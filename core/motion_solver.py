"""Motion core — turn 3D landmarks into a hierarchical capture rig.

This is the pure-``numpy`` heart of the pipeline. It mirrors the capture rig
that the Blender add-on builds (``blender_addon/vid_an/mocap.py``) so that an
existing user bone-mapping keeps working: same bone names, same rest pose,
same orientation convention (local ``+Y`` points down the bone).

Input is a :class:`core.pose_schema.PoseSequence`. Output is a
:class:`SolvedSequence` holding, per frame, each bone's world matrix and its
local rotation quaternion / root translation — ready to be baked into Blender,
written to BVH, or exported as raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import partial_body
from .pose_schema import MEDIAPIPE_NAMES, PoseSequence

# MediaPipe landmark index -> short joint name (matches mocap.py).
MP_INDEX: Dict[str, int] = {
    "nose": 0,
    "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_foot": 31, "r_foot": 32,
}

REF_FWD = np.array([0.0, -1.0, 0.0], dtype=np.float64)
REF_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class Bone:
    name: str
    parent: Optional[str]
    head_joint: str
    tail_joint: str
    ref: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    head_rest: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tail_rest: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def ref_vec(self) -> np.ndarray:
        return np.array(self.ref, dtype=np.float64)


# Rest pose + hierarchy identical to the add-on's capture rig.
REST: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "Hips": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.1)),
    "Spine": ((0.0, 0.0, 0.1), (0.0, 0.0, 0.35)),
    "Neck": ((0.0, 0.0, 0.35), (0.0, 0.0, 0.5)),
    "Head": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.65)),
    "Shoulder.L": ((0.0, 0.0, 0.4), (0.1, 0.0, 0.45)),
    "UpperArm.L": ((0.1, 0.0, 0.45), (0.34, 0.0, 0.45)),
    "ForeArm.L": ((0.34, 0.0, 0.45), (0.58, 0.0, 0.45)),
    "Hand.L": ((0.58, 0.0, 0.45), (0.73, 0.0, 0.45)),
    "Shoulder.R": ((0.0, 0.0, 0.4), (-0.1, 0.0, 0.45)),
    "UpperArm.R": ((-0.1, 0.0, 0.45), (-0.34, 0.0, 0.45)),
    "ForeArm.R": ((-0.34, 0.0, 0.45), (-0.58, 0.0, 0.45)),
    "Hand.R": ((-0.58, 0.0, 0.45), (-0.73, 0.0, 0.45)),
    "Thigh.L": ((0.08, 0.0, 0.0), (0.08, 0.0, -0.4)),
    "Shin.L": ((0.08, 0.0, -0.4), (0.08, 0.0, -0.8)),
    "Foot.L": ((0.08, 0.0, -0.8), (0.08, 0.15, -0.8)),
    "Thigh.R": ((-0.08, 0.0, 0.0), (-0.08, 0.0, -0.4)),
    "Shin.R": ((-0.08, 0.0, -0.4), (-0.08, 0.0, -0.8)),
    "Foot.R": ((-0.08, 0.0, -0.8), (-0.08, 0.15, -0.8)),
}

# (name, parent, head_joint, tail_joint, ref)
_SPEC: List[Tuple[str, Optional[str], str, str, np.ndarray]] = [
    ("Hips", None, "mid_hip", "mid_shoulder", REF_FWD),
    ("Spine", "Hips", "mid_hip", "mid_shoulder", REF_FWD),
    ("Neck", "Spine", "mid_shoulder", "nose", REF_FWD),
    ("Head", "Neck", "nose", "nose_up", REF_FWD),
    ("Shoulder.L", "Spine", "mid_shoulder", "l_shoulder", REF_UP),
    ("UpperArm.L", "Shoulder.L", "l_shoulder", "l_elbow", REF_UP),
    ("ForeArm.L", "UpperArm.L", "l_elbow", "l_wrist", REF_UP),
    ("Hand.L", "ForeArm.L", "l_wrist", "l_index", REF_UP),
    ("Shoulder.R", "Spine", "mid_shoulder", "r_shoulder", REF_UP),
    ("UpperArm.R", "Shoulder.R", "r_shoulder", "r_elbow", REF_UP),
    ("ForeArm.R", "UpperArm.R", "r_elbow", "r_wrist", REF_UP),
    ("Hand.R", "ForeArm.R", "r_wrist", "r_index", REF_UP),
    ("Thigh.L", "Hips", "l_hip", "l_knee", REF_FWD),
    ("Shin.L", "Thigh.L", "l_knee", "l_ankle", REF_FWD),
    ("Foot.L", "Shin.L", "l_ankle", "l_foot", REF_UP),
    ("Thigh.R", "Hips", "r_hip", "r_knee", REF_FWD),
    ("Shin.R", "Thigh.R", "r_knee", "r_ankle", REF_FWD),
    ("Foot.R", "Shin.R", "r_ankle", "r_foot", REF_UP),
]


def rotation_from_direction(direction: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Build a rotation whose local ``+Y`` maps to ``direction``.

    Columns are ``(x, y, z)``; the returned matrix has determinant ``+1``.
    """
    y = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(y)
    if n < 1e-9:
        return np.eye(3)
    y = y / n
    r = np.asarray(ref, dtype=np.float64)
    r = r / (np.linalg.norm(r) + 1e-12)
    if abs(float(np.dot(y, r))) > 0.999:
        r = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(r, y)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / nx
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _rt_to_matrix(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = pos
    return m


@dataclass
class Skeleton:
    bones: List[Bone]

    @property
    def names(self) -> List[str]:
        return [b.name for b in self.bones]

    def by_name(self, name: str) -> Bone:
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(name)

    def rest_world(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for b in self.bones:
            head = np.array(b.head_rest, dtype=np.float64)
            tail = np.array(b.tail_rest, dtype=np.float64)
            rot = rotation_from_direction(tail - head, b.ref_vec)
            out[b.name] = _rt_to_matrix(rot, head)
        return out

    def rest_local(self) -> Dict[str, np.ndarray]:
        """Local rest transform of each bone relative to its parent."""
        rw = self.rest_world()
        out: Dict[str, np.ndarray] = {}
        for b in self.bones:
            if b.parent:
                out[b.name] = np.linalg.inv(rw[b.parent]) @ rw[b.name]
            else:
                out[b.name] = rw[b.name].copy()
        return out


def build_skeleton() -> Skeleton:
    bones = []
    for name, parent, hj, tj, ref in _SPEC:
        head, tail = REST[name]
        bones.append(Bone(
            name=name, parent=parent, head_joint=hj, tail_joint=tj,
            ref=tuple(float(v) for v in ref), head_rest=head, tail_rest=tail,
        ))
    return Skeleton(bones=bones)


SKELETON = build_skeleton()


def mediapipe_to_solver(coords: np.ndarray) -> np.ndarray:
    """Map MediaPipe world coords ``(x, y, z)`` to the solver/Blender frame.

    Same convention as the add-on: ``(x, z, -y)`` (X left, -Y forward, Z up).
    Both MediaPipe and Blender are right-handed, so the change of basis must
    be a proper rotation (determinant +1); the previous ``(x, -z, -y)`` was a
    reflection (determinant -1) that mirrored the body front-to-back.
    """
    c = np.asarray(coords, dtype=np.float64)
    out = np.empty_like(c)
    out[..., 0] = c[..., 0]
    out[..., 1] = c[..., 2]
    out[..., 2] = -c[..., 1]
    return out


def build_joints(
    landmarks_3d: np.ndarray,
    names: Sequence[str] = MEDIAPIPE_NAMES,
    confidence: Optional[np.ndarray] = None,
    min_vis: float = 0.3,
    fill_partial: bool = True,
    stance_width: float = 0.18,
    silhouette_xz: Optional[tuple] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[Set[str]]]:
    """Build the named joint dictionary plus the set of *synthesized* joints.

    When ``fill_partial`` is true and a body part is not visible (low
    confidence / missing), it is rebuilt as a standing pose whose centre of
    gravity stays under the visible silhouette centroid — see
    :mod:`core.partial_body`.
    """
    if landmarks_3d is None or len(landmarks_3d) == 0:
        return {}, None
    idx = {n: i for i, n in enumerate(names)}
    pts: Dict[str, np.ndarray] = {}
    valid: Dict[str, bool] = {}
    for jname, wi in MP_INDEX.items():
        i = idx.get(jname)
        if i is None:
            if wi >= len(landmarks_3d):
                continue
            i = wi
        raw = np.asarray(landmarks_3d[i], dtype=np.float64)
        conf = 1.0
        if confidence is not None and i < len(confidence):
            conf = float(confidence[i])
        ok = np.all(np.isfinite(raw)) and conf >= min_vis
        pts[jname] = mediapipe_to_solver(raw)
        valid[jname] = bool(ok)

    _derive_base(pts, valid)

    synth: Optional[Set[str]] = None
    if fill_partial and not all(valid.get(k, False) for k in MP_INDEX):
        pts, synth = partial_body.complete(
            pts, valid, stance_width=stance_width, silhouette_xz=silhouette_xz)
    return pts, synth


def _derive_base(pts: Dict[str, np.ndarray], valid: Dict[str, bool]) -> None:
    if "l_hip" in pts and "r_hip" in pts:
        pts["mid_hip"] = (pts["l_hip"] + pts["r_hip"]) * 0.5
        valid["mid_hip"] = bool(valid.get("l_hip") and valid.get("r_hip"))
    if "l_shoulder" in pts and "r_shoulder" in pts:
        pts["mid_shoulder"] = (pts["l_shoulder"] + pts["r_shoulder"]) * 0.5
        valid["mid_shoulder"] = bool(valid.get("l_shoulder") and valid.get("r_shoulder"))
    if "nose" in pts:
        base = pts.get("mid_shoulder", pts.get("nose")) - pts.get("mid_hip", pts.get("nose"))
        n = np.linalg.norm(base)
        up = base / n if n > 1e-6 else REF_UP.copy()
        pts["nose_up"] = pts["nose"] + up * 0.1
        valid["nose_up"] = valid.get("nose", False)


def joints_from_landmarks(
    landmarks_3d: np.ndarray,
    names: Sequence[str] = MEDIAPIPE_NAMES,
    confidence: Optional[np.ndarray] = None,
    min_vis: float = 0.3,
    fill_partial: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """Convenience wrapper returning just the joint dictionary."""
    pts, _synth = build_joints(
        landmarks_3d, names, confidence=confidence, min_vis=min_vis,
        fill_partial=fill_partial)
    if not pts:
        return None
    return pts


def solve_frame(
    skeleton: Skeleton,
    joints: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return ``name -> 4x4 world matrix`` for one frame."""
    rest = skeleton.rest_world()
    out: Dict[str, np.ndarray] = {}
    for b in skeleton.bones:
        head = joints.get(b.head_joint)
        tail = joints.get(b.tail_joint)
        if head is None or tail is None:
            continue
        direction = tail - head
        if np.linalg.norm(direction) < 1e-8:
            rot = out.get(b.parent, rest[b.name])[:3, :3] if b.parent else rest[b.name][:3, :3]
            out[b.name] = _rt_to_matrix(rot.copy(), head)
            continue
        rot = rotation_from_direction(direction, b.ref_vec)
        out[b.name] = _rt_to_matrix(rot, head)
    return out


def local_basis(
    skeleton: Skeleton,
    name: str,
    world: Dict[str, np.ndarray],
    rest_local: Dict[str, np.ndarray],
    rest_world: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(location, quaternion wxyz)`` for a bone's local basis.

    Uses the standard rest-binding formula so results match Blender's
    ``pose_bone.matrix_basis`` for the same world matrices.
    """
    b = skeleton.by_name(name)
    rest_w = rest_world if rest_world is not None else skeleton.rest_world()
    if b.parent and b.parent in world:
        base = world[b.parent] @ np.linalg.inv(rest_w[b.parent]) @ rest_w[name]
    else:
        base = rest_w[name]
    basis = np.linalg.inv(base) @ world[name]
    loc = basis[:3, 3].copy()
    quat = matrix_to_quat(basis[:3, :3])
    return loc, quat


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> quaternion ``(w, x, y, z)``."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


@dataclass
class SolvedSequence:
    """Per-frame solved capture-rig motion."""

    skeleton: Skeleton = field(default_factory=build_skeleton)
    fps: float = 30.0
    source: str = ""
    world: List[Dict[str, np.ndarray]] = field(default_factory=list)
    local: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    root_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    missing: List[bool] = field(default_factory=list)
    synthesized: List[Set[str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.world)


def estimate_height(joints: Dict[str, np.ndarray]) -> float:
    """Approximate standing height from hip and head/ankle joints."""
    mid_hip = joints.get("mid_hip")
    nose = joints.get("nose")
    l_ankle = joints.get("l_ankle")
    r_ankle = joints.get("r_ankle")
    if mid_hip is None or nose is None or l_ankle is None or r_ankle is None:
        return 0.0
    ankle = (l_ankle + r_ankle) * 0.5
    return float(np.linalg.norm(nose - ankle) + 0.12)


def scale_skeleton(
    skeleton: Skeleton,
    factor: float,
) -> Skeleton:
    """Return a copy of ``skeleton`` with rest positions scaled by ``factor``."""
    bones = []
    for b in skeleton.bones:
        bones.append(Bone(
            name=b.name, parent=b.parent, head_joint=b.head_joint,
            tail_joint=b.tail_joint, ref=b.ref,
            head_rest=tuple(v * factor for v in b.head_rest),
            tail_rest=tuple(v * factor for v in b.tail_rest),
        ))
    return Skeleton(bones=bones)


def solve_sequence(
    sequence: PoseSequence,
    skeleton: Optional[Skeleton] = None,
    min_visibility: float = 0.3,
    fill_partial: bool = True,
) -> SolvedSequence:
    """Solve every frame of ``sequence`` into a :class:`SolvedSequence`.

    Frames without landmarks are marked missing and reuse the previous pose so
    downstream consumers always get a continuous track. With ``fill_partial``,
    hidden body parts are rebuilt as a balanced standing pose.
    """
    skel = skeleton or SKELETON
    rest_local = skel.rest_local()
    rest_world = skel.rest_world()
    result = SolvedSequence(skeleton=skel, fps=sequence.fps, source=sequence.source)

    prev_world: Optional[Dict[str, np.ndarray]] = None
    prev_root = np.zeros(3, dtype=np.float64)
    for fr in sequence.frames:
        if fr.landmarks_3d is None or len(fr.landmarks_3d) == 0:
            result.missing.append(True)
            if fill_partial:
                joints, synth = partial_body.standing_pose(anchor=prev_root, scale=1.0)
            else:
                joints, synth = None, set()
            if joints is None:
                if prev_world is None:
                    result.world.append({})
                    result.local.append({})
                    result.synthesized.append(set())
                    result.root_positions = np.vstack(
                        [result.root_positions, np.zeros(3, np.float32)])
                    continue
                world = prev_world
            else:
                world = solve_frame(skel, joints)
                prev_world = world
            result.synthesized.append(set(synth))
        else:
            result.missing.append(False)
            joints, synth = build_joints(
                fr.landmarks_3d, sequence.landmark_names,
                confidence=fr.confidence, min_vis=min_visibility,
                fill_partial=fill_partial)
            world = solve_frame(skel, joints)
            prev_world = world
            result.synthesized.append(set(synth or set()))

        local: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for b in skel.bones:
            if b.name not in world:
                continue
            loc, quat = local_basis(skel, b.name, world, rest_local, rest_world)
            local[b.name] = (loc, quat)
        result.world.append(world)
        result.local.append(local)

        root = world.get("Hips")
        rpos = root[:3, 3] if root is not None else np.zeros(3, np.float64)
        prev_root = np.asarray(rpos, dtype=np.float64)
        result.root_positions = np.vstack([result.root_positions, rpos.astype(np.float32)])

    return result
