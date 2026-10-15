"""Motion core — turn 3D landmarks into a hierarchical capture rig.

This is the pure-``numpy`` heart of the pipeline. It mirrors the capture rig
that the Blender add-on builds (``blender_addon/vid_an/mocap.py``) so that an
existing user bone-mapping keeps working: same bone names, same rest pose,
same orientation convention (local ``+Y`` points down the bone).

All knowledge about joints, reference axes and anatomical limits lives in
:mod:`core.anatomy`; this module only turns those specs into per-frame world
matrices and local rotations.

Input is a :class:`core.pose_schema.PoseSequence`. Output is a
:class:`SolvedSequence` holding, per frame, each bone's world matrix and its
local rotation quaternion / root translation — ready to be baked into Blender,
written to BVH, or exported as raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import anatomy, partial_body
from .anatomy import FORWARD, UP
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

REF_FWD = FORWARD
REF_UP = UP


@dataclass(frozen=True)
class Bone:
    name: str
    parent: Optional[str]
    head_joint: str
    tail_joint: str
    ref: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    head_rest: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tail_rest: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    secondary: Optional[Tuple[str, str]] = None
    group: str = "torso"
    weight: float = 1.0
    min_bend: float = 0.0
    max_bend: float = 180.0

    @property
    def ref_vec(self) -> np.ndarray:
        return np.array(self.ref, dtype=np.float64)


# Rest pose + hierarchy identical to the add-on's capture rig.
REST: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = anatomy.REST


def rotation_from_direction(direction: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Build a rotation whose local ``+Y`` maps to ``direction``.

    Columns are ``(x, y, z)``; the returned matrix has determinant ``+1``.
    Kept as the public rest-convention helper (``x = y x ref``); the richer
    two-axis builder lives in :func:`core.anatomy.bone_frame`.
    """
    return anatomy.bone_frame(direction, None, ref, None)


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
            rot = anatomy.bone_frame(tail - head, None, b.ref_vec, None)
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
    for spec in anatomy.BONES:
        head, tail = REST[spec.name]
        bones.append(Bone(
            name=spec.name, parent=spec.parent, head_joint=spec.head,
            tail_joint=spec.tail, ref=spec.ref,
            head_rest=head, tail_rest=tail, secondary=spec.secondary,
            group=spec.group, weight=spec.weight,
            min_bend=spec.min_bend, max_bend=spec.max_bend,
        ))
    return Skeleton(bones=bones)


SKELETON = build_skeleton()

# Backwards-compatible spec view (name, parent, head, tail, ref).
_SPEC: List[Tuple[str, Optional[str], str, str, np.ndarray]] = [
    (b.name, b.parent, b.head_joint, b.tail_joint, b.ref_vec) for b in SKELETON.bones
]


def mediapipe_to_solver(coords: np.ndarray) -> np.ndarray:
    """Map MediaPipe world coords ``(x, y, z)`` to the solver/Blender frame.

    Same convention as the add-on: ``(x, z, -y)`` (X left, -Y forward, Z up).
    Both MediaPipe and Blender are right-handed, so the change of basis must
    be a proper rotation (determinant +1); a reflection would mirror the body
    front-to-back.
    """
    return anatomy.mediapipe_to_solver(coords)


def _derive_head(pts: Dict[str, np.ndarray], valid: Dict[str, bool]) -> None:
    """Add a synthetic ``head_top`` from the face (nose + ears).

    The head bone must be able to nod/turn/tilt, so its axis comes from the face
    plane rather than from the torso. ``head_top = nose + up * 0.15`` where
    ``up = cross(forward, left)`` and ``forward = nose - mid_ear``. Falls back to
    the torso ``up`` when the ears are unreliable.
    """
    nose = pts.get("nose")
    if nose is None:
        return
    le = pts.get("l_ear")
    re = pts.get("r_ear")
    up = None
    if le is not None and re is not None and valid.get("l_ear", True) and valid.get("r_ear", True):
        left = np.asarray(le) - np.asarray(re)
        forward = np.asarray(nose) - (np.asarray(le) + np.asarray(re)) * 0.5
        c = np.cross(forward, left)
        n = float(np.linalg.norm(c))
        if n > 1e-6:
            up = c / n
    if up is None:
        base = pts.get("mid_shoulder", pts.get("nose")) - pts.get("mid_hip", pts.get("nose"))
        n = float(np.linalg.norm(base))
        up = base / n if n > 1e-6 else UP.copy()
    pts["head_top"] = np.asarray(nose, dtype=np.float64) + up * 0.15
    valid["head_top"] = valid.get("nose", False)


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
    _derive_head(pts, valid)
    return pts, synth


def _derive_base(pts: Dict[str, np.ndarray], valid: Dict[str, bool]) -> None:
    if "l_hip" in pts and "r_hip" in pts:
        pts["mid_hip"] = (pts["l_hip"] + pts["r_hip"]) * 0.5
        valid["mid_hip"] = bool(valid.get("l_hip") and valid.get("r_hip"))
    if "l_shoulder" in pts and "r_shoulder" in pts:
        pts["mid_shoulder"] = (pts["l_shoulder"] + pts["r_shoulder"]) * 0.5
        valid["mid_shoulder"] = bool(valid.get("l_shoulder") and valid.get("r_shoulder"))


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


def blend_joints(
    prev: Dict[str, np.ndarray],
    cur: Dict[str, np.ndarray],
    confidence: Optional[np.ndarray],
    names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Weight low-confidence joints toward their previous value.

    A single noisy landmark should not flip a whole limb between frames, so
    instead of trusting it blindly we blend it against the previous frame with a
    weight equal to its (scaled) confidence. Confident joints are untouched.
    Derived joints (``mid_hip``/``mid_shoulder``/``head_top``) are recomputed.
    """
    if confidence is None:
        return cur
    idx = {n: i for i, n in enumerate(names)}
    out = dict(cur)
    for jname in MP_INDEX:
        i = idx.get(jname)
        if i is None or i >= len(confidence) or jname not in prev:
            continue
        conf = float(confidence[i])
        if conf >= 0.5 or jname not in out:
            continue
        w = max(0.0, conf) / 0.5
        out[jname] = w * np.asarray(out[jname]) + (1.0 - w) * np.asarray(prev[jname])
    valid = {k: True for k in out}
    _derive_base(out, valid)
    _derive_head(out, valid)
    return out


def _secondary_vec(bone: Bone, joints: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if not bone.secondary:
        return None
    a = joints.get(bone.secondary[0])
    b = joints.get(bone.secondary[1])
    if a is None or b is None:
        return None
    v = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if float(np.linalg.norm(v)) < 1e-4:
        return None
    return v


def solve_frame(
    skeleton: Skeleton,
    joints: Dict[str, np.ndarray],
    prev: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Return ``name -> 4x4 world matrix`` for one frame.

    ``prev`` maps a bone name to the *previous frame's* 3x3 world rotation; it
    is used to keep each bone's roll continuous (no 180-degree flips) across
    frames. Bones are processed parent-first, and each direction is clamped to
    its anatomical bend range relative to the parent.
    """
    rest = skeleton.rest_world()
    out: Dict[str, np.ndarray] = {}
    for b in skeleton.bones:
        head = joints.get(b.head_joint)
        tail = joints.get(b.tail_joint)
        if head is None or tail is None:
            parent_prev = prev.get(b.name) if prev else None
            rot = parent_prev[:3, :3] if parent_prev is not None else rest[b.name][:3, :3]
            pos = np.asarray(head, dtype=np.float64) if head is not None else rest[b.name][:3, 3]
            out[b.name] = _rt_to_matrix(np.array(rot, dtype=np.float64), pos)
            continue

        head = np.asarray(head, dtype=np.float64)
        direction = np.asarray(tail, dtype=np.float64) - head

        parent_dir = None
        if b.parent and b.parent in out:
            parent_dir = out[b.parent][:3, 1]
        direction = anatomy.clamp_direction(direction, parent_dir, b.min_bend, b.max_bend)

        if float(np.linalg.norm(direction)) < 1e-8:
            rot = prev[b.name][:3, :3] if prev and b.name in prev else rest[b.name][:3, :3]
            out[b.name] = _rt_to_matrix(np.array(rot, dtype=np.float64), head)
            continue

        prev_x = None
        if prev and b.name in prev:
            prev_x = prev[b.name][:3, 0]
        frame = anatomy.bone_frame(direction, _secondary_vec(b, joints), b.ref_vec, prev_x)
        out[b.name] = _rt_to_matrix(frame, head)
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
            secondary=b.secondary, group=b.group, weight=b.weight,
            min_bend=b.min_bend, max_bend=b.max_bend,
        ))
    return Skeleton(bones=bones)


def solve_sequence(
    sequence: PoseSequence,
    skeleton: Optional[Skeleton] = None,
    min_visibility: float = 0.3,
    fill_partial: bool = True,
    smooth_rotations: bool = True,
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

    from .smoothing import QuatSmoother

    smoothers = {b.name: QuatSmoother(anatomy.group_smoothing(b.group)) for b in skel.bones}

    prev_world: Optional[Dict[str, np.ndarray]] = None
    prev_joints: Optional[Dict[str, np.ndarray]] = None
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
                world = solve_frame(skel, joints, prev_world)
                prev_world = world
                prev_joints = joints
            result.synthesized.append(set(synth))
        else:
            result.missing.append(False)
            joints, synth = build_joints(
                fr.landmarks_3d, sequence.landmark_names,
                confidence=fr.confidence, min_vis=min_visibility,
                fill_partial=fill_partial)
            if prev_joints is not None:
                joints = blend_joints(prev_joints, joints, fr.confidence, sequence.landmark_names)
            world = solve_frame(skel, joints, prev_world)
            prev_world = world
            prev_joints = joints
            result.synthesized.append(set(synth or set()))

        local: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for b in skel.bones:
            if b.name not in world:
                continue
            loc, quat = local_basis(skel, b.name, world, rest_local, rest_world)
            if smooth_rotations:
                quat = smoothers[b.name](quat)
            local[b.name] = (loc, quat)
        result.world.append(world)
        result.local.append(local)

        root = world.get("Hips")
        rpos = root[:3, 3] if root is not None else np.zeros(3, np.float64)
        prev_root = np.asarray(rpos, dtype=np.float64)
        result.root_positions = np.vstack([result.root_positions, rpos.astype(np.float32)])

    return result
