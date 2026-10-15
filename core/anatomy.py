"""Anatomical correspondence table and frame geometry — single source of truth.

This module holds *all* knowledge about how the detector landmarks turn into a
humanoid skeleton and how each bone's spatial orientation is built. Both the
pure-``numpy`` core (:mod:`core.motion_solver`) and the Blender add-on
(``blender_addon/vid_an/mocap.py``) read this table, so a joint/axis/limit
tweak only has to be made here.

Coordinate convention (solver / Blender space, right-handed)
------------------------------------------------------------
* ``+X``  : the person's LEFT (image right when they face the camera).
* ``+Y``  : BACKWARD; the character faces ``-Y`` (``FORWARD = (0, -1, 0)``).
* ``+Z``  : UP.
* origin   : solver origin (metres). MediaPipe world coords are converted with a
  proper rotation (determinant +1) by :func:`mediapipe_to_solver`.

Each bone is defined by:
* ``head`` / ``tail`` joints — the segment direction is ``tail - head`` (the
  bone's local ``+Y``).
* an optional ``secondary`` joint pair ``(a, b)`` giving the *left* axis
  ``a - b`` (e.g. ``("l_hip", "r_hip")``). Bones with a secondary axis get their
  roll (yaw/twist) from real landmarks — that is what makes the pelvis, chest and
  head move correctly instead of being built from a single direction.
* a fallback ``ref`` vector used only when no secondary axis is available.
* a ``group`` (for per-region smoothing/filter parameters), a reliability
  ``weight``, and a bend range ``(min_bend, max_bend)`` in degrees measured
  between this bone's direction and its parent's direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FORWARD = np.array([0.0, -1.0, 0.0], dtype=np.float64)
UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class BoneSpec:
    name: str
    parent: Optional[str]
    head: str
    tail: str
    secondary: Optional[Tuple[str, str]]
    ref: Tuple[float, float, float]
    group: str
    weight: float = 1.0
    min_bend: float = 0.0
    max_bend: float = 180.0

    @property
    def ref_vec(self) -> np.ndarray:
        return np.array(self.ref, dtype=np.float64)


# Rest pose (metres) of the canonical 18-bone capture rig. Head/tail are kept on
# the vertical axis; the actual per-frame orientation comes from landmarks.
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

_LEG_L = ("l_hip", "r_hip")
_SHO_L = ("l_shoulder", "r_shoulder")
_EAR_L = ("l_ear", "r_ear")

# The correspondence table. ``secondary`` axes are what give the torso, pelvis
# and head their real twist; limbs use the rest ``ref`` plus temporal continuity
# (previous frame) so their roll never flips.
BONES: List[BoneSpec] = [
    BoneSpec("Hips", None, "mid_hip", "mid_shoulder", _LEG_L, (0.0, -1.0, 0.0),
             group="torso", weight=1.0, min_bend=0.0, max_bend=180.0),
    BoneSpec("Spine", "Hips", "mid_hip", "mid_shoulder", _SHO_L, (0.0, -1.0, 0.0),
             group="torso", weight=1.0, min_bend=0.0, max_bend=45.0),
    BoneSpec("Neck", "Spine", "mid_shoulder", "nose", _SHO_L, (0.0, -1.0, 0.0),
             group="neck", weight=1.0, min_bend=0.0, max_bend=55.0),
    BoneSpec("Head", "Neck", "nose", "head_top", _EAR_L, (0.0, -1.0, 0.0),
             group="head", weight=1.0, min_bend=0.0, max_bend=70.0),
    BoneSpec("Shoulder.L", "Spine", "mid_shoulder", "l_shoulder", None, (0.0, -1.0, 0.0),
             group="arm.l", weight=1.0, min_bend=0.0, max_bend=60.0),
    BoneSpec("UpperArm.L", "Shoulder.L", "l_shoulder", "l_elbow", None, (0.0, -1.0, 0.0),
             group="arm.l", weight=1.0, min_bend=0.0, max_bend=180.0),
    BoneSpec("ForeArm.L", "UpperArm.L", "l_elbow", "l_wrist", None, (0.0, -1.0, 0.0),
             group="arm.l", weight=1.0, min_bend=0.0, max_bend=155.0),
    BoneSpec("Hand.L", "ForeArm.L", "l_wrist", "l_index", None, (0.0, -1.0, 0.0),
             group="arm.l", weight=0.8, min_bend=0.0, max_bend=90.0),
    BoneSpec("Shoulder.R", "Spine", "mid_shoulder", "r_shoulder", None, (0.0, -1.0, 0.0),
             group="arm.r", weight=1.0, min_bend=0.0, max_bend=60.0),
    BoneSpec("UpperArm.R", "Shoulder.R", "r_shoulder", "r_elbow", None, (0.0, -1.0, 0.0),
             group="arm.r", weight=1.0, min_bend=0.0, max_bend=180.0),
    BoneSpec("ForeArm.R", "UpperArm.R", "r_elbow", "r_wrist", None, (0.0, -1.0, 0.0),
             group="arm.r", weight=1.0, min_bend=0.0, max_bend=155.0),
    BoneSpec("Hand.R", "ForeArm.R", "r_wrist", "r_index", None, (0.0, -1.0, 0.0),
             group="arm.r", weight=0.8, min_bend=0.0, max_bend=90.0),
    BoneSpec("Thigh.L", "Hips", "l_hip", "l_knee", None, (0.0, -1.0, 0.0),
             group="leg.l", weight=1.0, min_bend=0.0, max_bend=130.0),
    BoneSpec("Shin.L", "Thigh.L", "l_knee", "l_ankle", None, (0.0, -1.0, 0.0),
             group="leg.l", weight=1.0, min_bend=0.0, max_bend=155.0),
    BoneSpec("Foot.L", "Shin.L", "l_ankle", "l_foot", None, (0.0, -1.0, 0.0),
             group="leg.l", weight=0.8, min_bend=0.0, max_bend=80.0),
    BoneSpec("Thigh.R", "Hips", "r_hip", "r_knee", None, (0.0, -1.0, 0.0),
             group="leg.r", weight=1.0, min_bend=0.0, max_bend=130.0),
    BoneSpec("Shin.R", "Thigh.R", "r_knee", "r_ankle", None, (0.0, -1.0, 0.0),
             group="leg.r", weight=1.0, min_bend=0.0, max_bend=155.0),
    BoneSpec("Foot.R", "Shin.R", "r_ankle", "r_foot", None, (0.0, -1.0, 0.0),
             group="leg.r", weight=0.8, min_bend=0.0, max_bend=80.0),
]

BONES_BY_NAME: Dict[str, BoneSpec] = {b.name: b for b in BONES}

# Per-group temporal smoothing strength for the rotation filter (0..1, higher =
# smoother/laggier). Torso/head are stabilised more; fast hand gestures kept.
SMOOTH: Dict[str, float] = {
    "torso": 0.55,
    "neck": 0.5,
    "head": 0.5,
    "arm": 0.25,
    "leg": 0.4,
}


def group_smoothing(group: str) -> float:
    """Map a bone group like ``arm.l`` to its filter strength."""
    base = group.split(".")[0]
    return SMOOTH.get(group, SMOOTH.get(base, 0.3))


def mediapipe_to_solver(coords: np.ndarray) -> np.ndarray:
    """Map MediaPipe world coords ``(x, y, z)`` to solver space.

    MediaPipe world: ``+x`` = person's left, ``+y`` = down, ``+z`` = toward the
    camera. Solver: ``+X`` = left, ``+Y`` = back (so the face looks ``-Y``),
    ``+Z`` = up. Both frames are right-handed, so this is a proper rotation
    (determinant +1): ``(x, y, z) -> (x, z, -y)``.
    """
    c = np.asarray(coords, dtype=np.float64)
    out = np.empty_like(c)
    out[..., 0] = c[..., 0]
    out[..., 1] = c[..., 2]
    out[..., 2] = -c[..., 1]
    return out


def bone_frame(
    direction: np.ndarray,
    secondary: Optional[np.ndarray] = None,
    ref: np.ndarray = FORWARD,
    prev_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a right-handed bone frame ``columns = (x, y, z)``.

    * ``y`` is the (normalised) bone direction.
    * ``x`` is the *left* axis: from ``secondary`` when available, otherwise
      ``y x ref`` (the rest convention used for the limbs).
    * ``z = x x y``.

    When ``prev_x`` is given, the roll is chosen to be the closest hemisphere to
    the previous frame, which removes the 180-degree flips that otherwise appear
    when the reference nearly aligns with the direction.
    """
    y = np.asarray(direction, dtype=np.float64)
    ny = float(np.linalg.norm(y))
    if ny < 1e-9:
        return np.eye(3, dtype=np.float64)
    y = y / ny

    x: Optional[np.ndarray] = None
    if secondary is not None:
        s = np.asarray(secondary, dtype=np.float64)
        s = s - float(np.dot(s, y)) * y
        if float(np.linalg.norm(s)) > 1e-6:
            x = s
    if x is None:
        r = np.asarray(ref, dtype=np.float64)
        r = r / (float(np.linalg.norm(r)) + 1e-12)
        if abs(float(np.dot(r, y))) > 0.999:
            r = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(y, r)
        if float(np.linalg.norm(x)) < 1e-9:
            x = np.array([1.0, 0.0, 0.0])
    x = x / (float(np.linalg.norm(x)) + 1e-12)

    if prev_x is not None and float(np.dot(x, np.asarray(prev_x, dtype=np.float64))) < 0.0:
        x = -x
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def clamp_direction(direction: np.ndarray, parent_dir: Optional[np.ndarray],
                    min_bend: float, max_bend: float) -> np.ndarray:
    """Clamp the bend between ``direction`` and ``parent_dir`` to a range.

    ``min_bend``/``max_bend`` are degrees. Returns a possibly-rotated direction
    of the same length; when the bend is inside the range the input is returned
    unchanged. This only nudges anatomically impossible poses back to the limit
    instead of hard-locking the joint.
    """
    d = np.asarray(direction, dtype=np.float64)
    nd = float(np.linalg.norm(d))
    if parent_dir is None or nd < 1e-9:
        return d
    p = np.asarray(parent_dir, dtype=np.float64)
    np_ = float(np.linalg.norm(p))
    if np_ < 1e-9:
        return d
    ang = angle_between(d, p)
    if min_bend <= ang <= max_bend:
        return d
    target = max_bend if ang > max_bend else min_bend
    axis = np.cross(p, d)
    na = float(np.linalg.norm(axis))
    if na < 1e-9:
        return d
    axis = axis / na
    theta = math.radians(target - ang)
    return _rot_axis(axis, theta) @ d


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype=np.float64)
