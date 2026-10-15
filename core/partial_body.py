"""Partial-body completion.

When the video does not show the whole person (legs out of frame, occlusion,
cropping), MediaPipe still returns all 33 landmarks but the hidden ones have a
low visibility and unreliable world coordinates. Blindly trusting them makes the
character collapse or drift.

This module rebuilds the missing body parts as a plausible **standing pose**
whose horizontal centre of gravity stays under the centre of gravity of the
*visible* joints — which is the closest available proxy for the centre of the
visible silhouette. In other words: if only the upper body is on screen, the
legs are placed in a stance underneath the visible mass instead of dangling from
garbage coordinates.

Only ``numpy`` is used. All geometry lives in solver space (``+X`` = the person's
LEFT, ``+Y`` = BACKWARD, ``+Z`` = UP, i.e. the character faces ``-Y``) — the same
space used by :mod:`core.motion_solver` and by the add-on's ``mocap._to_blender``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

# A canonical ~1.7 m standing template, offsets in metres from mid-hip, expressed
# in solver space (+Z up, character faces -Y).  ``+Z`` is height, ``-Y`` is forward.
STANDING: Dict[str, Tuple[float, float, float]] = {
    "mid_hip": (0.00, 0.00, 0.00),
    "l_hip": (0.09, 0.00, 0.00),
    "r_hip": (-0.09, 0.00, 0.00),
    "l_knee": (0.10, -0.02, -0.44),
    "r_knee": (-0.10, -0.02, -0.44),
    "l_ankle": (0.10, -0.03, -0.86),
    "r_ankle": (-0.10, -0.03, -0.86),
    "l_foot": (0.10, -0.16, -0.88),
    "r_foot": (-0.10, -0.16, -0.88),
    "l_shoulder": (0.18, 0.00, 0.48),
    "r_shoulder": (-0.18, 0.00, 0.48),
    "l_elbow": (0.22, 0.00, 0.20),
    "r_elbow": (-0.22, 0.00, 0.20),
    "l_wrist": (0.24, 0.00, -0.06),
    "r_wrist": (-0.24, 0.00, -0.06),
    "l_index": (0.25, -0.02, -0.12),
    "r_index": (-0.25, -0.02, -0.12),
    "nose": (0.00, -0.10, 0.62),
    "l_ear": (0.08, -0.02, 0.60),
    "r_ear": (-0.08, -0.02, 0.60),
}

# Rough segment-mass weights (normalised at use time) for centre of gravity.
COG_WEIGHTS: Dict[str, float] = {
    "nose": 0.060,
    "mid_shoulder": 0.220,
    "mid_hip": 0.280,
    "l_shoulder": 0.030, "r_shoulder": 0.030,
    "l_elbow": 0.020, "r_elbow": 0.020,
    "l_wrist": 0.020, "r_wrist": 0.020,
    "l_hip": 0.040, "r_hip": 0.040,
    "l_knee": 0.090, "r_knee": 0.090,
    "l_ankle": 0.040, "r_ankle": 0.040,
    "l_foot": 0.010, "r_foot": 0.010,
}

LOWER_BODY: Set[str] = {
    "mid_hip", "l_hip", "r_hip", "l_knee", "r_knee",
    "l_ankle", "r_ankle", "l_foot", "r_foot",
}

_MIRROR = {"l": "r", "r": "l"}


def _cog(pts: Dict[str, np.ndarray], valid: Dict[str, bool],
         only: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, float]:
    """Weighted mean position of the valid joints (horizontal + total mass)."""
    allowed = set(only) if only is not None else None
    num = np.zeros(3, dtype=np.float64)
    den = 0.0
    for name, w in COG_WEIGHTS.items():
        if allowed is not None and name not in allowed:
            continue
        if valid.get(name) and name in pts:
            num += w * np.asarray(pts[name], dtype=np.float64)
            den += w
    if den <= 0.0:
        return np.zeros(3, dtype=np.float64), 0.0
    return num / den, den


def estimate_scale(pts: Dict[str, np.ndarray], valid: Dict[str, bool]) -> Optional[float]:
    """Estimate metres-per-template-unit from any reliable body measurement."""
    estimates = []

    def dist(a: str, b: str) -> Optional[float]:
        if valid.get(a) and valid.get(b) and a in pts and b in pts:
            return float(np.linalg.norm(np.asarray(pts[a]) - np.asarray(pts[b])))
        return None

    sw = dist("l_shoulder", "r_shoulder")
    if sw and sw > 1e-4:
        estimates.append(sw / 0.36)     # template shoulder width
    hw = dist("l_hip", "r_hip")
    if hw and hw > 1e-4:
        estimates.append(hw / 0.18)     # template hip width
    torso = dist("nose", "mid_hip")
    if torso and torso > 1e-4:
        estimates.append(torso / 0.62)
    if not estimates:
        return None
    return float(np.median(estimates))


def _anchor_hip(pts: Dict[str, np.ndarray], valid: Dict[str, bool],
                scale: float) -> np.ndarray:
    """Best estimate of the (mid-)hip position from whatever is visible."""
    vis_hips = [pts[h] for h in ("l_hip", "r_hip") if valid.get(h) and h in pts]
    if vis_hips:
        return np.mean(np.asarray(vis_hips, dtype=np.float64), axis=0)
    if valid.get("mid_hip") and "mid_hip" in pts:
        return np.asarray(pts["mid_hip"], dtype=np.float64)
    vis_sh = [pts[s] for s in ("l_shoulder", "r_shoulder") if valid.get(s) and s in pts]
    if vis_sh:
        ms = np.mean(np.asarray(vis_sh, dtype=np.float64), axis=0)
        return ms + np.array([0.0, 0.0, -0.48 * scale])
    if valid.get("mid_shoulder") and "mid_shoulder" in pts:
        return np.asarray(pts["mid_shoulder"], dtype=np.float64) + np.array([0.0, 0.0, -0.48 * scale])
    if valid.get("nose") and "nose" in pts:
        return np.asarray(pts["nose"], dtype=np.float64) + np.array([0.0, 0.0, -0.62 * scale])
    return np.array([0.0, 0.0, -0.9 * scale], dtype=np.float64)


def _mirror_side(name: str, pts: Dict[str, np.ndarray], valid: Dict[str, bool],
                 plane_x: float) -> Optional[np.ndarray]:
    """Mirror a visible opposite-side joint across the sagittal plane."""
    side = name.split(".")[-1] if "." in name else None
    if side not in _MIRROR:
        return None
    other = name.replace("." + side, "." + _MIRROR[side])
    # handle names like l_knee / r_knee (underscore form)
    if other not in pts:
        return None
    if not valid.get(other) or other not in pts:
        return None
    q = np.asarray(pts[other], dtype=np.float64).copy()
    q[0] = 2.0 * plane_x - q[0]
    return q


def _name_variants(name: str) -> Tuple[str, Optional[str]]:
    """Return (underscore_name, dot_name) for a template joint name."""
    if name in ("l_hip", "r_hip", "l_knee", "r_knee", "l_shoulder", "r_shoulder",
                "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_ankle", "r_ankle",
                "l_foot", "r_foot", "l_ear", "r_ear"):
        return name, name[0] + "." + name[2:]
    if name in ("l_index", "r_index"):
        return name, name[0] + "." + name[2:]
    return name, None


def complete(
    pts: Dict[str, np.ndarray],
    valid: Dict[str, bool],
    scale: Optional[float] = None,
    stance_width: float = 0.18,
    silhouette_xz: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, np.ndarray], Set[str]]:
    """Fill missing joints and balance the result.

    Returns ``(points, synthesized_names)``. ``silhouette_xz`` optionally
    overrides the horizontal centre-of-gravity target with a silhouette
    centroid (solver-space horizontal ``x, y``).
    """
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in pts.items()}
    vis = dict(valid)
    synth: Set[str] = set()

    if scale is None:
        scale = estimate_scale(out, vis) or 1.0
    hip = _anchor_hip(out, vis, scale)
    plane_x = float(hip[0])

    # 1) mirror a visible side onto the hidden side when possible
    for tname in list(STANDING.keys()):
        us, dot = _name_variants(tname)
        key = dot if (dot and dot in out) else us
        if vis.get(key):
            continue
        m = _mirror_side(us, out, vis, plane_x)
        if m is None and dot:
            m = _mirror_side(dot, out, vis, plane_x)
        if m is not None:
            out[us] = m
            vis[us] = True
            synth.add(us)

    # 2) place the rest from the standing template
    for name, off in STANDING.items():
        us, dot = _name_variants(name)
        if vis.get(us) or vis.get(name):
            continue
        p = hip + np.array(off, dtype=np.float64) * scale
        if us in ("l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
                  "l_foot", "r_foot"):
            p[0] = hip[0] + (off[0] * scale if stance_width is None else
                             np.sign(off[0]) * stance_width * 0.5 if off[0] else 0.0)
        out[us] = p
        vis[us] = True
        synth.add(us)

    _derive(out, vis)

    # 3) balance: shift every synthesized joint horizontally (in the XY ground
    #    plane) so the synthesized centre of gravity lands under the target
    #    (visible CoG / silhouette).
    if silhouette_xz is not None:
        # ``silhouette_xz`` is a horizontal (x, y) centroid in solver space.
        target = np.array([float(silhouette_xz[0]), float(silhouette_xz[1]), 0.0])
    else:
        target, _ = _cog(out, vis, only=set(out) - synth)
        if not np.any(target):
            target, _ = _cog(out, vis)

    synth_cog, mass = _cog(out, vis, only=synth)
    if mass > 0.0:
        delta = target - synth_cog
        delta[2] = 0.0  # never move vertically; balance is a horizontal shift
        for name in synth:
            if name in out:
                out[name] = out[name] + delta
        _derive(out, vis)

    return out, synth


def _derive(pts: Dict[str, np.ndarray], valid: Dict[str, bool]) -> None:
    """Recompute mid-hip, mid-shoulder and nose_up from base joints."""
    if "l_hip" in pts and "r_hip" in pts:
        pts["mid_hip"] = (np.asarray(pts["l_hip"]) + np.asarray(pts["r_hip"])) * 0.5
        valid["mid_hip"] = True
    if "l_shoulder" in pts and "r_shoulder" in pts:
        pts["mid_shoulder"] = (np.asarray(pts["l_shoulder"]) + np.asarray(pts["r_shoulder"])) * 0.5
        valid["mid_shoulder"] = True
    if "mid_hip" in pts and "mid_shoulder" in pts:
        up = np.asarray(pts["mid_shoulder"]) - np.asarray(pts["mid_hip"])
        n = np.linalg.norm(up)
        if n < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
            n = 1.0
        if "nose" in pts:
            pts["nose_up"] = np.asarray(pts["nose"]) + up / n * 0.1
            valid["nose_up"] = True


def standing_pose(
    anchor: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], Set[str]]:
    """Return a full standing pose (used when a frame has no detected person)."""
    base = np.asarray(anchor, dtype=np.float64) if anchor is not None else np.zeros(3, dtype=np.float64)
    pts: Dict[str, np.ndarray] = {}
    valid: Dict[str, bool] = {}
    for name, off in STANDING.items():
        if name == "mid_hip":
            continue
        pts[name] = base + np.array(off, dtype=np.float64) * scale
        valid[name] = True
    _derive(pts, valid)
    return pts, set(pts)
