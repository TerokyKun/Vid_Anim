"""Temporal smoothing / gap-filling for pose sequences.

Implements the *One Euro filter* (Casiez et al., CHI 2012), a simple linear
interpolator for missing frames, and a convenience :func:`smooth_sequence`
that operates on a :class:`core.pose_schema.PoseSequence`.

All functions are pure ``numpy`` — no SciPy, no ``bpy`` — so they can run both
inside Blender and in the backend venv.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .pose_schema import PoseFrame, PoseSequence


class OneEuroFilter:
    """Low-latency adaptive low-pass filter for noisy, real-time signals.

    A single filter instance filters an N-dimensional signal over time. Smaller
    ``min_cutoff`` removes more jitter at rest; larger ``beta`` reacts faster to
    fast motion.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = max(1e-3, float(freq))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def _smoothing_factor(self, speed: float) -> float:
        cutoff = self.min_cutoff + self.beta * speed
        return self._alpha(cutoff, self.freq)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if timestamp is not None and self._t_prev is not None:
            dt = max(1e-6, float(timestamp) - self._t_prev)
            freq = 1.0 / dt
        else:
            freq = self.freq
        self._t_prev = timestamp

        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.astype(np.float32)

        dx = (x - self._x_prev) * freq
        a_d = self._alpha(self.d_cutoff, freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = float(np.linalg.norm(dx_hat))
        a = self._smoothing_factor(speed)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat.astype(np.float32)


def interpolate_missing(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Linearly interpolate NaNs along ``axis``; clamp-extrapolate at the edges.

    ``arr`` is normally ``(frames, landmarks, dims)``. Gadget frames that are
    fully NaN (person absent) are filled so downstream solvers never see NaNs.
    """
    a = np.asarray(arr, dtype=np.float32).copy()
    if a.size == 0:
        return a
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    flat = moved.reshape(n, -1)
    idx = np.arange(n)

    for col in range(flat.shape[1]):
        colv = flat[:, col]
        valid = ~np.isnan(colv)
        if valid.all():
            continue
        if not valid.any():
            flat[:, col] = 0.0
            continue
        flat[:, col] = np.interp(idx, idx[valid], colv[valid])
    return np.moveaxis(flat.reshape(moved.shape), 0, axis)


def smooth_array(
    arr: np.ndarray,
    freq: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.01,
    axis: int = 0,
) -> np.ndarray:
    """Apply a per-element One Euro filter along ``axis`` of ``arr``.

    NaNs are interpolated first, then each element is filtered independently
    along the time axis.
    """
    a = interpolate_missing(arr, axis=axis)
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    flat = moved.reshape(n, -1).astype(np.float64)
    out = np.empty_like(flat)
    filters = [OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta) for _ in range(flat.shape[1])]
    for t in range(n):
        ts = t / freq if freq else float(t)
        for c in range(flat.shape[1]):
            out[t, c] = filters[c](flat[t, c : c + 1], ts)[0]
    return np.moveaxis(out.reshape(moved.shape).astype(np.float32), 0, axis)


def smooth_sequence(
    sequence: PoseSequence,
    min_cutoff: float = 1.0,
    beta: float = 0.01,
    smooth_3d: bool = True,
    smooth_2d: bool = False,
    inplace: bool = False,
) -> PoseSequence:
    """Return a smoothed copy of ``sequence`` (or mutate it if ``inplace``).

    Missing frames are interpolated and no longer flagged ``is_missing`` only if
    they had neighbours on both sides — isolated dropouts are bridged, but a
    long cluster of missing frames is kept as zeros and still reported missing.
    """
    seq = sequence if inplace else PoseSequence(
        frames=list(sequence.frames),
        fps=sequence.fps,
        source=sequence.source,
        backend=sequence.backend,
        landmark_names=list(sequence.landmark_names),
        schema_version=sequence.schema_version,
        width=sequence.width,
        height=sequence.height,
        meta=dict(sequence.meta),
    )

    def _stack(attr: str) -> Optional[np.ndarray]:
        arrays = [getattr(f, attr) for f in seq.frames]
        if all(a is None for a in arrays):
            return None
        k = max((a.shape[0] if a is not None else 0) for a in arrays)
        if k == 0:
            return None
        dim = next(a.shape[1] for a in arrays if a is not None)
        out = np.full((len(arrays), k, dim), np.nan, dtype=np.float32)
        for i, a in enumerate(arrays):
            if a is not None:
                m = min(k, a.shape[0])
                out[i, :m] = a[:m]
        return out

    def _scatter(attr: str, data: Optional[np.ndarray]) -> None:
        if data is None:
            return
        for i, fr in enumerate(seq.frames):
            setattr(fr, attr, data[i].copy())

    if smooth_3d:
        data = _stack("landmarks_3d")
        if data is not None:
            _scatter("landmarks_3d", smooth_array(
                data, freq=seq.fps, min_cutoff=min_cutoff, beta=beta))
    if smooth_2d:
        data = _stack("landmarks_2d")
        if data is not None:
            xy = smooth_array(data[:, :, :2], freq=seq.fps,
                              min_cutoff=min_cutoff, beta=beta)
            for i, fr in enumerate(seq.frames):
                if fr.landmarks_2d is None:
                    continue
                vis = fr.landmarks_2d[:, 2:3]
                fr.landmarks_2d = np.concatenate([xy[i], vis], axis=1).astype(np.float32)

    seq.meta = dict(seq.meta)
    seq.meta["smoothed"] = {"min_cutoff": min_cutoff, "beta": beta}
    return seq


def foot_lock(
    joints: np.ndarray,
    foot_indices: list,
    ground_eps: float = 0.03,
    speed_eps: float = 0.05,
    fps: float = 30.0,
) -> np.ndarray:
    """Simple ground-contact foot locking.

    ``joints`` is ``(frames, joints, 3)``. For each foot, frames are found where
    the foot is within ``ground_eps`` of the lowest observed height *and* moving
    slowly; within those runs the foot position is held at the run's first
    position to remove sliding. Returns a modified copy.
    """
    out = np.array(joints, dtype=np.float32, copy=True)
    if out.ndim != 3 or out.shape[0] == 0:
        return out
    dt = 1.0 / max(1e-3, fps)
    for j in foot_indices:
        if j < 0 or j >= out.shape[1]:
            continue
        track = out[:, j, :]
        heights = track[:, 1]
        floor = np.nanmin(heights)
        speed = np.linalg.norm(np.gradient(track, axis=0), axis=1) / dt
        contact = (heights - floor <= ground_eps) & (speed <= speed_eps)
        i = 0
        n = out.shape[0]
        while i < n:
            if not contact[i]:
                i += 1
                continue
            start = i
            while i < n and contact[i]:
                i += 1
            anchor = out[start, j, :].copy()
            out[start:i, j, :] = anchor
    return out
