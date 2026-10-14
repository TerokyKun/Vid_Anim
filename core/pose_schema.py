"""PoseFrame / PoseSequence — the canonical in-memory pose format.

Design goals
------------
* Dependency-light: only ``numpy`` is required.
* Interoperable: reads the legacy single-JSON format produced by the first
  generation of the backend (``{source, fps, frames:[{t, landmarks, world}]}``)
  *and* a streaming JSONL/NPZ cache produced by the new backend.
* Deterministic caching: :class:`InferenceCache` lets the pipeline skip the
  expensive inference step when nothing about the input changed.

The JSON schema is versioned via :data:`SCHEMA_VERSION`. Consumers must ignore
unknown keys so newer writers stay backward compatible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

SCHEMA_VERSION = "vid_an.pose.v1"

# MediaPipe Pose Landmarker: 33 landmarks, stable ordering.
MEDIAPIPE_NAMES: List[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _as_float_array(value: Any, ndim: int) -> Optional[np.ndarray]:
    """Best-effort coercion of a nested list into a float32 ndarray."""
    if value is None:
        return None
    # Support legacy dict format: list of {"x":..., "y":..., "z":..., "v":...}
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            n = len(value)
            out = np.zeros((n, ndim), dtype=np.float32)
            for i, v in enumerate(value):
                if "x" in v:
                    out[i, 0] = float(v["x"])
                if "y" in v:
                    out[i, 1] = float(v["y"])
                if ndim > 2:
                    if "z" in v:
                        out[i, 2] = float(v["z"])
                    elif "v" in v:
                        out[i, 2] = float(v["v"])
                    elif "visibility" in v:
                        out[i, 2] = float(v["visibility"])
            return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, ndim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, ndim)
    return arr


@dataclass
class PoseFrame:
    """A single frame of pose inference.

    ``landmarks_2d`` is ``(K, 3)`` = ``x, y, visibility`` in normalized image
    coordinates; ``landmarks_3d`` is ``(K, 3)`` = ``x, y, z`` in metres with the
    origin at the hips (MediaPipe world landmarks). Either may be ``None``.

    A frame with no detected person is represented with ``is_missing=True`` and
    empty/``None`` landmark arrays rather than raising — a missing person is a
    normal condition, not an error.
    """

    frame: int
    timestamp: float
    person_id: int = 0
    landmarks_2d: Optional[np.ndarray] = None
    landmarks_3d: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    is_missing: bool = False
    backend: str = "mediapipe"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_landmarks(self) -> int:
        for arr in (self.landmarks_2d, self.landmarks_3d, self.confidence):
            if arr is not None:
                return int(arr.shape[0])
        return 0

    @property
    def mean_confidence(self) -> float:
        if self.confidence is None or self.confidence.size == 0:
            return 0.0
        return float(np.mean(self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        def _lores(arr: Optional[np.ndarray]) -> Optional[List[List[float]]]:
            if arr is None:
                return None
            return [[round(float(v), 6) for v in row] for row in arr]

        return {
            "frame": int(self.frame),
            "timestamp": round(float(self.timestamp), 6),
            "person_id": int(self.person_id),
            "landmarks_2d": _lores(self.landmarks_2d),
            "landmarks_3d": _lores(self.landmarks_3d),
            "confidence": None if self.confidence is None
            else [round(float(v), 6) for v in self.confidence],
            "is_missing": bool(self.is_missing),
            "backend": self.backend,
            "meta": self.meta or {},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PoseFrame":
        return cls(
            frame=int(d.get("frame", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
            person_id=int(d.get("person_id", 0)),
            landmarks_2d=_as_float_array(d.get("landmarks_2d"), 3),
            landmarks_3d=_as_float_array(d.get("landmarks_3d"), 3),
            confidence=_as_float_array(d.get("confidence"), 1).reshape(-1)
            if d.get("confidence") is not None else None,
            is_missing=bool(d.get("is_missing", False)),
            backend=str(d.get("backend", "mediapipe")),
            meta=dict(d.get("meta", {}) or {}),
        )

    @classmethod
    def missing(cls, frame: int, timestamp: float, **kwargs: Any) -> "PoseFrame":
        return cls(frame=frame, timestamp=timestamp, is_missing=True, **kwargs)


@dataclass
class PoseSequence:
    """An ordered collection of :class:`PoseFrame` plus sequence metadata."""

    frames: List[PoseFrame]
    fps: float
    source: str = ""
    backend: str = "mediapipe"
    landmark_names: List[str] = field(default_factory=lambda: list(MEDIAPIPE_NAMES))
    schema_version: str = SCHEMA_VERSION
    width: int = 0
    height: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[PoseFrame]:
        return iter(self.frames)

    def stats(self) -> Dict[str, Any]:
        n = len(self.frames)
        missing = sum(1 for f in self.frames if f.is_missing or f.landmarks_3d is None)
        confs = [f.mean_confidence for f in self.frames if not f.is_missing]
        return {
            "frames": n,
            "missing": missing,
            "present": n - missing,
            "missing_ratio": (missing / n) if n else 0.0,
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "fps": self.fps,
            "duration_s": (n / self.fps) if self.fps else 0.0,
        }

    # -- JSONL streaming cache -------------------------------------------
    def to_jsonl(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            header = {
                "type": "meta",
                "schema_version": self.schema_version,
                "fps": self.fps,
                "source": self.source,
                "backend": self.backend,
                "landmark_names": self.landmark_names,
                "width": self.width,
                "height": self.height,
                "meta": self.meta,
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for fr in self.frames:
                rec = fr.to_dict()
                rec["type"] = "frame"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "PoseSequence":
        frames: List[PoseFrame] = []
        header: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "meta" or (
                    "landmarks_2d" not in rec and "frames" not in rec and "frame" not in rec
                ):
                    header = rec
                    continue
                frames.append(PoseFrame.from_dict(rec))
        return cls(
            frames=frames,
            fps=float(header.get("fps", 30.0)),
            source=header.get("source", ""),
            backend=header.get("backend", "mediapipe"),
            landmark_names=header.get("landmark_names", list(MEDIAPIPE_NAMES)),
            schema_version=header.get("schema_version", SCHEMA_VERSION),
            width=int(header.get("width", 0)),
            height=int(header.get("height", 0)),
            meta=dict(header.get("meta", {}) or {}),
        )

    # -- Legacy single-JSON (first-gen backend) --------------------------
    @classmethod
    def from_legacy(cls, data: Dict[str, Any]) -> "PoseSequence":
        raw = data.get("frames", [])
        fps = float(data.get("fps", 30.0))
        frames: List[PoseFrame] = []
        for i, fr in enumerate(raw):
            t = fr.get("t")
            if t is None:
                t = i / fps if fps else float(i)
            landmarks_2d = fr.get("landmarks") or None
            world = fr.get("world") or None
            missing = not landmarks_2d and not world
            frames.append(
                PoseFrame(
                    frame=i,
                    timestamp=float(t),
                    landmarks_2d=_as_float_array(landmarks_2d, 3) if landmarks_2d else None,
                    landmarks_3d=_as_float_array(world, 3) if world else None,
                    confidence=_as_float_array(
                        [[p.get("v", 1.0)] for p in landmarks_2d], 1
                    ).reshape(-1) if landmarks_2d else None,
                    is_missing=missing,
                    backend="mediapipe-legacy",
                )
            )
        return cls(
            frames=frames,
            fps=fps,
            source=data.get("source", ""),
            backend="mediapipe-legacy",
        )

    # -- NPZ (compact binary) --------------------------------------------
    def to_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        n = len(self.frames)
        k = len(self.landmark_names)
        l2 = np.full((n, k, 3), np.nan, dtype=np.float32)
        l3 = np.full((n, k, 3), np.nan, dtype=np.float32)
        conf = np.full((n, k), np.nan, dtype=np.float32)
        ts = np.zeros((n,), dtype=np.float64)
        ids = np.zeros((n,), dtype=np.int32)
        miss = np.zeros((n,), dtype=np.bool_)
        for i, fr in enumerate(self.frames):
            ts[i] = fr.timestamp
            ids[i] = fr.person_id
            miss[i] = fr.is_missing
            if fr.landmarks_2d is not None:
                m = min(k, fr.landmarks_2d.shape[0])
                l2[i, :m] = fr.landmarks_2d[:m]
            if fr.landmarks_3d is not None:
                m = min(k, fr.landmarks_3d.shape[0])
                l3[i, :m] = fr.landmarks_3d[:m]
            if fr.confidence is not None:
                m = min(k, fr.confidence.shape[0])
                conf[i, :m] = fr.confidence[:m]
        meta = json.dumps(
            {
                "schema_version": self.schema_version,
                "source": self.source,
                "backend": self.backend,
                "fps": self.fps,
                "landmark_names": self.landmark_names,
                "width": self.width,
                "height": self.height,
                "meta": self.meta,
            }
        )
        np.savez_compressed(
            path,
            landmarks_2d=l2,
            landmarks_3d=l3,
            confidence=conf,
            timestamp=ts,
            person_id=ids,
            is_missing=miss,
            meta=np.array(meta),
        )

    @classmethod
    def from_npz(cls, path: str) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            l2 = z["landmarks_2d"]
            l3 = z["landmarks_3d"]
            conf = z["confidence"]
            ts = z["timestamp"]
            ids = z["person_id"]
            miss = z["is_missing"].astype(bool)
            n = l2.shape[0]

            def _clean(arr: np.ndarray) -> Optional[np.ndarray]:
                row = arr
                if row.shape[1] == 0:
                    return None
                # a fully-NaN row means "no data"
                return None if np.all(np.isnan(row)) else row

            frames = []
            for i in range(n):
                frames.append(
                    PoseFrame(
                        frame=i,
                        timestamp=float(ts[i]),
                        person_id=int(ids[i]),
                        landmarks_2d=_clean(l2[i]),
                        landmarks_3d=_clean(l3[i]),
                        confidence=_clean(conf[i].reshape(-1, 1)).reshape(-1)
                        if not np.all(np.isnan(conf[i])) else None,
                        is_missing=bool(miss[i]),
                        backend=meta.get("backend", "mediapipe"),
                    )
                )
        return cls(
            frames=frames,
            fps=float(meta.get("fps", 30.0)),
            source=meta.get("source", ""),
            backend=meta.get("backend", "mediapipe"),
            landmark_names=meta.get("landmark_names", list(MEDIAPIPE_NAMES)),
            schema_version=meta.get("schema_version", SCHEMA_VERSION),
            width=int(meta.get("width", 0)),
            height=int(meta.get("height", 0)),
            meta=dict(meta.get("meta", {}) or {}),
        )


def load_sequence(path: str) -> PoseSequence:
    """Auto-detect JSONL, NPZ, or legacy single-JSON and load accordingly."""
    lower = path.lower()
    if lower.endswith(".npz"):
        return PoseSequence.from_npz(path)
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return PoseSequence.from_jsonl(path)
    # .json — could be legacy object; also accept JSONL saved under .json
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.lstrip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return PoseSequence.from_jsonl(path)
    if isinstance(obj, dict) and "frames" in obj:
        return PoseSequence.from_legacy(obj)
    # a dict without "frames" is a single meta/frame record -> JSONL-ish
    if stripped.startswith("{") and "\n" in text.strip():
        return PoseSequence.from_jsonl(path)
    return PoseSequence.from_legacy(obj if isinstance(obj, dict) else {})


class InferenceCache:
    """Content-addressed cache of an inference result.

    The cache key folds in the input file identity (size+mtime), the backend,
    model and frame-decimation parameters. Re-running the pipeline with the same
    inputs reuses the cached sequence instead of recomputing inference.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.key_path = path + ".key.json"

    @staticmethod
    def make_key(source: str, **params: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {"source": os.path.abspath(source)}
        try:
            st = os.stat(source)
            info["size"] = int(st.st_size)
            info["mtime"] = int(st.st_mtime)
        except OSError:
            info["size"] = -1
            info["mtime"] = -1
        info["params"] = {k: params[k] for k in sorted(params)}
        return info

    def is_valid(self, key: Dict[str, Any]) -> bool:
        if not (os.path.exists(self.path) and os.path.exists(self.key_path)):
            return False
        try:
            with open(self.key_path, "r", encoding="utf-8") as fh:
                return json.load(fh) == key
        except (OSError, json.JSONDecodeError):
            return False

    def load(self) -> PoseSequence:
        return load_sequence(self.path)

    def save(self, sequence: PoseSequence, key: Dict[str, Any]) -> None:
        if self.path.lower().endswith(".npz"):
            sequence.to_npz(self.path)
        else:
            sequence.to_jsonl(self.path)
        with open(self.key_path, "w", encoding="utf-8") as fh:
            json.dump(key, fh, ensure_ascii=False, indent=2)

    def invalidate(self) -> None:
        for p in (self.path, self.key_path):
            try:
                os.remove(p)
            except OSError:
                pass


# Module-level convenience API (matches test-style calls)
def to_jsonl(path: str, sequence: PoseSequence | None = None) -> None:
    if sequence is None:
        raise TypeError("sequence must be provided")
    sequence.to_jsonl(path)


def from_jsonl(path: str) -> PoseSequence:
    return PoseSequence.from_jsonl(path)


def to_npz(path: str, sequence: PoseSequence | None = None) -> None:
    if sequence is None:
        raise TypeError("sequence must be provided")
    sequence.to_npz(path)


def from_npz(path: str) -> PoseSequence:
    return PoseSequence.from_npz(path)
