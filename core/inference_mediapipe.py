"""FAST MODE inference via MediaPipe Pose Landmarker.

Wraps ``mediapipe.tasks.vision.PoseLandmarker`` and converts its output into the
canonical :class:`core.pose_schema.PoseFrame`. MediaPipe provides both normalized
2D landmarks (with visibility) and metric 3D "world" landmarks (origin at hips).

MediaPipe is imported lazily; importing this module without MediaPipe installed
is fine.
"""

from __future__ import annotations

import os
from typing import Callable, Iterator, Optional

import numpy as np

from . import model_store
from .capture import CaptureError, VideoSource
from .pose_schema import MEDIAPIPE_NAMES, InferenceCache, PoseFrame, PoseSequence

_KIND_TO_MODEL = {
    "lite": "mediapipe.pose_landmarker.lite",
    "full": "mediapipe.pose_landmarker.full",
    "heavy": "mediapipe.pose_landmarker.heavy",
}


class InferenceError(RuntimeError):
    pass


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class MediaPipePose:
    """Stateful pointer wrapper around a MediaPipe PoseLandmarker (VIDEO mode)."""

    def __init__(
        self,
        kind: str = "full",
        min_conf: float = 0.5,
        delegate: str = "auto",
        model_id: Optional[str] = None,
    ) -> None:
        if not mediapipe_available():
            raise InferenceError(
                "MediaPipe is not installed in this Python. FAST MODE needs "
                "'mediapipe'. Install it in the backend venv (backend/setup_*)."
            )
        import mediapipe as mp

        self._mp = mp
        self.kind = kind
        self.min_conf = float(min_conf)
        mid = model_id or _KIND_TO_MODEL.get(kind, _KIND_TO_MODEL["full"])
        try:
            model_path = model_store.resolve_model(mid, download=True)
        except model_store.ModelError as exc:
            raise InferenceError(str(exc)) from exc

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        base_kwargs = {"model_asset_path": model_path}
        if delegate in ("gpu", "auto"):
            try:
                base_kwargs["delegate"] = BaseOptions.Delegate.GPU
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(**base_kwargs),
                    running_mode=RunningMode.VIDEO,
                    min_pose_detection_confidence=self.min_conf,
                    min_tracking_confidence=self.min_conf,
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
                self.delegate = "GPU"
            except Exception:  # noqa: BLE001 - fall back to CPU
                base_kwargs.pop("delegate", None)
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(**base_kwargs),
                    running_mode=RunningMode.VIDEO,
                    min_pose_detection_confidence=self.min_conf,
                    min_tracking_confidence=self.min_conf,
                )
                self._landmarker = PoseLandmarker.create_from_options(options)
                self.delegate = "CPU"
        else:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(**base_kwargs),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=self.min_conf,
                min_tracking_confidence=self.min_conf,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self.delegate = "CPU"

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "MediaPipePose":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def process(self, bgr: np.ndarray, frame_index: int, timestamp_ms: int) -> PoseFrame:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            res = self._landmarker.detect_for_video(mp_img, int(timestamp_ms))
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("MediaPipe inference failed: {}".format(exc)) from exc

        if not res.pose_landmarks:
            return PoseFrame.missing(frame_index, timestamp_ms / 1000.0)

        lm = res.pose_landmarks[0]
        l2 = np.array([[p.x, p.y, getattr(p, "visibility", 1.0)] for p in lm], dtype=np.float32)
        conf = np.array([getattr(p, "visibility", 1.0) for p in lm], dtype=np.float32)
        l3 = None
        if res.pose_world_landmarks:
            w = res.pose_world_landmarks[0]
            l3 = np.array([[p.x, p.y, p.z] for p in w], dtype=np.float32)
        return PoseFrame(
            frame=frame_index,
            timestamp=timestamp_ms / 1000.0,
            landmarks_2d=l2,
            landmarks_3d=l3,
            confidence=conf,
            is_missing=l3 is None,
            backend="mediapipe",
            meta={"delegate": self.delegate, "model": self.kind},
        )


def iter_inference(
    source: VideoSource,
    pose: MediaPipePose,
    step: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Iterator[PoseFrame]:
    """Yield a :class:`PoseFrame` for every ``step``-th frame of ``source``."""
    step = max(1, int(step))
    for fr in source:
        if fr.index % step != 0:
            continue
        ts_ms = int(round(fr.timestamp * 1000.0))
        yield pose.process(fr.image, fr.index, ts_ms)
        if progress is not None:
            progress(fr.index + 1, source.total)


def run_source(
    source,
    out_path: Optional[str] = None,
    is_camera: bool = False,
    kind: str = "full",
    min_conf: float = 0.5,
    step: int = 1,
    resize: int = 0,
    delegate: str = "auto",
    use_cache: bool = True,
    max_frames: int = 0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PoseSequence:
    """Run FAST MODE inference over a video/webcam and return a PoseSequence.

    If ``out_path`` is given and the cache is valid, the cached sequence is
    returned without re-running inference.
    """
    cache = InferenceCache(out_path) if out_path else None
    if cache is not None and not is_camera and use_cache:
        key = cache.make_key(str(source), kind=kind, step=step, resize=resize, min_conf=min_conf)
        if cache.is_valid(key):
            seq = cache.load()
            if len(seq):
                return seq
    else:
        key = None

    vs = VideoSource(source, is_camera=is_camera, resize=resize)
    vs.open()
    try:
        pose = MediaPipePose(kind=kind, min_conf=min_conf, delegate=delegate)
    except InferenceError:
        vs.close()
        raise

    frames = []
    try:
        total = vs.total
        for fr in vs:
            if fr.index % max(1, step) != 0:
                continue
            frames.append(pose.process(fr.image, fr.index, int(round(fr.timestamp * 1000.0))))
            if progress is not None:
                progress(fr.index + 1, total)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        pose.close()
        vs.close()

    seq = PoseSequence(
        frames=frames,
        fps=vs.fps,
        source=str(source),
        backend="mediapipe",
        width=vs.width,
        height=vs.height,
        meta={"model": kind, "step": step, "resize": resize},
    )
    if cache is not None and key is not None:
        cache.save(seq, key)
    return seq
