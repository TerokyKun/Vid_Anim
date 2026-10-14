"""QUALITY MODE inference via ONNX Runtime (RTMPose / RTMO).

ONNX Runtime and the model files are optional. When ``onnxruntime`` is not
installed or no model file is present, this module raises :class:`InferenceError`
and the caller should fall back to FAST MODE (MediaPipe).

The execution provider is chosen automatically: CUDA (TensorRT optional) when
available, otherwise CPU. No PyTorch / MMCV / MMEngine dependency is used.
"""

from __future__ import annotations

import os
from typing import Callable, Iterator, Optional

import numpy as np

from . import model_store
from .pose_schema import MEDIAPIPE_NAMES, InferenceCache, PoseFrame, PoseSequence

# COCO-17 -> MediaPipe-33 index mapping (only the joints that overlap are filled;
# the rest stay missing and are handled by core.partial_body / smoothing).
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 2,    # left eye
    2: 5,    # right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}

COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class InferenceError(RuntimeError):
    pass


def onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def available_providers() -> list:
    if not onnx_available():
        return []
    import onnxruntime as ort
    return list(ort.get_available_providers())


def _select_providers(device: str) -> list:
    prov = available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    ordered = []
    for pref in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
        if pref in prov:
            ordered.append(pref)
    ordered.append("CPUExecutionProvider")
    return ordered


class OnnxPose:
    """Thin wrapper around an ONNX Runtime session doing top-down pose."""

    def __init__(self, model_id: str = "onnx.rtmpose.body", device: str = "auto",
                 model_path: Optional[str] = None) -> None:
        if not onnx_available():
            raise InferenceError(
                "onnxruntime is not installed. QUALITY MODE needs 'onnxruntime'"
                " (or onnxruntime-gpu) in the backend venv."
            )
        import onnxruntime as ort

        self.model_id = model_id
        if model_path is None:
            try:
                model_path = model_store.resolve_model(model_id, download=True)
            except model_store.ModelError as exc:
                raise InferenceError(str(exc)) from exc
        self.model_path = model_path

        self.providers = _select_providers(device)
        self.device = "cuda" if any("CUDA" in p or "Tensorrt" in p for p in self.providers) else "cpu"
        self._session = ort.InferenceSession(model_path, providers=self.providers)
        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name
        shape = self._input.shape
        try:
            self._in_h = int(shape[2])
            self._in_w = int(shape[3])
        except Exception:  # noqa: BLE001
            self._in_h, self._in_w = 256, 192

    def close(self) -> None:
        self._session = None

    def __enter__(self) -> "OnnxPose":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def process(self, bgr: np.ndarray, frame_index: int, timestamp_ms: int,
                bbox: Optional[tuple] = None) -> PoseFrame:
        import cv2

        h, w = bgr.shape[:2]
        if bbox is None:
            bbox = (0, 0, w, h)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return PoseFrame.missing(frame_index, timestamp_ms / 1000.0)

        crop = bgr[y1:y2, x1:x2]
        inp, scale = self._preprocess(crop)
        outputs = self._session.run(None, {self._input_name: inp})
        kpts, scores = self._postprocess(outputs, crop.shape[:2], (x1, y1))
        return self._to_frame(kpts, scores, frame_index, timestamp_ms, w, h)

    def _preprocess(self, crop: np.ndarray):
        import cv2

        resized = cv2.resize(crop, (self._in_w, self._in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        rgb = (rgb - mean) / std
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        return blob, None

    def _postprocess(self, outputs, crop_hw, offset):
        crop_h, crop_w = crop_hw
        ox, oy = offset
        n = len(COCO_NAMES)
        if len(outputs) >= 2:
            simcc_x, simcc_y = outputs[0], outputs[1]
            kx = int(np.argmax(simcc_x[0], axis=-1))
            ky = int(np.argmax(simcc_y[0], axis=-1))
            sx = simcc_x[0]
            sy = simcc_y[0]
            scores = np.minimum(sx.max(axis=-1), sy.max(axis=-1))
            kpts = np.stack([
                kx / sx.shape[-1] * crop_w + ox,
                ky / sy.shape[-1] * crop_h + oy,
            ], axis=-1).astype(np.float32)
            return kpts[:n], scores[:n].astype(np.float32)
        out = outputs[0].reshape(-1, n, 3) if outputs[0].ndim > 2 else outputs[0].reshape(n, 3)
        out = out[0]
        kpts = out[:, :2].astype(np.float32)
        kpts[:, 0] = kpts[:, 0] / self._in_w * crop_w + ox
        kpts[:, 1] = kpts[:, 1] / self._in_h * crop_h + oy
        scores = out[:, 2].astype(np.float32)
        return kpts, scores

    def _to_frame(self, kpts, scores, frame_index, timestamp_ms, w, h):
        l2 = np.zeros((len(MEDIAPIPE_NAMES), 3), dtype=np.float32)
        conf = np.zeros((len(MEDIAPIPE_NAMES),), dtype=np.float32)
        for ci, mi in COCO_TO_MEDIAPIPE.items():
            if ci >= len(kpts):
                continue
            l2[mi, 0] = kpts[ci, 0] / max(1, w)
            l2[mi, 1] = kpts[ci, 1] / max(1, h)
            conf[mi] = scores[ci] if ci < len(scores) else 0.0
            l2[mi, 2] = conf[mi]
        meta = {"device": self.device, "model": self.model_id,
                "kpts_px": kpts.tolist(), "scores": scores.tolist()}
        return PoseFrame(
            frame=frame_index,
            timestamp=timestamp_ms / 1000.0,
            landmarks_2d=l2,
            landmarks_3d=None,
            confidence=conf,
            is_missing=bool(conf.max() < 0.3),
            backend="onnx",
            meta=meta,
        )


def run_source(
    source,
    out_path: Optional[str] = None,
    is_camera: bool = False,
    model_id: str = "onnx.rtmpose.body",
    device: str = "auto",
    step: int = 1,
    resize: int = 0,
    use_cache: bool = True,
    max_frames: int = 0,
    bbox_fn: Optional[Callable[[np.ndarray], Optional[tuple]]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PoseSequence:
    from .capture import VideoSource

    cache = InferenceCache(out_path) if out_path else None
    key = None
    if cache is not None and not is_camera and use_cache:
        key = cache.make_key(str(source), kind=model_id, step=step, resize=resize,
                             backend="onnx")
        if cache.is_valid(key):
            seq = cache.load()
            if len(seq):
                return seq

    vs = VideoSource(source, is_camera=is_camera, resize=resize)
    vs.open()
    try:
        pose = OnnxPose(model_id=model_id, device=device)
    except InferenceError:
        vs.close()
        raise
    frames = []
    try:
        for fr in vs:
            if fr.index % max(1, step) != 0:
                continue
            bbox = bbox_fn(fr.image) if bbox_fn else None
            frames.append(pose.process(fr.image, fr.index, int(round(fr.timestamp * 1000.0)),
                                       bbox=bbox))
            if progress is not None:
                progress(fr.index + 1, vs.total)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        pose.close()
        vs.close()

    seq = PoseSequence(frames=frames, fps=vs.fps, source=str(source), backend="onnx",
                       width=vs.width, height=vs.height,
                       meta={"model": model_id, "device": device})
    if cache is not None and key is not None:
        cache.save(seq, key)
    return seq
