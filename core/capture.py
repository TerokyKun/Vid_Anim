"""Frame sources: webcam, video file, image sequence, or a folder of videos.

Thin, defensive wrappers around OpenCV so the rest of the pipeline never has to
deal with ``cv2`` quirks or cryptic failure codes. Every failure path raises
:class:`CaptureError` with a human-readable message so Blender can show it
without crashing.

OpenCV is imported lazily: importing this module does not require ``cv2``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


class CaptureError(RuntimeError):
    """Raised for any recoverable capture problem (busy camera, bad codec...)."""


@dataclass
class Frame:
    index: int
    timestamp: float
    image: "np.ndarray"  # BGR uint8 HxWx3


class VideoSource:
    """Iterable frame source over a webcam, a video file or a folder of images."""

    def __init__(
        self,
        path: Union[str, int],
        fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        resize: int = 0,
        roi: Optional[Tuple[int, int, int, int]] = None,  # x, y, w, h
        backend: int = 0,
        is_camera: bool = False,
    ) -> None:
        self.path = path
        self.is_camera = is_camera
        self.roi = roi
        self.resize = int(resize or 0)
        self._cap = None
        self._fps = fps
        self._width = width
        self._height = height
        self._backend = backend
        self._total = 0
        self._index = 0

    # -- lifecycle --------------------------------------------------------
    def open(self) -> "VideoSource":
        import cv2

        if self.is_camera:
            cap = cv2.VideoCapture(int(self.path), self._backend) if isinstance(self.path, int) \
                else cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise CaptureError(
                    "Cannot open webcam {!r}. It may be busy or the index is "
                    "wrong. Try another index (0, 1, 2...) or close the app "
                    "that is using it.".format(self.path)
                )
        else:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise CaptureError(
                    "Cannot open video {!r}. The file may be missing or its "
                    "codec unsupported by the system FFmpeg/OpenCV build "
                    "(H.264/H.265 usually work).".format(self.path)
                )
        self._cap = cap
        if self._fps is None:
            detected = cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(detected) if detected and detected > 0 else 30.0
        self._width = int(self._width or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._height or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.is_camera else 0
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- properties -------------------------------------------------------
    @property
    def fps(self) -> float:
        return float(self._fps or 30.0)

    @property
    def width(self) -> int:
        return int(self._width or 0)

    @property
    def height(self) -> int:
        return int(self._height or 0)

    @property
    def total(self) -> int:
        return self._total

    # -- iteration --------------------------------------------------------
    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise CaptureError("Source is not open; call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        frame = self._postprocess(frame)
        ts = self._index / self.fps
        out = Frame(index=self._index, timestamp=ts, image=frame)
        self._index += 1
        return out

    def _postprocess(self, frame: "np.ndarray") -> "np.ndarray":
        import cv2

        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[max(0, y):y + h, max(0, x):x + w]
        if self.resize:
            h, w = frame.shape[:2]
            m = max(h, w)
            if m > self.resize:
                s = float(self.resize) / m
                frame = cv2.resize(
                    frame, (max(1, int(w * s)), max(1, int(h * s))),
                    interpolation=cv2.INTER_AREA,
                )
        return frame

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            self.open()
        while True:
            fr = self.read()
            if fr is None:
                break
            yield fr


def list_webcams(max_probe: int = 6) -> List[int]:
    """Probe indices ``0..max_probe-1`` and return ones that open successfully."""
    try:
        import cv2
    except ImportError:
        return []
    found: List[int] = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                found.append(i)
        finally:
            cap.release()
    return found


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")


def list_videos(folder: str) -> List[str]:
    """Sorted list of video files directly inside ``folder``."""
    if not os.path.isdir(folder):
        raise CaptureError("Not a folder: {!r}".format(folder))
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    return sorted(os.path.join(folder, f) for f in files)


def open_source(
    source: Union[str, int],
    is_camera: bool = False,
    **kwargs: object,
) -> VideoSource:
    """Convenience factory; validates and opens a source."""
    if is_camera:
        return VideoSource(source, is_camera=True, **kwargs).open()  # type: ignore[arg-type]
    if not os.path.exists(str(source)):
        raise CaptureError("Input not found: {!r}".format(source))
    return VideoSource(source, **kwargs).open()  # type: ignore[arg-type]


def open_image_sequence(folder: str, pattern: str = "*") -> List[str]:
    """Return a sorted list of image files in ``folder``."""
    if not os.path.isdir(folder):
        raise CaptureError("Not a folder: {!r}".format(folder))
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    images = [f for f in files
              if os.path.splitext(f)[1].lower() in
              (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")]
    if not images:
        raise CaptureError("No images found in {!r}".format(folder))
    return images
