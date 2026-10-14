"""Device detection and lightweight timing/profiling helpers.

Used to pick the fastest available backend / execution provider and to report
throughput (fps, ms/frame) to the UI and logs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


def cuda_available() -> bool:
    """Best-effort CUDA detection without importing heavy frameworks."""
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        return any("CUDA" in p or "Tensorrt" in p for p in provs)
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, check=True)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


@dataclass
class DeviceInfo:
    platform: str
    python: str
    cpu_count: int
    cuda: bool = False
    gpu_name: str = ""
    onnx_providers: list = field(default_factory=list)
    mediapipe: bool = False
    onnxruntime: bool = False
    opencv: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "platform": self.platform,
            "python": self.python,
            "cpu_count": self.cpu_count,
            "cuda": self.cuda,
            "gpu_name": self.gpu_name,
            "onnx_providers": list(self.onnx_providers),
            "mediapipe": self.mediapipe,
            "onnxruntime": self.onnxruntime,
            "opencv": self.opencv,
        }


def _gpu_name() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


def device_report() -> DeviceInfo:
    info = DeviceInfo(
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
    )
    try:
        import mediapipe  # noqa: F401
        info.mediapipe = True
    except Exception:  # noqa: BLE001
        info.mediapipe = False
    try:
        import onnxruntime as ort
        info.onnxruntime = True
        info.onnx_providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        info.onnxruntime = False
    try:
        import cv2
        info.opencv = cv2.__version__
    except Exception:  # noqa: BLE001
        info.opencv = ""
    info.gpu_name = _gpu_name()
    info.cuda = bool(info.gpu_name) or any(
        "CUDA" in p or "Tensorrt" in p for p in info.onnx_providers)
    return info


def suggest_device() -> str:
    return "cuda" if cuda_available() else "cpu"


class Timer:
    """Context manager measuring wall time of a block."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start


def throughput(frames: int, seconds: float) -> Dict[str, float]:
    seconds = max(1e-9, seconds)
    return {
        "frames": float(frames),
        "seconds": seconds,
        "fps": frames / seconds,
        "ms_per_frame": 1000.0 * seconds / max(1, frames),
    }
