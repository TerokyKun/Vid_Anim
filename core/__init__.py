"""vid_an core — pure-Python motion-capture building blocks.

This package is intentionally free of ``bpy`` and of heavy ML frameworks.
It is shared by the Blender add-on (``blender_addon/vid_an``), by the optional
MediaPipe/ONNX backend (``backend/``) and by the test-suite.

Only ``numpy`` is a hard requirement here. Optional dependencies (OpenCV,
MediaPipe, ONNX Runtime) are imported lazily inside the modules that need them
so that importing :mod:`core` never fails when they are missing.
"""

from . import partial_body
from . import pose_schema
from . import smoothing
from . import motion_solver
from . import capture
from . import model_store
from . import exporter
from . import retarget_map
from . import profiling
from .pose_schema import (
    SCHEMA_VERSION,
    MEDIAPIPE_NAMES,
    PoseFrame,
    PoseSequence,
)
from .smoothing import OneEuroFilter, interpolate_missing, smooth_sequence
from .motion_solver import (
    SKELETON,
    Bone,
    Skeleton,
    build_joints,
    solve_sequence,
    rotation_from_direction,
)

__all__ = [
    "SCHEMA_VERSION",
    "MEDIAPIPE_NAMES",
    "PoseFrame",
    "PoseSequence",
    "OneEuroFilter",
    "interpolate_missing",
    "smooth_sequence",
    "SKELETON",
    "Bone",
    "Skeleton",
    "build_joints",
    "solve_sequence",
    "rotation_from_direction",
    "partial_body",
    "pose_schema",
    "smoothing",
    "motion_solver",
    "capture",
    "model_store",
    "exporter",
    "retarget_map",
    "profiling",
]

__version__ = "0.2.0"
