"""Exporters for pure-Python / backend paths.

Contents
--------
* :class:`BVHWriter` — a dependency-free BVH text writer (tested).
* :func:`bvh_from_solved` — writes a :class:`SolvedSequence` to BVH.
* :func:`write_csv`, :func:`export_npz`, :func:`export_jsonl` — raw joint data.

FBX/GLB export is done through Blender (see ``addon/blender_bridge.py``); this
module never imports ``bpy``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .motion_solver import Skeleton, SolvedSequence, matrix_to_quat, SKELETON


# --- Euler ZXY helpers --------------------------------------------------
def euler_to_matrix(z: float, x: float, y: float) -> np.ndarray:
    """Matrix for the BVH channel order ``Zrotation Xrotation Yrotation``."""
    cz, sz = math.cos(z), math.sin(z)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return rz @ rx @ ry


def matrix_to_euler(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`euler_to_matrix`; returns ``(z, x, y)`` radians."""
    r = np.asarray(m, dtype=np.float64)
    sx = max(-1.0, min(1.0, float(r[2, 1])))
    x = math.asin(sx)
    if abs(sx) > 0.9999:  # gimbal lock
        y = 0.0
        z = math.atan2(-r[0, 2], r[0, 0])
    else:
        y = math.atan2(-r[2, 0], r[2, 2])
        z = math.atan2(-r[0, 1], r[1, 1])
    return np.array([z, x, y], dtype=np.float64)


@dataclass
class BVHJoint:
    name: str
    parent: Optional[str]
    offset: Tuple[float, float, float]
    children: List[str] = field(default_factory=list)


class BVHWriter:
    """Minimal, correct BVH text writer.

    Joints are given as ``(name, parent, offset)``. The root carries 6 channels
    (position + rotation); every other joint carries 3 rotation channels. The
    rotation channel order is ``Zrotation Xrotation Yrotation``.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = float(fps)
        self.joints: Dict[str, BVHJoint] = {}
        self.order: List[str] = []

    def add_joint(self, name: str, parent: Optional[str],
                  offset: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        if name in self.joints:
            raise ValueError("duplicate joint {!r}".format(name))
        self.joints[name] = BVHJoint(name, parent, tuple(float(v) for v in offset))
        self.order.append(name)
        if parent is not None:
            if parent not in self.joints:
                raise ValueError("parent {!r} must be added first".format(parent))
            self.joints[parent].children.append(name)

    @property
    def root(self) -> BVHJoint:
        for j in self.joints.values():
            if j.parent is None:
                return j
        raise ValueError("no root joint")

    def _hierarchy(self) -> str:
        lines: List[str] = ["HIERARCHY"]

        def emit(j: BVHJoint, indent: int) -> None:
            pad = "\t" * indent
            tag = "ROOT" if j.parent is None else "JOINT"
            channels = 6 if j.parent is None else 3
            lines.append("{}{} {}".format(pad, tag, j.name))
            lines.append("{}{{".format(pad))
            lines.append("{}OFFSET {} {} {}".format(
                pad, j.offset[0], j.offset[1], j.offset[2]))
            if j.parent is None:
                lines.append("{}CHANNELS 6 Xposition Yposition Zposition "
                             "Zrotation Xrotation Yrotation".format(pad))
            else:
                lines.append("{}CHANNELS 3 Zrotation Xrotation Yrotation".format(pad))
            for child in j.children:
                emit(self.joints[child], indent + 1)
            lines.append("{}}}".format(pad))

        emit(self.root, 0)
        return "\n".join(lines)

    def write(
        self,
        path: str,
        rotations: Dict[str, List[Tuple[float, float, float]]],
        root_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Write BVH.

        ``rotations[name]`` is a list over frames of ``(z, x, y)`` radians.
        ``root_positions`` is ``(frames, 3)``.
        """
        frames = len(next(iter(rotations.values()))) if rotations else 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = [self._hierarchy(), "MOTION", "Frames: {}".format(frames),
               "Frame Time: {:.8f}".format(1.0 / self.fps if self.fps else 1 / 30.0)]

        for t in range(frames):
            vals: List[float] = []
            for name in self.order:
                j = self.joints[name]
                if j.parent is None:
                    if root_positions is not None and t < len(root_positions):
                        rp = root_positions[t]
                        vals.extend([float(rp[0]), float(rp[1]), float(rp[2])])
                    else:
                        vals.extend([j.offset[0], j.offset[1], j.offset[2]])
                z, x, y = rotations[name][t]
                vals.extend([math.degrees(z), math.degrees(x), math.degrees(y)])
            out.append(" ".join("{:.6f}".format(v) for v in vals))

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")


def _relative_rotations(solved: SolvedSequence) -> Dict[str, List[Tuple[float, float, float]]]:
    """Per-bone local euler channels (ZXY) from solved world matrices.

    Each bone's local rotation is expressed relative to its parent's posed
    frame, so replaying the channels reproduces the motion's orientations.
    """
    rot: Dict[str, List[Tuple[float, float, float]]] = {
        b.name: [] for b in solved.skeleton.bones
    }
    prev_quat: Dict[str, np.ndarray] = {}
    for world in solved.world:
        for b in solved.skeleton.bones:
            if b.name not in world:
                continue
            w_rot = world[b.name][:3, :3]
            if b.parent and b.parent in world:
                p_rot = world[b.parent][:3, :3]
                local = p_rot.T @ w_rot
            else:
                local = w_rot
            # quaternion continuity to avoid euler flips
            q = matrix_to_quat(local)
            pq = prev_quat.get(b.name)
            if pq is not None and float(np.dot(pq, q)) < 0.0:
                q = -q
            prev_quat[b.name] = q
            e = matrix_to_euler(local)
            rot[b.name].append((float(e[0]), float(e[1]), float(e[2])))
    return rot


def _rest_offsets(skeleton: Skeleton) -> Dict[str, Tuple[float, float, float]]:
    rest = skeleton.rest_world()
    offsets: Dict[str, Tuple[float, float, float]] = {}
    for b in skeleton.bones:
        head = rest[b.name][:3, 3]
        if b.parent:
            parent_head = rest[b.parent][:3, 3]
            d = head - parent_head
        else:
            d = head
        offsets[b.name] = (float(d[0]), float(d[1]), float(d[2]))
    return offsets


def bvh_from_solved(solved: SolvedSequence, path: str) -> None:
    """Write a :class:`SolvedSequence` to BVH.

    Orientation-accurate: it replays the solved bone orientations through the
    BVH channel hierarchy. Offsets come from the capture rig's rest pose.
    """
    writer = BVHWriter(fps=solved.fps)
    offsets = _rest_offsets(solved.skeleton)
    for b in solved.skeleton.bones:
        writer.add_joint(b.name, b.parent, offsets[b.name])
    rotations = _relative_rotations(solved)
    root_pos = solved.root_positions
    writer.write(path, rotations, root_pos if len(root_pos) else None)


# --- Raw data exports ---------------------------------------------------
def write_csv(path: str, solved: SolvedSequence) -> None:
    """CSV with per-frame global joint positions (solver space, metres)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joints = solved.skeleton.names
    header = ["frame", "timestamp", "missing"]
    for j in joints:
        header += ["{}_x".format(j), "{}_y".format(j), "{}_z".format(j)]
    n = len(solved)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            world = solved.world[i]
            row: List[object] = [i, round(i / solved.fps, 6) if solved.fps else i,
                                 bool(solved.missing[i]) if i < len(solved.missing) else False]
            for j in joints:
                if j in world:
                    p = world[j][:3, 3]
                    row += [round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6)]
                else:
                    row += ["", "", ""]
            w.writerow(row)


def export_npz(path: str, solved: SolvedSequence) -> None:
    """NPZ with per-frame joint positions, rotations and confidence."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joints = solved.skeleton.names
    n = len(solved)
    pos = np.full((n, len(joints), 3), np.nan, dtype=np.float32)
    rot = np.full((n, len(joints), 4), np.nan, dtype=np.float32)
    for i in range(n):
        world = solved.world[i]
        local = solved.local[i] if i < len(solved.local) else {}
        for k, j in enumerate(joints):
            if j in world:
                pos[i, k] = world[j][:3, 3]
            if j in local:
                rot[i, k] = local[j][1]
    np.savez_compressed(
        path,
        joint_names=np.array(joints),
        positions=pos,
        rotations=rot,
        root_positions=solved.root_positions,
        is_missing=np.array(solved.missing, dtype=np.bool_),
        fps=np.array([solved.fps], dtype=np.float64),
    )


def export_jsonl(path: str, solved: SolvedSequence) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joints = solved.skeleton.names
    with open(path, "w", encoding="utf-8") as fh:
        meta = {"type": "meta", "fps": solved.fps, "joint_names": joints,
                "source": solved.source}
        fh.write(json.dumps(meta) + "\n")
        for i in range(len(solved)):
            world = solved.world[i]
            rec = {
                "type": "frame",
                "frame": i,
                "timestamp": round(i / solved.fps, 6) if solved.fps else i,
                "missing": solved.missing[i] if i < len(solved.missing) else False,
                "positions": {j: [round(float(v), 6) for v in world[j][:3, 3]]
                              for j in joints if j in world},
            }
            fh.write(json.dumps(rec) + "\n")
