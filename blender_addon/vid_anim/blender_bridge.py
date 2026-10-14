"""Blender <-> core glue: import/export helpers used by the operators.

Keeps every ``bpy`` call in one place. Native Blender exporters are preferred for
BVH/FBX/GLB because they produce fully correct rest-pose aware files; the pure
``core.exporter`` writer is used only for standalone/backend output.
"""

from __future__ import annotations

import json
import os
import tempfile

import bpy

from ._core_bridge import core


def _selected_armature(context, prefer_target=True):
    props = context.scene.vid_an
    obj = props.target_object if prefer_target else props.source_object
    if obj is None:
        obj = props.source_object or props.target_object
    if obj is None or obj.type != 'ARMATURE':
        raise RuntimeError("No armature selected (Source or Target).")
    return obj


def _activate(obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_bvh(context, path, prefer_target=True):
    obj = _selected_armature(context, prefer_target)
    _activate(obj)
    bpy.ops.export_anim.bvh(filepath=path)
    return path


def export_fbx(context, path, prefer_target=True):
    obj = _selected_armature(context, prefer_target)
    _activate(obj)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        bake_anim=True,
    )
    return path


def export_glb(context, path, prefer_target=True):
    obj = _selected_armature(context, prefer_target)
    _activate(obj)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True)
    return path


def session_to_legacy_json(session_path, out_json=None):
    """Convert a core PoseSequence (JSONL/NPZ/legacy JSON) to the add-on's
    legacy ``{source, fps, frames:[{t, landmarks, world}]}`` JSON so the existing
    ``mocap.build_source_from_json`` rig builder can consume it.

    Returns the path to the produced JSON.
    """
    if core is None:
        raise RuntimeError(
            "core package not found; cannot import external mocap sessions."
        )
    seq = core.pose_schema.load_sequence(session_path)
    if len(seq) == 0:
        raise RuntimeError("Session contains no frames: {}".format(session_path))

    if out_json is None:
        out_json = os.path.join(tempfile.gettempdir(),
                                os.path.splitext(os.path.basename(session_path))[0] + "_legacy.json")

    frames = []
    for fr in seq.frames:
        world = []
        landmarks = []
        if fr.landmarks_3d is not None:
            world = [{"x": float(a[0]), "y": float(a[1]), "z": float(a[2])}
                     for a in fr.landmarks_3d]
        if fr.landmarks_2d is not None:
            for i, a in enumerate(fr.landmarks_2d):
                v = 1.0
                if fr.confidence is not None and i < len(fr.confidence):
                    v = float(fr.confidence[i])
                landmarks.append({"x": float(a[0]), "y": float(a[1]), "z": 0.0, "v": v})
        frames.append({"t": float(fr.timestamp), "landmarks": landmarks, "world": world})

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"source": seq.source, "fps": seq.fps, "frames": frames}, fh)
    return out_json


def export_arrays_to_core(context, solved_or_none, out_path, fmt):
    """Export NPZ/JSONL/CSV from the built source rig via the core solver.

    This is a standalone-friendly path; it re-solves from the stored pose JSON
    when the core package is available.
    """
    if core is None:
        raise RuntimeError("core package not found; cannot export raw arrays.")
    props = context.scene.vid_an
    pose_json = bpy.path.abspath(props.pose_json) if props.pose_json else ""
    if not pose_json or not os.path.exists(pose_json):
        raise RuntimeError("No pose JSON available to export raw arrays.")
    seq = core.pose_schema.load_sequence(pose_json)
    solved = core.motion_solver.solve_sequence(seq)
    fmt = fmt.lower()
    if fmt == "npz":
        core.exporter.export_npz(out_path, solved)
    elif fmt in ("jsonl", "json"):
        core.exporter.export_jsonl(out_path, solved)
    elif fmt == "csv":
        core.exporter.write_csv(out_path, solved)
    elif fmt == "bvh":
        core.exporter.bvh_from_solved(solved, out_path)
    else:
        raise RuntimeError("Unsupported array format: " + fmt)
    return out_path
