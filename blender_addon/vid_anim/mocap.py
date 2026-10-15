import json
from collections import namedtuple

import bpy
from mathutils import Matrix, Vector

from ._core_bridge import core

MP = {
    "nose": 0,
    "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_foot": 31, "r_foot": 32,
}

REF_FWD = Vector((0.0, -1.0, 0.0))
REF_UP = Vector((0.0, 0.0, 1.0))

Spec = namedtuple("Spec", "name parent head tail secondary ref min_bend max_bend")

_LOCAL_SPECS = [
    ("Hips", None, "mid_hip", "mid_shoulder", ("l_hip", "r_hip"), REF_FWD, 0.0, 180.0),
    ("Spine", "Hips", "mid_hip", "mid_shoulder", ("l_shoulder", "r_shoulder"), REF_FWD, 0.0, 45.0),
    ("Neck", "Spine", "mid_shoulder", "nose", ("l_shoulder", "r_shoulder"), REF_FWD, 0.0, 55.0),
    ("Head", "Neck", "nose", "head_top", ("l_ear", "r_ear"), REF_FWD, 0.0, 70.0),
    ("Shoulder.L", "Spine", "mid_shoulder", "l_shoulder", None, REF_FWD, 0.0, 60.0),
    ("UpperArm.L", "Shoulder.L", "l_shoulder", "l_elbow", None, REF_FWD, 0.0, 180.0),
    ("ForeArm.L", "UpperArm.L", "l_elbow", "l_wrist", None, REF_FWD, 0.0, 155.0),
    ("Hand.L", "ForeArm.L", "l_wrist", "l_index", None, REF_FWD, 0.0, 90.0),
    ("Shoulder.R", "Spine", "mid_shoulder", "r_shoulder", None, REF_FWD, 0.0, 60.0),
    ("UpperArm.R", "Shoulder.R", "r_shoulder", "r_elbow", None, REF_FWD, 0.0, 180.0),
    ("ForeArm.R", "UpperArm.R", "r_elbow", "r_wrist", None, REF_FWD, 0.0, 155.0),
    ("Hand.R", "ForeArm.R", "r_wrist", "r_index", None, REF_FWD, 0.0, 90.0),
    ("Thigh.L", "Hips", "l_hip", "l_knee", None, REF_FWD, 0.0, 130.0),
    ("Shin.L", "Thigh.L", "l_knee", "l_ankle", None, REF_FWD, 0.0, 155.0),
    ("Foot.L", "Shin.L", "l_ankle", "l_foot", None, REF_FWD, 0.0, 80.0),
    ("Thigh.R", "Hips", "r_hip", "r_knee", None, REF_FWD, 0.0, 130.0),
    ("Shin.R", "Thigh.R", "r_knee", "r_ankle", None, REF_FWD, 0.0, 155.0),
    ("Foot.R", "Shin.R", "r_ankle", "r_foot", None, REF_FWD, 0.0, 80.0),
]


def _load_specs():
    """Prefer the shared core anatomy table; fall back to a local copy."""
    anat = getattr(core, "anatomy", None) if core is not None else None
    if anat is not None:
        return [Spec(s.name, s.parent, s.head, s.tail, s.secondary,
                     Vector(s.ref), s.min_bend, s.max_bend) for s in anat.BONES]
    return [Spec(n, p, h, t, sec, r, mn, mx) for (n, p, h, t, sec, r, mn, mx) in _LOCAL_SPECS]


BONES = _load_specs()


def _anatomy():
    return getattr(core, "anatomy", None) if core is not None else None


def _to_blender(p):
    # MediaPipe world landmarks are right-handed: +x = the person's LEFT
    # (image right when facing the camera), +y = DOWN, +z = depth toward the
    # camera.  Blender is right-handed with +Y forward and +Z up, and the
    # target rig faces -Y.  A coordinate-system change between two
    # right-handed frames MUST be a proper rotation (determinant +1):
    #     X = x        (left stays +X)
    #     Y = +z       (toward-camera => -Y = character front)
    #     Z = -y       (up)
    # The previous (x, -z, -y) negated two axes, i.e. it was a REFLECTION
    # (determinant -1), which mirrored the body front-to-back.
    return Vector((p["x"], p["z"], -p["y"]))


def _vec3(v):
    return [float(v.x), float(v.y), float(v.z)]


def _derive_head(pts):
    """Synthetic ``head_top`` so the head bone can nod/turn/tilt from the face."""
    nose = pts["nose"]
    up = None
    if "l_ear" in pts and "r_ear" in pts:
        left = pts["l_ear"] - pts["r_ear"]
        forward = nose - (pts["l_ear"] + pts["r_ear"]) * 0.5
        c = forward.cross(left)
        if c.length > 1e-6:
            up = c.normalized()
    if up is None:
        up = (pts["mid_shoulder"] - pts["mid_hip"])
        up = up.normalized() if up.length > 1e-6 else REF_UP
    pts["head_top"] = nose + up * 0.15


def _world_joints(frame, fill_partial=True):
    world = frame.get("world") or []
    if not world:
        return None
    landmarks = frame.get("landmarks") or []
    pts = {}
    valid = {}
    for name, idx in MP.items():
        pts[name] = _to_blender(world[idx])
        v = 1.0
        if idx < len(landmarks):
            try:
                v = float(landmarks[idx].get("v", 1.0))
            except Exception:
                v = 1.0
        valid[name] = v >= 0.3

    # Rebuild hidden parts as a balanced standing pose (centre of gravity under
    # the visible silhouette) using the shared core package when available.
    if fill_partial and core is not None and any(not v for v in valid.values()):
        try:
            import numpy as _np

            pts_np = {k: _np.array([p.x, p.y, p.z], dtype=float) for k, p in pts.items()}
            valid_np = dict(valid)
            pts_np, _synth = core.partial_body.complete(pts_np, valid_np)
            pts = {k: Vector((float(a[0]), float(a[1]), float(a[2])))
                   for k, a in pts_np.items()}
        except Exception:
            pass

    pts["mid_hip"] = (pts["l_hip"] + pts["r_hip"]) * 0.5
    pts["mid_shoulder"] = (pts["l_shoulder"] + pts["r_shoulder"]) * 0.5
    up = (pts["mid_shoulder"] - pts["mid_hip"])
    if up.length < 1e-6:
        up = REF_UP
    pts["nose_up"] = pts["nose"] + up.normalized() * 0.1
    _derive_head(pts)
    return pts


def _secondary_vec(spec, pts):
    if not spec.secondary:
        return None
    a = pts.get(spec.secondary[0])
    b = pts.get(spec.secondary[1])
    if a is None or b is None:
        return None
    v = a - b
    return v if v.length > 1e-4 else None


def _bone_frame(direction, secondary, ref, prev_x):
    """Right-handed bone frame (X=left, Y=direction, Z=XxY); shared with core."""
    anat = _anatomy()
    if anat is not None:
        m = anat.bone_frame(_vec3(direction),
                            _vec3(secondary) if secondary is not None else None,
                            _vec3(ref),
                            _vec3(prev_x) if prev_x is not None else None)
        x = Vector((m[0][0], m[1][0], m[2][0]))
        y = Vector((m[0][1], m[1][1], m[2][1]))
        z = Vector((m[0][2], m[1][2], m[2][2]))
        return Matrix((x, y, z)).transposed()

    y = direction.normalized()
    if secondary is not None:
        s = secondary - y * secondary.dot(y)
        x = s if s.length > 1e-6 else y.cross(ref).normalized()
    else:
        r = ref.normalized()
        if abs(y.dot(r)) > 0.999:
            r = Vector((1.0, 0.0, 0.0)) if abs(y.x) < 0.9 else Vector((0.0, 0.0, 1.0))
        x = y.cross(r)
        x = x.normalized() if x.length > 1e-9 else Vector((1.0, 0.0, 0.0))
    if prev_x is not None and x.dot(prev_x) < 0.0:
        x = -x
    z = x.cross(y)
    return Matrix((x, y, z)).transposed()


def _clamp_dir(direction, parent_dir, min_bend, max_bend):
    anat = _anatomy()
    if anat is not None and parent_dir is not None:
        import numpy as _np

        out = anat.clamp_direction(_vec3(direction), _vec3(parent_dir), min_bend, max_bend)
        return Vector((float(out[0]), float(out[1]), float(out[2])))
    return direction


def _rotation_from_dir(direction, ref):
    """Public rest-convention helper (kept for compatibility)."""
    return _bone_frame(direction, None, ref, None)


REST = {
    "Hips": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.1)),
    "Spine": ((0.0, 0.0, 0.1), (0.0, 0.0, 0.35)),
    "Neck": ((0.0, 0.0, 0.35), (0.0, 0.0, 0.5)),
    "Head": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.65)),
    "Shoulder.L": ((0.0, 0.0, 0.4), (0.1, 0.0, 0.45)),
    "UpperArm.L": ((0.1, 0.0, 0.45), (0.34, 0.0, 0.45)),
    "ForeArm.L": ((0.34, 0.0, 0.45), (0.58, 0.0, 0.45)),
    "Hand.L": ((0.58, 0.0, 0.45), (0.73, 0.0, 0.45)),
    "Shoulder.R": ((0.0, 0.0, 0.4), (-0.1, 0.0, 0.45)),
    "UpperArm.R": ((-0.1, 0.0, 0.45), (-0.34, 0.0, 0.45)),
    "ForeArm.R": ((-0.34, 0.0, 0.45), (-0.58, 0.0, 0.45)),
    "Hand.R": ((-0.58, 0.0, 0.45), (-0.73, 0.0, 0.45)),
    "Thigh.L": ((0.08, 0.0, 0.0), (0.08, 0.0, -0.4)),
    "Shin.L": ((0.08, 0.0, -0.4), (0.08, 0.0, -0.8)),
    "Foot.L": ((0.08, 0.0, -0.8), (0.08, 0.15, -0.8)),
    "Thigh.R": ((-0.08, 0.0, 0.0), (-0.08, 0.0, -0.4)),
    "Shin.R": ((-0.08, 0.0, -0.4), (-0.08, 0.0, -0.8)),
    "Foot.R": ((-0.08, 0.0, -0.8), (-0.08, 0.15, -0.8)),
}


def _ensure_armature(name="vidan_src"):
    arm_data = bpy.data.armatures.new(name + "_data")
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm)
    return arm


def iter_build_source_from_json(context, json_path, bone_prefix="", fill_partial=True):
    """Build the source rig from pose JSON, yielding (done, total) per frame."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    frames = data.get("frames", [])
    if not frames:
        raise RuntimeError("No frames in JSON")

    arm = _ensure_armature()
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    for spec in BONES:
        b = eb.new(bone_prefix + spec.name)
        head, tail = REST[spec.name]
        b.head = Vector(head)
        b.tail = Vector(tail)
        if spec.parent:
            b.parent = eb[bone_prefix + spec.parent]
    bpy.ops.object.mode_set(mode='OBJECT')

    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    scene = context.scene
    src_fps = data.get("fps") or scene.render.fps
    try:
        scene.render.fps = max(1, min(240, int(round(src_fps))))
    except Exception:
        pass

    last_frame = 1
    for frame in frames:
        if frame.get("t") is not None:
            last_frame = max(last_frame, int(round(frame["t"] * src_fps)) + 1)
    scene.frame_start = 1
    scene.frame_end = last_frame

    data_bones = {bone_prefix + s.name: arm.data.bones[bone_prefix + s.name] for s in BONES}
    pose_bones = {bone_prefix + s.name: arm.pose.bones[bone_prefix + s.name] for s in BONES}

    prev_joints = None
    prev_x = {}
    total = len(frames)
    for fi, frame in enumerate(frames):
        f = int(round((frame.get("t") or 0.0) * src_fps)) + 1
        pts = _world_joints(frame, fill_partial=fill_partial)
        if pts is None:
            pts = prev_joints
        if pts is None:
            yield fi + 1, total
            continue
        prev_joints = pts

        scene.frame_set(f)
        for pb in arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)

        pose_arm = {}
        for spec in BONES:
            head = pts[spec.head]
            direction = pts[spec.tail] - head
            stored = prev_x.get(spec.name)
            if direction.length < 1e-8:
                if stored is None:
                    continue
                frame_m = Matrix((stored[0], stored[1], stored[2])).transposed()
                M = Matrix.Translation(head) @ frame_m.to_4x4()
            else:
                parent_dir = None
                if spec.parent:
                    pname = bone_prefix + spec.parent
                    pm = pose_arm.get(pname)
                    if pm is not None:
                        parent_dir = pm.to_3x3() @ Vector((0.0, 1.0, 0.0))
                direction = _clamp_dir(direction, parent_dir, spec.min_bend, spec.max_bend)
                frame_m = _bone_frame(direction, _secondary_vec(spec, pts),
                                      spec.ref, stored[0] if stored else None)
                M = Matrix.Translation(head) @ frame_m.to_4x4()

            name = bone_prefix + spec.name
            tbone = data_bones[name]
            if spec.parent:
                pname = bone_prefix + spec.parent
                base = pose_arm[pname] @ data_bones[pname].matrix_local.inverted() @ tbone.matrix_local
                basis = base.inverted() @ M
            else:
                basis = tbone.matrix_local.inverted() @ M
            loc, q, _s = basis.decompose()
            pb = pose_bones[name]
            pb.rotation_quaternion = q
            pb.location = loc
            pose_arm[name] = M
            prev_x[spec.name] = (Vector(frame_m.col[0]), Vector(frame_m.col[1]),
                                 Vector(frame_m.col[2]))

        for spec in BONES:
            pose_bones[bone_prefix + spec.name].keyframe_insert("rotation_quaternion", frame=f)
        root = pose_bones[bone_prefix + "Hips"]
        root.keyframe_insert("location", frame=f)

        yield fi + 1, total

    scene.frame_set(1)
    return arm


def build_source_from_json(context, json_path, bone_prefix="", fill_partial=True):
    gen = iter_build_source_from_json(context, json_path, bone_prefix=bone_prefix,
                                      fill_partial=fill_partial)
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break
    return result
