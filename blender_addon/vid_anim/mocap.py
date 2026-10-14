import json

import bpy
from mathutils import Matrix, Vector

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

BONES = [
    ("Hips", None, ("mid_hip", "mid_shoulder"), REF_FWD),
    ("Spine", "Hips", ("mid_hip", "mid_shoulder"), REF_FWD),
    ("Neck", "Spine", ("mid_shoulder", "nose"), REF_FWD),
    ("Head", "Neck", ("nose", "nose_up"), REF_FWD),
    ("Shoulder.L", "Spine", ("mid_shoulder", "l_shoulder"), REF_UP),
    ("UpperArm.L", "Shoulder.L", ("l_shoulder", "l_elbow"), REF_UP),
    ("ForeArm.L", "UpperArm.L", ("l_elbow", "l_wrist"), REF_UP),
    ("Hand.L", "ForeArm.L", ("l_wrist", "l_index"), REF_UP),
    ("Shoulder.R", "Spine", ("mid_shoulder", "r_shoulder"), REF_UP),
    ("UpperArm.R", "Shoulder.R", ("r_shoulder", "r_elbow"), REF_UP),
    ("ForeArm.R", "UpperArm.R", ("r_elbow", "r_wrist"), REF_UP),
    ("Hand.R", "ForeArm.R", ("r_wrist", "r_index"), REF_UP),
    ("Thigh.L", "Hips", ("l_hip", "l_knee"), REF_FWD),
    ("Shin.L", "Thigh.L", ("l_knee", "l_ankle"), REF_FWD),
    ("Foot.L", "Shin.L", ("l_ankle", "l_foot"), REF_UP),
    ("Thigh.R", "Hips", ("r_hip", "r_knee"), REF_FWD),
    ("Shin.R", "Thigh.R", ("r_knee", "r_ankle"), REF_FWD),
    ("Foot.R", "Shin.R", ("r_ankle", "r_foot"), REF_UP),
]


def _to_blender(p):
    return Vector((p["x"], -p["z"], -p["y"]))


def _world_joints(frame):
    world = frame.get("world") or []
    if not world:
        return None
    pts = {}
    for name, idx in MP.items():
        pts[name] = _to_blender(world[idx])
    pts["mid_hip"] = (pts["l_hip"] + pts["r_hip"]) * 0.5
    pts["mid_shoulder"] = (pts["l_shoulder"] + pts["r_shoulder"]) * 0.5
    up = (pts["mid_shoulder"] - pts["mid_hip"])
    if up.length < 1e-6:
        up = REF_UP
    pts["nose_up"] = pts["nose"] + up.normalized() * 0.1
    return pts


def _rotation_from_dir(direction, ref):
    y = direction.normalized()
    if abs(y.dot(ref.normalized())) > 0.999:
        ref = Vector((1.0, 0.0, 0.0)) if abs(y.x) < 0.9 else Vector((0.0, 0.0, 1.0))
    x = ref.cross(y).normalized()
    z = x.cross(y)
    return Matrix((x, y, z)).transposed()


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


def iter_build_source_from_json(context, json_path, bone_prefix=""):
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
    for bname, parent, _jt, _ref in BONES:
        b = eb.new(bone_prefix + bname)
        head, tail = REST[bname]
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent:
            b.parent = eb[bone_prefix + parent]
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

    data_bones = {bone_prefix + b: arm.data.bones[bone_prefix + b] for b, _p, _j, _r in BONES}
    pose_bones = {bone_prefix + b: arm.pose.bones[bone_prefix + b] for b, _p, _j, _r in BONES}

    prev_joints = None
    total = len(frames)
    for fi, frame in enumerate(frames):
        f = int(round((frame.get("t") or 0.0) * src_fps)) + 1
        pts = _world_joints(frame)
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
        for bname, parent, (h, t), ref in BONES:
            head = pts[h]
            direction = pts[t] - pts[h]
            if direction.length < 1e-8:
                continue
            rot = _rotation_from_dir(direction, ref)
            M = Matrix.Translation(head) @ rot.to_4x4()

            name = bone_prefix + bname
            tbone = data_bones[name]
            if parent:
                pname = bone_prefix + parent
                base = pose_arm[pname] @ data_bones[pname].matrix_local.inverted() @ tbone.matrix_local
                basis = base.inverted() @ M
            else:
                basis = tbone.matrix_local.inverted() @ M
            loc, q, _s = basis.decompose()
            pb = pose_bones[name]
            pb.rotation_quaternion = q
            pb.location = loc
            pose_arm[name] = M

        for bname, _p, _j, _r in BONES:
            pose_bones[bone_prefix + bname].keyframe_insert("rotation_quaternion", frame=f)
        root = pose_bones[bone_prefix + "Hips"]
        root.keyframe_insert("location", frame=f)

        yield fi + 1, total

    scene.frame_set(1)
    return arm


def build_source_from_json(context, json_path, bone_prefix=""):
    gen = iter_build_source_from_json(context, json_path, bone_prefix=bone_prefix)
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break
    return result
