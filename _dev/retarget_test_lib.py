import math

import bpy
from mathutils import Vector, Quaternion

GENERIC_TO_CTRL = {
    "hips": "hips",
    "spine": "spine_fk",
    "neck": "neck",
    "head": "head",
    "upper_arm.L": "upper_arm_fk.L",
    "forearm.L": "forearm_fk.L",
    "hand.L": "hand_fk.L",
    "upper_arm.R": "upper_arm_fk.R",
    "forearm.R": "forearm_fk.R",
    "hand.R": "hand_fk.R",
    "thigh.L": "thigh_fk.L",
    "shin.L": "shin_fk.L",
    "foot.L": "foot_fk.L",
    "thigh.R": "thigh_fk.R",
    "shin.R": "shin_fk.R",
    "foot.R": "foot_fk.R",
}

PARENTS = {
    "spine": "hips", "neck": "spine", "head": "neck",
    "upper_arm.L": "spine", "forearm.L": "upper_arm.L", "hand.L": "forearm.L",
    "upper_arm.R": "spine", "forearm.R": "upper_arm.R", "hand.R": "forearm.R",
    "thigh.L": "hips", "shin.L": "thigh.L", "foot.L": "shin.L",
    "thigh.R": "hips", "shin.R": "thigh.R", "foot.R": "shin.R",
}

CONNECT = {"forearm.L", "hand.L", "forearm.R", "hand.R", "shin.L", "foot.L", "shin.R", "foot.R"}


def wipe():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def gen_rigify():
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.rigify_generate()
    bpy.ops.object.mode_set(mode='OBJECT')
    return bpy.data.objects["rig"]


def make_source_matching(rig):
    arm_data = bpy.data.armatures.new("srcdata")
    src = bpy.data.objects.new("src", arm_data)
    bpy.context.collection.objects.link(src)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    heads, tails = {}, {}
    for gen, ctrl in GENERIC_TO_CTRL.items():
        b = rig.data.bones[ctrl]
        heads[gen] = rig.matrix_world @ b.head_local
        tails[gen] = rig.matrix_world @ b.tail_local

    order = ["hips", "spine", "neck", "head",
             "upper_arm.L", "forearm.L", "hand.L",
             "upper_arm.R", "forearm.R", "hand.R",
             "thigh.L", "shin.L", "foot.L",
             "thigh.R", "shin.R", "foot.R"]
    for gen in order:
        b = eb.new(gen)
        b.head = heads[gen]
        b.tail = tails[gen]
        parent = PARENTS.get(gen)
        if parent:
            b.parent = eb[parent]
            b.use_connect = gen in CONNECT
    bpy.ops.object.mode_set(mode='OBJECT')
    return src


def animate_swing(src):
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode='POSE')
    for pb in src.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    def key(name, axis, angles):
        pb = src.pose.bones[name]
        for f, deg in angles:
            bpy.context.scene.frame_set(f)
            pb.rotation_quaternion = Quaternion(axis, math.radians(deg))
            pb.keyframe_insert("rotation_quaternion", frame=f)

    key("upper_arm.L", Vector((1, 0, 0)), [(1, 0), (10, 60), (20, 0)])
    key("upper_arm.R", Vector((1, 0, 0)), [(1, 0), (10, -45), (20, 0)])
    key("thigh.L", Vector((1, 0, 0)), [(1, 0), (10, 40), (20, 0)])
    key("spine", Vector((1, 0, 0)), [(1, 0), (10, 20), (20, 0)])
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 20
