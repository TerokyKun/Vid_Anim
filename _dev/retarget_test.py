import os
import sys
import math

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender_addon"))
sys.path.insert(0, HERE)

import retarget_test_lib as lib
from vid_an.bone_maps import auto_pairs
from vid_an import retarget as rt

lib.wipe()
rig = lib.gen_rigify()
src = lib.make_source_matching(rig)
lib.animate_swing(src)

pairs = auto_pairs(src, rig)
print("PAIRS", len(pairs))
for s, t, sem in pairs:
    print("  MAP %-14s -> %-20s [%s]" % (s, t, sem))

rt.retarget(bpy.context, src, rig, pairs, 1, 20)


def wdir(obj, name):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).col[1].xyz.normalized()


def delta(obj, name):
    b = obj.data.bones[name]
    rest = obj.matrix_world @ b.matrix_local
    cur = obj.matrix_world @ obj.pose.bones[name].matrix
    return (cur @ rest.inverted()).to_quaternion()


checks = [
    ("upper_arm.L", "upper_arm_fk.L", "DEF-upper_arm.L"),
    ("forearm.L", "forearm_fk.L", "DEF-forearm.L"),
    ("thigh.L", "thigh_fk.L", "DEF-thigh.L"),
    ("shin.L", "shin_fk.L", "DEF-shin.L"),
    ("spine", "spine_fk", "DEF-spine"),
]
worst_dir = 0.0
worst_delta = 0.0
for f in range(1, 21):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    for sname, cname, dname in checks:
        sd = wdir(src, sname)
        cd = wdir(rig, cname)
        dd = wdir(rig, dname)
        de = math.degrees(sd.angle(dd))
        dde = math.degrees(delta(src, sname).rotation_difference(delta(rig, dname)).angle)
        worst_dir = max(worst_dir, de)
        worst_delta = max(worst_delta, dde)
        if f in (1, 10, 20):
            print("  f%02d %-12s dir_err=%.2f  delta_err=%.2f  src=(%.2f,%.2f,%.2f) def=(%.2f,%.2f,%.2f)"
                  % (f, sname, de, dde, sd.x, sd.y, sd.z, dd.x, dd.y, dd.z))
print("WORST_DIR_ERR %.3f  WORST_DELTA_ERR %.3f" % (worst_dir, worst_delta))
print("TEST_DONE")
