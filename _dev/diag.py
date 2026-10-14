import os
import sys
import math

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender_addon"))
sys.path.insert(0, HERE)

import retarget_test_lib as lib

lib.wipe()
src = lib.make_source()
lib.animate_source(src)
rig = lib.gen_rigify()
from vid_an.bone_maps import auto_pairs
from vid_an import retarget as rt

pairs = auto_pairs(src, rig)
rt.retarget(bpy.context, src, rig, pairs, 1, 20)


def wdir(obj, name):
    m = obj.matrix_world @ obj.pose.bones[name].matrix
    return m.col[1].xyz.normalized()


def ang(a, b):
    return math.degrees(a.angle(b))


cands = ["upper_arm_fk.L", "ORG-upper_arm.L", "DEF-upper_arm.L", "DEF-upper_arm.L.001"]
print("SRC action:", src.animation_data.action if src.animation_data else None)
for f in (1, 5, 10, 15, 20):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    q = src.pose.bones["upper_arm.L"].rotation_quaternion
    print("SRCQ f%d = %.3f %.3f %.3f %.3f" % (f, q.w, q.x, q.y, q.z))
    s = wdir(src, "upper_arm.L")
    print("FRAME", f, "src_dir=%.2f,%.2f,%.2f" % (s.x, s.y, s.z))
    for c in cands:
        if c in rig.pose.bones:
            d = wdir(rig, c)
            print("   %-22s dir_err=%.2f" % (c, ang(s, d)))
