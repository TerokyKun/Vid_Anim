import os, sys, math
import bpy
from mathutils import Vector, Quaternion
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE),"blender_addon"))
sys.path.insert(0, HERE)
import retarget_test_lib as lib
from vid_an.bone_maps import auto_pairs
from vid_an import retarget as rt
lib.wipe()
rig = lib.gen_rigify()
src = lib.make_source_matching(rig)
lib.animate_swing(src)
pairs = auto_pairs(src, rig)
rt.retarget(bpy.context, src, rig, pairs, 1, 20)
def wdir(obj, name):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).col[1].xyz.normalized()
for mode in ("single","double","depsgraph"):
    print("=== MODE", mode)
    for f in range(1,21):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        if mode=="double":
            bpy.context.view_layer.update()
        if mode=="depsgraph":
            dg=bpy.context.evaluated_depsgraph_get(); dg.update()
        if f==10:
            d=wdir(rig,"DEF-upper_arm.L")
            print("  f10 DEF dir= %.2f %.2f %.2f"%(d.x,d.y,d.z))
