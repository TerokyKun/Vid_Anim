import os, sys, math
import bpy
from mathutils import Vector
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
act = rig.animation_data.action
print("ACTION", act.name)
print("HAS slots", hasattr(act,"slots"), "layers", hasattr(act,"layers"))
try:
    print("slots", [s.identifier for s in act.slots])
except Exception as e:
    print("sloterr", e)
for f in range(1,21):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    q = rig.pose.bones["upper_arm_fk.L"].rotation_quaternion
    if f in (1,5,10,15,20):
        print("  f%02d fkq= %.3f %.3f %.3f %.3f"%(f,q.w,q.x,q.y,q.z))
def wdir_eval(obj, name):
    dg = bpy.context.evaluated_depsgraph_get()
    eobj = obj.evaluated_get(dg)
    return (eobj.matrix_world @ eobj.pose.bones[name].matrix).col[1].xyz.normalized()
for f in (1,5,10,15,20):
    bpy.context.scene.frame_set(f)
    d=wdir_eval(rig,"DEF-upper_arm.L")
    print("  DEF f%02d = %.2f %.2f %.2f"%(f,d.x,d.y,d.z))
