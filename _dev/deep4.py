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
print("ACTION", act.name, "fcurves", len(act.fcurves))
for fc in act.fcurves:
    if "upper_arm_fk.L" in fc.data_path:
        print("FC", fc.data_path, "idx", fc.array_index, "kps", len(fc.keyframe_points), "val@10=%.3f"%fc.evaluate(10))
# also check hips/root translation target
for fc in act.fcurves:
    if fc.data_path.endswith('["location"]') or "location" in fc.data_path:
        pass
print("root fcurves:")
for fc in act.fcurves:
    if '"root"' in fc.data_path or '"hips"' in fc.data_path:
        print("  ", fc.data_path, fc.array_index, "val@10=%.3f"%fc.evaluate(10))
