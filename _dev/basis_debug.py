import os
import sys
import math

import bpy
from mathutils import Matrix, Vector

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

tgt_bones = {t: rig.data.bones[t] for (_, t, _) in pairs}
pairs_sorted = sorted(pairs, key=lambda p: rt._bone_depth(tgt_bones[p[1]]))

target_name = "thigh_fk.L"
T = rig.pose.bones[target_name]
tb = tgt_bones[target_name]
parent = tb.parent
print("TARGET", target_name, "PARENT", parent.name if parent else None,
      "connected", tb.use_connect, "parent_in_pairs", any(p[1] == (parent.name if parent else "") for p in pairs))

frame = 10
bpy.context.scene.frame_set(frame)
bpy.context.view_layer.update()

# find source bone for target_name
sname = [s for (s, t, sem) in pairs if t == target_name][0]
print("SRC", sname)

s_bone = src.pose.bones[sname]
s_pose_world = src.matrix_world @ s_bone.matrix
tgt_rest_world = rt._world_rest(rig, tb)
desired_rot = (Matrix.Rotation(0, 4, 'Z').to_3x3() @ s_pose_world.to_3x3()).to_4x4()
desired_rot.translation = tgt_rest_world.translation
desired_arm = rig.matrix_world.inverted() @ desired_rot

# ---- method A: original pb.matrix ----
T.matrix = desired_arm
bpy.context.view_layer.update()
qA = T.rotation_quaternion.copy()
lA = T.location.copy()

# reset
bpy.context.scene.frame_set(frame)
bpy.context.view_layer.update()
for pb in rig.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
bpy.context.view_layer.update()

# rebuild parent desired pose the analytic way (hips root etc.)
# We'll just replicate the analytic basis for this bone using the CURRENT (rest) parent pose
pp = rig.pose.bones[parent.name].matrix if parent else None
if parent is not None:
    base = pp @ parent.matrix_local.inverted() @ tb.matrix_local
    basis = base.inverted() @ desired_arm
else:
    basis = tb.matrix_local.inverted() @ desired_arm
qB = basis.to_quaternion()
lB = basis.to_translation()

print("qA", tuple(round(x, 4) for x in qA))
print("qB", tuple(round(x, 4) for x in qB))
print("angle A vs B", math.degrees(qA.rotation_difference(qB).angle))
print("lA", tuple(round(x, 4) for x in lA))
print("lB", tuple(round(x, 4) for x in lB))
print("DEBUG_DONE")
