import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget_test_lib as lib

lib.wipe()
rig = lib.gen_rigify()

for name in ["upper_arm_fk.L", "forearm_fk.L", "thigh_fk.L", "spine_fk",
             "DEF-upper_arm.L", "DEF-forearm.L", "DEF-thigh.L", "DEF-spine",
             "hips", "torso", "ORB-upper_arm.L", "ORG-upper_arm.L"]:
    if name in rig.data.bones:
        b = rig.data.bones[name]
        d = (rig.matrix_world @ b.matrix_local).col[1].xyz.normalized()
        print("REST %-18s dir=(%6.3f,%6.3f,%6.3f)" % (name, d.x, d.y, d.z))
print("PROBE_DONE")
