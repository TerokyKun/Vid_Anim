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
for f in (1,10):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    q = rig.pose.bones["upper_arm_fk.L"].rotation_quaternion
    print("F",f,"fkq= %.3f %.3f %.3f %.3f"%(q.w,q.x,q.y,q.z))
    for n in ["upper_arm_fk.L","ORG-upper_arm.L","upper_arm_ik.L","DEF-upper_arm.L"]:
        d=(rig.matrix_world @ rig.pose.bones[n].matrix).col[1].xyz.normalized()
        print("   %-16s dir= %.2f %.2f %.2f"%(n,d.x,d.y,d.z))
    # delta comparison source vs fk control
    sb=src.data.bones["upper_arm.L"]; cb=rig.data.bones["upper_arm_fk.L"]
    sd=(src.matrix_world@src.pose.bones["upper_arm.L"].matrix)@(src.matrix_world@sb.matrix_local).inverted()
    cd=(rig.matrix_world@rig.pose.bones["upper_arm_fk.L"].matrix)@(rig.matrix_world@cb.matrix_local).inverted()
    sdir=(src.matrix_world@src.pose.bones["upper_arm.L"].matrix).col[1].xyz.normalized()
    print("   SRCDIR %.2f %.2f %.2f"%(sdir.x,sdir.y,sdir.z))
    print("   DELTA_ANGLE %.2f"%math.degrees(sd.to_quaternion().rotation_difference(cd.to_quaternion()).angle))
    for c in rig.pose.bones["ORG-upper_arm.L"].constraints:
        print("   CON", c.subtarget, "inf=%.2f"%c.influence, "drv=", end=" ")
        try:
            print(c.driver_add("influence"))
        except Exception as e:
            print("none")
    # IK_FK prop
    for pk in rig.pose.bones["upper_arm_parent.L"].keys():
        print("   PROP", pk, rig.pose.bones["upper_arm_parent.L"][pk])
