import bpy, math
from mathutils import Quaternion
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
pb = rig.pose.bones["upper_arm_fk.L"]; pb.rotation_mode='QUATERNION'
for f,deg in [(1,0),(10,60),(20,0)]:
    bpy.context.scene.frame_set(f)
    pb.rotation_quaternion = Quaternion((1,0,0), math.radians(deg))
    pb.keyframe_insert("rotation_quaternion", frame=f)
for ikfk in (0.0, 1.0):
    rig.pose.bones["upper_arm_parent.L"]["IK_FK"] = ikfk
    bpy.context.scene.frame_set(1); bpy.context.view_layer.update()
    bpy.context.scene.frame_set(10); bpy.context.view_layer.update()
    dg=bpy.context.evaluated_depsgraph_get(); eo=rig.evaluated_get(dg)
    d=(eo.matrix_world @ eo.pose.bones["DEF-upper_arm.L"].matrix).col[1].xyz.normalized()
    infl=eo.pose.bones["ORG-upper_arm.L"].constraints["Copy Transforms.001"].influence
    print("IK_FK=%.1f  DEFdir=(%.2f,%.2f,%.2f) ikInfluence=%.2f"%(ikfk,d.x,d.y,d.z,infl))
