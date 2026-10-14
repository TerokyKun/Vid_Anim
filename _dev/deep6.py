import bpy, math
from mathutils import Quaternion
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
pb = rig.pose.bones["upper_arm_fk.L"]
pb.rotation_mode='QUATERNION'
for f,deg in [(1,0),(10,60),(20,0)]:
    bpy.context.scene.frame_set(f)
    pb.rotation_quaternion = Quaternion((1,0,0), math.radians(deg))
    pb.keyframe_insert("rotation_quaternion", frame=f)
for f in (1,5,10,20):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    dg=bpy.context.evaluated_depsgraph_get(); eo=rig.evaluated_get(dg)
    d=(eo.matrix_world @ eo.pose.bones["DEF-upper_arm.L"].matrix).col[1].xyz.normalized()
    q=rig.pose.bones["upper_arm_fk.L"].rotation_quaternion
    print("f%02d fkq=(%.2f,%.2f,%.2f,%.2f) DEFdir=(%.2f,%.2f,%.2f)"%(f,q.w,q.x,q.y,q.z,d.x,d.y,d.z))
