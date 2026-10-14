import bpy, math
from mathutils import Quaternion
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
# remove all drivers, then force ik influence 0
for d in list(rig.animation_data.drivers):
    if 'ORG-' in d.data_path and 'Copy Transforms.001' in d.data_path:
        rig.animation_data.drivers.remove(d)
for n in ("ORG-upper_arm.L","ORG-upper_arm.R","ORG-thigh.L","ORG-thigh.R"):
    for c in rig.pose.bones[n].constraints:
        if c.type=='COPY_TRANSFORMS':
            c.influence = 1.0 if '.001' not in c.name else 0.0
pb = rig.pose.bones["upper_arm_fk.L"]; pb.rotation_mode='QUATERNION'
for f,deg in [(1,0),(10,60),(20,0)]:
    bpy.context.scene.frame_set(f)
    pb.rotation_quaternion = Quaternion((1,0,0), math.radians(deg))
    pb.keyframe_insert("rotation_quaternion", frame=f)
for f in (1,10,20):
    bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
    dg=bpy.context.evaluated_depsgraph_get(); eo=rig.evaluated_get(dg)
    for n in ["upper_arm_fk.L","ORG-upper_arm.L","DEF-upper_arm.L"]:
        d=(eo.matrix_world @ eo.pose.bones[n].matrix).col[1].xyz.normalized()
        print("f%02d %-16s (%.2f,%.2f,%.2f)"%(f,n,d.x,d.y,d.z))
