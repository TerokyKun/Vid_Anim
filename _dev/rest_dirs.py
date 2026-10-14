import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
for n in ["upper_arm_fk.L","forearm_fk.L","thigh_fk.L","shin_fk.L","foot_fk.L","spine_fk","hips","head","neck"]:
    b = rig.data.bones[n]
    v = (rig.matrix_world @ b.matrix_local).col[1].xyz.normalized()
    print("RD", n, "%.3f %.3f %.3f" % (v.x,v.y,v.z), "len=%.3f"%b.length)
