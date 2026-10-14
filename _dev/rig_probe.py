import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
try:
    bpy.ops.pose.rigify_generate()
except Exception as e:
    print("GENERR", e)
arm = [o for o in bpy.data.objects if o.type=='ARMATURE']
for a in arm:
    print("ARMATURE", a.name, len(a.data.bones))
    for b in a.data.bones:
        print("BONE", b.name)
