import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
ad = rig.animation_data
print("ndrivers", len(ad.drivers))
for d in ad.drivers:
    if "upper_arm" in d.data_path and "constraints" in d.data_path:
        print("DRV", d.data_path, d.array_index)
        for v in d.driver.variables:
            print("   VAR", v.name, v.type)
            for t in v.targets:
                print("      TGT id=", t.id, "bone=", t.bone_target, "data_path=", t.data_path)
