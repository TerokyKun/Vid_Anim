import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
for d in rig.animation_data.drivers:
    if 'ORG-upper_arm.L' in d.data_path and 'Copy Transforms.001' in d.data_path:
        print("DRV expr=", repr(d.driver.expression), "type=", d.driver.type, "mute=", d.mute)
prop = rig.pose.bones["upper_arm_parent.L"]
print("IK_FK default=", prop["IK_FK"])
def infl():
    return rig.pose.bones["ORG-upper_arm.L"].constraints["Copy Transforms.001"].influence
print("infl before update=", infl())
bpy.context.view_layer.update()
print("infl after update=", infl())
prop["IK_FK"] = 1
bpy.context.view_layer.update()
print("infl with IK_FK=1 =", infl())
prop["IK_FK"] = 0
bpy.context.view_layer.update()
print("infl with IK_FK=0 =", infl())
