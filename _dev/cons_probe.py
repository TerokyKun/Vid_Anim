import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
names = ["upper_arm_fk.L","ORG-upper_arm.L","DEF-upper_arm.L","MCH-upper_arm_parent.L",
         "upper_arm_parent.L","thigh_fk.L","ORG-thigh.L","DEF-thigh.L"]
for n in names:
    pb = rig.pose.bones.get(n)
    if not pb: 
        print("MISS", n); continue
    print("BONE", n, "parent=", pb.parent.name if pb.parent else None)
    for c in pb.constraints:
        print("   CON", c.type, "sub=", getattr(c,'subtarget',''), "inf=%.2f"%c.influence,
              "owner=", getattr(c,'owner_space',''), "target=", getattr(c,'target_space',''))
