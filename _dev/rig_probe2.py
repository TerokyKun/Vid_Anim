import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
rig = bpy.data.objects.get("rig")
want = ["root","torso","hips","chest","spine_fk","spine_fk.001","spine_fk.002","spine_fk.003",
        "neck","head","shoulder.L","upper_arm_fk.L","forearm_fk.L","hand_fk.L",
        "thigh_fk.L","shin_fk.L","foot_fk.L","toe_fk.L","upper_arm_parent.L","thigh_parent.L"]
for n in want:
    b = rig.data.bones.get(n)
    if not b:
        print("MISS", n); continue
    print("CTRL", n, "parent=", b.parent.name if b.parent else None,
          "len=%.3f"%b.length, "use_connect=", b.use_connect)
