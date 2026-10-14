import bpy
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
bpy.ops.object.mode_set(mode='OBJECT')
rig = bpy.data.objects["rig"]
ad = rig.animation_data
print("AD", ad)
print("ad attrs:", [a for a in dir(ad) if not a.startswith("__")])
act = ad.action
print("ACT", act, type(act))
print("act attrs:", [a for a in dir(act) if not a.startswith("__")])
# drivers count via new API
try:
    print("slots", [(s.identifier, s.name_display) for s in act.slots])
except Exception as e: print("e1",e)
try:
    for layer in act.layers:
        for strip in layer.strips:
            print("STRIP", strip.type)
            for slot in act.slots:
                cb = strip.channelbag(slot)
                if cb:
                    print("  channelbag", slot.identifier, "fcurves", len(cb.fcurves), "drivers", len(cb.drivers))
except Exception as e: print("e2",e)
