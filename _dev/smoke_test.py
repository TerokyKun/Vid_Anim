import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender_addon"))

import vid_an
from vid_an import operators, panel, properties, mocap, retarget, bone_maps

vid_an.register()
print("REGISTER_OK classes", len(operators.CLASSES))
print("PROPS", [p for p in dir(bpy.types.Scene.vid_an) if not p.startswith("_")][:5])
print("SCENE_OK", hasattr(bpy.types.Scene, "vid_an"))
vid_an.unregister()
print("UNREGISTER_OK")
print("SMOKE_DONE")
