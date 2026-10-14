import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender_addon"))

import vid_an
from vid_an import operators

vid_an.register()
props = bpy.context.scene.vid_an
py, script = operators._resolve_backend(props)
print("ROOTS", operators._candidate_roots())
print("BACKEND_PY", py)
print("SCRIPT", script)
print("PY_EXISTS", os.path.exists(py) if py else False)
print("SCRIPT_EXISTS", os.path.exists(script) if script else False)
print("CHECK_DONE")
