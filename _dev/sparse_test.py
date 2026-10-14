import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "blender_addon"))

import bpy
from vid_an import mocap

D = mocap._to_blender
world = []
for i in range(33):
    world.append({"x": 0.0, "y": 0.0, "z": 0.0})
world[mocap.MP["l_shoulder"]] = {"x": -0.2, "y": 0.0, "z": 0.0}
world[mocap.MP["r_shoulder"]] = {"x": 0.2, "y": 0.0, "z": 0.0}
world[mocap.MP["l_hip"]] = {"x": -0.1, "y": 0.0, "z": -0.9}
world[mocap.MP["r_hip"]] = {"x": 0.1, "y": 0.0, "z": -0.9}
world[mocap.MP["nose"]] = {"x": 0.0, "y": 0.0, "z": 0.3}

frames = [{"t": 0.0, "landmarks": [], "world": world},
          {"t": 0.5, "landmarks": [], "world": world},
          {"t": 1.0, "landmarks": [], "world": world}]
p = os.path.join(HERE, "_tmp_sparse.json")
json.dump({"source": "x", "fps": 30.0, "frames": frames}, open(p, "w"))

arm = mocap.build_source_from_json(bpy.context, p)
sc = bpy.context.scene
print("FRAME_END", sc.frame_end, "FPS", sc.render.fps)
act = arm.animation_data.action
frames_keyed = set()
for fc in act.fcurves:
    for kp in fc.keyframe_points:
        frames_keyed.add(int(round(kp.co[0])))
print("KEYED", sorted(frames_keyed)[:6], "...", sorted(frames_keyed)[-3:])
print("SPARSE_DONE")
os.remove(p)
