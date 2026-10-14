import os
import sys
import json
import math

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender_addon"))
sys.path.insert(0, HERE)

import retarget_test_lib as lib
from vid_an import mocap
from vid_an.bone_maps import auto_pairs
from vid_an import retarget as rt

N_FRAMES = 10


def p(x, y, z):
    return {"x": x, "y": y, "z": z}


def make_world(theta_deg):
    th = math.radians(theta_deg)
    d = Vector((math.cos(th), -math.sin(th), 0.0))
    sh = Vector((0.18, -0.52, 0.0))
    elbow = sh + d * 0.24
    wrist = elbow + d * 0.23
    index = wrist + d * 0.10
    pts = [p(0, 0, 0) for _ in range(33)]
    pts[0] = p(0, -0.72, -0.15)
    pts[7] = p(0.08, -0.70, -0.05)
    pts[8] = p(-0.08, -0.70, -0.05)
    pts[11] = p(sh.x, sh.y, sh.z)
    pts[12] = p(-0.18, -0.52, 0.0)
    pts[13] = p(elbow.x, elbow.y, elbow.z)
    pts[14] = p(-0.42, -0.52, 0.0)
    pts[15] = p(wrist.x, wrist.y, wrist.z)
    pts[16] = p(-0.65, -0.52, 0.0)
    pts[19] = p(index.x, index.y, index.z)
    pts[20] = p(-0.75, -0.52, 0.0)
    pts[23] = p(0.09, 0.0, 0.0)
    pts[24] = p(-0.09, 0.0, 0.0)
    pts[25] = p(0.09, 0.45, 0.0)
    pts[26] = p(-0.09, 0.45, 0.0)
    pts[27] = p(0.09, 0.85, 0.0)
    pts[28] = p(-0.09, 0.85, 0.0)
    pts[31] = p(0.09, 0.90, -0.15)
    pts[32] = p(-0.09, 0.90, -0.15)
    return pts


frames = []
for i in range(N_FRAMES):
    theta = 90.0 * i / (N_FRAMES - 1)
    frames.append({"t": i / 30.0, "landmarks": [], "world": make_world(theta)})

data = {"source": "synthetic", "fps": 30.0, "frames": frames}
json_path = os.path.join(HERE, "mocap_synth.json")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(data, fh)

lib.wipe()
src = mocap.build_source_from_json(bpy.context, json_path)
print("SOURCE", src.name, "bones", len(src.pose.bones))


def wdir(obj, name):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).col[1].xyz.normalized()


worst = 0.0
for i in range(N_FRAMES):
    f = i + 1
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    th = math.radians(90.0 * i / (N_FRAMES - 1))
    expect = Vector((math.cos(th), 0.0, math.sin(th)))
    got = wdir(src, "UpperArm.L")
    err = math.degrees(expect.angle(got))
    worst = max(worst, err)
    if f in (1, N_FRAMES):
        print("  f%02d UpperArm.L expect=(%.2f,%.2f,%.2f) got=(%.2f,%.2f,%.2f) err=%.2f"
              % (f, expect.x, expect.y, expect.z, got.x, got.y, got.z, err))
print("SOLVER_WORST_DIR_ERR %.3f" % worst)

rig = lib.gen_rigify()
pairs = auto_pairs(src, rig)
print("PAIRS", len(pairs))
rt.retarget(bpy.context, src, rig, pairs, 1, N_FRAMES)


def delta(obj, name):
    b = obj.data.bones[name]
    rest = obj.matrix_world @ b.matrix_local
    cur = obj.matrix_world @ obj.pose.bones[name].matrix
    return (cur @ rest.inverted()).to_quaternion()


worst_dir = 0.0
for f in range(1, N_FRAMES + 1):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    for sname, dname in [("UpperArm.L", "DEF-upper_arm.L"), ("ForeArm.L", "DEF-forearm.L"),
                         ("Thigh.L", "DEF-thigh.L"), ("Spine", "DEF-spine")]:
        sd = wdir(src, sname)
        dd = wdir(rig, dname)
        worst_dir = max(worst_dir, math.degrees(sd.angle(dd)))
print("RETARGET_WORST_DIR_ERR %.3f" % worst_dir)
print("TEST_DONE")
