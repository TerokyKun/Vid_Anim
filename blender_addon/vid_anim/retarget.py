import math

import bpy
from mathutils import Matrix, Quaternion, Vector

from .bone_maps import auto_pairs, semantic_key


def _world_rest(obj, bone):
    return obj.matrix_world @ bone.matrix_local


def _bone_depth(bone):
    depth = 0
    p = bone.parent
    while p is not None:
        depth += 1
        p = p.parent
    return depth


def force_fk(target_arm):
    """Make limb chains follow their FK controls.

    On control rigs like Rigify, an IK constraint is stacked on top of the FK
    control and normally wins. Here we drop the IK constraint's influence,
    remove its drivers and set the FK/IK switch to FK so FK keyframes drive
    the deformation regardless of the evaluation context.
    """
    ad = target_arm.animation_data
    for pb in target_arm.pose.bones:
        for con in pb.constraints:
            sub = getattr(con, "subtarget", "") or ""
            if con.type == 'COPY_TRANSFORMS' and "ik" in sub.lower():
                con.influence = 0.0
                if ad is not None:
                    path = 'pose.bones["%s"].constraints["%s"].influence' % (pb.name, con.name)
                    for drv in [d for d in ad.drivers if d.data_path == path]:
                        try:
                            ad.drivers.remove(drv)
                        except Exception:
                            pass
        for key in list(pb.keys()):
            if key == "IK_FK":
                try:
                    pb[key] = 0.0
                except Exception:
                    pass


def build_pairs(source_arm, target_arm, target_preset="AUTO", manual_map=None):
    if target_preset == "AUTO" and not manual_map:
        return auto_pairs(source_arm, target_arm)
    if manual_map:
        result = []
        src_index = {}
        for b in source_arm.data.bones:
            k = semantic_key(b.name)
            if k:
                src_index[k] = b.name
        for sem, tname in manual_map.items():
            sname = src_index.get(sem)
            if sname and tname in target_arm.pose.bones:
                result.append((sname, tname, sem))
        return result
    return auto_pairs(source_arm, target_arm)


def iter_retarget(context, source_arm, target_arm, pairs, frame_start, frame_end,
                  step=1, yaw_offset_deg=0.0, loc_scale=1.0, root_sem="hips",
                  scene_fps=None, fk_mode=True):
    """Bake the source motion onto the target, yielding (done, total) per frame.

    Local bone matrices are computed analytically instead of writing
    ``pose_bone.matrix`` and calling ``view_layer.update()`` for every bone.
    That removes tens of thousands of depsgraph updates and lets the caller
    run the bake in small, UI-friendly slices.
    """
    if not pairs:
        raise RuntimeError("No bone pairs to retarget.")

    if fk_mode:
        force_fk(target_arm)

    scene = context.scene
    view_layer = context.view_layer

    yaw = Matrix.Rotation(math.radians(yaw_offset_deg), 4, 'Z')

    tgt_bones = {t: target_arm.data.bones[t] for (_, t, _) in pairs}
    pairs_sorted = sorted(pairs, key=lambda p: _bone_depth(tgt_bones[p[1]]))

    tgt_rest_world = {t: _world_rest(target_arm, target_arm.data.bones[t])
                      for (_, t, _) in pairs}
    src_rest_world = {s: _world_rest(source_arm, source_arm.data.bones[s])
                      for (s, _, _) in pairs}

    loop_count = (frame_end - frame_start) // step + 1
    root_bone = None
    for (_, t, sem) in pairs_sorted:
        if sem == root_sem:
            root_bone = t
            break
    if root_bone is None and pairs_sorted:
        root_bone = pairs_sorted[0][1]

    for pb in target_arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    prev_q = {}
    for i in range(loop_count):
        frame = frame_start + i * step
        scene.frame_set(frame)
        view_layer.update()

        pose_arm = {}
        needs_update = True
        for (s, t, sem) in pairs_sorted:
            s_bone = source_arm.pose.bones[s]
            s_pose_world = source_arm.matrix_world @ s_bone.matrix

            desired_rot = (yaw.to_3x3() @ s_pose_world.to_3x3()).to_4x4()
            desired_rot.translation = tgt_rest_world[t].translation
            if sem == root_sem:
                disp = s_pose_world.translation - src_rest_world[s].translation
                desired_rot.translation = (yaw @ disp) + tgt_rest_world[t].translation
            desired_arm = target_arm.matrix_world.inverted() @ desired_rot

            tbone = tgt_bones[t]
            parent = tbone.parent
            if parent is not None:
                pp = pose_arm.get(parent.name)
                if pp is None:
                    if needs_update:
                        view_layer.update()
                        needs_update = False
                    pp = target_arm.pose.bones[parent.name].matrix
                base = pp @ parent.matrix_local.inverted() @ tbone.matrix_local
                basis = base.inverted() @ desired_arm
            else:
                basis = tbone.matrix_local.inverted() @ desired_arm

            loc, rot, _scale = basis.decompose()
            if sem != root_sem:
                loc = Vector((0.0, 0.0, 0.0))
            elif loc_scale != 1.0:
                loc = loc * loc_scale

            pb = target_arm.pose.bones[t]
            pq = prev_q.get(t)
            if pq is not None and pq.dot(rot) < 0.0:
                rot.negate()
            prev_q[t] = rot.copy()

            pb.rotation_quaternion = rot
            pb.location = loc
            pose_arm[t] = desired_arm
            needs_update = True

            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if sem == root_sem:
                pb.keyframe_insert(data_path="location", frame=frame)

        yield i + 1, loop_count

    scene.frame_start = frame_start
    scene.frame_end = frame_start + (loop_count - 1) * step
    scene.frame_set(frame_start)

    for fc in _action_fcurves(target_arm):
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
        fc.update()

    return loop_count


def retarget(context, source_arm, target_arm, pairs, frame_start, frame_end,
             step=1, yaw_offset_deg=0.0, loc_scale=1.0, root_sem="hips",
             scene_fps=None, fk_mode=True):
    gen = iter_retarget(context, source_arm, target_arm, pairs, frame_start, frame_end,
                        step=step, yaw_offset_deg=yaw_offset_deg, loc_scale=loc_scale,
                        root_sem=root_sem, scene_fps=scene_fps, fk_mode=fk_mode)
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break
    return result


def _action_fcurves(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    try:
        return list(ad.action.fcurves)
    except AttributeError:
        return []
