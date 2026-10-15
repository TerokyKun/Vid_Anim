# Vid_Anim — quick tests
import sys
sys.path.insert(0, '.')

from core import PoseFrame, PoseSequence, partial_body, motion_solver
import numpy as np


def test_partial_body_fills_legs():
    # Solver space: +X left, +Y backward, +Z up (character faces -Y).
    pts = {
        'nose': (0, 0.1, 1.7),
        'l_shoulder': (-0.15, 0.0, 1.5),
        'r_shoulder': (0.15, 0.0, 1.5),
        'l_hip': (-0.1, 0.0, 0.9),
        'r_hip': (0.1, 0.0, 0.9),
        'l_elbow': (-0.3, 0.0, 1.2),
        'r_elbow': (0.3, 0.0, 1.2),
        'l_wrist': (-0.4, 0.0, 1.0),
        'r_wrist': (0.3, 0.0, 1.0),
        'l_ear': (-0.05, 0.1, 1.8),
        'r_ear': (0.05, 0.1, 1.8),
    }
    valid = {k: True for k in pts}
    p2, synth = partial_body.complete(pts, valid)
    # hip sits below the nose, and the filled leg chain descends (Z decreases).
    assert p2['l_hip'][2] < p2['nose'][2]
    assert p2['l_knee'][2] < p2['l_hip'][2]
    assert p2['l_ankle'][2] < p2['l_knee'][2]
    # legs stay under the body (not swung backward along -Y).
    assert abs(p2['l_knee'][1]) < 0.2
    assert synth and ('l_foot' in synth)


def test_build_joints_fill_legs_descend():
    # Real path: build_joints maps MediaPipe (Y-down) to solver space, then fills
    # the low-confidence legs. The filled legs must descend (+Z up => knee/ankle
    # below the hip), not swing backward along -Y.
    lm = np.zeros((33, 3), np.float32)
    conf = np.ones((33,), np.float32)

    def setp(i, x, y, z, c=1.0):
        lm[i] = (x, y, z)
        conf[i] = c

    setp(0, 0.0, -1.7, 0.0)                 # nose (up = -y in MP)
    setp(11, -0.15, -1.5, 0.0)
    setp(12, 0.15, -1.5, 0.0)               # shoulders
    setp(23, -0.10, -0.9, 0.0)
    setp(24, 0.10, -0.9, 0.0)               # hips (visible)
    for i in (25, 26, 27, 28, 31, 32):      # knees/ankles/feet hidden
        setp(i, 0.0, 0.0, 0.0, c=0.0)

    joints, synth = motion_solver.build_joints(lm, confidence=conf, fill_partial=True)
    assert joints['l_knee'][2] < joints['l_hip'][2]
    assert joints['l_ankle'][2] < joints['l_knee'][2]
    assert abs(joints['l_knee'][1]) < 0.2   # not thrown backward


def test_head_orients_from_face_not_torso():
    # Regression: the head axis used to be forced equal to the torso axis
    # (nose_up = nose + torso_up), so the head could never nod/turn/tilt.
    # Now head_top comes from the face plane (nose + ears) and the head's roll
    # comes from the ear line, so a tilted face must tilt the head frame.
    skel = motion_solver.SKELETON
    joints = {
        'mid_hip': np.array([0.0, 0.0, 0.9]),
        'mid_shoulder': np.array([0.0, 0.0, 1.4]),
        'l_shoulder': np.array([0.18, 0.0, 1.4]),
        'r_shoulder': np.array([-0.18, 0.0, 1.4]),
        'l_hip': np.array([0.09, 0.0, 0.9]),
        'r_hip': np.array([-0.09, 0.0, 0.9]),
        'nose': np.array([0.0, -0.12, 1.66]),
        'l_ear': np.array([0.10, 0.0, 1.70]),   # left ear raised => tilted head
        'r_ear': np.array([-0.06, 0.0, 1.58]),
    }
    valid = {k: True for k in joints}
    motion_solver._derive_head(joints, valid)     # adds head_top from the face
    world = motion_solver.solve_frame(skel, joints, None)
    head_x = world['Head'][:3, 0]
    hips_x = world['Hips'][:3, 0]
    assert abs(head_x[2]) > 0.1                    # ear-line tilt => +Z component
    assert abs(hips_x[2]) < 0.1                    # torso stays level
    assert not np.allclose(world['Head'][:3, :3], world['Hips'][:3, :3], atol=0.05)


def test_motion_solver_basic():
    f = PoseFrame(frame=0, timestamp=0.0, landmarks_3d=np.zeros((33,3), np.float32),
                  confidence=np.ones((33,), np.float32), backend='test')
    seq = PoseSequence(frames=[f], fps=30, source='t')
    s = motion_solver.solve_sequence(seq)
    assert len(s.skeleton.bones) == 18


def test_mediapipe_mapping_is_proper_rotation():
    # Regression: MediaPipe (x=left, y=down, z=toward camera) -> Blender
    # (X=left, -Y=front, Z=up) must be a proper rotation between right-handed
    # frames. The old (x, -z, -y) was a reflection (det -1) that mirrored the
    # body front-to-back and twisted the limbs.
    basis = np.eye(3)
    mapped = motion_solver.mediapipe_to_solver(basis)   # columns -> mapped axes
    assert np.linalg.det(mapped) > 0.999
    # toward-camera direction (0, 0, -1) must become character front (-Y)
    front = motion_solver.mediapipe_to_solver(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(front, [0.0, -1.0, 0.0])
    # person's left (+x) stays +X; up (MP -y) becomes +Z
    assert np.allclose(motion_solver.mediapipe_to_solver(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    assert np.allclose(motion_solver.mediapipe_to_solver(np.array([0.0, -1.0, 0.0])), [0.0, 0.0, 1.0])


def test_pose_schema_roundtrip(tmp_path):
    from core import pose_schema

    f = PoseFrame(frame=0, timestamp=0.0, landmarks_2d=np.zeros((33,3),np.float32),
                  landmarks_3d=np.zeros((33,3),np.float32), confidence=np.ones((33,),np.float32),
                  backend='test')
    seq = PoseSequence(frames=[f], fps=30, source='t')
    jl = tmp_path / 's.jsonl'
    npz = tmp_path / 's.npz'
    pose_schema.to_jsonl(str(jl), seq)
    pose_schema.to_npz(str(npz), seq)
    s2 = pose_schema.from_jsonl(str(jl))
    s3 = pose_schema.from_npz(str(npz))
    assert len(s2) == 1 and len(s3) == 1
