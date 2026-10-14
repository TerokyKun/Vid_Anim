# Vid_Anim — quick tests
import sys
sys.path.insert(0, '.')

from core import PoseFrame, PoseSequence, partial_body, motion_solver
import numpy as np


def test_partial_body_fills_legs():
    pts = {
        'nose': (0, 1.7, -0.1),
        'l_shoulder': (-0.15, 1.5, 0),
        'r_shoulder': (0.15, 1.5, 0),
        'l_hip': (-0.1, 0.9, 0),
        'r_hip': (0.1, 0.9, 0),
        'l_knee': (-0.1, 0.5, 0.2),
        'r_knee': (0.1, 0.5, 0.2),
        'l_ankle': (-0.1, 0.1, 0.25),
        'r_ankle': (0.1, 0.1, 0.25),
        'l_foot_index': (-0.1, 0.0, 0.25),
        'r_foot_index': (0.1, 0.0, 0.25),
        'l_elbow': (-0.3, 1.2, 0),
        'r_elbow': (0.3, 1.2, 0),
        'l_wrist': (-0.4, 1.0, 0),
        'r_wrist': (0.3, 1.0, 0),
        'l_ear': (-0.05, 1.8, -0.1),
        'r_ear': (0.05, 1.8, -0.1),
    }
    valid = {k: True for k in pts}
    for k in ('l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle','l_foot_index','r_foot_index'):
        valid[k] = False
    p2, synth = partial_body.complete(pts, valid)
    assert p2['l_hip'][1] < p2['nose'][1]
    assert (synth and ('l_foot_index' in synth or 'l_foot' in synth))
    assert p2['l_ankle'][1] > 0


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
