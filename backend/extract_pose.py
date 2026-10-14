"""Extract 3D human pose from a video file and dump it as JSON.

Runs in its own virtualenv (see requirements.txt) so Blender's bundled Python
stays untouched. Uses the MediaPipe Tasks PoseLandmarker, which provides both
2D normalized landmarks and metric 3D "world" landmarks (origin at the hips).

Usage:
    python extract_pose.py input.mp4 output.json [--fps 30] [--model lite|full|heavy]

Output JSON schema:
    {
      "source": "input.mp4",
      "fps": 30.0,
      "frames": [
        {"t": 0.0,
         "landmarks": [{"x","y","z","v"} ... 33],
         "world":     [{"x","y","z"} ... 33]},
        ...
      ]
    }
"""

import argparse
import json
import os
import sys
import urllib.request

MODELS = {
    "lite": "pose_landmarker_lite",
    "full": "pose_landmarker_full",
    "heavy": "pose_landmarker_heavy",
}
MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/{name}/float16/latest/{name}.task"
CACHE_DIR = os.path.expanduser("~/.vid_an")


def ensure_model(kind):
    name = MODELS.get(kind, MODELS["lite"])
    path = os.path.join(CACHE_DIR, name + ".task")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        url = MODEL_BASE.format(name=name)
        print("[vid_an] downloading model", url)
        urllib.request.urlretrieve(url, path)
    return path


def extract(video_path, out_path, fps, kind, min_conf, step=1, resize=0):
    import cv2
    import mediapipe as mp

    step = max(1, int(step))
    model_path = ensure_model(kind)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total:
        print("[vid_an] total %d frames" % total, flush=True)
    out_frames = []
    idx = 0
    kept = 0
    landmarker = PoseLandmarker.create_from_options(options)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            if resize:
                h, w = frame.shape[:2]
                m = max(h, w)
                if m > resize:
                    s = float(resize) / m
                    frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                                       interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(idx * (1000.0 / src_fps))
            res = landmarker.detect_for_video(mp_img, ts)

            if not res.pose_landmarks:
                out_frames.append({"t": idx / src_fps, "landmarks": [], "world": []})
            else:
                lm = res.pose_landmarks[0]
                landmarks = [{"x": p.x, "y": p.y, "z": p.z, "v": getattr(p, "visibility", 1.0)} for p in lm]
                world = []
                if res.pose_world_landmarks:
                    world = [{"x": p.x, "y": p.y, "z": p.z} for p in res.pose_world_landmarks[0]]
                out_frames.append({"t": idx / src_fps, "landmarks": landmarks, "world": world})

            kept += 1
            idx += 1
            if kept % 10 == 0:
                print("[vid_an] frame %d/%d" % (idx, total), flush=True)
    finally:
        landmarker.close()
        cap.release()

    data = {"source": os.path.basename(video_path), "fps": src_fps, "frames": out_frames}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print("[vid_an] wrote", len(out_frames), "frames to", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("output")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--model", choices=list(MODELS), default="full")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame (1 = all)")
    ap.add_argument("--resize", type=int, default=0, help="downscale longest side to N px (0 = off)")
    args = ap.parse_args()
    extract(args.video, args.output, args.fps, args.model, args.min_conf,
            step=args.step, resize=args.resize)


if __name__ == "__main__":
    sys.exit(main())
