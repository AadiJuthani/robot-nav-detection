"""
Extra credit: track cones across video frames with Lucas-Kanade optical flow.

Running the detector on every frame is the expensive part on an edge device.
Optical flow is cheap, so the idea here is:

  1. run the detector every --detect-every frames (a "keyframe")
  2. inside each detected box, pick good corners to follow (Shi-Tomasi)
  3. on the frames in between, track those corners with pyramidal Lucas-Kanade
     and slide each box by the median motion of its points
  4. distance is recomputed from the moved box every frame

Each track also keeps a short trail so you can see the cones move through the frame.

    python scripts/track_flow.py --video clip.mp4 --detect-every 5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.annotate import annotate  # noqa: E402
from navdet.paths import repo_path  # noqa: E402
from navdet.pipeline import Detection, NavPipeline  # noqa: E402

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@dataclass
class Track:
    det: Detection
    points: np.ndarray  # Nx1x2 float32 corners inside the box
    trail: list = field(default_factory=list)


def seed_points(gray: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in box)
    mask = np.zeros_like(gray)
    mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01, minDistance=4, mask=mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="outputs/weights/best.pt")
    ap.add_argument("--camera", default="configs/camera.yaml")
    ap.add_argument("--classes", nargs="+", default=["cone"], help="which classes to track")
    ap.add_argument("--detect-every", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="outputs/extra/optical_flow.mp4")
    args = ap.parse_args()

    pipe = NavPipeline(str(repo_path(args.weights)), str(repo_path(args.camera)), conf=args.conf)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks: list[Track] = []
    prev_gray = None
    t_detect = t_flow = 0.0
    n_detect = n_flow = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i % args.detect_every == 0 or prev_gray is None:
            t0 = time.perf_counter()
            dets = [d for d in pipe.detect(frame) if d.class_name in args.classes]
            t_detect += time.perf_counter() - t0
            n_detect += 1
            new_tracks = []
            for d in dets:
                pts = seed_points(gray, d.xyxy)
                if pts is None:
                    continue
                # carry the trail over from the nearest old track so the line doesn't reset every keyframe
                cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
                trail = []
                if tracks:
                    near = min(tracks, key=lambda t: np.hypot(t.trail[-1][0] - cx, t.trail[-1][1] - cy) if t.trail else 1e9)
                    if near.trail and np.hypot(near.trail[-1][0] - cx, near.trail[-1][1] - cy) < 0.5 * (d.xyxy[3] - d.xyxy[1]):
                        trail = near.trail
                new_tracks.append(Track(d, pts, trail + [(cx, cy)]))
            tracks = new_tracks
        else:
            t0 = time.perf_counter()
            kept = []
            for t in tracks:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, t.points, None, **LK)
                good_old = t.points[status.ravel() == 1]
                good_new = nxt[status.ravel() == 1]
                if len(good_new) < 3:
                    continue  # lost it, the next keyframe will pick it back up
                dx, dy = np.median((good_new - good_old).reshape(-1, 2), axis=0)
                x1, y1, x2, y2 = t.det.xyxy
                t.det.xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                t.det.distance = pipe.estimator.estimate(t.det.class_name, t.det.xyxy, (w, h))
                t.points = good_new.reshape(-1, 1, 2)
                t.trail.append(((t.det.xyxy[0] + t.det.xyxy[2]) / 2, (t.det.xyxy[1] + t.det.xyxy[3]) / 2))
                kept.append(t)
            tracks = kept
            t_flow += time.perf_counter() - t0
            n_flow += 1

        vis = annotate(frame, [t.det for t in tracks])
        for t in tracks:
            pts = np.int32(t.trail[-30:])
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
            for p in t.points.reshape(-1, 2):
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        tag = "detector" if i % args.detect_every == 0 else "optical flow"
        cv2.putText(vis, f"frame {i}  [{tag}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)
        prev_gray = gray
        i += 1

    cap.release()
    writer.release()
    if n_detect:
        print(f"detector: {1000 * t_detect / n_detect:.1f} ms/frame over {n_detect} keyframes")
    if n_flow:
        print(f"optical flow: {1000 * t_flow / n_flow:.1f} ms/frame over {n_flow} frames")
    print(f"saved {out} ({i} frames)")


if __name__ == "__main__":
    main()
