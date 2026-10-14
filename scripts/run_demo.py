"""
The whole pipeline on images or a video: detect -> distance -> annotated output.

    python scripts/run_demo.py --source path/to/images_or_video --out outputs/demo

Every detection gets a label like "Cone, 1.5m". A "<" in front (e.g. "Cone, <2.1m")
means the box is cut off by the image edge, so the real object is closer than
that. "?m" means the box was too small to trust.
A CSV next to the images lists every detection with its distance.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402
from navdet.pipeline import NavPipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".ogv"}


def run_images(pipe: NavPipeline, paths: list[Path], out: Path, writer) -> None:
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skipping {p} (couldn't read it)")
            continue
        annotated, dets = pipe(img)
        cv2.imwrite(str(out / f"{p.stem}_annotated.jpg"), annotated)
        for d in dets:
            writer.writerow([p.name, d.class_name, f"{d.confidence:.3f}",
                             *(f"{v:.1f}" for v in d.xyxy),
                             "" if d.distance.meters is None else f"{d.distance.meters:.2f}",
                             int(d.distance.truncated)])
        summary = ", ".join(f"{d.class_name} {d.distance.label()}" for d in dets) or "nothing"
        print(f"{p.name}: {summary}")


def run_video(pipe: NavPipeline, path: Path, out: Path, writer) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out / f"{path.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, dets = pipe(frame)
        vw.write(annotated)
        for d in dets:
            writer.writerow([f"{path.name}#{i}", d.class_name, f"{d.confidence:.3f}",
                             *(f"{v:.1f}" for v in d.xyxy),
                             "" if d.distance.meters is None else f"{d.distance.meters:.2f}",
                             int(d.distance.truncated)])
        i += 1
    cap.release()
    vw.release()
    print(f"{path.name}: {i} frames -> {out / f'{path.stem}_annotated.mp4'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="an image, a folder of images, or a video")
    ap.add_argument("--weights", default="outputs/weights/best.pt")
    ap.add_argument("--camera", default="configs/camera.yaml")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--class-conf", nargs="*", default=[], metavar="CLASS=CONF",
                    help="per-class thresholds, e.g. stop_sign=0.6 cone=0.25")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()

    class_conf = {k: float(v) for k, v in (kv.split("=") for kv in args.class_conf)}
    pipe = NavPipeline(str(repo_path(args.weights)), str(repo_path(args.camera)),
                       conf=args.conf, imgsz=args.imgsz, device=args.device, class_conf=class_conf)
    src = Path(args.source)
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "class", "conf", "x1", "y1", "x2", "y2", "distance_m", "cut_off"])
        if src.is_dir():
            run_images(pipe, sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS), out, writer)
        elif src.suffix.lower() in VID_EXTS:
            run_video(pipe, src, out, writer)
        else:
            run_images(pipe, [src], out, writer)


if __name__ == "__main__":
    main()
