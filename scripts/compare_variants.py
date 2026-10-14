"""
Before/after picture: the same image through the FP32 baseline and an optimized
model, side by side, so you can see the INT8 model still finds the same things.

    python scripts/compare_variants.py --image some.jpg \
        --a optimization/models/best_fp32.pt --b optimization/models/best_int8.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402
from navdet.pipeline import NavPipeline  # noqa: E402


def run(weights: str, img: np.ndarray, camera: str, device: str | None, title: str) -> np.ndarray:
    pipe = NavPipeline(weights, camera, device=device)
    pipe(img)  # warm-up
    t0 = time.perf_counter()
    out, dets = pipe(img)
    ms = (time.perf_counter() - t0) * 1000
    label = f"{title}  |  {len(dets)} objects  |  {ms:.0f} ms end-to-end"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for d in dets:
        print(f"  {title}: {d.class_name} {d.confidence:.2f} {d.distance.label()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--a", default="optimization/models/best_fp32.pt")
    ap.add_argument("--a-title", default="Before: PyTorch FP32 (CPU)")
    ap.add_argument("--a-device", default="cpu")
    ap.add_argument("--b", default="optimization/models/best_int8.onnx")
    ap.add_argument("--b-title", default="After: ONNX Runtime INT8 (CPU)")
    ap.add_argument("--b-device", default="cpu")
    ap.add_argument("--camera", default="configs/camera.yaml")
    ap.add_argument("--out", default="optimization/before_after.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    cam = str(repo_path(args.camera))
    left = run(str(repo_path(args.a)), img, cam, args.a_device, args.a_title)
    right = run(str(repo_path(args.b)), img, cam, args.b_device, args.b_title)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([left, np.full((img.shape[0], 8, 3), 255, np.uint8), right]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
