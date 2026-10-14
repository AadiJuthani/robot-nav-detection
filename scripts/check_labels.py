"""
Draws the ground-truth boxes on random images and tiles them into one mosaic.

This is the 5-minute check that catches a wrong class mapping or a broken box
conversion before burning an hour of training on it.

    python scripts/check_labels.py --split train --n 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.annotate import draw_box  # noqa: E402
from navdet.classes import CLASS_COLORS, CLASS_NAMES, display_name  # noqa: E402
from navdet.paths import repo_path  # noqa: E402


def draw_gt(img_path: Path, label_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            name = CLASS_NAMES[int(c)]
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            draw_box(img, box, display_name(name), CLASS_COLORS[name])
    return img


def tile(images: list[np.ndarray], cols: int = 5, cell: int = 320) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        s = cell / max(h, w)
        small = cv2.resize(img, (int(w * s), int(h * s)))
        r, c = divmod(i, cols)
        y0, x0 = r * cell, c * cell
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only", default=None, help="only images containing this class, e.g. barrier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/label_check")
    args = ap.parse_args()

    cfg = yaml.safe_load(repo_path(args.data).read_text())
    root = repo_path(cfg["path"])
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    images = sorted(img_dir.iterdir())
    if args.only:
        want = str(CLASS_NAMES.index(args.only))
        images = [p for p in images
                  if (lbl_dir / f"{p.stem}.txt").exists()
                  and any(l.split()[0] == want for l in (lbl_dir / f"{p.stem}.txt").read_text().splitlines())]
    random.Random(args.seed).shuffle(images)
    picks = images[: args.n]

    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drawn = [draw_gt(p, lbl_dir / f"{p.stem}.txt") for p in picks]
    tag = f"{args.split}_{args.only or 'all'}"
    cv2.imwrite(str(out / f"mosaic_{tag}.jpg"), tile(drawn))
    print(f"saved {out / f'mosaic_{tag}.jpg'} from {len(picks)} images")


if __name__ == "__main__":
    main()
