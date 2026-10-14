"""
Builds one YOLO-format dataset with exactly three classes:

    0 cone      1 barrier      2 stop_sign

out of two sources, because no single public dataset has all three:

  * Roboflow Universe "road-obstacle-detection" (laasya-wdzur, CC BY 4.0)
    -> traffic cones and barriers. Its other classes (potholes, debris...) get dropped.
  * COCO 2017 images that contain a stop sign (annotations CC BY 4.0)
    -> stop signs. Every other COCO class in those images gets dropped.

BDD100K was the suggested dataset, but its detection labels don't have cones or
barriers and fold stop signs into a generic "traffic sign", so it can't teach
these three classes on its own.

Splits: the Roboflow export keeps its own train/valid/test split (it was split by
the dataset author, which avoids near-duplicate frames leaking across splits).
COCO train2017 stop-sign images are split 90/10 into train/val, and COCO val2017
stop-sign images become test images.

Usage:
    python scripts/prepare_dataset.py --roboflow data/raw/road_obstacle --coco data/raw --out datasets/nav3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.classes import CLASS_NAMES  # noqa: E402

COCO_STOP_SIGN_ID = 11  # index in the usual 80-class COCO list
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def target_class_for(source_name: str) -> str | None:
    """Maps a source dataset's class name onto ours (or None to drop it)."""
    n = source_name.lower().replace("-", "_").replace(" ", "_")
    if "cone" in n:
        return "cone"
    if "barrier" in n or "barricade" in n:
        return "barrier"
    if "stop" in n and "sign" in n:
        return "stop_sign"
    return None


def yolo_line_to_box(parts: list[str]) -> tuple[float, float, float, float]:
    """Handles plain boxes and polygon rows (some exports mix them in) -> (cx, cy, w, h)."""
    vals = [float(v) for v in parts]
    if len(vals) == 4:
        return tuple(vals)
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def remap_labels(label_path: Path, id_map: dict[int, int]) -> list[str]:
    """Reads a YOLO label file and keeps only the classes we care about, renumbered."""
    if not label_path.exists():
        return []
    kept = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        src_id = int(float(parts[0]))
        if src_id not in id_map:
            continue
        cx, cy, w, h = yolo_line_to_box(parts[1:])
        if w <= 0 or h <= 0:
            continue
        kept.append(f"{id_map[src_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return kept


def place_image(src: Path, dst: Path) -> None:
    """Hard link if we can (no extra disk), plain copy if not."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_sample(out: Path, split: str, name: str, img: Path, lines: list[str], stats: Counter) -> None:
    place_image(img, out / "images" / split / f"{name}{img.suffix.lower()}")
    label_file = out / "labels" / split / f"{name}.txt"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    stats[(split, "images")] += 1
    for line in lines:
        stats[(split, CLASS_NAMES[int(line.split()[0])])] += 1


def add_roboflow(src: Path, out: Path, stats: Counter, negative_frac: float, rng: random.Random) -> None:
    meta = yaml.safe_load((src / "data.yaml").read_text())
    names = meta["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    id_map = {}
    for i, n in enumerate(names):
        tgt = target_class_for(n)
        if tgt is not None:
            id_map[i] = CLASS_NAMES.index(tgt)
    print(f"[roboflow] classes {names} -> keeping {{{', '.join(f'{names[k]}: {CLASS_NAMES[v]}' for k, v in id_map.items())}}}")

    split_names = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for src_split, split in split_names.items():
        img_dir = src / src_split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lines = remap_labels(src / src_split / "labels" / f"{img.stem}.txt", id_map)
            # keep a few road images with none of our classes, so the model also sees "nothing here"
            if not lines and rng.random() > negative_frac:
                continue
            write_sample(out, split, f"rf_{img.stem}", img, lines, stats)


def add_coco_stop_signs(coco_root: Path, out: Path, stats: Counter, val_frac: float, rng: random.Random) -> None:
    id_map = {COCO_STOP_SIGN_ID: CLASS_NAMES.index("stop_sign")}
    for coco_split in ("train2017", "val2017"):
        img_dir = coco_root / "coco_stop" / coco_split
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(images)
        n_val = int(len(images) * val_frac)
        for i, img in enumerate(images):
            lines = remap_labels(coco_root / "coco" / "labels" / coco_split / f"{img.stem}.txt", id_map)
            if not lines:
                continue
            if coco_split == "val2017":
                split = "test"
            else:
                split = "val" if i < n_val else "train"
            write_sample(out, split, f"coco_{img.stem}", img, lines, stats)
        print(f"[coco] {coco_split}: {len(images)} images")


def print_stats(stats: Counter) -> dict:
    table = {}
    print(f"\n{'split':<6} {'images':>7} " + " ".join(f"{c:>10}" for c in CLASS_NAMES))
    for split in ("train", "val", "test"):
        row = {k: stats[(split, k)] for k in ["images", *CLASS_NAMES]}
        table[split] = row
        print(f"{split:<6} {row['images']:>7} " + " ".join(f"{row[c]:>10}" for c in CLASS_NAMES))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roboflow", required=True, type=Path, help="unzipped Roboflow yolov8 export")
    ap.add_argument("--coco", required=True, type=Path, help="folder holding coco/labels/* and coco_stop/*")
    ap.add_argument("--out", default="datasets/nav3", type=Path)
    ap.add_argument("--coco-val-frac", default=0.10, type=float)
    ap.add_argument("--negative-frac", default=0.10, type=float, help="share of no-target road images to keep")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)  # always rebuild from scratch so stale files can't sneak in
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    add_roboflow(args.roboflow, args.out, stats, args.negative_frac, rng)
    add_coco_stop_signs(args.coco, args.out, stats, args.coco_val_frac, rng)
    table = print_stats(stats)
    (args.out / "stats.json").write_text(json.dumps(table, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
