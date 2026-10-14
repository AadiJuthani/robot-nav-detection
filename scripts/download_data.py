"""
Downloads the raw data into data/raw/ (which is gitignored).

  * Roboflow Universe "road-obstacle-detection" (cones + barriers).
    Needs a free Roboflow API key:  export ROBOFLOW_API_KEY=...   (or put it in .env)
  * COCO 2017 labels in YOLO format, plus only the images that contain a stop sign
    (about 1,800 images instead of the full 20 GB).

    python scripts/download_data.py
    python scripts/prepare_dataset.py --roboflow data/raw/road_obstacle --coco data/raw

Note on the exact Roboflow export: I forked the Universe project into my own
workspace and generated a version with only auto-orient (original image sizes,
no resize, no augmentation). The public Universe version 3 that this script
pulls by default was resized to 512x512 by its author, so numbers can differ a
little if you rebuild from it.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402

COCO_LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_IMG_URL = "http://images.cocodataset.org/{split}/{name}.jpg"
COCO_STOP_SIGN_ID = 11


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def get_roboflow(out: Path, workspace: str, project: str, version: int) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = repo_path(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        print("no ROBOFLOW_API_KEY set, skipping the cone/barrier download")
        return
    from roboflow import Roboflow  # pip install roboflow

    rf = Roboflow(api_key=key)
    rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(out))


def get_coco_stop_signs(root: Path, workers: int) -> None:
    zip_path = root / "coco2017labels.zip"
    if not (root / "coco" / "labels").exists():
        print("downloading COCO labels (~50 MB)")
        fetch(COCO_LABELS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        zip_path.unlink()

    for split in ("train2017", "val2017"):
        label_dir = root / "coco" / "labels" / split
        names = sorted(p.stem for p in label_dir.glob("*.txt")
                       if any(l.split()[0] == str(COCO_STOP_SIGN_ID) for l in p.read_text().splitlines() if l))
        print(f"{split}: {len(names)} images with a stop sign")
        out = root / "coco_stop" / split
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda n: fetch(COCO_IMG_URL.format(split=split, name=n), out / f"{n}.jpg"), names))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--rf-workspace", default="laasya-wdzur")
    ap.add_argument("--rf-project", default="road-obstacle-detection-6zwj5")
    ap.add_argument("--rf-version", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = repo_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    get_roboflow(root / "road_obstacle", args.rf_workspace, args.rf_project, args.rf_version)
    get_coco_stop_signs(root, args.workers)


if __name__ == "__main__":
    main()
