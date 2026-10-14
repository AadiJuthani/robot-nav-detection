"""
Fine-tunes a COCO-pretrained YOLO on our three classes (transfer learning).

The pretrained backbone already knows edges, shapes and textures from COCO,
so only the later layers really have to figure out what a cone or a barrier
looks like. That's why this works with a few thousand images and a laptop.

    # quick end-to-end check that paths/labels/logging work (1 epoch, 5% of data)
    python scripts/train.py --smoke

    # the real run, capped by wall-clock time
    python scripts/train.py --epochs 60 --hours 1.5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import REPO_ROOT, repo_path, resolved_data_yaml  # noqa: E402
from navdet.pipeline import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting weights")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hours", type=float, default=None, help="wall-clock cap, overrides epochs when set")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15, help="stop early if val mAP stalls this many epochs")
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (backbone is 0-10)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=None, help="'ram' keeps decoded images in memory, a decent speedup")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu, picked automatically if empty")
    ap.add_argument("--name", default="yolo11n_nav3")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of the data")
    args = ap.parse_args()

    from ultralytics import YOLO

    device = args.device or pick_device()
    data = resolved_data_yaml(args.data)
    model = YOLO(args.model)

    extra = {}
    if args.smoke:
        extra = dict(epochs=1, fraction=0.05, name=f"{args.name}_smoke", plots=False)
    else:
        extra = dict(epochs=args.epochs, name=args.name)
        if args.hours:
            extra["time"] = args.hours

    print(f"training {args.model} on {device} | data={data}")
    model.train(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        freeze=args.freeze or None,
        workers=args.workers,
        cache=args.cache or False,
        project=str(REPO_ROOT / "runs"),
        exist_ok=True,
        seed=0,
        deterministic=False,  # faster on mps, and we're not chasing bit-exact reruns
        **extra,
    )

    best = Path(model.trainer.best)
    if not args.smoke and best.exists():
        dst = repo_path("outputs/weights/best.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"copied {best} -> {dst}")


if __name__ == "__main__":
    main()
