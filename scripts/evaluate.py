"""
Scores a model on the held-out test split and saves the numbers.

Works for any format Ultralytics can load (.pt, .onnx, .mlpackage), which is what
makes the before/after comparison fair: every variant is scored on the exact
same unseen images with the same settings.

    python scripts/evaluate.py --weights outputs/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.classes import CLASS_NAMES  # noqa: E402
from navdet.paths import REPO_ROOT, repo_path, resolved_data_yaml  # noqa: E402
from navdet.pipeline import pick_device  # noqa: E402


def evaluate(weights: str, data: str = "configs/data.yaml", split: str = "test", imgsz: int = 640,
             device: str | None = None, half: bool = False) -> dict:
    """Runs model.val() and pulls out the overall and per-class numbers."""
    from ultralytics import YOLO

    model = YOLO(str(repo_path(weights)), task="detect")
    m = model.val(
        data=resolved_data_yaml(data),
        split=split,
        imgsz=imgsz,
        device=device or pick_device(),
        half=half,
        plots=False,
        verbose=False,
        project=str(REPO_ROOT / "runs" / "val"),
        name=Path(weights).stem,
        exist_ok=True,
    )
    box = m.box
    per_class = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[int(cls_idx)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50_95": float(box.ap[i]),
        }
    return {
        "weights": str(weights),
        "split": split,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }


def to_markdown(r: dict) -> str:
    lines = [
        f"Overall ({r['split']} split): mAP@50 **{r['mAP50']:.3f}**, mAP@50-95 {r['mAP50_95']:.3f}, "
        f"precision {r['precision']:.3f}, recall {r['recall']:.3f}",
        "",
        "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|",
    ]
    for name, c in r["per_class"].items():
        lines.append(f"| {name} | {c['precision']:.3f} | {c['recall']:.3f} | {c['mAP50']:.3f} | {c['mAP50_95']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/weights/best.pt")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/metrics")
    args = ap.parse_args()

    r = evaluate(args.weights, args.data, args.split, args.imgsz, args.device)
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.weights).stem}_{args.split}"
    (out / f"{stem}.json").write_text(json.dumps(r, indent=2))
    (out / f"{stem}.md").write_text(to_markdown(r) + "\n")
    print(to_markdown(r))


if __name__ == "__main__":
    main()
