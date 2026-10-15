"""
Speed and accuracy for every model variant, so the before/after is one table.

For each variant it measures:
  * file size
  * FPS on CPU and on GPU (Apple GPU through MPS / CoreML here, CUDA if you run it on NVIDIA)
  * mAP on the same held-out test split, because a speedup means nothing if accuracy falls apart

FPS here is model inference only (batch 1, 640x640, after warm-up, averaged over
--runs frames). Pre/post-processing is the same for every variant, so leaving it
out keeps the comparison about the model itself.

    python scripts/benchmark.py            # after scripts/optimize.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import argparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate  # noqa: E402
from optimize import letterbox  # noqa: E402

MODELS = repo_path("optimization/models")

# (label, file, backend, precision)
VARIANTS = [
    ("Baseline (PyTorch)", "best_fp32.pt", "torch", "FP32"),
    ("CoreML FP16", "best_fp16.mlpackage", "coreml", "FP16"),
    ("ONNX Runtime INT8", "best_int8.onnx", "onnx", "INT8 weights + activations"),
    ("Pruned 30%", "best_pruned30.pt", "torch", "FP32, 30% of conv weights zeroed"),
]


@dataclass
class Row:
    variant: str
    file: str
    precision: str
    size_mb: float = 0.0
    fps: dict = field(default_factory=dict)  # {"cpu": .., "gpu": .., "ane": ..}
    mAP50: float | None = None
    mAP50_95: float | None = None
    notes: list = field(default_factory=list)


def size_mb(path: Path) -> float:
    if path.is_dir():  # .mlpackage is a folder
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


def hardware() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version()}
    if sys.platform == "darwin":
        q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()  # noqa: E731
        info["chip"] = q("machdep.cpu.brand_string")
        info["ram_gb"] = round(int(q("hw.memsize")) / 2**30)
        info["cpu_cores"] = q("hw.ncpu")
    else:
        info["chip"] = platform.processor()
    import torch
    info["torch"] = torch.__version__
    info["torch_threads"] = torch.get_num_threads()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["gpu"] = f"{info.get('chip', 'Apple')} GPU (MPS / CoreML)"
    return info


def timed_fps(run, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        run()
    t0 = time.perf_counter()
    for _ in range(runs):
        run()
    return runs / (time.perf_counter() - t0)


def torch_runner(path: Path, device: str, x: np.ndarray):
    import torch
    from ultralytics import YOLO

    model = YOLO(str(path)).model.fuse().eval().to(device)
    t = torch.from_numpy(x).to(device)

    def run():
        with torch.inference_mode():
            model(t)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    return run


def onnx_runner(path: Path, x: np.ndarray):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda: sess.run(None, {name: x})


def coreml_runner(path: Path, unit: str, canvas_rgb: np.ndarray):
    import coremltools as ct
    from PIL import Image

    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.ALL}
    model = ct.models.MLModel(str(path), compute_units=units[unit])
    name = model.get_spec().description.input[0].name
    img = Image.fromarray(canvas_rgb)
    return lambda: model.predict({name: img})


def gpu_device() -> str | None:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def bench_variant(label, fname, backend, precision, x, canvas_rgb, args) -> Row | None:
    path = MODELS / fname
    if not path.exists():
        print(f"skip {label}: {path.name} not found")
        return None
    row = Row(label, fname, precision, size_mb=round(size_mb(path), 2))
    gpu = gpu_device()

    def attempt(key, make):
        try:
            row.fps[key] = round(timed_fps(make(), args.warmup, args.runs), 1)
            print(f"  {label:<20} {key:<4} {row.fps[key]:>7.1f} FPS")
        except Exception as e:  # one broken backend shouldn't kill the whole table
            row.notes.append(f"{key} failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    if backend == "torch":
        attempt("cpu", lambda: torch_runner(path, "cpu", x))
        if gpu:
            attempt("gpu", lambda: torch_runner(path, gpu, x))
    elif backend == "onnx":
        attempt("cpu", lambda: onnx_runner(path, x))
    elif backend == "coreml":
        for unit in ("cpu", "gpu", "ane"):
            attempt(unit, lambda unit=unit: coreml_runner(path, unit, canvas_rgb))

    if not args.skip_map:
        try:
            dev = (gpu or "cpu") if backend.startswith("torch") else "cpu"
            r = evaluate(str(path), args.data, "test", args.imgsz, dev)
            row.mAP50, row.mAP50_95 = round(r["mAP50"], 4), round(r["mAP50_95"], 4)
        except Exception as e:
            row.notes.append(f"mAP failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    return row


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def write_markdown(rows: list[Row], hw: dict, args, out: Path) -> None:
    base = next((r for r in rows if r.file == "best_fp32.pt" and r.precision == "FP32"), None)
    lines = [
        "# Edge optimization results",
        "",
        f"Measured on **{hw.get('chip')}**, {hw.get('ram_gb', '?')} GB RAM, {hw.get('os')}, "
        f"torch {hw.get('torch')} ({hw.get('torch_threads')} CPU threads). GPU = {hw.get('gpu', 'none')}.",
        f"Batch 1, {args.imgsz}x{args.imgsz} input, {args.warmup} warm-up runs then the average over {args.runs} runs. "
        "FPS is model inference only. mAP is on the held-out test split for every row.",
        "",
        "| Variant | Precision | Size (MB) | CPU FPS | GPU FPS | Neural Engine FPS | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.variant} | {r.precision} | {r.size_mb:.1f} | {fmt(r.fps.get('cpu'))} | {fmt(r.fps.get('gpu'))} | "
            f"{fmt(r.fps.get('ane'))} | {fmt(r.mAP50, 3)} | {fmt(r.mAP50_95, 3)} |"
        )
    if base:
        lines += ["", "Change vs. the FP32 PyTorch baseline:", "",
                  "| Variant | Size | Best CPU FPS | Best GPU-side FPS | mAP@50 |", "|---|---|---|---|---|"]
        b_cpu = base.fps.get("cpu")
        b_gpu = base.fps.get("gpu")
        for r in rows:
            if r is base:
                continue
            gpu_side = max([v for k, v in r.fps.items() if k in ("gpu", "ane")], default=None)
            cpu = r.fps.get("cpu")
            parts = [
                f"{r.size_mb / base.size_mb:.2f}x",
                f"{cpu / b_cpu:.2f}x" if cpu and b_cpu else "–",
                f"{gpu_side / b_gpu:.2f}x" if gpu_side and b_gpu else "–",
                f"{(r.mAP50 - base.mAP50):+.3f}" if r.mAP50 is not None and base.mAP50 is not None else "–",
            ]
            lines.append(f"| {r.variant} | " + " | ".join(parts) + " |")
    notes = [(r.variant, n) for r in rows for n in r.notes]
    if notes:
        lines += ["", "Notes from the run:", ""] + [f"- {v}: {n}" for v, n in notes]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--image", default=None, help="image used as the timing input (any test image works)")
    args = ap.parse_args()

    img_path = Path(args.image) if args.image else sorted(repo_path("datasets/nav3/images/test").iterdir())[0]
    img = cv2.imread(str(img_path))
    x = letterbox(img, args.imgsz)
    canvas_rgb = (x[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)

    hw = hardware()
    print(json.dumps(hw, indent=2))
    rows = [r for v in VARIANTS if (r := bench_variant(*v, x, canvas_rgb, args))]

    out_dir = repo_path("optimization")
    (out_dir / "results.json").write_text(json.dumps({"hardware": hw, "rows": [asdict(r) for r in rows]}, indent=2))
    write_markdown(rows, hw, args, out_dir / "results.md")
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
