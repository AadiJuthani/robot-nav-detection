"""
Makes the smaller / faster versions of the trained model that benchmark.py compares.

  * CoreML FP16         half precision, runs on the Apple GPU / Neural Engine
  * ONNX INT8           static post-training quantization with ONNX Runtime
                        (weights AND activations in 8-bit, calibrated on val images).
                        Goes through a plain FP32 ONNX export first.
  * Pruned 30%          L1 unstructured pruning of the conv weights, no retraining

Everything lands in optimization/models/.

    python scripts/optimize.py --weights outputs/weights/best.pt
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402

MODELS = repo_path("optimization/models")


def letterbox(img_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """Same preprocessing Ultralytics uses: keep aspect ratio, pad with gray, RGB, 0-1, NCHW."""
    h, w = img_bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def export_onnx(weights: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    out = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
    return out


def export_coreml_fp16(weights: Path, imgsz: int, dst: Path) -> Path:
    from ultralytics import YOLO

    out = Path(YOLO(str(weights)).export(format="coreml", imgsz=imgsz, half=True))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(out), dst)
    return dst


def detect_head_float_nodes(onnx_path: Path) -> list[str]:
    """
    Nodes to leave in float: the last 1x1 conv of each head branch (raw box
    regression + class scores), the DFL conv, and the decode math after them
    (reshapes, softmax, sigmoid, the multiply by stride). These spit out numbers
    that get turned straight into pixel coordinates, and their value ranges are
    huge, so they quantize badly. Everything else goes to int8.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    idx = [int(m.group(1)) for n in model.graph.node if (m := re.search(r"/model\.(\d+)/", n.name))]
    head = f"/model.{max(idx)}/"  # the Detect layer is the last one
    final = re.compile(r"(cv[23]\.\d+/cv[23]\.\d+\.2|dfl)/")
    keep = []
    for n in model.graph.node:
        if head not in n.name and "/model." in n.name:
            continue  # backbone / neck: quantize all of it
        if re.search(r"/cv[23]\.\d+/", n.name):
            # inside the head's box/class branches: conv+SiLU blocks are fine in int8,
            # only the last conv of each branch stays float
            if n.op_type == "Conv" and final.search(n.name):
                keep.append(n.name)
            continue
        # what's left is the decode math after the branches (+ the DFL conv)
        keep.append(n.name)
    return keep


def quantize_onnx_int8(fp32_onnx: Path, calib_images: list[Path], imgsz: int, dst: Path) -> Path:
    """
    Static INT8 quantization (QDQ format) calibrated on real images.

    Everything ONNX Runtime can quantize goes to int8 (convs, the SiLU's
    sigmoid/mul, adds, concats, pooling), not just the convs. Quantizing only the
    convs leaves every activation in float, and the runtime then spends its time
    converting int8 <-> float around each layer.
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    input_name = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.paths = iter(calib_images)

        def get_next(self):
            for p in self.paths:
                img = cv2.imread(str(p))
                if img is not None:
                    return {input_name: letterbox(img, imgsz)}
            return None

    pre = dst.with_name(dst.stem + "_pre.onnx")
    try:
        quant_pre_process(str(fp32_onnx), str(pre))
        src = pre
    except Exception as e:  # pre-processing is a nice-to-have, not required
        print(f"quant_pre_process failed ({e}), quantizing the raw export instead")
        src = fp32_onnx

    skip = detect_head_float_nodes(src)
    print(f"keeping {len(skip)} detect-head nodes in float")
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=skip,
        calibrate_method=CalibrationMethod.MinMax,
    )
    pre.unlink(missing_ok=True)

    # quantization drops the Ultralytics metadata (class names, stride, imgsz),
    # copy it back so YOLO() can load the int8 file like any other export
    src_meta = {p.key: p.value for p in onnx.load(str(fp32_onnx)).metadata_props}
    q = onnx.load(str(dst))
    del q.metadata_props[:]
    for k, v in src_meta.items():
        q.metadata_props.add(key=k, value=v)
    onnx.save(q, str(dst))
    return dst


def prune_weights(weights: Path, amount: float, dst: Path) -> dict:
    """
    Zeroes the smallest `amount` of conv weights (by |w|, ranked across the whole
    network). The tensors stay dense, so expect the file to compress better but
    not to run faster on normal hardware.
    """
    import torch
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    convs = [(m, "weight") for m in yolo.model.modules() if isinstance(m, torch.nn.Conv2d)]
    prune.global_unstructured(convs, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in convs:
        prune.remove(m, name)  # bake the mask in
    total = sum(m.weight.numel() for m, _ in convs)
    zeros = sum(int((m.weight == 0).sum()) for m, _ in convs)
    yolo.save(str(dst))
    return {"conv_params": total, "zeroed": zeros, "sparsity": zeros / total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-dir", default="datasets/nav3/images/val")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--prune", type=float, default=0.3, help="share of conv weights to zero")
    ap.add_argument("--steps", nargs="+", default=["onnx", "int8", "coreml", "prune"],
                    help="any of: onnx int8 coreml prune")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    base = MODELS / "best_fp32.pt"
    shutil.copy2(repo_path(args.weights), base)
    failed = []

    def step(name, fn):
        # one broken exporter shouldn't take the other variants down with it
        print(f"== {name} ==")
        try:
            print(fn())
        except Exception as e:
            failed.append(name)
            print(f"{name} failed: {type(e).__name__}: {e}")

    onnx_fp32 = MODELS / "best_fp32.onnx"
    if "onnx" in args.steps:
        step("ONNX FP32", lambda: export_onnx(base, args.imgsz))
    if "int8" in args.steps:
        calib = sorted(repo_path(args.calib_dir).iterdir())
        random.Random(0).shuffle(calib)
        step("ONNX INT8 (static, ORT)",
             lambda: quantize_onnx_int8(onnx_fp32, calib[: args.calib_n], args.imgsz, MODELS / "best_int8.onnx"))
    if "coreml" in args.steps and sys.platform == "darwin":
        # coremltools lags behind new torch releases, see README (exported from a torch 2.7 env)
        step("CoreML FP16", lambda: export_coreml_fp16(base, args.imgsz, MODELS / "best_fp16.mlpackage"))
    if "prune" in args.steps:
        tag = int(round(args.prune * 100))
        step(f"pruned {tag}%", lambda: prune_weights(base, args.prune, MODELS / f"best_pruned{tag}.pt"))
    if failed:
        print("failed steps:", failed)


if __name__ == "__main__":
    main()
