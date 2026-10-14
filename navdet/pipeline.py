"""
Glues the pieces together: image in -> detections -> distances -> annotated image out.

The detector is anything Ultralytics can load (best.pt, an .onnx export, a CoreML
.mlpackage...), so the same pipeline runs on the optimized models too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotate import annotate
from .distance import DistanceEstimate, DistanceEstimator


@dataclass
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple  # (x1, y1, x2, y2) in original image pixels
    distance: DistanceEstimate = field(default_factory=lambda: DistanceEstimate(None))


def pick_device() -> str:
    """cuda if there's an NVIDIA GPU, mps on Apple silicon, otherwise cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NavPipeline:
    def __init__(self, weights: str, camera_cfg: str, conf: float = 0.35, imgsz: int = 640, device: str | None = None,
                 class_conf: dict | None = None):
        from ultralytics import YOLO  # imported here so distance-only code doesn't need torch

        self.model = YOLO(weights, task="detect")
        self.estimator = DistanceEstimator.from_yaml(camera_cfg)
        self.conf = conf
        # per-class overrides, e.g. {"stop_sign": 0.6} when red tail lights keep fooling it
        self.class_conf = dict(class_conf or {})
        self.imgsz = imgsz
        self.device = device or pick_device()

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        floor = min([self.conf, *self.class_conf.values()])
        result = self.model.predict(image_bgr, conf=floor, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        names = result.names
        h, w = image_bgr.shape[:2]
        dets = []
        for xyxy, cls_id, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy().astype(int),
                                      result.boxes.conf.cpu().numpy()):
            name = names[int(cls_id)]
            if conf < self.class_conf.get(name, self.conf):
                continue
            box = tuple(float(v) for v in xyxy)
            dist = self.estimator.estimate(name, box, (w, h))
            dets.append(Detection(name, float(conf), box, dist))
        # closest first, since that's what a robot cares about
        dets.sort(key=lambda d: d.distance.meters if d.distance.meters is not None else float("inf"))
        return dets

    def __call__(self, image_bgr: np.ndarray):
        dets = self.detect(image_bgr)
        return annotate(image_bgr, dets), dets
