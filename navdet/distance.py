"""
Distance from a bounding box, using the pinhole camera model.

It's really just similar triangles. An object that is H meters tall and Z meters
away shows up h pixels tall on the image:

    h_px / f_px = H_real / Z        so        Z = f_px * H_real / h_px

f_px is the focal length measured in pixels. If you know it and roughly how tall
the object is in real life, the box height gives you the distance. Twice as far
away means half as many pixels tall.

I use box *height* and not width on purpose: when a cone or sign is turned at an
angle its width shrinks but its height mostly doesn't.

Where this breaks (and the README says so too):
  * the real-world height is a guess per class (a 45 cm cone vs a 70 cm one)
  * a tilted camera or lens distortion bends the geometry a bit
  * a box cut off by the image edge is too short, so the object looks farther
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


def focal_from_hfov(hfov_deg: float, width_px: int) -> float:
    """Focal length in px from the horizontal field of view: f = (W/2) / tan(HFOV/2)."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def focal_from_specs(focal_mm: float, sensor_width_mm: float, width_px: int) -> float:
    """Focal length in px from lens specs: f_px = f_mm * W_px / sensor_width_mm."""
    return focal_mm * width_px / sensor_width_mm


@dataclass
class CameraModel:
    """Just the bits of the intrinsics we need: focal lengths at a reference image size."""

    fx: float
    fy: float
    ref_width: int
    ref_height: int

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> "CameraModel":
        # square pixels, so fy == fx
        f = focal_from_hfov(hfov_deg, width)
        return cls(fx=f, fy=f, ref_width=width, ref_height=height)

    def scaled_to(self, width: int, height: int) -> "CameraModel":
        """Same lens, different image size (e.g. a downscaled frame). f scales with resolution."""
        return CameraModel(
            fx=self.fx * width / self.ref_width,
            fy=self.fy * height / self.ref_height,
            ref_width=width,
            ref_height=height,
        )


@dataclass
class DistanceEstimate:
    """What came out for one box. meters is None when we didn't trust the box."""

    meters: float | None
    truncated: bool = False  # box touches the image edge, so meters is really an upper bound
    note: str = ""

    def label(self) -> str:
        """The distance part of the on-image label, e.g. '1.5m' or '<2.1m'."""
        if self.meters is None:
            return "?m"
        prefix = "<" if self.truncated else ""
        return f"{prefix}{self.meters:.1f}m"


class DistanceEstimator:
    """Turns (class, box, image size) into meters using configs/camera.yaml."""

    def __init__(self, config: dict):
        cam = config.get("camera", {})
        self.hfov_deg = cam.get("hfov_deg", 65.0)
        self.fx = cam.get("fx")
        self.fy = cam.get("fy")
        self.ref_width = cam.get("ref_width")
        self.ref_height = cam.get("ref_height")
        self.heights = dict(config.get("object_heights_m", {}))
        limits = config.get("limits", {})
        self.min_box_px = limits.get("min_box_px", 12)
        self.edge_margin_px = limits.get("edge_margin_px", 2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DistanceEstimator":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def camera_for(self, width: int, height: int) -> CameraModel:
        """Calibrated intrinsics if we have them (rescaled to this frame), else the HFOV guess."""
        if self.fx and self.fy and self.ref_width and self.ref_height:
            base = CameraModel(self.fx, self.fy, self.ref_width, self.ref_height)
            return base.scaled_to(width, height)
        return CameraModel.from_hfov(self.hfov_deg, width, height)

    def distance_from_height(self, class_name: str, box_height_px: float, fy: float) -> float:
        """The formula itself: Z = f * H / h."""
        real_h = self.heights[class_name]
        return fy * real_h / box_height_px

    def estimate(self, class_name: str, box_xyxy, image_size) -> DistanceEstimate:
        """
        box_xyxy is (x1, y1, x2, y2) in pixels of the original image,
        image_size is (width, height).
        """
        if class_name not in self.heights:
            return DistanceEstimate(None, note=f"no real-world height for {class_name}")

        width, height = image_size
        x1, y1, x2, y2 = box_xyxy
        h_px = float(y2 - y1)
        if h_px < self.min_box_px:
            # tiny boxes are a few pixels of noise away from a totally different answer
            return DistanceEstimate(None, note="box too small to trust")

        cam = self.camera_for(width, height)
        meters = self.distance_from_height(class_name, h_px, cam.fy)

        # cut off at the top or bottom -> the box is shorter than the object,
        # so the real thing is closer than this number
        m = self.edge_margin_px
        truncated = y1 <= m or y2 >= height - 1 - m
        return DistanceEstimate(meters, truncated=truncated, note="cut off by image edge" if truncated else "")
