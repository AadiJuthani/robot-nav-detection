"""Drawing boxes with labels like 'Cone, 1.5m' onto an image."""

from __future__ import annotations

import cv2
import numpy as np

from .classes import CLASS_COLORS, display_name


def make_label(class_name: str, distance_label: str) -> str:
    """'cone' + '1.5m' -> 'Cone, 1.5m' (the exact format the assignment asks for)."""
    return f"{display_name(class_name)}, {distance_label}"


def draw_box(image: np.ndarray, xyxy, text: str, color=(0, 255, 0)) -> None:
    """Draws one box and a filled label tag above it (or inside it if there's no room up top)."""
    h, w = image.shape[:2]
    # scale line width and font with the image so labels stay readable on big and small frames
    thickness = max(2, round(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 900)
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 4
    tag_h = th + baseline + 2 * pad
    top = y1 - tag_h if y1 - tag_h >= 0 else y1  # flip inside the box near the top edge
    left = min(max(x1, 0), max(w - tw - 2 * pad, 0))
    cv2.rectangle(image, (left, top), (left + tw + 2 * pad, top + tag_h), color, -1)
    # dark text on light colors, white text otherwise
    text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)
    cv2.putText(image, text, (left + pad, top + pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, thickness, cv2.LINE_AA)


def annotate(image: np.ndarray, detections) -> np.ndarray:
    """
    Returns a copy of the image with every detection drawn on it.
    Each detection needs .class_name, .xyxy and .distance (a DistanceEstimate).
    """
    out = image.copy()
    for det in detections:
        text = make_label(det.class_name, det.distance.label())
        draw_box(out, det.xyxy, text, CLASS_COLORS.get(det.class_name, (0, 255, 0)))
    return out
