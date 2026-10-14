"""
Sanity checks for the distance math. Run with:  python -m pytest -q

The main one projects a fake object through a pinhole camera at a known
distance and checks we get that distance back from the box height.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.distance import CameraModel, DistanceEstimator, focal_from_hfov, focal_from_specs  # noqa: E402

CFG = {
    "camera": {"hfov_deg": 65.0, "fx": None, "fy": None},
    "object_heights_m": {"cone": 0.70, "barrier": 1.0, "stop_sign": 0.75},
    "limits": {"min_box_px": 12, "edge_margin_px": 2},
}


def test_hand_computed_example():
    # the example from the plan: f = 800 px, 0.7 m cone, 373 px box -> about 1.5 m
    est = DistanceEstimator(CFG)
    assert est.distance_from_height("cone", 373, fy=800) == pytest.approx(1.501, abs=1e-3)


@pytest.mark.parametrize("true_z", [1.5, 2.0, 3.0, 7.5, 15.0])
def test_round_trip_through_pinhole_projection(true_z):
    w, h = 1280, 720
    est = DistanceEstimator(CFG)
    cam = est.camera_for(w, h)
    real_h = CFG["object_heights_m"]["stop_sign"]
    box_h = cam.fy * real_h / true_z  # project: h = f * H / Z
    cy = h / 2
    box = (600, cy - box_h / 2, 680, cy + box_h / 2)
    if box_h >= h:  # too close to fit in frame, not a useful case
        pytest.skip("object taller than the frame at this distance")
    got = est.estimate("stop_sign", box, (w, h))
    assert got.meters == pytest.approx(true_z, rel=1e-6)
    assert not got.truncated


def test_twice_as_far_is_half_as_tall():
    est = DistanceEstimator(CFG)
    z1 = est.distance_from_height("cone", 200, fy=1000)
    z2 = est.distance_from_height("cone", 100, fy=1000)
    assert z2 == pytest.approx(2 * z1)


def test_focal_formulas_agree():
    # 65 deg HFOV on a 1920 px wide image
    f = focal_from_hfov(65, 1920)
    assert f == pytest.approx(960 / math.tan(math.radians(32.5)))
    # 4.25 mm lens on a 5.6 mm wide sensor at 4032 px -> about 3060 px
    assert focal_from_specs(4.25, 5.6, 4032) == pytest.approx(3060, rel=1e-3)


def test_focal_scales_with_resolution():
    cam = CameraModel(fx=1000, fy=1000, ref_width=1920, ref_height=1080)
    half = cam.scaled_to(960, 540)
    assert half.fx == pytest.approx(500)
    assert half.fy == pytest.approx(500)


def test_calibrated_values_win_over_hfov():
    cfg = {**CFG, "camera": {"hfov_deg": 65, "fx": 900, "fy": 910, "ref_width": 1280, "ref_height": 720}}
    est = DistanceEstimator(cfg)
    assert est.camera_for(640, 360).fy == pytest.approx(455)


def test_edge_and_tiny_boxes_are_flagged():
    est = DistanceEstimator(CFG)
    cut = est.estimate("cone", (10, 0, 60, 120), (640, 480))  # touches the top edge
    assert cut.truncated and cut.label().startswith("<")
    tiny = est.estimate("cone", (10, 100, 15, 108), (640, 480))  # 8 px tall
    assert tiny.meters is None and tiny.label() == "?m"


def test_label_format():
    est = DistanceEstimator(CFG)
    cam = est.camera_for(640, 480)
    box_h = cam.fy * 0.70 / 1.5
    d = est.estimate("cone", (100, 200, 140, 200 + box_h), (640, 480))
    assert d.label() == "1.5m"
