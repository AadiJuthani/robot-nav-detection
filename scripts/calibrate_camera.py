"""
Checkerboard calibration with OpenCV, then writes fx / fy into configs/camera.yaml.

Print a checkerboard (e.g. 9x6 inner corners), take ~15 photos of it from
different angles with the robot's camera, then:

    python scripts/calibrate_camera.py --images calib_photos/ --cols 9 --rows 6 --square-mm 25

After this, distance estimates use the real focal length instead of the
assumed 65 degree field of view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
    ap.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--camera", default="configs/camera.yaml")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0

    obj_pts, img_pts, size = [], [], None
    for p in sorted(Path(args.images).iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            print(f"no board found in {p.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(objp)
        img_pts.append(corners)

    if len(obj_pts) < 5:
        sys.exit(f"only {len(obj_pts)} usable photos, need at least 5")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"RMS reprojection error {rms:.3f} px from {len(obj_pts)} photos")
    print(f"fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}  at {size[0]}x{size[1]}")

    cam_path = repo_path(args.camera)
    cfg = yaml.safe_load(cam_path.read_text())
    cfg["camera"].update(source="calibrated", fx=float(K[0, 0]), fy=float(K[1, 1]),
                         ref_width=int(size[0]), ref_height=int(size[1]))
    cam_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"updated {cam_path}")


if __name__ == "__main__":
    main()
