"""
A sanity check on the distance numbers that doesn't need ground truth.

In the I-84 clip the cones sit still on the road and the car filming them drives
forward at a pretty steady speed. So for any one cone, the true distance should
drop in a straight line over time, and the slope of that line is the car's speed.

If the geometry (focal length + assumed cone height) is roughly right, then:
  * each cone's estimated distance vs. time should be close to a straight line
  * every cone should give about the same slope (same car, same speed)
  * that speed should be believable for a crew laying cones (walking-ish pace)

This reads the per-frame detections written by run_demo.py, links cones across
frames with a tiny nearest-neighbour tracker, fits a line per cone, and plots it.

    python scripts/distance_vs_time.py --csv outputs/demo/detections.csv --fps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdet.paths import repo_path  # noqa: E402

# reference categorical palette, used in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def load_cones(path: Path, cls: str, min_conf: float):
    """frame index -> list of (cx, cy, box_h, distance)"""
    per_frame = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if "#" not in r["source"] or r["class"] != cls or not r["distance_m"]:
                continue
            if int(r["cut_off"]) or float(r["conf"]) < min_conf:
                continue  # cut-off boxes are only upper bounds, leave them out of the fit
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            frame = int(r["source"].split("#")[1])
            per_frame[frame].append(((x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, float(r["distance_m"])))
    return per_frame


def link(per_frame, max_gap: int = 5):
    """Greedy nearest-centre linking. Good enough for a handful of slow-moving cones."""
    tracks, active = [], []
    for frame in sorted(per_frame):
        dets = per_frame[frame]
        used = set()
        for t in active:
            last_f, (cx, cy, h, _) = t[-1]
            if frame - last_f > max_gap:
                continue
            best, best_d = None, None
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = np.hypot(d[0] - cx, d[1] - cy)
                if dist < 0.8 * max(h, d[2]) and (best_d is None or dist < best_d):
                    best, best_d = i, dist
            if best is not None:
                t.append((frame, dets[best]))
                used.add(best)
        for i, d in enumerate(dets):
            if i not in used:
                t = [(frame, d)]
                tracks.append(t)
                active.append(t)
        active = [t for t in active if frame - t[-1][0] <= max_gap]
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/demo/detections.csv")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cls", default="cone")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--out", default="outputs/extra/distance_vs_time.png")
    args = ap.parse_args()

    tracks = [t for t in link(load_cones(repo_path(args.csv), args.cls, args.min_conf)) if len(t) >= args.min_points]
    tracks.sort(key=lambda t: t[0][0])
    fits = []
    for k, t in enumerate(tracks):
        ts = np.array([f for f, _ in t]) / args.fps
        zs = np.array([d[3] for _, d in t])
        slope, icpt = np.polyfit(ts, zs, 1)
        pred = slope * ts + icpt
        r2 = 1 - ((zs - pred) ** 2).sum() / max(((zs - zs.mean()) ** 2).sum(), 1e-9)
        fits.append({"track": k + 1, "points": len(t), "t_start_s": round(ts[0], 2), "t_end_s": round(ts[-1], 2),
                     "z_start_m": round(zs[0], 2), "z_end_m": round(zs[-1], 2),
                     "speed_mps": round(-slope, 2), "r2": round(r2, 3),
                     "ts": ts.tolist(), "zs": zs.tolist()})

    for f in fits:
        print(f"cone {f['track']}: {f['points']} frames, {f['z_start_m']} -> {f['z_end_m']} m, "
              f"speed {f['speed_mps']} m/s ({f['speed_mps'] * 2.237:.1f} mph), R^2 {f['r2']}")
    speeds = [f["speed_mps"] for f in fits if f["r2"] > 0.8]
    if speeds:
        print(f"median speed over {len(speeds)} clean tracks: {np.median(speeds):.2f} m/s ({np.median(speeds) * 2.237:.1f} mph)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = sorted(fits, key=lambda f: -f["points"])[: len(SERIES)]
    shown.sort(key=lambda f: f["t_start_s"])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for color, f in zip(SERIES, shown):
        ts, zs = np.array(f["ts"]), np.array(f["zs"])
        ax.scatter(ts, zs, s=10, color=color, edgecolors="none", alpha=0.75)
        line_t = np.array([ts[0], ts[-1]])
        ax.plot(line_t, -f["speed_mps"] * line_t + (zs.mean() + f["speed_mps"] * ts.mean()), color=color, lw=2,
                label=f"cone {f['track']}: {f['speed_mps']:.1f} m/s, R² {f['r2']:.2f}")
    ax.set_xlabel("time in clip (s)", color=INK_2)
    ax.set_ylabel("estimated distance (m)", color=INK_2)
    ax.set_title("Each cone's estimated distance vs. time (dots) with a straight-line fit", color=INK, fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=INK_2)
    ax.set_ylim(bottom=0)
    if shown:
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
        for txt in leg.get_texts():
            txt.set_color(INK)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    out.with_suffix(".json").write_text(json.dumps([{k: v for k, v in f.items() if k not in ("ts", "zs")} for f in fits], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
