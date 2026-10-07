"""Topic A: Lệch yaw bao nhiêu độ thì điểm LiDAR rơi ra khỏi 2D box của vật thể.

Mở rộng từ codelab: phân rã theo cự ly và class.
Chạy từ gốc repo:
    python -m src.exp_yaw_sweep --data-root data/kitti_mini --frames 000008 000011 000049
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from starter.datasets import load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, velo_to_cam

CLASSES = ("Car", "Van", "Pedestrian", "Cyclist")


def points_in_box(points_cam: np.ndarray, obj) -> np.ndarray:
    """Mask (N,) các điểm (đã ở camera frame) nằm trong 3D box của label."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ R        # toạ độ của điểm trong hệ trục gắn với box
    return ((np.abs(local[:, 0]) <= l / 2) & (local[:, 1] <= 0) & (local[:, 1] >= -h)
            & (np.abs(local[:, 2]) <= w / 2))


def run_one(fr: dict, yaw_deg: float) -> dict:
    pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
    cam_true = velo_to_cam(pts[:, :3], fr["calib"])           # vị trí thật, theo calib gốc
    calib = perturb_extrinsic(fr["calib"], yaw_deg=yaw_deg)   # calib đã bị lệch
    uv, _, mask = project_velo_to_image(pts, calib, fr["image"].shape)
    uv_all = np.full((len(pts), 2), np.nan)
    uv_all[mask] = uv

    obj_pts = hits = 0
    for obj in fr["labels"]:
        if obj.type not in CLASSES:
            continue
        sel = points_in_box(cam_true, obj) & mask
        u, v = uv_all[sel, 0], uv_all[sel, 1]
        x1, y1, x2, y2 = obj.bbox
        hits += int(((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)).sum())
        obj_pts += int(sel.sum())
    return {"n_points": len(pts), "inside_image": int(mask.sum()), "object_points": obj_pts,
            "hit_ratio": round(hits / obj_pts, 4) if obj_pts else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quét góc lệch yaw, đo % điểm của vật thể nằm trong 2D box")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=["000008", "000011", "000049"])
    ap.add_argument("--yaw-levels", nargs="+", type=float, default=[0, 0.5, 1, 2, 3])
    ap.add_argument("--out", default="results/yaw_perturb_sweep.csv")
    args = ap.parse_args()

    rows = []
    for frame in args.frames:
        fr = load_frame(args.data_root, frame)
        for yaw in args.yaw_levels:
            row = {"dataset": Path(args.data_root).name, "frame": frame, "yaw_deg": yaw, **run_one(fr, yaw)}
            rows.append(row)
            print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out} ({len(rows)} dòng)")


if __name__ == "__main__":
    main()
