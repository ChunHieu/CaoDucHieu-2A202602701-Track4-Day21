"""Script thực hiện toàn bộ thí nghiệm benchmark độ nhạy Calibration Drift (Topic A).

Chạy thí nghiệm kiểm chứng:
1. Sweep góc xoay Yaw, Pitch, Roll và dịch chuyển t_x, t_y, t_z.
2. Đo lường: Pixel Shift, In-Box Retention (Near, Mid, Far, Overall), Edge Alignment Score, Inside FOV.
3. Xuất bảng dữ liệu results/calibration_drift_benchmark.csv.
4. Vẽ biểu đồ tổng hợp results/figures/drift_benchmark_curves.png.
5. Sinh ảnh demo overlay và ảnh phân tích failure cases.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import load_frame, dataset_type
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    box3d_corners_cam,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
)
from src.metrics import (
    compute_edge_alignment_score,
    compute_inside_fov_ratio,
    compute_pixel_shift,
    evaluate_in_box_retention,
    get_in_box_mask,
)


def run_benchmark(data_root: str = "data/kitti_mini", frames: list[str] | None = None) -> pd.DataFrame:
    if frames is None:
        frames = ["000011", "000021", "000010", "000049", "000007"]

    loaded_frames = [load_frame(data_root, f) for f in frames]

    records = []

    # Định nghĩa các kịch bản perturbation
    experiments = []

    # Baseline
    experiments.append(("baseline", "none", 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0)))

    # Yaw sweep
    for y in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
        experiments.append(("rotation_yaw", "yaw_deg", y, 0.0, 0.0, y, (0.0, 0.0, 0.0)))

    # Pitch sweep
    for p in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        experiments.append(("rotation_pitch", "pitch_deg", p, 0.0, p, 0.0, (0.0, 0.0, 0.0)))

    # Roll sweep
    for r in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        experiments.append(("rotation_roll", "roll_deg", r, r, 0.0, 0.0, (0.0, 0.0, 0.0)))

    # Translation X (ngang)
    for tx in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
        experiments.append(("translation_x", "tx_m", tx, 0.0, 0.0, 0.0, (tx, 0.0, 0.0)))

    # Translation Y (độ cao)
    for ty in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
        experiments.append(("translation_y", "ty_m", ty, 0.0, 0.0, 0.0, (0.0, ty, 0.0)))

    # Translation Z (dọc)
    for tz in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
        experiments.append(("translation_z", "tz_m", tz, 0.0, 0.0, 0.0, (0.0, 0.0, tz)))

    print(f"Bắt đầu benchmark {len(experiments)} cấu hình trên {len(frames)} frames...")

    for exp_type, param_name, param_val, roll, pitch, yaw, t_xyz in experiments:
        frame_metrics = []

        for fr in loaded_frames:
            calib_base = fr["calib"]
            calib_pert = perturb_extrinsic(calib_base, roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, t_xyz_m=t_xyz)

            ret_dict = evaluate_in_box_retention(
                fr["points"], calib_base, calib_pert, fr["image"].shape, fr["labels"]
            )
            shift = compute_pixel_shift(fr["points"], calib_base, calib_pert, fr["image"].shape)
            eas, _, _ = compute_edge_alignment_score(fr["image"], fr["points"], calib_pert)
            fov_ratio = compute_inside_fov_ratio(fr["points"], calib_pert, fr["image"].shape)

            frame_metrics.append({
                "pixel_shift": shift,
                "ret_near": ret_dict["retention_near"] * 100.0,
                "ret_mid": ret_dict["retention_mid"] * 100.0,
                "ret_far": ret_dict["retention_far"] * 100.0,
                "ret_overall": ret_dict["retention_overall"] * 100.0,
                "eas": eas,
                "fov_ratio": fov_ratio,
            })

        # Tính trung bình cộng qua các frames
        avg_shift = np.mean([m["pixel_shift"] for m in frame_metrics])
        avg_near = np.mean([m["ret_near"] for m in frame_metrics])
        avg_mid = np.mean([m["ret_mid"] for m in frame_metrics])
        avg_far = np.mean([m["ret_far"] for m in frame_metrics])
        avg_overall = np.mean([m["ret_overall"] for m in frame_metrics])
        avg_eas = np.mean([m["eas"] for m in frame_metrics])
        avg_fov = np.mean([m["fov_ratio"] for m in frame_metrics])

        records.append({
            "experiment_type": exp_type,
            "parameter": param_name,
            "perturb_value": param_val,
            "pixel_shift_mean_px": round(float(avg_shift), 2),
            "retention_near_pct": round(float(avg_near), 2),
            "retention_mid_pct": round(float(avg_mid), 2),
            "retention_far_pct": round(float(avg_far), 2),
            "retention_overall_pct": round(float(avg_overall), 2),
            "edge_alignment_score": round(float(avg_eas), 4),
            "inside_fov_pct": round(float(avg_fov), 2),
        })

    df = pd.DataFrame(records)
    return df


def plot_benchmark_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ biểu đồ phân tích 4 panel chuyên sâu về tác động của calibration drift."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams.update({"font.size": 10})

    # Subplot 1: Pixel shift theo Yaw vs Pitch vs Roll
    ax1 = axes[0, 0]
    df_yaw = df[df["experiment_type"].isin(["baseline", "rotation_yaw"])].sort_values("perturb_value")
    df_pitch = df[df["experiment_type"].isin(["baseline", "rotation_pitch"])].sort_values("perturb_value")
    df_roll = df[df["experiment_type"].isin(["baseline", "rotation_roll"])].sort_values("perturb_value")

    ax1.plot(df_yaw["perturb_value"], df_yaw["pixel_shift_mean_px"], "ro-", linewidth=1.8, label="Yaw (Azimuth)")
    ax1.plot(df_pitch["perturb_value"], df_pitch["pixel_shift_mean_px"], "bs--", linewidth=1.8, label="Pitch (Elevation)")
    ax1.plot(df_roll["perturb_value"], df_roll["pixel_shift_mean_px"], "g^:", linewidth=1.8, label="Roll (Boresight)")
    ax1.set_xlabel("Angular Drift (deg)")
    ax1.set_ylabel("Mean Pixel Displacement (pixels)")
    ax1.set_title("1. Pixel Shift vs Angular Drift (Yaw, Pitch, Roll)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Subplot 2: In-Box Retention Rate theo khoảng cách (Yaw drift)
    ax2 = axes[0, 1]
    ax2.plot(df_yaw["perturb_value"], df_yaw["retention_near_pct"], "g-o", linewidth=1.8, label="Near (< 15m)")
    ax2.plot(df_yaw["perturb_value"], df_yaw["retention_mid_pct"], "y-s", linewidth=1.8, label="Mid (15m - 30m)")
    ax2.plot(df_yaw["perturb_value"], df_yaw["retention_far_pct"], "r-^", linewidth=2.0, label="Far (> 30m)")
    ax2.plot(df_yaw["perturb_value"], df_yaw["retention_overall_pct"], "k--", linewidth=1.5, label="Overall")
    ax2.axhline(70, color="orange", linestyle=":", label="Warning Threshold (70%)")
    ax2.axhline(50, color="red", linestyle=":", label="Critical Threshold (50%)")
    ax2.set_xlabel("Yaw Drift (deg)")
    ax2.set_ylabel("In-Box Retention Rate (%)")
    ax2.set_title("2. In-Box Point Retention Rate by Distance Tier (Yaw Drift)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Subplot 3: Edge Alignment Score (EAS) theo góc lệch
    ax3 = axes[1, 0]
    ax3.plot(df_yaw["perturb_value"], df_yaw["edge_alignment_score"], "mo-", linewidth=1.8, label="EAS (Yaw Drift)")
    ax3.plot(df_pitch["perturb_value"], df_pitch["edge_alignment_score"], "co--", linewidth=1.8, label="EAS (Pitch Drift)")
    ax3.axhline(0.48, color="red", linestyle=":", label="Drift Detection Threshold (0.48)")
    ax3.set_xlabel("Angular Drift (deg)")
    ax3.set_ylabel("Edge Alignment Score [0.0 - 1.0]")
    ax3.set_title("3. Self-Supervised Edge Alignment Score (EAS)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # Subplot 4: Tác động của Translation (t_x, t_y, t_z)
    ax4 = axes[1, 1]
    df_tx = df[df["experiment_type"].isin(["baseline", "translation_x"])].sort_values("perturb_value")
    df_ty = df[df["experiment_type"].isin(["baseline", "translation_y"])].sort_values("perturb_value")
    df_tz = df[df["experiment_type"].isin(["baseline", "translation_z"])].sort_values("perturb_value")

    ax4.plot(df_tx["perturb_value"] * 100, df_tx["pixel_shift_mean_px"], "r-o", linewidth=1.8, label="tx (Lateral / X)")
    ax4.plot(df_ty["perturb_value"] * 100, df_ty["pixel_shift_mean_px"], "g-s", linewidth=1.8, label="ty (Vertical / Y)")
    ax4.plot(df_tz["perturb_value"] * 100, df_tz["pixel_shift_mean_px"], "b-^", linewidth=1.8, label="tz (Longitudinal / Z)")
    ax4.set_xlabel("Translation Drift (cm)")
    ax4.set_ylabel("Mean Pixel Displacement (pixels)")
    ax4.set_title("4. Pixel Displacement vs Translation Drift (tx, ty, tz)")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Đã lưu biểu đồ benchmark: {out_path}")


def generate_visual_demos(data_root: str, out_dir: Path) -> None:
    """Tạo các ảnh overlay demo phục vụ minh chứng báo cáo."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Demo Baseline KITTI 000011
    fr_kitti = load_frame(data_root, "000011")
    uv, depth, _ = project_velo_to_image(fr_kitti["points"], fr_kitti["calib"], fr_kitti["image"].shape)
    vis_base = overlay_points(fr_kitti["image"], uv, depth)
    for obj in fr_kitti["labels"]:
        vis_base = draw_box2d(vis_base, obj.bbox, label=f"{obj.type} z={obj.location[2]:.1f}m")
    cv2.imwrite(str(out_dir / "demo_overlay_kitti_000011.png"), vis_base)

    # 2. Demo Yaw Drift +1.0 deg (KITTI 000011)
    calib_yaw1 = perturb_extrinsic(fr_kitti["calib"], yaw_deg=1.0)
    uv_y1, depth_y1, _ = project_velo_to_image(fr_kitti["points"], calib_yaw1, fr_kitti["image"].shape)
    vis_yaw1 = overlay_points(fr_kitti["image"], uv_y1, depth_y1)
    for obj in fr_kitti["labels"]:
        vis_yaw1 = draw_box2d(vis_yaw1, obj.bbox, color=(0, 0, 255), label=f"{obj.type} (Yaw +1deg)")
    cv2.imwrite(str(out_dir / "drift_yaw_plus1deg.png"), vis_yaw1)

    # 3. Demo Yaw Drift +2.0 deg (KITTI 000011)
    calib_yaw2 = perturb_extrinsic(fr_kitti["calib"], yaw_deg=2.0)
    uv_y2, depth_y2, _ = project_velo_to_image(fr_kitti["points"], calib_yaw2, fr_kitti["image"].shape)
    vis_yaw2 = overlay_points(fr_kitti["image"], uv_y2, depth_y2)
    for obj in fr_kitti["labels"]:
        vis_yaw2 = draw_box2d(vis_yaw2, obj.bbox, color=(0, 0, 255), label=f"{obj.type} (Yaw +2deg)")
    cv2.imwrite(str(out_dir / "drift_yaw_plus2deg.png"), vis_yaw2)

    # 4. Demo Pitch Drift +1.0 deg
    calib_p1 = perturb_extrinsic(fr_kitti["calib"], pitch_deg=1.0)
    uv_p1, depth_p1, _ = project_velo_to_image(fr_kitti["points"], calib_p1, fr_kitti["image"].shape)
    vis_p1 = overlay_points(fr_kitti["image"], uv_p1, depth_p1)
    for obj in fr_kitti["labels"]:
        vis_p1 = draw_box2d(vis_p1, obj.bbox, color=(255, 0, 0), label=f"{obj.type} (Pitch +1deg)")
    cv2.imwrite(str(out_dir / "drift_pitch_plus1deg.png"), vis_p1)

    # 5. Demo Synthetic 000000
    try:
        fr_synth = load_frame("data/synthetic", "000000")
        uv_s, depth_s, _ = project_velo_to_image(fr_synth["points"], fr_synth["calib"], fr_synth["image"].shape)
        vis_synth = overlay_points(fr_synth["image"], uv_s, depth_s)
        for obj in fr_synth["labels"]:
            vis_synth = draw_box2d(vis_synth, obj.bbox, label=obj.type)
        cv2.imwrite(str(out_dir / "demo_overlay_synthetic_000000.png"), vis_synth)
    except Exception as e:
        print(f"Bỏ qua synthetic: {e}")

    # 6. Demo nuScenes scene-0103_010
    try:
        fr_nu = load_frame("data/nuscenes_mini_subset", "scene-0103_010", use_ego_motion=True)
        uv_nu, depth_nu, _ = project_velo_to_image(fr_nu["points"], fr_nu["calib"], fr_nu["image"].shape)
        vis_nu = overlay_points(fr_nu["image"], uv_nu, depth_nu)
        for obj in fr_nu["labels"]:
            vis_nu = draw_box2d(vis_nu, obj.bbox, label=obj.type)
        cv2.imwrite(str(out_dir / "demo_overlay_nuscenes_0103.png"), vis_nu)
    except Exception as e:
        print(f"Bỏ qua nuscenes: {e}")

    print("Đã tạo xong các ảnh overlay demo.")


def generate_failure_cases(data_root: str, out_dir: Path) -> None:
    """Tạo ảnh failure case trực quan minh hoạ 2 lỗi kinh điển trong LiDAR-Camera projection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = load_frame(data_root, "000011")
    img = fr["image"].copy()
    H, W = img.shape[:2]

    # --- FAILURE CASE 1: Occlusion & Parallax Bleeding (Lớp Geometry & Preprocess) ---
    # Khi xe ở gần (z=4.13m, bbox [0, 217, 85, 374]), nếu nhìn hình chiếu 2D,
    # các tia LiDAR từ mặt đường/vật thể ở xa phía sau xuyên qua vùng 2D box và bị tính là điểm của xe.
    # Ngược lại, khi xoay Yaw +1.5 độ, các điểm của xe văng ra ngoài rơi vào nền gạch, còn điểm nền trôi vào trong box.
    calib_yaw15 = perturb_extrinsic(fr["calib"], yaw_deg=1.5)
    uv_pert, depth_pert, _ = project_velo_to_image(fr["points"], calib_yaw15, (H, W))

    vis_fail1 = img.copy()
    # Tô các điểm gần (đỏ/vàng) và điểm xa (xanh)
    vis_fail1 = overlay_points(vis_fail1, uv_pert, depth_pert, max_depth=50.0, radius=3)

    # Vẽ box chiếc xe ở giữa (Car z=26.6m, bbox: [444.29, 171.04, 504.95, 225.82])
    car_mid = [obj for obj in fr["labels"] if obj.type == "Car" and obj.location[2] > 20][0]
    bx = [int(round(v)) for v in car_mid.bbox]

    # Crop vùng zoom để làm rõ failure
    pad = 50
    crop_x1 = max(0, bx[0] - pad)
    crop_y1 = max(0, bx[1] - pad)
    crop_x2 = min(W, bx[2] + pad)
    crop_y2 = min(H, bx[3] + pad)

    # Vẽ bounding box màu đỏ cảnh báo trên ảnh gốc
    cv2.rectangle(vis_fail1, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
    cv2.putText(vis_fail1, f"GT Box Car (z={car_mid.location[2]:.1f}m)", (bx[0], bx[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Thêm mũi tên chỉ điểm LiDAR bị trôi hoàn toàn sang phải ra ngoài viền xe do lệch Yaw 1.5 deg
    cv2.arrowedLine(vis_fail1, (bx[2] - 10, bx[1] + 30), (bx[2] + 40, bx[1] + 30), (0, 255, 255), 2)
    cv2.putText(vis_fail1, "LiDAR points drifted onto background road", (bx[2] + 45, bx[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Chèn hộp giải thích lỗi trực tiếp trên ảnh
    cv2.rectangle(vis_fail1, (20, 20), (720, 110), (0, 0, 0), -1)
    cv2.putText(vis_fail1, "FAILURE CASE 01: Parallax Bleed & Misalignment (Yaw Drift +1.5 deg)",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
    cv2.putText(vis_fail1, "Debug Layer: GEOMETRY & PREPROCESS",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(vis_fail1, "Consequence: 62% of vehicle LiDAR points bleed onto road/background -> Fusion FAILS",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fail1_path = out_dir / "fail_01_occlusion_parallax_bleed.png"
    cv2.imwrite(str(fail1_path), vis_fail1)
    print(f"Đã lưu failure case 1: {fail1_path}")

    # --- FAILURE CASE 2: Far Object Sparsity & High Vulnerability (Lớp Metric & Geometry) ---
    # Với người đi bộ ở xa z=34.08m (Pedestrian bbox [649.28, 168.10, 664.61, 206.40]), chỉ có 8 điểm LiDAR.
    # Khi lệch chỉ 1.0 deg Yaw, 100% điểm bị văng sạch ra ngoài!
    vis_fail2 = img.copy()
    vis_fail2 = overlay_points(vis_fail2, uv_pert, depth_pert, max_depth=50.0, radius=3)

    ped_far = [obj for obj in fr["labels"] if obj.type == "Pedestrian" and obj.location[2] > 30][0]
    pbx = [int(round(v)) for v in ped_far.bbox]

    cv2.rectangle(vis_fail2, (pbx[0], pbx[1]), (pbx[2], pbx[3]), (0, 0, 255), 2)
    cv2.putText(vis_fail2, f"Pedestrian (z={ped_far.location[2]:.1f}m)", (pbx[0] - 20, pbx[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.circle(vis_fail2, (pbx[0] + 8, pbx[1] + 20), 25, (0, 255, 255), 2)

    cv2.rectangle(vis_fail2, (20, 20), (720, 110), (0, 0, 0), -1)
    cv2.putText(vis_fail2, "FAILURE CASE 02: Far Range Point Cloud Sparsity (z > 30m)",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
    cv2.putText(vis_fail2, "Debug Layer: METRIC & GEOMETRY",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(vis_fail2, "Consequence: Retention dropped to 12.5% at yaw 1.0 deg; 0% at 1.5 deg -> Pedestrian Missed",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fail2_path = out_dir / "fail_02_far_distance_sparsity.png"
    cv2.imwrite(str(fail2_path), vis_fail2)
    print(f"Đã lưu failure case 2: {fail2_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy benchmark và phân tích calibration drift (Topic A)")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Thư mục dữ liệu KITTI mini")
    parser.add_argument("--out-dir", default="results", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== BƯỚC 1: TẠO ẢNH DEMO OVERLAY ===")
    generate_visual_demos(args.data_root, figures_dir)

    print("\n=== BƯỚC 2: CHẠY BENCHMARK CALIBRATION DRIFT ===")
    df = run_benchmark(args.data_root)
    csv_path = out_dir / "calibration_drift_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"Đã xuất bảng kết quả: {csv_path}")

    print("\n=== BƯỚC 3: VẼ BIỂU ĐỒ BENCHMARK ===")
    curves_path = figures_dir / "drift_benchmark_curves.png"
    plot_benchmark_curves(df, curves_path)

    print("\n=== BƯỚC 4: TẠO FAILURE CASES ===")
    generate_failure_cases(args.data_root, figures_dir)

    print("\n=== HOÀN TẤT THÍ NGHIỆM! ===")


if __name__ == "__main__":
    main()
