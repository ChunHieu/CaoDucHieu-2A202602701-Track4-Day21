"""Module tính toán các metric đánh giá chất lượng LiDAR-Camera Calibration và Projection.

Bao gồm:
1. In-Box Point Retention Rate (phân nhóm theo cự ly Near <15m, Mid 15-30m, Far >30m).
2. Edge Alignment Score (EAS) tự giám sát dựa trên Canny edge và LiDAR depth jump.
3. Mean Pixel Displacement (độ dịch chuyển pixel trung bình do calibration drift).
4. Inside FOV Ratio (tỉ lệ điểm nằm trong tầm nhìn camera).
"""
from __future__ import annotations

import cv2
import numpy as np
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import project_velo_to_image


def get_in_box_mask(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Kiểm tra điểm uv (M, 2) có nằm trong bbox 2D [x1, y1, x2, y2] hay không."""
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def evaluate_in_box_retention(
    points_velo: np.ndarray,
    calib_base: KittiCalib,
    calib_perturbed: KittiCalib,
    image_shape: tuple[int, ...],
    labels: list[KittiObject],
) -> dict[str, float]:
    """Tính toán tỉ lệ điểm LiDAR của vật thể còn giữ được trong 2D box khi bị calibration drift.

    Phân rã theo cự ly:
    - near (< 15m)
    - mid (15m - 30m)
    - far (> 30m)
    - overall
    """
    uv_base, depth_base, mask_base = project_velo_to_image(points_velo, calib_base, image_shape)
    uv_pert, depth_pert, mask_pert = project_velo_to_image(points_velo, calib_perturbed, image_shape)

    base_indices = np.where(mask_base)[0]
    pert_indices = np.where(mask_pert)[0]

    pert_idx_to_pos = {int(idx): i for i, idx in enumerate(pert_indices)}

    categories: dict[str, list[tuple[float, int]]] = {
        "near": [],
        "mid": [],
        "far": [],
        "overall": [],
    }

    for obj in labels:
        if obj.type in ["DontCare"]:
            continue

        obj_depth = float(obj.location[2])
        cat = "near" if obj_depth < 15.0 else ("mid" if obj_depth <= 30.0 else "far")

        in_box_base_mask = get_in_box_mask(uv_base, obj.bbox)
        n_base = int(np.sum(in_box_base_mask))
        if n_base == 0:
            continue

        obj_point_indices = base_indices[in_box_base_mask]

        retained = 0
        for orig_idx in obj_point_indices:
            orig_idx_int = int(orig_idx)
            if orig_idx_int in pert_idx_to_pos:
                pos = pert_idx_to_pos[orig_idx_int]
                u_p, v_p = uv_pert[pos]
                if obj.bbox[0] <= u_p <= obj.bbox[2] and obj.bbox[1] <= v_p <= obj.bbox[3]:
                    retained += 1

        ratio = retained / n_base
        categories[cat].append((ratio, n_base))
        categories["overall"].append((ratio, n_base))

    def weighted_avg(items: list[tuple[float, int]]) -> float:
        if not items:
            return 1.0
        tot_pts = sum(cnt for _, cnt in items)
        if tot_pts == 0:
            return 1.0
        return sum(r * cnt for r, cnt in items) / tot_pts

    return {
        "retention_near": float(weighted_avg(categories["near"])),
        "retention_mid": float(weighted_avg(categories["mid"])),
        "retention_far": float(weighted_avg(categories["far"])),
        "retention_overall": float(weighted_avg(categories["overall"])),
    }


def find_depth_edges_grid(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple[int, ...],
    depth_jump_thresh: float = 2.0,
    downsample_factor: int = 4,
) -> np.ndarray:
    """Tìm các điểm LiDAR nằm trên ranh giới vật thể (depth discontinuity).
    
    Sử dụng grid 2D với morphological dilation để xác định foreground silhouette.
    """
    H, W = image_shape[:2]
    gh = H // downsample_factor + 1
    gw = W // downsample_factor + 1

    grid_max = np.zeros((gh, gw), dtype=np.float32)
    u_grid = np.clip((uv[:, 0] / downsample_factor).astype(int), 0, gw - 1)
    v_grid = np.clip((uv[:, 1] / downsample_factor).astype(int), 0, gh - 1)

    np.maximum.at(grid_max, (v_grid, u_grid), depth)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_max = cv2.dilate(grid_max, kernel)

    depth_jump = dilated_max[v_grid, u_grid] - depth
    return depth_jump >= depth_jump_thresh


def compute_edge_alignment_score(
    image: np.ndarray,
    points_velo: np.ndarray,
    calib: KittiCalib,
    sigma: float = 2.5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Tính Edge Alignment Score (EAS) giữa depth edges của LiDAR và Canny edges của ảnh.

    Trả về:
    - score: float trong [0.0, 1.0]
    - uv: toạ độ pixel (M, 2)
    - is_edge: mask bool (M,) các điểm depth edge
    """
    uv, depth, mask = project_velo_to_image(points_velo, calib, image.shape)
    if len(uv) == 0:
        return 0.0, np.zeros((0, 2)), np.zeros(0, dtype=bool)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    dist_map = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    is_edge = find_depth_edges_grid(uv, depth, image.shape)
    if not np.any(is_edge):
        is_edge = depth < 40.0

    edge_uv = uv[is_edge].astype(int)
    H, W = image.shape[:2]
    valid_u = np.clip(edge_uv[:, 0], 0, W - 1)
    valid_v = np.clip(edge_uv[:, 1], 0, H - 1)

    dists = dist_map[valid_v, valid_u]
    scores = np.exp(-(dists ** 2) / (2.0 * (sigma ** 2)))
    return float(np.mean(scores)), uv, is_edge


def compute_pixel_shift(
    points_velo: np.ndarray,
    calib_base: KittiCalib,
    calib_perturbed: KittiCalib,
    image_shape: tuple[int, ...],
) -> float:
    """Tính độ lệch pixel Euclidean trung bình giữa calibration chuẩn và calibration bị drift."""
    uv_base, _, mask_base = project_velo_to_image(points_velo, calib_base, image_shape)
    uv_pert, _, mask_pert = project_velo_to_image(points_velo, calib_perturbed, image_shape)

    common_mask = mask_base & mask_pert
    if not np.any(common_mask):
        return 0.0

    idx_base = np.where(mask_base)[0]
    map_base = {int(idx): i for i, idx in enumerate(idx_base)}

    idx_pert = np.where(mask_pert)[0]
    map_pert = {int(idx): i for i, idx in enumerate(idx_pert)}

    idx_common = np.where(common_mask)[0]
    shifts = []
    for idx in idx_common:
        i_b = map_base[int(idx)]
        i_p = map_pert[int(idx)]
        u0, v0 = uv_base[i_b]
        u1, v1 = uv_pert[i_p]
        shifts.append(np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2))

    return float(np.mean(shifts)) if shifts else 0.0


def compute_inside_fov_ratio(points_velo: np.ndarray, calib: KittiCalib, image_shape: tuple[int, ...]) -> float:
    """Tỉ lệ điểm nằm trong FOV camera (% so với tổng điểm point cloud)."""
    uv, depth, mask = project_velo_to_image(points_velo, calib, image_shape)
    return float(len(uv) / len(points_velo) * 100.0) if len(points_velo) > 0 else 0.0
