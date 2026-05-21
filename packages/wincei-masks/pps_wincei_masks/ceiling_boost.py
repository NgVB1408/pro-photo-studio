"""Ceiling boost — dùng lamp/light/downlight làm anchor để loang mực mask trần.

Cho ảnh modern không có phào trần, SegFormer ADE20K thường UNDER-detect ceiling
(coverage < 2% khi thực tế trần chiếm 15-25%). Lý do:
    - ADE20K phân lớp ceiling không bao phủ vùng có lamp/light
    - Góc chụp ngang/dưới-lên → ceiling visible nhưng confidence thấp

FIX KIẾN TRÚC HIỆN ĐẠI:
    1. Detect lamp (ADE id 36) + light fixture (id 82) trong TOP 60% of image
    2. Dùng vị trí lamp làm SEED POINTS
    3. Region growing qua color similarity từ seed → expand ceiling mask
    4. Union với ceiling mask gốc
    5. Crop về vùng trên top edge của wall mask (an toàn)

Hiệu ứng tương đương VLM-SAM2 multi-point nhưng không cần Ollama.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _detect_seed_points(
    lamp_mask: np.ndarray,
    light_mask: np.ndarray,
    *,
    top_fraction: float = 0.6,
    min_blob_area_pct: float = 0.0005,
) -> list[tuple[int, int]]:
    """Trích centroid của mỗi lamp/light blob (ở top fraction).

    Returns:
        List[(x, y)] seed points.
    """
    h, w = lamp_mask.shape
    combined = ((lamp_mask > 0) | (light_mask > 0)).astype(np.uint8)
    # Restrict to top portion
    combined[int(h * top_fraction):, :] = 0
    if combined.sum() < 10:
        return []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    min_area = h * w * min_blob_area_pct
    seeds: list[tuple[int, int]] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cx, cy = centroids[i]
            seeds.append((int(cx), int(cy)))
    return seeds


def _flood_fill_from_seeds(
    image_bgr: np.ndarray,
    seeds: list[tuple[int, int]],
    *,
    color_tol: int = 18,
    wall_mask: np.ndarray | None = None,
    floor_mask: np.ndarray | None = None,
) -> np.ndarray:
    """OpenCV floodFill từ mỗi seed point. Tránh wall + floor.

    Args:
        image_bgr: full-res ảnh.
        seeds: list (x, y) anchor points.
        color_tol: tolerance LAB color (lower=stricter).
        wall_mask: exclude wall pixels from flood.
        floor_mask: exclude floor pixels from flood.

    Returns:
        uint8 0/255 mask kết hợp các flood region.
    """
    if not seeds:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    h, w = image_bgr.shape[:2]
    # Convert to LAB for perceptual color flood
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    combined = np.zeros((h, w), dtype=np.uint8)

    # Build "no-go" mask (floor + wall) as floodFill obstacle
    obstacle = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if wall_mask is not None:
        obstacle[1:-1, 1:-1] |= (wall_mask > 128).astype(np.uint8)
    if floor_mask is not None:
        obstacle[1:-1, 1:-1] |= (floor_mask > 128).astype(np.uint8)

    for (sx, sy) in seeds:
        if sy >= h or sx >= w or sy < 0 or sx < 0:
            continue
        local_mask = obstacle.copy()
        try:
            cv2.floodFill(
                lab.copy(),
                local_mask,
                seedPoint=(sx, sy),
                newVal=0,
                loDiff=(color_tol, color_tol, color_tol),
                upDiff=(color_tol, color_tol, color_tol),
                flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4,
            )
            # local_mask filled với 255 (inside fill) vs 0 (outside) vs 1 (obstacle)
            filled = (local_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
            combined = np.maximum(combined, filled)
        except cv2.error as exc:
            log.debug("floodFill fail @ (%d,%d): %s", sx, sy, exc)

    return combined


def _geometric_above_wall(wall_mask: np.ndarray, *, min_pct: float = 0.05) -> np.ndarray:
    """Fallback geometric: vùng ABOVE top edge của wall mask = nghi ngờ ceiling.

    Per column: tìm top wall pixel → mọi pixel ABOVE đó = candidate ceiling.
    """
    binary = (wall_mask > 128).astype(np.uint8)
    h, w = binary.shape
    out = np.zeros((h, w), dtype=np.uint8)

    has_wall = binary.any(axis=0)
    if not has_wall.any():
        return out

    # Vectorized: tìm top wall pixel per column
    # argmax returns first index where binary > 0 (since uint8 1)
    top_y = np.argmax(binary, axis=0)  # (W,) — 0 nếu wall ở row 0
    # Columns no wall → set top_y = h (so no fill)
    no_wall_cols = ~has_wall
    top_y[no_wall_cols] = 0  # skip

    # Vectorized fill: row_idx[:, None] < top_y[None, :]
    row_idx = np.arange(h, dtype=np.int32)
    fill = row_idx[:, None] < top_y[None, :]
    out[fill] = 255
    return out


def _top_region_minus_wall_floor(
    wall_mask: np.ndarray,
    floor_mask: np.ndarray,
    *,
    top_fraction: float = 0.40,
    min_band_above_wall: int = 30,
    door_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
    opening_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Vùng top X% trừ wall/floor/door/window → candidate ceiling cho ảnh modern.

    Logic: ceiling = top fraction MINUS (wall ∪ floor ∪ door ∪ window ∪ opening).
    Loại trừ door+window vì transom phía trên cửa không phải ceiling.
    """
    h, w = wall_mask.shape
    wall_b = (wall_mask > 128).astype(np.uint8)
    floor_b = (floor_mask > 128).astype(np.uint8) if floor_mask is not None else np.zeros_like(wall_b)
    door_b = (door_mask > 128).astype(np.uint8) if door_mask is not None else np.zeros_like(wall_b)
    window_b = (window_mask > 128).astype(np.uint8) if window_mask is not None else np.zeros_like(wall_b)
    opening_b = (opening_mask > 128).astype(np.uint8) if opening_mask is not None else np.zeros_like(wall_b)

    # Transom buffer: exclude door/window region + 20px buffer above (lintel/header)
    # NOT extending all the way to top — would kill legit ceiling above door.
    transom_buffer_px = 20
    dw_combined = door_b | window_b | opening_b
    transom_mask = np.zeros_like(wall_b)
    if dw_combined.any():
        # Dilate door/window UP only by transom_buffer_px
        kernel = np.zeros((transom_buffer_px * 2 + 1, 3), dtype=np.uint8)
        kernel[: transom_buffer_px + 1, :] = 1  # only top half = dilate upward
        transom_mask = cv2.dilate(dw_combined, kernel, iterations=1)

    top_region = np.zeros_like(wall_b)
    top_cap = int(h * top_fraction)
    top_region[:top_cap, :] = 1

    excluded = wall_b | floor_b | dw_combined | transom_mask
    candidate = top_region & ~excluded

    return candidate.astype(np.uint8) * 255


def boost_ceiling_mask(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    *,
    lamp_mask: np.ndarray | None = None,
    light_mask: np.ndarray | None = None,
    wall_mask: np.ndarray | None = None,
    floor_mask: np.ndarray | None = None,
    door_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
    opening_mask: np.ndarray | None = None,
    min_cov_to_boost: float = 0.05,
    color_tol: int = 22,
) -> tuple[np.ndarray, dict]:
    """Boost ceiling mask khi coverage thấp (kiến trúc modern).

    Pipeline:
        1. Nếu ceiling cov >= min_cov_to_boost → no-op (đã đủ).
        2. Else: lấy seed từ lamp/light blobs trong top 60%.
        3. Flood fill từ seeds qua color similarity (avoid wall+floor).
        4. Nếu flood fail/empty: fallback geometric above-wall.
        5. Union ceiling gốc + boost mask.

    Returns:
        (boosted_mask, info_dict)
    """
    info = {
        "boosted": False,
        "method": None,
        "n_seeds": 0,
        "original_cov_pct": 0.0,
        "boosted_cov_pct": 0.0,
    }
    original_cov = (ceiling_mask > 128).mean()
    info["original_cov_pct"] = round(original_cov * 100, 2)

    if original_cov >= min_cov_to_boost:
        info["method"] = "skip_already_sufficient"
        info["boosted_cov_pct"] = info["original_cov_pct"]
        return ceiling_mask, info

    # Step 1: lamp/light seeds
    if lamp_mask is not None and light_mask is not None:
        seeds = _detect_seed_points(lamp_mask, light_mask, top_fraction=0.6)
        info["n_seeds"] = len(seeds)
        if seeds:
            flood = _flood_fill_from_seeds(
                image_bgr, seeds,
                color_tol=color_tol,
                wall_mask=wall_mask, floor_mask=floor_mask,
            )
            # Restrict to top half + above wall top
            if wall_mask is not None:
                wall_binary = (wall_mask > 128).astype(np.uint8)
                # Only keep flood pixels above the bottom of wall region per column
                # Simple: limit to top 60% (modern interior ceiling rarely below)
                h = flood.shape[0]
                flood[int(h * 0.6):, :] = 0

            if (flood > 0).mean() > original_cov + 0.005:
                boosted = np.maximum(ceiling_mask, flood)
                info["boosted"] = True
                info["method"] = "lamp_anchor_flood"
                info["boosted_cov_pct"] = round((boosted > 128).mean() * 100, 2)
                log.info(
                    "Ceiling boost: lamp_flood %d seeds, cov %.2f%% → %.2f%%",
                    len(seeds), info["original_cov_pct"], info["boosted_cov_pct"]
                )
                return boosted, info

    # Step 2: top-region-minus-wall-floor fallback (cho ảnh modern không phào, không lamp)
    if wall_mask is not None:
        top_minus = _top_region_minus_wall_floor(
            wall_mask, floor_mask if floor_mask is not None else np.zeros_like(wall_mask),
            top_fraction=0.40,
            door_mask=door_mask,
            window_mask=window_mask,
            opening_mask=opening_mask,
        )
        top_cov = (top_minus > 0).mean()
        log.info("Ceiling boost candidate: top_minus_wf cov=%.2f%% (threshold >0.5%% or improve)",
                 top_cov * 100)
        # Apply rất nới — bất kỳ improvement nào > 0.5%
        if top_cov >= 0.005 and top_cov > original_cov * 0.5:
            # Soften with morph open + feather
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            top_minus = cv2.morphologyEx(top_minus, cv2.MORPH_OPEN, k)
            top_minus = cv2.GaussianBlur(top_minus, (15, 15), 0)
            boosted = np.maximum(ceiling_mask, top_minus)
            info["boosted"] = True
            info["method"] = "top_minus_wall_floor"
            info["boosted_cov_pct"] = round((boosted > 128).mean() * 100, 2)
            log.info(
                "Ceiling boost: top_minus_wf, cov %.2f%% → %.2f%%",
                info["original_cov_pct"], info["boosted_cov_pct"]
            )
            return boosted, info

    # Step 3: pure geometric above-wall fallback
    if wall_mask is not None:
        geo = _geometric_above_wall(wall_mask)
        geo_cov = (geo > 0).mean()
        if geo_cov > original_cov + 0.01:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            geo = cv2.morphologyEx(geo, cv2.MORPH_OPEN, k)
            geo = cv2.GaussianBlur(geo, (9, 9), 0)
            boosted = np.maximum(ceiling_mask, geo)
            info["boosted"] = True
            info["method"] = "geometric_above_wall"
            info["boosted_cov_pct"] = round((boosted > 128).mean() * 100, 2)
            log.info(
                "Ceiling boost: geometric, cov %.2f%% → %.2f%%",
                info["original_cov_pct"], info["boosted_cov_pct"]
            )
            return boosted, info

    info["method"] = "no_boost_possible"
    info["boosted_cov_pct"] = info["original_cov_pct"]
    return ceiling_mask, info
