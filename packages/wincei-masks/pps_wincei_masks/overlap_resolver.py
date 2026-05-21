"""Overlap leakage resolver — phân tách mask wall/ceiling/floor khi overlap.

Kỹ thuật: Sobel directional edge analysis
    - Vertical edges (∂x dominant)  → thuộc về WALL (boundary giữa các tường)
    - Horizontal edges (∂y dominant) → thuộc về CEILING/FLOOR (boundary với trần/sàn)

Khi mask_ceiling ∩ mask_wall > 0:
    Per overlapping pixel:
        Tính Sobel gradient direction tại pixel đó
        Nếu |Gx| > |Gy| * 1.5 → assign WALL (vertical edge dominant)
        Nếu |Gy| > |Gx| * 1.5 → assign CEILING (horizontal edge dominant)
        Khác → ưu tiên class có higher score
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def compute_sobel_direction(image_bgr: np.ndarray, *, ksize: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients trên gray.

    Returns:
        (abs_gx, abs_gy): magnitude của ∂x và ∂y, same shape.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.abs(gx), np.abs(gy)


def resolve_ceiling_wall_overlap(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    wall_mask: np.ndarray,
    *,
    direction_ratio: float = 1.5,
    smooth_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve overlap ceiling∩wall qua Sobel direction.

    Args:
        image_bgr: full-res ảnh.
        ceiling_mask, wall_mask: uint8 0/255.
        direction_ratio: ngưỡng |Gx|/|Gy| (1.5 = mạnh vertical thì gán wall).
        smooth_sigma: blur Sobel mag để stable hơn.

    Returns:
        (ceiling_resolved, wall_resolved) — overlap được phân lại exclusive.
    """
    ceiling_b = (ceiling_mask > 128).astype(np.uint8)
    wall_b = (wall_mask > 128).astype(np.uint8)
    overlap = (ceiling_b & wall_b).astype(bool)

    if not overlap.any():
        return ceiling_mask, wall_mask

    log.info("Resolve overlap: %d pixels (%.2f%%) ceiling∩wall",
             int(overlap.sum()), 100.0 * overlap.sum() / overlap.size)

    abs_gx, abs_gy = compute_sobel_direction(image_bgr)
    # Smooth để direction stable (đỡ noisy pixel-by-pixel)
    if smooth_sigma > 0:
        abs_gx = cv2.GaussianBlur(abs_gx, (0, 0), smooth_sigma)
        abs_gy = cv2.GaussianBlur(abs_gy, (0, 0), smooth_sigma)

    # Vertical-dominant (|Gx| > |Gy| * ratio) → wall wins
    vertical_dominant = abs_gx > abs_gy * direction_ratio
    horizontal_dominant = abs_gy > abs_gx * direction_ratio

    ceiling_out = ceiling_mask.copy()
    wall_out = wall_mask.copy()

    # In overlap region:
    #   vertical_dominant → set ceiling = 0 (keep wall)
    #   horizontal_dominant → set wall = 0 (keep ceiling)
    #   neither → keep both (use class confidence later if needed)
    kill_ceiling = overlap & vertical_dominant
    kill_wall = overlap & horizontal_dominant

    ceiling_out[kill_ceiling] = 0
    wall_out[kill_wall] = 0

    n_kept_ceiling = int((overlap & ~vertical_dominant).sum())
    n_kept_wall = int((overlap & ~horizontal_dominant).sum())
    log.info("Overlap resolved: ceiling kept %d, wall kept %d (Sobel directional)",
             n_kept_ceiling, n_kept_wall)
    return ceiling_out, wall_out


def resolve_ceiling_floor_overlap(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    floor_mask: np.ndarray,
    *,
    smooth_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ceiling∩floor overlap qua position rule.

    Trong ảnh BĐS, ceiling LUÔN nằm trên floor → đơn giản hơn Sobel:
        Pixel ở top 50% → ceiling wins
        Pixel ở bottom 50% → floor wins
    """
    ceiling_b = (ceiling_mask > 128).astype(np.uint8)
    floor_b = (floor_mask > 128).astype(np.uint8)
    overlap = (ceiling_b & floor_b).astype(bool)
    if not overlap.any():
        return ceiling_mask, floor_mask

    h = image_bgr.shape[0]
    row_idx = np.arange(h, dtype=np.int32)[:, None]
    is_top = (row_idx < h * 0.5).astype(bool)

    ceiling_out = ceiling_mask.copy()
    floor_out = floor_mask.copy()
    ceiling_out[overlap & ~is_top] = 0  # bottom overlap → floor wins
    floor_out[overlap & is_top] = 0     # top overlap → ceiling wins
    return ceiling_out, floor_out


def resolve_all_overlaps(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    apply_sobel: bool = True,
    apply_position: bool = True,
) -> dict[str, np.ndarray]:
    """Bộ resolver tổng — apply trên dict[name -> mask] in-place style.

    Returns new dict với masks đã phân lại exclusive (ceiling, wall, floor).
    """
    out = {k: v.copy() for k, v in masks.items()}

    if apply_sobel and "ceiling" in out and "wall" in out:
        out["ceiling"], out["wall"] = resolve_ceiling_wall_overlap(
            image_bgr, out["ceiling"], out["wall"]
        )

    if apply_position and "ceiling" in out and "floor" in out:
        out["ceiling"], out["floor"] = resolve_ceiling_floor_overlap(
            image_bgr, out["ceiling"], out["floor"]
        )

    return out
