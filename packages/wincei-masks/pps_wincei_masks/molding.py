"""Phào chỉ detection — không có sẵn trong ADE20K, build heuristic.

3 LOẠI MOLDING TRONG ẢNH BĐS:
    ┌──────────────────────────────────────────────────────────────┐
    │ Crown molding  : phào trần — seam ceiling|wall, ngang        │
    │ Baseboard      : phào chân tường — seam floor|wall, ngang    │
    │ Casing         : nẹp cửa/cửa sổ — viền window|door|wall      │
    └──────────────────────────────────────────────────────────────┘

PIPELINE:
    1. Lấy seam giữa 2 region từ semantic argmax
    2. Dilate seam ra thành band ±N px (default ±20)
    3. Trong band: tính edge map (Canny) + LSD line detect
    4. Mask = band ∩ strong_edge → strip mask thin
    5. Morphology close → kết nối các đoạn rời
    6. Connected components → keep large only (drop noise)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MoldingMasks:
    """3 loại phào chỉ + composite."""
    crown: np.ndarray       # phào trần
    baseboard: np.ndarray   # phào chân tường
    casing: np.ndarray      # nẹp cửa/cửa sổ
    combined: np.ndarray    # union all


def _seam_mask(region_a: np.ndarray, region_b: np.ndarray, dilate_px: int = 20) -> np.ndarray:
    """Compute seam (giao tuyến) giữa 2 region rồi dilate thành band.

    Args:
        region_a, region_b: binary uint8 0/1 hoặc 0/255.
        dilate_px: ±N px buffer around seam.

    Returns:
        uint8 0/255 — band ngang qua biên 2 region.
    """
    a = (region_a > 0).astype(np.uint8)
    b = (region_b > 0).astype(np.uint8)
    # Edge of a ∩ neighbor of b
    a_dilate = cv2.dilate(a, np.ones((3, 3), np.uint8), iterations=1)
    b_dilate = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)
    seam = (a_dilate & b_dilate).astype(np.uint8)
    if seam.sum() == 0:
        return np.zeros_like(a, dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    band = cv2.dilate(seam, k, iterations=1)
    return band * 255


def _strong_edge_in_band(
    image_bgr: np.ndarray,
    band: np.ndarray,
    *,
    canny_low: int = 60,
    canny_high: int = 150,
    line_min_length: int = 80,
    line_orientation: str | None = None,  # 'horizontal' | 'vertical' | None
    angle_tol: float = 15.0,
) -> np.ndarray:
    """Find strong edges trong band + filter theo orientation nếu có.

    Returns: uint8 0/255 — chỉ pixels có edge mạnh + đúng hướng.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3, L2gradient=True)
    edges_in_band = cv2.bitwise_and(edges, band)

    if line_orientation is None:
        return edges_in_band

    # LSD line detection để filter theo hướng
    try:
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
    except (AttributeError, cv2.error):
        log.warning("LSD line detector không có → trả raw edges")
        return edges_in_band

    lines = lsd.detect(gray)[0]
    if lines is None:
        return edges_in_band

    mask = np.zeros_like(gray, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < line_min_length:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        is_h = abs(angle) < angle_tol or abs(abs(angle) - 180) < angle_tol
        is_v = abs(abs(angle) - 90) < angle_tol
        if line_orientation == "horizontal" and not is_h:
            continue
        if line_orientation == "vertical" and not is_v:
            continue
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=4)

    # Intersect with band
    return cv2.bitwise_and(mask, band)


def _clean_strip(mask: np.ndarray, *, close_px: int = 5, min_area_pct: float = 0.0005) -> np.ndarray:
    """Morphology close + remove small components."""
    if mask.sum() == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px * 2 + 1, close_px * 2 + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # Connected components filter
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    total_area = closed.shape[0] * closed.shape[1]
    min_area = total_area * min_area_pct
    out = np.zeros_like(closed)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _seam_band_fallback(mask_a: np.ndarray, dilate_px: int) -> np.ndarray:
    """Fallback nếu không có region B — chỉ lấy biên dilate của A."""
    a = (mask_a > 0).astype(np.uint8)
    if a.sum() == 0:
        return np.zeros_like(a, dtype=np.uint8) * 255
    grad = cv2.morphologyEx(a, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    return cv2.dilate(grad, k, iterations=1) * 255


def detect_moldings(
    image_bgr: np.ndarray,
    wall_mask: np.ndarray,
    ceiling_mask: np.ndarray,
    floor_mask: np.ndarray,
    window_mask: np.ndarray,
    door_mask: np.ndarray,
    *,
    crown_dilate: int = 18,
    baseboard_dilate: int = 18,
    casing_dilate: int = 8,            # giảm từ 16 → 8 (tránh tràn vào sofa)
    line_min_length_crown: int = 60,
    line_min_length_base: int = 60,
    line_min_length_casing: int = 50,  # tăng từ 25 → 50 (chỉ giữ line dài)
    canny_low: int = 60,
    canny_high: int = 150,
    casing_max_pct: float = 0.06,      # cap casing ≤ 6% diện tích
) -> MoldingMasks:
    """Detect 3 loại phào chỉ (relaxed thresholds — bắt cả phào mảnh modern)."""

    def _seam_or_grad(a: np.ndarray, b: np.ndarray, dilate: int) -> np.ndarray:
        seam = _seam_mask(a, b, dilate_px=dilate)
        if seam.sum() > 0:
            return seam
        # Region B vắng (vd no ceiling detected) → fallback dùng gradient của A
        return _seam_band_fallback(a, dilate)

    # Crown (phào trần): wall|ceiling, ngang
    crown_band = _seam_or_grad(wall_mask, ceiling_mask, crown_dilate)
    crown = _strong_edge_in_band(
        image_bgr, crown_band,
        canny_low=canny_low, canny_high=canny_high,
        line_orientation="horizontal",
        line_min_length=line_min_length_crown,
    )
    crown = _clean_strip(crown, close_px=5)

    # Baseboard (phào chân tường): wall|floor, ngang
    base_band = _seam_or_grad(wall_mask, floor_mask, baseboard_dilate)
    baseboard = _strong_edge_in_band(
        image_bgr, base_band,
        canny_low=canny_low, canny_high=canny_high,
        line_orientation="horizontal",
        line_min_length=line_min_length_base,
    )
    baseboard = _clean_strip(baseboard, close_px=5)

    # Casing (nẹp cửa/cửa sổ): wall|(window∪door), bất kỳ hướng
    win_or_door = cv2.bitwise_or((window_mask > 0).astype(np.uint8) * 255,
                                  (door_mask > 0).astype(np.uint8) * 255)
    casing_band = _seam_or_grad(wall_mask, win_or_door, casing_dilate)
    casing = _strong_edge_in_band(
        image_bgr, casing_band,
        canny_low=canny_low, canny_high=canny_high,
        line_orientation=None,
        line_min_length=line_min_length_casing,
    )
    casing = _clean_strip(casing, close_px=4)

    # Cap casing coverage — nếu > casing_max_pct → khả năng tràn ra sofa/floor → reject
    casing_cov = (casing > 0).mean()
    if casing_cov > casing_max_pct:
        log.warning("casing %.1f%% > cap %.1f%% → clear (likely false positive)",
                    casing_cov * 100, casing_max_pct * 100)
        casing = np.zeros_like(casing)

    combined = cv2.bitwise_or(cv2.bitwise_or(crown, baseboard), casing)
    return MoldingMasks(crown=crown, baseboard=baseboard, casing=casing, combined=combined)
