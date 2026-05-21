"""Overlap leakage resolver v0.3.1 — sửa 3 lỗi structural detection:

[FIX-1] subtract_casing_from_opening — Opening (cửa kính + sky) đang nuốt thanh đố
        của khung cửa → trừ mask casing dilated khỏi opening để giữ structure.

[FIX-2] reclaim_ceiling_from_wall — Wall mask đang ăn vùng trần phía trên đèn chùm.
        Reclaim các pixel wall ABOVE row của chandelier có color flat (low gradient
        ngang) → push thành ceiling.

[FIX-3] enforce_baseboard_continuity — Baseboard đứt đoạn ở góc chậu cây.
        Hough horizontal line fit trong band wall|floor seam, morphological close
        ép nối liền sofa-trái → sofa-phải.

Kỹ thuật cốt lõi (giữ từ v0.3.0):
    resolve_ceiling_wall_overlap — Sobel directional separator
    resolve_ceiling_floor_overlap — position rule (top vs bottom)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# CORE 1: Sobel directional resolver (giữ nguyên từ v0.3.0)
# ───────────────────────────────────────────────────────────────────

def compute_sobel_direction(image_bgr: np.ndarray, *, ksize: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradient magnitude per direction.

    Returns:
        (abs_gx, abs_gy)
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
    """Resolve ceiling∩wall qua Sobel direction.

    vertical edge dominant → wall wins
    horizontal edge dominant → ceiling wins
    """
    ceiling_b = (ceiling_mask > 128).astype(np.uint8)
    wall_b = (wall_mask > 128).astype(np.uint8)
    overlap = (ceiling_b & wall_b).astype(bool)

    if not overlap.any():
        return ceiling_mask, wall_mask

    log.info("Resolve ceiling∩wall overlap: %d pixels (%.2f%%)",
             int(overlap.sum()), 100.0 * overlap.sum() / overlap.size)

    abs_gx, abs_gy = compute_sobel_direction(image_bgr)
    if smooth_sigma > 0:
        abs_gx = cv2.GaussianBlur(abs_gx, (0, 0), smooth_sigma)
        abs_gy = cv2.GaussianBlur(abs_gy, (0, 0), smooth_sigma)

    vertical_dominant = abs_gx > abs_gy * direction_ratio
    horizontal_dominant = abs_gy > abs_gx * direction_ratio

    ceiling_out = ceiling_mask.copy()
    wall_out = wall_mask.copy()
    ceiling_out[overlap & vertical_dominant] = 0
    wall_out[overlap & horizontal_dominant] = 0

    return ceiling_out, wall_out


def resolve_ceiling_floor_overlap(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    floor_mask: np.ndarray,
    *,
    smooth_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Position-rule resolver: top half → ceiling, bottom half → floor."""
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
    ceiling_out[overlap & ~is_top] = 0
    floor_out[overlap & is_top] = 0
    return ceiling_out, floor_out


# ───────────────────────────────────────────────────────────────────
# FIX-1: Subtract casing/mullion from opening mask
# ───────────────────────────────────────────────────────────────────

def subtract_casing_from_opening(
    opening_mask: np.ndarray,
    casing_mask: np.ndarray,
    door_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
    *,
    dilate_casing_px: int = 3,
    image_bgr: np.ndarray | None = None,
    detect_mullions: bool = True,
) -> np.ndarray:
    """Trừ thanh đố (casing + mullion) khỏi opening mask.

    Args:
        opening_mask: opening uint8 0/255 (window ∪ door ∪ sky).
        casing_mask: casing detected từ molding heuristic.
        door_mask, window_mask: để extract mullion bằng edge detection.
        dilate_casing_px: dilate casing trước khi subtract (an toàn biên).
        image_bgr: nếu có → detect mullion lines bên trong opening.
        detect_mullions: True = thêm Hough line detect THẲNG ĐỨNG/NGANG
                         bên trong opening để tách thanh đố kính.

    Returns:
        opening_cleaned uint8.
    """
    opening_b = (opening_mask > 128).astype(np.uint8) * 255
    casing_b = (casing_mask > 128).astype(np.uint8) * 255

    # Step 1: Dilate casing → buffer an toàn quanh viền cửa
    if dilate_casing_px > 0 and casing_b.sum() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_casing_px * 2 + 1,) * 2)
        casing_dilated = cv2.dilate(casing_b, k, iterations=1)
    else:
        casing_dilated = casing_b

    # Step 2: Trừ casing dilated khỏi opening
    cleaned = cv2.bitwise_and(opening_b, cv2.bitwise_not(casing_dilated))
    removed_by_casing = int((opening_b > 0).sum() - (cleaned > 0).sum())
    log.info("Subtract casing from opening: -%d pixels", removed_by_casing)

    # Step 3: Detect mullion BÊN TRONG opening qua Hough line
    if detect_mullions and image_bgr is not None and cleaned.sum() > 0:
        mullion_mask = _detect_mullion_lines(image_bgr, cleaned)
        if mullion_mask is not None and mullion_mask.sum() > 0:
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(mullion_mask))
            removed_by_mullion = int(mullion_mask.sum() / 255)
            log.info("Subtract mullion lines: -%d pixels", removed_by_mullion)

    return cleaned


def _detect_mullion_lines(
    image_bgr: np.ndarray,
    opening_mask: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    line_thickness: int = 6,
    min_line_length_pct: float = 0.04,
) -> np.ndarray | None:
    """Detect mullion (thanh đố) bên trong opening qua LSD line detector.

    Mullion = đường thẳng dài ĐỨNG hoặc NGANG có edge mạnh bên trong vùng kính.
    """
    h, w = opening_mask.shape
    min_len = int(min(h, w) * min_line_length_pct)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    edges_in_opening = cv2.bitwise_and(edges, opening_mask)

    try:
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
    except (AttributeError, cv2.error):
        log.debug("LSD không có → skip mullion detect")
        return None

    lines = lsd.detect(gray)[0]
    if lines is None or len(lines) == 0:
        return None

    out = np.zeros_like(opening_mask)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        is_v = abs(angle - 90) < 8.0
        is_h = angle < 8.0 or angle > 172.0
        if not (is_v or is_h):
            continue
        # Verify line center INSIDE opening
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < w and 0 <= cy < h and opening_mask[cy, cx] > 0:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     255, thickness=line_thickness)

    # Intersect với opening để tránh leak ra ngoài
    return cv2.bitwise_and(out, opening_mask)


# ───────────────────────────────────────────────────────────────────
# FIX-2: Reclaim ceiling from wall (above chandelier band)
# ───────────────────────────────────────────────────────────────────

def reclaim_ceiling_from_wall(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    wall_mask: np.ndarray,
    *,
    lamp_mask: np.ndarray | None = None,
    light_mask: np.ndarray | None = None,
    top_safe_fraction: float = 0.30,
    color_flatness_threshold: float = 12.0,
    min_horizontal_run_px: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Reclaim wall pixels ABOVE chandelier band → ceiling.

    Logic:
        1. Tìm hàng y_lamp = top row của lamp/light mask (chandelier).
        2. Vùng wall ABOVE y_lamp có color flat (low ∂x gradient horizontal)
           → push thành ceiling.
        3. Đảm bảo continuous horizontal run ≥ min_horizontal_run_px (tránh noise).

    Args:
        image_bgr: full-res BGR.
        ceiling_mask, wall_mask: uint8 0/255.
        lamp_mask, light_mask: ADE lamp(36) + light(82) softmax binary.
        top_safe_fraction: chỉ reclaim trong top X% of image.
        color_flatness_threshold: max gradient horizontal cho pixel "flat ceiling".
        min_horizontal_run_px: min run width để consider ceiling vùng (lọc noise).

    Returns:
        (ceiling_reclaimed, wall_reduced)
    """
    h, w = image_bgr.shape[:2]
    wall_b = (wall_mask > 128).astype(np.uint8)
    ceiling_b = (ceiling_mask > 128).astype(np.uint8)

    # Bước 1: Xác định y_anchor — top row chandelier hoặc cap_top
    y_anchor = int(h * top_safe_fraction)
    if lamp_mask is not None or light_mask is not None:
        combined_light = np.zeros((h, w), dtype=np.uint8)
        if lamp_mask is not None:
            combined_light = np.maximum(combined_light, (lamp_mask > 128).astype(np.uint8))
        if light_mask is not None:
            combined_light = np.maximum(combined_light, (light_mask > 128).astype(np.uint8))
        if combined_light.sum() > 100:
            ys = np.where(combined_light > 0)[0]
            if len(ys) > 0:
                y_lamp_top = int(ys.min())
                y_lamp_bottom = int(np.percentile(ys, 50))
                # Chandelier nằm ở rows [y_lamp_top, y_lamp_bottom]
                # Reclaim ngay BÊN TRÊN chandelier band → ceiling
                y_anchor = max(y_anchor, y_lamp_bottom)
                log.info("Chandelier detected at rows %d-%d, y_anchor=%d",
                         y_lamp_top, y_lamp_bottom, y_anchor)

    # Bước 2: Compute horizontal gradient (∂x trên gray)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gx_smooth = cv2.GaussianBlur(gx, (0, 0), 6.0)
    color_flat = gx_smooth < color_flatness_threshold

    # Bước 3: Candidate = wall ABOVE y_anchor AND color_flat
    candidate = wall_b.copy()
    candidate[y_anchor:, :] = 0  # only rows above y_anchor
    candidate = candidate & color_flat.astype(np.uint8)

    if candidate.sum() < 500:
        log.info("Reclaim ceiling: candidate < 500 pixels, skip")
        return ceiling_mask, wall_mask

    # Bước 4: Lọc — chỉ giữ rows có horizontal run đủ dài
    row_runs = candidate.sum(axis=1)  # số pixel flat ở mỗi row
    valid_rows = row_runs >= min_horizontal_run_px
    if not valid_rows.any():
        log.info("Reclaim ceiling: no row có run ≥ %d", min_horizontal_run_px)
        return ceiling_mask, wall_mask

    # Mask cuối cùng: candidate AND row trong valid_rows
    final_reclaim = candidate.copy()
    final_reclaim[~valid_rows, :] = 0

    # Bước 5: Apply morphological close → smooth boundary
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    final_reclaim = cv2.morphologyEx(final_reclaim, cv2.MORPH_CLOSE, k)

    # Soft alpha feather
    feathered = cv2.GaussianBlur(final_reclaim * 255, (15, 15), 0).astype(np.uint8)

    n_reclaimed = int((final_reclaim > 0).sum())
    log.info("Reclaim ceiling from wall: +%d pixels (%.2f%%) above row %d",
             n_reclaimed, 100.0 * n_reclaimed / (h * w), y_anchor)

    ceiling_new = np.maximum(ceiling_mask, feathered)
    wall_new = wall_mask.copy()
    wall_new[(final_reclaim > 0)] = 0

    return ceiling_new, wall_new


# ───────────────────────────────────────────────────────────────────
# FIX-3: Enforce baseboard continuity (Hough horizontal line)
# ───────────────────────────────────────────────────────────────────

def enforce_baseboard_continuity(
    image_bgr: np.ndarray,
    baseboard_mask: np.ndarray,
    wall_mask: np.ndarray,
    floor_mask: np.ndarray,
    *,
    seam_band_px: int = 30,
    hough_min_len_pct: float = 0.06,
    hough_max_gap_pct: float = 0.04,
    close_kernel_w: int = 35,
) -> np.ndarray:
    """Ép baseboard liên tục dọc seam wall|floor.

    Logic:
        1. Compute seam wall|floor → band ±N px.
        2. Hough horizontal line trong band (allow gap, min length 6% width).
        3. Union với baseboard gốc.
        4. Morphological close (kernel ngang dài 35px) → nối đứt đoạn.
    """
    h, w = image_bgr.shape[:2]
    wall_b = (wall_mask > 128).astype(np.uint8)
    floor_b = (floor_mask > 128).astype(np.uint8)

    # Seam wall|floor
    wall_d = cv2.dilate(wall_b, np.ones((3, 3), np.uint8), iterations=1)
    floor_d = cv2.dilate(floor_b, np.ones((3, 3), np.uint8), iterations=1)
    seam = wall_d & floor_d
    if seam.sum() < 50:
        return baseboard_mask

    # Band around seam
    k_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (seam_band_px * 2 + 1,) * 2)
    band = cv2.dilate(seam, k_band, iterations=1) * 255

    # Edges trong band
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150, apertureSize=3)
    edges_in_band = cv2.bitwise_and(edges, band)

    # HoughLinesP horizontal
    min_len = int(w * hough_min_len_pct)
    max_gap = int(w * hough_max_gap_pct)
    out = baseboard_mask.copy()

    try:
        lines = cv2.HoughLinesP(
            edges_in_band, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=min_len, maxLineGap=max_gap,
        )
    except cv2.error:
        lines = None

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Chỉ giữ line gần ngang (±10°)
            if angle < 10.0 or angle > 170.0:
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)
                if 0 <= cy < h and 0 <= cx < w and band[cy, cx] > 0:
                    cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                             255, thickness=5)
        log.info("Baseboard Hough: detected %d lines", len(lines))

    # Morphological close ngang (kernel rộng) → nối đứt đoạn
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_w, 3))
    closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close)

    # Restrict to band (tránh leak lên tường / sàn)
    return cv2.bitwise_and(closed, band)


# ───────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────

def resolve_all_overlaps(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    apply_sobel: bool = True,
    apply_position: bool = True,
    apply_casing_subtract: bool = True,
    apply_ceiling_reclaim: bool = True,
    apply_baseboard_continuity: bool = True,
    lamp_mask: np.ndarray | None = None,
    light_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bộ resolver tổng — fix all overlaps + 3 lỗi structural v0.3.1.

    Pipeline:
        1. Sobel directional ceiling|wall
        2. Position ceiling|floor
        3. Subtract casing from opening (FIX-1)
        4. Reclaim ceiling from wall above chandelier (FIX-2)
        5. Enforce baseboard continuity Hough horizontal (FIX-3)

    Returns: new dict[name -> mask] phân lại exclusive.
    """
    out = {k: v.copy() for k, v in masks.items()}

    # Step 1: Sobel ceiling|wall
    if apply_sobel and "ceiling" in out and "wall" in out:
        out["ceiling"], out["wall"] = resolve_ceiling_wall_overlap(
            image_bgr, out["ceiling"], out["wall"]
        )

    # Step 2: Position ceiling|floor
    if apply_position and "ceiling" in out and "floor" in out:
        out["ceiling"], out["floor"] = resolve_ceiling_floor_overlap(
            image_bgr, out["ceiling"], out["floor"]
        )

    # Step 3 (FIX-1): subtract casing from opening
    if apply_casing_subtract and "opening" in out and "casing" in out:
        out["opening"] = subtract_casing_from_opening(
            out["opening"],
            out["casing"],
            door_mask=out.get("door"),
            window_mask=out.get("window"),
            image_bgr=image_bgr,
            dilate_casing_px=3,
            detect_mullions=True,
        )

    # Step 4 (FIX-2): reclaim ceiling from wall
    if apply_ceiling_reclaim and "ceiling" in out and "wall" in out:
        out["ceiling"], out["wall"] = reclaim_ceiling_from_wall(
            image_bgr,
            out["ceiling"],
            out["wall"],
            lamp_mask=lamp_mask,
            light_mask=light_mask,
        )

    # Step 5 (FIX-3): baseboard continuity
    if apply_baseboard_continuity and "baseboard" in out \
       and "wall" in out and "floor" in out:
        out["baseboard"] = enforce_baseboard_continuity(
            image_bgr,
            out["baseboard"],
            out["wall"],
            out["floor"],
        )

    return out
