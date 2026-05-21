"""Overlap leakage resolver v0.3.2 — Localized + Dynamic + Geometry-strict.

4 NGUYÊN TẮC THIẾT KẾ:
    [1] NON-PROPERTY EXCLUSION: build exclusion_mask từ ADE20K furniture class IDs
        (sofa, bàn, ghế, gối, thảm, rèm, tranh, ottoman, ...).
        ÉP TOÁN HỌC zero-out tất cả casing/baseboard/crown/ceiling pixels trong exclusion.

    [2] DYNAMIC CEILING CLOSE: kernel size = 2% của image width (auto co giãn theo góc chụp).
        + Connected component noise filter (lọc đốm < 0.02% area).
        → Ceiling mask phẳng mịn không bị rách lốm đốm bóng quạt trần.

    [3] DISTANCE-TRANSFORM CASING CONSTRAINT: casing/baseboard pixels chỉ được tồn tại
        cách lõi kính (opening) tối đa 1.5% width. Vượt là zero-out via cv2.distanceTransform.
        → Pink không tràn vào rèm, tủ, bếp.

    [4] KHÔNG còn heuristic line drawing fixed-pixel. Tất cả threshold theo % image dim.

Public API:
    resolve_all_overlaps(image_bgr, masks, *, ade_argmax_id=None, ...) -> dict[name, mask]
    build_furniture_exclusion_mask(ade_argmax_id) -> uint8 mask
    apply_exclusion_to_strip_masks(masks, exclusion) -> in-place mutation
    dynamic_close_ceiling(image_bgr, ceiling_mask, *, width_pct=0.02) -> mask
    constrain_strip_to_opening(strip_mask, opening_mask, image_bgr, *, max_pct=0.015) -> mask
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# ADE20K FURNITURE / NON-PROPERTY CLASS IDS
# ───────────────────────────────────────────────────────────────────
# Tất cả "movable / non-architectural" — zero-out khỏi structural masks.
ADE_NON_PROPERTY_IDS = frozenset({
    7,   # bed
    10,  # cabinet
    15,  # table
    18,  # curtain
    19,  # chair
    22,  # painting
    23,  # sofa
    24,  # shelf
    28,  # rug
    31,  # armchair
    33,  # fence
    34,  # desk
    36,  # lamp (đèn — nhưng treo trần nên đôi khi cần giữ)
    37,  # bathtub
    39,  # cushion
    41,  # box
    44,  # counter
    50,  # refrigerator
    57,  # bookcase
    62,  # bench
    64,  # coffee table
    65,  # toilet
    66,  # flower
    67,  # book
    70,  # countertop
    74,  # kitchen island
    75,  # computer
    78,  # car (rare nhưng có khi xuất hiện)
    81,  # bus
    83,  # television
    89,  # towel
    90,  # arcade machine
    97,  # awning
    99,  # bottle
    105, # tray
    108, # ottoman
    110, # pillow
    115, # bag
    119, # monitor
    124, # microwave
    125, # pot/plant
    130, # plate
    132, # screen
    134, # sculpture
    135, # hood
    142, # ashcan
    143, # fan (quạt — quạt trần)
    144, # pier
    149, # flag
})


def build_furniture_exclusion_mask(
    ade_argmax_id: np.ndarray | None,
    *,
    dilate_px: int = 3,
    extra_class_ids: set[int] | None = None,
) -> np.ndarray | None:
    """Build binary mask của TẤT CẢ non-property regions từ ADE20K argmax.

    Args:
        ade_argmax_id: (H, W) int32 argmax map từ SegFormer (toàn bộ 150 classes).
        dilate_px: dilate exclusion thêm vài px để bao biên an toàn.
        extra_class_ids: thêm class IDs custom vào exclusion.

    Returns:
        (H, W) uint8 0/255 — vùng cấm structural masks chạm vào.
        None nếu argmax không có.
    """
    if ade_argmax_id is None:
        return None

    exclusion_ids = set(ADE_NON_PROPERTY_IDS)
    if extra_class_ids:
        exclusion_ids |= extra_class_ids

    # Vectorized check: pixel ∈ exclusion_ids
    flat = np.isin(ade_argmax_id, list(exclusion_ids))
    exclusion = (flat.astype(np.uint8)) * 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_px * 2 + 1,) * 2)
        exclusion = cv2.dilate(exclusion, k, iterations=1)

    cov_pct = (exclusion > 0).mean() * 100
    log.info("Furniture exclusion mask: %.2f%% coverage (%d ADE classes)",
             cov_pct, len(exclusion_ids))
    return exclusion


def apply_exclusion_to_strip_masks(
    masks: dict[str, np.ndarray],
    exclusion: np.ndarray | None,
    target_classes: tuple[str, ...] = ("casing", "baseboard", "crown", "ceiling"),
) -> dict[str, np.ndarray]:
    """ÉP tất cả pixel trong exclusion về 0 cho strip + ceiling masks.

    Casing/baseboard/crown KHÔNG ĐƯỢC nằm trên sofa, bàn, rèm, tranh.
    Ceiling KHÔNG ĐƯỢC chứa pixel của quạt trần/đồ treo.
    """
    if exclusion is None:
        return masks

    out = dict(masks)
    for cls in target_classes:
        if cls not in out:
            continue
        before = int((out[cls] > 128).sum())
        out[cls] = cv2.bitwise_and(out[cls], cv2.bitwise_not(exclusion))
        after = int((out[cls] > 128).sum())
        removed = before - after
        if removed > 0:
            log.info("Exclusion applied to '%s': -%d pixels (%.2f%% of class)",
                     cls, removed, 100.0 * removed / max(1, before))
    return out


# ───────────────────────────────────────────────────────────────────
# DYNAMIC CEILING CLOSE + CC NOISE FILTER
# ───────────────────────────────────────────────────────────────────

def dynamic_close_ceiling(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    *,
    width_pct: float = 0.02,
    min_blob_area_pct: float = 0.0002,
    fill_holes_pct: float = 0.0005,
) -> np.ndarray:
    """Dynamic morphological close + blob filter cho ceiling.

    Args:
        image_bgr: full-res ảnh (chỉ cần shape).
        ceiling_mask: uint8 0/255.
        width_pct: kernel size = width * width_pct (auto co giãn).
        min_blob_area_pct: drop connected components < area_pct * total.
        fill_holes_pct: fill INTERNAL holes < this pct (bóng quạt trần).

    Returns:
        ceiling cleaned uint8.
    """
    h, w = image_bgr.shape[:2]
    total_area = h * w

    if (ceiling_mask > 128).sum() < 50:
        return ceiling_mask

    # Step 1: Dynamic kernel close
    k_size = max(5, int(w * width_pct))
    if k_size % 2 == 0:
        k_size += 1  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(ceiling_mask, cv2.MORPH_CLOSE, kernel)
    log.info("Ceiling dynamic close: kernel %dx%d (%.2f%% of width)",
             k_size, k_size, width_pct * 100)

    # Step 2: Drop small connected components (đốm nhiễu rời rạc)
    bin_mask = (closed > 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    min_area = total_area * min_blob_area_pct
    keep = np.zeros_like(bin_mask)
    n_dropped = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
        else:
            n_dropped += 1
    if n_dropped > 0:
        log.info("Ceiling noise filter: dropped %d blobs (< %.2f%% area each)",
                 n_dropped, min_blob_area_pct * 100)

    # Step 3: Fill INTERNAL holes (bóng quạt trần, đèn âm trần)
    # Trick: invert + connected components ngoài → giữ vùng "hole" trong mask
    keep_u8 = keep * 255
    inv = cv2.bitwise_not(keep_u8)
    n_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    fill_area_max = total_area * fill_holes_pct * 50  # holes có thể to hơn blobs
    holes_filled = 0
    for i in range(1, n_inv):
        area = stats_inv[i, cv2.CC_STAT_AREA]
        # External background = largest CC. Skip nó.
        # Internal holes = nhỏ hơn và không chạm biên ảnh
        if area > fill_area_max:
            continue
        # Verify not touching border
        x = stats_inv[i, cv2.CC_STAT_LEFT]
        y = stats_inv[i, cv2.CC_STAT_TOP]
        ww = stats_inv[i, cv2.CC_STAT_WIDTH]
        hh = stats_inv[i, cv2.CC_STAT_HEIGHT]
        touches_border = (x == 0 or y == 0 or x + ww >= image_bgr.shape[1]
                          or y + hh >= image_bgr.shape[0])
        if touches_border:
            continue
        keep[labels_inv == i] = 1
        holes_filled += 1
    if holes_filled > 0:
        log.info("Ceiling holes filled: %d internal holes", holes_filled)

    return (keep * 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────────
# DISTANCE-TRANSFORM STRIP CONSTRAINT
# ───────────────────────────────────────────────────────────────────

def constrain_strip_to_opening(
    strip_mask: np.ndarray,
    opening_mask: np.ndarray,
    image_bgr: np.ndarray,
    *,
    max_pct: float = 0.015,
) -> np.ndarray:
    """Strip mask (casing/baseboard) chỉ được tồn tại trong khoảng cách max_pct * width
    tính từ biên opening.

    Args:
        strip_mask: uint8 mask cần constrain.
        opening_mask: opening mask làm reference center.
        image_bgr: chỉ cần shape.
        max_pct: ngưỡng khoảng cách (default 1.5% width).

    Returns:
        strip cleaned (vùng ngoài tầm bị zero-out).
    """
    w = image_bgr.shape[1]
    max_dist = max(8, int(w * max_pct))

    if (opening_mask > 128).sum() < 100:
        # Không có opening → strip không cần constrain
        return strip_mask
    if (strip_mask > 128).sum() < 50:
        return strip_mask

    # Distance từ mỗi pixel → opening gần nhất
    # opening = 1 → dist=0; xa hơn → dist tăng dần
    opening_b = (opening_mask > 128).astype(np.uint8)
    inv_opening = (1 - opening_b).astype(np.uint8)
    dist = cv2.distanceTransform(inv_opening, cv2.DIST_L2, 3)

    cleaned = strip_mask.copy()
    far_mask = dist > max_dist
    before = int((cleaned > 128).sum())
    cleaned[far_mask] = 0
    after = int((cleaned > 128).sum())

    removed = before - after
    if removed > 0:
        log.info("Distance constrain: -%d pixels (> %dpx from opening, "
                 "%.2f%% width)", removed, max_dist, max_pct * 100)
    return cleaned


# ───────────────────────────────────────────────────────────────────
# CORE: Sobel directional resolver (giữ nhưng disable cho strip classes)
# ───────────────────────────────────────────────────────────────────

def resolve_ceiling_wall_overlap(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    wall_mask: np.ndarray,
    *,
    direction_ratio: float = 1.5,
    smooth_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sobel directional: vertical edge → wall wins, horizontal → ceiling wins."""
    ceiling_b = (ceiling_mask > 128).astype(np.uint8)
    wall_b = (wall_mask > 128).astype(np.uint8)
    overlap = (ceiling_b & wall_b).astype(bool)

    if not overlap.any():
        return ceiling_mask, wall_mask

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5))
    abs_gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5))
    if smooth_sigma > 0:
        abs_gx = cv2.GaussianBlur(abs_gx, (0, 0), smooth_sigma)
        abs_gy = cv2.GaussianBlur(abs_gy, (0, 0), smooth_sigma)

    vertical_dom = abs_gx > abs_gy * direction_ratio
    horizontal_dom = abs_gy > abs_gx * direction_ratio

    ceiling_out = ceiling_mask.copy()
    wall_out = wall_mask.copy()
    ceiling_out[overlap & vertical_dom] = 0
    wall_out[overlap & horizontal_dom] = 0

    log.info("Sobel resolver ceiling∩wall: %d overlap px → split by direction",
             int(overlap.sum()))
    return ceiling_out, wall_out


def resolve_ceiling_floor_overlap(
    image_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    floor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Position rule: top half → ceiling, bottom half → floor."""
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
# OPENING / CASING / MULLION CLEANUP
# ───────────────────────────────────────────────────────────────────

def subtract_casing_from_opening(
    opening_mask: np.ndarray,
    casing_mask: np.ndarray,
    *,
    image_bgr: np.ndarray | None = None,
    dilate_casing_px: int = 3,
    detect_mullions: bool = True,
) -> np.ndarray:
    """Trừ casing + mullion (đường thẳng đố) khỏi opening."""
    opening_b = (opening_mask > 128).astype(np.uint8) * 255
    casing_b = (casing_mask > 128).astype(np.uint8) * 255

    if dilate_casing_px > 0 and casing_b.sum() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_casing_px * 2 + 1,) * 2)
        casing_dil = cv2.dilate(casing_b, k, iterations=1)
    else:
        casing_dil = casing_b

    cleaned = cv2.bitwise_and(opening_b, cv2.bitwise_not(casing_dil))

    if detect_mullions and image_bgr is not None and cleaned.sum() > 0:
        mullion = _detect_mullion_lines_in(image_bgr, cleaned)
        if mullion is not None and mullion.sum() > 0:
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(mullion))
    return cleaned


def _detect_mullion_lines_in(
    image_bgr: np.ndarray, opening_mask: np.ndarray,
) -> np.ndarray | None:
    """LSD line detect bên trong opening, chỉ giữ line đứng/ngang dài."""
    h, w = opening_mask.shape
    min_len = int(min(h, w) * 0.04)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
    except (AttributeError, cv2.error):
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
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < w and 0 <= cy < h and opening_mask[cy, cx] > 0:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     255, thickness=6)
    return cv2.bitwise_and(out, opening_mask)


# ───────────────────────────────────────────────────────────────────
# ROI BOUNDARY RULES — Khoanh vùng cứng v0.3.3
# ───────────────────────────────────────────────────────────────────

def enforce_ceiling_roi(
    ceiling_mask: np.ndarray,
    image_bgr: np.ndarray,
    *,
    max_height_pct: float = 0.45,
) -> np.ndarray:
    """[RULE 1] Ceiling pixels CHỈ được phép tồn tại trong vùng top X% chiều cao.

    Bất kỳ pixel ceiling nào lọt xuống dưới y = H * max_height_pct → zero-out.
    Tránh ceiling lan vào sofa/bàn trà/sàn.

    Args:
        ceiling_mask: uint8 0/255.
        image_bgr: chỉ cần shape.
        max_height_pct: ngưỡng % chiều cao (default 0.45 = top 45%).

    Returns:
        ceiling cleaned uint8.
    """
    h = image_bgr.shape[0]
    cap_y = int(h * max_height_pct)
    before = int((ceiling_mask > 128).sum())
    if before == 0:
        return ceiling_mask
    out = ceiling_mask.copy()
    out[cap_y:, :] = 0  # ép zero tất cả pixel bên dưới cap_y
    after = int((out > 128).sum())
    removed = before - after
    if removed > 0:
        log.info("Ceiling ROI cap @ y=%d (%.0f%% height): -%d pixels",
                 cap_y, max_height_pct * 100, removed)
    return out


def _bbox_of_mask(mask: np.ndarray, *, min_area: int = 100) -> tuple[int, int, int, int] | None:
    """Find tight bounding box of mask, return (y_min, x_min, y_max, x_max) hoặc None."""
    binary = (mask > 128).astype(np.uint8)
    if binary.sum() < min_area:
        return None
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def restrict_casing_window_to_bbox(
    casing_mask: np.ndarray,
    window_mask: np.ndarray,
    opening_mask: np.ndarray,
    image_bgr: np.ndarray,
    *,
    padding_pct: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """[RULE 2] Casing + window CHỈ được tồn tại bên trong bbox(opening) + padding.

    bbox = bounding box của opening mask + padding (2.5% width default).
    Pixel ngoài bbox → zero-out.

    Args:
        casing_mask, window_mask: uint8 0/255.
        opening_mask: reference bbox (window ∪ door ∪ sky).
        image_bgr: shape.
        padding_pct: padding ngoài bbox (theo % width).

    Returns:
        (casing_restricted, window_restricted)
    """
    h, w = image_bgr.shape[:2]
    bbox = _bbox_of_mask(opening_mask, min_area=200)
    if bbox is None:
        # Không có opening → không restrict gì cả (để VLM/SAM logic khác handle)
        return casing_mask, window_mask

    y0, x0, y1, x1 = bbox
    pad = max(20, int(w * padding_pct))
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad)
    x1 = min(w, x1 + pad)

    # Build ROI binary mask
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y0:y1, x0:x1] = 1

    cas_before = int((casing_mask > 128).sum())
    win_before = int((window_mask > 128).sum())

    casing_out = cv2.bitwise_and(casing_mask, casing_mask, mask=roi)
    window_out = cv2.bitwise_and(window_mask, window_mask, mask=roi)

    cas_removed = cas_before - int((casing_out > 128).sum())
    win_removed = win_before - int((window_out > 128).sum())

    log.info("Casing/window bbox restrict @ [%d:%d, %d:%d] padding=%dpx: "
             "casing -%d px, window -%d px",
             y0, y1, x0, x1, pad, cas_removed, win_removed)
    return casing_out, window_out


# ───────────────────────────────────────────────────────────────────
# OUTPUT PRODUCT FILTER — clean masks trước khi export
# ───────────────────────────────────────────────────────────────────

def clean_output_masks(
    masks: dict[str, np.ndarray],
    image_bgr: np.ndarray,
    ade_argmax_id: np.ndarray | None,
    *,
    ceiling_close_pct: float = 0.02,
    casing_max_dist_pct: float = 0.015,
    baseboard_max_dist_to_wall_pct: float = 0.01,
    ceiling_max_height_pct: float = 0.45,
    bbox_padding_pct: float = 0.025,
) -> dict[str, np.ndarray]:
    """Final cleanup — 4 nguyên tắc v0.3.2 + 3 ROI rules v0.3.3."""
    out = dict(masks)

    # ━━ ROI HARD-CAP TRƯỚC TIÊN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [RULE 1] Ceiling y ≤ 45% height (no sofa, no table, no floor)
    if "ceiling" in out:
        out["ceiling"] = enforce_ceiling_roi(
            out["ceiling"], image_bgr, max_height_pct=ceiling_max_height_pct,
        )

    # [RULE 2] Casing + window within bbox(opening) + 2.5% padding
    if "casing" in out and "window" in out and "opening" in out:
        out["casing"], out["window"] = restrict_casing_window_to_bbox(
            out["casing"], out["window"], out["opening"], image_bgr,
            padding_pct=bbox_padding_pct,
        )

    # ━━ EXCLUSION MASK (49 ADE furniture classes) ━━━━━━━━━━━━━━━━━━━
    exclusion = build_furniture_exclusion_mask(ade_argmax_id)
    out = apply_exclusion_to_strip_masks(
        out, exclusion,
        target_classes=("casing", "baseboard", "crown", "ceiling", "window"),
    )

    # ━━ DYNAMIC CEILING CLOSE + CC NOISE FILTER ━━━━━━━━━━━━━━━━━━━━━
    if "ceiling" in out:
        out["ceiling"] = dynamic_close_ceiling(
            image_bgr, out["ceiling"], width_pct=ceiling_close_pct,
        )
        # Re-apply ROI cap sau morph (close có thể đẩy mask xuống dưới cap_y)
        out["ceiling"] = enforce_ceiling_roi(
            out["ceiling"], image_bgr, max_height_pct=ceiling_max_height_pct,
        )

    # ━━ DISTANCE-TRANSFORM STRIP CONSTRAINT ━━━━━━━━━━━━━━━━━━━━━━━━━
    if "casing" in out and "opening" in out:
        out["casing"] = constrain_strip_to_opening(
            out["casing"], out["opening"], image_bgr,
            max_pct=casing_max_dist_pct,
        )
    if "baseboard" in out and "floor" in out:
        out["baseboard"] = constrain_strip_to_opening(
            out["baseboard"], out["floor"], image_bgr,
            max_pct=baseboard_max_dist_to_wall_pct,
        )

    return out


# ───────────────────────────────────────────────────────────────────
# MAIN ENTRY: resolve_all_overlaps
# ───────────────────────────────────────────────────────────────────

def resolve_all_overlaps(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    ade_argmax_id: np.ndarray | None = None,
    apply_sobel: bool = True,
    apply_position: bool = True,
    apply_casing_subtract: bool = True,
    apply_final_cleanup: bool = True,
    lamp_mask: np.ndarray | None = None,  # kept for backwards compat (unused)
    light_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Master orchestrator v0.3.2 — clean structural masks.

    Pipeline:
        1. Sobel ceiling∩wall (split overlap)
        2. Position ceiling∩floor
        3. Subtract casing + mullions from opening (preserve frame structure)
        4. FINAL CLEANUP — 4 nguyên tắc:
           a. Furniture exclusion mask zero-out
           b. Dynamic ceiling close (kernel = 2% width) + CC noise filter
           c. Casing distance transform constraint (≤1.5% width from opening)
           d. Baseboard distance transform constraint (≤1% width from floor)
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

    # Step 3: Subtract casing + mullions from opening
    if apply_casing_subtract and "opening" in out and "casing" in out:
        out["opening"] = subtract_casing_from_opening(
            out["opening"], out["casing"],
            image_bgr=image_bgr,
            dilate_casing_px=3, detect_mullions=True,
        )

    # Step 4: FINAL CLEANUP (4 nguyên tắc)
    if apply_final_cleanup:
        out = clean_output_masks(out, image_bgr, ade_argmax_id)

    return out
