"""Perfect Window v0.3.4 — GEOMETRIC BBOX CROP ONLY.

KHÔNG sobel. KHÔNG SAM. KHÔNG Canny. KHÔNG morph. KHÔNG mask manipulation.

Chỉ làm 3 thứ:
    1. Nhận bbox [ymin, xmin, ymax, xmax] (từ VLM hoặc semantic fallback)
    2. img[ymin:ymax, xmin:xmax] giữ nguyên 100% màu gốc
    3. Output BGRA: alpha=255 trong bbox, alpha=0 ngoài bbox
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PerfectWindowResult:
    rgba: np.ndarray              # (H, W, 4) uint8 BGRA
    bbox: tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
    method: str                   # "vlm_bbox" | "semantic_fallback"


def _ade_opening_bbox(
    sem_result,
    image_shape: tuple[int, int],
    *,
    soft_threshold: float = 0.45,
    min_area_px: int = 500,
) -> tuple[int, int, int, int] | None:
    """Fallback bbox = union(window, door, sky) bounding box."""
    from .semantic import ADE20K_CLASSES
    soft_w = sem_result.get_soft(ADE20K_CLASSES["window"])
    soft_d = sem_result.get_soft(ADE20K_CLASSES["door"])
    soft_s = sem_result.get_soft(ADE20K_CLASSES["sky"])
    union = np.maximum.reduce([soft_w, soft_d, soft_s])
    binary = (union >= soft_threshold).astype(np.uint8)
    if binary.sum() < min_area_px:
        return None
    ys, xs = np.where(binary > 0)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def crop_window_rgba(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """CORE: BGRA với alpha=255 trong bbox, alpha=0 ngoài bbox.

    Không can thiệp pixel màu gốc. Hình chữ nhật cứng.
    """
    h, w = image_bgr.shape[:2]
    y0, x0, y1, x1 = bbox
    # Clamp coordinates trong ảnh
    y0 = max(0, min(h - 1, y0))
    x0 = max(0, min(w - 1, x0))
    y1 = max(y0 + 1, min(h, y1))
    x1 = max(x0 + 1, min(w, x1))

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 0  # transparent toàn bộ
    bgra[y0:y1, x0:x1, 3] = 255  # opaque trong bbox

    log.info("BBOX crop: [%d:%d, %d:%d] = %dx%d px (%.2f%% of image)",
             y0, y1, x0, x1, y1 - y0, x1 - x0,
             100.0 * (y1 - y0) * (x1 - x0) / (h * w))
    return bgra


def extract_perfect_window(
    image_bgr: np.ndarray,
    *,
    vlm_bbox: tuple[int, int, int, int] | None = None,
    sem_result=None,
    sam_engine=None,  # IGNORED (backwards compat)
    bbox_padding_pct: float = 0.0,  # default 0 — strict bbox
    canny_low: int = None,          # IGNORED
    canny_high: int = None,         # IGNORED
    feather_px: int = 0,            # IGNORED — no feather, hard edge
) -> PerfectWindowResult:
    """Geometric BBOX crop pipeline.

    Args:
        image_bgr: full-res BGR uint8.
        vlm_bbox: bbox từ VLM (ymin, xmin, ymax, xmax). None → semantic fallback.
        sem_result: SemanticResult (cần nếu vlm_bbox=None).
        bbox_padding_pct: padding theo % width (default 0 — strict bbox).

    Returns:
        PerfectWindowResult.
    """
    h, w = image_bgr.shape[:2]
    method = "vlm_bbox" if vlm_bbox is not None else "semantic_fallback"

    if vlm_bbox is None:
        if sem_result is None:
            raise ValueError("Cần vlm_bbox hoặc sem_result")
        bbox = _ade_opening_bbox(sem_result, (h, w))
        if bbox is None:
            raise RuntimeError("Không phát hiện window/opening trong ảnh")
    else:
        bbox = vlm_bbox

    # Optional padding (default 0 — strict)
    if bbox_padding_pct > 0:
        pad = int(w * bbox_padding_pct)
        y0, x0, y1, x1 = bbox
        bbox = (
            max(0, y0 - pad),
            max(0, x0 - pad),
            min(h, y1 + pad),
            min(w, x1 + pad),
        )

    rgba = crop_window_rgba(image_bgr, bbox)
    return PerfectWindowResult(rgba=rgba, bbox=bbox, method=method)


def extract_perfect_window_from_path(
    image_path: Path | str,
    *,
    vlm_bbox: tuple[int, int, int, int] | None = None,
    sem_result=None,
    output_png: Path | str | None = None,
    **kwargs,
) -> PerfectWindowResult:
    """Convenience: đọc file → crop → save PNG."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {image_path}")
    result = extract_perfect_window(
        img, vlm_bbox=vlm_bbox, sem_result=sem_result, **kwargs,
    )
    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        log.info("Saved: %s", output_png)
    return result


# ═══════════════════════════════════════════════════════════════════════
# ZOOM & CROP STRATEGY — cho window nhỏ <3% area / ngược sáng mạnh
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZoomCropResult:
    rgba: np.ndarray              # (H, W, 4) BGRA full-res
    window_mask: np.ndarray       # (H, W) uint8 mask full-res
    bbox: tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax) bbox VLM
    sam_score: float
    method: str                   # "vlm_zoom_sam" | "vlm_zoom_grabcut" | etc.
    clahe_applied: bool
    crop_size: tuple[int, int]    # (h, w) của crop


def zoom_crop_window(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid: int = 8,
    sam_engine=None,
    bbox_padding_pct: float = 0.05,
    feather_px: int = 3,
) -> ZoomCropResult:
    """ZOOM & CROP strategy cho window nhỏ.

    Workflow:
        1. Crop window_zone = img[bbox + padding]
        2. CLAHE LOCAL trên window_zone (làm sáng khung tối ngược sáng)
        3. SAM 2 predict_from_box trên enhanced crop
        4. Rescale mask về tọa độ full-res
        5. Output BGRA — alpha = mask (chỉ vùng cửa thật), KHÔNG can thiệp màu

    Args:
        image_bgr: full-res ảnh gốc (BGR).
        bbox: VLM bbox (ymin, xmin, ymax, xmax).
        clahe_clip_limit: CLAHE strength (3.0 = vừa mạnh, làm sáng khung tối).
        clahe_tile_grid: CLAHE tile grid (8×8 default).
        sam_engine: SAMEngine instance (None → GrabCut fallback).
        bbox_padding_pct: padding ngoài bbox % width (default 5% — rộng hơn standard).
        feather_px: feather alpha biên (giảm răng cưa).

    Returns:
        ZoomCropResult.
    """
    h, w = image_bgr.shape[:2]
    y0_raw, x0_raw, y1_raw, x1_raw = bbox

    # Step 1: Padded crop region
    pad = int(w * bbox_padding_pct)
    y0 = max(0, y0_raw - pad)
    x0 = max(0, x0_raw - pad)
    y1 = min(h, y1_raw + pad)
    x1 = min(w, x1_raw + pad)
    crop = image_bgr[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    log.info("Zoom crop bbox: [%d:%d, %d:%d] = %d×%d px", y0, y1, x0, x1, cw, ch)

    # Step 2: CLAHE local trên crop (làm sáng khung ngược sáng)
    clahe_applied = False
    if clahe_clip_limit > 0:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid, clahe_tile_grid),
        )
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        crop_enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        clahe_applied = True
        log.info("CLAHE applied on crop (clip=%.1f, grid=%d)",
                 clahe_clip_limit, clahe_tile_grid)
    else:
        crop_enhanced = crop

    # Step 3: SAM 2 predict_from_box trên enhanced crop
    crop_mask = None
    sam_score = 0.0
    method = "fallback_threshold"

    if sam_engine is not None and getattr(sam_engine, "_predictor", None) is not None:
        try:
            sam_engine.set_image(crop_enhanced)
            # Box = toàn bộ crop (vì crop đã là vùng VLM chỉ định)
            box_xyxy = np.array([0, 0, cw - 1, ch - 1])
            masks, scores, _ = sam_engine._predictor.predict(
                box=box_xyxy[None, :],
                multimask_output=False,
            )
            crop_mask = (masks[0] * 255).astype(np.uint8)
            sam_score = float(scores[0])
            method = "vlm_zoom_sam"
            log.info("SAM2 box on crop → score=%.3f", sam_score)
        except Exception as exc:
            log.warning("SAM box on crop fail: %s — GrabCut fallback", exc)

    # Fallback: GrabCut với rect = full crop
    if crop_mask is None:
        try:
            gc_mask = np.zeros((ch, cw), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect_margin = max(2, int(min(ch, cw) * 0.02))
            rect = (rect_margin, rect_margin,
                    cw - 2 * rect_margin, ch - 2 * rect_margin)
            cv2.grabCut(crop_enhanced, gc_mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)
            crop_mask = np.where(
                (gc_mask == cv2.GC_PR_FGD) | (gc_mask == cv2.GC_FGD), 255, 0
            ).astype(np.uint8)
            sam_score = 0.4
            method = "vlm_zoom_grabcut"
            log.info("GrabCut on crop → mask coverage %.1f%%",
                     (crop_mask > 128).mean() * 100)
        except cv2.error as exc:
            log.warning("GrabCut fail: %s — full-bbox alpha", exc)
            crop_mask = np.full((ch, cw), 255, dtype=np.uint8)
            method = "fallback_full_bbox"

    # Step 4: Feather alpha biên (giảm răng cưa)
    if feather_px > 0:
        crop_mask = cv2.GaussianBlur(
            crop_mask, (feather_px * 2 + 1,) * 2, 0,
        )

    # Step 5: Rescale mask về full-res tọa độ (chỉ paste vào region bbox)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = crop_mask

    # Step 6: BGRA output — KHÔNG can thiệp màu gốc
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = full_mask

    log.info("Zoom & Crop output: method=%s, mask cov %.2f%%",
             method, 100.0 * (full_mask > 128).sum() / (h * w))

    return ZoomCropResult(
        rgba=bgra,
        window_mask=full_mask,
        bbox=(y0, x0, y1, x1),
        sam_score=sam_score,
        method=method,
        clahe_applied=clahe_applied,
        crop_size=(ch, cw),
    )


def zoom_crop_window_from_path(
    image_path: Path | str,
    bbox: tuple[int, int, int, int],
    *,
    output_png: Path | str | None = None,
    **kwargs,
) -> ZoomCropResult:
    """Convenience: đọc file → zoom_crop → save PNG."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {image_path}")
    result = zoom_crop_window(img, bbox, **kwargs)
    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        log.info("Saved zoom-crop output: %s", output_png)
    return result
