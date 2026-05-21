"""Alpha refinement — biên mask đẹp sub-pixel.

2 backends:
    [A] PyMatting closed-form (chất tốt nhất, ~3-8s/mask 6K)
    [B] OpenCV guided filter (fallback, nhanh hơn 10x, biên ổn nhưng cứng hơn)

Auto-fallback nếu pymatting không cài.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    from pymatting import estimate_alpha_cf
    _HAS_PYMATTING = True
except ImportError:
    _HAS_PYMATTING = False
    log.info("pymatting không cài → fallback guided filter")


def _build_trimap(
    soft: np.ndarray,
    *,
    fg_thresh: float = 0.85,
    bg_thresh: float = 0.15,
    unknown_dilate: int = 4,
) -> np.ndarray:
    """Trimap 3-state từ soft prob:
        255 = chắc chắn FG, 0 = chắc chắn BG, 128 = unknown (cần matting).
    """
    fg = (soft >= fg_thresh).astype(np.uint8)
    bg = (soft <= bg_thresh).astype(np.uint8)
    unknown = ((soft > bg_thresh) & (soft < fg_thresh)).astype(np.uint8)

    # Dilate unknown band gần biên
    if unknown_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (unknown_dilate * 2 + 1,) * 2)
        edge_band = cv2.dilate(unknown, k, iterations=1)
        unknown = edge_band

    trimap = np.full(soft.shape, 128, dtype=np.uint8)
    trimap[fg.astype(bool) & ~unknown.astype(bool)] = 255
    trimap[bg.astype(bool) & ~unknown.astype(bool)] = 0
    return trimap


def refine_alpha_masks(
    image_bgr: np.ndarray,
    soft_masks: dict[str, np.ndarray],
    *,
    use_matting: bool = True,
    matting_max_side: int = 1600,
    guide_radius: int = 8,
    guide_eps: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Refine soft masks → alpha 0..255 với biên đẹp.

    Args:
        image_bgr: full-res ảnh gốc.
        soft_masks: dict[name -> (H,W) float32 0..1]
        use_matting: True = PyMatting closed-form (slower, đẹp hơn);
                     False = guided filter (faster).
        matting_max_side: matting tốn O(N²) RAM → downscale rồi upscale.
        guide_radius, guide_eps: params cho guided filter fallback.

    Returns:
        dict[name -> (H,W) uint8 0..255 alpha]
    """
    out: dict[str, np.ndarray] = {}
    h, w = image_bgr.shape[:2]
    rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    rgb_full = np.clip(rgb_full, 0.0, 1.0)

    if use_matting and _HAS_PYMATTING:
        scale = min(1.0, matting_max_side / max(h, w))
        if scale < 1.0:
            sw, sh = int(w * scale), int(h * scale)
            rgb_small = cv2.resize(rgb_full, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb_full
            sw, sh = w, h
        # pymatting needs float64 contiguous
        rgb_small = np.ascontiguousarray(rgb_small, dtype=np.float64)

        for name, soft in soft_masks.items():
            try:
                soft_small = (
                    cv2.resize(soft, (sw, sh), interpolation=cv2.INTER_AREA)
                    if scale < 1.0 else soft
                )
                trimap = _build_trimap(soft_small)
                trimap_f = np.ascontiguousarray(trimap.astype(np.float64) / 255.0)
                # Skip matting nếu trimap không có FG đủ — fallback luôn
                if (trimap_f >= 0.9).sum() < 100 or (trimap_f <= 0.1).sum() < 100:
                    raise ValueError("trimap insufficient FG/BG anchor")
                alpha_small = estimate_alpha_cf(rgb_small, trimap_f)
                if scale < 1.0:
                    alpha = cv2.resize(alpha_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    alpha = alpha_small
                out[name] = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
            except Exception as exc:
                log.warning("matting fail '%s' (%s) → guided filter fallback", name, exc)
                out[name] = _guided_refine(image_bgr, soft, guide_radius, guide_eps)
        return out

    # Fallback: guided filter
    for name, soft in soft_masks.items():
        out[name] = _guided_refine(image_bgr, soft, guide_radius, guide_eps)
    return out


def _guided_refine(image_bgr: np.ndarray, soft: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter refinement — nhanh, biên smooth nhưng không matting thực sự."""
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = soft.astype(np.float32)
    try:
        alpha = cv2.ximgproc.guidedFilter(guide, src, radius=radius, eps=eps)
    except (AttributeError, cv2.error):
        # ximgproc không có → bilateral fallback (cứng hơn)
        alpha = cv2.bilateralFilter(src, d=9, sigmaColor=0.1, sigmaSpace=8)
    return (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
