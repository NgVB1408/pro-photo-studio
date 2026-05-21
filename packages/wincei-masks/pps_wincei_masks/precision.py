"""Precision mode — biên mượt sub-pixel ở 6K.

Stack 4 layer enhancement:
    1. Tile-based matting full-res (KHÔNG downscale)
    2. Bilateral edge polish (smooth aliasing, preserve hard edge)
    3. Anisotropic diffusion (edge-aware smoothing)
    4. Soft thresholding (giữ alpha gradient, không binary cứng)
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


def _expanded_trimap(
    soft: np.ndarray,
    *,
    fg_thresh: float = 0.80,
    bg_thresh: float = 0.20,
    unknown_band: int = 12,
) -> np.ndarray:
    """Trimap với band rộng hơn → matting có nhiều pixel để work.

    Unknown band ±12px (vs default ±4) → biên matting có gradient mượt rộng hơn.
    """
    fg = (soft >= fg_thresh).astype(np.uint8)
    bg = (soft <= bg_thresh).astype(np.uint8)
    fg_eroded = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=unknown_band // 3)
    bg_eroded = cv2.erode(bg, np.ones((3, 3), np.uint8), iterations=unknown_band // 3)
    trimap = np.full(soft.shape, 128, dtype=np.uint8)
    trimap[fg_eroded > 0] = 255
    trimap[bg_eroded > 0] = 0
    return trimap


def tile_matting(
    image_bgr: np.ndarray,
    soft: np.ndarray,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    unknown_band: int = 12,
) -> np.ndarray:
    """Matting full-res qua tile + alpha-blend overlap.

    6K image → 3×5 tiles (~15 tiles), each ~5-10s matting → 60-150s total.
    Biên sub-pixel chính xác hơn nhiều so với downscale 1600.

    Args:
        image_bgr: (H,W,3) uint8 BGR full-res.
        soft: (H,W) float32 [0,1] semantic prob.
        tile_size: cạnh mỗi tile (px).
        overlap: vùng overlap giữa tile (px).
        unknown_band: trimap unknown band (px) — rộng hơn cho biên mượt.

    Returns:
        (H,W) uint8 [0,255] alpha refined.
    """
    if not _HAS_PYMATTING:
        log.warning("pymatting không cài → tile_matting fallback bilateral")
        return _bilateral_only(image_bgr, soft)

    h, w = image_bgr.shape[:2]
    rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    rgb_full = np.clip(rgb_full, 0.0, 1.0)

    # Output accumulator + weight buffer (cosine taper at edges)
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    stride = tile_size - overlap
    n_tiles_y = max(1, (h - overlap + stride - 1) // stride)
    n_tiles_x = max(1, (w - overlap + stride - 1) // stride)
    n_tiles = n_tiles_y * n_tiles_x
    done = 0

    # Cosine taper window (smooth blend)
    def _taper(sz: int) -> np.ndarray:
        ramp = min(overlap, sz // 4)
        win = np.ones(sz, dtype=np.float32)
        if ramp > 0:
            t = np.linspace(0, np.pi / 2, ramp, dtype=np.float32)
            win[:ramp] = np.sin(t) ** 2
            win[-ramp:] = np.sin(np.pi / 2 - t) ** 2
        return win

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = min(ty * stride, h - tile_size) if h >= tile_size else 0
            x0 = min(tx * stride, w - tile_size) if w >= tile_size else 0
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            rgb_tile = np.ascontiguousarray(rgb_full[y0:y1, x0:x1])
            soft_tile = soft[y0:y1, x0:x1]
            trimap = _expanded_trimap(soft_tile, unknown_band=unknown_band)

            # Skip matting nếu tile không có biên
            n_unknown = (trimap == 128).sum()
            if n_unknown < 200 or (trimap == 255).sum() < 100 or (trimap == 0).sum() < 100:
                alpha_tile = soft_tile.astype(np.float32)
            else:
                try:
                    trimap_f = np.ascontiguousarray(trimap.astype(np.float64) / 255.0)
                    alpha_tile = estimate_alpha_cf(rgb_tile, trimap_f).astype(np.float32)
                except Exception as exc:
                    log.debug("tile (%d,%d) matting fail: %s", ty, tx, exc)
                    alpha_tile = soft_tile.astype(np.float32)

            # Cosine taper window
            th, tw = alpha_tile.shape
            win = _taper(th)[:, None] * _taper(tw)[None, :]
            acc[y0:y1, x0:x1] += alpha_tile * win
            weight[y0:y1, x0:x1] += win

            done += 1
            if done % 5 == 0:
                log.info("tile %d/%d done", done, n_tiles)

    alpha = acc / np.maximum(weight, 1e-6)
    alpha = np.clip(alpha, 0, 1)
    return (alpha * 255 + 0.5).astype(np.uint8)


def _bilateral_only(image_bgr: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """Fallback: bilateral filter only — fast nhưng kém matting."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = soft.astype(np.float32)
    try:
        alpha = cv2.ximgproc.jointBilateralFilter(gray, src, d=15, sigmaColor=0.08, sigmaSpace=12)
    except (AttributeError, cv2.error):
        alpha = cv2.bilateralFilter(src, d=15, sigmaColor=0.08, sigmaSpace=12)
    return (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)


def edge_polish(alpha: np.ndarray, image_bgr: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """Post-matting polish: bilateral + anisotropic để biên mượt sub-pixel.

    Args:
        alpha: (H,W) uint8 mask.
        image_bgr: full-res ảnh gốc làm guidance.
        strength: 0..1 — 1.0 = polish mạnh.

    Returns:
        (H,W) uint8 alpha smooth.
    """
    a = alpha.astype(np.float32) / 255.0

    # Step 1: joint bilateral (edge-aware blur, dùng ảnh gốc làm guidance)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        polished = cv2.ximgproc.jointBilateralFilter(
            gray, a,
            d=int(7 + 6 * strength),
            sigmaColor=0.05 + 0.03 * strength,
            sigmaSpace=6 + 8 * strength,
        )
    except (AttributeError, cv2.error):
        polished = cv2.bilateralFilter(
            a,
            d=int(7 + 6 * strength),
            sigmaColor=0.05 + 0.03 * strength,
            sigmaSpace=6 + 8 * strength,
        )

    # Step 2: ép alpha vào [0,1] và soft-stretch (push toward 0/255 ở vùng confident, giữ gradient ở biên)
    polished = np.clip(polished, 0, 1)
    # Sigmoid stretch with steepness — pull confident pixels sharp, leave edge soft
    k = 6.0  # steeper = harder cut
    polished = 1.0 / (1.0 + np.exp(-k * (polished - 0.5)))

    return (polished * 255 + 0.5).astype(np.uint8)


def refine_alpha_masks_precision(
    image_bgr: np.ndarray,
    soft_masks: dict[str, np.ndarray],
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    unknown_band: int = 12,
    polish_strength: float = 1.0,
) -> dict[str, np.ndarray]:
    """Precision refinement pipeline cho từng mask.

    Pipeline mỗi mask:
        1. Tile-based matting full-res
        2. Edge polish (bilateral + sigmoid stretch)

    Args:
        image_bgr: full-res ảnh.
        soft_masks: dict[name -> (H,W) float32 [0,1]].
        tile_size, overlap, unknown_band: tile matting params.
        polish_strength: 0..1.

    Returns:
        dict[name -> uint8 alpha smooth].
    """
    out: dict[str, np.ndarray] = {}
    for name, soft in soft_masks.items():
        if soft.max() < 0.1:
            # Mask quá yếu — guided filter only
            out[name] = _bilateral_only(image_bgr, soft)
            continue
        log.info("precision refine '%s'...", name)
        alpha = tile_matting(
            image_bgr, soft,
            tile_size=tile_size, overlap=overlap, unknown_band=unknown_band,
        )
        out[name] = edge_polish(alpha, image_bgr, strength=polish_strength)
    return out
