"""AI Inpainting via LaMa (simple-lama-inpainting wrapper).

Khác Telea/NS classical:
- LaMa redraw texture có context (không lặp pattern)
- Dùng được cho vùng lớn (>200×200px)
- Chuyển từ classical → AI cho:
  - Photographer Removal trong gương
  - TV Blackout (xoá UI/nội dung TV)
  - Watermark lớn

CPU: ~10-30s cho ảnh 4K. GPU CUDA: ~1-3s.

Lazy-loaded — không import torch nếu không gọi.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LAMA_INSTANCE: Any = None
_LOAD_LOCK = threading.Lock()


def _get_lama():
    """Lazy-load LaMa model (download big-lama.pt lần đầu)."""
    global _LAMA_INSTANCE
    if _LAMA_INSTANCE is not None:
        return _LAMA_INSTANCE
    with _LOAD_LOCK:
        if _LAMA_INSTANCE is not None:
            return _LAMA_INSTANCE
        from simple_lama_inpainting import SimpleLama
        _LAMA_INSTANCE = SimpleLama()
        logger.info("LaMa loaded")
        return _LAMA_INSTANCE


def inpaint_ai(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    dilate: int = 3,
) -> np.ndarray:
    """Inpaint qua LaMa.

    Args:
        image: BGR uint8 [H,W,3].
        mask: uint8 [H,W], pixel >0 = vùng cần xoá.
        dilate: nở mask trước khi inpaint để cover edge.

    Returns:
        BGR uint8, vùng mask đã được redraw.
    """
    if image.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("image + mask phải uint8")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image vs mask khác size: {image.shape[:2]} vs {mask.shape[:2]}")
    if not np.any(mask):
        return image.copy()

    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate*2+1, dilate*2+1))
        mask = cv2.dilate(mask, kernel)

    from PIL import Image as PILImage
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    pil_mask = PILImage.fromarray(mask)

    lama = _get_lama()
    out_pil = lama(pil_img, pil_mask)
    out_rgb = np.array(out_pil)
    if out_rgb.ndim == 3 and out_rgb.shape[2] == 3:
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    else:
        out_bgr = out_rgb
    return out_bgr


def inpaint_smart(
    image: np.ndarray, mask: np.ndarray, *, force_classical: bool = False,
) -> np.ndarray:
    """Tự chọn classical/AI dựa trên kích thước mask."""
    H, W = image.shape[:2]
    mask_pixels = int((mask > 0).sum())
    rel = mask_pixels / max(H * W, 1)

    if force_classical or rel < 0.001:
        # Mask nhỏ < 0.1% pixel — Telea fast + đủ chất lượng
        from .inpaint import inpaint_opencv
        return inpaint_opencv(image, mask, method="telea", radius=3)

    try:
        return inpaint_ai(image, mask, dilate=3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LaMa fail (%s), fallback classical NS", exc)
        from .inpaint import inpaint_opencv
        return inpaint_opencv(image, mask, method="ns", radius=5)
