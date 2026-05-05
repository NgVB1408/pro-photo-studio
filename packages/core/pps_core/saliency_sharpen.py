"""Saliency-based selective sharpening — sharpen subject, leave background soft.

Pro photographer kỹ thuật:
- Subject (sản phẩm, người, thuộc về scene chính) cần sharpening mạnh
- Background (tường xa, sky, foliage) nên giữ soft để tăng depth perception
- Áp uniform sharpen → ảnh nhìn "crispy" nhưng artificial

Pipeline:
1. Generate saliency mask qua spectral residual (OpenCV built-in) hoặc gradient-based
2. Smooth mask + dilate để cover toàn bộ subject
3. Sharpen full image
4. Blend: result = sharp * mask + smooth * (1 - mask)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _gradient_saliency(img: np.ndarray) -> np.ndarray:
    """Fallback saliency: gradient magnitude + box smooth.

    Subject thường có high gradient (edge, texture detail) cao hơn background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    # Smooth thành saliency map mượt
    h, w = gray.shape
    box_k = max(15, min(h, w) // 80)
    grad_smooth = cv2.boxFilter(grad, cv2.CV_32F, (box_k, box_k))
    # Normalize 0..1
    g_min, g_max = grad_smooth.min(), grad_smooth.max()
    if g_max - g_min < 1e-6:
        return np.full_like(grad_smooth, 0.5)
    return (grad_smooth - g_min) / (g_max - g_min)


def _opencv_saliency(img: np.ndarray) -> np.ndarray | None:
    """OpenCV saliency module (StaticSaliencyFineGrained). Return None if unavailable."""
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, mask = sal.computeSaliency(img)
        if ok:
            return mask.astype(np.float32)  # 0..1
    except (AttributeError, cv2.error):
        pass
    return None


def compute_saliency(img: np.ndarray) -> np.ndarray:
    """Trả saliency map [H, W] in 0..1. Subject ~ 1, background ~ 0."""
    cv_sal = _opencv_saliency(img)
    if cv_sal is not None:
        return cv_sal
    return _gradient_saliency(img)


def saliency_sharpen(
    img: np.ndarray,
    *,
    sharp_amount: float = 0.6,
    sharp_sigma: float = 1.5,
    bg_smooth: float = 0.0,
    saliency_threshold: float = 0.35,
    saliency_blur: int = 51,
) -> np.ndarray:
    """Sharpen-on-subject + soft-on-background.

    Args:
        sharp_amount: 0..1, strength sharpening trên subject.
        sharp_sigma: unsharp Gaussian sigma.
        bg_smooth: 0..1, blur strength trên background. 0 = không blur, just less sharpen.
        saliency_threshold: 0..1, mask threshold để decide subject vs background.
        saliency_blur: smooth saliency mask để tránh hard edge giữa subject/bg.

    Returns:
        BGR uint8 với sharpening selective.
    """
    # 1. Saliency mask
    saliency = compute_saliency(img)
    # Soft threshold + blur
    mask = np.clip((saliency - saliency_threshold) / max(1e-6, 1 - saliency_threshold), 0, 1)
    k = saliency_blur if saliency_blur % 2 == 1 else saliency_blur + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = mask[..., None]  # broadcast

    # 2. Sharpen full
    blurred = cv2.GaussianBlur(img, (0, 0), sharp_sigma)
    sharp = cv2.addWeighted(img, 1 + sharp_amount, blurred, -sharp_amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    # 3. Optional bg smooth
    if bg_smooth > 0:
        bg = cv2.GaussianBlur(img, (0, 0), 1.5 + bg_smooth * 3)
        bg = bg.astype(np.float32) * (1 - bg_smooth) + img.astype(np.float32) * bg_smooth
        bg = bg.astype(np.uint8)
    else:
        bg = img

    # 4. Blend: subject = sharp, background = bg
    out = sharp.astype(np.float32) * mask + bg.astype(np.float32) * (1 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)
