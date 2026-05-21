"""DenseCRF post-processing — boundary refinement classical.

Krähenbühl & Koltun 2011 — fully-connected CRF, MAP via mean-field.
Tinh chỉnh biên dựa vào ảnh gốc (color + position). Cải biên rõ rệt
cho semantic mask vốn smooth at low-res rồi upsample.

Fallback graceful nếu pydensecrf không cài.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False
    log.info("pydensecrf không cài → skip CRF (fallback đã có)")


def crf_refine(
    image_bgr: np.ndarray,
    probs: np.ndarray,
    *,
    n_iters: int = 5,
    sxy_gaussian: float = 3.0,
    sxy_bilateral: float = 80.0,
    srgb_bilateral: float = 13.0,
    compat_gaussian: float = 3.0,
    compat_bilateral: float = 10.0,
) -> np.ndarray:
    """Refine softmax probs bằng DenseCRF.

    Args:
        image_bgr: (H,W,3) uint8.
        probs: (H,W,K) float32 softmax probabilities.
        n_iters: số lần mean-field inference.
        sxy_gaussian, sxy_bilateral, srgb_bilateral: kernel widths.
        compat_*: label compatibility (Potts-like).

    Returns:
        (H,W,K) float32 refined probs.
    """
    if not _HAS_CRF:
        return probs

    h, w, k = probs.shape
    if k < 2:
        return probs

    # CRF wants (K, H*W) softmax probs
    probs_chw = probs.transpose(2, 0, 1).copy()
    probs_chw = np.clip(probs_chw, 1e-6, 1.0)
    probs_chw /= probs_chw.sum(axis=0, keepdims=True)

    d = dcrf.DenseCRF2D(w, h, k)
    unary = unary_from_softmax(probs_chw)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    # Color-aware bilateral (rgb space — use RGB not BGR)
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=rgb, compat=compat_bilateral
    )

    Q = d.inference(n_iters)
    refined = np.array(Q).reshape(k, h, w).transpose(1, 2, 0).astype(np.float32)
    return refined
