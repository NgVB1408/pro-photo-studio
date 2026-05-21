"""Exposure fusion engine — Mertens (LDR space, không cần CRF).

Mertens 2007 paper: fuse N exposures qua 3 quality measures:
    - Contrast (Laplacian magnitude)
    - Saturation (RGB std)
    - Well-exposedness (Gaussian around mid-grey)

Output: 1 LDR ảnh same resolution, 0..1 float32 → uint8.
Không cần tonemap, không cần biết EV thực.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def align_brackets(
    images: list[np.ndarray],
    *,
    enabled: bool = True,
    max_bits: int = 6,
    exclude_range: int = 4,
    cut: bool = True,
) -> list[np.ndarray]:
    """Align handheld bracket bằng MTB (Median Threshold Bitmap).

    AlignMTB không cần feature matching, robust với ảnh exposure khác nhau.
    Auto-shift to compensate handheld jitter. Đầu vào tripod → no-op.

    Args:
        images: list BGR uint8 cùng shape.
        enabled: tắt nếu chắc chắn tripod (tiết kiệm 0.5-2s/group).
        max_bits, exclude_range, cut: AlignMTB params (defaults work for most).

    Returns:
        Aligned list (same length).
    """
    if not enabled or len(images) < 2:
        return images
    aligner = cv2.createAlignMTB(max_bits=max_bits, exclude_range=exclude_range, cut=cut)
    aligned = list(images)
    aligner.process(images, aligned)
    return aligned


def fuse_mertens(
    images: list[np.ndarray],
    *,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Exposure fusion (Mertens) trên N ảnh BGR uint8.

    Args:
        images: list BGR uint8 cùng shape, exposure ordering tự do.
        contrast/saturation/exposure_weight: 3 quality measures.
            Defaults (1,1,1) — bias đều, output natural.
            Bias đẩy outdoor recovery: lower exposure_weight, raise contrast.
        gamma: post-fusion gamma correction (1.0 = none).

    Returns:
        BGR uint8 same shape, exposure-fused.
    """
    if not images:
        raise ValueError("fuse_mertens: empty image list")
    if len(images) == 1:
        log.warning("fuse_mertens called with 1 image — passthrough")
        return images[0]

    h, w = images[0].shape[:2]
    for i, im in enumerate(images[1:], 1):
        if im.shape[:2] != (h, w):
            raise ValueError(
                f"shape mismatch: image[0]={h}x{w} vs image[{i}]={im.shape[0]}x{im.shape[1]}"
            )

    merger = cv2.createMergeMertens(
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
    )
    fused = merger.process(images)  # float32, can have values <0 or >1
    fused = np.clip(fused, 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-3:
        fused = np.power(fused, 1.0 / gamma)

    return (fused * 255.0 + 0.5).astype(np.uint8)


def fuse_bracket_files(
    paths: list[Path],
    *,
    align: bool = True,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Convenience: đọc N path → align → Mertens fuse → BGR uint8."""
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Không đọc được: {p}")
        images.append(img)
    images = align_brackets(images, enabled=align)
    return fuse_mertens(
        images,
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
        gamma=gamma,
    )
