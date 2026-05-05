"""Image quality metrics — PSNR, SSIM, MAE, watermark coverage check.

Dùng cho QA / regression testing sau khi inpaint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class QualityReport:
    psnr: float
    ssim: float
    mae: float
    max_diff: int
    different_pixels_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB). >40dB = mắt khó phân biệt."""
    a32 = a.astype(np.float64)
    b32 = b.astype(np.float64)
    mse = float(np.mean((a32 - b32) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM rút gọn (luminance-only, không cửa sổ trượt — đủ cho regression)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) if b.ndim == 3 else b.astype(np.float64)
    mu_a, mu_b = ga.mean(), gb.mean()
    sa, sb = ga.var(), gb.var()
    cov = ((ga - mu_a) * (gb - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    )


def compare(a: np.ndarray, b: np.ndarray) -> QualityReport:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = cv2.absdiff(a, b)
    return QualityReport(
        psnr=round(psnr(a, b), 3),
        ssim=round(ssim(a, b), 5),
        mae=round(float(np.mean(diff)), 4),
        max_diff=int(diff.max()),
        different_pixels_ratio=round(float((diff.sum(-1) > 0).mean()), 5)
        if a.ndim == 3 else round(float((diff > 0).mean()), 5),
    )


def compare_files(original: str | Path, restored: str | Path) -> QualityReport:
    """Read 2 ảnh + so sánh. Tự resize nếu mismatch nhẹ (≤2px)."""
    from .utils import read_image
    a = read_image(original)
    b = read_image(restored)
    if a.shape != b.shape:
        # accept tiny resize artifacts from JPEG decoder
        if abs(a.shape[0] - b.shape[0]) <= 2 and abs(a.shape[1] - b.shape[1]) <= 2:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        else:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return compare(a, b)


def watermark_residual(
    cleaned: np.ndarray,
    mask: np.ndarray,
    *,
    bright_threshold: int = 240,
    dark_threshold: int = 15,
) -> dict:
    """Đếm pixel có thể là 'tàn dư' watermark trong vùng đã inpaint.

    Sử dụng heuristic: pixel quá sáng hoặc quá tối trong vùng mask thường
    là watermark sót lại."""
    if cleaned.ndim == 3:
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    else:
        gray = cleaned
    region = mask > 0
    n_in_region = int(region.sum())
    if n_in_region == 0:
        return {"checked": 0, "bright_residual": 0, "dark_residual": 0, "ratio": 0.0}
    bright = int(((gray > bright_threshold) & region).sum())
    dark = int(((gray < dark_threshold) & region).sum())
    return {
        "checked": n_in_region,
        "bright_residual": bright,
        "dark_residual": dark,
        "ratio": round((bright + dark) / n_in_region, 5),
    }
