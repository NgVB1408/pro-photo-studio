"""Local adjustments — Lightroom radial / graduated filter equivalent.

3 loại mask + 1 adjustment engine:

- ``radial_mask(h, w, cx, cy, rx, ry, feather)`` — elliptical mask cho
  vignette/dodge spot
- ``graduated_mask(h, w, x0, y0, x1, y1, feather)`` — linear gradient mask
  (Lightroom Graduated Filter — phổ biến cho darken sky / brighten foreground)
- ``brush_mask(h, w, strokes, brush_size, feather)`` — mask vẽ tay (Brush tool)

Adjustment engine:
- ``apply_local(img, mask, exposure/temp/tint/saturation/clarity/...)`` — áp
  bất kỳ tone params nào ở vùng có mask, blend với original

Tất cả mask trả float32 [0..1] (giống Lightroom feather).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Mask generators
# =====================================================================

def radial_mask(
    h: int, w: int,
    *,
    cx_frac: float = 0.5, cy_frac: float = 0.5,
    rx_frac: float = 0.4, ry_frac: float = 0.4,
    feather: float = 0.5,
    inverted: bool = False,
) -> np.ndarray:
    """Elliptical radial mask. Trả float32 [0..1].

    inverted=False: trong = 1 (effect áp vùng giữa)
    inverted=True : ngoài = 1 (effect áp vùng rìa — vignette)

    feather: 0..1, mức làm mềm rìa. 0 = hard edge, 1 = max feather.
    """
    cx, cy = int(cx_frac * w), int(cy_frac * h)
    rx, ry = max(1, int(rx_frac * w)), max(1, int(ry_frac * h))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    norm_dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # 1 trong, 0 ngoài (smooth)
    f = max(0.05, min(2.0, 1.0 - feather))  # feather=0 → f=1 (sharp), feather=1 → f=0 (very soft)
    mask = np.clip(1.0 - norm_dist ** (1.0 / max(f, 0.05)), 0, 1)
    # Smooth thêm Gaussian
    sigma = max(1.0, feather * min(rx, ry) * 0.5)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    if inverted:
        mask = 1.0 - mask
    return mask.astype(np.float32)


def graduated_mask(
    h: int, w: int,
    *,
    x0_frac: float = 0.5, y0_frac: float = 0.0,
    x1_frac: float = 0.5, y1_frac: float = 0.5,
    feather: float = 0.3,
) -> np.ndarray:
    """Linear gradient mask (Lightroom Graduated Filter).

    Mask = 1 ở line (x0,y0) → 0 ở line (x1,y1). Vuông góc với line này là
    đẳng giá trị. Ví dụ: y0=0, y1=0.5 → mask = 1 ở top, fade về 0 ở giữa
    ảnh (dùng để darken sky).

    feather: 0..1, mức smooth ở rìa transition.
    """
    x0, y0 = x0_frac * w, y0_frac * h
    x1, y1 = x1_frac * w, y1_frac * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Vector từ (x0,y0) đến (x1,y1)
    dx, dy = x1 - x0, y1 - y0
    length = max(np.hypot(dx, dy), 1e-3)
    nx, ny = dx / length, dy / length
    # Projection of (xx,yy) - (x0,y0) onto normal vector
    proj = (xx - x0) * nx + (yy - y0) * ny
    t = proj / length  # 0 ở (x0,y0), 1 ở (x1,y1)
    # Mask = 1 ở t≤0, 0 ở t≥1, smooth ở giữa
    mask = np.clip(1.0 - t, 0, 1)
    # Apply feather ở rìa transition (smoothstep)
    if feather > 0:
        sigma = max(1.0, feather * length * 0.3)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    return mask.astype(np.float32)


def brush_mask(
    h: int, w: int,
    strokes: list[dict],
    *,
    feather: float = 0.5,
) -> np.ndarray:
    """Mask vẽ tay từ list strokes.

    strokes: list dict, mỗi dict:
      {"x": int, "y": int, "size": int, "flow": float (0..1)}
      Hoặc với line: {"x0":, "y0":, "x1":, "y1":, "size":, "flow":}
    """
    mask = np.zeros((h, w), dtype=np.float32)
    for s in strokes:
        flow = float(s.get("flow", 1.0))
        size = int(s.get("size", 30))
        if "x0" in s:
            cv2.line(
                mask, (int(s["x0"]), int(s["y0"])),
                (int(s["x1"]), int(s["y1"])),
                color=flow, thickness=size,
            )
        else:
            cv2.circle(
                mask, (int(s["x"]), int(s["y"])),
                radius=size // 2, color=flow, thickness=-1,
            )
    if feather > 0:
        sigma = max(1.0, feather * 10.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    return np.clip(mask, 0, 1).astype(np.float32)


# =====================================================================
# Local adjustment engine
# =====================================================================

@dataclass
class LocalParams:
    """Adjust params trong vùng mask. Range -1..+1 (như Lightroom)."""
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    saturation: float = 0.0
    temp_shift: float = 0.0
    tint_shift: float = 0.0
    clarity: float = 0.0
    sharpness: float = 0.0


def apply_local(
    img: np.ndarray,
    mask: np.ndarray,
    params: LocalParams,
) -> np.ndarray:
    """Áp params trong vùng mask. mask: float32 [0..1].

    Mặt độ áp = mask * effect_strength. Pixel ngoài mask giữ nguyên.
    """
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
        if mask.max() > 1.5:
            mask = mask / 255.0
    mask = np.clip(mask, 0, 1)

    # Tạo ảnh "fully adjusted" (áp 100% effect lên toàn ảnh)
    full = _apply_global(img, params)

    # Blend: out = orig * (1 - mask) + full * mask
    mask_3d = mask[..., None]
    out = img.astype(np.float32) * (1 - mask_3d) + full.astype(np.float32) * mask_3d
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_global(img: np.ndarray, params: LocalParams) -> np.ndarray:
    """Áp params toàn ảnh — helper cho apply_local."""
    out = img.copy()

    # WB
    if abs(params.temp_shift) > 1e-3 or abs(params.tint_shift) > 1e-3:
        from .tone import temperature_tint
        out = temperature_tint(
            out, temp_shift=params.temp_shift, tint_shift=params.tint_shift,
        )

    # Tone curve
    needs_tone = any(abs(getattr(params, k)) > 1e-3 for k in (
        "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    ))
    if needs_tone:
        from .tone import ToneParams, parametric_tone
        tp = ToneParams(
            exposure=params.exposure,
            contrast=params.contrast,
            highlights=params.highlights,
            lights=params.highlights * 0.5,  # cascade nhẹ
            darks=params.shadows * 0.5,
            shadows=params.shadows,
            whites=params.whites,
            blacks=params.blacks,
        )
        out = parametric_tone(out, tp)

    # Saturation
    if abs(params.saturation) > 1e-3:
        sat = float(np.clip(params.saturation, -1.0, 1.0))
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Clarity
    if abs(params.clarity) > 1e-3:
        from .tone import clarity
        out = clarity(out, amount=params.clarity)

    # Sharpness
    if abs(params.sharpness) > 1e-3:
        amount = float(np.clip(params.sharpness, -1.0, 1.0))
        if amount > 0:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
            out = cv2.addWeighted(out, 1 + amount, blurred, -amount, 0)
        else:
            # Negative = soften (Gaussian blur)
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=abs(amount) * 2.0)

    return out


# =====================================================================
# Pre-built common adjustments
# =====================================================================

def vignette(
    img: np.ndarray, *,
    amount: float = -0.3,           # -1..0 dark vignette / 0..1 bright
    midpoint: float = 0.5,           # 0..1 size of vignette
    feather: float = 0.5,            # 0..1 edge softness
    roundness: float = 0.0,          # -1 (oval H) .. +1 (oval V), 0 = circle
) -> np.ndarray:
    """Lightroom Effects → Vignette."""
    h, w = img.shape[:2]
    rx_frac = 0.5 + midpoint * 0.4
    ry_frac = 0.5 + midpoint * 0.4
    if roundness > 0:
        ry_frac *= 1.0 - roundness * 0.3
    elif roundness < 0:
        rx_frac *= 1.0 + roundness * 0.3
    mask = radial_mask(
        h, w, rx_frac=rx_frac, ry_frac=ry_frac,
        feather=feather, inverted=True,  # rìa = 1
    )
    params = LocalParams(exposure=amount * 1.0, contrast=amount * 0.3)
    return apply_local(img, mask, params)


def darken_sky_grad(
    img: np.ndarray, *,
    horizon_y_frac: float = 0.45,
    amount: float = -0.4,
    feather: float = 0.4,
    contrast_boost: float = 0.2,
) -> np.ndarray:
    """Graduated filter darkens sky above horizon."""
    h, w = img.shape[:2]
    mask = graduated_mask(
        h, w,
        x0_frac=0.5, y0_frac=horizon_y_frac - feather * 0.3,
        x1_frac=0.5, y1_frac=horizon_y_frac + feather * 0.3,
        feather=feather,
    )
    params = LocalParams(
        exposure=amount,
        contrast=contrast_boost,
        clarity=0.15,
    )
    return apply_local(img, mask, params)


def brighten_foreground_grad(
    img: np.ndarray, *,
    horizon_y_frac: float = 0.55,
    amount: float = 0.25,
    feather: float = 0.4,
) -> np.ndarray:
    """Inverse graduated — brighten dưới horizon."""
    h, w = img.shape[:2]
    mask = graduated_mask(
        h, w,
        x0_frac=0.5, y0_frac=horizon_y_frac + feather * 0.3,
        x1_frac=0.5, y1_frac=horizon_y_frac - feather * 0.3,
        feather=feather,
    )
    params = LocalParams(exposure=amount, shadows=amount * 0.6)
    return apply_local(img, mask, params)
