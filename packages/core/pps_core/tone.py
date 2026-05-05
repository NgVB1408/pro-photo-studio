"""Pro-grade tone & color controls — Lightroom Develop module equivalent.

7 module:
- ``parametric_tone(img, h/l/d/s, contrast)`` — Highlights/Lights/Darks/Shadows
  (4-region parametric tone curve, KHÔNG phải CLAHE)
- ``texture(img, amount)`` — Mid-frequency contrast (Lightroom Texture)
- ``clarity(img, amount)`` — Local contrast vùng midtone (Lightroom Clarity)
- ``dehaze(img, amount)`` — Dark channel prior dehaze (atmospheric clarity)
- ``white_balance_picker(img, x, y, radius)`` — eyedropper neutral pick
- ``temperature_tint(img, kelvin_shift, tint_shift)`` — Kelvin + Tint sliders
- ``apply_tone_full(img, params)`` — full pipeline với 1 dataclass

Tất cả deterministic, CPU-only, KHÔNG cần ML.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToneParams:
    """Lightroom-style tone params (range -1.0..+1.0)."""
    # White balance
    temp_shift: float = 0.0      # -1 cool / +1 warm (≈ ±2000K)
    tint_shift: float = 0.0      # -1 green / +1 magenta
    # Tone curve regions (4-region parametric)
    highlights: float = 0.0      # giảm = pull down highlights
    lights: float = 0.0          # giảm = pull down upper midtones
    darks: float = 0.0           # tăng = lift lower midtones
    shadows: float = 0.0         # tăng = lift shadows
    # Global
    exposure: float = 0.0        # ±1 EV (-1=half, +1=double)
    contrast: float = 0.0        # ±1 S-curve strength
    blacks: float = 0.0          # ±1 black point shift
    whites: float = 0.0          # ±1 white point shift
    # Detail/texture
    texture: float = 0.0         # ±1 mid-freq contrast
    clarity: float = 0.0         # ±1 local midtone contrast
    dehaze: float = 0.0          # 0..1 dehaze strength


# =====================================================================
# 1. White Balance — Kelvin + Tint sliders, eyedropper pick
# =====================================================================

def temperature_tint(
    img: np.ndarray, *,
    temp_shift: float = 0.0,    # -1 cool .. +1 warm
    tint_shift: float = 0.0,    # -1 green .. +1 magenta
) -> np.ndarray:
    """Lightroom WB sliders.

    temp_shift: âm = giảm R/tăng B (cool), dương = tăng R/giảm B (warm).
    tint_shift: âm = tăng G (greener), dương = giảm G (magenta).
    """
    if abs(temp_shift) < 1e-3 and abs(tint_shift) < 1e-3:
        return img
    f = img.astype(np.float32)
    # Temperature: ±0.18 multiplier full range (≈ ±2000K natural)
    t = float(np.clip(temp_shift, -1.0, 1.0)) * 0.18
    if t != 0:
        f[..., 2] = f[..., 2] * (1.0 + t)      # R
        f[..., 0] = f[..., 0] * (1.0 - t * 0.85)  # B
    # Tint: ±0.12 multiplier on G
    g = float(np.clip(tint_shift, -1.0, 1.0)) * 0.12
    if g != 0:
        f[..., 1] = f[..., 1] * (1.0 - g)      # G (magenta = low G)
    return np.clip(f, 0, 255).astype(np.uint8)


def white_balance_picker(
    img: np.ndarray,
    x: int, y: int,
    *,
    radius: int = 15,
    target_neutral: float = 0.6,  # 0..1, mức gray target
) -> tuple[np.ndarray, dict]:
    """Eyedropper WB: pick 1 vùng pixel mà user coi là "trung tính" (gray/white)
    → tự tính scale per-channel để vùng đó thành gray neutral.

    Args:
        img: BGR uint8.
        (x, y): tâm pick.
        radius: bán kính patch để lấy median.
        target_neutral: mức V gray sau correction (0.6 = không quá sáng).

    Returns:
        (corrected BGR, info dict {scale_b, scale_g, scale_r, picked_color}).
    """
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius)
    y0, y1 = max(0, y - radius), min(h, y + radius)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return img.copy(), {"error": "empty patch"}
    # Median per channel
    b_med = float(np.median(patch[..., 0]))
    g_med = float(np.median(patch[..., 1]))
    r_med = float(np.median(patch[..., 2]))
    target = target_neutral * 255.0
    # Scale từng channel để patch → target
    scale_b = target / max(b_med, 1.0)
    scale_g = target / max(g_med, 1.0)
    scale_r = target / max(r_med, 1.0)
    # Soften scale để không over-correct (clamp 0.5..2.0)
    scale_b = float(np.clip(scale_b, 0.5, 2.0))
    scale_g = float(np.clip(scale_g, 0.5, 2.0))
    scale_r = float(np.clip(scale_r, 0.5, 2.0))

    f = img.astype(np.float32)
    f[..., 0] *= scale_b
    f[..., 1] *= scale_g
    f[..., 2] *= scale_r
    out = np.clip(f, 0, 255).astype(np.uint8)
    return out, {
        "picked_bgr": (round(b_med, 1), round(g_med, 1), round(r_med, 1)),
        "scale_bgr": (round(scale_b, 3), round(scale_g, 3), round(scale_r, 3)),
        "patch": (x0, y0, x1 - x0, y1 - y0),
    }


# =====================================================================
# 2. Parametric Tone Curve — Lightroom 4-region equivalent
# =====================================================================

def _build_parametric_lut(
    *,
    highlights: float = 0.0,
    lights: float = 0.0,
    darks: float = 0.0,
    shadows: float = 0.0,
    exposure: float = 0.0,
    contrast: float = 0.0,
    blacks: float = 0.0,
    whites: float = 0.0,
) -> np.ndarray:
    """Build 256-entry LUT cho parametric tone curve.

    4 region (giống Lightroom):
      - shadows : 0..0.25
      - darks   : 0.25..0.50
      - lights  : 0.50..0.75
      - highlights: 0.75..1.0
    Mỗi slider shift midpoint của region đó ±0.18 max.

    Plus:
      - exposure: shift toàn ±1 EV (±0.5 in 0..1 space)
      - contrast: S-curve sigmoid centered at 0.5
      - blacks/whites: shift endpoints
    """
    x = np.linspace(0, 1, 256, dtype=np.float32)
    y = x.copy()

    # 1. Exposure — multiplier
    if abs(exposure) > 1e-3:
        y = y * (2.0 ** (np.clip(exposure, -1.0, 1.0)))

    # 2. Black/white point shift
    if abs(blacks) > 1e-3:
        # blacks +ve = lift shadows
        y = y + np.clip(blacks, -1.0, 1.0) * 0.10 * np.clip(0.25 - y, 0, None) / 0.25
    if abs(whites) > 1e-3:
        # whites +ve = boost highlights
        y = y + np.clip(whites, -1.0, 1.0) * 0.10 * np.clip(y - 0.75, 0, None) / 0.25

    # 3. Parametric 4-region — soft anchor shifts
    region_centers = [
        (0.125, shadows),
        (0.375, darks),
        (0.625, lights),
        (0.875, highlights),
    ]
    for center, amount in region_centers:
        if abs(amount) < 1e-3:
            continue
        amt = float(np.clip(amount, -1.0, 1.0))
        # Gaussian bump centered at this region (sigma 0.15)
        bump = np.exp(-((x - center) ** 2) / (2 * 0.15 ** 2))
        y = y + amt * 0.15 * bump

    # 4. Contrast S-curve
    if abs(contrast) > 1e-3:
        c = float(np.clip(contrast, -1.0, 1.0))
        steepness = 4 + abs(c) * 8 if c > 0 else 0
        if c > 0:
            sig = 1.0 / (1.0 + np.exp(-steepness * (y - 0.5)))
            sig0 = 1.0 / (1.0 + np.exp(steepness * 0.5))
            sig1 = 1.0 / (1.0 + np.exp(-steepness * 0.5))
            y = (sig - sig0) / (sig1 - sig0)
        elif c < 0:
            # Invert S-curve: pull toward midtone (lower contrast)
            y = y * (1.0 + c * 0.3) + 0.5 * (-c * 0.3)

    y = np.clip(y, 0, 1)
    return (y * 255.0).astype(np.uint8)


def parametric_tone(img: np.ndarray, params: ToneParams) -> np.ndarray:
    """Áp parametric tone curve theo 4-region (Lightroom equivalent)."""
    lut = _build_parametric_lut(
        highlights=params.highlights,
        lights=params.lights,
        darks=params.darks,
        shadows=params.shadows,
        exposure=params.exposure,
        contrast=params.contrast,
        blacks=params.blacks,
        whites=params.whites,
    )
    # Áp LUT trên kênh L của LAB để giữ saturation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# =====================================================================
# 3. Texture / Clarity / Dehaze — Lightroom Develop module
# =====================================================================

def texture(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Mid-frequency contrast boost (Lightroom Texture).

    Detail at the level of bricks, fabric weave — KHÁC clarity (lower freq).
    Implementation: high-pass filter ở freq mid (Gaussian sigma ~3-8 px).
    """
    if abs(amount) < 1e-3:
        return img
    a = float(np.clip(amount, -1.0, 1.0))
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=2.5)
    high_freq = f - blur
    out = f + a * 0.5 * high_freq
    return np.clip(out, 0, 255).astype(np.uint8)


def clarity(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Local midtone contrast (Lightroom Clarity).

    Detail at architecture level (windows, room features).
    Implementation: unsharp mask ở freq lower (sigma ~25-50 px).
    """
    if abs(amount) < 1e-3:
        return img
    a = float(np.clip(amount, -1.0, 1.0))
    f = img.astype(np.float32)
    # Sigma tỉ lệ với image size
    sigma = max(15, min(img.shape[:2]) / 80)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    low_freq = f - blur
    # Áp chỉ ở midtones (mask theo luminance — không boost shadows/highlights)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    midtone_mask = 1.0 - 4.0 * (gray - 0.5) ** 2  # parabola peak ở 0.5
    midtone_mask = np.clip(midtone_mask, 0, 1)[..., None]
    out = f + a * 0.4 * low_freq * midtone_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def dehaze(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Dark Channel Prior dehaze (He et al. 2009 + Lightroom Dehaze).

    Removes atmospheric haze — useful cho ảnh ngoài trời thiếu contrast,
    ảnh underwater, foggy weather.

    amount: 0..1.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a < 1e-3:
        return img

    f = img.astype(np.float32) / 255.0

    # Dark channel: per-pixel min trong patch
    patch_size = max(7, min(img.shape[:2]) // 100)
    if patch_size % 2 == 0:
        patch_size += 1
    dark = np.min(f, axis=2)
    kernel = np.ones((patch_size, patch_size), dtype=np.float32)
    dark_min = cv2.erode(dark, kernel)

    # Atmospheric light: top 0.1% brightest pixels in dark channel
    flat = dark_min.flatten()
    n_top = max(1, int(flat.size * 0.001))
    top_idx = np.argpartition(-flat, n_top)[:n_top]
    rows, cols = np.unravel_index(top_idx, dark_min.shape)
    A = f[rows, cols].mean(axis=0)  # 3-channel atmospheric light
    A = np.clip(A, 0.6, 1.0)

    # Transmission map
    omega = 0.85 * a  # mức dehaze
    t = 1.0 - omega * dark_min / max(A.max(), 1e-6)
    t = np.clip(t, 0.1, 1.0)

    # Soft transmission (guided filter would be better; box blur is faster)
    t_smooth = cv2.boxFilter(t, cv2.CV_32F, (patch_size * 4, patch_size * 4))
    t_smooth = np.clip(t_smooth, 0.1, 1.0)

    # Recover scene radiance: J = (I - A) / t + A
    t_3 = t_smooth[..., None]
    J = (f - A) / t_3 + A
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


# =====================================================================
# 4. Full pipeline
# =====================================================================

def apply_tone_full(img: np.ndarray, params: ToneParams) -> np.ndarray:
    """Áp full Lightroom-style pipeline theo thứ tự pro chuẩn:

    1. WB (temp/tint) — luôn đầu tiên
    2. Exposure + parametric tone
    3. Dehaze (atmospheric clarity trước detail)
    4. Texture + Clarity (detail levels khác nhau)
    """
    out = img
    out = temperature_tint(
        out, temp_shift=params.temp_shift, tint_shift=params.tint_shift,
    )
    out = parametric_tone(out, params)
    if params.dehaze > 0:
        out = dehaze(out, amount=params.dehaze)
    if abs(params.texture) > 1e-3:
        out = texture(out, amount=params.texture)
    if abs(params.clarity) > 1e-3:
        out = clarity(out, amount=params.clarity)
    return out
