"""Fixers v0.2 — quality-grade window highlight recovery + ceiling neutralization.

WINDOW FIX:
    1. Identify blown highlights inside mask (luminance > 235).
    2. Generate synthetic underexposed exposure via inverse-gamma + dehaze.
    3. Multi-band blend (Laplacian pyramid 3 levels) — preserve edges.
    4. Reinhard local tone-mapping inside mask region.
    5. Guided filter on highlight transition để KHÔNG có halo.
    6. Soft chroma recovery (boost saturation cho desaturated highlights).

CEILING FIX:
    1. Estimate illuminant under ceiling region via gray-world + bright-pixel mix.
    2. Compute chromatic adaptation matrix (Bradford CAT) target → D65 neutral.
    3. Apply CAT in XYZ space (proper color science, not LAB shift).
    4. Local luminance equalization (uniform brightness across ceiling).
    5. Guided filter cho edge preservation tránh smudge sang wall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Color science helpers ─────────────────────────────────────────────────────

# Bradford CAT — gold standard chromatic adaptation transform
_BRADFORD_M = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
_BRADFORD_M_INV = np.linalg.inv(_BRADFORD_M)

# sRGB ↔ XYZ matrices (D65 illuminant assumed for sRGB).
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """sRGB gamma → linear. arr in [0,1]."""
    out = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return out


def _linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    out = np.where(arr <= 0.0031308, 12.92 * arr, 1.055 * np.power(np.maximum(arr, 0), 1 / 2.4) - 0.055)
    return out


def _bradford_cat(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    """Compute 3×3 Bradford CAT matrix (in XYZ space) for src_illum → dst_illum."""
    src_cone = _BRADFORD_M @ src_white
    dst_cone = _BRADFORD_M @ dst_white
    diag = np.diag(dst_cone / np.maximum(src_cone, 1e-6))
    return _BRADFORD_M_INV @ diag @ _BRADFORD_M


# ── Guided filter (edge-preserving smoother) ──────────────────────────────────


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 16, eps: float = 1e-3) -> np.ndarray:
    """OpenCV ximgproc-style guided filter, but vanilla cv2 implementation.

    Used to constrain mask-region adjustments to image structure → no halos.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    win = (radius * 2 + 1, radius * 2 + 1)

    mean_g = cv2.boxFilter(guide, -1, win)
    mean_s = cv2.boxFilter(src, -1, win)
    corr_gg = cv2.boxFilter(guide * guide, -1, win)
    corr_gs = cv2.boxFilter(guide * src, -1, win)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, -1, win)
    mean_b = cv2.boxFilter(b, -1, win)

    return mean_a * guide + mean_b


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.float32:
        m = mask.astype(np.float32) / 255.0
    else:
        m = np.clip(mask, 0.0, 1.0)
    return m


# ── Highlight roll-off (window) ───────────────────────────────────────────────


def _highlight_rolloff(
    luminance: np.ndarray,
    *,
    knee: float = 0.65,
    target_ceiling: float = 0.85,
    softness: float = 0.5,
) -> np.ndarray:
    """Smooth highlight roll-off — compress values above `knee` so brightest
    parts move from [knee..1.0] into [knee..target_ceiling].

    Hermite smoothstep blend prevents banding around the knee transition.

    Args:
        luminance: linear luminance, float32, expected in [0..1+] range.
        knee: where compression begins (default 0.65 in linear).
        target_ceiling: max output value (default 0.85 keeps headroom).
        softness: blend region width below knee (0..1).

    Returns:
        Compressed luminance, float32 in [0..target_ceiling].
    """
    lum = np.clip(luminance, 0.0, None)
    # Linear pass-through below knee
    below = lum
    # Above knee: log-style soft compress
    excess = np.maximum(lum - knee, 0.0)
    avail = target_ceiling - knee
    if avail <= 0:
        avail = 0.01
    # log1p compression — natural shoulder
    compressed_excess = avail * (1.0 - np.exp(-excess / max(avail, 0.05)))
    above = knee + compressed_excess

    # Soft blend in [knee-softness*knee, knee+softness*knee] band via smoothstep
    blend_lo = knee * (1.0 - softness)
    blend_hi = knee * (1.0 + softness)
    t = np.clip((lum - blend_lo) / max(blend_hi - blend_lo, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return below * (1.0 - smooth) + above * smooth


def _aces_filmic_curve(luminance: np.ndarray) -> np.ndarray:
    """ACES-approximate filmic tone-mapper (Krzysztof Narkowicz). Good shoulder.

    Useful as alternative to highlight_rolloff for very HDR-ish input.
    """
    x = np.clip(luminance, 0.0, None)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class FixerMetrics:
    applied: bool
    mask_pct: float = 0.0
    extra: dict = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        d: dict = {"applied": self.applied, "mask_pct": self.mask_pct}
        if self.extra:
            d.update(self.extra)
        return d


def fix_window_highlights(
    img_bgr: np.ndarray,
    window_mask: np.ndarray,
    *,
    strength: float = 1.0,
    chroma_recover: float = 1.04,  # ↓ from 1.12 — real-estate: subtle, not punchy
    guide_radius: int = 20,        # ↓ from 40 — tighter band, less bleed to wall
    knee: float = 0.55,
    target_ceiling: float = 0.82,
) -> tuple[np.ndarray, dict]:
    """Recover blown highlights inside window mask via:
       1. Smooth highlight roll-off (compress values > knee toward target_ceiling).
       2. Channel-ratio re-apply (preserves hue / chroma direction).
       3. Saturation boost (compensate desaturation from blowout).
       4. Guided-filter blend (edge-aware, no halo).

    Args:
        img_bgr: uint8 H×W×3.
        window_mask: uint8 0..255.
        strength: 0..1.5+ — interpolation factor between original and tonemapped.
        chroma_recover: saturation multiplier post-recovery.
        guide_radius: guided filter radius.
        knee: roll-off threshold in linear luminance (0..1).
        target_ceiling: max luminance after compression (default 0.86 = ~stop below pure white).
    """
    if window_mask.sum() == 0 or strength <= 0:
        return img_bgr.copy(), FixerMetrics(applied=False, extra={"reason": "empty_or_off"}).to_dict()

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    linear = _srgb_to_linear(rgb)
    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]

    # Highlight roll-off (compresses high values DOWNWARD)
    tm_lum = _highlight_rolloff(luminance, knee=knee, target_ceiling=target_ceiling, softness=0.4)
    new_lum = luminance * (1 - strength) + tm_lum * strength
    new_lum = np.clip(new_lum, 1e-6, 1.0)

    # Channel-ratio re-apply (preserve chroma)
    ratio = (new_lum / np.maximum(luminance, 1e-6))[..., None]
    recovered = np.clip(linear * ratio, 0.0, 1.0)
    sat_rgb = _linear_to_srgb(recovered)

    # Saturation recovery
    if chroma_recover != 1.0:
        u8 = (sat_rgb * 255).clip(0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * chroma_recover, 0, 255)
        sat_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Guided-filter mask blend
    mask_f = _normalize_mask(window_mask)
    guided_mask = _guided_filter(luminance, mask_f, radius=guide_radius, eps=2e-3)
    guided_mask = np.clip(guided_mask, 0.0, 1.0)[..., None]

    blended = sat_rgb * guided_mask + rgb * (1 - guided_mask)
    out_rgb = np.clip(blended, 0.0, 1.0)
    out_bgr = cv2.cvtColor((out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Metrics
    inside = mask_f > 0.2
    if inside.any():
        lum_before_mask = float(luminance[inside].mean())
        lum_after_mask = float(new_lum[inside].mean())
        clipped_before = float((luminance[inside] > 0.95).mean()) * 100
        clipped_after = float((new_lum[inside] > 0.95).mean()) * 100
    else:
        lum_before_mask = lum_after_mask = clipped_before = clipped_after = 0.0

    metrics = FixerMetrics(
        applied=True,
        mask_pct=float((window_mask > 64).mean()) * 100,
        extra={
            "lum_before": lum_before_mask,
            "lum_after": lum_after_mask,
            "clipped_pct_before": clipped_before,
            "clipped_pct_after": clipped_after,
            "knee": float(knee),
            "target_ceiling": float(target_ceiling),
            "strength": float(strength),
        },
    ).to_dict()
    return out_bgr, metrics


def fix_ceiling_neutrality(
    img_bgr: np.ndarray,
    ceiling_mask: np.ndarray,
    *,
    strength: float = 0.85,
    luminance_equalize: bool = False,  # ↓ default OFF — σ=40 spreads beyond ceiling
    guide_radius: int = 16,            # ↓ from 32 — tighter, no wall tint
) -> tuple[np.ndarray, dict]:
    """Neutralize ceiling color cast via Bradford CAT (proper chromatic adaptation).

    Steps:
        1. Estimate ceiling illuminant XYZ via mean of bright (top-25% luminance) ceiling pixels.
        2. Compute Bradford CAT matrix from estimated illuminant → D65 neutral.
        3. Apply CAT in linear XYZ space inside mask only.
        4. Optional: local luminance equalization (even-out brightness across ceiling).
        5. Guided filter blend so edges of ceiling/wall don't smear.

    Args:
        img_bgr: uint8 H×W×3.
        ceiling_mask: uint8 0..255.
        strength: 0..1 — how strongly to pull illuminant toward D65 neutral.
        luminance_equalize: True → equalize luminance across ceiling region.
        guide_radius: guided filter radius.

    Returns:
        (output BGR uint8, metrics dict).
    """
    if ceiling_mask.sum() == 0 or strength <= 0:
        return img_bgr.copy(), FixerMetrics(applied=False, extra={"reason": "empty_or_off"}).to_dict()

    h, w = img_bgr.shape[:2]
    mask_f = _normalize_mask(ceiling_mask)

    # 1. Convert to linear XYZ
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear.reshape(-1, 3) @ _SRGB_TO_XYZ.T.astype(np.float32)
    xyz = xyz.reshape(h, w, 3)

    # 2. Estimate illuminant from bright ceiling pixels (top 25% by luminance within mask)
    luminance = xyz[..., 1]
    inside = mask_f > 0.5
    if not inside.any():
        return img_bgr.copy(), FixerMetrics(applied=False, extra={"reason": "no_solid_mask"}).to_dict()

    inside_lum = luminance[inside]
    threshold = np.percentile(inside_lum, 75)
    bright_mask = inside & (luminance >= threshold)
    if bright_mask.sum() < 50:
        bright_mask = inside

    src_white = xyz[bright_mask].mean(axis=0).astype(np.float64)
    src_white_norm = src_white / max(src_white[1], 1e-6)  # normalize so Y=1

    # 3. Compute Bradford CAT toward D65, interpolated by `strength`
    cat_full = _bradford_cat(src_white_norm, _D65)
    cat_blend = (1.0 - strength) * np.eye(3) + strength * cat_full

    # 4. Apply CAT in XYZ space (full image, then mask-blend at end)
    xyz_corr = (xyz.reshape(-1, 3) @ cat_blend.T).reshape(h, w, 3).astype(np.float32)

    # Back to linear sRGB
    linear_corr = xyz_corr.reshape(-1, 3) @ _XYZ_TO_SRGB.T.astype(np.float32)
    linear_corr = np.clip(linear_corr.reshape(h, w, 3), 0.0, 1.0)

    # 5. Local luminance equalize (optional) — pulls darker corners of ceiling
    #    up to median brightness, keeps subtle texture.
    if luminance_equalize:
        Y_corr = 0.2126 * linear_corr[..., 0] + 0.7152 * linear_corr[..., 1] + 0.0722 * linear_corr[..., 2]
        med = float(np.median(Y_corr[inside])) if inside.any() else 0.7
        local_avg = cv2.GaussianBlur(Y_corr, (0, 0), sigmaX=40.0)
        eq_factor = np.where(local_avg > 1e-3, (med / np.maximum(local_avg, 1e-3)), 1.0)
        eq_factor = np.clip(eq_factor, 0.85, 1.18)  # cap to avoid overdrive
        linear_corr = np.clip(linear_corr * eq_factor[..., None], 0.0, 1.0)

    # 6. Guided filter mask blend
    Y_orig = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    guided_mask = _guided_filter(Y_orig, mask_f, radius=guide_radius, eps=2e-3)
    guided_mask = np.clip(guided_mask, 0.0, 1.0)[..., None]

    out_lin = linear_corr * guided_mask + linear * (1 - guided_mask)
    out_rgb = _linear_to_srgb(np.clip(out_lin, 0.0, 1.0))
    out_bgr = cv2.cvtColor((out_rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Metrics — measure cast magnitude in LAB
    lab_before = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_after = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cast_before = float(
        abs(lab_before[..., 1][inside].mean() - 128.0)
        + abs(lab_before[..., 2][inside].mean() - 128.0)
    )
    cast_after = float(
        abs(lab_after[..., 1][inside].mean() - 128.0)
        + abs(lab_after[..., 2][inside].mean() - 128.0)
    )

    metrics = FixerMetrics(
        applied=True,
        mask_pct=float((ceiling_mask > 64).mean()) * 100,
        extra={
            "src_illuminant_xyz": [float(v) for v in src_white_norm],
            "cast_magnitude_before": cast_before,
            "cast_magnitude_after": cast_after,
            "strength": float(strength),
        },
    ).to_dict()
    return out_bgr, metrics
