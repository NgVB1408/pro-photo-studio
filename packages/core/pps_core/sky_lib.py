"""Procedural sky library — thay thế gradient phẳng kiểu Autoenhance.ai.

Pipeline thực tế cho ảnh real estate:
- 6 preset có **mây Perlin noise** thật sự (không phải gradient phẳng)
- Atmospheric haze ở chân trời (warm tint giảm sat khi gần ground)
- Vertical gradient sử dụng power curve (đậm hơn ở zenith, sáng hơn ở horizon)
- Match color temperature của ảnh scene (warm scene → sky bias warm hơn)
- Spread sky color cast lên cửa kính/window pane (subtle, ~10%)

So với gradient cũ:
- Có cloud structure → nhìn không "paint bucket"
- Horizon haze tự nhiên → blend mượt với building/cây
- Color cast lên window → pro-grade reflection

API: generate_sky(h, w, preset, seed=None) -> BGR uint8
     match_sky_to_scene(sky, scene, mask) -> sky đã match warm/cool
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SkyPresetV2 = Literal[
    "blue_clear",  # trời xanh trong, không mây — open sky
    "blue_clouds",  # xanh + mây cumulus — phổ biến nhất cho RE
    "sunset_warm",  # hoàng hôn cam đỏ
    "golden_hour",  # vàng nhẹ + mây nhẹ — dramatic listing
    "dramatic_storm",  # mây đen + tia sáng — luxury listing
    "overcast_soft",  # nhiều mây mềm — neutral
]


# Preset config: (zenith_BGR, horizon_BGR, has_clouds, cloud_density, cloud_softness)
_PRESET_CONFIG: dict[str, dict] = {
    "blue_clear": {
        "zenith": (210, 130, 60),
        "horizon": (240, 200, 165),
        "clouds": False,
        "haze_color": (200, 215, 235),
        "haze_strength": 0.28,
    },
    "blue_clouds": {
        "zenith": (215, 150, 75),
        "horizon": (235, 215, 195),
        "clouds": True,
        "cloud_density": 0.55,  # threshold thấp hơn = nhiều mây hơn
        "cloud_softness": 2.5,
        "cloud_color": (252, 248, 244),
        "cloud_mix": 0.85,
        "haze_color": (210, 220, 235),
        "haze_strength": 0.22,
    },
    "sunset_warm": {
        "zenith": (140, 90, 60),
        "horizon": (90, 160, 245),
        "clouds": True,
        "cloud_density": 0.50,
        "cloud_softness": 2.0,
        "cloud_color": (130, 175, 245),  # mây cam
        "cloud_mix": 0.75,
        "haze_color": (110, 175, 250),
        "haze_strength": 0.40,
    },
    "golden_hour": {
        "zenith": (170, 130, 110),
        "horizon": (130, 200, 245),
        "clouds": True,
        "cloud_density": 0.45,
        "cloud_softness": 2.2,
        "cloud_color": (220, 240, 250),
        "cloud_mix": 0.78,
        "haze_color": (160, 215, 250),
        "haze_strength": 0.35,
    },
    "dramatic_storm": {
        "zenith": (60, 50, 45),
        "horizon": (140, 130, 120),
        "clouds": True,
        "cloud_density": 0.40,
        "cloud_softness": 1.8,
        "cloud_color": (100, 95, 92),  # mây đen
        "cloud_mix": 0.80,
        "haze_color": (130, 130, 125),
        "haze_strength": 0.18,
    },
    "overcast_soft": {
        "zenith": (200, 200, 200),
        "horizon": (228, 228, 228),
        "clouds": True,
        "cloud_density": 0.35,  # mây dày phủ kín
        "cloud_softness": 3.2,
        "cloud_color": (240, 240, 238),
        "cloud_mix": 0.55,
        "haze_color": (232, 232, 230),
        "haze_strength": 0.12,
    },
}


def _value_noise_2d(
    h: int,
    w: int,
    *,
    scale: int = 100,
    octaves: int = 5,
    persistence: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Multi-octave value noise (giống Perlin về tinh thần). Nhanh, deterministic.

    Trả về float32 [0,1] shape (h,w).
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    cur_scale = float(scale)
    for _ in range(octaves):
        cell_h = max(2, int(h / max(cur_scale, 1)))
        cell_w = max(2, int(w / max(cur_scale, 1)))
        # Giá trị random per cell, upsample mượt
        grid = rng.random((cell_h, cell_w)).astype(np.float32)
        upsampled = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += upsampled * amplitude
        total_amp += amplitude
        cur_scale *= 0.55
        amplitude *= persistence
    out /= max(total_amp, 1e-6)
    # Normalize to [0,1]
    lo, hi = float(out.min()), float(out.max())
    if hi - lo > 1e-6:
        out = (out - lo) / (hi - lo)
    return out


def _vertical_gradient(h: int, w: int, top_bgr, bot_bgr, *, power: float = 0.65) -> np.ndarray:
    """Vertical gradient với power curve (đậm hơn ở top, sáng hơn ở horizon)."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    t = np.power(y, power)
    top = np.array(top_bgr, dtype=np.float32).reshape(1, 1, 3)
    bot = np.array(bot_bgr, dtype=np.float32).reshape(1, 1, 3)
    grad = top * (1.0 - t) + bot * t
    return np.broadcast_to(grad, (h, w, 3)).copy()


def generate_sky(
    h: int,
    w: int,
    *,
    preset: SkyPresetV2 = "blue_clouds",
    seed: int | None = None,
) -> np.ndarray:
    """Tạo ảnh trời procedural BGR uint8.

    seed=None → random sky mỗi lần. seed=int → deterministic.
    """
    if preset not in _PRESET_CONFIG:
        raise ValueError(f"Preset không hợp lệ: {preset}. Có: {list(_PRESET_CONFIG)}")
    cfg = _PRESET_CONFIG[preset]
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    # 1. Vertical gradient
    sky = _vertical_gradient(h, w, cfg["zenith"], cfg["horizon"])

    # 2. Clouds (nếu preset có)
    if cfg.get("clouds"):
        # Scale tương đối: ảnh càng rộng thì cloud càng to
        scale = max(60, w // 12)
        noise = _value_noise_2d(
            h,
            w,
            scale=scale,
            octaves=5,
            persistence=0.55,
            seed=seed,
        )
        # Threshold soft → mask mây có giá trị 0..1
        density = float(cfg.get("cloud_density", 0.5))
        softness = float(cfg.get("cloud_softness", 2.0))
        cloud_mask = np.clip((noise - density) * softness, 0.0, 1.0)
        # Mây mỏng hơn ở zenith (top), dày hơn ở midsky (typical real-world)
        y_norm = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        # Cumulus thật thường ở mid-sky (y=0.3..0.7), giảm density ở top và horizon
        mid_bias = 1.0 - 4.0 * (y_norm - 0.5) ** 2  # parabola peak at 0.5
        mid_bias = np.clip(mid_bias, 0.3, 1.0)
        cloud_mask = cloud_mask * mid_bias
        cloud_mask_3d = cloud_mask[..., None]

        cloud_bgr = np.array(cfg.get("cloud_color", (250, 248, 245)), dtype=np.float32).reshape(
            1, 1, 3
        )
        cloud_mix = float(cfg.get("cloud_mix", 0.8))
        sky = sky * (1.0 - cloud_mask_3d * cloud_mix) + cloud_bgr * (cloud_mask_3d * cloud_mix)

    # 3. Atmospheric haze ở chân trời (bottom 30%)
    haze_strength = float(cfg.get("haze_strength", 0.25))
    if haze_strength > 0:
        y_norm = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        # Haze mạnh dần từ giữa xuống horizon
        haze_mask = np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0)[..., None]
        haze_bgr = np.array(cfg.get("haze_color", (210, 220, 235)), dtype=np.float32).reshape(
            1, 1, 3
        )
        sky = sky * (1.0 - haze_mask * haze_strength) + haze_bgr * (haze_mask * haze_strength)

    # 4. Mild noise để tránh banding khi save JPEG
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1.2, (h, w, 3)).astype(np.float32)
    sky = sky + grain

    return np.clip(sky, 0, 255).astype(np.uint8)


def estimate_scene_temperature(scene_bgr: np.ndarray, sky_mask: np.ndarray) -> float:
    """Ước lượng color temperature của phần KHÔNG phải sky (building/ground/cây).

    Trả về tỉ lệ R/B trong phần ground:
      > 1.05  → scene warm (đèn vàng, chiều)
      ≈ 1.0   → neutral (ngày nắng)
      < 0.95  → scene cool (mây xám, sớm sáng)
    """
    ground = sky_mask < 128
    if ground.sum() < 1000:
        return 1.0
    # Giảm sample size để nhanh
    h, w = scene_bgr.shape[:2]
    step = max(1, min(h, w) // 200)
    sample = scene_bgr[::step, ::step]
    sample_mask = ground[::step, ::step]
    pixels = sample[sample_mask]
    if len(pixels) < 100:
        return 1.0
    b_mean = float(pixels[:, 0].mean()) + 1e-6
    r_mean = float(pixels[:, 2].mean()) + 1e-6
    return r_mean / b_mean


def match_sky_to_scene(
    sky_bgr: np.ndarray,
    scene_bgr: np.ndarray,
    sky_mask: np.ndarray,
    *,
    max_shift: float = 0.06,
) -> np.ndarray:
    """Subtle warm/cool shift cho sky để match với scene temperature.

    max_shift: giới hạn shift để không phá tone preset (default ±18%).
    """
    temp = estimate_scene_temperature(scene_bgr, sky_mask)
    # Map temp → shift factor
    # temp 1.0 → no shift
    # temp 1.2 (warm scene) → sky cũng warm hơn (giảm B, tăng R nhẹ)
    # temp 0.8 (cool scene) → sky cũng cool hơn
    shift = float(np.clip((temp - 1.0) * 0.5, -max_shift, max_shift))
    if abs(shift) < 0.02:
        return sky_bgr  # gần như không đổi
    f = sky_bgr.astype(np.float32)
    # Apply shift: B *= (1 - shift), R *= (1 + shift)
    f[..., 0] = np.clip(f[..., 0] * (1.0 - shift), 0, 255)
    f[..., 2] = np.clip(f[..., 2] * (1.0 + shift), 0, 255)
    logger.debug("sky color match: scene temp=%.2f, shift=%+.3f", temp, shift)
    return f.astype(np.uint8)


def cast_sky_color_on_glass(
    img_bgr: np.ndarray,
    sky_bgr: np.ndarray,
    sky_mask: np.ndarray,
    *,
    strength: float = 0.12,
) -> np.ndarray:
    """Subtle reflection: cast sky color lên các bề mặt sáng KHÔNG phải sky
    (cửa kính, cửa sổ, water). Pro-grade detail.

    Detect bright low-saturation regions ngoài sky_mask.
    strength: 0..0.3, mặc định 0.12 (12% blend, rất subtle).
    """
    if strength <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] >= 180
    low_sat = hsv[..., 1] <= 70
    not_sky = sky_mask < 128
    glass_candidate = (bright & low_sat & not_sky).astype(np.uint8) * 255
    # Yêu cầu cluster đủ to (cửa sổ thật, không phải pixel noise)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        glass_candidate,
        connectivity=8,
    )
    keep = np.zeros_like(glass_candidate)
    min_area = (h * w) * 0.0008
    max_area = (h * w) * 0.10
    for i in range(1, n):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return img_bgr
    # Soften mask
    keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=4)
    alpha = (keep.astype(np.float32) / 255.0 * strength)[..., None]

    # Cast sky color: trung bình sky color (chỉ phần sky thật)
    sky_pixels = sky_bgr[sky_mask >= 128]
    if len(sky_pixels) < 10:
        return img_bgr
    sky_avg = sky_pixels.mean(axis=0).astype(np.float32)
    out = img_bgr.astype(np.float32)
    out = out * (1 - alpha) + sky_avg * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
