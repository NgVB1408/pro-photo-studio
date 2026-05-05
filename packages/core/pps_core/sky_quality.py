"""Sky quality detection — quyết định CÓ NÊN replace sky không.

Module này giải quyết catastrophic failure của pipeline cũ: ảnh twilight/golden
hour ĐẸP SẴN bị replace thành trời xanh trưa, phá storytelling + mâu thuẫn lighting.

Pipeline:
1. ``is_sky_already_beautiful(img, sky_mask)`` — detect golden hour, twilight,
   dramatic clouds, sunset. Nếu True → skip sky replace.
2. ``detect_warm_indoor_glow(img)`` — detect đèn nhà ấm rực qua cửa sổ. Nếu có
   → preset chọn phải là warm/twilight (không thể là noon blue clear).
3. ``auto_select_sky_preset(img, sky_mask, user_preset)`` — override user_preset
   khi context bắt buộc (vd: warm glow + user pick blue_clear → ép twilight_blue).

Tất cả CPU heuristic, không ML. Thresholds được calibrate trên ảnh BĐS thực.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SkyQualityReport:
    is_beautiful: bool
    category: str  # "golden_hour" | "twilight" | "dramatic_clouds" | "sunset" | "plain" | "boring_grey"
    saturation_mean: float
    saturation_p90: float
    hue_diversity: float    # std deviation hue (0..255 scale)
    warm_ratio: float       # fraction of warm pixel (orange/pink/purple)
    score: float            # 0..1 aesthetic score
    reason: str


# Hue ranges (OpenCV H in 0..179)
# Cool blue: 95-130
# Cyan: 80-94
# Warm pink/magenta: 145-179 + 0-10 (wraps)
# Orange: 5-25
# Yellow: 25-40
WARM_HUE_RANGES = [(0, 25), (145, 179)]   # pink/orange/red — twilight/sunset
COOL_BLUE_RANGE = (95, 130)


def _is_warm_hue(h_channel: np.ndarray) -> np.ndarray:
    """Mask boolean cho hue 'warm' (pink/orange/red — chỉ báo twilight/sunset)."""
    mask = np.zeros_like(h_channel, dtype=bool)
    for lo, hi in WARM_HUE_RANGES:
        mask |= (h_channel >= lo) & (h_channel <= hi)
    return mask


def is_sky_already_beautiful(
    img: np.ndarray,
    sky_mask: np.ndarray | None = None,
    *,
    min_score: float = 0.55,
) -> SkyQualityReport:
    """Detect xem trời gốc đã đẹp đủ để KHÔNG replace.

    Beautiful sky markers:
    - **Golden hour / sunset / twilight**: warm hue (orange/pink) + saturation cao.
      Đặc trưng: pixel ratio warm ≥ 12%, mean sat ≥ 35.
    - **Dramatic clouds**: hue diversity cao (gradient nhiều màu) + saturation
      đủ ở vùng warm. Đặc trưng: hue_std ≥ 22, mean sat ≥ 28.
    - **Vibrant blue clear**: saturation rất cao (≥ 60) + hue ổn định blue.
      Đặc trưng: vibrant clear blue cũng đáng giữ.

    Boring sky markers (NÊN replace):
    - Grey overcast: saturation < 18 + brightness uniform → giữ vô nghĩa
    - Pale washed-out: saturation < 22 + warm_ratio < 5% → không có character
    - Hazy white-blue: saturation < 25 và không có gradient → flat

    Args:
        img: BGR uint8 input.
        sky_mask: optional mask 0..255 vùng trời. None → dùng top 30% ảnh.
        min_score: ngưỡng "beautiful" (0..1). Default 0.55 = thận trọng.

    Returns:
        SkyQualityReport với is_beautiful + category + lý do.
    """
    h, w = img.shape[:2]

    # Region to analyze
    if sky_mask is not None and sky_mask.sum() > (h * w) * 0.005:
        # Use detected sky mask
        sky_pixels = img[sky_mask > 128]
        if len(sky_pixels) < 100:
            sky_pixels = img[: max(1, h // 4)].reshape(-1, 3)
        # Reshape to 2D for HSV conversion
        sample = sky_pixels.reshape(-1, 1, 3)
    else:
        # Fallback: top 30%
        top = img[: max(1, int(h * 0.30))]
        sample = top.reshape(-1, 1, 3)

    if sample.shape[0] < 100:
        return SkyQualityReport(
            is_beautiful=False, category="plain",
            saturation_mean=0.0, saturation_p90=0.0,
            hue_diversity=0.0, warm_ratio=0.0,
            score=0.0, reason="too_few_pixels",
        )

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # Bỏ pixel quá tối (S, V có thể nhiễu) — không đại diện sky
    valid = V >= 80
    if valid.sum() < 100:
        return SkyQualityReport(
            is_beautiful=False, category="plain",
            saturation_mean=0.0, saturation_p90=0.0,
            hue_diversity=0.0, warm_ratio=0.0,
            score=0.0, reason="too_dark",
        )

    H_v = H[valid]
    S_v = S[valid]
    V_v = V[valid]

    sat_mean = float(S_v.mean())
    sat_p90 = float(np.percentile(S_v, 90))

    # Hue diversity — nhưng phải xử lý wraparound (179 ≈ 0). Dùng circular std.
    # Convert hue → angle radian, tính circular variance
    hue_rad = (H_v.astype(np.float32) / 180.0) * 2.0 * np.pi
    sin_mean = float(np.sin(hue_rad).mean())
    cos_mean = float(np.cos(hue_rad).mean())
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)  # mean resultant length
    # circular variance = 1 - R, bound 0..1
    circular_var = 1.0 - R
    # Scale to 0..40 to match old "std deviation" feel
    hue_diversity = float(circular_var * 40.0)

    warm_mask = _is_warm_hue(H_v) & (S_v >= 25)
    warm_ratio = float(warm_mask.sum()) / max(1, len(H_v))

    blue_mask = (
        (H_v >= COOL_BLUE_RANGE[0])
        & (H_v <= COOL_BLUE_RANGE[1])
        & (S_v >= 30)
    )
    blue_ratio = float(blue_mask.sum()) / max(1, len(H_v))

    # ===== Categorization =====
    category = "plain"
    reason_parts: list[str] = []

    # Golden hour: warm dominant + sat đủ
    is_golden = warm_ratio >= 0.18 and sat_mean >= 35
    # Twilight: warm + cool gradient (hue diversity cao + có cả warm)
    is_twilight = (
        warm_ratio >= 0.08
        and hue_diversity >= 14
        and sat_mean >= 28
    )
    # Dramatic clouds: variance cao + có texture
    is_dramatic = hue_diversity >= 20 and sat_p90 >= 50
    # Vibrant blue: hoàn toàn xanh nhưng saturation rất tốt
    is_vibrant_blue = blue_ratio >= 0.55 and sat_mean >= 55 and warm_ratio < 0.05

    # Score (weighted)
    score = 0.0
    if is_golden:
        category = "golden_hour"
        score = min(1.0, 0.40 + warm_ratio * 1.5 + (sat_mean - 35) / 100)
        reason_parts.append(f"golden(warm={warm_ratio:.2f},sat={sat_mean:.0f})")
    elif is_twilight:
        category = "twilight"
        score = min(1.0, 0.35 + warm_ratio * 1.2 + hue_diversity / 60)
        reason_parts.append(f"twilight(warm={warm_ratio:.2f},div={hue_diversity:.1f})")
    elif is_dramatic:
        category = "dramatic_clouds"
        score = min(1.0, 0.30 + hue_diversity / 50 + sat_p90 / 200)
        reason_parts.append(f"dramatic(div={hue_diversity:.1f},p90={sat_p90:.0f})")
    elif is_vibrant_blue:
        category = "vibrant_blue"
        score = min(1.0, 0.30 + (sat_mean - 55) / 80 + blue_ratio / 2)
        reason_parts.append(f"vibrant_blue(sat={sat_mean:.0f},ratio={blue_ratio:.2f})")
    else:
        if sat_mean < 18:
            category = "boring_grey"
            reason_parts.append(f"flat_grey(sat={sat_mean:.0f})")
        elif sat_mean < 25 and warm_ratio < 0.05:
            category = "washed_out"
            reason_parts.append(f"washed(sat={sat_mean:.0f})")
        else:
            category = "plain"
            reason_parts.append(f"plain(sat={sat_mean:.0f},warm={warm_ratio:.2f})")
        score = 0.0

    is_beautiful = score >= min_score

    return SkyQualityReport(
        is_beautiful=is_beautiful,
        category=category,
        saturation_mean=sat_mean,
        saturation_p90=sat_p90,
        hue_diversity=hue_diversity,
        warm_ratio=warm_ratio,
        score=float(score),
        reason="|".join(reason_parts),
    )


# ======================================================================
# Warm indoor glow detection (lighting consistency)
# ======================================================================

@dataclass
class IndoorGlowReport:
    has_warm_glow: bool
    glow_ratio: float        # fraction of bright warm pixels (interior light)
    avg_hue_glow: float      # mean hue (0..179) of warm bright clusters
    suggests_time: str       # "twilight" | "evening" | "day" | "unknown"


def detect_warm_indoor_glow(
    img: np.ndarray,
    *,
    min_ratio: float = 0.005,  # ≥ 0.5% pixel ảnh có warm bright glow
    sky_mask: np.ndarray | None = None,
) -> IndoorGlowReport:
    """Detect đèn ấm rực rỡ qua cửa sổ — chỉ báo lighting time-of-day.

    Đặc trưng:
    - Pixel hue warm (orange-yellow 5-30) + sat ≥ 50 + V ≥ 200 = đèn nhà bật
    - Phải là cluster (component analysis) không phải pixel rời
    - Nếu glow_ratio đủ → scene là twilight/evening, KHÔNG match được trời trưa

    Args:
        img: BGR uint8.
        min_ratio: % pixel để confirm "có glow rõ ràng".
        sky_mask: optional, exclude vùng sky để không nhầm lẫn.

    Returns:
        IndoorGlowReport.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Warm bright pixels: orange-yellow saturated bright
    warm_glow = (
        ((H <= 30) | (H >= 160))
        & (S >= 50)
        & (V >= 200)
    )

    # Exclude sky region (sunset has warm sky too)
    if sky_mask is not None:
        warm_glow = warm_glow & (sky_mask < 128)

    # Connected component filter — avoid noise
    glow_u8 = warm_glow.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(glow_u8, connectivity=8)
    keep = np.zeros_like(glow_u8)
    min_area = max(20, (h * w) // 20000)  # at least 20 px or 0.005% of image
    total_area = 0
    hue_sum = 0.0
    hue_count = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == i] = 255
            total_area += area
            comp_pixels = H[labels == i]
            hue_sum += float(comp_pixels.sum())
            hue_count += int(area)

    glow_ratio = float(total_area) / (h * w)
    avg_hue = (hue_sum / hue_count) if hue_count > 0 else 0.0

    has_glow = glow_ratio >= min_ratio

    # Time-of-day suggestion
    if has_glow and glow_ratio >= 0.015:
        suggests_time = "twilight"  # strong glow → likely twilight/evening
    elif has_glow:
        suggests_time = "evening"
    elif glow_ratio < 0.001:
        suggests_time = "day"
    else:
        suggests_time = "unknown"

    return IndoorGlowReport(
        has_warm_glow=has_glow,
        glow_ratio=glow_ratio,
        avg_hue_glow=avg_hue,
        suggests_time=suggests_time,
    )


# ======================================================================
# Smart sky preset auto-selection
# ======================================================================

# Map preset name → time-of-day class
_PRESET_TOD_MAP: dict[str, str] = {
    "blue_clear":     "day",
    "blue_clouds":    "day",
    "overcast_soft":  "day",
    "sunset_warm":    "evening",
    "golden_hour":    "evening",
    "twilight_blue":  "twilight",
    "dramatic_storm": "any",      # storm có thể bất kỳ thời điểm
}


@dataclass
class SkyDecision:
    action: str               # "skip" | "replace"
    chosen_preset: str        # preset cuối cùng
    original_user_preset: str
    overridden: bool
    reason: str


def auto_decide_sky_action(
    img: np.ndarray,
    sky_mask: np.ndarray | None,
    user_preset: str,
    *,
    min_beautiful_score: float = 0.55,
    respect_user_preset: bool = True,
) -> SkyDecision:
    """Quyết định CÓ replace sky không + chọn preset phù hợp lighting context.

    Decision tree:
    1. Nếu sky gốc đã beautiful (golden/twilight/dramatic/vibrant_blue) → SKIP
    2. Nếu scene có warm indoor glow mạnh + user chọn day preset → ép evening preset
    3. Nếu scene có warm glow vừa → giữ user preset nhưng warn
    4. Default → replace với user preset

    Args:
        img: BGR uint8.
        sky_mask: optional sky mask để analyze quality chính xác hơn.
        user_preset: preset user chọn (vd "blue_clouds").
        min_beautiful_score: ngưỡng "đẹp sẵn".
        respect_user_preset: True = nếu user pick warm preset, không override.

    Returns:
        SkyDecision với action + lý do.
    """
    quality = is_sky_already_beautiful(
        img, sky_mask, min_score=min_beautiful_score,
    )

    # Path 1: sky đã đẹp → skip hoàn toàn
    if quality.is_beautiful:
        return SkyDecision(
            action="skip",
            chosen_preset=user_preset,
            original_user_preset=user_preset,
            overridden=False,
            reason=f"sky_beautiful({quality.category},score={quality.score:.2f},{quality.reason})",
        )

    # Path 2: lighting context ép preset
    glow = detect_warm_indoor_glow(img, sky_mask=sky_mask)
    user_tod = _PRESET_TOD_MAP.get(user_preset, "day")

    # Strong warm glow + user picked day preset → override
    if glow.has_warm_glow and glow.suggests_time in ("twilight", "evening"):
        if user_tod == "day":
            new_preset = (
                "twilight_blue" if glow.suggests_time == "twilight"
                else "sunset_warm"
            )
            return SkyDecision(
                action="replace",
                chosen_preset=new_preset,
                original_user_preset=user_preset,
                overridden=True,
                reason=(
                    f"warm_indoor_glow_overrides_day_preset("
                    f"glow={glow.glow_ratio:.3f},user={user_preset}→{new_preset})"
                ),
            )

    # Default: replace với user preset như cũ
    return SkyDecision(
        action="replace",
        chosen_preset=user_preset,
        original_user_preset=user_preset,
        overridden=False,
        reason=f"replace_default(quality={quality.category})",
    )
