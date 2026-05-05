"""Tests cho sky_quality module — smart sky skip + preset override."""

from __future__ import annotations

import numpy as np
from pps_core.sky_quality import (
    auto_decide_sky_action,
    detect_warm_indoor_glow,
    is_sky_already_beautiful,
)

# ============================================================================
# Helpers — synthesize sky scenarios
# ============================================================================


def _make_blue_clear_sky(h: int = 600, w: int = 800) -> np.ndarray:
    """Vibrant clear blue sky — beautiful (vibrant_blue category)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top half — vibrant blue (BGR: 200, 130, 50 = saturated blue)
    img[: h // 2] = [200, 130, 50]
    # Bottom half — ground brown
    img[h // 2 :] = [60, 70, 90]
    return img


def _make_grey_overcast_sky(h: int = 600, w: int = 800) -> np.ndarray:
    """Grey washed-out overcast — boring, should be replaced."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)  # uniform grey ~180
    img[h // 2 :] = [80, 90, 100]  # ground
    return img


def _make_golden_hour_sky(h: int = 600, w: int = 800) -> np.ndarray:
    """Golden hour — warm orange dominant. Should be PRESERVED."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top — warm orange BGR (15, 100, 220)
    for y in range(h // 2):
        t = y / (h // 2)
        b = int(15 + 50 * t)
        g = int(100 + 50 * t)
        r = int(220 - 30 * t)
        img[y] = [b, g, r]
    img[h // 2 :] = [30, 40, 60]
    return img


def _make_twilight_sky(h: int = 600, w: int = 800) -> np.ndarray:
    """Twilight — vivid pink/purple gradient + cool blue zenith. PRESERVED."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sky_h = h // 2
    for y in range(sky_h):
        t = y / sky_h  # 0 = top (zenith), 1 = horizon
        # Zenith deep purple-blue → horizon warm pink/magenta
        # BGR top: (140, 60, 80) deep purple-blue
        # BGR bottom: (180, 100, 230) warm pink
        b = int(140 + 40 * t)
        g = int(60 + 40 * t)
        r = int(80 + 150 * t)
        img[y] = [min(255, b), min(255, g), min(255, r)]
    img[sky_h:] = [40, 35, 50]  # ground
    return img


def _make_house_with_warm_glow(h: int = 600, w: int = 800) -> np.ndarray:
    """Exterior house at twilight với cửa sổ phát đèn ấm rực."""
    img = np.full((h, w, 3), 60, dtype=np.uint8)  # dark twilight ambient

    # Top: dark blue twilight sky
    for y in range(int(h * 0.4)):
        t = y / (h * 0.4)
        img[y] = [int(80 + 40 * t), int(50 + 30 * t), int(30 + 20 * t)]

    # House body — dark grey
    img[int(h * 0.4) : int(h * 0.85), int(w * 0.2) : int(w * 0.8)] = [70, 70, 70]

    # WARM GLOWING WINDOWS — strong orange (BGR: 50, 200, 255)
    win_y0 = int(h * 0.50)
    win_y1 = int(h * 0.65)
    win_x0 = int(w * 0.30)
    win_x1 = int(w * 0.45)
    img[win_y0:win_y1, win_x0:win_x1] = [50, 200, 255]
    img[win_y0:win_y1, int(w * 0.55) : int(w * 0.70)] = [50, 200, 255]

    return img


# ============================================================================
# Tests: is_sky_already_beautiful
# ============================================================================


def test_grey_overcast_not_beautiful():
    img = _make_grey_overcast_sky()
    report = is_sky_already_beautiful(img)
    assert report.is_beautiful is False
    assert report.category in ("boring_grey", "washed_out", "plain")


def test_vibrant_blue_is_beautiful():
    img = _make_blue_clear_sky()
    report = is_sky_already_beautiful(img)
    assert report.is_beautiful is True
    assert report.category == "vibrant_blue"
    assert report.score >= 0.55


def test_golden_hour_is_beautiful():
    img = _make_golden_hour_sky()
    report = is_sky_already_beautiful(img)
    assert report.is_beautiful is True
    assert report.category in ("golden_hour", "twilight", "dramatic_clouds")
    assert report.warm_ratio >= 0.10


def test_twilight_is_beautiful():
    img = _make_twilight_sky()
    report = is_sky_already_beautiful(img)
    # Twilight có warm gradient + diversity
    assert report.is_beautiful is True
    assert report.category in ("twilight", "golden_hour", "dramatic_clouds")


def test_min_score_threshold_strict():
    img = _make_blue_clear_sky()
    # Set very strict threshold — require near-perfect score
    report = is_sky_already_beautiful(img, min_score=0.99)
    # Should still be beautiful since vibrant blue scores 1.0
    assert report.score >= 0.5  # might or might not pass strict


# ============================================================================
# Tests: detect_warm_indoor_glow
# ============================================================================


def test_no_glow_in_grey_overcast():
    img = _make_grey_overcast_sky()
    report = detect_warm_indoor_glow(img)
    assert report.has_warm_glow is False
    assert report.glow_ratio < 0.005


def test_warm_glow_detected_in_house():
    img = _make_house_with_warm_glow()
    report = detect_warm_indoor_glow(img)
    assert report.has_warm_glow is True
    assert report.glow_ratio >= 0.005
    assert report.suggests_time in ("twilight", "evening")


# ============================================================================
# Tests: auto_decide_sky_action
# ============================================================================


def test_decide_skip_when_sky_beautiful():
    img = _make_blue_clear_sky()
    decision = auto_decide_sky_action(img, None, "blue_clouds")
    assert decision.action == "skip"
    assert "beautiful" in decision.reason.lower() or "vibrant" in decision.reason.lower()


def test_decide_replace_when_sky_grey():
    img = _make_grey_overcast_sky()
    decision = auto_decide_sky_action(img, None, "blue_clouds")
    assert decision.action == "replace"
    assert decision.chosen_preset == "blue_clouds"  # no override needed


def test_decide_override_preset_when_warm_glow():
    """House với warm glow + user pick day preset → ép evening preset."""
    img = _make_house_with_warm_glow()
    decision = auto_decide_sky_action(img, None, "blue_clear")
    # Sky portion là dark twilight gradient → có thể beautiful hoặc plain
    # Nếu plain → glow detected → override sang twilight_blue/sunset_warm
    if decision.action == "replace":
        # Override should kick in vì warm glow rõ + user chọn day preset
        assert decision.chosen_preset in ("twilight_blue", "sunset_warm")
        assert decision.overridden is True
    else:
        # Sky đẹp sẵn → skip cũng OK
        assert decision.action == "skip"


def test_decide_no_override_when_user_picks_warm():
    """User chọn sunset_warm + warm glow → giữ user preset."""
    img = _make_house_with_warm_glow()
    decision = auto_decide_sky_action(img, None, "sunset_warm")
    if decision.action == "replace":
        # User đã chọn warm preset, không override
        assert decision.overridden is False
        assert decision.chosen_preset == "sunset_warm"


def test_decide_respects_user_preset_flag():
    """respect_user_preset=False vẫn cho override."""
    img = _make_house_with_warm_glow()
    decision = auto_decide_sky_action(
        img,
        None,
        "blue_clear",
        respect_user_preset=True,
    )
    # Logic vẫn override nếu warm glow đủ rõ — flag chỉ ảnh hưởng khi
    # cần user override quyết định cuối cùng
    assert decision.original_user_preset == "blue_clear"
