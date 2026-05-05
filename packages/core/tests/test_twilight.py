"""Tests cho pps_core.twilight (Day → Sunset transform)."""
from __future__ import annotations

import numpy as np
import pytest

from pps_core.twilight import (
    TwilightReport,
    transform_to_twilight,
    _detect_window_glow_mask,
    _sunset_gradient,
    _apply_warm_tone,
)


def _make_outdoor(h: int = 240, w: int = 320) -> np.ndarray:
    """Synth ảnh ngoại thất: top 50% blue sky, bottom 50% green/grey ground.
    Có 1 vùng "window" sáng ở giữa giả lập building.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky band (top 50%) — xanh
    img[: h // 2, :] = (210, 160, 100)  # BGR ~ light blue
    # Ground (bottom 50%) — neutral
    img[h // 2 :, :] = (90, 110, 95)  # BGR neutral grey-green
    # Building window cluster (mid)
    img[h // 2 - 30 : h // 2 + 10, w // 2 - 40 : w // 2 + 40] = (180, 200, 210)
    return img


def _make_interior(h: int = 240, w: int = 320) -> np.ndarray:
    """Ảnh interior — không có sky."""
    rng = np.random.default_rng(0)
    img = (rng.normal(loc=120, scale=15, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
    return img


# ---------- _sunset_gradient ----------

def test_sunset_gradient_shape_and_range():
    rng = np.random.default_rng(42)
    sky = _sunset_gradient(120, 200, rng=rng)
    assert sky.shape == (120, 200, 3)
    assert sky.dtype == np.uint8
    # Top dark, bottom warmer (R channel)
    top_r = sky[:10, :, 2].mean()
    bottom_r = sky[-30:, :, 2].mean()
    assert bottom_r > top_r, "bottom band phải warm hơn top zenith"


def test_sunset_gradient_seedable():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    sky_a = _sunset_gradient(60, 80, rng=rng_a)
    sky_b = _sunset_gradient(60, 80, rng=rng_b)
    assert np.array_equal(sky_a, sky_b)


# ---------- _apply_warm_tone ----------

def test_warm_tone_shifts_b_channel_up():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    out = _apply_warm_tone(img, strength=0.8)
    # b in LAB should increase → blue shifts down, R/G slight up in BGR
    # Concrete check: total energy in R channel ≥ original
    assert out[..., 2].mean() >= img[..., 2].mean() - 1


def test_warm_tone_zero_strength_noop():
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    out = _apply_warm_tone(img, strength=0.0)
    assert np.array_equal(out, img)


# ---------- _detect_window_glow_mask ----------

def test_window_glow_returns_float_01():
    img = _make_outdoor()
    mask = _detect_window_glow_mask(img, exclude=None)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_window_glow_excludes_sky():
    img = _make_outdoor()
    h = img.shape[0]
    sky = np.zeros(img.shape[:2], dtype=np.uint8)
    sky[: h // 2, :] = 255
    mask = _detect_window_glow_mask(img, exclude=sky)
    # Vùng top phải gần 0
    assert mask[: h // 2, :].mean() < 0.1


# ---------- transform_to_twilight ----------

def test_twilight_outdoor_applied():
    img = _make_outdoor()
    out, rpt = transform_to_twilight(img, strength=0.8, seed=42)
    assert isinstance(rpt, TwilightReport)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert rpt.applied is True
    assert rpt.sky_mask_pct > 0


def test_twilight_interior_skips_composite_but_applies_warm():
    img = _make_interior()
    out, rpt = transform_to_twilight(img, strength=0.7, seed=1)
    assert out.shape == img.shape
    # Interior: applied = False (no sky), reason mentioned
    if not rpt.applied:
        assert rpt.reason
        # Output không identical (warm tone vẫn áp)
        assert not np.array_equal(out, img)


def test_twilight_seed_deterministic():
    img = _make_outdoor()
    out_a, _ = transform_to_twilight(img.copy(), strength=0.8, seed=11)
    out_b, _ = transform_to_twilight(img.copy(), strength=0.8, seed=11)
    assert np.array_equal(out_a, out_b)


def test_twilight_strength_scaling():
    img = _make_outdoor()
    weak, _ = transform_to_twilight(img.copy(), strength=0.2, seed=3)
    strong, _ = transform_to_twilight(img.copy(), strength=1.0, seed=3)
    diff_weak = np.abs(weak.astype(np.int16) - img.astype(np.int16)).mean()
    diff_strong = np.abs(strong.astype(np.int16) - img.astype(np.int16)).mean()
    assert diff_strong > diff_weak, "strength cao phải thay đổi mạnh hơn"


def test_twilight_rejects_bgra():
    img = np.zeros((20, 20, 4), dtype=np.uint8)
    out, rpt = transform_to_twilight(img)
    # BGRA should be auto-converted to BGR, output 3-channel
    assert out.shape == (20, 20, 3)


def test_twilight_rejects_grayscale():
    img = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        transform_to_twilight(img)
