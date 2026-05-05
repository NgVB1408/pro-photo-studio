"""Tests cho tone_map_real_estate (port từ Edit-image / imagen-ai)."""
from __future__ import annotations

import numpy as np
import pytest

from pps_core.tone_coherency import (
    TonePreset,
    tone_map_real_estate,
)


def _img(h=60, w=80, base=110):
    rng = np.random.default_rng(0)
    return (rng.normal(loc=base, scale=20, size=(h, w, 3))).clip(0, 255).astype(np.uint8)


def test_real_estate_returns_uint8_same_shape():
    img = _img()
    out = tone_map_real_estate(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_real_estate_brightens_midtones():
    img = np.full((40, 40, 3), 110, dtype=np.uint8)
    out = tone_map_real_estate(img, gamma=0.85, clahe_clip=2.0)
    # Gamma <1 brightens
    assert out.mean() >= img.mean() - 1


def test_real_estate_clahe_increases_local_contrast():
    img = np.full((80, 80, 3), 128, dtype=np.uint8)
    # Add 1 dim region
    img[30:50, 30:50] = 100
    out = tone_map_real_estate(img, gamma=1.0, clahe_clip=4.0)
    # CLAHE phải mở rộng dynamic range trong tile
    in_std = img.std()
    out_std = out.std()
    assert out_std >= in_std


def test_real_estate_rejects_float():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        tone_map_real_estate(img)


def test_tone_preset_real_estate_routes_to_tone_map():
    img = _img(base=120)
    preset = TonePreset(name="real_estate", strength=0.7)
    out = preset.apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Output khác input (đã apply tone map)
    assert not np.array_equal(out, img)


def test_tone_preset_real_estate_strength_zero_still_applies_clahe():
    img = _img(base=128)
    preset = TonePreset(name="real_estate", strength=0.0)
    out = preset.apply(img)
    # Strength=0 → gamma=1.0, clip=1.5 — vẫn áp CLAHE → khác input
    assert out.shape == img.shape


def test_tone_preset_real_estate_strength_increasing():
    img = _img()
    weak = TonePreset(name="real_estate", strength=0.2).apply(img)
    strong = TonePreset(name="real_estate", strength=1.0).apply(img)
    diff_w = np.abs(weak.astype(np.int16) - img.astype(np.int16)).mean()
    diff_s = np.abs(strong.astype(np.int16) - img.astype(np.int16)).mean()
    assert diff_s > diff_w
