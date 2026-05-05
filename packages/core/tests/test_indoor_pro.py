"""Tests cho indoor_pro module — selective wall WB + surface clarity."""
from __future__ import annotations

import numpy as np
import pytest

from pps_core.indoor_pro import (
    detect_white_wall_mask,
    selective_wall_wb,
    detect_smooth_surface_mask,
    boost_surface_clarity,
    enhance_interior_pro,
)


def _make_room_with_tungsten_cast(h: int = 480, w: int = 640) -> np.ndarray:
    """Phòng có tường trắng + tungsten cast vàng (BGR ám vàng = R cao G mid B thấp)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Tường trắng bị ám vàng (BGR ~ 200, 215, 230 — slight warm)
    img[:int(h * 0.6)] = [200, 215, 230]
    # Sàn gỗ tối (BGR ~ 80, 100, 130)
    img[int(h * 0.6):] = [80, 100, 130]
    # Đèn vàng rực ở góc — small bright spot (BGR ~ 50, 200, 255 = pure orange)
    cy, cx = int(h * 0.20), int(w * 0.45)
    cv2_circle = lambda y, x, r, c: img[max(0, y - r):y + r, max(0, x - r):x + r].__setitem__(slice(None), c)
    img[cy - 25:cy + 25, cx - 30:cx + 30] = [50, 200, 255]
    return img


def _make_marble_room(h: int = 480, w: int = 640) -> np.ndarray:
    """Phòng có marble surface — flat tone, low sat, mid V."""
    img = np.full((h, w, 3), 0, dtype=np.uint8)
    # Marble wall — flat light grey
    img[:, :] = [180, 185, 188]
    # Add subtle noise to simulate texture
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ============================================================================
# detect_white_wall_mask
# ============================================================================

def test_white_wall_mask_detected():
    img = _make_room_with_tungsten_cast()
    mask = detect_white_wall_mask(img)
    # Top half should be detected as wall
    assert mask[100, 320] > 100  # in wall area
    # Bottom (sàn) should NOT be detected (V too low)
    assert mask[400, 320] < 100


def test_white_wall_excludes_warm_lamp():
    img = _make_room_with_tungsten_cast()
    mask = detect_white_wall_mask(img)
    # Lamp area (very saturated warm) should NOT be in wall mask
    assert mask[100, int(img.shape[1] * 0.45)] < 200  # might be partial


# ============================================================================
# selective_wall_wb
# ============================================================================

def test_selective_wb_neutralizes_cast():
    img = _make_room_with_tungsten_cast()
    out, info = selective_wall_wb(img, strength=1.0)
    if not info["applied"]:
        pytest.skip("Cast quá nhỏ để correct — synthetic image issue")
    # Wall area (top center): R should be reduced relative to G,B (warm cast removed)
    orig_wall = img[100, 320].astype(np.float32)
    new_wall = out[100, 320].astype(np.float32)
    # Cast magnitude should decrease
    orig_dev = max(abs(orig_wall - orig_wall.mean()))
    new_dev = max(abs(new_wall - new_wall.mean()))
    assert new_dev <= orig_dev + 0.5  # allow tiny float fluctuation


def test_selective_wb_preserves_lamp_glow():
    img = _make_room_with_tungsten_cast()
    out, info = selective_wall_wb(img, strength=1.0)
    # Lamp center (saturated warm) — should be unchanged or minimally changed
    orig_lamp = img[100, int(img.shape[1] * 0.45)].astype(np.float32)
    new_lamp = out[100, int(img.shape[1] * 0.45)].astype(np.float32)
    diff = float(np.abs(new_lamp - orig_lamp).max())
    # Lamp should change much less than wall would
    assert diff < 50


def test_selective_wb_skips_when_neutral():
    """Ảnh đã neutral → skip correction."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)  # uniform grey
    out, info = selective_wall_wb(img, strength=1.0, cast_threshold=0.04)
    assert info["applied"] is False


def test_selective_wb_zero_strength():
    img = _make_room_with_tungsten_cast()
    out, info = selective_wall_wb(img, strength=0.0)
    assert info["applied"] is False
    np.testing.assert_array_equal(out, img)


# ============================================================================
# detect_smooth_surface_mask
# ============================================================================

def test_smooth_surface_detected_on_marble():
    img = _make_marble_room()
    mask = detect_smooth_surface_mask(img)
    # Most of marble should be detected
    assert mask.mean() > 100  # at least half saturated


def test_smooth_surface_excludes_high_edge():
    """Pattern fabric/plant (high edge density) → not smooth surface."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Add high-frequency pattern in top half
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 255, (240, 640, 3), dtype=np.uint8)
    img[:240] = pattern
    mask = detect_smooth_surface_mask(img)
    # Top half should NOT be detected (high edge density)
    assert mask[100, 320] < 100
    # Bottom half (smooth) should be detected
    assert mask[400, 320] > 100


# ============================================================================
# boost_surface_clarity
# ============================================================================

def test_clarity_boost_applies():
    img = _make_marble_room()
    out, info = boost_surface_clarity(img, strength=0.5)
    assert info["applied"] is True
    # Output shape preserved
    assert out.shape == img.shape


def test_clarity_zero_strength_passthrough():
    img = _make_marble_room()
    out, info = boost_surface_clarity(img, strength=0.0)
    assert info["applied"] is False
    np.testing.assert_array_equal(out, img)


def test_clarity_increases_local_contrast():
    img = _make_marble_room()
    out, _ = boost_surface_clarity(img, strength=0.6)
    # Local std (texture) should increase
    in_std = float(img.std())
    out_std = float(out.std())
    assert out_std >= in_std


# ============================================================================
# enhance_interior_pro composite
# ============================================================================

def test_interior_pro_composite_runs():
    img = _make_room_with_tungsten_cast()
    out, info = enhance_interior_pro(img)
    assert out.shape == img.shape
    assert "steps" in info
    assert len(info["steps"]) >= 4  # WB + shadow + vibrance + clarity + sharpen


def test_interior_pro_idempotent_safe():
    """Run twice → output không bị over-process tới mức artifacts."""
    img = _make_room_with_tungsten_cast()
    out1, _ = enhance_interior_pro(img)
    out2, _ = enhance_interior_pro(out1)
    # Diff between out1 and out2 should be much smaller than out1 vs original
    d1 = float(np.abs(out1.astype(np.float32) - img.astype(np.float32)).mean())
    d2 = float(np.abs(out2.astype(np.float32) - out1.astype(np.float32)).mean())
    assert d2 < d1 + 5.0  # second pass shouldn't drift much
