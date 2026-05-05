"""Tests for AI sky segmentation wrapper.

Hai chế độ:
  1. Khi rembg cài → integration test thực sự
  2. Khi không cài → fallback heuristic phải work, không raise
"""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest
from pps_core.sky_seg_ai import (
    detect_sky_mask_ai,
    detect_sky_mask_smart,
    is_available,
)

REMBG_AVAILABLE = importlib.util.find_spec("rembg") is not None


def _outdoor_with_sky(h: int = 480, w: int = 720) -> np.ndarray:
    """Synth ảnh exterior: gradient sky trên + building giữa + grass dưới."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient (BGR — cao = blue tươi, thấp = blend)
    for y in range(int(h * 0.5)):
        t = y / max(1, int(h * 0.5))
        img[y, :] = [220 - int(60 * t), 170 - int(40 * t), 110 - int(30 * t)]
    # Building dải giữa
    cv2.rectangle(img, (240, int(h * 0.45)), (520, int(h * 0.85)), (90, 90, 90), -1)
    # Grass dưới
    img[int(h * 0.85) :, :] = (50, 130, 60)
    return img


def test_fallback_when_rembg_unavailable_no_raise():
    """Function luôn trả mask hợp lệ, không raise dù rembg không cài."""
    img = _outdoor_with_sky()
    mask = detect_sky_mask_ai(img, fallback=True)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    # Sky của ảnh synth có ratio đủ → mask không zero
    assert mask.sum() > 0


def test_smart_routes_through_correct_backend():
    """detect_sky_mask_smart không raise và chọn đúng backend."""
    img = _outdoor_with_sky()
    info: dict = {}
    mask = detect_sky_mask_smart(img, prefer="ai", debug_info=info)
    assert mask.shape == img.shape[:2]
    assert "mode" in info
    if REMBG_AVAILABLE:
        assert info["mode"].startswith("rembg") or info["mode"] in (
            "skip_indoor",
            "fallback_heuristic",
            "fallback_after_error",
        )
    else:
        assert info["mode"] == "fallback_heuristic"


def test_indoor_returns_empty():
    """Ảnh indoor (top half = ceiling beige) → outdoor check fail → mask rỗng."""
    img = np.full((480, 720, 3), (220, 220, 215), dtype=np.uint8)  # beige flat
    cv2.rectangle(img, (200, 200), (520, 460), (180, 150, 130), -1)  # furniture
    info: dict = {}
    mask = detect_sky_mask_ai(img, fallback=True, debug_info=info)
    # Outdoor gate sẽ reject ảnh indoor
    assert info.get("mode") == "skip_indoor" or mask.sum() == 0


def test_no_fallback_raises_when_unavailable():
    """fallback=False + rembg không cài → RuntimeError."""
    if REMBG_AVAILABLE:
        pytest.skip("rembg đã cài — không test path này")
    img = _outdoor_with_sky()
    with pytest.raises(RuntimeError):
        detect_sky_mask_ai(img, fallback=False, require_outdoor=False)


def test_is_available_returns_bool():
    """is_available không raise, trả bool."""
    assert isinstance(is_available(), bool)


@pytest.mark.skipif(not REMBG_AVAILABLE, reason="rembg chưa cài — skip integration test")
def test_rembg_path_actually_runs():
    """End-to-end với rembg installed — chỉ chạy khi extras [sky-ai] cài."""
    img = _outdoor_with_sky()
    info: dict = {}
    mask = detect_sky_mask_ai(img, fallback=False, require_outdoor=False, debug_info=info)
    assert mask.shape == img.shape[:2]
    assert info["mode"].startswith("rembg")
    # Sky phải được detect (synth scene có 50% sky)
    assert mask.sum() > 0
