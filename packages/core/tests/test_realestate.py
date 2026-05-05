"""Tests for realestate.py — sky/window/lawn/vertical/classify."""
import cv2
import numpy as np
import pytest

from pps_core.realestate import (
    classify_scene,
    correct_vertical,
    detect_blown_windows,
    detect_lawn_mask,
    detect_sky_mask,
    enhance_lawn,
    enhance_realestate_full,
    replace_sky,
    window_pull,
)


# ---- Synthetic image builders ----

def _exterior_with_sky_and_grass(h=400, w=600):
    """Top half = blue sky, bottom = green grass, middle band = building."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Blue sky (BGR)
    img[: int(h * 0.45)] = (220, 180, 130)  # light blue
    # Building
    img[int(h * 0.45) : int(h * 0.65)] = (110, 110, 115)
    # Grass — vibrant green (BGR)
    img[int(h * 0.65) :] = (60, 160, 80)
    return img


def _interior_with_blown_window(h=400, w=600):
    """Dim room with one bright square 'window'."""
    img = np.full((h, w, 3), 70, dtype=np.uint8)  # dim
    # Bright window
    cv2.rectangle(img, (240, 90), (440, 270), (252, 252, 252), thickness=-1)
    # Some edges (furniture)
    cv2.rectangle(img, (30, 290), (200, 380), (50, 50, 50), thickness=-1)
    cv2.line(img, (0, 380), (w, 380), (40, 40, 40), 3)
    cv2.line(img, (50, 0), (50, h), (40, 40, 40), 3)
    return img


def _tilted_image(angle: float, h=400, w=600):
    """Build image with strong vertical lines, then rotate it."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # Vertical lines (will be detected)
    for x in (100, 200, 300, 400, 500):
        cv2.line(img, (x, 20), (x, h - 20), (0, 0, 0), 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        borderValue=(200, 200, 200),
    )


# ---- detect_sky_mask ----

def test_detect_sky_mask_finds_top_band():
    img = _exterior_with_sky_and_grass()
    mask = detect_sky_mask(img)
    assert mask.shape == img.shape[:2]
    # Top of image should have sky
    assert (mask[:50] > 128).mean() > 0.5
    # Bottom should NOT
    assert (mask[-50:] > 128).mean() < 0.05


def test_detect_sky_mask_empty_for_dark():
    img = np.full((200, 200, 3), 30, dtype=np.uint8)
    mask = detect_sky_mask(img)
    assert mask.sum() == 0


# ---- replace_sky ----

def test_replace_sky_changes_top_band_only():
    img = _exterior_with_sky_and_grass()
    out, mask = replace_sky(img, preset="sunset", blend_strength=1.0, feather=5)
    assert out.shape == img.shape
    # Top should differ (sky was replaced)
    diff_top = np.abs(out[:50].astype(int) - img[:50].astype(int)).mean()
    assert diff_top > 20
    # Bottom (grass) should be virtually identical
    diff_bot = np.abs(out[-50:].astype(int) - img[-50:].astype(int)).mean()
    assert diff_bot < 5


def test_replace_sky_with_custom_image():
    img = _exterior_with_sky_and_grass()
    custom = np.full((100, 100, 3), (10, 50, 200), dtype=np.uint8)
    out, _ = replace_sky(img, sky_image=custom, blend_strength=1.0)
    # Top pixels should be tinted toward red (high R channel in BGR=(10,50,200))
    top_avg = out[:30].mean(axis=(0, 1))
    assert top_avg[2] > top_avg[0]  # R > B


def test_replace_sky_no_sky_returns_original():
    img = np.full((200, 200, 3), 30, dtype=np.uint8)
    out, mask = replace_sky(img)
    assert mask.sum() == 0
    assert np.array_equal(out, img)


# ---- detect_blown_windows + window_pull ----

def test_detect_blown_windows_finds_bright_rect():
    img = _interior_with_blown_window()
    mask = detect_blown_windows(img, value_threshold=240)
    assert mask.shape == img.shape[:2]
    # Window region should be in mask
    assert (mask[120:240, 280:400] > 100).mean() > 0.5


def test_window_pull_darkens_bright_region():
    img = _interior_with_blown_window()
    out, mask = window_pull(img, strength=0.8)
    assert out.shape == img.shape
    # Window region average V should drop
    win_orig = img[120:240, 280:400].mean()
    win_out = out[120:240, 280:400].mean()
    assert win_out < win_orig


# ---- detect_lawn_mask + enhance_lawn ----

def test_detect_lawn_mask_finds_bottom_green():
    img = _exterior_with_sky_and_grass()
    mask = detect_lawn_mask(img)
    # Bottom should be lawn
    assert (mask[-50:] > 100).mean() > 0.4
    # Top (sky) should NOT
    assert (mask[:50] > 100).mean() < 0.05


def test_enhance_lawn_boosts_saturation():
    img = _exterior_with_sky_and_grass()
    out, mask = enhance_lawn(img, sat_boost=0.6)
    # Compare HSV S channel in lawn region
    s_orig = cv2.cvtColor(img[-50:], cv2.COLOR_BGR2HSV)[..., 1].mean()
    s_out = cv2.cvtColor(out[-50:], cv2.COLOR_BGR2HSV)[..., 1].mean()
    assert s_out > s_orig


# ---- correct_vertical ----

def test_correct_vertical_detects_tilt():
    img = _tilted_image(angle=3.0)
    out, report = correct_vertical(img, max_angle=8.0)
    # Should detect non-trivial angle
    assert report.line_count > 0
    # The output may be cropped — just check it has reasonable size
    assert out.shape[0] > 100 and out.shape[1] > 100


def test_correct_vertical_no_tilt_returns_original():
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    out, report = correct_vertical(img)
    assert report.rotated is False
    assert out.shape == img.shape


# ---- classify_scene ----

def test_classify_exterior():
    img = _exterior_with_sky_and_grass()
    report = classify_scene(img)
    assert report.tag in ("exterior", "aerial")
    assert report.confidence > 0.5
    assert report.sky_ratio > 0.1


def test_classify_interior():
    img = _interior_with_blown_window()
    report = classify_scene(img)
    assert report.tag in ("interior", "unknown")
    assert report.sky_ratio < 0.1


# ---- enhance_realestate_full (composite) ----

def test_enhance_realestate_full_exterior():
    img = _exterior_with_sky_and_grass()
    # Disable smart_sky_skip — synthetic test sky là vibrant blue saturated nên
    # smart skip sẽ giữ nguyên (đúng pipeline). Force replace để test path replace.
    out, report = enhance_realestate_full(
        img, sky_preset="blue", smart_sky_skip=False,
    )
    assert out.shape[:2] != (0, 0)
    assert report.scene.tag in ("exterior", "aerial")
    # For exterior: sky should be replaced
    assert report.sky_replaced is True


def test_enhance_realestate_full_smart_skip_beautiful_sky():
    """Smart skip path: vibrant blue test sky → not replaced."""
    img = _exterior_with_sky_and_grass()
    out, report = enhance_realestate_full(
        img, sky_preset="blue", smart_sky_skip=True,
    )
    # Vibrant test sky → smart logic should skip
    assert report.sky_replaced is False
    assert "skip" in report.sky_decision
    assert "beautiful" in report.sky_decision.lower() or "vibrant" in report.sky_decision.lower()


def test_enhance_realestate_full_interior_no_sky():
    img = _interior_with_blown_window()
    out, report = enhance_realestate_full(img)
    # Should NOT replace sky (interior tag)
    assert report.sky_replaced is False


def test_enhance_realestate_full_disable_all():
    img = _exterior_with_sky_and_grass()
    out, report = enhance_realestate_full(
        img,
        enable_sky=False,
        enable_window_pull=False,
        enable_lawn=False,
        enable_vertical=False,
    )
    assert report.sky_replaced is False
    assert report.lawn_enhanced is False
    assert report.windows_recovered is False
    assert report.vertical.rotated is False
    # Output should be unchanged
    assert np.array_equal(out, img)
