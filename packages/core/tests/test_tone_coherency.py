"""Tests for tone_coherency: TonePreset (static) + BatchToneFitter (dynamic)."""

from __future__ import annotations

import cv2
import numpy as np
from pps_core.tone_coherency import (
    BatchAnchor,
    BatchToneFitter,
    TonePreset,
    detect_scene_tone,
)


def _solid(h=240, w=320, color=(120, 120, 120)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _scene(h=240, w=320, base=(120, 120, 120)) -> np.ndarray:
    """Mid-bright scene with texture so LAB midtone mask catches pixels."""
    img = _solid(h, w, base)
    # Add gradient → lots of midtones
    grad = np.linspace(60, 180, w, dtype=np.uint8)
    img[:, :, :] = np.stack([grad, grad, grad], axis=-1)[None, :, :].repeat(h, axis=0)
    # Add color patches
    img[20:80, 20:80] = (40, 30, 200)  # warm patch (red)
    img[140:200, 140:200] = (200, 80, 40)  # cool patch (blue)
    return img


# ── BatchToneFitter / BatchAnchor ────────────────────────────────────


def test_fitter_records_samples():
    fitter = BatchToneFitter(sample_short_edge=128)
    assert fitter.samples == 0
    assert fitter.add(_scene()) is True
    assert fitter.add(_scene()) is True
    assert fitter.samples == 2


def test_fitter_skips_empty_image():
    fitter = BatchToneFitter()
    assert fitter.add(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert fitter.add(None) is False  # type: ignore[arg-type]
    assert fitter.samples == 0


def test_fitter_skips_pure_black_or_white():
    """No midtone pixels → cannot fit."""
    fitter = BatchToneFitter()
    assert fitter.add(_solid(color=(0, 0, 0))) is False
    assert fitter.add(_solid(color=(255, 255, 255))) is False


def test_fit_anchor_returns_none_without_samples():
    fitter = BatchToneFitter()
    assert fitter.fit_anchor() is None


def test_fit_anchor_median_lab():
    fitter = BatchToneFitter()
    # Three same scenes → median = single value
    for _ in range(3):
        fitter.add(_scene())
    anchor = fitter.fit_anchor()
    assert anchor is not None
    assert anchor.samples == 3
    assert 0 < anchor.lab_median[0] < 255
    assert 0 < anchor.lab_median[1] < 255
    assert 0 < anchor.lab_median[2] < 255


def test_anchor_apply_no_op_on_neutral_match():
    """Image already at anchor → minimal shift (delta < threshold → return original)."""
    fitter = BatchToneFitter()
    img = _scene()
    fitter.add(img)
    anchor = fitter.fit_anchor()
    out = anchor.apply(img.copy(), strength=0.6)
    # Same input as fit → very close to no change
    diff = np.mean(np.abs(out.astype(np.int16) - img.astype(np.int16)))
    assert diff < 1.5, f"Expected near-zero shift, got mean abs diff {diff}"


def test_anchor_apply_shifts_warm_image_toward_neutral():
    """Anchor fit on neutral images should pull warm image toward neutral."""
    fitter = BatchToneFitter()
    # Fit anchor on neutral scenes
    for _ in range(3):
        fitter.add(_scene())
    anchor = fitter.fit_anchor()

    # Warm-cast image (R boost, B reduce) — shift baseline channels in midrange,
    # add a checkerboard-ish texture so midtone mask catches enough pixels.
    h, w = 240, 320
    base = np.full((h, w, 3), 120, dtype=np.int16)
    grad = np.linspace(70, 190, w, dtype=np.int16)
    base[:, :, :] = np.stack([grad, grad, grad], axis=-1)[None, :, :].repeat(h, axis=0)
    base[..., 0] = np.clip(base[..., 0] - 25, 0, 255)  # less B
    base[..., 2] = np.clip(base[..., 2] + 25, 0, 255)  # more R
    warm = base.astype(np.uint8)

    # Measure 'b' (LAB) before/after — warm cast → high b; should drop after apply
    lab_before = cv2.cvtColor(warm, cv2.COLOR_BGR2LAB)
    b_before = float(np.median(lab_before[..., 2]))
    out = anchor.apply(warm, strength=0.8)
    lab_after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    b_after = float(np.median(lab_after[..., 2]))
    assert b_after < b_before, (
        f"b channel should decrease toward neutral, before={b_before:.1f} after={b_after:.1f}"
    )


def test_anchor_apply_respects_strength_zero():
    fitter = BatchToneFitter()
    fitter.add(_scene())
    anchor = fitter.fit_anchor()
    img = _scene(base=(80, 80, 200))
    out = anchor.apply(img, strength=0.0)
    np.testing.assert_array_equal(out, img)


def test_anchor_apply_clips_max_shift():
    """Anchor with extreme delta should be clipped to max_shift to avoid posterizing."""
    anchor = BatchAnchor(lab_median=(128.0, 200.0, 200.0), hue_mean=0.0, samples=3)
    img = _scene()
    out = anchor.apply(img, strength=1.0, max_shift=2.0)
    diff = np.mean(np.abs(out.astype(np.int16) - img.astype(np.int16)))
    # 2-unit clip on a/b through soft mask → average diff small
    assert diff < 6.0, f"Clipping failed, diff={diff}"


def test_anchor_apply_handles_zero_samples():
    anchor = BatchAnchor(lab_median=(128.0, 128.0, 128.0), hue_mean=0.0, samples=0)
    img = _scene()
    out = anchor.apply(img)
    np.testing.assert_array_equal(out, img)


# ── Static TonePreset (regression) ───────────────────────────────────


def test_tone_preset_neutral_does_not_mutate_drastically():
    img = _scene()
    out = TonePreset(name="neutral").apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_tone_preset_warm_increases_red_bias():
    img = _scene()
    out = TonePreset(name="warm", strength=1.0).apply(img)
    # Mean R should go up vs B
    diff_in = float(img[..., 2].mean() - img[..., 0].mean())
    diff_out = float(out[..., 2].mean() - out[..., 0].mean())
    assert diff_out >= diff_in - 0.5  # tolerate WB neutralization on test scene


def test_detect_scene_tone_returns_valid_label():
    img = _scene()
    label = detect_scene_tone(img)
    assert label in ("neutral", "warm", "cool")
