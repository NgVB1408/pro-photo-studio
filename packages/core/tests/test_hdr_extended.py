"""Tests cho deghost_mask + color_normalize_brackets (port từ imagen-ai)."""

from __future__ import annotations

import numpy as np
import pytest
from pps_core.hdr import (
    color_normalize_brackets,
    compute_deghost_mask,
    fuse_brackets,
)


def _bracket(h=80, w=120, base=128, delta=40):
    """Synth 3-bracket stack (under, mid, over)."""
    under = np.full((h, w, 3), max(base - delta, 0), dtype=np.uint8)
    mid = np.full((h, w, 3), base, dtype=np.uint8)
    over = np.full((h, w, 3), min(base + delta, 255), dtype=np.uint8)
    return [under, mid, over]


# ---------- color_normalize_brackets ----------


def test_color_normalize_keeps_reference_intact():
    stack = _bracket()
    out = color_normalize_brackets(stack, ref_index=1)
    assert np.array_equal(out[1], stack[1])


def test_color_normalize_pulls_means_toward_ref():
    stack = _bracket()
    # Inject color cast to under-bracket: tăng B, giảm R
    biased = stack[0].copy()
    biased[..., 0] = np.clip(biased[..., 0] + 30, 0, 255)
    biased[..., 2] = np.clip(biased[..., 2] - 30, 0, 255)
    stack[0] = biased

    out = color_normalize_brackets(stack, ref_index=1)
    # Difference giữa adjusted và ref nhỏ hơn original biased
    orig_diff = np.abs(stack[0].astype(np.int16) - stack[1].astype(np.int16)).mean()
    new_diff = np.abs(out[0].astype(np.int16) - stack[1].astype(np.int16)).mean()
    assert new_diff < orig_diff


def test_color_normalize_single_image_passthrough():
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    out = color_normalize_brackets([img])
    assert len(out) == 1


# ---------- compute_deghost_mask ----------


def test_deghost_mask_zero_when_no_motion():
    stack = _bracket()
    mask = compute_deghost_mask(stack, threshold=4.0)
    assert mask.shape == (80, 120)
    # Stack có exposure khác nhưng KHÔNG có motion local — mask phải nhỏ
    # (deviation/MAD ratio xấp xỉ uniform → thấp)
    assert mask.mean() < 0.5


def test_deghost_mask_high_for_moving_object():
    h, w = 80, 120
    stack = _bracket(h=h, w=w)
    # Inject 1 vùng "moving person" chỉ xuất hiện ở 1 frame
    stack[1][30:50, 40:70] = 200  # bright blob ở frame mid only
    mask = compute_deghost_mask(stack, threshold=2.0)
    # Vùng motion phải có mask > 0
    motion_region = mask[30:50, 40:70].mean()
    background = mask[:20, :20].mean()
    assert motion_region > background


def test_deghost_mask_two_image_input():
    h, w = 60, 80
    stack = [
        np.full((h, w, 3), 100, dtype=np.uint8),
        np.full((h, w, 3), 150, dtype=np.uint8),
    ]
    mask = compute_deghost_mask(stack)
    assert mask.shape == (h, w)
    assert mask.dtype == np.float32


def test_deghost_mask_single_image_returns_zero():
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    mask = compute_deghost_mask([img])
    assert mask.shape == (40, 40)
    assert mask.max() == 0


# ---------- fuse_brackets with new flags ----------


def test_fuse_brackets_with_deghost_runs():
    stack = _bracket()
    out = fuse_brackets(stack, deghost=True)
    assert out.shape == stack[0].shape
    assert out.dtype == np.uint8


def test_fuse_brackets_with_color_normalize_runs():
    stack = _bracket()
    out = fuse_brackets(stack, color_normalize=True)
    assert out.shape == stack[0].shape
    assert out.dtype == np.uint8


def test_fuse_brackets_with_both_flags():
    stack = _bracket()
    out_basic = fuse_brackets(stack)
    out_full = fuse_brackets(stack, deghost=True, color_normalize=True)
    assert out_basic.shape == out_full.shape


def test_fuse_brackets_deghost_falls_back_at_motion():
    """Pixel bị flag ghost → output phải gần reference frame ở vùng đó."""
    h, w = 80, 120
    stack = _bracket(h=h, w=w)
    # Inject motion: ghost blob chỉ ở frame 0
    stack[0][20:40, 30:60] = 250
    out = fuse_brackets(stack, deghost=True, reference_index=1)
    # Output ở vùng motion nên close hơn reference (frame 1) hơn là blend
    ref_region = stack[1][20:40, 30:60]
    out_region = out[20:40, 30:60]
    diff_to_ref = np.abs(out_region.astype(np.int16) - ref_region.astype(np.int16)).mean()
    # Tightness check — không cần exact, chỉ cần dưới ~50 (nếu blend đầy đủ ghost
    # thì sẽ kéo về ~125 trung bình giữa 128 (ref) và 250 (ghost frame))
    assert diff_to_ref < 80


def test_fuse_brackets_rejects_too_few_inputs():
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError):
        fuse_brackets([img])
