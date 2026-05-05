import numpy as np
import pytest

from pps_core.quality import (
    QualityReport,
    compare,
    compare_files,
    psnr,
    ssim,
    watermark_residual,
)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


def test_psnr_identical_is_inf():
    a = _img()
    assert psnr(a, a) == float("inf")


def test_psnr_decreases_with_noise():
    a = _img()
    b = a.copy()
    b[:5, :5] = 0
    score = psnr(a, b)
    assert 0 < score < 50


def test_ssim_identical_is_one():
    a = _img()
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_compare_returns_report():
    a = _img(0)
    b = _img(1)
    report = compare(a, b)
    assert isinstance(report, QualityReport)
    d = report.as_dict()
    assert {"psnr", "ssim", "mae", "max_diff", "different_pixels_ratio"} <= set(d)


def test_compare_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare(_img(), np.zeros((10, 10, 3), dtype=np.uint8))


def test_compare_files(tmp_path):
    from pps_core.utils import write_image
    a = _img(7)
    b = a.copy()
    b[10:20, 10:20] = 0
    pa = write_image(tmp_path / "a.png", a)
    pb = write_image(tmp_path / "b.png", b)
    report = compare_files(pa, pb)
    assert report.psnr < float("inf")
    assert report.different_pixels_ratio > 0


def test_watermark_residual_counts_bright_dark_in_mask():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    img[10:20, 10:20] = 255  # bright
    img[30:40, 30:40] = 0    # dark
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[30:40, 30:40] = 255
    info = watermark_residual(img, mask)
    assert info["bright_residual"] == 100
    assert info["dark_residual"] == 100
    assert info["ratio"] == 1.0


def test_watermark_residual_empty_mask():
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    info = watermark_residual(img, np.zeros((10, 10), dtype=np.uint8))
    assert info["checked"] == 0
    assert info["ratio"] == 0.0
