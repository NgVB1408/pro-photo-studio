"""Tests for preflight QC module."""
from __future__ import annotations

import numpy as np
import pytest

from pps_core.preflight import (
    analyze_image,
    PreflightReport,
    BLUR_FAIL,
    BLUR_WARN,
)


def _solid_color(h: int, w: int, color=(120, 120, 120)) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def _sharp_image(h: int = 1200, w: int = 1800) -> np.ndarray:
    """Tạo ảnh sharp với edge và texture."""
    img = _solid_color(h, w, (90, 90, 90))
    # Thêm checkerboard pattern → high Laplacian variance
    for i in range(0, h, 16):
        for j in range(0, w, 16):
            if (i + j) // 16 % 2 == 0:
                img[i:i+16, j:j+16] = (200, 200, 200)
    # Thêm vài ô màu để có saturation
    img[40:100, 40:100] = (60, 30, 200)   # đỏ
    img[140:200, 140:200] = (50, 200, 80)  # xanh
    return img


def test_sharp_image_passes():
    img = _sharp_image()
    rpt = analyze_image(img)
    assert isinstance(rpt, PreflightReport)
    assert rpt.severity in ("ok", "info"), f"got {rpt.severity}: {rpt.warnings}"
    assert rpt.blur_score > BLUR_WARN
    assert rpt.width == 1800 and rpt.height == 1200


def test_blurry_image_flagged_fail():
    """Gaussian blur mạnh → blur_score giảm rõ → fail."""
    import cv2
    img = _sharp_image()
    img = cv2.GaussianBlur(img, (101, 101), sigmaX=30)
    rpt = analyze_image(img)
    assert rpt.severity == "fail"
    assert rpt.blur_score < BLUR_FAIL
    assert any("Blur" in w for w in rpt.warnings)
    assert "retake" in rpt.suggested_action.lower()


def test_overexposed_image_flagged():
    """Ảnh cháy ≥18% pixel → fail (highlight clipping)."""
    img = _sharp_image()  # 1200x1800
    img[:300, :] = 254  # 300/1200 = 25% ảnh cháy
    rpt = analyze_image(img)
    assert rpt.highlight_clip_pct >= 18.0
    assert rpt.severity == "fail"
    assert any("Cháy" in w for w in rpt.warnings)


def test_dark_image_flagged():
    """Ảnh quá tối → fail."""
    img = np.full((480, 720, 3), 15, dtype=np.uint8)
    rpt = analyze_image(img)
    assert rpt.severity == "fail"
    assert rpt.avg_brightness < 35


def test_low_resolution_flagged():
    """Ảnh < 720 short side → fail; 720..1080 → info."""
    img = _sharp_image(h=500, w=600)
    rpt = analyze_image(img)
    assert rpt.severity == "fail"
    assert any("nhỏ" in w.lower() for w in rpt.warnings)


def test_resolution_warn_zone():
    """Ảnh 800x1200 → info (low-res nhưng còn dùng được, ≥720)."""
    img = _sharp_image(h=820, w=1200)
    rpt = analyze_image(img)
    # 820 short side ∈ [720, 1080) → info, không fail
    assert rpt.severity in ("ok", "info"), f"got {rpt.severity}: {rpt.warnings}"


def test_color_cast_detected():
    """Ảnh ngả vàng đậm (B<<R) → cảnh báo WB."""
    img = _sharp_image()
    img = img.astype(np.int16)
    img[..., 0] = np.clip(img[..., 0] - 60, 0, 255)  # giảm B
    img[..., 2] = np.clip(img[..., 2] + 30, 0, 255)  # tăng R
    img = img.astype(np.uint8)
    rpt = analyze_image(img)
    # Color cast metric > 0; có thể trigger info hoặc warn
    assert rpt.color_cast > 5.0


def test_csv_summary_format():
    img = _sharp_image()
    rpt = analyze_image(img)
    summary = rpt.csv_summary()
    if rpt.severity == "ok":
        assert summary == "ok"
    else:
        assert ":" in summary  # "warn: ..." or "fail: ..."


def test_empty_image_handled():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    rpt = analyze_image(img)
    assert rpt.severity == "fail"


def test_as_dict_serializable():
    """Verify as_dict() để CSV writer dùng được."""
    img = _sharp_image()
    rpt = analyze_image(img)
    d = rpt.as_dict()
    assert "blur_score" in d and isinstance(d["blur_score"], float)
    assert "severity" in d
    assert "warnings" in d and isinstance(d["warnings"], list)
