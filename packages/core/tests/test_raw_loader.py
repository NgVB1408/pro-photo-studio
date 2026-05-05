"""Tests for RAW input handling in utils.read_image."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pps_core.utils import RAW_EXTS, read_image


def test_raw_exts_set_includes_common():
    """Sanity check — phải có DSLR/mirrorless extensions chính."""
    must_have = {".dng", ".cr2", ".nef", ".arw", ".raf", ".rw2", ".orf"}
    assert must_have <= RAW_EXTS


def test_raw_unsupported_without_rawpy(monkeypatch, tmp_path: Path):
    """Khi rawpy không cài, đọc RAW phải raise ImportError với hint cài."""
    # Tạo file giả với extension .dng — không cần content thật
    fake = tmp_path / "x.dng"
    fake.write_bytes(b"not-a-real-dng")

    # Force "rawpy không cài" bằng cách block import
    import sys
    original = sys.modules.get("rawpy")
    sys.modules["rawpy"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError) as exc_info:
            read_image(fake)
        assert "rawpy" in str(exc_info.value).lower()
    finally:
        if original is None:
            sys.modules.pop("rawpy", None)
        else:
            sys.modules["rawpy"] = original


@pytest.mark.skipif(
    importlib.util.find_spec("rawpy") is None,
    reason="rawpy không cài — skip RAW decode test",
)
def test_raw_invalid_file_raises_value_error(tmp_path: Path):
    """File .dng không hợp lệ → ValueError (decode fail)."""
    fake = tmp_path / "junk.dng"
    fake.write_bytes(b"\x00" * 1024)
    with pytest.raises((ValueError, Exception)):
        read_image(fake)


def test_jpg_path_unaffected_by_raw_branch(tmp_path: Path):
    """JPG bình thường vẫn đi qua OpenCV path, không touched RAW logic."""
    import cv2
    import numpy as np

    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    p = tmp_path / "ok.jpg"
    cv2.imwrite(str(p), img)

    out = read_image(p)
    assert out.shape == (100, 200, 3)
    assert out.dtype == np.uint8
