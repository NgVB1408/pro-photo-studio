import numpy as np
import pytest

from pps_core.utils import ensure_dir, read_image, safe_filename, write_image


def test_safe_filename_strips_specials():
    assert safe_filename("hello world!@#.jpg") == "hello_world_.jpg"
    assert safe_filename("../etc/passwd") == "etc_passwd"
    assert safe_filename("") == "file"


def test_safe_filename_truncates():
    long = "a" * 200
    assert len(safe_filename(long, max_len=80)) == 80


def test_ensure_dir_creates(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    out = ensure_dir(target)
    assert out.is_dir()


def test_read_write_roundtrip(tmp_path):
    img = np.full((20, 30, 3), 100, dtype=np.uint8)
    img[5, 5] = (10, 20, 30)
    path = tmp_path / "x.png"
    write_image(path, img)
    loaded = read_image(path)
    assert loaded.shape[:2] == (20, 30)
    assert tuple(loaded[5, 5]) == (10, 20, 30)


def test_read_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.png")
