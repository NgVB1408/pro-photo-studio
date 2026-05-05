import pytest
from pps_core.config import load_settings


def test_load_defaults(monkeypatch):
    for key in [
        "UNSPLASH_ACCESS_KEY",
        "INPAINT_BACKEND",
        "OPENCV_METHOD",
        "OPENCV_RADIUS",
        "LAMA_DEVICE",
        "OUTPUT_DIR",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    s = load_settings(env_file=None)
    assert s.inpaint_backend == "opencv"
    assert s.opencv_method == "telea"
    assert s.opencv_radius == 3
    assert s.lama_device == "auto"
    assert s.unsplash_access_key is None


def test_invalid_backend(monkeypatch):
    monkeypatch.setenv("INPAINT_BACKEND", "wrong")
    with pytest.raises(ValueError):
        load_settings(env_file=None)


def test_invalid_method(monkeypatch):
    monkeypatch.setenv("OPENCV_METHOD", "foo")
    with pytest.raises(ValueError):
        load_settings(env_file=None)


def test_invalid_radius(monkeypatch):
    monkeypatch.setenv("OPENCV_RADIUS", "0")
    with pytest.raises(ValueError):
        load_settings(env_file=None)


def test_require_unsplash(monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    s = load_settings(env_file=None)
    with pytest.raises(RuntimeError, match="UNSPLASH_ACCESS_KEY"):
        s.require_unsplash()
