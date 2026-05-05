import numpy as np
import pytest
from pps_core.inpaint import InpaintBackend, inpaint, inpaint_opencv


def _gradient(h=40, w=60):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :] = (y * 3 % 256, (y * 5 + 30) % 256, (y * 7 + 80) % 256)
    return img


def test_opencv_telea_fills_masked_region():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[10:20, 20:40] = 255

    # Đặt vùng mask thành màu lạ rõ rệt để thấy khác biệt sau inpaint
    polluted = img.copy()
    polluted[10:20, 20:40] = (0, 255, 0)

    out = inpaint_opencv(polluted, mask, method="telea", radius=3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Sau inpaint, vùng được khôi phục KHÔNG còn màu xanh lá nguyên thuỷ
    assert not np.array_equal(out[10:20, 20:40], polluted[10:20, 20:40])


def test_opencv_ns_works():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[5:8, 5:8] = 255
    out = inpaint_opencv(img, mask, method="ns", radius=2)
    assert out.shape == img.shape


def test_dispatcher_routes_to_opencv():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = inpaint(img, mask, backend=InpaintBackend.OPENCV)
    assert out.shape == img.shape


def test_dispatcher_string_backend():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = 255
    out = inpaint(img, mask, backend="opencv")
    assert out.shape == img.shape


def test_invalid_backend_raises():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[0, 0] = 255
    with pytest.raises(ValueError):
        inpaint(img, mask, backend="nonsense")


def test_empty_mask_raises():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    with pytest.raises(ValueError):
        inpaint_opencv(img, mask)


def test_size_mismatch_raises():
    img = _gradient(40, 60)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5, 5] = 255
    with pytest.raises(ValueError):
        inpaint_opencv(img, mask)


def test_invalid_method_raises():
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[0, 0] = 255
    with pytest.raises(ValueError):
        inpaint_opencv(img, mask, method="zzz")  # type: ignore[arg-type]


def test_lama_backend_without_iopaint_gives_clear_error(monkeypatch):
    img = _gradient()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[1:3, 1:3] = 255
    # iopaint không có trong môi trường test mặc định -> phải raise RuntimeError có hint
    with pytest.raises(RuntimeError, match="iopaint"):
        inpaint(img, mask, backend="lama")
