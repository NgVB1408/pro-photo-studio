import cv2
import numpy as np

from pps_core.detect import (
    auto_mask,
    detect_bright_logo,
    detect_edge_anomaly,
    detect_text_mser,
)


def _gradient_with_white_logo(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :] = (50 + y // 4, 80 + y // 5, 120 - y // 6)
    cv2.putText(
        img, "LOGO",
        (w - 110, h - 20), cv2.FONT_HERSHEY_DUPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return img


def test_detect_bright_finds_white_logo():
    img = _gradient_with_white_logo()
    mask = detect_bright_logo(img, brightness_threshold=230)
    assert mask.shape == img.shape[:2]
    # Logo nằm ở góc dưới-phải
    assert mask[180:, 200:].sum() > 0
    # Nền không có pixel sáng
    assert mask[:50, :100].sum() == 0


def test_detect_text_mser_returns_mask():
    img = _gradient_with_white_logo()
    mask = detect_text_mser(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8


def test_detect_edge_anomaly_returns_uint8():
    img = _gradient_with_white_logo()
    mask = detect_edge_anomaly(img, block=32)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8


def test_auto_mask_combines_strategies():
    img = _gradient_with_white_logo()
    mask = auto_mask(img, strategy="auto", border_only=True, dilate_iters=2)
    assert mask.shape == img.shape[:2]
    # Logo phải bị bắt
    assert mask[180:, 200:].sum() > 0


def test_auto_mask_specific_strategy():
    img = _gradient_with_white_logo()
    bright = auto_mask(img, strategy="bright", dilate_iters=0)
    text = auto_mask(img, strategy="text", dilate_iters=0)
    edge = auto_mask(img, strategy="edge", dilate_iters=0)
    # mỗi cái có thể trống hay không tuỳ ảnh, nhưng phải là uint8 hợp lệ
    for m in (bright, text, edge):
        assert m.dtype == np.uint8
        assert m.shape == img.shape[:2]
