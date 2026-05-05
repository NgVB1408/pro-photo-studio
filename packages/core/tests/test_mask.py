import numpy as np
import pytest

from pps_core.mask import (
    build_mask_from_boxes,
    build_mask_from_color,
    build_mask_from_image,
    combine_masks,
    dilate_mask,
)


def _img(h=20, w=30):
    return np.full((h, w, 3), 50, dtype=np.uint8)


def test_mask_from_single_box():
    # box = (x=5, y=5, w=10, h=8) -> vùng filled là rows 5..12, cols 5..14
    img = _img(20, 30)
    mask = build_mask_from_boxes(img, [(5, 5, 10, 8)])
    assert mask.shape == (20, 30)
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 255          # góc trên-trái
    assert mask[12, 14] == 255        # góc dưới-phải (row 12, col 14)
    assert mask[8, 10] == 255         # giữa
    assert mask[4, 5] == 0            # ngay trên top edge
    assert mask[5, 4] == 0            # ngay trái left edge
    assert mask[13, 14] == 0          # ngay dưới bottom edge
    assert mask[12, 15] == 0          # ngay phải right edge (exclusive)


def test_mask_from_multiple_boxes_clamped_to_image():
    img = _img(20, 30)
    mask = build_mask_from_boxes(img, [(0, 0, 5, 5), (28, 18, 100, 100)])
    assert mask[0, 0] == 255
    assert mask[19, 29] == 255  # clamped to last pixel
    assert mask[10, 10] == 0


def test_mask_from_box_rejects_invalid():
    img = _img()
    with pytest.raises(ValueError):
        build_mask_from_boxes(img, [(0, 0, 0, 5)])
    with pytest.raises(ValueError):
        build_mask_from_boxes(img, [(0, 0, 5)])  # type: ignore[arg-type]


def test_mask_from_color_threshold():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[3:7, 3:7] = (250, 250, 250)
    mask = build_mask_from_color(img, lower=(200, 200, 200), upper=(255, 255, 255))
    assert mask[5, 5] == 255
    assert mask[0, 0] == 0


def test_mask_from_image_resizes_if_needed(tmp_path):
    import cv2

    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[10:30, 10:30] = 255
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), raw)

    img = _img(20, 30)
    mask = build_mask_from_image(img, p)
    assert mask.shape == (20, 30)


def test_dilate_mask_grows_region():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    grown = dilate_mask(mask, iterations=2, kernel_size=3)
    assert grown[10, 10] == 255
    assert grown[8, 8] == 255  # 2-iter cross of 3x3 reaches diag-2


def test_combine_masks_or():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0, 0] = 255
    b[9, 9] = 255
    out = combine_masks([a, b])
    assert out[0, 0] == 255
    assert out[9, 9] == 255
    assert out[5, 5] == 0


def test_image_must_be_uint8():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        build_mask_from_boxes(img, [(0, 0, 1, 1)])
