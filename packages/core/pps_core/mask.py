"""Mask construction utilities.

Một mask trong watermark removal là ảnh nhị phân (uint8, giá trị 0 hoặc 255):
- 255 = pixel cần inpaint (vùng có watermark/logo).
- 0   = pixel giữ nguyên.

Các API ở đây nhận BGR image (numpy uint8) — phù hợp với OpenCV — và trả về
mask cùng kích thước HxW.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x, y, w, h)


def _ensure_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image phải là numpy.ndarray, nhận {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"image phải có dtype uint8, nhận {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image phải có 2 hoặc 3 chiều, nhận shape={image.shape}")


def build_mask_from_boxes(image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    """Mask từ danh sách bounding box (x, y, w, h)."""
    _ensure_image(image)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for idx, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f"box[{idx}] phải có 4 phần tử (x, y, w, h)")
        x, y, bw, bh = (int(v) for v in box)
        if bw <= 0 or bh <= 0:
            raise ValueError(f"box[{idx}] có width/height không hợp lệ: {box}")
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] = 255
    return mask


def build_mask_from_color(
    image: np.ndarray,
    *,
    lower: tuple[int, int, int] = (200, 200, 200),
    upper: tuple[int, int, int] = (255, 255, 255),
    color_space: str = "bgr",
) -> np.ndarray:
    """Mask theo dải màu — phù hợp watermark sáng/đậm đồng nhất.

    color_space:
      - "bgr": lower/upper là (B, G, R)
      - "hsv": lower/upper là (H, S, V) sau khi convert sang HSV
    """
    _ensure_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color mask cần ảnh BGR 3 kênh")

    cs = color_space.lower()
    if cs == "bgr":
        src = image
    elif cs == "hsv":
        src = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("color_space phải là 'bgr' hoặc 'hsv'")

    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(src, lo, hi)
    return mask


def build_mask_from_image(
    image: np.ndarray,
    mask_path: str | Path,
    *,
    threshold: int = 127,
) -> np.ndarray:
    """Đọc mask từ file ảnh (grayscale hoặc bất kỳ), nhị phân hoá theo threshold.

    Mask file phải cùng kích thước với image (H, W).
    """
    _ensure_image(image)
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mask: {path}")

    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Không đọc được mask từ {path}")

    h, w = image.shape[:2]
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)
    return mask


def dilate_mask(mask: np.ndarray, *, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    """Phồng mask để bao phủ rìa watermark — quan trọng cho chất lượng inpaint."""
    if mask.dtype != np.uint8:
        raise ValueError("mask phải là uint8")
    if iterations <= 0:
        return mask
    if kernel_size < 1:
        raise ValueError("kernel_size >= 1")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    """OR hợp nhiều mask cùng kích thước thành một."""
    out: np.ndarray | None = None
    for m in masks:
        if m.dtype != np.uint8:
            raise ValueError("mask phải là uint8")
        if out is None:
            out = m.copy()
        else:
            if out.shape != m.shape:
                raise ValueError(f"mask shapes khác nhau: {out.shape} vs {m.shape}")
            out = cv2.bitwise_or(out, m)
    if out is None:
        raise ValueError("Cần ít nhất một mask")
    return out
