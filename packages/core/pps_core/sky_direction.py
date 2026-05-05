"""Sky light-direction matching — flip sky asset để match shadow trong scene.

Lý do: sky asset có "sun position" cố định (vd: nắng từ phải). Scene của user có
"shadow direction" khác (vd: nắng từ trái). Composite nắng mâu thuẫn = uncanny.

Pipeline:
1. Detect dominant shadow direction trong scene (gradient của brightness theo vertical/horizontal)
2. Detect sun direction trong sky asset (brightest cluster center vs frame center)
3. Flip sky asset horizontally nếu mâu thuẫn

Trả: sky_asset đã flip nếu cần.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_scene_light_direction(img: np.ndarray) -> str:
    """Detect direction nắng trong scene: 'left', 'right', 'top', 'unknown'.

    Heuristic: chia ảnh 4 phần, đo brightness mỗi phần. Phần sáng nhất = nguồn sáng.
    Real-world: shadows OPPOSITE to sun → ngược lại brightness.

    Đơn giản hơn: assume nắng từ vùng có building lit nhất (top-left vs top-right).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Top half (sky+upper part) → đo brightness left vs right half
    top = gray[: h // 2]
    left_mean = float(top[:, : w // 2].mean())
    right_mean = float(top[:, w // 2 :].mean())

    diff = left_mean - right_mean
    if abs(diff) < 5:  # < 5 brightness units = không xác định
        return "unknown"
    return "left" if diff > 0 else "right"


def detect_sky_sun_position(sky_asset: np.ndarray) -> str:
    """Detect 'left' / 'right' / 'center' / 'unknown' dựa vào pixel sáng nhất.

    Sun trong sky asset thường là cluster pixel V > 240. Nếu không có (sky đều đặn,
    không có sun visible) → return 'unknown'.
    """
    hsv = cv2.cvtColor(sky_asset, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    h, w = V.shape
    bright_mask = (V > 240).astype(np.uint8)
    if bright_mask.sum() < (h * w) * 0.001:
        # < 0.1% pixel sun → no visible sun
        return "unknown"
    # Center of mass cho bright cluster
    M = cv2.moments(bright_mask)
    if M["m00"] == 0:
        return "unknown"
    cx = M["m10"] / M["m00"]
    rel_x = cx / w  # 0..1
    if rel_x < 0.35:
        return "left"
    elif rel_x > 0.65:
        return "right"
    return "center"


def match_sky_to_scene_direction(
    sky_asset: np.ndarray, scene: np.ndarray,
) -> np.ndarray:
    """Flip sky asset horizontally nếu sun direction mâu thuẫn với scene shadow."""
    scene_dir = detect_scene_light_direction(scene)
    sky_dir = detect_sky_sun_position(sky_asset)
    if scene_dir == "unknown" or sky_dir == "unknown" or sky_dir == "center":
        return sky_asset
    # Mâu thuẫn → flip
    if scene_dir != sky_dir:
        logger.info("sky direction match: scene=%s sky=%s → flip horizontal",
                    scene_dir, sky_dir)
        return cv2.flip(sky_asset, 1)
    return sky_asset
