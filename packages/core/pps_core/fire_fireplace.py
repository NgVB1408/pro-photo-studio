"""Fire in Fireplace — detect lò sưởi + composite ngọn lửa procedural.

Fireplace photos thường chụp khi lò KHÔNG đốt → trống đen. Auto add fire để
tạo cảm giác ấm cúng + bán nhà tốt hơn.

Steps:
1. Detect dark rectangular opening (fireplace) — gần đáy frame, dark interior
2. Generate procedural fire texture (Perlin noise + warm gradient)
3. Composite vào opening với glow effect lan ra hearth surround
4. Adjust local exposure quanh fire (warm cast + slight brightness)

Hoàn toàn local, không depend ML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FireplaceDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float


@dataclass
class FireReport:
    detected: int = 0
    fires_added: int = 0
    detections: list[FireplaceDetection] = field(default_factory=list)


def _detect_fireplace_openings(img: np.ndarray) -> list[FireplaceDetection]:
    """Heuristic: tìm dark rectangular opening ở phần dưới ảnh."""
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold dark areas
    _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Morphology để gộp opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[FireplaceDetection] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        rel_area = (w * h) / (w_img * h_img)
        if not (0.005 < rel_area < 0.15):
            continue
        # Phần dưới ảnh
        cy = y + h / 2
        if cy < h_img * 0.4:
            continue
        # Aspect: fireplace opening thường rộng hơn cao
        ar = w / h
        if not (0.7 < ar < 2.5):
            continue
        # Confidence: rectangularity + position score
        contour_area = cv2.contourArea(c)
        rect_area = w * h
        rect_score = contour_area / rect_area if rect_area > 0 else 0
        pos_score = (cy - h_img * 0.4) / (h_img * 0.6)
        conf = float(rect_score * 0.7 + pos_score * 0.3)
        if conf < 0.5:
            continue
        candidates.append(
            FireplaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=conf,
            )
        )
    candidates.sort(key=lambda d: -d.confidence)
    return candidates


def _generate_fire_texture(w: int, h: int, *, seed: int = 42) -> np.ndarray:
    """Procedural fire — gradient warm + Perlin-ish noise + flame shape mask."""
    rng = np.random.default_rng(seed)
    # Multi-octave value noise
    noise = np.zeros((h, w), dtype=np.float32)
    for octave in range(4):
        scale = 2**octave
        small = rng.random((max(1, h // (8 * scale)), max(1, w // (8 * scale)))).astype(np.float32)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += big * (0.6**octave)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    # Vertical flame falloff (nến cháy hướng lên trên)
    yy = np.arange(h, dtype=np.float32) / h  # 0 (top) → 1 (bottom)
    flame_mask_v = np.clip(1.4 - yy * 1.2, 0, 1)  # mạnh nhất ở dưới-giữa
    flame_mask_v = flame_mask_v[:, None]
    # Horizontal taper
    xx = np.arange(w, dtype=np.float32) / w
    flame_mask_h = 1.0 - np.abs(xx - 0.5) * 1.6
    flame_mask_h = np.clip(flame_mask_h, 0, 1)[None, :]

    intensity = noise * flame_mask_v * flame_mask_h

    # Color: dark red → orange → yellow theo intensity
    fire = np.zeros((h, w, 3), dtype=np.float32)
    fire[..., 2] = np.clip(intensity * 1.4, 0, 1) * 255  # R
    fire[..., 1] = np.clip(intensity * 1.0 - 0.2, 0, 1) * 255  # G (hơn 0.2 mới có)
    fire[..., 0] = np.clip(intensity * 0.4 - 0.5, 0, 1) * 255  # B (chỉ vùng nóng nhất)

    # Add white-hot core
    hot_core = (intensity > 0.85).astype(np.float32)
    fire[..., 0] = np.maximum(fire[..., 0], hot_core * 200)
    fire[..., 1] = np.maximum(fire[..., 1], hot_core * 230)
    fire[..., 2] = np.maximum(fire[..., 2], hot_core * 250)

    return fire.astype(np.uint8)


def add_fire_to_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    intensity: float = 0.85,
    glow: bool = True,
) -> np.ndarray:
    """Composite fire vào bbox + glow surround in-place."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return img

    fire = _generate_fire_texture(x2 - x, y2 - y)
    fire_alpha = cv2.cvtColor(fire, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    fire_alpha = np.clip(fire_alpha * 1.5, 0, 1)[..., None]

    out = img.copy()
    region = out[y:y2, x:x2].astype(np.float32)
    # Blend
    blended = (
        region * (1 - fire_alpha * intensity) + fire.astype(np.float32) * fire_alpha * intensity
    )
    out[y:y2, x:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Glow lan ra surround (gradient warm cast)
    if glow:
        glow_radius = int(max(w, h) * 0.4)
        gx1 = max(0, x - glow_radius)
        gy1 = max(0, y - glow_radius)
        gx2 = min(W, x2 + glow_radius)
        gy2 = min(H, y2 + glow_radius)
        gw = gx2 - gx1
        gh = gy2 - gy1

        # Distance map từ fire bbox
        cx = (x + x2) // 2 - gx1
        cy = (y + y2) // 2 - gy1
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = max(np.hypot(cx, cy), np.hypot(gw - cx, gh - cy), 1.0)
        dist_norm = np.clip(dist / max_dist, 0, 1)
        glow_strength = (1 - dist_norm) ** 2 * 0.25  # soft falloff

        glow_patch = out[gy1:gy2, gx1:gx2].astype(np.float32) / 255.0
        # Warm cast: tăng R, giảm B
        glow_patch[..., 2] = np.clip(glow_patch[..., 2] + glow_strength * 0.4, 0, 1)
        glow_patch[..., 1] = np.clip(glow_patch[..., 1] + glow_strength * 0.2, 0, 1)
        glow_patch[..., 0] = np.clip(glow_patch[..., 0] - glow_strength * 0.1, 0, 1)
        out[gy1:gy2, gx1:gx2] = (glow_patch * 255).astype(np.uint8)

    return out


def fire_in_fireplace(
    img: np.ndarray,
    *,
    max_fires: int = 1,
) -> tuple[np.ndarray, FireReport]:
    """Auto detect fireplace opening + add fire."""
    out = img.copy()
    report = FireReport()
    candidates = _detect_fireplace_openings(out)
    report.detected = len(candidates)
    for det in candidates[:max_fires]:
        out = add_fire_to_bbox(out, det.bbox)
        report.fires_added += 1
        report.detections.append(det)
    logger.info("Fire in Fireplace: detected=%d added=%d", report.detected, report.fires_added)
    return out, report
