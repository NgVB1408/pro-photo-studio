"""TV Blackout — detect TV screens trong nội thất + thay nền đen / asset.

Lý do: TV ON khi chụp BĐS sẽ hiển thị logo brand (Samsung/LG), reflection, glare;
chèn TV "off" cho ảnh sạch và pro.

Heuristic detect (không cần ML):
1. Tìm rectangle/quadrilateral có aspect ratio ~ 16:9 hoặc 4:3
2. Trong vùng ảnh có brightness contrast cao (TV thường gần wall color trở đi)
3. Edge sharp (TV bezel rõ)
4. Diện tích 0.5%-15% ảnh

Sau detect → fill black + thêm reflection nhẹ (gradient) cho realistic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TVDetection:
    polygon: np.ndarray  # 4 points (x, y)
    area: int
    aspect_ratio: float
    confidence: float


@dataclass
class TVBlackoutReport:
    detected: int = 0
    blacked_out: int = 0
    detections: list[TVDetection] = field(default_factory=list)


def _detect_quadrilaterals(
    img: np.ndarray, *, min_aspect: float = 1.55, max_aspect: float = 2.10,
    min_area_rel: float = 0.020, max_area_rel: float = 0.25,
) -> list[TVDetection]:
    """Tighter TV-screen detector — avoid false-positive on framed pictures.

    Updates v2:
    - min_aspect 1.2→1.55, max_aspect 2.5→2.1 (TV gần 16:9 = 1.78, picture frame 4:3=1.33)
    - min_area_rel 0.5%→2% (small framed picture < 2% reject)
    - Interior content check: TV either DARK uniform (off) hoặc HIGH brightness (on
      with display). Framed picture có moderate variance + multi-color photo.
    """
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bil = cv2.bilateralFilter(gray, 9, 50, 50)
    edges = cv2.Canny(bil, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[TVDetection] = []
    img_area = h_img * w_img
    for c in contours:
        area = cv2.contourArea(c)
        rel = area / img_area
        if not (min_area_rel < rel < max_area_rel):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        poly = approx.reshape(4, 2)
        x, y, w, h = cv2.boundingRect(poly)
        if h <= 0:
            continue
        ar = w / h
        if not (min_aspect < ar < max_aspect):
            continue

        # Interior content check — phải là "screen-like"
        x0 = max(0, x + int(w * 0.1))
        y0 = max(0, y + int(h * 0.1))
        x1 = min(w_img, x + int(w * 0.9))
        y1 = min(h_img, y + int(h * 0.9))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = img[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]
        # Uniformity: TV off = std rất nhỏ; TV on = std vừa nhưng mean cao
        roi_std = float(roi_gray.std())
        roi_mean = float(roi_gray.mean())
        # Color variance — tranh có nhiều màu khác nhau
        bgr_std = float(roi.reshape(-1, 3).std(axis=0).mean())

        is_screen = False
        # Path 1: TV off (uniform dark)
        if roi_mean < 60 and roi_std < 25:
            is_screen = True
        # Path 2: TV on (high brightness, single-tone dominant)
        elif roi_mean > 100 and bgr_std < 35:
            is_screen = True

        if not is_screen:
            continue

        rect_area = w * h
        rectangularity = area / rect_area
        ar_score = 1 - abs(ar - 1.78) / 1.78
        conf = float(rectangularity * 0.5 + ar_score * 0.5)
        if conf < 0.65:  # tighter conf threshold
            continue
        candidates.append(TVDetection(
            polygon=poly.astype(np.int32),
            area=int(area),
            aspect_ratio=float(ar),
            confidence=conf,
        ))
    candidates.sort(key=lambda d: -d.confidence)
    return candidates


def _draw_tv_off(
    img: np.ndarray, polygon: np.ndarray, *, with_reflection: bool = True,
) -> None:
    """Vẽ TV off (đen + reflection nhẹ) in-place."""
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    # Inset 3% để giữ bezel
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    inset_poly = polygon.copy().astype(np.float32)
    for i in range(4):
        dx = inset_poly[i, 0] - cx
        dy = inset_poly[i, 1] - cy
        inset_poly[i, 0] = cx + dx * 0.97
        inset_poly[i, 1] = cy + dy * 0.97
    inset_poly = inset_poly.astype(np.int32)

    # Fill screen với màu đen sâu
    cv2.fillConvexPoly(img, inset_poly, (10, 10, 10))

    # Reflection nhẹ — gradient diagonal
    if with_reflection:
        x, y, w, h = cv2.boundingRect(inset_poly)
        if w > 20 and h > 20:
            grad = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(h):
                for_factor = (1.0 - i / h) * 0.15  # mạnh nhất ở top
                grad[i] = for_factor
            # Apply only where TV mask
            x2, y2 = x + w, y + h
            tv_mask = np.zeros((h, w), dtype=np.uint8)
            local_poly = inset_poly - np.array([x, y])
            cv2.fillConvexPoly(tv_mask, local_poly, 255)
            tv_mask_3 = (tv_mask > 0).astype(np.float32)[..., None]
            patch = img[y:y2, x:x2].astype(np.float32) / 255.0
            # Add reflection theo mask
            patch += grad * tv_mask_3 * 30 / 255.0
            img[y:y2, x:x2] = np.clip(patch * 255, 0, 255).astype(np.uint8)


def tv_blackout(
    img: np.ndarray, *,
    max_screens: int = 3,
    confidence_threshold: float = 0.55,
) -> tuple[np.ndarray, TVBlackoutReport]:
    """Detect + blackout TV screens trong ảnh nội thất.

    Args:
        img: BGR uint8.
        max_screens: tối đa bao nhiêu screen blackout (sort theo confidence).
        confidence_threshold: skip detection có conf thấp hơn.

    Returns:
        ảnh đã xử lý + report.
    """
    out = img.copy()
    report = TVBlackoutReport()

    candidates = _detect_quadrilaterals(out)
    selected = [c for c in candidates if c.confidence >= confidence_threshold][:max_screens]
    report.detected = len(candidates)

    for det in selected:
        _draw_tv_off(out, det.polygon, with_reflection=True)
        report.blacked_out += 1
        report.detections.append(det)

    logger.info("TV Blackout: detected=%d blacked=%d", report.detected, report.blacked_out)
    return out, report


def tv_blackout_manual(
    img: np.ndarray, polygon: np.ndarray,
) -> np.ndarray:
    """User-defined TV polygon — 4 điểm."""
    out = img.copy()
    if polygon.shape != (4, 2):
        raise ValueError(f"polygon phải 4×2, nhận {polygon.shape}")
    _draw_tv_off(out, polygon.astype(np.int32), with_reflection=True)
    return out
