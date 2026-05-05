"""Photographer Removal — xoá người chụp trong gương phản chiếu.

Nhiếp ảnh BĐS thường lộ chính mình + tripod + camera trong gương phòng tắm,
gương trang trí. Tool này:

1. Detect mirror surface (rectangular smooth-glass region)
2. Detect photographer silhouette + tripod trong mirror
3. LaMa inpaint vùng đó với context từ surround mirror reflection

Backends:
- detect_mirror: heuristic rectangle + reflective texture detection
- detect_person: OpenCV HOG + people detector trên mirror crop
- inpaint: LaMa AI (cho chất lượng pro), Telea fallback

Manual mode: user click bbox → inpaint trực tiếp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhotogRemovalReport:
    mirrors_detected: int = 0
    photographers_detected: int = 0
    regions_inpainted: int = 0
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)


def _detect_mirrors(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Heuristic: tìm large rectangular regions có texture giống reflection."""
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection cho mirror frame
    edges = cv2.Canny(gray, 50, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mirrors: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 100 or h < 100:
            continue
        rel = (w * h) / (w_img * h_img)
        if not (0.02 < rel < 0.5):
            continue
        # Aspect ratio reasonable cho mirror (tránh portrait stuff)
        ar = w / h
        if ar > 4 or ar < 0.25:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 4:
            continue
        contour_area = cv2.contourArea(c)
        if contour_area / (w * h) < 0.6:
            continue
        mirrors.append((int(x), int(y), int(w), int(h)))
    return mirrors


def _detect_people_hog(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """OpenCV HOG people detector — detect person silhouettes."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, _ = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]


def remove_photographer(
    img: np.ndarray,
    *,
    mirror_bboxes: list[tuple[int, int, int, int]] | None = None,
    use_ai_inpaint: bool = True,
) -> tuple[np.ndarray, PhotogRemovalReport]:
    """Auto detect mirror + photographer + inpaint.

    Args:
        img: BGR uint8.
        mirror_bboxes: nếu truyền vào, dùng làm hint thay vì auto detect.
        use_ai_inpaint: True = LaMa, False = Telea classical.

    Returns:
        ảnh đã xoá photographer + report.
    """
    out = img.copy()
    H, W = out.shape[:2]
    report = PhotogRemovalReport()

    if mirror_bboxes is None:
        mirror_bboxes = _detect_mirrors(out)
    report.mirrors_detected = len(mirror_bboxes)

    if not mirror_bboxes:
        return out, report

    # Build mask combining all photographer detections inside mirrors
    mask = np.zeros((H, W), dtype=np.uint8)

    for mx, my, mw, mh in mirror_bboxes:
        mirror_crop = out[my : my + mh, mx : mx + mw]
        if mirror_crop.size == 0:
            continue
        people = _detect_people_hog(mirror_crop)
        for px, py, pw, ph in people:
            # Convert tới global coord
            gx = mx + px
            gy = my + py
            gw, gh = pw, ph
            # Mở rộng bbox 10% cover camera/tripod
            ex = int(gw * 0.1)
            ey = int(gh * 0.1)
            x0 = max(0, gx - ex)
            y0 = max(0, gy - ey)
            x1 = min(W, gx + gw + ex)
            y1 = min(H, gy + gh + ey)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            report.regions.append((x0, y0, x1 - x0, y1 - y0))
            report.photographers_detected += 1

    if not np.any(mask):
        return out, report

    # Inpaint vùng đã mark
    if use_ai_inpaint:
        try:
            from .inpaint_ai import inpaint_ai

            out = inpaint_ai(out, mask, dilate=5)
        except Exception as exc:
            logger.warning("LaMa fail (%s) — fallback Telea", exc)
            from .inpaint import inpaint_opencv

            out = inpaint_opencv(out, mask, method="telea", radius=5)
    else:
        from .inpaint import inpaint_opencv

        out = inpaint_opencv(out, mask, method="telea", radius=5)

    report.regions_inpainted = len(report.regions)
    logger.info(
        "Photographer Removal: mirrors=%d people=%d inpainted=%d",
        report.mirrors_detected,
        report.photographers_detected,
        report.regions_inpainted,
    )
    return out, report


def remove_object_manual(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    use_ai_inpaint: bool = True,
) -> np.ndarray:
    """User-defined object bbox → inpaint."""
    out = img.copy()
    H, W = out.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = bbox
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    if use_ai_inpaint:
        from .inpaint_ai import inpaint_ai

        return inpaint_ai(out, mask, dilate=5)
    from .inpaint import inpaint_opencv

    return inpaint_opencv(out, mask, method="telea", radius=5)
