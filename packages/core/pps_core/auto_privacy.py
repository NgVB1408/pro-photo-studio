"""Auto Privacy — blur khuôn mặt + biển số xe trong ảnh BĐS.

Yêu cầu cho ngành BĐS:
- Hàng xóm/người đi đường lọt vào ảnh exterior → blur mặt
- Xe đậu trước nhà → blur biển số xe
- Người chụp/phản chiếu trong gương → flag để dùng module Photographer Removal

Backend:
- Face: OpenCV Haar cascade (nhẹ, không deps) hoặc DNN SSD/YuNet (chính xác hơn)
- License plate: YOLO/contour detection + OCR confidence

Output: ảnh đã blur + report số region đã blur.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyReport:
    faces_blurred: int = 0
    plates_blurred: int = 0
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x,y,w,h)


def _gaussian_blur_region(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    kernel_scale: float = 0.4,
) -> None:
    """Blur in-place 1 region. Kernel ~ tỉ lệ với region size."""
    if w <= 0 or h <= 0:
        return
    H, W = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    k = max(11, int(min(roi.shape[:2]) * kernel_scale) | 1)  # odd
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    # Pixelate effect (mosaic) cho blur mạnh hơn
    pix = max(8, k // 4)
    small = cv2.resize(
        blurred,
        (max(1, roi.shape[1] // pix), max(1, roi.shape[0] // pix)),
        interpolation=cv2.INTER_LINEAR,
    )
    pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    img[y:y2, x:x2] = pixelated


def _has_skin_tone(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    min_ratio: float = 0.35,
    center_min_ratio: float = 0.55,
) -> bool:
    """Check if face bbox có ratio đủ skin-color pixels.

    Tighter v2 — yêu cầu skin trong CENTER 50% của bbox, không phải any pixel.
    Pillow patchwork có thể có vài patch màu da random, nhưng KHÔNG concentrate
    ở center vùng face.

    Skin-tone YCrCb space (Cr 135-180, Cb 85-140) — cover Á/Âu/Phi.
    """
    H, W = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return False
    roi = img[y:y2, x:x2]
    if roi.size < 400:
        return False
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    skin_mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 140)
    ratio = float(skin_mask.sum()) / skin_mask.size
    if ratio < min_ratio:
        return False

    # Check skin trong center 50% (face center = mũi/má — phải có skin tập trung)
    h_roi, w_roi = skin_mask.shape
    cy0 = h_roi // 4
    cy1 = h_roi - h_roi // 4
    cx0 = w_roi // 4
    cx1 = w_roi - w_roi // 4
    center = skin_mask[cy0:cy1, cx0:cx1]
    if center.size < 100:
        return False
    center_ratio = float(center.sum()) / center.size
    return center_ratio >= center_min_ratio


_YUNET_DETECTOR: cv2.FaceDetectorYN | None = None
_YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _get_yunet_model_path() -> Path | None:
    """Return path to yunet.onnx, download if missing. Return None if download fail."""
    target = Path.home() / ".pps_core" / "models" / "yunet.onnx"
    if target.is_file() and target.stat().st_size > 100_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        req = urllib.request.Request(
            _YUNET_MODEL_URL, headers={"User-Agent": "watermark-toolkit/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        target.write_bytes(data)
        logger.info("Downloaded YuNet model %.0f KB", len(data) / 1024)
        return target
    except Exception as exc:
        logger.warning("YuNet model download fail: %s — fallback Haar", exc)
        return None


def _yunet_detector(img_w: int, img_h: int) -> cv2.FaceDetectorYN | None:
    """Lazy-load YuNet detector. Return None nếu không có model → caller fallback Haar."""
    global _YUNET_DETECTOR
    model_path = _get_yunet_model_path()
    if model_path is None:
        return None
    if _YUNET_DETECTOR is None:
        _YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (img_w, img_h),
            score_threshold=0.92,  # very strict — BĐS thường không có people
            nms_threshold=0.3,
            top_k=20,
        )
    else:
        _YUNET_DETECTOR.setInputSize((img_w, img_h))
    return _YUNET_DETECTOR


def detect_faces_yunet(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """YuNet DNN face detector — chính xác hơn Haar nhiều, không nhầm wall-light/pillow."""
    h, w = img.shape[:2]
    detector = _yunet_detector(w, h)
    if detector is None:
        return []  # caller fallback Haar
    _, faces = detector.detect(img)
    if faces is None:
        return []
    boxes = []
    for f in faces:
        x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        if fw < 10 or fh < 10:
            continue
        boxes.append((x, y, fw, fh))
    return boxes


def _haar_face_cascade() -> cv2.CascadeClassifier:
    """Lazy-load Haar face cascade từ OpenCV bundled."""
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Haar cascade missing: {path}")
    cls = cv2.CascadeClassifier(str(path))
    if cls.empty():
        raise RuntimeError("Haar cascade load fail")
    return cls


def _haar_profile_cascade() -> cv2.CascadeClassifier:
    """Cascade nhận mặt nghiêng (lookat-side)."""
    path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
    if not path.is_file():
        return None  # optional
    cls = cv2.CascadeClassifier(str(path))
    return cls if not cls.empty() else None


def detect_faces(img: np.ndarray, *, min_size: int = 30) -> list[tuple[int, int, int, int]]:
    """Detect faces — primary YuNet DNN, fallback Haar cascade nếu YuNet load fail.

    YuNet ~5MB ONNX, accuracy >95% trên WIDER FACE benchmark, không nhầm
    wall-sconce/pillow như Haar.
    """
    # Prefer YuNet — chính xác hơn Haar nhiều
    yunet_boxes = detect_faces_yunet(img)
    if yunet_boxes:
        # Belt-and-suspenders: YuNet detection vẫn check skin-tone center
        # để loại sót case wall sconce được ML detect nhầm
        verified = []
        for x, y, w, h in yunet_boxes:
            if _has_skin_tone(img, x, y, w, h):
                verified.append((x, y, w, h))
        return _merge_overlapping(verified)
    # Nếu YuNet không trả gì (model fail) → fallback Haar với strict gates

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    H, W = gray.shape[:2]
    short = min(H, W)
    # Auto-scale min_size theo image size (4% short edge — face đủ lớn)
    # Trong BĐS, face xa < 4% thường không quan trọng riêng tư + thường là pillow false-positive
    auto_min = max(min_size, int(short * 0.04))
    boxes: list[tuple[int, int, int, int]] = []

    frontal = _haar_face_cascade()
    found = frontal.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(auto_min, auto_min),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    for x, y, w, h in found:
        # Aspect filter: face square-ish (0.7 < ar < 1.4)
        ar = w / max(h, 1)
        if ar < 0.7 or ar > 1.4:
            continue
        # Skin-tone gate: face roi phải có ít nhất 15% skin-color pixels
        # Tránh false-positive trên pillow patterns / printed art / windows
        if not _has_skin_tone(img, x, y, w, h, min_ratio=0.15):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    profile = _haar_profile_cascade()
    if profile is not None:
        for flip in (False, True):
            g = cv2.flip(gray, 1) if flip else gray
            found = profile.detectMultiScale(
                g,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(auto_min, auto_min),
            )
            for x, y, w, h in found:
                ar = w / max(h, 1)
                if ar < 0.7 or ar > 1.4:
                    continue
                if flip:
                    x = W - x - w
                if not _has_skin_tone(img, x, y, w, h, min_ratio=0.15):
                    continue
                boxes.append((int(x), int(y), int(w), int(h)))

    return _merge_overlapping(boxes)


def _merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Non-max suppress overlapping boxes (IoU > 0.3)."""
    if not boxes:
        return []
    rects = np.array(boxes)
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = x1 + rects[:, 2]
    y2 = y1 + rects[:, 3]
    areas = rects[:, 2] * rects[:, 3]
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < 0.3]
    return [tuple(int(v) for v in rects[i]) for i in keep]


def detect_license_plates(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Heuristic license plate detector — không cần deep learning.

    Tìm rectangle có aspect ratio plate (3:1 → 5:1) ở vùng dưới ảnh,
    màu sáng (white plate VN/EU) hoặc vàng (US/specific countries),
    có text-like contrast.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilateral filter giữ edge
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edges = cv2.Canny(gray, 30, 200)
    # Morphology để connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plates: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh < 10 or ww < 30:
            continue
        ar = ww / hh
        # plate aspect ratio (chặt hơn để tránh window/door false positive)
        if not (2.5 < ar < 5.5):
            continue
        # plate size: 0.05% - 1.5% of frame (license plate thường nhỏ)
        area = ww * hh
        rel = area / (w * h)
        if not (0.0005 < rel < 0.015):
            continue
        # Plate ở phần dưới + trung tâm theo chiều ngang
        cy = y + hh / 2
        if cy < h * 0.5:
            continue
        # Verify có text-like contrast (variance cao trên mảnh sáng)
        roi = img[y : y + hh, x : x + ww]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if gray_roi.size < 100:
            continue
        # Plate VN/EU thường có background sáng (white/yellow) > mean 130
        if gray_roi.mean() < 100:
            continue
        # Variance cao = có text
        if gray_roi.std() < 30:
            continue
        plates.append((x, y, ww, hh))
    return plates


def auto_privacy(
    img: np.ndarray,
    *,
    blur_faces: bool = True,
    blur_plates: bool = True,
    min_face_size: int = 20,
) -> tuple[np.ndarray, PrivacyReport]:
    """Pipeline đầy đủ: detect + blur faces + plates."""
    out = img.copy()
    report = PrivacyReport()

    if blur_faces:
        faces = detect_faces(out, min_size=min_face_size)
        for x, y, w, h in faces:
            # Mở rộng region 20% để cover tóc/cổ
            exp = int(min(w, h) * 0.2)
            _gaussian_blur_region(out, x - exp, y - exp, w + 2 * exp, h + 2 * exp)
            report.regions.append((x, y, w, h))
        report.faces_blurred = len(faces)

    if blur_plates:
        plates = detect_license_plates(out)
        for x, y, w, h in plates:
            _gaussian_blur_region(out, x, y, w, h, kernel_scale=0.5)
            report.regions.append((x, y, w, h))
        report.plates_blurred = len(plates)

    logger.info("Auto Privacy: faces=%d plates=%d", report.faces_blurred, report.plates_blurred)
    return out, report
