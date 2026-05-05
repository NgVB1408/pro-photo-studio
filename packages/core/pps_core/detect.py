"""Auto-detect watermark — không cần user nhập box.

Ba chiến lược, mỗi cái mạnh ở 1 loại watermark:

1. detect_text_mser   — MSER tìm các blob giống chữ (logo có chữ, ©, brand name).
2. detect_bright_logo — tìm vùng pixel sáng/đặc (logo trắng / overlay sáng).
3. detect_edge_anomaly — tìm vùng có edge-density cao hơn nền xung quanh (logo
                         đồ hoạ trên ảnh tự nhiên).

`auto_mask()` kết hợp cả 3 + filter geometry để loại noise.

LƯU Ý: auto-detect KHÔNG bao giờ hoàn hảo cho mọi ảnh. Luôn `--save-mask`
trước khi inpaint thật. Nếu auto thiếu/sai, fallback sang painter thủ công
hoặc box.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from .mask import combine_masks, dilate_mask

logger = logging.getLogger(__name__)

DetectStrategy = Literal["text", "bright", "edge", "logo", "auto"]
_ALLOWED_STRATEGIES = {"text", "bright", "edge", "logo", "auto"}


def detect_text_mser(
    image: np.ndarray,
    *,
    min_area: int = 60,
    max_area_ratio: float = 0.05,
    delta: int = 5,
) -> np.ndarray:
    """MSER-based text/logo detector. Trả mask uint8."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    max_area = int(h * w * max_area_ratio)

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    regions, _ = mser.detectRegions(gray)
    mask = np.zeros((h, w), dtype=np.uint8)
    for points in regions:
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def detect_bright_logo(
    image: np.ndarray,
    *,
    brightness_threshold: int = 230,
    saturation_threshold: int = 60,
    min_blob_area: int = 50,
) -> np.ndarray:
    """Logo trắng/sáng-pastel: pixel V cao + S thấp (ít màu)."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] >= brightness_threshold
    desat = hsv[..., 1] <= saturation_threshold
    raw = (bright & desat).astype(np.uint8) * 255

    # Loại bỏ blob quá nhỏ (noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            keep[labels == i] = 255
    return keep


def detect_corner_logo(
    image: np.ndarray,
    *,
    corner: str = "bottom-right",
    roi_w_ratio: float = 0.40,
    roi_h_ratio: float = 0.20,
    saturation_min: int = 120,
    pad_px: int = 20,
) -> np.ndarray:
    """Logo commercial (xanh, đỏ, đen có chữ) đặt cố định ở góc.

    Strategy: tìm cluster pixel có saturation cao (logo có màu nền) trong ROI
    góc, lấy bounding box + padding nhỏ. KHÔNG bắt pixel trắng (tường, thảm)
    để mask không phình ra ngoài logo.

    Nếu góc đó không có cluster saturated nào (logo trắng đơn thuần), trả mask
    rỗng — để các strategy khác (`bright`) lo phần đó.
    """
    h, w = image.shape[:2]
    rw = int(w * roi_w_ratio)
    rh = int(h * roi_h_ratio)

    if corner == "bottom-right":
        y0, y1, x0, x1 = h - rh, h, w - rw, w
    elif corner == "bottom-left":
        y0, y1, x0, x1 = h - rh, h, 0, rw
    elif corner == "top-right":
        y0, y1, x0, x1 = 0, rh, w - rw, w
    elif corner == "top-left":
        y0, y1, x0, x1 = 0, rh, 0, rw
    else:
        raise ValueError(
            f"corner phải là bottom-right/bottom-left/top-right/top-left, nhận {corner}"
        )

    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # CHỈ saturated pixels — pixel trắng/xám (tường/thảm) bị loại
    saturated = (hsv[..., 1] >= saturation_min).astype(np.uint8) * 255

    # Cleanup noise nhỏ
    kernel = np.ones((3, 3), dtype=np.uint8)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, kernel, iterations=1)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, connectivity=8)
    full = np.zeros((h, w), dtype=np.uint8)

    if n <= 1:
        return full

    img_area = h * w
    edge_tolerance = 8  # cluster bbox phải chạm edge ảnh trong khoảng này
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < img_area * 0.0005 or area > img_area * 0.04:
            continue
        bx_local = stats[i, cv2.CC_STAT_LEFT]
        by_local = stats[i, cv2.CC_STAT_TOP]
        bw_local = stats[i, cv2.CC_STAT_WIDTH]
        bh_local = stats[i, cv2.CC_STAT_HEIGHT]

        # Toạ độ tuyệt đối trong ảnh
        ax0 = x0 + bx_local
        ay0 = y0 + by_local
        ax1 = ax0 + bw_local
        ay1 = ay0 + bh_local

        # Phải chạm edge tương ứng với corner
        touches_edge = False
        if corner == "bottom-right":
            touches_edge = (w - ax1) <= edge_tolerance and (h - ay1) <= edge_tolerance
        elif corner == "bottom-left":
            touches_edge = ax0 <= edge_tolerance and (h - ay1) <= edge_tolerance
        elif corner == "top-right":
            touches_edge = (w - ax1) <= edge_tolerance and ay0 <= edge_tolerance
        elif corner == "top-left":
            touches_edge = ax0 <= edge_tolerance and ay0 <= edge_tolerance

        if not touches_edge:
            continue

        # Aspect ratio hợp lý (logo thường 1:1 → 5:1)
        aspect = bw_local / max(bh_local, 1)
        if aspect < 0.2 or aspect > 8:
            continue

        candidates.append((area, i, bx_local, by_local, bw_local, bh_local))

    if not candidates:
        return full

    # Lấy candidate lớn nhất (logo thường là cluster saturated lớn nhất chạm edge)
    candidates.sort(key=lambda t: -t[0])
    _, idx, bx_local, by_local, bw_local, bh_local = candidates[0]

    # Bbox tuyệt đối của cluster + padding rộng để cover rounded corner + viền soft
    abs_x0 = max(0, x0 + bx_local - pad_px)
    abs_y0 = max(0, y0 + by_local - pad_px)
    abs_x1 = min(w, x0 + bx_local + bw_local + pad_px)
    abs_y1 = min(h, y0 + by_local + bh_local + pad_px)

    # Fill toàn bbox solid — logo có rounded corners, antialiased text,
    # gradient → bất kỳ approach pixel-level nào cũng bỏ sót pixel rìa.
    # Solid bbox đảm bảo cover 100%, inpaint sẽ phục dựng từ context xung quanh.
    full[abs_y0:abs_y1, abs_x0:abs_x1] = 255
    return full


def detect_edge_anomaly(
    image: np.ndarray,
    *,
    block: int = 64,
    z_threshold: float = 2.0,
) -> np.ndarray:
    """Tìm các block có edge-density cao bất thường so với phần còn lại của ảnh.

    Phù hợp logo đồ hoạ (đen/màu) trên ảnh nhiếp ảnh tự nhiên.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 80, 200)

    # Edge density per block
    bh = max(1, h // block)
    bw = max(1, w // block)
    density = cv2.resize(edges.astype(np.float32), (bw, bh), interpolation=cv2.INTER_AREA)

    mean = float(density.mean())
    std = float(density.std()) or 1.0
    z = (density - mean) / std
    hot = (z >= z_threshold).astype(np.uint8) * 255

    # Up-sample về kích thước ảnh
    mask = cv2.resize(hot, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _filter_by_geometry(
    mask: np.ndarray,
    *,
    min_area: int = 100,
    min_aspect: float = 0.05,
    max_aspect: float = 20.0,
    border_only: bool = False,
    border_ratio: float = 0.25,
) -> np.ndarray:
    """Loại bỏ blob không-logo: quá nhỏ, aspect kỳ cục, hoặc nằm sâu trong ảnh.

    border_only=True: chỉ giữ blob có ít nhất 1 phần nằm trong vùng rìa
    (border_ratio * min(h, w))."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    border = int(min(h, w) * border_ratio)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bh == 0 or bw == 0:
            continue
        aspect = bw / bh
        if aspect < min_aspect or aspect > max_aspect:
            continue

        if border_only:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            x1 = x0 + bw
            y1 = y0 + bh
            in_border = (
                x0 < border or y0 < border
                or x1 > w - border or y1 > h - border
            )
            if not in_border:
                continue
        out[labels == i] = 255
    return out


def auto_mask(
    image: np.ndarray,
    *,
    strategy: DetectStrategy = "auto",
    border_only: bool = True,
    dilate_iters: int = 3,
) -> np.ndarray:
    """Auto-build mask. `strategy="auto"` chạy cả 4 strategy và OR kết quả."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh BGR/BGRA")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if strategy not in _ALLOWED_STRATEGIES:
        raise ValueError(
            f"strategy phải thuộc {sorted(_ALLOWED_STRATEGIES)}, nhận {strategy!r}"
        )

    masks = []
    if strategy in ("text", "auto"):
        m = detect_text_mser(image)
        m = _filter_by_geometry(
            m, min_area=80, min_aspect=0.1, max_aspect=15,
            border_only=border_only,
        )
        masks.append(m)
        logger.debug("text-MSER blobs: %d", int((m > 0).sum()))
    if strategy in ("bright", "auto"):
        m = detect_bright_logo(image)
        m = _filter_by_geometry(
            m, min_area=100, border_only=border_only,
        )
        masks.append(m)
        logger.debug("bright-logo pixels: %d", int((m > 0).sum()))
    if strategy in ("logo", "auto"):
        # Quét cả 4 góc cho logo commercial — strategy "logo" KHÔNG dùng bright
        # để tránh phủ nhầm tường/thảm trắng.
        for corner in ("bottom-right", "bottom-left", "top-right", "top-left"):
            m = detect_corner_logo(image, corner=corner)
            masks.append(m)
            logger.debug("corner-logo[%s] pixels: %d", corner, int((m > 0).sum()))
    if strategy in ("edge", "auto"):
        m = detect_edge_anomaly(image)
        m = _filter_by_geometry(
            m, min_area=200, border_only=border_only,
        )
        masks.append(m)
        logger.debug("edge-anomaly pixels: %d", int((m > 0).sum()))

    if not masks:
        # Mặc dù đã validate strategy ở trên, giữ guard này để defensive.
        raise ValueError(f"Không strategy nào được áp dụng cho strategy={strategy!r}")

    combined = combine_masks(masks)
    if dilate_iters > 0:
        combined = dilate_mask(combined, iterations=dilate_iters)

    coverage = float((combined > 0).sum()) / combined.size
    logger.info("Auto-mask coverage: %.2f%% diện tích ảnh", coverage * 100)
    if coverage > 0.4:
        logger.warning(
            "Auto-mask phủ %.0f%% ảnh — có thể quá rộng. Cân nhắc strategy "
            "cụ thể hoặc tăng border_only.",
            coverage * 100,
        )
    return combined
