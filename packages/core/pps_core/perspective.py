"""4-point perspective correction (Adobe Upright equivalent).

Sửa **converging verticals** khi chụp building từ thấp lên (camera tilt up)
hoặc từ trên xuống (tilt down). Khác hoàn toàn với rotate 2D — đây là warp
homography thực sự để các đường thẳng dọc song song với cạnh ảnh.

Algorithm:
1. Canny + Probabilistic Hough → line segments
2. Filter vertical-ish lines (góc < 25° từ vertical, độ dài ≥ 1/5 chiều cao)
3. Tìm vanishing point dọc bằng least-squares qua tất cả vertical lines
4. Nếu VP cách đủ xa khỏi center → tính skew amount
5. Build homography 4-point → expand top hoặc bottom tương ứng
6. Crop về kích thước gốc ở giữa

Output: (warped, report) — report dict có applied/skew/vp_y/lines_used/direction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UprightReport:
    applied: bool
    reason: str = ""
    skew: float = 0.0          # tỉ lệ skew (0..max_skew)
    vp_x: float = 0.0
    vp_y: float = 0.0
    lines_used: int = 0
    direction: str = ""        # "up" (tilt up) | "down" (tilt down)
    angle_estimate_deg: float = 0.0


def _detect_vertical_lines(
    img: np.ndarray,
    *,
    angle_tolerance_deg: float = 25.0,
    min_length_frac: float = 0.18,
    max_lines: int = 200,
):
    """Trả về list (x_top, y_top, x_bot, y_bot, angle_from_vert_deg)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)

    min_len = int(h * min_length_frac)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=max(60, min_len // 2),
        minLineLength=min_len, maxLineGap=18,
    )
    if lines is None:
        return []

    out = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dy = y2 - y1
        dx = x2 - x1
        if abs(dy) < h * 0.12:  # quá ngắn theo trục dọc
            continue
        if abs(dx) > abs(dy) * 0.65:  # quá nghiêng (gần horizontal)
            continue
        # Góc từ vertical (0 = dọc tuyệt đối)
        angle = np.degrees(np.arctan2(dx, dy))
        if abs(angle) > angle_tolerance_deg:
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        out.append((float(x1), float(y1), float(x2), float(y2), float(angle)))
        if len(out) >= max_lines:
            break
    return out


def _vanishing_point(lines) -> tuple[float, float] | None:
    """Least-squares VP qua N≥2 đường. Trả None nếu không converge tốt."""
    if len(lines) < 4:
        return None
    A, b_vec = [], []
    for x1, y1, x2, y2, _ in lines:
        # Đường: a*x + b*y + c = 0, normalize a²+b² = 1
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        n = np.hypot(a, b)
        if n < 1e-6:
            continue
        A.append([a / n, b / n])
        b_vec.append(-c / n)
    if len(A) < 4:
        return None
    A = np.array(A, dtype=np.float64)
    b_vec = np.array(b_vec, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    return float(sol[0]), float(sol[1])


def correct_upright(
    img: np.ndarray,
    *,
    max_skew: float = 0.18,
    min_lines: int = 12,
    min_vp_distance: float = 0.70,   # |dy/h| tối thiểu
    safe_max_skew: float = 0.22,
) -> tuple[np.ndarray, UprightReport]:
    """Sửa perspective chính xác hơn rotate 2D — phù hợp cho ảnh BĐS chụp wide.

    Defaults TIGHTENED so as not to over-warp interior photos with ambiguous
    vertical lines (furniture/window frames). Ảnh BĐS thật rất hiếm khi chụp
    nghiêng > 5°. Aggressive warp gây hỏng cảnh thay vì sửa.

    Args:
        max_skew: 0..0.5, default 0.18 (≈10° tilt — đủ cho ảnh BĐS bình thường).
            Cũ là 0.32 (≈17.7°) → quá aggressive trên interior.
        min_lines: 12 lines (cũ 8) — yêu cầu evidence mạnh hơn.
        min_vp_distance: 0.70 (cũ 0.55) — chỉ apply khi VP rất xa center.
        safe_max_skew: 0.22 (cũ 0.40) — clamp tuyệt đối.

    Returns:
        (warped_image, UprightReport).
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh BGR/BGRA")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    lines = _detect_vertical_lines(img)
    if len(lines) < min_lines:
        return (
            img.copy(),
            UprightReport(applied=False,
                          reason=f"Chỉ có {len(lines)} vertical line (cần ≥{min_lines})"),
        )

    vp = _vanishing_point(lines)
    if vp is None:
        return (
            img.copy(),
            UprightReport(applied=False, reason="Không tính được vanishing point"),
        )
    vp_x, vp_y = vp
    cy = h / 2.0

    dy_norm = abs(vp_y - cy) / h
    if dy_norm < min_vp_distance:
        return (
            img.copy(),
            UprightReport(
                applied=False,
                reason=f"VP quá gần center ({dy_norm:.2f}h < {min_vp_distance}h) — "
                       f"có thể warp gây méo lớn",
                vp_x=vp_x, vp_y=vp_y, lines_used=len(lines),
            ),
        )

    # Skew amount để gần như reverse hoàn toàn warp gốc.
    # Calibration: tilt_factor 0.18 → VP cách center ≈ 4.6h → skew cần ≈ 0.18
    # → coefficient 0.85 (đo thực nghiệm).
    # Clamp về [0, max_skew] để tránh over-correct ở case VP rất gần.
    skew = float(np.clip(0.85 / dy_norm, 0.0, min(max_skew, safe_max_skew)))

    # Ước lượng góc tilt camera (sin của góc tilt ≈ skew một cách thô)
    angle_estimate = float(np.degrees(np.arctan(skew)))

    if vp_y < cy:
        # VP ở phía trên → camera tilt up → top of image bị "nén" → mở rộng top
        direction = "up"
        src = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        dst = np.array([
            [-w * skew * 0.5, 0],
            [w * (1 + skew * 0.5), 0],
            [w, h],
            [0, h],
        ], dtype=np.float32)
    else:
        # VP phía dưới → camera tilt down → mở rộng bottom
        direction = "down"
        src = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        dst = np.array([
            [0, 0], [w, 0],
            [w * (1 + skew * 0.5), h],
            [-w * skew * 0.5, h],
        ], dtype=np.float32)

    # Offset dst để toàn bộ trong canvas mới (rộng hơn)
    new_w = int(w * (1 + skew))
    offset_x = w * skew * 0.5
    dst_shifted = dst + np.array([offset_x, 0], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst_shifted)
    warped = cv2.warpPerspective(
        img, H, (new_w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop về size gốc (centered) — phần warped ngoài rìa thường có distortion mạnh
    crop_x0 = int(round(offset_x))
    crop_x1 = crop_x0 + w
    if crop_x1 > new_w:
        crop_x1 = new_w
        crop_x0 = max(0, crop_x1 - w)
    out = warped[:, crop_x0:crop_x1]

    # Đảm bảo output có cùng size hoặc ít nhất tỉ lệ tương tự
    if out.shape[1] != w or out.shape[0] != h:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return out, UprightReport(
        applied=True,
        skew=skew,
        vp_x=vp_x,
        vp_y=vp_y,
        lines_used=len(lines),
        direction=direction,
        angle_estimate_deg=angle_estimate,
    )
