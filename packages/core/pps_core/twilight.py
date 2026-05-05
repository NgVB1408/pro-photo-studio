"""Virtual Twilight — chuyển ảnh ngoại thất ban ngày sang giờ vàng/hoàng hôn.

Đây là feature **AutoEnhance.ai bán riêng** ($/ảnh) — tạo ảnh "twilight"
từ 1 ảnh chụp ban ngày: replace sky bằng sunset gradient + thêm warm glow
ở cửa sổ (đèn nội thất) + ấm hoá tone toàn ảnh.

Pipeline:
1. Detect sky mask (dùng `realestate.detect_sky_mask` đã pro-grade — touches
   top edge, GrabCut, alpha matting).
2. Generate sunset gradient (top deep purple → middle orange → horizon yellow)
   + Perlin-style noise cho cloud texture (seedable).
3. Composite sky vào vùng mask với feather edge.
4. Detect bright window-like regions trong vùng building (ngoài sky mask) →
   add warm yellow glow (mô phỏng đèn bật khi tối).
5. Apply warm tone shift toàn ảnh (R+ B-) qua LAB midtone — không đụng
   vùng sky đã composite.

Khác bản tham khảo (`imagen-ai/backend/services/virtual_twilight.py`):
- BGR uint8 thay vì CHW float (toolkit standard)
- Detect sky bằng `detect_sky_mask` thay brightness threshold (chính xác hơn
  nhiều — không nhầm tường trắng thành sky)
- Seedable noise cho determinism trong batch
- Strength control 0..1
- Glow chỉ apply cho cửa sổ bên trong building, không apply lên ground/sky
- Warm tone shift áp ở LAB midtone không đụng shadows/highlights đã clip

API:
    transform_to_twilight(img, *, strength=0.85, seed=None) -> (out, info)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwilightReport:
    applied: bool
    sky_mask_pct: float = 0.0  # % pixel coi là sky
    glow_windows_pct: float = 0.0  # % pixel có warm glow
    reason: str = ""


def _sunset_gradient(
    h: int,
    w: int,
    *,
    rng: np.random.Generator,
    cloud_amount: float = 0.18,
) -> np.ndarray:
    """Generate sunset sky BGR uint8.

    Vertical bands:
      0..25% (top zenith): deep purple/blue
      25..55% (upper sunset): magenta → orange
      55..85% (lower sunset): orange → gold
      85..100% (horizon): warm yellow with haze
    Plus noisy cloud texture (additive, low-frequency).
    """
    # Build gradient in BGR (since OpenCV uses BGR)
    sky = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        t = y / max(h - 1, 1)
        if t < 0.25:
            # Deep purple zenith → indigo
            tt = t / 0.25
            r = 0.20 + 0.22 * tt
            g = 0.12 + 0.10 * tt
            b = 0.42 + 0.10 * tt
        elif t < 0.55:
            # Magenta to orange
            tt = (t - 0.25) / 0.30
            r = 0.42 + 0.45 * tt
            g = 0.22 + 0.25 * tt
            b = 0.52 - 0.30 * tt
        elif t < 0.85:
            # Orange to gold
            tt = (t - 0.55) / 0.30
            r = 0.87 + 0.05 * tt
            g = 0.47 + 0.25 * tt
            b = 0.22 - 0.10 * tt
        else:
            # Horizon haze
            tt = (t - 0.85) / 0.15
            r = 0.92 - 0.05 * tt
            g = 0.72 + 0.05 * tt
            b = 0.12 + 0.18 * tt
        # OpenCV BGR ordering
        sky[y, :, 0] = b
        sky[y, :, 1] = g
        sky[y, :, 2] = r

    if cloud_amount > 0:
        # Low-frequency noise as cloud texture
        small_h, small_w = max(8, h // 32), max(12, w // 32)
        noise_small = rng.random((small_h, small_w), dtype=np.float32)
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
        # Push noise to clouds (above 0.6) only — keeps sky color clean
        noise = np.clip((noise - 0.55) * 2.0, 0, 1)
        # Multiply more cloud near upper band, less near horizon
        vertical_falloff = np.linspace(1.0, 0.3, h, dtype=np.float32)[:, None]
        cloud = noise * vertical_falloff * cloud_amount
        # Brighten under cloud (warm pinks reflect sun)
        sky[..., 1] += cloud * 0.25  # G slightly up
        sky[..., 2] += cloud * 0.30  # R slightly up

    return np.clip(sky * 255.0, 0, 255).astype(np.uint8)


def _detect_window_glow_mask(
    img: np.ndarray,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Tìm vùng cửa sổ bên trong building (bright + có cấu trúc khung).

    Khác `detect_blown_areas` ở chỗ chỉ care vùng có brightness vừa phải
    (160-220 V) + có structure (Sobel response high) — không phải toàn bộ
    bright pixel. Loại trừ vùng đã là sky (passed in `exclude`).

    Trả mask float32 [0,1] đã feather.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]

    # Bright midtone (windows often look like 170-220 in V — not blown)
    bright = ((v >= 165) & (v <= 235)).astype(np.uint8) * 255

    # Edge response (windows have frame structure)
    sobel_x = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(sobel_x, sobel_y)
    edge_norm = np.clip(edge / max(edge.max(), 1e-6), 0, 1)

    # Smooth edge map → high in regions with frame-like content
    structure = cv2.GaussianBlur(edge_norm, (0, 0), sigmaX=8.0)
    structure_mask = (structure > 0.04).astype(np.uint8) * 255

    # AND with bright mask
    raw = cv2.bitwise_and(bright, structure_mask)

    # Loại sky
    if exclude is not None:
        excl = (exclude > 64).astype(np.uint8) * 255
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(excl))

    # Cleanup small noise + close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    # Connected component filter — keep only mid-size blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    total = h * w
    for i in range(1, n):
        ratio = stats[i, cv2.CC_STAT_AREA] / total
        if 0.0005 <= ratio <= 0.05:
            keep[labels == i] = 255

    feathered = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), sigmaX=6.0)
    return np.clip(feathered, 0, 1)


def _apply_warm_tone(img: np.ndarray, strength: float) -> np.ndarray:
    """Add warm tone (R+, B-) chỉ ở midtone, qua LAB."""
    if strength <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    # Midtone mask in L (40-200 typical for ~RGB 50-220)
    mid = np.clip(1.0 - np.abs(l - 128.0) / 128.0, 0, 1)
    # b channel up = warm (yellow), a channel up = warm-red
    shift_b = 8.0 * strength
    shift_a = 3.5 * strength
    lab[..., 2] += mid * shift_b  # +b = warmer
    lab[..., 1] += mid * shift_a
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def transform_to_twilight(
    img: np.ndarray,
    *,
    strength: float = 0.85,
    seed: int | None = None,
    use_ai_sky: bool = True,
    glow_intensity: float = 0.30,
    warm_tone: bool = True,
) -> tuple[np.ndarray, TwilightReport]:
    """Convert daytime exterior shot → twilight/sunset version.

    Args:
        img: BGR uint8 (3-channel).
        strength: 0..1, mức blend overall (sky composite + tone shift).
        seed: nếu set, RNG cho cloud noise sẽ deterministic.
        use_ai_sky: dùng `detect_sky_mask_smart` (rembg + heuristic). False
                    = chỉ dùng heuristic.
        glow_intensity: 0..1, độ mạnh warm glow ở cửa sổ.
        warm_tone: True = áp warm shift toàn ảnh ngoài sky.

    Returns:
        (output BGR uint8, TwilightReport)
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh BGR/BGRA")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed if seed is not None else 0)

    # 1. Sky mask
    try:
        if use_ai_sky:
            from .sky_seg_ai import detect_sky_mask_smart

            sky_mask = detect_sky_mask_smart(img)
        else:
            from .realestate import detect_sky_mask

            sky_mask = detect_sky_mask(img, refine=True)
    except Exception as exc:
        logger.warning("Twilight: sky detect fail (%s) — fallback heuristic", exc)
        from .realestate import detect_sky_mask

        sky_mask = detect_sky_mask(img, refine=True)

    sky_pct = float((sky_mask > 64).mean()) * 100.0

    if sky_pct < 0.5:
        # Không phải ảnh ngoại thất có sky → skip composite, chỉ áp warm tone
        out = _apply_warm_tone(img, strength * 0.6) if warm_tone else img.copy()
        return out, TwilightReport(
            applied=False,
            sky_mask_pct=sky_pct,
            reason="Không tìm được sky đáng kể (ảnh interior?)",
        )

    # 2. Generate sunset gradient
    sunset = _sunset_gradient(h, w, rng=rng)

    # 3. Composite sky
    alpha = (sky_mask.astype(np.float32) / 255.0)[..., None]
    alpha = np.clip(alpha * strength, 0, 1)
    sky_composite = img.astype(np.float32) * (1 - alpha) + sunset.astype(np.float32) * alpha
    out = np.clip(sky_composite, 0, 255).astype(np.uint8)

    # 4. Window glow
    glow_mask = _detect_window_glow_mask(img, exclude=sky_mask)
    glow_pct = float((glow_mask > 0.05).mean()) * 100.0
    if glow_pct > 0.05 and glow_intensity > 0:
        # Warm yellow glow color in BGR (B≈100, G≈210, R≈255)
        glow_color = np.array([100, 210, 255], dtype=np.float32)
        glow_alpha = (glow_mask * glow_intensity * strength)[..., None]
        out_f = out.astype(np.float32)
        out_f = out_f + glow_alpha * (glow_color - out_f) * 0.55
        # Cũng nâng nhẹ luminance tại tâm glow
        bloom = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=18.0)[..., None]
        out_f = out_f + bloom * 18.0 * strength
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    # 5. Warm tone toàn ảnh (ngoài sky)
    if warm_tone:
        warmed = _apply_warm_tone(out, strength * 0.6)
        # Chỉ apply ngoài sky (giữ sky composite nguyên)
        non_sky = 1.0 - alpha
        blend = out.astype(np.float32) * alpha + warmed.astype(np.float32) * non_sky
        out = np.clip(blend, 0, 255).astype(np.uint8)

    return out, TwilightReport(
        applied=True,
        sky_mask_pct=round(sky_pct, 2),
        glow_windows_pct=round(glow_pct, 2),
    )
