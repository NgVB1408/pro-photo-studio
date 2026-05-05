"""Indoor real-estate pro grade enhancement chain.

Module này giải quyết 2 vấn đề pro retoucher BĐS hay sửa nhưng generic
enhance không phủ:

1. **Tungsten cast trên tường trắng** — đèn dây tóc/halogen/3000K LED phủ ám
   vàng lên tường+trần+rèm. Generic gray-world WB sẽ neutralize TOÀN BỘ ảnh
   → mất glow nghệ thuật quanh đèn (đèn ấm phải vàng, không thể trắng).

   `selective_wall_wb` giải quyết bằng cách:
   - Phát hiện vùng "tường/trần trắng" (V cao + S thấp + L* gần neutral)
   - Neutralize CHỈ vùng đó về truly neutral
   - Giữ vùng warm bright (glow đèn) intact
   - Smooth blend mask → không seam

2. **Bề mặt đá/marble/gỗ phẳng thiếu sparkle** — chuyên gia BĐS dùng dehaze +
   clarity local trên đá đắt tiền để bring out texture.

   `boost_surface_clarity` giải quyết bằng:
   - Detect bề mặt "flat luminance + low saturation" (stone/marble) hoặc
     "moderate hue uniform" (gỗ sàn)
   - Tăng micro-contrast local trên các vùng này (guided filter base/detail)
   - Avoid faces, fabric, plants
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_white_wall_mask(
    img: np.ndarray,
    *,
    v_min: int = 150,           # tường sáng
    s_max: int = 90,             # ít saturation (không phải fabric)
    s_max_strict: int = 60,      # vùng pure-white càng strict
    feather_sigma: float = 12.0,
) -> np.ndarray:
    """Mask 0..255 cho vùng tường/trần trắng (kể cả bị tungsten ám vàng nhẹ).

    Heuristic:
    - V ≥ v_min (sáng)
    - S ≤ s_max (không phải fabric/plant)
    - Loại pixel saturated cao (đèn vàng rực) bằng s_max_strict trên V cao
    - Smooth feather để blend tự nhiên
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Wall: bright + low sat
    wall = (V >= v_min) & (S <= s_max)

    # Loại đèn ấm rực (V rất cao + S còn cao = đèn vàng): wall pixel V>200 cần S≤60
    very_bright_warm = (V >= 200) & (S > s_max_strict) & ((H <= 30) | (H >= 160))
    wall = wall & (~very_bright_warm)

    raw = wall.astype(np.uint8) * 255

    # Morphological clean để bỏ lốm đốm
    h, w = img.shape[:2]
    k = max(3, min(h, w) // 250)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k * 2, k * 2), np.uint8))

    # Feather edge
    if feather_sigma > 0:
        raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=feather_sigma)

    return raw


def selective_wall_wb(
    img: np.ndarray,
    *,
    strength: float = 0.65,
    cast_threshold: float = 0.025,
    mask: np.ndarray | None = None,
    max_chroma_shift: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Neutralize cast CHỈ trên tường trắng — LAB-based để KHÔNG over-correct.

    Pipeline (v2 — LAB safe):
    1. Detect white wall mask
    2. Convert ảnh sang LAB. Đo deviation A/B mean trên wall pixels (A=green/red,
       B=blue/yellow axis). A=128 + B=128 = neutral.
    3. Tính shift A/B cần thiết để wall mean → neutral. **Cap shift ở
       max_chroma_shift** để tránh over-correct (visual cyan/magenta cast).
    4. Apply shift trên wall pixels VỚI alpha mask blend. L channel KHÔNG đổi
       → luminance/contrast bảo toàn nguyên.

    Tại sao LAB tốt hơn BGR:
    - BGR scale factors có thể tạo cross-channel artifact khi cast severe
    - LAB tách rời chroma (A,B) khỏi luminance (L) → correction surgical
    - Cap chroma shift = guaranteed visually subtle, không bị "AI looking"

    Args:
        strength: 0..1 — 0.65 default phù hợp BĐS (dial back từ 0.85 vì tránh
            over-correct trên cast severe).
        cast_threshold: chỉ correct nếu chroma deviation ≥ threshold.
        mask: optional precomputed wall mask 0..255.
        max_chroma_shift: max shift A hoặc B (LAB scale 0..255). 8.0 = subtle.

    Returns:
        (corrected BGR uint8, info dict).
    """
    info: dict = {}
    if strength <= 0:
        info["applied"] = False
        info["reason"] = "strength=0"
        return img.copy(), info

    if mask is None:
        mask = detect_white_wall_mask(img)

    if mask.sum() < (img.shape[0] * img.shape[1]) * 0.02:
        info["applied"] = False
        info["reason"] = "no_significant_wall"
        info["wall_ratio"] = float(mask.sum()) / (img.shape[0] * img.shape[1] * 255)
        return img.copy(), info

    # Đo cast trên wall pixels (mask >= 200 = chắc chắn wall)
    wall_solid = mask >= 200
    if wall_solid.sum() < 100:
        wall_solid = mask >= 128

    # Convert sang LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

    a_mean = float(A[wall_solid].mean())
    b_mean = float(B[wall_solid].mean())

    # Deviation từ neutral (128). Dùng tỷ lệ chuẩn hoá BGR-equivalent
    # cast magnitude = sqrt(dA^2 + dB^2) / 128 (rough analogy)
    da = a_mean - 128.0
    db = b_mean - 128.0
    cast = float(np.sqrt(da * da + db * db) / 128.0)
    info["cast_magnitude"] = cast
    info["wall_lab_AB"] = (a_mean, b_mean)

    if cast < cast_threshold:
        info["applied"] = False
        info["reason"] = f"cast_too_small({cast:.3f}<{cast_threshold})"
        return img.copy(), info

    # Shift để neutralize, với cap
    target_da = -da * strength
    target_db = -db * strength
    # Cap shift để tránh over-correct
    target_da = float(np.clip(target_da, -max_chroma_shift, max_chroma_shift))
    target_db = float(np.clip(target_db, -max_chroma_shift, max_chroma_shift))

    info["shift_AB"] = (target_da, target_db)

    # Apply shift trên wall pixels với alpha
    alpha = mask.astype(np.float32) / 255.0
    A_corrected = A + alpha * target_da
    B_corrected = B + alpha * target_db

    lab_out = np.stack([L, A_corrected, B_corrected], axis=-1)
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    info["applied"] = True
    return out, info


# ======================================================================
# Surface clarity boost (marble / stone / wood)
# ======================================================================

def detect_smooth_surface_mask(
    img: np.ndarray,
    *,
    s_max: int = 80,
    v_min: int = 70,
    v_max: int = 235,
    edge_density_max: float = 0.10,
) -> np.ndarray:
    """Mask 0..255 cho bề mặt phẳng tone trung tính (marble/stone/wood/cabinet).

    Đặc trưng:
    - V trong dải mid (70..235) — không phải highlight blow hay shadow đen
    - S thấp (đá/gỗ thường desaturated)
    - Edge density thấp (smooth surface, không phải pattern fabric/plant)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_ok = (V >= v_min) & (V <= v_max) & (S <= s_max)

    # Edge density (block-wise)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = cv2.boxFilter(edges, cv2.CV_32F, (51, 51)) / 255.0
    smooth = edge_density <= edge_density_max

    raw = (color_ok & smooth).astype(np.uint8) * 255

    # Morph clean
    h, w = img.shape[:2]
    k = max(3, min(h, w) // 200)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=8.0)
    return raw


def boost_surface_clarity(
    img: np.ndarray,
    *,
    strength: float = 0.30,
    radius: int = 12,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Boost local micro-contrast trên smooth surfaces — bring out texture đá/gỗ.

    Pipeline: guided filter base/detail decomposition → boost detail → recombine,
    chỉ trong vùng mask (smooth surface).

    Args:
        strength: 0..1 mức boost (0.3 = subtle pro touch, 0.6 = aggressive).
        radius: kernel radius cho guided filter (lớn = smooth base nhiều hơn).
        mask: optional precomputed surface mask.

    Returns:
        (output BGR uint8, info dict).
    """
    info: dict = {}
    if strength <= 0:
        info["applied"] = False
        return img.copy(), info

    if mask is None:
        mask = detect_smooth_surface_mask(img)

    if mask.sum() < (img.shape[0] * img.shape[1]) * 0.03:
        info["applied"] = False
        info["reason"] = "no_smooth_surface"
        return img.copy(), info

    from .enhance import guided_filter

    base = guided_filter(img, radius=radius, eps=1e-2)
    detail = img.astype(np.float32) - base.astype(np.float32)
    boosted = base.astype(np.float32) + detail * (1.0 + strength)
    boosted = np.clip(boosted, 0, 255)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + boosted * alpha
    info["applied"] = True
    info["surface_ratio"] = float(mask.sum()) / (img.shape[0] * img.shape[1] * 255)
    return np.clip(out, 0, 255).astype(np.uint8), info


# ======================================================================
# Composite indoor pipeline
# ======================================================================

def _measure_pristine(img: np.ndarray) -> dict:
    """Đo xem ảnh có đã pristine chưa: cast nhỏ + dynamic range OK + không cháy/bệt."""
    f = img.astype(np.float32)
    m = (img.max(axis=2) < 245) & (img.min(axis=2) > 10)
    if m.sum() < 100:
        return {"is_pristine": False, "reason": "too_few_valid_pixels"}
    b = float(f[..., 0][m].mean())
    g = float(f[..., 1][m].mean())
    r = float(f[..., 2][m].mean())
    avg = (b + g + r) / 3.0
    cast = max(abs(b - avg), abs(g - avg), abs(r - avg)) / max(avg, 1.0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    dyn_range = p99 - p1
    clip_high = float((gray >= 250).mean())
    clip_low = float((gray <= 5).mean())

    is_pristine = (
        cast < 0.030
        and 130 <= dyn_range <= 250
        and clip_high < 0.005
        and clip_low < 0.020
    )
    return {
        "is_pristine": is_pristine,
        "cast": cast,
        "dyn_range": dyn_range,
        "clip_high": clip_high,
        "clip_low": clip_low,
    }


def enhance_interior_pro(
    img: np.ndarray,
    *,
    wb_strength: float = 0.85,
    clarity_strength: float = 0.30,
    shadow_lift_amount: float = 0.30,
    vibrance_amount: float = 0.15,
    sharpen_amount: float = 0.30,
    skip_if_pristine: bool = True,
) -> tuple[np.ndarray, dict]:
    """Pro indoor enhancement chain:
    1. Selective wall WB — neutralize tungsten cast trên tường, giữ glow đèn
    2. Shadow lift — hiển hình vùng tối (sàn gỗ, ghế tối)
    3. Vibrance — boost màu mid-sat tự nhiên
    4. Surface clarity — bring out texture đá/gỗ
    5. Subtle unsharp — final crispness

    Default values are CONSERVATIVE — phù hợp ảnh BĐS đã chụp tốt.
    Để aggressive hơn, tăng strength/amount.

    Args:
        skip_if_pristine: True = nếu ảnh đã pristine (cast<3%, range OK, không
            cháy/bệt) thì skip toàn bộ chain để tránh "sửa thứ không cần sửa"
            → bảo vệ điểm 8 batch consistency cho ảnh đẹp sẵn.

    Returns:
        (output BGR uint8, debug info dict).
    """
    info: dict = {"steps": []}

    # Skip nếu ảnh đã pristine — pro retoucher không sửa thứ không hỏng
    if skip_if_pristine:
        prist = _measure_pristine(img)
        info["pristine_check"] = prist
        if prist["is_pristine"]:
            info["skipped"] = True
            info["reason"] = (
                f"pristine(cast={prist['cast']:.3f},range={prist['dyn_range']:.0f})"
            )
            return img.copy(), info

    # Step 1: Selective wall WB
    out, wb_info = selective_wall_wb(img, strength=wb_strength)
    info["steps"].append(("selective_wall_wb", wb_info))

    # Step 2: Shadow lift (toàn ảnh — wall WB đã handle warm cast separately)
    if shadow_lift_amount > 0:
        from .enhance import shadow_lift
        out = shadow_lift(out, amount=shadow_lift_amount)
        info["steps"].append(("shadow_lift", {"amount": shadow_lift_amount}))

    # Step 3: Vibrance
    if vibrance_amount > 0:
        from .enhance import vibrance
        out = vibrance(out, amount=vibrance_amount)
        info["steps"].append(("vibrance", {"amount": vibrance_amount}))

    # Step 4: Surface clarity boost
    out, clar_info = boost_surface_clarity(out, strength=clarity_strength)
    info["steps"].append(("surface_clarity", clar_info))

    # Step 5: Subtle final sharpen
    if sharpen_amount > 0:
        from .enhance import unsharp_mask
        out = unsharp_mask(out, sigma=1.2, amount=sharpen_amount)
        info["steps"].append(("unsharp", {"amount": sharpen_amount}))

    return out, info
