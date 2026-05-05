"""HDR exposure fusion — recover detail trong vùng cháy/tối THẬT SỰ.

So với `window_pull` cũ chỉ dùng gamma compress (làm xám vùng cháy):
- HDR Mertens **blend nhiều exposure** → recover detail thật ngoài cửa sổ
- Pixel đã 255 trong 1 exposure thường có info trong exposure tối hơn → fusion
  trả lại detail (cây, building, trời)

Hai mode:
1. **Bracket fusion** — user upload 2-5 ảnh exposure khác nhau, dùng
   Mertens (Tom Mertens, 2007) — chất lượng cao nhất, đây là cách pro làm.
2. **Single image pseudo-HDR** — synthesize 3 exposure từ 1 ảnh bằng
   gamma + tone curve, fuse Mertens. Không thay được bracket thật nhưng
   tốt hơn nhiều so với gamma compress đơn thuần.

API chính:
- `fuse_brackets(images, weights=...)` — Mertens fusion từ N≥2 exposures
- `pseudo_hdr_single(img, ev_steps=(-1.5, 0, 1.0))` — synth bracket → fuse
- `recover_blown_windows(img, mask=None, mode='auto')` — wrapper detect
  cửa sổ cháy + áp pseudo-HDR + blend với original
"""
from __future__ import annotations

import logging
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Bracket fusion (REAL HDR — pro workflow)
# =====================================================================

def color_normalize_brackets(
    images: Sequence[np.ndarray],
    *,
    ref_index: int = 0,
) -> list[np.ndarray]:
    """LAB mean/std matching across brackets — giảm WB/temperature drift.

    Khi 3 brackets được chụp với cùng setting nhưng auto-WB của camera nudge
    màu hơi khác giữa frames, fuse Mertens sẽ ra ảnh với halo màu. Pre-normalize
    về LAB stats của reference frame (thường là 0 EV) trước khi fuse.

    Port từ imagen-ai/backend/services/color_normalize.py.

    Args:
        images: list ≥1 BGR uint8 cùng size.
        ref_index: index ảnh dùng làm reference (default 0 = ảnh đầu).

    Returns:
        list cùng độ dài, mỗi ảnh đã match LAB mean/std của reference.
    """
    if len(images) < 2:
        return list(images)
    ref_lab = cv2.cvtColor(images[ref_index], cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_mean = ref_lab.reshape(-1, 3).mean(axis=0)
    ref_std = ref_lab.reshape(-1, 3).std(axis=0) + 1e-6

    out: list[np.ndarray] = []
    for i, im in enumerate(images):
        if i == ref_index:
            out.append(im)
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        flat = lab.reshape(-1, 3)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-6
        adjusted = (lab - mean) * (ref_std / std) + ref_mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
        out.append(bgr)
    return out


def compute_deghost_mask(
    images: Sequence[np.ndarray],
    *,
    threshold: float = 4.0,
    feather_sigma: float = 2.0,
) -> np.ndarray:
    """Detect motion/ghost pixels across aligned exposures.

    Đối với HDR merge, pixel di chuyển (người/xe/lá cây gió) sẽ tạo ghost
    artifact khi blend các exposure. Mask này flag những pixel có deviation
    lớn so với median across frames → sau đó fuse_brackets() sẽ fallback
    về reference frame ở những pixel đó thay vì blend.

    Port từ imagen-ai/backend/services/deghost_mask.py — adapted cho BGR uint8.

    Args:
        images: list ≥2 BGR uint8 đã align cùng size.
        threshold: bội số của MAD coi là outlier (cao hơn = strict hơn).
        feather_sigma: Gaussian blur sigma sau threshold để smooth edge.

    Returns:
        float32 mask H×W [0,1] — 1 = ghost pixel, 0 = stable.
    """
    if len(images) < 2:
        h, w = images[0].shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    ys = []
    for im in images:
        if im.ndim == 3:
            y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            y = im
        ys.append(y.astype(np.float32))
    stack = np.stack(ys, axis=0)  # (T, H, W)

    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median[None, ...]), axis=0) + 1e-6
    deviation = np.max(np.abs(stack - median[None, ...]) / mad[None, ...], axis=0)

    mask = (deviation > threshold).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if feather_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma)
    return np.clip(mask, 0.0, 1.0)


def fuse_brackets(
    images: Sequence[np.ndarray],
    *,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposedness_weight: float = 1.0,
    deghost: bool = False,
    color_normalize: bool = False,
    reference_index: int = 0,
) -> np.ndarray:
    """Mertens exposure fusion (Tom Mertens et al. 2007).

    Args:
        images: list/tuple ≥2 BGR uint8 ảnh cùng size, khác exposure
                (vd: -2 EV, 0 EV, +2 EV).
        contrast_weight: ưu tiên pixel có local contrast cao (edges).
        saturation_weight: ưu tiên pixel có saturation cao (màu sống).
        exposedness_weight: ưu tiên pixel ở midtone (xa 0 và 255).
        deghost: True = detect motion pixels và fallback về reference frame
                 ở vùng đó để tránh ghost.
        color_normalize: True = LAB-match all brackets về reference trước
                         khi fuse → giảm WB drift halo.
        reference_index: ảnh nào làm reference cho deghost / color_normalize.
                         Thường là 0 EV (giữa stack).

    Returns:
        BGR uint8 ảnh fused — cùng size với input.
    """
    imgs = list(images)
    if len(imgs) < 2:
        raise ValueError("Cần ít nhất 2 ảnh exposure để fuse")
    h0, w0 = imgs[0].shape[:2]
    aligned = []
    for i, im in enumerate(imgs):
        if im.dtype != np.uint8:
            raise ValueError(f"Ảnh {i} phải uint8, nhận {im.dtype}")
        if im.shape[:2] != (h0, w0):
            logger.warning(
                "Ảnh %d size %s khác ảnh đầu %dx%d — resize",
                i, im.shape[:2], h0, w0,
            )
            im = cv2.resize(im, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        aligned.append(im)

    if color_normalize:
        aligned = color_normalize_brackets(aligned, ref_index=reference_index)

    ghost_mask = None
    if deghost:
        ghost_mask = compute_deghost_mask(aligned)

    merger = cv2.createMergeMertens(
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposedness_weight,
    )
    fused = merger.process(aligned)
    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    if ghost_mask is not None and ghost_mask.max() > 0.05:
        # Ở vùng ghost, replace với reference frame để tránh blend artifacts
        ref = aligned[max(0, min(reference_index, len(aligned) - 1))]
        alpha = ghost_mask[..., None]
        blended = (fused.astype(np.float32) * (1 - alpha) +
                   ref.astype(np.float32) * alpha)
        fused = np.clip(blended, 0, 255).astype(np.uint8)
    return fused


def align_brackets(
    images: Sequence[np.ndarray],
    *,
    method: Literal["mtb", "ecc"] = "mtb",
) -> list[np.ndarray]:
    """Align nhiều bracket trước khi fuse (cho ảnh chụp tay không tripod).

    method:
      - "mtb" (Median Threshold Bitmap, OpenCV `createAlignMTB`) — nhanh,
        robust với exposure khác nhau. Khuyến nghị.
      - "ecc" (Enhanced Correlation Coefficient) — chính xác hơn nhưng chậm.
    """
    imgs = list(images)
    if len(imgs) < 2:
        return imgs
    if method == "mtb":
        aligner = cv2.createAlignMTB()
        aligner.process(imgs, imgs)  # in-place align
        return imgs
    # ECC fallback
    ref = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY) if imgs[0].ndim == 3 else imgs[0]
    out = [imgs[0]]
    for im in imgs[1:]:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref.astype(np.float32) / 255.0,
                gray.astype(np.float32) / 255.0,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          200, 1e-5),
            )
            warped = cv2.warpAffine(
                im, warp_matrix, (im.shape[1], im.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            out.append(warped)
        except cv2.error as exc:
            logger.warning("ECC align fail: %s — dùng ảnh nguyên", exc)
            out.append(im)
    return out


# =====================================================================
# 2. Pseudo-HDR từ 1 ảnh (synthesize bracket → fuse)
# =====================================================================

def synthesize_bracket(
    img: np.ndarray,
    ev_steps: tuple[float, ...] = (-1.5, 0.0, 1.2),
) -> list[np.ndarray]:
    """Tạo các "exposure giả" từ 1 ảnh bằng gamma + tone curve.

    Cho mỗi EV step:
      - EV < 0 (under): nén highlights, recover detail vùng cháy
      - EV = 0 : ảnh gốc
      - EV > 0 (over): nâng shadow, recover detail vùng tối

    Đây không thay được bracket THẬT (vì pixel đã 255 không có info), nhưng
    cho Mertens 1 mức "exposedness" khác để chọn pixel midtone tốt hơn → kết
    quả tự nhiên hơn nhiều so với gamma compress single-pass.
    """
    out = []
    for ev in ev_steps:
        if abs(ev) < 0.05:
            out.append(img.copy())
            continue
        f = img.astype(np.float32) / 255.0
        # Gamma cho EV: gamma = 2^(-EV) → EV +1 → gamma 0.5 (sáng hơn)
        gamma = float(2.0 ** (-ev))
        adjusted = np.power(f, gamma)
        # Tone curve nhẹ để giảm artifact ở extreme:
        if ev < 0:
            # Under-exposure: roll-off highlights mạnh
            highlight = np.clip((adjusted - 0.7) / 0.3, 0, 1)
            adjusted = adjusted - 0.15 * highlight * (adjusted - 0.7)
        else:
            # Over-exposure: lift shadows
            shadow = np.clip((0.3 - adjusted) / 0.3, 0, 1)
            adjusted = adjusted + 0.12 * shadow * (0.3 - adjusted)
        out.append(np.clip(adjusted * 255.0, 0, 255).astype(np.uint8))
    return out


def pseudo_hdr_single(
    img: np.ndarray,
    *,
    ev_steps: tuple[float, ...] = (-1.5, 0.0, 1.2),
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposedness_weight: float = 1.5,  # ưu tiên midtone hơn
) -> np.ndarray:
    """Pseudo-HDR pipeline từ 1 ảnh."""
    bracket = synthesize_bracket(img, ev_steps=ev_steps)
    return fuse_brackets(
        bracket,
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposedness_weight=exposedness_weight,
    )


# =====================================================================
# 3. Recover blown windows — wrapper áp dụng có chọn lọc lên vùng cháy
# =====================================================================

def detect_blown_areas(
    img: np.ndarray,
    *,
    threshold_v: int = 245,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.40,
) -> np.ndarray:
    """Detect vùng cháy (V ≥ threshold) đủ to để là cửa sổ/lighting.

    Trả mask uint8 (0 hoặc 255) đã feathered.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 2] >= threshold_v).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    keep = np.zeros_like(bright)
    total = h * w
    for i in range(1, n):
        ratio = stats[i, cv2.CC_STAT_AREA] / total
        if min_area_ratio <= ratio <= max_area_ratio:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return keep
    # Smooth + dilate nhẹ để cover viền
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=8)
    return keep


def recover_blown_windows(
    img: np.ndarray,
    *,
    mode: Literal["auto", "single", "bracket"] = "auto",
    brackets: Sequence[np.ndarray] | None = None,
    align: bool = True,
    strength: float = 1.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """High-level wrapper: tự nhận diện cửa sổ cháy + áp HDR thật/pseudo.

    Args:
        img: BGR uint8 ảnh "0 EV" (hoặc bất kỳ exposure tham chiếu).
        mode:
          - "bracket": dùng `brackets` (yêu cầu ≥1 ảnh thêm)
          - "single":  pseudo-HDR từ 1 ảnh
          - "auto":    nếu có brackets → bracket; không thì single
        brackets: list ảnh thêm (-2 EV, +2 EV, etc.). KHÔNG cần kèm `img` —
                  function sẽ tự thêm img vào head.
        align: True = MTB align trước khi fuse (nên bật khi chụp tay).
        strength: 0..1, mức blend HDR result vs original ở vùng mask.
        mask: nếu None thì auto-detect blown areas. None → áp toàn ảnh nếu
              `mask` không tự detect được vùng nào.

    Returns:
        (output BGR uint8, info dict {mode, blown_pct, fused})
    """
    h, w = img.shape[:2]
    info: dict = {"mode": mode, "fused": False, "blown_pct": 0.0}

    # Quyết định mode thật sự
    use_bracket = mode == "bracket" or (mode == "auto" and brackets and len(brackets) >= 1)

    if use_bracket:
        if not brackets:
            raise ValueError("Mode bracket cần ít nhất 1 ảnh thêm trong `brackets`")
        all_imgs = [img] + list(brackets)
        if align:
            all_imgs = align_brackets(all_imgs, method="mtb")
        fused = fuse_brackets(all_imgs)
        info["mode"] = "bracket"
    else:
        fused = pseudo_hdr_single(img)
        info["mode"] = "single"
    info["fused"] = True

    # Auto-detect blown areas nếu mask=None
    if mask is None:
        mask = detect_blown_areas(img)

    blown_pct = float((mask > 128).mean()) * 100
    info["blown_pct"] = round(blown_pct, 2)

    if mask.sum() == 0:
        # Không có vùng cháy → blend toàn ảnh với strength giảm
        global_strength = float(np.clip(strength * 0.6, 0, 1))
        out = cv2.addWeighted(img, 1.0 - global_strength, fused, global_strength, 0)
        info["mask_used"] = "global"
    else:
        alpha = (mask.astype(np.float32) / 255.0 * strength)[..., None]
        out = (img.astype(np.float32) * (1 - alpha) +
               fused.astype(np.float32) * alpha)
        out = np.clip(out, 0, 255).astype(np.uint8)
        info["mask_used"] = "blown_windows"

    return out, info
