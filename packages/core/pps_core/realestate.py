"""Real-estate-specific enhancements (Autoenhance.ai parity layer).

Module này gom 5 tính năng đặc trưng của Autoenhance.ai mà generic enhance
pipeline không phủ:

1. ``replace_sky``        — thay trời xám bằng trời xanh / sunset
2. ``window_pull``        — kéo sáng cửa sổ cháy trong ảnh nội thất
3. ``enhance_lawn``       — boost cỏ xanh selectively
4. ``correct_vertical``   — kéo thẳng đường dọc nghiêng
5. ``classify_scene``     — auto-tag interior / exterior / aerial

Tất cả CPU-only (numpy + OpenCV), không cần ML, chạy < 1s/4K. Có thể
gộp qua ``enhance_realestate_full(img, ...)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SkyPreset = Literal["blue", "sunset", "overcast", "dramatic"]
SceneTag = Literal["interior", "exterior", "aerial", "unknown"]


# ======================================================================
# 1. SKY REPLACEMENT
# ======================================================================

# Map preset cũ → preset mới (sky_lib v2). Giữ tương thích ngược.
_SKY_PRESET_MAP: dict[str, str] = {
    "blue":      "blue_clouds",     # nâng cấp: gradient → có mây
    "sunset":    "sunset_warm",
    "overcast":  "overcast_soft",
    "dramatic":  "dramatic_storm",
    # Preset mới
    "blue_clear":     "blue_clear",
    "blue_clouds":    "blue_clouds",
    "sunset_warm":    "sunset_warm",
    "golden_hour":    "golden_hour",
    "dramatic_storm": "dramatic_storm",
    "overcast_soft":  "overcast_soft",
}


def is_outdoor_scene(img: np.ndarray, *, min_blue_ratio: float = 0.03) -> tuple[bool, dict]:
    """Indoor/outdoor classifier — chỉ outdoor mới có sky để replace.

    Tighter heuristic v2 — block interior white ceiling:
    1. Top 25% phải có **blue hue** ratio ≥3% (chứ không chỉ bright white).
       White ceiling/wall không có blue hue → sẽ fail.
    2. Top edge (1px) phải có blue hue ≥20% HOẶC (overcast white + variance > 80).
       Pure white wall variance gần 0 — sẽ fail. Cloudy sky variance cao.
    3. Top region phải có texture variance đủ — uniform white wall sẽ fail.

    Trả: (is_outdoor, debug_info_dict)
    """
    h, w = img.shape[:2]
    info: dict = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Top 25% region
    top_h = max(int(h * 0.25), 1)
    top_H = H[:top_h]
    top_S = S[:top_h]
    top_V = V[:top_h]

    # CHỈ accept blue hue (95-135) với SATURATION đủ — sky thật S≥30,
    # tường trắng/grey LED tint S<20 (không qua được)
    blue_hue = (top_H >= 95) & (top_H <= 135) & (top_V >= 130) & (top_S >= 30)
    blue_ratio = float(blue_hue.sum()) / blue_hue.size
    info["top_blue_ratio"] = blue_ratio

    # Texture variance trong top — sky có cloud texture, white wall thì uniform
    gray_top = cv2.cvtColor(img[:top_h], cv2.COLOR_BGR2GRAY)
    top_variance = float(gray_top.var())
    info["top_variance"] = top_variance

    # Top edge (1px row) — strictly blue OR (overcast with variance)
    edge_row_H = H[0]
    edge_row_S = S[0]
    edge_row_V = V[0]
    edge_blue = (edge_row_H >= 95) & (edge_row_H <= 135) & (edge_row_V >= 130) & (edge_row_S >= 30)
    edge_blue_ratio = float(edge_blue.sum()) / edge_blue.size
    # Overcast: bright + truly desaturated (gần white) — phải có V variance để không nhầm wall
    edge_overcast = (edge_row_S <= 25) & (edge_row_V >= 215) & (edge_row_V <= 245)
    edge_overcast_ratio = float(edge_overcast.sum()) / edge_overcast.size
    info["edge_blue_ratio"] = edge_blue_ratio
    info["edge_overcast_ratio"] = edge_overcast_ratio

    # Verdict: 2 paths to outdoor
    # Path 1: clear blue sky — blue dominant
    path_blue = (blue_ratio >= min_blue_ratio) and (edge_blue_ratio >= 0.20)
    # Path 2: overcast sky — bright but with cloud texture
    path_overcast = (edge_overcast_ratio >= 0.50) and (top_variance >= 80)

    is_outdoor = path_blue or path_overcast
    info["is_outdoor"] = is_outdoor
    info["reason"] = (
        "blue_sky" if path_blue else
        "overcast_sky" if path_overcast else
        "indoor"
    )
    return is_outdoor, info


def detect_sky_mask(
    img: np.ndarray,
    *,
    saturation_max: int = 35,    # tighten 60 → 35 (avoid white walls)
    value_min: int = 180,         # tighten 140 → 180
    top_bias: float = 0.45,       # tighten 0.55 → 0.45
    refine: bool = True,
    grabcut_refine: bool = True,
    alpha_matting: bool = True,
    require_outdoor: bool = True,
) -> np.ndarray:
    """Trả về mask 0..255 cho vùng trời — pro-grade với 4 tầng refinement.

    Pipeline mới (v3):
    1. **HSV color filter** — blue hue OR bright+desat
    2. **Edge density check** — sky có low edge density (loại tường có texture)
    3. **Connected component touching top** — sky phải chạm top edge
    4. **GrabCut refinement** — tinh chỉnh boundary với probabilistic segmentation
       (tách rìa cây/tóc/cột chính xác hơn nhiều threshold thuần)
    5. **Alpha matting trên rìa** — mask ra 0..255 thay 0/255, hair-detail-aware

    Args:
        saturation_max: max sat để pixel được coi là sky candidate.
        value_min: min V (brightness) để pixel là sky candidate.
        top_bias: chỉ pixel ở 0..top_bias*h được xét là sky.
        refine: True = áp connected component filter.
        grabcut_refine: True = áp GrabCut sau morph (chậm hơn ~0.3s/4K nhưng
                        rìa cây/tóc chính xác hơn nhiều).
        alpha_matting: True = mask trả 0..255 (soft) thay 0/255 (binary).
    """
    # Outdoor gate — không phải outdoor → return empty mask
    if require_outdoor:
        outdoor, info = is_outdoor_scene(img)
        if not outdoor:
            logger.info("detect_sky_mask: indoor scene (top_blue=%.3f, edge=%.3f) — skip",
                        info.get("top_blue_ratio", 0), info.get("edge_top_sky_ratio", 0))
            return np.zeros(img.shape[:2], dtype=np.uint8)

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Tier 1: HSV color filter — strict
    # Sky predominantly blue, OR very bright + truly desaturated (overcast)
    blue_hue = (H >= 95) & (H <= 135) & (V >= 130)
    bright_overcast = (S <= 20) & (V >= 200)  # very white sky
    bright_low_sat = (S <= saturation_max) & (V >= value_min) & (H >= 80) & (H <= 140)
    sky_color = blue_hue | bright_overcast | bright_low_sat

    # Tier 2: Edge density check — sky có low edge density (loại tường gạch)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Kernel lớn để smooth edge density
    edge_density = cv2.boxFilter(edges, cv2.CV_32F, (51, 51)) / 255.0
    low_edge = edge_density < 0.05  # sky thường < 5% edge pixel trong block

    # Top bias: chỉ giữ pixel ở phần trên
    y_grid = np.arange(h, dtype=np.float32)[:, None] / h
    top_mask = y_grid <= top_bias
    raw = (sky_color & top_mask & low_edge).astype(np.uint8) * 255

    if not refine or raw.sum() == 0:
        return raw

    # Morphological cleanup
    k = max(3, min(h, w) // 200)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k // 2 or 1,) * 2, np.uint8))

    # Tier 3: Keep components touching top edge
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    for i in range(1, n):
        _, y, _, _, area = stats[i]
        if y == 0 and area > (h * w) * 0.005:
            keep[labels == i] = 255

    if keep.sum() == 0:
        return np.zeros_like(raw)

    # Tier 4: GrabCut refinement — tinh chỉnh boundary
    if grabcut_refine and keep.sum() > h * w * 0.02:
        try:
            keep = _grabcut_refine_mask(img, keep)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GrabCut refine fail: %s — dùng mask gốc", exc)

    # Tier 5: Alpha matting on edges
    if alpha_matting:
        keep = _alpha_matting_edges(img, keep)
    else:
        keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=max(3, k))

    return keep


def _grabcut_refine_mask(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    iter_count: int = 3,
    erode_px: int = 8,
    dilate_px: int = 12,
) -> np.ndarray:
    """Refine mask bằng GrabCut — tách rìa cây/cột/tóc chính xác.

    Workflow:
    1. Eroded mask = sure foreground
    2. Dilated mask = possible foreground
    3. Outside dilated = sure background
    4. Run GrabCut với MASK init
    """
    h, w = img.shape[:2]
    # Resize image xuống ≤ 1080p để GrabCut chạy nhanh
    scale = min(1.0, 1080.0 / max(h, w))
    if scale < 1.0:
        small_img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        small_mask = cv2.resize(
            mask, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        small_img = img
        small_mask = mask
    sh, sw = small_img.shape[:2]

    erode_s = max(2, int(erode_px * scale))
    dilate_s = max(4, int(dilate_px * scale))

    sure_fg = cv2.erode(small_mask, np.ones((erode_s, erode_s), np.uint8))
    possible_fg = cv2.dilate(small_mask, np.ones((dilate_s, dilate_s), np.uint8))

    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[possible_fg > 128] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 128] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(small_img, gc_mask, None, bgd_model, fgd_model,
                iter_count, mode=cv2.GC_INIT_WITH_MASK)
    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0,
    ).astype(np.uint8)
    if scale < 1.0:
        refined = cv2.resize(
            refined, (w, h), interpolation=cv2.INTER_LINEAR,
        )
        # Re-binarize sau resize
        refined = ((refined > 128).astype(np.uint8)) * 255
    return refined


def _alpha_matting_edges(
    img: np.ndarray,
    binary_mask: np.ndarray,
    *,
    edge_radius: int = 8,
    feather_sigma: float = 2.5,
) -> np.ndarray:
    """Alpha matting trên rìa mask — output 0..255 mượt cho rìa cây/tóc.

    Phương pháp: detect rìa mask → trong vùng rìa, tính alpha bằng
    closed-form-ish guided filter với image làm guide.
    """
    h, w = img.shape[:2]
    # Edge zone = vùng quanh boundary
    eroded = cv2.erode(binary_mask, np.ones((edge_radius, edge_radius), np.uint8))
    dilated = cv2.dilate(binary_mask, np.ones((edge_radius, edge_radius), np.uint8))
    edge_zone = (dilated > 128) & (eroded < 128)

    if not edge_zone.any():
        # Không có rìa → chỉ cần Gaussian feather
        return cv2.GaussianBlur(binary_mask, (0, 0), sigmaX=feather_sigma)

    # Alpha base = Gaussian-blurred binary mask (continuous 0..1)
    alpha = cv2.GaussianBlur(binary_mask.astype(np.float32) / 255.0,
                              (0, 0), sigmaX=feather_sigma)

    # Edge-aware refinement: trong edge_zone, dùng guided filter với img làm guide
    # để alpha bám theo edge của ảnh (sky/cây/tóc).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # OpenCV ximgproc có guidedFilter — không phải mọi build có. Fallback box mean.
    try:
        from cv2 import ximgproc  # type: ignore
        alpha = ximgproc.guidedFilter(
            guide=gray, src=alpha, radius=edge_radius, eps=1e-3,
        )
    except (ImportError, AttributeError):
        # Fallback: bilateral filter alpha bằng image guide (gần đúng)
        alpha_u8 = (alpha * 255).astype(np.uint8)
        alpha_u8 = cv2.bilateralFilter(
            alpha_u8, d=edge_radius * 2 + 1,
            sigmaColor=30, sigmaSpace=edge_radius,
        )
        alpha = alpha_u8.astype(np.float32) / 255.0
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def _build_sky_gradient(h: int, w: int, preset: SkyPreset) -> np.ndarray:
    """Tạo trời procedural — đã nâng cấp sang sky_lib v2 với clouds + haze.

    Wrapper giữ tương thích ngược với code/test cũ. Code mới nên gọi trực tiếp
    `sky_lib.generate_sky()` để có thêm tham số seed.
    """
    from .sky_lib import generate_sky
    new_preset = _SKY_PRESET_MAP.get(preset, preset)
    return generate_sky(h, w, preset=new_preset)


def replace_sky(
    img: np.ndarray,
    *,
    preset: SkyPreset = "blue",
    sky_image: np.ndarray | None = None,
    sky_source: Literal["procedural", "real_photo", "auto"] = "auto",
    sky_id: str | None = None,
    mask: np.ndarray | None = None,
    blend_strength: float = 1.0,
    feather: int = 21,
    match_temperature: bool = True,
    cast_on_glass: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Thay trời PRO-grade với atmospheric matching.

    Pipeline so với phiên bản cũ:
      1. Procedural sky có **mây Perlin** + horizon haze (không còn gradient phẳng)
      2. Match color temperature scene → sky bias warm/cool nhẹ
      3. Match luminance để brightness sky tương đồng scene
      4. Feather mask + alpha blend
      5. **Cast sky color lên window/glass** subtle (~12% blend) — pro detail

    Args:
        img: BGR uint8 input.
        preset: tên preset (cũ: `blue/sunset/overcast/dramatic`, mới:
            `blue_clear/blue_clouds/sunset_warm/golden_hour/dramatic_storm/
            overcast_soft/twilight_blue`).
        sky_image: BGR uint8 ảnh trời CUSTOM cao nhất ưu tiên (override hết).
        sky_source:
            - "real_photo": LẤY TỪ SKY LIBRARY ẢNH THẬT (Unsplash CC0 50+ ảnh,
              auto-download cache lần đầu)
            - "procedural": gen procedural từ sky_lib.generate_sky (Perlin)
            - "auto" (default): thử real_photo trước, fallback procedural nếu
              không có internet hoặc download fail
        sky_id: chỉ định sky cụ thể từ library (Unsplash photo id). Khi None
            + source=real_photo → random theo category map từ preset.
        mask: nếu None thì auto-detect bằng ``detect_sky_mask``.
        blend_strength: 0-1.
        feather: kernel Gaussian blur cho mask edge.
        match_temperature: bias warm/cool sky theo scene temp (default True).
        cast_on_glass: cast sky color lên cửa kính (default True, subtle 12%).
        seed: int để có sky deterministic, None = random mỗi lần.

    Returns:
        (output BGR uint8, mask uint8 đã dùng).
    """
    from .sky_lib import (
        generate_sky, match_sky_to_scene, cast_sky_color_on_glass,
    )

    h, w = img.shape[:2]
    if mask is None:
        mask = detect_sky_mask(img)

    if mask.sum() == 0:
        logger.info("replace_sky: không phát hiện vùng trời, trả ảnh gốc")
        return img.copy(), mask

    # 1. Resolve preset name + load/generate sky
    new_preset = _SKY_PRESET_MAP.get(preset, preset)
    new_sky: np.ndarray | None = None

    if sky_image is not None:
        # User cấp ảnh tuỳ chỉnh — ưu tiên cao nhất
        new_sky = cv2.resize(sky_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    elif sky_source in ("real_photo", "auto"):
        # Thử load ảnh trời thật từ library
        try:
            from .sky_assets import load_sky_by_id, random_sky
            real = None
            if sky_id:
                real = load_sky_by_id(sky_id)
            else:
                # Map preset → category trong sky_assets
                real, entry = random_sky(category=new_preset, seed=seed)
                if real is not None and entry:
                    logger.info(
                        "replace_sky: dùng ảnh trời thật %s (cat=%s)",
                        entry["id"], entry["category"],
                    )
            if real is not None:
                new_sky = cv2.resize(real, (w, h), interpolation=cv2.INTER_LANCZOS4)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sky library lỗi (%s) — fallback procedural", exc,
            )

    if new_sky is None:
        # Fallback: procedural sky
        if sky_source == "real_photo":
            logger.warning(
                "sky_source=real_photo nhưng load thất bại — fallback procedural",
            )
        new_sky = generate_sky(h, w, preset=new_preset, seed=seed)

    # 1.5. Match light direction — flip sky asset nếu sun position mâu thuẫn
    try:
        from .sky_direction import match_sky_to_scene_direction
        new_sky = match_sky_to_scene_direction(new_sky, img)
    except Exception as exc:  # noqa: BLE001
        logger.debug("sky direction match skipped: %s", exc)

    # 2. Match color temperature với scene
    if match_temperature:
        new_sky = match_sky_to_scene(new_sky, img, mask)

    # 3. Match luminance — sky brightness tương đồng vùng trời gốc
    sky_pixels = img[mask > 128]
    if len(sky_pixels) > 100:
        target_v = float(cv2.cvtColor(
            sky_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV,
        )[..., 2].mean())
        new_v = float(cv2.cvtColor(new_sky, cv2.COLOR_BGR2HSV)[..., 2].mean())
        if new_v > 1:
            scale = np.clip((target_v / new_v) * 0.6 + 0.4, 0.6, 1.4)
            new_sky = np.clip(
                new_sky.astype(np.float32) * scale, 0, 255,
            ).astype(np.uint8)

    # 4. Extend mask DOWN bằng dilate để cover horizon haze (sunset glow leak)
    # Chỉ dilate xuống dưới — không nở lên trên (tránh đè cây/building cao)
    h_img, w_img = img.shape[:2]
    extend_px = max(8, int(h_img * 0.012))
    kernel_down = np.zeros((extend_px * 2 + 1, 3), dtype=np.uint8)
    kernel_down[extend_px:, :] = 1  # asymmetric, only below
    mask = cv2.dilate(mask, kernel_down)

    # 4b. Auto-scale feather theo image size — feather phải ≥ 1% cạnh dài để
    # blend mượt trên ảnh 6K. Default 21 quá nhỏ cho ảnh lớn.
    auto_feather = max(feather, max(h_img, w_img) // 100)
    if auto_feather > 1:
        k = auto_feather if auto_feather % 2 == 1 else auto_feather + 1
        mask_f = cv2.GaussianBlur(mask, (k, k), 0)
    else:
        mask_f = mask
    alpha = (mask_f.astype(np.float32) / 255.0) * blend_strength
    alpha = alpha[..., None]
    out = img.astype(np.float32) * (1 - alpha) + new_sky.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 5. Cast sky color lên cửa kính (subtle, pro detail)
    if cast_on_glass:
        out = cast_sky_color_on_glass(out, new_sky, mask_f, strength=0.12)

    return out, mask_f


# ======================================================================
# 2. WINDOW PULL (interior overexposed window recovery)
# ======================================================================

def detect_blown_windows(
    img: np.ndarray,
    *,
    value_threshold: int = 240,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.35,
) -> np.ndarray:
    """Detect cửa sổ bị cháy trong ảnh nội thất."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    bright = (V >= value_threshold).astype(np.uint8) * 255

    h, w = img.shape[:2]
    total = h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    keep = np.zeros_like(bright)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        ratio = area / total
        if min_area_ratio <= ratio <= max_area_ratio:
            keep[labels == i] = 255
    # Smooth edges
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=5)
    return keep


def window_pull(
    img: np.ndarray,
    *,
    strength: float = 0.7,
    mask: np.ndarray | None = None,
    brackets: list[np.ndarray] | None = None,
    use_hdr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Kéo sáng cửa sổ cháy → recover detail.

    Pipeline (mới, đã upgrade):
      - Nếu `use_hdr=True` (default) và có `brackets` → **HDR Mertens fusion**
        (pro workflow, recover detail thật từ multiple exposures)
      - Nếu chỉ 1 ảnh + `use_hdr=True` → **pseudo-HDR**: synthesize 3
        exposure giả, fuse Mertens (tốt hơn nhiều gamma compress đơn thuần)
      - Nếu `use_hdr=False` → fallback gamma compress + CLAHE (legacy, cho
        backward compat với test cũ)

    Args:
        img: BGR uint8 — ảnh tham chiếu (thường là "0 EV").
        strength: 0..1 mức blend recover vs original (mặc định 0.7).
        mask: nếu None thì auto-detect bằng ``detect_blown_windows``.
        brackets: list ảnh exposure khác (ngoài img). 0-len = pseudo-HDR.
        use_hdr: True (mặc định) = dùng HDR Mertens. False = gamma cũ.

    Returns:
        (output BGR uint8, mask uint8 đã dùng).
    """
    if mask is None:
        mask = detect_blown_windows(img)

    if mask.sum() == 0:
        return img.copy(), mask

    if use_hdr:
        # HDR pipeline thật sự
        from .hdr import recover_blown_windows
        out, _ = recover_blown_windows(
            img,
            mode="bracket" if brackets else "single",
            brackets=brackets,
            align=True,
            strength=float(np.clip(strength, 0, 1)),
            mask=mask,
        )
        return out, mask

    # ===== Legacy gamma + CLAHE (backward compat) =====
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    V = hsv[..., 2] / 255.0
    gamma = 1.0 + 1.5 * strength
    V_compressed = np.power(V, gamma)
    V_u8 = (V_compressed * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    V_clahe = clahe.apply(V_u8).astype(np.float32) / 255.0
    hsv[..., 2] = V_clahe * 255
    recovered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    recovered_lab = cv2.cvtColor(recovered, cv2.COLOR_BGR2LAB).astype(np.float32)
    recovered_lab[..., 2] -= 6
    recovered = cv2.cvtColor(
        np.clip(recovered_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR,
    )
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + recovered.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), mask


# ======================================================================
# 3. LAWN / GRASS ENHANCEMENT
# ======================================================================

def detect_lawn_mask(
    img: np.ndarray,
    *,
    hue_min: int = 30,
    hue_max: int = 90,
    sat_min: int = 25,
    bottom_bias: float = 0.45,
) -> np.ndarray:
    """Detect cỏ: hue green + bias phần dưới ảnh."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    green = (H >= hue_min) & (H <= hue_max) & (S >= sat_min) & (V >= 30) & (V <= 220)

    y_grid = np.arange(h, dtype=np.float32)[:, None] / h
    bottom_mask = y_grid >= bottom_bias

    raw = (green & bottom_mask).astype(np.uint8) * 255
    if raw.sum() == 0:
        return raw

    k = max(3, min(h, w) // 250)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=max(3, k))
    return raw


def enhance_lawn(
    img: np.ndarray,
    *,
    sat_boost: float = 0.5,
    hue_shift: int = -3,
    value_lift: float = 0.08,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boost cỏ: tăng saturation, dịch hue về xanh tươi, lift shadows."""
    if mask is None:
        mask = detect_lawn_mask(img)
    if mask.sum() == 0:
        return img.copy(), mask

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    alpha = (mask.astype(np.float32) / 255.0)

    # Hue shift (toward richer green)
    hsv[..., 0] = (hsv[..., 0] + hue_shift * alpha) % 180
    # Saturation boost
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + sat_boost * alpha), 0, 255)
    # Value lift (lift shadows in grass)
    V = hsv[..., 2] / 255.0
    V_lifted = V + value_lift * alpha * (1 - V)  # only lift dark areas
    hsv[..., 2] = np.clip(V_lifted * 255, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, mask


# ======================================================================
# 4. VERTICAL / PERSPECTIVE CORRECTION
# ======================================================================

@dataclass
class VerticalReport:
    angle_deg: float       # tilt angle detected (positive = right tilt)
    line_count: int        # number of vertical-ish lines used
    rotated: bool          # whether correction was applied
    upright_skew: float = 0.0           # skew áp dụng nếu dùng upright warp
    upright_direction: str = ""          # "up"/"down" nếu warp perspective


def correct_vertical(
    img: np.ndarray,
    *,
    max_angle: float = 8.0,
    min_lines: int = 6,
    crop: bool = True,
    upright: bool = True,
) -> tuple[np.ndarray, VerticalReport]:
    """Sửa nghiêng dọc — 2 cấp:

    1. **Upright (4-point perspective warp)** — sửa converging verticals khi
       chụp building từ thấp/cao. Đây là pro-grade, tương đương Adobe Upright.
    2. Nếu không đủ điều kiện upright (VP quá gần center hoặc ảnh thẳng rồi),
       fallback về **rotate 2D** (sửa tilt nhẹ do cầm máy lệch).

    Args:
        max_angle: chỉ rotate fallback nếu |tilt| ≤ max_angle.
        min_lines: cần ít nhất N đường dọc.
        crop: True = crop biên đen sau rotate.
        upright: True = thử perspective warp trước, False = chỉ dùng rotate 2D.
    """
    h, w = img.shape[:2]

    # ========== TIER 1: Perspective upright warp ==========
    if upright:
        from .perspective import correct_upright
        try:
            warped, report = correct_upright(img, max_skew=0.32, min_lines=8)
            if report.applied:
                return warped, VerticalReport(
                    angle_deg=report.angle_estimate_deg,
                    line_count=report.lines_used,
                    rotated=True,
                    upright_skew=report.skew,
                    upright_direction=report.direction,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("upright fail: %s — fallback rotate 2D", exc)

    # ========== TIER 2: Rotate 2D fallback ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180, apertureSize=3)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 360, threshold=200)
    if lines is None:
        return img.copy(), VerticalReport(0.0, 0, False)

    angles = []
    for rho, theta in lines[:, 0, :]:
        deg = np.degrees(theta)
        if deg < 90:
            tilt = deg
        else:
            tilt = deg - 180
        if abs(tilt) <= max_angle:
            angles.append(tilt)

    if len(angles) < min_lines:
        return img.copy(), VerticalReport(0.0, len(angles), False)

    median_tilt = float(np.median(angles))
    if abs(median_tilt) < 0.3:
        return img.copy(), VerticalReport(median_tilt, len(angles), False)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), -median_tilt, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if crop:
        rad = np.radians(abs(median_tilt))
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        new_w = int(w * cos_a - h * sin_a) if w * cos_a > h * sin_a else int(h * cos_a - w * sin_a)
        new_h = int(h * cos_a - w * sin_a) if h * cos_a > w * sin_a else int(w * cos_a - h * sin_a)
        new_w = max(new_w, int(w * 0.9))
        new_h = max(new_h, int(h * 0.9))
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        rotated = rotated[y0:y0 + new_h, x0:x0 + new_w]

    return rotated, VerticalReport(median_tilt, len(angles), True)


# ======================================================================
# 5. SCENE CLASSIFICATION (interior / exterior / aerial)
# ======================================================================

@dataclass
class SceneReport:
    tag: SceneTag
    confidence: float
    sky_ratio: float
    grass_ratio: float
    avg_brightness: float
    edge_density: float


def classify_scene(img: np.ndarray) -> SceneReport:
    """Heuristic scene tagger — không cần ML.

    Features used:
      - sky_ratio: phần trăm ảnh detected là trời
      - grass_ratio: phần trăm cỏ
      - avg brightness
      - edge density top-half (indoor có ceiling/đèn → density cao)
    """
    h, w = img.shape[:2]

    sky = detect_sky_mask(img, refine=True)
    grass = detect_lawn_mask(img)
    sky_ratio = float((sky > 128).mean())
    grass_ratio = float((grass > 128).mean())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(gray.mean()) / 255.0

    edges = cv2.Canny(gray, 80, 180)
    top_edges = edges[: h // 2]
    edge_density = float(top_edges.mean()) / 255.0

    # Decision tree
    if sky_ratio > 0.15 and grass_ratio > 0.10 and avg_brightness > 0.35:
        # Lots of sky AND grass → likely aerial / drone
        if sky_ratio > 0.35 or grass_ratio > 0.30:
            tag: SceneTag = "aerial"
            conf = min(0.95, sky_ratio + grass_ratio)
        else:
            tag = "exterior"
            conf = 0.7 + 0.2 * (sky_ratio + grass_ratio)
    elif sky_ratio > 0.10 and edge_density < 0.08:
        tag = "exterior"
        conf = 0.6 + 0.3 * sky_ratio
    elif edge_density > 0.10 and sky_ratio < 0.05:
        tag = "interior"
        conf = 0.6 + 0.3 * edge_density
    elif sky_ratio < 0.03 and grass_ratio < 0.03:
        tag = "interior"
        conf = 0.55
    else:
        tag = "unknown"
        conf = 0.4

    return SceneReport(
        tag=tag,
        confidence=float(np.clip(conf, 0, 0.99)),
        sky_ratio=sky_ratio,
        grass_ratio=grass_ratio,
        avg_brightness=avg_brightness,
        edge_density=edge_density,
    )


# ======================================================================
# COMPOSITE PIPELINE: tag-aware all-in-one
# ======================================================================

@dataclass
class RealEstateReport:
    scene: SceneReport
    vertical: VerticalReport
    sky_replaced: bool
    windows_recovered: bool
    lawn_enhanced: bool
    sky_decision: str = ""           # "skip:reason" | "replace:preset:reason"
    sky_preset_used: str = ""        # preset cuối cùng dùng (sau auto-override)


def enhance_realestate_full(
    img: np.ndarray,
    *,
    sky_preset: SkyPreset = "blue",
    sky_image: np.ndarray | None = None,
    brackets: list[np.ndarray] | None = None,
    enable_sky: bool = True,
    enable_window_pull: bool = True,
    enable_lawn: bool = True,
    enable_vertical: bool = True,
    sky_blend: float = 0.85,
    window_strength: float = 0.7,
    lawn_boost: float = 0.5,
    seed: int | None = None,
    smart_sky_skip: bool = True,
    smart_preset_override: bool = True,
    enable_indoor_color: bool = True,
    indoor_wb_strength: float = 0.85,
    indoor_clarity: float = 0.30,
    indoor_shadow_lift: float = 0.30,
    indoor_vibrance: float = 0.15,
    indoor_sharpen: float = 0.30,
    use_ai_sky: bool = True,
) -> tuple[np.ndarray, RealEstateReport]:
    """Tag-aware full pipeline.

    Pipeline:
    1. classify_scene → tag interior/exterior/aerial
    2. correct_vertical
    3. **Smart sky decision** (mới — Pro v3):
       - Nếu sky gốc đã đẹp (golden/twilight/dramatic) → skip replace
       - Nếu scene có warm indoor glow + user chọn day preset → override sang
         twilight_blue/sunset_warm để không mâu thuẫn lighting
       - Otherwise: replace với user preset
    4. window_pull cho interior
    5. enhance_lawn cho exterior

    Args:
        smart_sky_skip: True = bỏ qua replace nếu sky gốc đẹp (default True).
        smart_preset_override: True = auto chọn preset khớp lighting (default True).
    """
    scene = classify_scene(img)
    out = img.copy()

    # Vertical correction first (rotate + crop) to keep coords clean
    vert_report = VerticalReport(0.0, 0, False)
    if enable_vertical:
        out, vert_report = correct_vertical(out)

    sky_done = False
    sky_decision_str = ""
    sky_preset_used = sky_preset

    if enable_sky and scene.tag in ("exterior", "aerial") and scene.sky_ratio > 0.05:
        # Compute sky mask — ưu tiên AI segmentation (rembg) nếu có cài,
        # fallback sang HSV heuristic. Cả 2 đều trả mask 0..255 cùng shape.
        if use_ai_sky:
            try:
                from .sky_seg_ai import detect_sky_mask_ai
                sky_mask = detect_sky_mask_ai(out, fallback=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AI sky fail: %s — fallback HSV", exc)
                sky_mask = detect_sky_mask(out)
        else:
            sky_mask = detect_sky_mask(out)

        # Smart decision: skip / replace + chọn preset
        if smart_sky_skip or smart_preset_override:
            from .sky_quality import auto_decide_sky_action
            decision = auto_decide_sky_action(
                out, sky_mask, str(sky_preset),
                respect_user_preset=not smart_preset_override,
            )
            sky_decision_str = f"{decision.action}:{decision.chosen_preset}:{decision.reason}"
            logger.info("sky decision: %s", sky_decision_str)

            if decision.action == "skip" and smart_sky_skip:
                # Sky đẹp sẵn → giữ nguyên
                sky_preset_used = sky_preset  # ghi user preset, không override
                # sky_done = False (không replace)
            else:
                # Replace với preset đã quyết
                effective_preset = (
                    decision.chosen_preset if smart_preset_override
                    else sky_preset
                )
                sky_preset_used = effective_preset
                out, _ = replace_sky(
                    out, preset=effective_preset, sky_image=sky_image,
                    blend_strength=sky_blend, mask=sky_mask, seed=seed,
                )
                sky_done = True
        else:
            # Disable smart logic — replace luôn
            out, _ = replace_sky(
                out, preset=sky_preset, sky_image=sky_image,
                blend_strength=sky_blend, mask=sky_mask, seed=seed,
            )
            sky_done = True
            sky_preset_used = sky_preset

    win_done = False
    if enable_window_pull and scene.tag == "interior":
        out, mask = window_pull(out, strength=window_strength, brackets=brackets)
        win_done = mask.sum() > 0

    # Pro indoor enhance — selective WB tường + clarity bề mặt đá/gỗ + shadow lift
    # Generic pipeline trước đây bỏ qua interior color enhance → tungsten cast giữ
    # nguyên trên tường trắng. Bước này fix điều đó.
    if enable_indoor_color and scene.tag == "interior":
        from .indoor_pro import enhance_interior_pro
        out, _ = enhance_interior_pro(
            out,
            wb_strength=indoor_wb_strength,
            clarity_strength=indoor_clarity,
            shadow_lift_amount=indoor_shadow_lift,
            vibrance_amount=indoor_vibrance,
            sharpen_amount=indoor_sharpen,
        )

    lawn_done = False
    if enable_lawn and scene.tag in ("exterior", "aerial"):
        out, mask = enhance_lawn(out, sat_boost=lawn_boost)
        lawn_done = mask.sum() > 0

    return out, RealEstateReport(
        scene=scene,
        vertical=vert_report,
        sky_replaced=sky_done,
        windows_recovered=win_done,
        lawn_enhanced=lawn_done,
        sky_decision=sky_decision_str,
        sky_preset_used=str(sky_preset_used),
    )


# ======================================================================
# IO helpers
# ======================================================================

def load_sky_from_path(path: str | Path) -> np.ndarray:
    """Load custom sky image (BGR uint8). Raises FileNotFoundError nếu sai path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sky image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode sky image: {p}")
    return img
