"""AI-augmented sky segmentation.

Strategy: rembg (U²-Net / BiRefNet — salient object detection) cho ta một
foreground/subject mask **chính xác hơn** HSV thuần. Sky của ảnh real estate
thường = "không phải subject" + top region + sky-color filter.

Pipeline:
    1. Lazy-import rembg + onnxruntime.
    2. Downscale ảnh xuống 1024px short-side (rembg đủ accuracy ở scale này).
    3. rembg.remove() → RGBA, alpha = foreground saliency.
    4. Sky candidate = (1 - alpha) ∩ top_region ∩ HSV(blue OR bright).
    5. Connected-component touching top edge filter.
    6. Alpha matting trên rìa để soft edge.
    7. Upscale mask về size gốc (LANCZOS).

Fallback: nếu rembg không cài / model fail → fallback sang
`realestate.detect_sky_mask` heuristic. Caller không cần biết.

Usage:
    from pps_core.sky_seg_ai import detect_sky_mask_ai
    mask = detect_sky_mask_ai(img)  # luôn có mask, không raise
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REMBG_MODEL_DEFAULT = "u2net"     # general SOD; ~167MB download lần đầu
REMBG_MAX_SIDE = 1024              # rembg scale — 4K ảnh không cần full-res segmentation


_rembg_lock = threading.Lock()


@lru_cache(maxsize=1)
def _is_rembg_available() -> bool:
    """Có cài rembg + onnxruntime không. Cache result để khỏi probe nhiều lần."""
    try:
        import rembg  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError as exc:
        logger.info("rembg/onnxruntime không cài (%s) — AI sky tắt, fallback heuristic", exc)
        return False


@lru_cache(maxsize=2)
def _get_rembg_session(model_name: str):
    """Lazy create rembg session — cache để khỏi load model mỗi ảnh."""
    from rembg import new_session
    logger.info("rembg: loading session model=%s (lần đầu sẽ download ~170MB)", model_name)
    return new_session(model_name)


def is_available() -> bool:
    """Public probe — UI có thể check để show indicator."""
    return _is_rembg_available()


def _foreground_alpha(img_bgr: np.ndarray, model_name: str = REMBG_MODEL_DEFAULT) -> np.ndarray:
    """Trả uint8 alpha mask (0..255) — high = foreground/subject."""
    from rembg import remove
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    if short > REMBG_MAX_SIDE:
        scale = REMBG_MAX_SIDE / short
        small = cv2.resize(
            img_bgr, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img_bgr

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with _rembg_lock:
        session = _get_rembg_session(model_name)
        rgba = remove(rgb, session=session)  # numpy uint8 HxWx4

    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[..., 3]
    else:
        alpha = np.full(small.shape[:2], 255, dtype=np.uint8)

    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha


def _hsv_sky_color_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask uint8 — pixel có màu sky-like (blue / overcast / bright low-sat)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    blue_hue = (H >= 95) & (H <= 135) & (V >= 130) & (S >= 25)
    bright_overcast = (S <= 30) & (V >= 200)
    bright_low_sat = (S <= 50) & (V >= 170) & (H >= 80) & (H <= 140)
    sunset_warm = (H <= 25) & (V >= 180) & (S >= 60)  # sunset/golden hour
    return ((blue_hue | bright_overcast | bright_low_sat | sunset_warm)
            .astype(np.uint8) * 255)


def _alpha_feather_edges(img_bgr: np.ndarray, mask: np.ndarray, *, radius: int = 4) -> np.ndarray:
    """Smooth rìa để mask 0..255 thay vì 0/255 — match `realestate._alpha_matting_edges`
    nhưng cheaper.
    """
    if mask.sum() == 0:
        return mask
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius)
    return soft


def detect_sky_mask_ai(
    img: np.ndarray,
    *,
    top_bias: float = 0.65,
    require_outdoor: bool = True,
    fallback: bool = True,
    model_name: str = REMBG_MODEL_DEFAULT,
    debug_info: dict | None = None,
) -> np.ndarray:
    """Detect sky mask dùng rembg + HSV refinement.

    Args:
        img: BGR uint8.
        top_bias: max y-fraction để consider là sky (0..1).
        require_outdoor: True → check outdoor scene trước; indoor → empty mask.
        fallback: True → fallback heuristic nếu rembg unavailable.
        model_name: rembg model — "u2net" | "isnet-general-use" | "birefnet-general".
        debug_info: nếu pass dict, function ghi metadata vào (mode used, etc.)

    Returns:
        Mask uint8 (0..255), shape = img.shape[:2].
    """
    info = debug_info if debug_info is not None else {}
    h, w = img.shape[:2]

    # Outdoor gate — reuse existing
    if require_outdoor:
        try:
            from .realestate import is_outdoor_scene
            outdoor, scene_info = is_outdoor_scene(img)
            info["outdoor_check"] = scene_info
            if not outdoor:
                info["mode"] = "skip_indoor"
                return np.zeros((h, w), dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            logger.debug("outdoor check fail: %s — proceed", exc)

    if not _is_rembg_available():
        if not fallback:
            raise RuntimeError(
                "AI sky cần rembg + onnxruntime. Cài: `pip install -e .[sky-ai]`"
            )
        info["mode"] = "fallback_heuristic"
        from .realestate import detect_sky_mask
        return detect_sky_mask(img, require_outdoor=False)

    try:
        fg_alpha = _foreground_alpha(img, model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rembg fail: %s — fallback heuristic", exc)
        info["mode"] = "fallback_after_error"
        info["error"] = str(exc)
        if fallback:
            from .realestate import detect_sky_mask
            return detect_sky_mask(img, require_outdoor=False)
        raise

    info["mode"] = f"rembg:{model_name}"

    # Background = inverse of foreground saliency
    bg = 255 - fg_alpha
    sky_color = _hsv_sky_color_mask(img)

    # Top-region constraint
    y_grid = np.arange(h, dtype=np.float32)[:, None] / h
    top_region = (y_grid <= top_bias).astype(np.uint8) * 255
    top_region = np.broadcast_to(top_region, (h, w))

    # Combine: sky = pixel-AND(bg ≥ 128, sky_color, top_region) — soft on bg
    raw = ((bg >= 128) & (sky_color > 0) & (top_region > 0)).astype(np.uint8) * 255
    info["raw_pct"] = float((raw > 0).mean()) * 100

    if raw.sum() == 0:
        info["final_pct"] = 0.0
        return raw

    # Morph cleanup
    k = max(3, min(h, w) // 250)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((max(2, k // 2),) * 2, np.uint8))

    # Connected-component touching top
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros_like(raw)
    min_area = max(int((h * w) * 0.003), 1)
    for i in range(1, n):
        _, y, _, _, area = stats[i]
        if y <= 4 and area >= min_area:
            keep[labels == i] = 255

    if keep.sum() == 0:
        # rembg + HSV failed to find sky touching top → may be tightly cropped
        # → fallback to whole-mask to avoid empty result on legit sky shots
        if raw.sum() > h * w * 0.02:
            keep = raw.copy()
        else:
            info["final_pct"] = 0.0
            return keep

    keep = _alpha_feather_edges(img, keep, radius=4)
    info["final_pct"] = float((keep > 64).mean()) * 100
    return keep


def detect_sky_mask_smart(
    img: np.ndarray,
    *,
    prefer: Literal["ai", "heuristic"] = "ai",
    debug_info: dict | None = None,
) -> np.ndarray:
    """Top-level entry: chọn AI hoặc heuristic dựa trên availability + preference.

    Caller bình thường nên dùng function này — nó tự fallback đẹp.
    """
    if prefer == "ai":
        return detect_sky_mask_ai(img, fallback=True, debug_info=debug_info)
    from .realestate import detect_sky_mask
    if debug_info is not None:
        debug_info["mode"] = "heuristic_forced"
    return detect_sky_mask(img)
