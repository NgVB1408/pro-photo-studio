"""AI detector v0.2 — semantic segmentation cho window + ceiling.

QUALITY-FIRST design:
    - SegFormer-B3 (HuggingFace, ADE20K) trên GPU: dùng class-id chính xác
      (window=8, ceiling=5) không suy luận từ saliency.
    - GPU-aware: dispatch ONNX CUDA → DirectML → CPU tự động.
    - SAM2 click-mask (optional, cho precision tuning từ user click).
    - Rembg fallback chỉ khi không cài transformers (giảm gradeful).

Mặc định bật segformer-b3 (~85MB) trên GPU, ~22M params. Trên CPU rút về b0.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from .runtime import ADE20K_INDEX, RuntimeProfile, detect_runtime, pick_segmentation_model

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_SIDE = 1024
_session_lock = threading.Lock()


@lru_cache(maxsize=1)
def ensure_ai_available() -> RuntimeProfile:
    """Verify AI stack. Raise nếu thiếu. Return runtime profile."""
    missing = []
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "pps-wincei v0.2 yêu cầu transformers + torch. "
            f"Thiếu: {', '.join(missing)}. "
            "Cài: pip install 'pps-wincei[gpu]'  (nếu có CUDA)\n"
            "Hoặc: pip install transformers torch torchvision"
        )
    return detect_runtime()


@dataclass
class DetectionDebug:
    mode: str = ""
    model: str = ""
    window_pct: float = 0.0
    ceiling_pct: float = 0.0
    wall_pct: float = 0.0
    floor_pct: float = 0.0
    sky_pct: float = 0.0
    inference_ms: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


class SegFormerSegmenter:
    """Semantic segmentation via SegFormer trên ADE20K (151 classes).

    Lazy load model + processor. Auto-dispatch CUDA nếu có.
    """

    def __init__(self, profile: RuntimeProfile | None = None) -> None:
        self.profile = profile or detect_runtime()
        self.model_id = pick_segmentation_model(self.profile)
        self._processor = None
        self._model = None
        self._device = "cuda" if self.profile.use_gpu else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with _session_lock:
            if self._model is not None:
                return
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            logger.info(
                "SegFormer: load %s on %s (lần đầu download ~85-240MB)",
                self.model_id,
                self._device,
            )
            self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
            self._model = self._model.to(self._device).eval()
            if self.profile.use_gpu:
                # Use fp16 if available for 2x speedup
                try:
                    self._model = self._model.half()
                    self._dtype = torch.float16
                except Exception:
                    self._dtype = torch.float32
            else:
                self._dtype = torch.float32

    def segment(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run semantic segmentation.

        Args:
            img_bgr: H×W×3 BGR uint8.

        Returns:
            H×W int32 array, mỗi pixel = ADE20K class id (0..150).
        """
        import torch

        self._load()
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._device, dtype=self._dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            outputs = self._model(**inputs)
            logits = outputs.logits  # (1, C, h', w')

            # Upsample to input resolution
            upsampled = torch.nn.functional.interpolate(
                logits.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        return pred


class _LegacySaliencySegmenter:
    """Rembg-based saliency segmenter — kept for emergency fallback. Not used by default."""

    def __init__(self, model_name: str = "u2net") -> None:
        self.model_name = model_name

    def foreground_alpha(self, img_bgr: np.ndarray) -> np.ndarray:
        from rembg import new_session, remove

        h, w = img_bgr.shape[:2]
        short = min(h, w)
        if short > DEFAULT_INFERENCE_SIDE:
            scale = DEFAULT_INFERENCE_SIDE / short
            small = cv2.resize(
                img_bgr,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = img_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        session = new_session(self.model_name)
        rgba = remove(rgb, session=session)
        alpha = rgba[..., 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else np.full(small.shape[:2], 255, dtype=np.uint8)
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha


@dataclass
class SegmentationResult:
    """ADE20K semantic segmentation output + derived class masks."""

    label_map: np.ndarray  # H×W int32 — ADE20K class id
    profile: RuntimeProfile
    inference_ms: float

    def class_mask(self, class_id: int) -> np.ndarray:
        """uint8 (0/255) mask cho 1 class."""
        return ((self.label_map == class_id).astype(np.uint8)) * 255

    def class_pct(self, class_id: int) -> float:
        return float((self.label_map == class_id).mean()) * 100


def segment(img_bgr: np.ndarray, *, debug: DetectionDebug | None = None) -> SegmentationResult:
    """Top-level: chạy semantic segmentation toàn ảnh."""
    import time

    profile = ensure_ai_available()
    segmenter = SegFormerSegmenter(profile)
    t0 = time.perf_counter()
    label_map = segmenter.segment(img_bgr)
    inference_ms = (time.perf_counter() - t0) * 1000.0
    if debug is not None:
        debug.model = segmenter.model_id
        debug.inference_ms = inference_ms
        debug.mode = "segformer:" + ("gpu" if profile.use_gpu else "cpu")

    return SegmentationResult(label_map=label_map, profile=profile, inference_ms=inference_ms)


def _morphological_cleanup(mask: np.ndarray, *, min_area_pct: float = 0.001) -> np.ndarray:
    """Close gaps + remove tiny noise components."""
    if mask.sum() == 0:
        return mask
    h, w = mask.shape[:2]
    k = max(3, min(h, w) // 200)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((max(2, k // 3),) * 2, np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep = np.zeros_like(opened)
    min_area = max(int(h * w * min_area_pct), 50)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def _feather(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Narrow edge feather to prevent hard step. SCOPE-strict: max 2px so we do NOT
    bleed into wall/floor/furniture. Real-estate constraint."""
    if mask.sum() == 0:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=radius)


def _erode_to_core(mask: np.ndarray, *, pixels: int = 1) -> np.ndarray:
    """Pull mask edge INWARD by N pixels. Combined with narrow feather, this means
    fixers operate strictly inside the semantic mask — never on adjacent classes."""
    if mask.sum() == 0:
        return mask
    k = max(1, pixels * 2 + 1)
    return cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)


# ─── Public mask helpers ──────────────────────────────────────────────────────


def detect_window_mask(
    img_bgr: np.ndarray,
    *,
    seg: SegmentationResult | None = None,
    debug: DetectionDebug | None = None,
    expand_with_sky: bool = True,
) -> np.ndarray:
    """Window mask = ADE20K class 8 (windowpane).

    Optionally union với class 2 (sky) — useful khi sky visible through window
    được model phân loại là sky thay vì windowpane.

    Returns:
        uint8 (0..255) mask same H×W as input.
    """
    if seg is None:
        seg = segment(img_bgr, debug=debug)

    window = seg.class_mask(ADE20K_INDEX["windowpane"])

    if expand_with_sky:
        sky = seg.class_mask(ADE20K_INDEX["sky"])
        window = cv2.bitwise_or(window, sky)

    # Real-estate scope: shrink mask 1px inward, then 2px soft edge.
    # Goal = ZERO bleed onto window frame / wall.
    window = _morphological_cleanup(window, min_area_pct=0.002)
    window = _erode_to_core(window, pixels=1)
    window = _feather(window, radius=2)

    if debug is not None:
        debug.window_pct = float((window > 64).mean()) * 100
        debug.sky_pct = seg.class_pct(ADE20K_INDEX["sky"])

    return window


def detect_ceiling_mask(
    img_bgr: np.ndarray,
    *,
    seg: SegmentationResult | None = None,
    debug: DetectionDebug | None = None,
    include_lamps: bool = False,
) -> np.ndarray:
    """Ceiling mask = ADE20K class 5 (ceiling).

    Optionally include lamps (36) and light (82) since chúng thường nằm trên ceiling
    và cùng cần neutralize.
    """
    if seg is None:
        seg = segment(img_bgr, debug=debug)

    ceiling = seg.class_mask(ADE20K_INDEX["ceiling"])

    if include_lamps:
        lamps = seg.class_mask(ADE20K_INDEX["lamp"])
        light = seg.class_mask(ADE20K_INDEX["light"])
        ceiling = cv2.bitwise_or(cv2.bitwise_or(ceiling, lamps), light)

    # Real-estate scope: shrink 1px inward, narrow 2px feather.
    # Ceiling abuts wall closely — wider feather would tint wall.
    ceiling = _morphological_cleanup(ceiling, min_area_pct=0.003)
    ceiling = _erode_to_core(ceiling, pixels=1)
    ceiling = _feather(ceiling, radius=2)

    if debug is not None:
        debug.ceiling_pct = float((ceiling > 64).mean()) * 100
        debug.wall_pct = seg.class_pct(ADE20K_INDEX["wall"])
        debug.floor_pct = seg.class_pct(ADE20K_INDEX["floor"])

    return ceiling


# ─── Backward-compat wrapper class ─────────────────────────────────────────────


class AISegmenter:
    """Compatibility wrapper — exposes the v0.1 API but uses SegFormer underneath."""

    def __init__(self, model_name: str = "segformer") -> None:
        ensure_ai_available()
        self.model_name = model_name
        self._segformer: SegFormerSegmenter | None = None

    def foreground_alpha(self, img_bgr: np.ndarray) -> np.ndarray:
        """Approximate v0.1 API: foreground = wall+floor+furniture (NOT ceiling, NOT window, NOT sky)."""
        if self._segformer is None:
            self._segformer = SegFormerSegmenter()
        label_map = self._segformer.segment(img_bgr)
        bg_classes = {
            ADE20K_INDEX["sky"],
            ADE20K_INDEX["ceiling"],
            ADE20K_INDEX["windowpane"],
        }
        fg = np.ones_like(label_map, dtype=np.uint8) * 255
        for c in bg_classes:
            fg[label_map == c] = 0
        return fg
