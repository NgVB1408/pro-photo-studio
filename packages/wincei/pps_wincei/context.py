"""Scene context classifier — đọc môi trường trước khi fix.

Output `SceneContext` mô tả:
  - room_type:       living / bedroom / kitchen / bathroom / dining / office / hallway / exterior
  - lighting:        daylight_natural / overcast / sunset_warm / evening_artificial / mixed / low_light
  - mood_style:      modern_minimal / traditional_warm / industrial / scandinavian / unknown
  - window_state:    blown_severe / blown_mild / balanced / dark / none  (heuristic + CLIP)
  - ceiling_state:   white_cool / warm_yellow / ambient_blue / mixed / dirty / none

Implementation:
  - Vision-language model: CLIP ViT-B/32 (transformers.CLIPModel, ~150MB).
  - Zero-shot scoring trên các prompt sets defined below.
  - Lightweight numeric heuristics complement CLIP (clipped %, color temp, lum dist).

Tốc độ: ~1-2s CPU, <0.3s GPU sau warm-up.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from .detector import SegmentationResult
from .runtime import ADE20K_INDEX, RuntimeProfile, detect_runtime

logger = logging.getLogger(__name__)


# ── Prompt taxonomy ───────────────────────────────────────────────────────────

ROOM_PROMPTS = {
    "living_room": "a photo of a living room with sofa and TV",
    "bedroom": "a photo of a bedroom with bed and pillows",
    "kitchen": "a photo of a kitchen with countertop and cabinets",
    "bathroom": "a photo of a bathroom with shower and sink",
    "dining_room": "a photo of a dining room with table and chairs",
    "office": "a photo of an office room with desk and chair",
    "hallway": "a photo of a hallway corridor inside a house",
    "exterior": "a photo of building exterior or outdoor scene",
    "studio": "a photo of an open-plan studio apartment",
}

LIGHTING_PROMPTS = {
    "daylight_natural": "interior photo with bright natural daylight through windows",
    "overcast": "interior photo with diffuse overcast soft daylight",
    "sunset_warm": "interior photo with warm golden hour sunset light",
    "evening_artificial": "interior photo at night with only artificial warm lamp lighting",
    "mixed": "interior photo with mixed daylight and artificial indoor lights",
    "low_light": "interior photo in dim low light",
}

MOOD_PROMPTS = {
    "modern_minimal": "modern minimalist clean interior design",
    "traditional_warm": "traditional warm cozy interior with wood and fabric",
    "industrial": "industrial loft interior with concrete and metal",
    "scandinavian": "scandinavian bright white interior",
    "luxury": "luxury high-end interior with marble and gold",
    "everyday_amateur": "ordinary everyday home interior photo without staging",
}

WINDOW_STATE_PROMPTS = {
    "blown_severe": "interior photo with windows completely blown out to pure white",
    "blown_mild": "interior photo with slightly bright over-exposed windows",
    "balanced": "interior photo with properly exposed windows showing outdoor scene",
    "dark": "interior photo with dark unlit windows or no view",
    "none": "interior photo with no visible windows",
}

CEILING_STATE_PROMPTS = {
    "white_cool": "interior with clean neutral white ceiling",
    "warm_yellow": "interior with yellow or orange warm-tinted ceiling",
    "ambient_blue": "interior with bluish cool-tinted ceiling",
    "mixed": "interior with ceiling mixing different light colors",
    "dirty": "interior with stained dirty ceiling",
    "none": "interior with no visible ceiling",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class SceneContext:
    """Inferred scene metadata used to tune fixer parameters."""

    room_type: str = "unknown"
    room_confidence: float = 0.0
    lighting: str = "unknown"
    lighting_confidence: float = 0.0
    mood_style: str = "unknown"
    mood_confidence: float = 0.0
    window_state: str = "unknown"
    window_state_confidence: float = 0.0
    ceiling_state: str = "unknown"
    ceiling_state_confidence: float = 0.0

    # Numeric measurements (heuristic, complementary to CLIP)
    window_clipped_pct: float = 0.0
    window_mean_luminance: float = 0.0
    ceiling_cast_magnitude: float = 0.0
    ceiling_color_temp_k: float = 0.0
    overall_color_temp_k: float = 0.0

    extras: dict = field(default_factory=dict)

    def is_exterior(self) -> bool:
        return self.room_type == "exterior" and self.room_confidence >= 0.4

    def is_warm_mood(self) -> bool:
        return self.mood_style in {"traditional_warm", "luxury"} or self.lighting in {
            "sunset_warm",
            "evening_artificial",
        }

    def needs_window_fix(self) -> bool:
        # Numeric measurement is GROUND TRUTH (CLIP can mis-classify)
        if self.window_clipped_pct >= 1.0:
            return True
        # CLIP fallback when no clipping detected
        if self.window_state == "none":
            return False
        if self.window_state == "balanced" and self.window_state_confidence > 0.4:
            return False
        return self.window_state in {"blown_severe", "blown_mild"}

    def needs_ceiling_fix(self) -> bool:
        # Numeric ground truth — if measurable cast exists, FIX (don't trust CLIP)
        if self.ceiling_cast_magnitude >= 5.0:
            return True
        if self.ceiling_state == "none":
            return False
        # CLIP can override only when numeric is borderline (<5)
        if self.ceiling_state == "white_cool" and self.ceiling_state_confidence > 0.4:
            return False
        return self.ceiling_cast_magnitude >= 3.0 or self.ceiling_state in {
            "warm_yellow",
            "ambient_blue",
            "mixed",
        }

    def summary(self) -> str:
        return (
            f"🏠 {self.room_type}({self.room_confidence:.0%}) "
            f"💡 {self.lighting}({self.lighting_confidence:.0%}) "
            f"🎨 {self.mood_style}({self.mood_confidence:.0%}) "
            f"🪟 {self.window_state}({self.window_state_confidence:.0%}) "
            f"🏠 {self.ceiling_state}({self.ceiling_state_confidence:.0%})"
        )


# ── CLIP zero-shot scorer ─────────────────────────────────────────────────────

_clip_lock = threading.Lock()


class _CLIPScorer:
    """Singleton CLIP zero-shot scorer. Lazy-loads model + text embeddings cache."""

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, profile: RuntimeProfile | None = None) -> None:
        self.profile = profile or detect_runtime()
        self._model = None
        self._processor = None
        self._device = "cuda" if self.profile.use_gpu else "cpu"
        self._text_cache: dict[str, "np.ndarray"] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        with _clip_lock:
            if self._model is not None:
                return
            import torch
            from transformers import CLIPModel, CLIPProcessor

            logger.info("CLIP: load %s on %s (lần đầu download ~600MB)", self.MODEL_ID, self._device)
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
            self._model = CLIPModel.from_pretrained(self.MODEL_ID).to(self._device).eval()
            if self.profile.use_gpu:
                try:
                    self._model = self._model.half()
                    self._dtype = torch.float16
                except Exception:
                    self._dtype = torch.float32
            else:
                self._dtype = torch.float32

    @staticmethod
    def _unwrap(feats) -> "torch.Tensor":
        """Handle both tensor and ModelOutput returns from CLIP get_*_features."""
        if hasattr(feats, "pooler_output"):
            return feats.pooler_output
        if hasattr(feats, "last_hidden_state"):
            return feats.last_hidden_state.mean(dim=1)
        return feats

    def _encode_texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        import torch

        key = "|".join(prompts)
        if key in self._text_cache:
            return self._text_cache[key]

        with torch.inference_mode():
            tokens = self._processor(text=list(prompts), return_tensors="pt", padding=True)
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            feats = self._unwrap(self._model.get_text_features(**tokens))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.cpu().float().numpy()
        self._text_cache[key] = arr
        return arr

    def score(self, img_bgr: np.ndarray, prompt_dict: dict[str, str]) -> dict[str, float]:
        """Return label → softmax confidence."""
        import torch

        self._load()
        labels = tuple(prompt_dict.keys())
        prompts = tuple(prompt_dict.values())

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        text_feats = self._encode_texts(prompts)  # (N, D)

        with torch.inference_mode():
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._device, dtype=self._dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            img_feats = self._unwrap(self._model.get_image_features(**inputs))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_arr = img_feats.cpu().float().numpy()  # (1, D)

        # Cosine sim * 100 → softmax
        logits = (img_arr @ text_feats.T) * 100.0
        exp = np.exp(logits - logits.max())
        probs = (exp / exp.sum()).squeeze(0)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}


@lru_cache(maxsize=1)
def _get_clip_scorer() -> _CLIPScorer:
    return _CLIPScorer()


# ── Numeric heuristic measurements ────────────────────────────────────────────


def _measure_window_metrics(img_bgr: np.ndarray, window_mask: np.ndarray) -> tuple[float, float]:
    if window_mask.sum() == 0:
        return 0.0, 0.0
    m = window_mask > 64
    if not m.any():
        return 0.0, 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum_in = gray[m]
    clipped_pct = float((lum_in > 0.95).mean()) * 100
    mean_lum = float(lum_in.mean())
    return clipped_pct, mean_lum


def _measure_ceiling_metrics(img_bgr: np.ndarray, ceiling_mask: np.ndarray) -> tuple[float, float]:
    if ceiling_mask.sum() == 0:
        return 0.0, 0.0
    m = ceiling_mask > 64
    if not m.any():
        return 0.0, 0.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_dev = abs(lab[..., 1][m].mean() - 128.0)
    b_dev = abs(lab[..., 2][m].mean() - 128.0)
    cast_magnitude = float(a_dev + b_dev)

    # Approx color temp from B channel: B>128 = warm, B<128 = cool
    # Mapping: ΔB +30 ≈ 4000K, 0 ≈ 6500K, -30 ≈ 9000K
    delta_b = float(lab[..., 2][m].mean() - 128.0)
    color_temp_k = 6500.0 - delta_b * 80.0
    color_temp_k = max(2500.0, min(10000.0, color_temp_k))
    return cast_magnitude, color_temp_k


def _estimate_overall_color_temp(img_bgr: np.ndarray) -> float:
    """Estimate scene-level color temperature (from gray-world assumption)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Use only mid-tone pixels (0.2 < L < 0.8)
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mid = (L > 0.2) & (L < 0.8)
    if not mid.any():
        return 6500.0
    r_avg = rgb[..., 0][mid].mean()
    b_avg = rgb[..., 2][mid].mean()
    # r/b ratio ≈ 1.0 at 6500K, > 1 at 3000K (warm), < 1 at 9000K (cool)
    if b_avg < 1e-3:
        return 6500.0
    ratio = r_avg / b_avg
    # Empirical map: ratio 1.4→3000K, 1.0→6500K, 0.7→9000K
    color_temp = 6500.0 + (1.0 - ratio) * 8000.0
    return float(max(2500.0, min(10000.0, color_temp)))


# ── Public API ────────────────────────────────────────────────────────────────


def classify_scene(
    img_bgr: np.ndarray,
    *,
    seg: SegmentationResult | None = None,
    window_mask: np.ndarray | None = None,
    ceiling_mask: np.ndarray | None = None,
    enable_clip: bool = True,
) -> SceneContext:
    """Run full scene classification + measurement.

    Args:
        img_bgr: input.
        seg: optional SegmentationResult to skip re-segmentation downstream.
        window_mask/ceiling_mask: optional precomputed masks.
        enable_clip: True → run CLIP zero-shot. False → numeric heuristics only.

    Returns:
        SceneContext with classification + measurements.
    """
    ctx = SceneContext()

    # Numeric measurements (always run, cheap)
    if window_mask is not None:
        clipped, mean_lum = _measure_window_metrics(img_bgr, window_mask)
        ctx.window_clipped_pct = clipped
        ctx.window_mean_luminance = mean_lum
    if ceiling_mask is not None:
        cast, k = _measure_ceiling_metrics(img_bgr, ceiling_mask)
        ctx.ceiling_cast_magnitude = cast
        ctx.ceiling_color_temp_k = k
    ctx.overall_color_temp_k = _estimate_overall_color_temp(img_bgr)

    # Provide seg-based extras
    if seg is not None:
        ctx.extras["seg_wall_pct"] = seg.class_pct(ADE20K_INDEX["wall"])
        ctx.extras["seg_floor_pct"] = seg.class_pct(ADE20K_INDEX["floor"])
        ctx.extras["seg_sky_pct"] = seg.class_pct(ADE20K_INDEX["sky"])
        ctx.extras["seg_window_pct"] = seg.class_pct(ADE20K_INDEX["windowpane"])
        ctx.extras["seg_ceiling_pct"] = seg.class_pct(ADE20K_INDEX["ceiling"])

    if not enable_clip:
        # Fallback heuristic-only classification
        return _fill_context_heuristic(ctx)

    # CLIP zero-shot classification
    try:
        scorer = _get_clip_scorer()
        room_scores = scorer.score(img_bgr, ROOM_PROMPTS)
        lighting_scores = scorer.score(img_bgr, LIGHTING_PROMPTS)
        mood_scores = scorer.score(img_bgr, MOOD_PROMPTS)
        window_scores = scorer.score(img_bgr, WINDOW_STATE_PROMPTS)
        ceiling_scores = scorer.score(img_bgr, CEILING_STATE_PROMPTS)

        ctx.room_type, ctx.room_confidence = _argmax_score(room_scores)
        ctx.lighting, ctx.lighting_confidence = _argmax_score(lighting_scores)
        ctx.mood_style, ctx.mood_confidence = _argmax_score(mood_scores)
        ctx.window_state, ctx.window_state_confidence = _argmax_score(window_scores)
        ctx.ceiling_state, ctx.ceiling_state_confidence = _argmax_score(ceiling_scores)

        ctx.extras["clip_room_scores"] = room_scores
        ctx.extras["clip_lighting_scores"] = lighting_scores
    except Exception as exc:
        logger.warning("CLIP scoring failed: %s — fallback heuristic", exc)
        return _fill_context_heuristic(ctx)

    return ctx


def _argmax_score(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0], float(best[1])


def _fill_context_heuristic(ctx: SceneContext) -> SceneContext:
    """When CLIP unavailable, infer rough labels from numeric heuristics only."""
    # window
    if ctx.window_clipped_pct > 8:
        ctx.window_state = "blown_severe"
        ctx.window_state_confidence = 0.7
    elif ctx.window_clipped_pct > 2:
        ctx.window_state = "blown_mild"
        ctx.window_state_confidence = 0.6
    elif ctx.extras.get("seg_window_pct", 0) > 0.5:
        ctx.window_state = "balanced"
        ctx.window_state_confidence = 0.55
    else:
        ctx.window_state = "none"
        ctx.window_state_confidence = 0.5

    # ceiling
    if ctx.ceiling_cast_magnitude > 12:
        ctx.ceiling_state = "warm_yellow" if ctx.ceiling_color_temp_k < 6000 else "ambient_blue"
        ctx.ceiling_state_confidence = 0.65
    elif ctx.ceiling_cast_magnitude > 5:
        ctx.ceiling_state = "mixed"
        ctx.ceiling_state_confidence = 0.55
    else:
        ctx.ceiling_state = "white_cool"
        ctx.ceiling_state_confidence = 0.5

    # lighting (overall color temp)
    if ctx.overall_color_temp_k < 3800:
        ctx.lighting = "evening_artificial"
    elif ctx.overall_color_temp_k < 5200:
        ctx.lighting = "sunset_warm"
    elif ctx.overall_color_temp_k < 7000:
        ctx.lighting = "daylight_natural"
    else:
        ctx.lighting = "overcast"
    ctx.lighting_confidence = 0.5

    ctx.room_type = "unknown"
    ctx.mood_style = "unknown"
    return ctx
