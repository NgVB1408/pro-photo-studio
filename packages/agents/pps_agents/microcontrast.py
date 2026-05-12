"""Micro-Contrast specialist — SOP Phần 3.

Pop materials (multi-band texture: wood + marble + cabinet panels), selective
dehaze for windows, dodging & burning for cinematic light. Property-aware
parameters keyed off ``ctx.property_type``.

Gene blending: when ``Orchestrator`` injects ``ctx.metadata["genes_microcontrast"]``
(a list of param dicts retrieved from EmbedStore for photos similar to the
input), this agent blends the property baseline with the mean of those genes
at weight ``GENE_BLEND_WEIGHT``. This is the "lấy gene của ảnh đẹp tương tự"
shortcut — falls back to baseline cleanly when no genes are present.

The multi-band texture op below already implements a Laplacian-pyramid-style
band-pass (Gaussian blurs at σ = 1.2 / 4.0 / 10.0, then ``f - b1``, ``b1 - b2``,
``b2 - b3`` detail layers) so detail is added per-frequency-band rather than in
one global pass — this is what avoids the white-fringe / ghost halos visible
in naive texture boosters.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .base import BaseAgent
from .types import JobContext, PropertyType, StagePlan, StageReport

log = logging.getLogger(__name__)

# Multi-band amounts per property type. Keys map to apply() parameters.
# Each tuple = (fine_amount, mid_amount, macro_amount).
_TEXTURE_PROFILES: dict[PropertyType, tuple[float, float, float]] = {
    "villa_luxury": (0.45, 0.40, 0.18),
    "apartment_modern": (0.35, 0.30, 0.12),
    "studio_minimal": (0.25, 0.20, 0.08),
    "commercial_showroom": (0.40, 0.30, 0.10),
    "twilight_cabin": (0.35, 0.30, 0.20),
    "generic": (0.35, 0.30, 0.15),
}

_DEHAZE_AMOUNT = {
    "villa_luxury": 0.18,
    "apartment_modern": 0.10,
    "studio_minimal": 0.05,
    "commercial_showroom": 0.10,
    "twilight_cabin": 0.20,
    "generic": 0.10,
}

# How much weight the retrieved "good photo gene" gets vs the property
# baseline. 0.0 = ignore genes entirely, 1.0 = use only genes. 0.4 leaves the
# property baseline dominant but still nudges toward what worked on similar
# photos. Tunable; the Director's QC scores will tell us if we should raise it.
GENE_BLEND_WEIGHT: float = 0.4

# Hard sanity bounds on each blended parameter — protects against poisoned
# or out-of-distribution genes ever exploding the pipeline.
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "fine": (0.0, 0.7),
    "mid": (0.0, 0.7),
    "macro": (0.0, 0.4),
    "dehaze": (0.0, 0.4),
    "wood": (0.0, 0.8),
    "marble": (0.0, 0.6),
    "sharpen_amount": (0.0, 0.9),
    "sharpen_sigma": (0.5, 3.0),
}


def _gene_field(gene: dict[str, Any], *path: str, default: float | None = None) -> float | None:
    """Walk a dotted path in a gene dict; returns default if any hop missing."""
    cur: Any = gene
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _blend(baseline: float, samples: list[float], *, weight: float) -> float:
    """Convex blend baseline with mean(samples). No samples → baseline."""
    if not samples:
        return baseline
    mean = sum(samples) / len(samples)
    return (1.0 - weight) * baseline + weight * mean


def _clip_param(name: str, value: float) -> float:
    lo, hi = _PARAM_BOUNDS.get(name, (-1e9, 1e9))
    return max(lo, min(hi, value))


def blend_microcontrast_genes(
    baseline: dict[str, float],
    genes: list[dict[str, Any]],
    *,
    weight: float = GENE_BLEND_WEIGHT,
) -> dict[str, float]:
    """Convex-blend a microcontrast baseline param dict with retrieved genes.

    ``baseline`` keys: ``fine``, ``mid``, ``macro``, ``dehaze``, ``wood``,
    ``marble``, ``sharpen_amount``, ``sharpen_sigma``.

    Genes that fail validation (wrong agent tag) are skipped silently. The
    return shape mirrors ``baseline`` and every value is clipped to its sane
    range in ``_PARAM_BOUNDS``.
    """
    valid: list[dict[str, Any]] = [
        g for g in genes if isinstance(g, dict) and g.get("agent") in (None, "microcontrast")
    ]
    out: dict[str, float] = dict(baseline)
    if not valid or weight <= 0.0:
        return {k: _clip_param(k, v) for k, v in out.items()}

    field_paths: dict[str, tuple[str, ...]] = {
        "fine": ("texture", "fine"),
        "mid": ("texture", "mid"),
        "macro": ("texture", "macro"),
        "dehaze": ("dehaze_amount",),
        "wood": ("wood_clarity",),
        "marble": ("marble_clarity",),
        "sharpen_amount": ("sharpen_amount",),
        "sharpen_sigma": ("sharpen_sigma",),
    }
    for key, path in field_paths.items():
        samples = [
            v for v in (_gene_field(g, *path) for g in valid) if v is not None
        ]
        out[key] = _clip_param(key, _blend(baseline[key], samples, weight=weight))
    return out


class MicroContrastAgent(BaseAgent):
    name = "microcontrast"

    def _analyze(self, ctx: JobContext) -> StagePlan:
        img = ctx.image
        h, w = img.shape[:2]
        prop = ctx.property_type

        fine, mid, macro = _TEXTURE_PROFILES.get(prop, _TEXTURE_PROFILES["generic"])
        dehaze_amt = _DEHAZE_AMOUNT.get(prop, 0.10)

        baseline = {
            "fine": float(fine),
            "mid": float(mid),
            "macro": float(macro),
            "dehaze": float(dehaze_amt),
            "wood": 0.45,
            "marble": 0.25,
            "sharpen_amount": 0.55,
            "sharpen_sigma": 1.4,
        }
        # Orchestrator may inject "good photo gene" params from EmbedStore.
        genes_raw = ctx.metadata.get("genes_microcontrast") or []
        genes = list(genes_raw) if isinstance(genes_raw, (list, tuple)) else []
        params = blend_microcontrast_genes(baseline, genes, weight=GENE_BLEND_WEIGHT)
        if genes:
            log.info(
                "microcontrast: blended baseline with %d gene(s) at weight %.2f "
                "(fine %.3f→%.3f, dehaze %.3f→%.3f)",
                len(genes),
                GENE_BLEND_WEIGHT,
                baseline["fine"],
                params["fine"],
                baseline["dehaze"],
                params["dehaze"],
            )

        # Surface masks for hue-aware boost. Wood = warm hue, marble = low sat.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        wood = ((H >= 8) & (H <= 30) & (S >= 40) & (S <= 180) & (V >= 60) & (V <= 220))
        marble = ((S < 35) & (V >= 130) & (V <= 235))
        wood_mask = (wood.astype(np.uint8) * 255)
        marble_mask = (marble.astype(np.uint8) * 255)
        # Smooth + clean
        k = max(3, min(h, w) // 200)
        kern = np.ones((k, k), np.uint8)
        wood_mask = cv2.morphologyEx(wood_mask, cv2.MORPH_OPEN, kern)
        marble_mask = cv2.morphologyEx(marble_mask, cv2.MORPH_OPEN, kern)
        wood_mask = cv2.GaussianBlur(wood_mask, (0, 0), sigmaX=8.0)
        marble_mask = cv2.GaussianBlur(marble_mask, (0, 0), sigmaX=8.0)

        # Window-area mask for selective dehaze (bright + variance moderate).
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        window_area = ((gray > 170) & (gray < 250)).astype(np.uint8) * 255
        window_area = cv2.morphologyEx(window_area, cv2.MORPH_OPEN, kern)
        window_area = cv2.GaussianBlur(window_area, (0, 0), sigmaX=12.0)

        ops = [
            {
                "op": "multi_band_texture",
                "fine": params["fine"],
                "mid": params["mid"],
                "macro": params["macro"],
            },
            {"op": "selective_dehaze_windows", "amount": params["dehaze"]},
            {
                "op": "boost_surface_clarity_hue_aware",
                "wood_strength": params["wood"],
                "marble_strength": params["marble"],
            },
            {
                "op": "saliency_sharpen_skin_safe",
                "amount": params["sharpen_amount"],
                "sigma": params["sharpen_sigma"],
            },
        ]

        return StagePlan(
            name=self.name,
            operations=ops,
            masks={
                "wood": wood_mask,
                "marble": marble_mask,
                "window_area": window_area,
            },
            metadata={
                "property_type": prop,
                "wood_ratio": float(wood_mask.mean() / 255),
                "marble_ratio": float(marble_mask.mean() / 255),
                "window_area_ratio": float(window_area.mean() / 255),
                "gene_count": len(genes),
                "gene_blend_weight": GENE_BLEND_WEIGHT if genes else 0.0,
                "params": dict(params),
            },
        )

    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        report = StageReport(name=self.name, metrics={})
        out = image
        wood = plan.masks.get("wood")
        marble = plan.masks.get("marble")
        window = plan.masks.get("window_area")

        for op in plan.operations:
            if op["op"] == "multi_band_texture":
                out = self._multi_band_texture(
                    out, fine=op["fine"], mid=op["mid"], macro=op["macro"]
                )
                report.metrics["texture_bands"] = (op["fine"], op["mid"], op["macro"])
            elif op["op"] == "selective_dehaze_windows" and window is not None:
                out = self._selective_dehaze(out, mask=window, amount=op["amount"])
                report.metrics["dehaze_amount"] = op["amount"]
            elif op["op"] == "boost_surface_clarity_hue_aware":
                if wood is not None and wood.any():
                    out = self._guided_boost(out, mask=wood, strength=op["wood_strength"], radius=8)
                    report.metrics["wood_clarity"] = op["wood_strength"]
                if marble is not None and marble.any():
                    out = self._guided_boost(out, mask=marble, strength=op["marble_strength"], radius=14)
                    report.metrics["marble_clarity"] = op["marble_strength"]
            elif op["op"] == "saliency_sharpen_skin_safe":
                out = self._skin_safe_saliency_sharpen(out, amount=op["amount"], sigma=op["sigma"])
                report.metrics["sharpen_amount"] = op["amount"]
        return out, report

    # ------------------------------------------------------------------
    # implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _luminance_protect(img: np.ndarray) -> np.ndarray:
        """Mask in [0,1] that suppresses detail boost at very dark / very bright
        regions to prevent halos and noise amplification."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Suppress below 0.15 and above 0.85, full strength in [0.25, 0.75]
        low = np.clip((gray - 0.15) / 0.10, 0, 1)
        high = np.clip((0.85 - gray) / 0.10, 0, 1)
        return (low * high).astype(np.float32)[..., None]

    def _multi_band_texture(
        self, img: np.ndarray, *, fine: float, mid: float, macro: float
    ) -> np.ndarray:
        f = img.astype(np.float32)
        protect = self._luminance_protect(img)
        b1 = cv2.GaussianBlur(f, (0, 0), sigmaX=1.2)
        b2 = cv2.GaussianBlur(f, (0, 0), sigmaX=4.0)
        b3 = cv2.GaussianBlur(f, (0, 0), sigmaX=10.0)
        # band-pass detail layers
        fine_d = (f - b1) * fine * 0.5
        mid_d = (b1 - b2) * mid * 0.5
        macro_d = (b2 - b3) * macro * 0.5
        boost = (fine_d + mid_d + macro_d) * protect
        out = f + boost
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _selective_dehaze(
        img: np.ndarray, *, mask: np.ndarray, amount: float
    ) -> np.ndarray:
        if amount <= 0 or mask is None or not mask.any():
            return img
        try:
            from pps_core.tone import dehaze
        except Exception:
            return img
        dehazed = dehaze(img, amount=amount)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = img.astype(np.float32) * (1 - alpha) + dehazed.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _guided_boost(
        img: np.ndarray, *, mask: np.ndarray, strength: float, radius: int
    ) -> np.ndarray:
        if strength <= 0 or mask is None or not mask.any():
            return img
        try:
            from pps_core.enhance import guided_filter
        except Exception:
            return img
        base = guided_filter(img, radius=radius, eps=1e-2)
        detail = img.astype(np.float32) - base.astype(np.float32)
        boosted = base.astype(np.float32) + detail * (1.0 + strength)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        out = img.astype(np.float32) * (1 - alpha) + boosted * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _skin_safe_saliency_sharpen(
        img: np.ndarray, *, amount: float, sigma: float
    ) -> np.ndarray:
        try:
            from pps_core.saliency_sharpen import compute_saliency
        except Exception:
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
            return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        sal = compute_saliency(img)
        sal = np.clip((sal - 0.35) / 0.65, 0, 1)
        sal = cv2.GaussianBlur(sal, (51, 51), 0)
        # Skin protect (reduce mask by 70% on skin hue)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        skin = ((H <= 25) | (H >= 170)) & (S >= 40) & (S <= 175) & (V >= 60) & (V <= 230)
        sal = sal * (1.0 - 0.7 * skin.astype(np.float32))
        mask = sal[..., None]
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        out = sharp.astype(np.float32) * mask + img.astype(np.float32) * (1 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)
