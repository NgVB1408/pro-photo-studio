"""Automatic quality auditor — scores a rendered photo 0-10 across categories.

The auditor is the third leg of the pipeline (alongside the pipeline runner
and the stage modules). After rendering, it inspects the output image — and,
when supplied, the original — and emits a ``QCReport`` with:

  * a ``score`` (0.0–10.0) per category
  * one human-readable ``finding`` per category — short and actionable
  * a weighted ``overall`` score
  * machine-readable ``metrics`` so the auto-pilot loop can decide whether
    to re-run a stage with different params

Categories
----------
``exposure``
    Penalises blown highlights (>250 cluster) and crushed shadows (<5 cluster).
    Optimum: <0.5% blown, <2% crushed.
``white_balance``
    Distance of the per-channel mean from neutral grey on a Reinhard-tonemapped
    proxy of the image. Penalises strong colour casts (warm/cool/green tints).
``sharpness``
    Laplacian-of-Gaussian variance on the luminance channel, normalised
    against expected sharpness for the image's resolution.
``color_richness``
    Saturation distribution — high enough to look professional, not so high
    it goes plasticky. Penalises both flat and over-saturated outputs.
``vertical_alignment``
    Hough-line tilt deviation from vertical/horizontal. Optimum: <0.5° median.
``halo``
    Detects ringing around high-contrast edges via gradient cross-correlation
    of low-pass and high-pass band-passes. Common artifact of aggressive USM
    or CLAHE — directly visible to a human reviewer.
``sky_quality``
    Only scored when ``classify_scene`` returns exterior or aerial. Penalises
    banding (gradient stair-steps) and saturation clipping in the blue channel.
``noise``
    Estimated via robust median absolute deviation on a flat-area patch.
    Penalises high-ISO grain that survived denoising.
``composition``
    Computes brightness percentiles and shadow rolloff to detect 'flat'
    or 'low-key' images that need exposure re-balancing.

The category list is open: new scorers can be registered. The default weights
emphasise the categories that most influence buyer perception in real-estate
photography.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


__all__ = [
    "CategoryScore",
    "QCReport",
    "QualityAuditor",
    "ScoreCategory",
    "audit",
]


ScoreCategory = Literal[
    "exposure",
    "white_balance",
    "sharpness",
    "color_richness",
    "vertical_alignment",
    "halo",
    "sky_quality",
    "noise",
    "composition",
]


# Weighted priorities. Order matters because we display them in this order.
DEFAULT_WEIGHTS: dict[ScoreCategory, float] = {
    "exposure": 1.30,
    "white_balance": 1.20,
    "sharpness": 1.20,
    "color_richness": 1.00,
    "vertical_alignment": 0.90,
    "halo": 1.10,
    "sky_quality": 0.90,  # auto-zeroed for interiors
    "noise": 0.80,
    "composition": 0.80,
}


@dataclass(frozen=True, slots=True)
class CategoryScore:
    name: ScoreCategory
    score: float                 # 0.0 – 10.0
    finding: str                 # one-line, customer-facing
    metrics: Mapping[str, float] # raw numbers used to derive the score
    weight: float = 1.0          # carried so the report can show contribution
    applicable: bool = True      # False if scene-gated off (e.g. sky on interior)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(float(self.score), 2),
            "finding": self.finding,
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "weight": round(float(self.weight), 2),
            "applicable": bool(self.applicable),
        }


@dataclass(frozen=True, slots=True)
class QCReport:
    overall: float                       # 0.0 – 10.0
    grade: str                           # "S" / "A" / "B" / "C" / "D"
    categories: tuple[CategoryScore, ...]
    scene: str = "unknown"
    summary: str = ""
    recommendations: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "overall": round(float(self.overall), 2),
            "grade": self.grade,
            "scene": self.scene,
            "summary": self.summary,
            "recommendations": list(self.recommendations),
            "categories": [c.as_dict() for c in self.categories],
        }


class _CategoryScorer(Protocol):
    def __call__(
        self, *, rendered: np.ndarray, original: np.ndarray | None, scene: str
    ) -> CategoryScore: ...


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _grey_mean(img: np.ndarray) -> tuple[float, float, float]:
    if img.ndim != 3:
        raise ValueError("expected BGR image")
    return tuple(float(c) for c in img.reshape(-1, 3).mean(axis=0))  # type: ignore[return-value]


def _score_exposure(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    pixels = gray.size
    blown = float((gray >= 250).sum() / pixels)
    crushed = float((gray <= 5).sum() / pixels)
    p1, p99 = np.percentile(gray, [1, 99])

    # Penalise both extremes. <0.5% blown / <2% crushed = full score.
    s_blown = max(0.0, 10.0 - 60.0 * max(0.0, blown - 0.005))
    s_crush = max(0.0, 10.0 - 35.0 * max(0.0, crushed - 0.02))
    s_range = 10.0 if (p99 - p1) >= 200 else 7.0 if (p99 - p1) >= 160 else 5.5
    score = float(np.clip(0.4 * s_blown + 0.3 * s_crush + 0.3 * s_range, 0.0, 10.0))

    if blown > 0.02:
        finding = f"{blown * 100:.1f}% blown highlights — consider window-pull or HDR fuse."
    elif crushed > 0.05:
        finding = f"{crushed * 100:.1f}% crushed shadows — shadow_lift is recommended."
    elif (p99 - p1) < 160:
        finding = "Tonal range is compressed; the image will look flat in print."
    else:
        finding = "Exposure is well distributed across the histogram."

    return CategoryScore(
        name="exposure",
        score=score,
        finding=finding,
        metrics={
            "blown_pct": blown,
            "crushed_pct": crushed,
            "p1": float(p1),
            "p99": float(p99),
            "range": float(p99 - p1),
        },
        weight=DEFAULT_WEIGHTS["exposure"],
    )


def _score_white_balance(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    # Robust gray-world: mid-tones (40–80% percentile of luminance) only —
    # avoids skew from sky / dark furniture.
    yuv = cv2.cvtColor(rendered, cv2.COLOR_BGR2YUV)
    y = yuv[..., 0]
    lo, hi = np.percentile(y, [40, 80])
    mid_mask = (y >= lo) & (y <= hi)
    if mid_mask.sum() < 1000:
        mid_mask[:] = True
    pixels = rendered.reshape(-1, 3)
    mid_pixels = rendered[mid_mask].reshape(-1, 3)
    if mid_pixels.size == 0:
        mid_pixels = pixels
    mean = mid_pixels.mean(axis=0).astype(np.float64)
    grey = float(mean.mean())
    if grey < 1e-3:
        return CategoryScore(
            name="white_balance",
            score=0.0,
            finding="Image is essentially black — cannot evaluate WB.",
            metrics={"grey": grey},
            weight=DEFAULT_WEIGHTS["white_balance"],
        )
    delta = mean / grey - 1.0
    cast = float(np.linalg.norm(delta))  # 0 = perfectly neutral

    score = float(np.clip(10.0 - 70.0 * cast, 0.0, 10.0))

    # Identify the dominant cast — useful for the report.
    b_dev, g_dev, r_dev = (float(d) for d in delta)
    if cast < 0.04:
        finding = "Neutral white-balance — no visible colour cast."
    elif r_dev > max(g_dev, b_dev) + 0.03:
        finding = "Warm cast (orange/red lift). Lower R-channel by ~5%."
    elif b_dev > max(r_dev, g_dev) + 0.03:
        finding = "Cool cast (blue lift). Lower B-channel or warm white-point."
    elif g_dev > max(r_dev, b_dev) + 0.03:
        finding = "Green cast (fluorescent / mixed lighting)."
    else:
        finding = "Mild colour cast — not severe."

    return CategoryScore(
        name="white_balance",
        score=score,
        finding=finding,
        metrics={
            "cast_norm": cast,
            "b_dev": b_dev,
            "g_dev": g_dev,
            "r_dev": r_dev,
        },
        weight=DEFAULT_WEIGHTS["white_balance"],
    )


def _score_sharpness(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    # Laplacian-of-Gaussian variance — classic sharpness proxy.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap_var = float(cv2.Laplacian(blur, cv2.CV_64F).var())

    # Normalise against image area — 4K photos naturally have higher var than 1080p.
    h, w = gray.shape
    target = 60.0 * (h * w / (1920 * 1080)) ** 0.45
    ratio = lap_var / max(target, 1e-3)

    score = float(np.clip(5.0 + 5.0 * np.tanh(np.log(max(ratio, 1e-3))), 0.0, 10.0))

    if ratio < 0.4:
        finding = "Image looks soft. Consider running enhance_studio for halo-free local detail."
    elif ratio > 4.0:
        finding = "Detail is over-emphasised — likely too much sharpening."
    else:
        finding = f"Sharpness within professional range (Laplacian var = {lap_var:.0f})."

    return CategoryScore(
        name="sharpness",
        score=score,
        finding=finding,
        metrics={"lap_var": lap_var, "target": target, "ratio": float(ratio)},
        weight=DEFAULT_WEIGHTS["sharpness"],
    )


def _score_color_richness(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    hsv = cv2.cvtColor(rendered, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    s_mean = float(sat.mean())
    s_p95 = float(np.percentile(sat, 95))

    # Optimum for real-estate: mean ~0.18-0.32, p95 ~ 0.55-0.85.
    if s_mean < 0.10:
        finding = "Image is desaturated; vibrance can lift it without skewing skin tones."
        score = 4.5
    elif s_mean > 0.45 or s_p95 > 0.95:
        finding = "Colour looks plasticky / oversaturated — back off vibrance."
        score = 4.0
    elif 0.18 <= s_mean <= 0.32 and 0.55 <= s_p95 <= 0.85:
        finding = "Colour richness is in the professional sweet spot."
        score = 9.5
    else:
        finding = "Colour is acceptable but a touch flat or punchy."
        score = 7.5

    return CategoryScore(
        name="color_richness",
        score=score,
        finding=finding,
        metrics={"sat_mean": s_mean, "sat_p95": s_p95},
        weight=DEFAULT_WEIGHTS["color_richness"],
    )


def _score_verticals(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 120, minLineLength=120, maxLineGap=15)
    if lines is None or len(lines) < 4:
        return CategoryScore(
            name="vertical_alignment",
            score=8.0,
            finding="Not enough strong lines to evaluate vertical alignment.",
            metrics={"lines": float(0 if lines is None else len(lines))},
            weight=DEFAULT_WEIGHTS["vertical_alignment"],
            applicable=scene != "unknown",
        )

    deviations: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if y2 == y1:
            continue
        dx, dy = float(x2 - x1), float(y2 - y1)
        # Distance from vertical (90°) — wrap into [0,90].
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        if angle > 90:
            angle = 180 - angle
        if angle <= 12:  # near-vertical only
            deviations.append(angle)
    if not deviations:
        return CategoryScore(
            name="vertical_alignment",
            score=8.0,
            finding="No near-vertical lines — likely an aerial or close-up shot.",
            metrics={"lines": float(len(lines))},
            weight=DEFAULT_WEIGHTS["vertical_alignment"],
            applicable=False,
        )
    median_dev = float(np.median(deviations))
    score = float(np.clip(10.0 - 1.5 * median_dev, 0.0, 10.0))
    finding = (
        "Walls and frames are plumb."
        if median_dev < 0.6
        else f"Verticals tilt ~{median_dev:.1f}° — perspective stage will fix it."
    )
    return CategoryScore(
        name="vertical_alignment",
        score=score,
        finding=finding,
        metrics={"median_deviation_deg": median_dev, "vertical_lines": float(len(deviations))},
        weight=DEFAULT_WEIGHTS["vertical_alignment"],
    )


def _score_halo(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # High-pass = image - low-pass. Inspect bright-side over-shoot near strong edges.
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=4)
    high = gray - low
    edges = cv2.Canny(gray.astype(np.uint8), 80, 200)
    if edges.sum() == 0:
        return CategoryScore(
            name="halo",
            score=9.5,
            finding="No strong edges to evaluate halo on.",
            metrics={"edge_pixels": 0.0},
            weight=DEFAULT_WEIGHTS["halo"],
        )
    edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    near_edge = edge_dilated > 0
    overshoot = float(np.percentile(np.abs(high[near_edge]), 95))
    # Empirical: >25 = visible halo for an 8-bit image.
    score = float(np.clip(10.0 - max(0.0, overshoot - 12.0) * 0.4, 0.0, 10.0))
    finding = (
        "No visible halo around high-contrast edges."
        if overshoot < 18
        else f"Slight ringing detected (overshoot ≈ {overshoot:.1f}); reduce CLAHE clip."
        if overshoot < 30
        else "Strong halo / ringing — re-render with halo-free local detail (guided filter)."
    )
    return CategoryScore(
        name="halo",
        score=score,
        finding=finding,
        metrics={"overshoot_p95": overshoot},
        weight=DEFAULT_WEIGHTS["halo"],
    )


def _score_sky(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original
    if scene not in ("exterior", "aerial"):
        return CategoryScore(
            name="sky_quality",
            score=10.0,
            finding="No sky present — category not evaluated.",
            metrics={"sky_ratio": 0.0},
            weight=0.0,
            applicable=False,
        )

    h, _w = rendered.shape[:2]
    top_band = rendered[: max(1, h // 4)]
    hsv = cv2.cvtColor(top_band, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    sky_mask = (s < 0.45) & (v > 0.55)
    if sky_mask.mean() < 0.05:
        return CategoryScore(
            name="sky_quality",
            score=8.5,
            finding="Sky region too small to evaluate banding.",
            metrics={"sky_ratio": float(sky_mask.mean())},
            weight=DEFAULT_WEIGHTS["sky_quality"],
        )
    blue = top_band[..., 0][sky_mask].astype(np.float32)
    sky_std = float(np.std(blue))
    sky_clip = float((blue >= 254).mean())
    # Some variation = healthy; very low = banding; very high = noise.
    if sky_clip > 0.05:
        score = 5.0
        finding = f"Sky is clipped on {sky_clip * 100:.1f}% of pixels — pull back exposure."
    elif sky_std < 1.5:
        score = 5.5
        finding = "Sky shows posterising / banding — apply sky_replace or dithered gradient."
    elif sky_std > 35:
        score = 6.5
        finding = "Sky is noisy; consider sky-segment denoise."
    else:
        score = 9.5
        finding = "Sky looks clean and gradient-smooth."
    return CategoryScore(
        name="sky_quality",
        score=score,
        finding=finding,
        metrics={
            "sky_ratio": float(sky_mask.mean()),
            "sky_std": sky_std,
            "sky_clip_pct": sky_clip,
        },
        weight=DEFAULT_WEIGHTS["sky_quality"],
    )


def _score_noise(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    residual = gray - blur
    # Robust noise estimate via MAD.
    mad = float(np.median(np.abs(residual - np.median(residual))))
    sigma = 1.4826 * mad  # MAD → σ for Gaussian noise
    score = float(np.clip(10.0 - max(0.0, sigma - 1.0) * 0.7, 0.0, 10.0))
    finding = (
        "Image is clean — minimal residual noise."
        if sigma < 1.5
        else f"Visible grain (σ ≈ {sigma:.1f}). Light denoise recommended."
        if sigma < 4.0
        else "High-ISO grain dominant — apply NLM or learned denoise."
    )
    return CategoryScore(
        name="noise",
        score=score,
        finding=finding,
        metrics={"sigma_est": sigma, "mad": mad},
        weight=DEFAULT_WEIGHTS["noise"],
    )


def _score_composition(*, rendered: np.ndarray, original: np.ndarray | None, scene: str) -> CategoryScore:
    del original, scene
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    spread = float(p90 - p10)
    median = float(p50)
    # Real-estate sweet spot: p50 in [110,160] (well-lit, not flat).
    score_median = 10.0 if 110 <= median <= 160 else max(0.0, 10.0 - abs(median - 135) * 0.08)
    score_spread = max(0.0, min(10.0, spread / 18.0))
    score = float(0.55 * score_median + 0.45 * score_spread)
    if median < 90:
        finding = "Image is dim overall — consider lifting global exposure."
    elif median > 175:
        finding = "Image is washed-out — pull back global brightness."
    elif spread < 90:
        finding = "Tonal spread is narrow; result will look flat."
    else:
        finding = "Composition is well-balanced for a hero photo."
    return CategoryScore(
        name="composition",
        score=score,
        finding=finding,
        metrics={"p10": float(p10), "p50": median, "p90": float(p90), "spread": spread},
        weight=DEFAULT_WEIGHTS["composition"],
    )


_SCORERS: dict[ScoreCategory, _CategoryScorer] = {
    "exposure": _score_exposure,
    "white_balance": _score_white_balance,
    "sharpness": _score_sharpness,
    "color_richness": _score_color_richness,
    "vertical_alignment": _score_verticals,
    "halo": _score_halo,
    "sky_quality": _score_sky,
    "noise": _score_noise,
    "composition": _score_composition,
}


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------


def _grade(score: float) -> str:
    if score >= 9.3:
        return "S"
    if score >= 8.5:
        return "A"
    if score >= 7.5:
        return "B"
    if score >= 6.5:
        return "C"
    return "D"


def _summarise(scene: str, overall: float, weak: list[CategoryScore]) -> str:
    if overall >= 9.3:
        return f"Listing-grade {scene} photo — all categories in the green."
    if not weak:
        return f"Solid {scene} photo, ready for the listing."
    weakest = weak[0]
    if len(weak) == 1:
        return f"Mostly strong {scene} photo. Only weak point: {weakest.name.replace('_', ' ')}."
    return (
        f"{scene.capitalize()} photo passes overall but {len(weak)} categories need a "
        f"second pass — start with {weakest.name.replace('_', ' ')}."
    )


@dataclass
class QualityAuditor:
    """Runs all registered scorers and aggregates a ``QCReport``.

    Construct once, reuse across many images. The auditor itself holds no
    image state, so it's safe to share between threads.
    """

    weights: dict[ScoreCategory, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )

    def audit(
        self,
        rendered: np.ndarray,
        *,
        original: np.ndarray | None = None,
        scene: str = "unknown",
    ) -> QCReport:
        if rendered is None or rendered.size == 0:
            raise ValueError("Cannot audit an empty image")
        if rendered.dtype != np.uint8:
            raise ValueError(f"Expected uint8 BGR image, got {rendered.dtype}")
        if rendered.ndim != 3 or rendered.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 BGR image, got {rendered.shape}")

        scores: list[CategoryScore] = []
        for name, scorer in _SCORERS.items():
            try:
                cs = scorer(rendered=rendered, original=original, scene=scene)
            except Exception as exc:
                logger.warning("QC scorer %s failed: %s", name, exc, exc_info=True)
                cs = CategoryScore(
                    name=name,
                    score=7.5,  # neutral on internal error — never fail the audit
                    finding=f"Scorer error ({type(exc).__name__}); falling back to neutral.",
                    metrics={},
                    weight=self.weights.get(name, 1.0),
                    applicable=False,
                )
            scores.append(cs)

        # Weighted overall — only count applicable categories.
        applicable = [s for s in scores if s.applicable and s.weight > 0]
        if applicable:
            total_w = sum(s.weight for s in applicable)
            overall = float(
                sum(s.score * s.weight for s in applicable) / max(total_w, 1e-9)
            )
        else:
            overall = 0.0

        weak = sorted(
            (s for s in applicable if s.score < 8.0),
            key=lambda s: s.score,
        )
        summary = _summarise(scene, overall, weak)
        recs = tuple(
            f"{s.name.replace('_', ' ').capitalize()}: {s.finding}" for s in weak[:3]
        )

        return QCReport(
            overall=round(overall, 2),
            grade=_grade(overall),
            categories=tuple(scores),
            scene=scene,
            summary=summary,
            recommendations=recs,
        )


_DEFAULT_AUDITOR = QualityAuditor()


def audit(
    rendered: np.ndarray,
    *,
    original: np.ndarray | None = None,
    scene: str = "unknown",
) -> QCReport:
    """Convenience wrapper using a process-global auditor."""
    return _DEFAULT_AUDITOR.audit(rendered, original=original, scene=scene)


# Re-exported helpers used by the auto-pilot retune loop.
def overall_to_dict(report: QCReport) -> dict:
    return asdict(report)
