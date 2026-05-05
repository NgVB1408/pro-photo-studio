"""Auto-pilot — one-shot enhancement: classify → run baseline → studio review.

The auto-pilot is the customer-facing default: drop a photo in, get a final
image plus a per-agent scorecard. No knobs, no toggle list — just an
intelligent pipeline that makes the right call per scene.

Flow
----
1. ``classify_scene`` decides interior / exterior / aerial.
2. The baseline pipeline (perspective + real_estate + enhance_studio) runs
   with scene-aware defaults.
3. ``StudioOrchestrator`` reviews the rendered output, intervenes per
   specialist, rolls back any regression.
4. The combined ``AutopilotReport`` is returned alongside the image.

The auto-pilot is deterministic: same input + same seed → byte-identical
output. ML stages (Phase 3) can be added by passing a custom roster.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .agents import StudioOrchestrator, StudioReport

logger = logging.getLogger(__name__)


__all__ = ["AutoPilot", "AutopilotReport", "auto_enhance"]


@dataclass(frozen=True, slots=True)
class AutopilotReport:
    scene: str
    baseline_stages: tuple[str, ...]
    baseline_duration_ms: float
    studio: StudioReport
    total_duration_ms: float

    @property
    def overall(self) -> float:
        return self.studio.overall_after

    @property
    def grade(self) -> str:
        return self.studio.grade

    def as_dict(self) -> dict:
        return {
            "scene": self.scene,
            "baseline_stages": list(self.baseline_stages),
            "baseline_duration_ms": round(float(self.baseline_duration_ms), 1),
            "total_duration_ms": round(float(self.total_duration_ms), 1),
            "studio": self.studio.as_dict(),
        }


@dataclass
class AutoPilot:
    """Stateless façade around the studio orchestrator + baseline pipeline."""

    orchestrator: StudioOrchestrator | None = None
    enable_perspective: bool = True
    enable_real_estate: bool = True
    enable_enhance_studio: bool = True
    twilight: bool = False  # opt-in only

    def run(
        self,
        image: np.ndarray,
        *,
        scene: str | None = None,
    ) -> tuple[np.ndarray, AutopilotReport]:
        if image is None or image.size == 0:
            raise ValueError("Cannot auto-enhance an empty image")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 uint8 BGR image, got {image.shape}/{image.dtype}")

        t0 = time.perf_counter()
        actual_scene = scene if scene else _classify(image)
        logger.info("autopilot scene=%s shape=%s", actual_scene, image.shape)

        rendered, baseline_stages, baseline_ms = self._baseline(image, scene=actual_scene)
        orchestrator = self.orchestrator or StudioOrchestrator()
        rendered, studio = orchestrator.run(rendered, scene=actual_scene)

        return rendered, AutopilotReport(
            scene=actual_scene,
            baseline_stages=tuple(baseline_stages),
            baseline_duration_ms=baseline_ms,
            studio=studio,
            total_duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _baseline(
        self, image: np.ndarray, *, scene: str
    ) -> tuple[np.ndarray, list[str], float]:
        from .enhance import enhance_studio
        from .realestate import correct_vertical, enhance_realestate_full

        t0 = time.perf_counter()
        applied: list[str] = []
        out = image

        if self.enable_perspective and scene in ("interior", "exterior"):
            corrected, _ = correct_vertical(out, max_angle=4.0)
            out = corrected
            applied.append("perspective")

        if self.enable_real_estate:
            out, _ = enhance_realestate_full(
                out,
                seed=42,
                enable_sky=scene in ("exterior", "aerial"),
                use_ai_sky=False,  # heuristic only — keeps autopilot fast + offline
            )
            applied.append("real_estate")

        if self.enable_enhance_studio:
            params = _scene_enhance_params(scene)
            out = enhance_studio(out, params=params)
            applied.append("enhance_studio")

        if self.twilight and scene == "exterior":
            from .twilight import transform_to_twilight

            out, _ = transform_to_twilight(out, strength=0.85)
            applied.append("twilight")

        return out, applied, (time.perf_counter() - t0) * 1000.0


def _classify(image: np.ndarray) -> str:
    from .realestate import classify_scene

    return classify_scene(image).tag


def _scene_enhance_params(scene: str):
    """Per-scene defaults — interiors need shadow lift, exteriors need WB."""
    from .enhance import EnhanceParams

    if scene == "interior":
        return EnhanceParams(
            white_balance="auto",
            clahe_clip=2.4,
            highlight_recovery=0.35,
            shadow_lift=0.40,
            vibrance=0.22,
            unsharp_amount=0.45,
            unsharp_sigma=1.4,
            gamma=0.96,
        )
    if scene == "exterior":
        return EnhanceParams(
            white_balance="auto",
            clahe_clip=2.0,
            highlight_recovery=0.45,
            shadow_lift=0.25,
            vibrance=0.30,
            unsharp_amount=0.50,
            unsharp_sigma=1.5,
            gamma=0.97,
        )
    if scene == "aerial":
        return EnhanceParams(
            white_balance="auto",
            clahe_clip=1.8,
            highlight_recovery=0.35,
            shadow_lift=0.20,
            vibrance=0.25,
            unsharp_amount=0.55,
            unsharp_sigma=1.6,
            gamma=0.98,
        )
    return EnhanceParams()


def auto_enhance(
    image: np.ndarray,
    *,
    scene: str | None = None,
) -> tuple[np.ndarray, AutopilotReport]:
    """One-shot helper using a default ``AutoPilot`` instance."""
    return AutoPilot().run(image, scene=scene)
