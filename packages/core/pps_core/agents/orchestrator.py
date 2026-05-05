"""Studio orchestrator — runs the agent roster end-to-end on one image.

Workflow per agent:
  1. ``evaluate`` (before) — produces the public checklist + score.
  2. ``apply`` — only intervenes if the evaluation indicates a problem.
  3. ``evaluate`` (after) — proves the intervention helped (or rolls back).

The orchestrator is deterministic: same input + same roster + same scene
classification → byte-identical output. No randomness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .base import (
    AgentApplyReport,
    AgentEvaluation,
    AgentReport,
    PostProductionAgent,
    StudioReport,
    aggregate_score,
    derive_grade,
)
from .color import ColorAgent
from .composition import CompositionAgent
from .exposure import ExposureAgent
from .halo import HaloAgent
from .noise import NoiseAgent
from .sharpness import SharpnessAgent
from .sky import SkyAgent
from .vertical import VerticalAgent
from .white_balance import WhiteBalanceAgent

logger = logging.getLogger(__name__)


# Default order. Geometry first, light/colour next, finishing last.
DEFAULT_ROSTER: tuple[PostProductionAgent, ...] = (
    VerticalAgent(),
    ExposureAgent(),
    WhiteBalanceAgent(),
    NoiseAgent(),
    SkyAgent(),
    SharpnessAgent(),
    HaloAgent(),
    ColorAgent(),
    CompositionAgent(),
)

# Priorities (matched to QC weights).
DEFAULT_WEIGHTS: dict[str, float] = {
    "vertical_alignment": 0.90,
    "exposure": 1.30,
    "white_balance": 1.20,
    "noise": 0.80,
    "sky_quality": 0.90,
    "sharpness": 1.20,
    "halo": 1.10,
    "color_richness": 1.00,
    "composition": 0.80,
}


@dataclass
class StudioOrchestrator:
    """Routes one image through every specialist, aggregates a ``StudioReport``."""

    roster: tuple[PostProductionAgent, ...] = DEFAULT_ROSTER
    weights: dict[str, float] | None = None
    rollback_on_regression: bool = True
    """If an agent's intervention makes the per-category score *worse* by 0.5+
    points, revert the change. This stops a noise-misclassification from
    softening a perfectly sharp image, etc."""

    def run(
        self,
        image: np.ndarray,
        *,
        scene: str = "unknown",
    ) -> tuple[np.ndarray, StudioReport]:
        if image is None or image.size == 0:
            raise ValueError("Cannot orchestrate on an empty image")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 uint8 BGR image, got {image.shape}/{image.dtype}")

        weights = self.weights or DEFAULT_WEIGHTS
        t0 = time.perf_counter()
        rendered = image.copy()

        agents_reports: list[AgentReport] = []
        before_evals: list[AgentEvaluation] = []

        for agent in self.roster:
            try:
                before = agent.evaluate(rendered, scene=scene)
                before_evals.append(before)
                # Snapshot pre-apply image in case we need to roll back.
                pre = rendered
                applied_image, apply_report = agent.apply(
                    rendered, scene=scene, evaluation=before
                )
                if apply_report.applied:
                    after = agent.evaluate(applied_image, scene=scene)
                    if (
                        self.rollback_on_regression
                        and after.score + 0.5 < before.score
                    ):
                        logger.warning(
                            "agent %s regressed score (%.2f → %.2f) — rolling back",
                            agent.category,
                            before.score,
                            after.score,
                        )
                        rendered = pre
                        rollback_msg = f"rolled back ({before.score:.2f} → {after.score:.2f})"
                        apply_report = AgentApplyReport(
                            applied=False,
                            actions=(*apply_report.actions, rollback_msg),
                            params=apply_report.params,
                            duration_ms=apply_report.duration_ms,
                            notes="Intervention regressed quality — change discarded.",
                        )
                        after = before
                    else:
                        rendered = applied_image
                else:
                    after = before
            except Exception as exc:
                logger.warning(
                    "agent %s raised %s — skipping",
                    agent.category,
                    exc,
                    exc_info=True,
                )
                # Synthesize a neutral evaluation so the report is still well-formed.
                neutral = AgentEvaluation(
                    score=7.5,
                    checklist=(),
                    summary=f"Agent error: {type(exc).__name__}",
                    metrics={},
                )
                before_evals.append(neutral)
                agents_reports.append(
                    AgentReport(
                        name=agent.name,
                        role=agent.role,
                        before=neutral,
                        after=neutral,
                        apply_report=AgentApplyReport(applied=False, notes="agent raised exception"),
                    )
                )
                continue
            agents_reports.append(
                AgentReport(
                    name=agent.name,
                    role=agent.role,
                    before=before,
                    after=after,
                    apply_report=apply_report,
                )
            )

        # Aggregate scores using each agent's category weight.
        weight_list = [weights.get(a.category, 1.0) for a in self.roster]
        # Skip non-applicable agents (weight 0 in their evaluation).
        before_weights = [
            w if ev.metrics.get("applicable", 1.0) > 0 else 0.0
            for w, ev in zip(weight_list, before_evals, strict=False)
        ]
        after_weights = [
            w if r.after.metrics.get("applicable", 1.0) > 0 else 0.0
            for w, r in zip(weight_list, agents_reports, strict=False)
        ]
        overall_before = aggregate_score(before_evals, before_weights)
        overall_after = aggregate_score([r.after for r in agents_reports], after_weights)

        weak = sorted(
            (r for r in agents_reports if r.after.score < 8.0 and r.after.metrics.get("applicable", 1.0) > 0),
            key=lambda r: r.after.score,
        )
        if overall_after >= 9.3:
            summary = f"S-grade {scene} render — listing-ready."
        elif not weak:
            summary = f"Solid {scene} render across every category."
        elif len(weak) == 1:
            summary = f"Strong {scene} render. One category to watch: {weak[0].name}."
        else:
            summary = (
                f"{scene.capitalize()} render passes overall — {len(weak)} categories "
                f"need a closer look (start with {weak[0].name})."
            )

        report = StudioReport(
            scene=scene,
            overall_before=overall_before,
            overall_after=overall_after,
            grade=derive_grade(overall_after),
            agents=tuple(agents_reports),
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            summary=summary,
        )
        return rendered, report


