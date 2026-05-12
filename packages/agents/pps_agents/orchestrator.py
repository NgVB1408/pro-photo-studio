"""Orchestrator — runs the 5 specialists in parallel, composes results, gates
through the Director, then hands off to the user.

Phases:

1. **Analyze (parallel)** — each agent's ``analyze(ctx)`` runs in its own thread.
   They share an immutable ``JobContext`` (image is read-only). OpenCV/numpy
   release the GIL during the heavy ops, so threads scale on multi-core.
2. **Apply (deterministic, serial)** — the orchestrator runs ``apply()`` of each
   agent in fixed order so each stage sees a consistent pixel grid:
   geometry → lightblend → microcontrast → cleanup → output.
3. **Director QC** — read-only review on (original, final). Encodes the user's
   3 self-questions + 5 SOP scorers and emits a verdict + recommendations.
4. **User review** — orchestrator returns ``PipelineResult`` to the caller; the
   CLI prints a digest and waits for explicit user approval before final write.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .base import BaseAgent
from .cleanup import CleanupAgent
from .director import DirectorAgent, DirectorReview
from .geometry import GeometryAgent
from .lightblend import LightBlendAgent
from .microcontrast import MicroContrastAgent
from .output import OutputAgent
from .types import JobContext, StagePlan, StageReport

log = logging.getLogger(__name__)

# Apply order is fixed to keep the pipeline deterministic.
DEFAULT_APPLY_ORDER: tuple[str, ...] = (
    "geometry",
    "lightblend",
    "microcontrast",
    "cleanup",
    "output",
)

# Metadata keys an Orchestrator-injected gene_provider populates on
# ``ctx.metadata`` before parallel analyse runs. Agents read these read-only.
META_GENE_KEY_PREFIX = "genes_"
GeneProvider = Callable[[np.ndarray], list[dict[str, Any]]]


@dataclass
class PipelineResult:
    image: np.ndarray
    plans: dict[str, StagePlan] = field(default_factory=dict)
    reports: dict[str, StageReport] = field(default_factory=dict)
    director: DirectorReview | None = None
    total_duration_s: float = 0.0
    analyze_duration_s: float = 0.0
    apply_duration_s: float = 0.0

    def summary(self) -> dict:
        return {
            "verdict": self.director.verdict if self.director else "no_review",
            "overall_score": (
                round(self.director.overall_score, 3) if self.director else None
            ),
            "question_scores": (
                {k: round(v, 3) for k, v in self.director.question_scores.items()}
                if self.director
                else {}
            ),
            "sop_scores": (
                {k: round(v, 3) for k, v in self.director.sop_scores.items()}
                if self.director
                else {}
            ),
            "stage_durations_s": {
                k: round(r.duration_s, 3) for k, r in self.reports.items()
            },
            "analyze_duration_s": round(self.analyze_duration_s, 3),
            "apply_duration_s": round(self.apply_duration_s, 3),
            "total_duration_s": round(self.total_duration_s, 3),
            "skipped_stages": [
                k for k, r in self.reports.items() if r.skipped
            ],
            "findings": self.director.findings if self.director else [],
            "recommendations": (
                self.director.recommendations if self.director else []
            ),
        }


class Orchestrator:
    def __init__(
        self,
        agents: Iterable[BaseAgent] | None = None,
        director: DirectorAgent | None = None,
        max_workers: int = 5,
        *,
        gene_providers: dict[str, GeneProvider] | None = None,
    ) -> None:
        self.agents: dict[str, BaseAgent] = {
            a.name: a
            for a in (agents or self._default_agents())
        }
        self.director = director or DirectorAgent()
        self.max_workers = max_workers
        # Mapping ``agent_name -> sync callable`` returning a list of param
        # dicts to inject as "good photo gene" hints. Optional; agents fall
        # back to baseline parameters when absent.
        self.gene_providers: dict[str, GeneProvider] = dict(gene_providers or {})

    @staticmethod
    def _default_agents() -> list[BaseAgent]:
        return [
            GeometryAgent(),
            LightBlendAgent(),
            MicroContrastAgent(),
            CleanupAgent(),
            OutputAgent(),
        ]

    def _fetch_genes(self, ctx: JobContext) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for agent_name, provider in self.gene_providers.items():
            try:
                genes = provider(ctx.image)
            except Exception:  # noqa: BLE001 — gene fetch must never break pipeline
                log.exception("gene_provider for %s failed", agent_name)
                continue
            if genes:
                out[agent_name] = list(genes)
                log.info(
                    "fetched %d %s gene(s) from provider", len(genes), agent_name
                )
        return out

    def run(self, ctx: JobContext) -> PipelineResult:
        t0 = time.perf_counter()

        # Phase 0: Optional gene retrieval. We don't mutate the caller's ctx —
        # we replace metadata with a merged dict.
        if self.gene_providers:
            genes_by_agent = self._fetch_genes(ctx)
            if genes_by_agent:
                merged_meta = dict(ctx.metadata)
                for agent_name, genes in genes_by_agent.items():
                    merged_meta[f"{META_GENE_KEY_PREFIX}{agent_name}"] = genes
                ctx = dataclasses.replace(ctx, metadata=merged_meta)

        # Phase 1: Analyze in parallel.
        plans: dict[str, StagePlan] = {}
        analyze_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(agent.analyze, ctx): name
                for name, agent in self.agents.items()
            }
            for fut in as_completed(futures):
                name = futures[fut]
                plans[name] = fut.result()
        analyze_dt = time.perf_counter() - analyze_t0

        # Phase 2: Apply deterministically.
        apply_t0 = time.perf_counter()
        image = ctx.image.copy()
        reports: dict[str, StageReport] = {}
        for stage_name in DEFAULT_APPLY_ORDER:
            if stage_name not in self.agents:
                continue
            agent = self.agents[stage_name]
            plan = plans.get(stage_name) or StagePlan(
                name=stage_name, skip=True, skip_reason="no_plan"
            )
            image, report = agent.apply(image, plan)
            reports[stage_name] = report
            log.info(
                "applied %s in %.3fs (skipped=%s)",
                stage_name,
                report.duration_s,
                report.skipped,
            )
        apply_dt = time.perf_counter() - apply_t0

        # Phase 3: Director QC.
        review = self.director.review(ctx.image, image)

        return PipelineResult(
            image=image,
            plans=plans,
            reports=reports,
            director=review,
            total_duration_s=time.perf_counter() - t0,
            analyze_duration_s=analyze_dt,
            apply_duration_s=apply_dt,
        )
