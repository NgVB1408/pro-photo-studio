"""Base specialist agent.

Subclasses implement ``_analyze`` and ``_apply``. The base wraps both with
timing, exception capture, and a guarantee that any exception turns into a
skipped stage rather than blowing up the whole pipeline. The Director can then
read warnings and decide whether to fail the job.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from .types import JobContext, StagePlan, StageReport, _Timer

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract specialist agent."""

    name: str = "base"

    # ------------------------------------------------------------------
    # public API used by Orchestrator
    # ------------------------------------------------------------------

    def analyze(self, ctx: JobContext) -> StagePlan:
        with _Timer() as t:
            try:
                plan = self._analyze(ctx)
            except Exception as exc:
                log.exception("%s.analyze failed", self.name)
                plan = StagePlan(
                    name=self.name,
                    skip=True,
                    skip_reason=f"analyze_error: {exc!r}",
                )
        plan.analyze_duration_s = t.elapsed
        plan.name = self.name
        return plan

    def apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        if plan.skip:
            return image, StageReport(
                name=self.name, skipped=True, skip_reason=plan.skip_reason
            )
        with _Timer() as t:
            try:
                out, report = self._apply(image, plan)
            except Exception as exc:
                log.exception("%s.apply failed", self.name)
                return image, StageReport(
                    name=self.name,
                    skipped=True,
                    skip_reason=f"apply_error: {exc!r}",
                )
        report.duration_s = t.elapsed
        report.name = self.name
        return out, report

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _analyze(self, ctx: JobContext) -> StagePlan: ...

    @abstractmethod
    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]: ...
