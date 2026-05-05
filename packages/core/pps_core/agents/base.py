"""Shared types for the post-production agent roster."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np


CheckStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class AgentChecklistItem:
    """One line item on an agent's public checklist.

    Customers see this verbatim on the scorecard, so phrasing matters: present-
    tense, customer-vocabulary, no engineering jargon.
    """

    label: str                 # e.g. "No blown highlights"
    status: CheckStatus = "pass"
    detail: str = ""           # e.g. "0.2% of pixels above 250"

    def as_dict(self) -> dict:
        return {"label": self.label, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class AgentEvaluation:
    """Outcome of an agent's ``evaluate`` step — diagnosis only, no edits."""

    score: float                                # 0.0 – 10.0
    checklist: tuple[AgentChecklistItem, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "score": round(float(self.score), 2),
            "summary": self.summary,
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "checklist": [c.as_dict() for c in self.checklist],
        }


@dataclass(frozen=True, slots=True)
class AgentApplyReport:
    """Outcome of an agent's ``apply`` step — what it did, with delta vs before."""

    applied: bool                                  # True if the agent intervened
    actions: tuple[str, ...] = ()                  # human-readable change list
    params: Mapping[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "applied": bool(self.applied),
            "actions": list(self.actions),
            "params": {k: round(float(v), 4) for k, v in self.params.items()},
            "duration_ms": round(float(self.duration_ms), 1),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class AgentReport:
    """Combined per-agent record — surfaced on the customer-facing scorecard."""

    name: str
    role: str
    before: AgentEvaluation
    after: AgentEvaluation
    apply_report: AgentApplyReport

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "apply": self.apply_report.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class StudioReport:
    """Top-level artefact returned by ``StudioOrchestrator.run``."""

    scene: str
    overall_before: float
    overall_after: float
    grade: str
    agents: tuple[AgentReport, ...]
    duration_ms: float
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "scene": self.scene,
            "overall_before": round(float(self.overall_before), 2),
            "overall_after": round(float(self.overall_after), 2),
            "grade": self.grade,
            "summary": self.summary,
            "duration_ms": round(float(self.duration_ms), 1),
            "agents": [a.as_dict() for a in self.agents],
        }


@runtime_checkable
class PostProductionAgent(Protocol):
    """The contract every specialist agent satisfies.

    Implementations must be pure (no global state) and idempotent — calling
    ``apply`` on an already-good image must be a no-op.
    """

    @property
    def name(self) -> str:
        """Display name shown to customers (e.g. 'Exposure Specialist')."""
        ...

    @property
    def role(self) -> str:
        """One-line job description (e.g. 'Recovers blown skies and lifted shadows')."""
        ...

    @property
    def category(self) -> str:
        """Stable identifier matching the QC category (e.g. 'exposure')."""
        ...

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        """Inspect the image; return the public checklist + a 0–10 score."""
        ...

    def apply(
        self,
        image: np.ndarray,
        *,
        scene: str,
        evaluation: AgentEvaluation,
    ) -> tuple[np.ndarray, AgentApplyReport]:
        """Correct the image when ``evaluation`` indicates a problem.

        Returns ``(image, report)``. When ``report.applied is False`` the
        returned image must be the same array passed in.
        """
        ...


def derive_grade(score: float) -> str:
    if score >= 9.3:
        return "S"
    if score >= 8.5:
        return "A"
    if score >= 7.5:
        return "B"
    if score >= 6.5:
        return "C"
    return "D"


def aggregate_score(evaluations: Sequence[AgentEvaluation], weights: Sequence[float]) -> float:
    """Weighted mean of agent scores; ignores zero-weight (non-applicable) agents."""
    if not evaluations:
        return 0.0
    total_w = 0.0
    total = 0.0
    for ev, w in zip(evaluations, weights, strict=False):
        if w <= 0:
            continue
        total += ev.score * w
        total_w += w
    if total_w <= 0:
        return 0.0
    return total / total_w
