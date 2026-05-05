"""Pydantic schemas for job submission and result retrieval.

Mirrors ``pps_core.types.{Job, StageReport, Report}`` 1:1 but lives in the
API layer because Pydantic adds validation, JSON serialisation, and OpenAPI
schema generation that we don't want in the core package.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    """Lifecycle of a job in the queue."""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobCreate(BaseModel):
    """Body of ``POST /v1/jobs`` — describes what to do.

    The image itself is uploaded as multipart form data alongside this JSON.
    """

    model_config = ConfigDict(extra="forbid")

    stages: list[str] = Field(
        default_factory=list,
        description="Ordered stage names to run. Empty = no-op (validates input only).",
        examples=[["preflight", "real_estate", "twilight"]],
    )
    params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-stage parameters keyed by stage name.",
        examples=[{"twilight": {"strength": 0.7}}],
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        lt=2**32,
        description="Deterministic seed (optional). Same input + same seed = same output.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form caller metadata (user, source). Not used by stages.",
    )


class StageReportOut(BaseModel):
    """One stage's outcome — JSON-serialised ``pps_core.types.StageReport``."""

    name: str
    applied: bool = False
    skipped: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    warnings: list[tuple[str, str]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Optional structured outputs (JSON-encoded). Stage-specific keys.",
    )


class ReportOut(BaseModel):
    """Job-level report aggregating every stage's outcome."""

    job_id: str
    duration_ms: float
    halted: bool = False
    stages: list[StageReportOut] = Field(default_factory=list)


class JobOut(BaseModel):
    """``GET /v1/jobs/{id}`` — current status and (when done) the report."""

    job_id: str
    status: JobStatus
    error: str | None = None
    report: ReportOut | None = None
    result_url: str | None = Field(
        default=None,
        description="URL to fetch the final image. Available when status=completed.",
    )
