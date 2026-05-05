"""Service layer — wires HTTP layer to pps_core.

Currently only exposes ``JobStore`` (in-memory job tracking) and
``run_pipeline_for_job`` (synchronous dispatcher). Phase 2.3 will add a
Celery-backed dispatcher and a Postgres-backed store sharing the same
interfaces.
"""

from __future__ import annotations

from .job_store import InMemoryJobStore, JobRecord, JobStore
from .pipeline_runner import run_pipeline_for_job

__all__ = [
    "InMemoryJobStore",
    "JobRecord",
    "JobStore",
    "run_pipeline_for_job",
]
