"""Job lifecycle store — abstract interface + in-memory implementation.

The store records every submitted job and is the source of truth for
``GET /v1/jobs/{id}``. Phase 2 ships an in-memory store that's perfect for
tests and single-process local dev. Phase 2.3 will add a SQLAlchemy-backed
implementation behind the same protocol.

Concurrency: the in-memory store uses ``threading.Lock`` so multiple FastAPI
workers (under uvicorn/gunicorn) on the same process can read/write safely.
For multi-process / multi-host deployments, switch to the Postgres backend.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field, replace
from typing import Protocol

from pps_core.types import Report

from pps_api.schemas import JobStatus


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One job as observed by the store."""

    job_id: str
    status: JobStatus
    error: str | None = None
    report: Report | None = None
    result_path: str | None = None
    """Local file path or S3 key to the final image. None until status=completed."""

    metadata: dict[str, str] = field(default_factory=dict)


class JobStore(Protocol):
    """Protocol every job store must satisfy. Async-friendly."""

    async def create(self, record: JobRecord) -> None: ...

    async def get(self, job_id: str) -> JobRecord | None: ...

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        error: str | None = None,
        report: Report | None = None,
        result_path: str | None = None,
    ) -> JobRecord: ...

    async def list_recent(self, limit: int = 50) -> list[JobRecord]: ...


class InMemoryJobStore:
    """Process-local store. Good for tests + local dev. NOT for production.

    Concurrency uses both ``threading.Lock`` (for sync access from background
    tasks) and an ``asyncio.Lock`` ... actually no, we serialise everything
    through one threading.Lock since the operations are O(1) dict ops.
    """

    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._order: list[str] = []  # insertion order for list_recent

    async def create(self, record: JobRecord) -> None:
        with self._lock:
            if record.job_id in self._records:
                raise ValueError(f"Job already exists: {record.job_id}")
            self._records[record.job_id] = record
            self._order.append(record.job_id)

    async def get(self, job_id: str) -> JobRecord | None:
        # Yield to event loop so callers can rely on this being async-safe.
        await asyncio.sleep(0)
        with self._lock:
            return self._records.get(job_id)

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        error: str | None = None,
        report: Report | None = None,
        result_path: str | None = None,
    ) -> JobRecord:
        await asyncio.sleep(0)
        with self._lock:
            existing = self._records.get(job_id)
            if existing is None:
                raise KeyError(f"Job not found: {job_id}")
            updated = replace(
                existing,
                status=status if status is not None else existing.status,
                error=error if error is not None else existing.error,
                report=report if report is not None else existing.report,
                result_path=(result_path if result_path is not None else existing.result_path),
            )
            self._records[job_id] = updated
            return updated

    async def list_recent(self, limit: int = 50) -> list[JobRecord]:
        await asyncio.sleep(0)
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._records[i] for i in ids]
