"""SQL-backed implementation of the JobStore Protocol.

Drop-in replacement for ``InMemoryJobStore``: same async interface, same
``JobRecord`` dataclass on the wire. Tests verify both implementations
satisfy the same behavioural contract.

The ``Report`` dataclass is serialised to JSON on write and deserialised
on read so the API layer never sees ORM objects. This keeps the boundary
clean and makes Protocol substitutability cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pps_core.types import Report, StageReport
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from pps_api.schemas import JobStatus
from pps_api.services.job_store import JobRecord

from .models import JobORM

logger = logging.getLogger(__name__)


SessionFactory = Callable[[], Awaitable[AsyncSession]] | Callable[[], AsyncSession]
"""Callable that yields a fresh session — typically a contextmanager
returned by ``pps_api.db.engine.get_session``. We accept either a sync
or async factory so tests can pass simple lambdas."""


class SqlJobStore:
    """Postgres / SQLite-backed job store.

    The constructor takes a session-factory callable so the store doesn't
    own engine lifecycle. In production wire it to ``pps_api.db.engine.get_session``;
    in tests pass a per-test factory.
    """

    def __init__(self, session_factory: Callable[[], object]) -> None:
        self._session_factory = session_factory

    async def create(self, record: JobRecord) -> None:
        async with self._session_factory() as session:
            orm = _from_record(record)
            session.add(orm)
            try:
                await session.flush()
            except Exception as exc:
                # Most likely IntegrityError on duplicate primary key.
                raise ValueError(f"Job already exists or invalid: {record.job_id}") from exc

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._session_factory() as session:
            orm = await session.get(JobORM, job_id)
            return _to_record(orm) if orm else None

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        error: str | None = None,
        report: Report | None = None,
        result_path: str | None = None,
    ) -> JobRecord:
        async with self._session_factory() as session:
            orm = await session.get(JobORM, job_id)
            if orm is None:
                raise KeyError(f"Job not found: {job_id}")
            if status is not None:
                orm.status = status.value
            if error is not None:
                orm.error = error
            if report is not None:
                orm.report_json = _serialise_report(report)
            if result_path is not None:
                orm.result_path = result_path
            await session.flush()
            return _to_record(orm)

    async def list_recent(self, limit: int = 50) -> list[JobRecord]:
        async with self._session_factory() as session:
            stmt = select(JobORM).order_by(desc(JobORM.created_at)).limit(limit)
            result = await session.execute(stmt)
            return [_to_record(orm) for orm in result.scalars().all()]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _from_record(record: JobRecord) -> JobORM:
    return JobORM(
        job_id=record.job_id,
        status=record.status.value,
        error=record.error,
        result_path=record.result_path,
        report_json=_serialise_report(record.report) if record.report else None,
        job_metadata=dict(record.metadata) if record.metadata else None,
    )


def _to_record(orm: JobORM) -> JobRecord:
    report = _deserialise_report(orm.report_json) if orm.report_json else None
    return JobRecord(
        job_id=orm.job_id,
        status=JobStatus(orm.status),
        error=orm.error,
        report=report,
        result_path=orm.result_path,
        metadata=dict(orm.job_metadata) if orm.job_metadata else {},
    )


def _serialise_report(report: Report) -> dict[str, object]:
    return {
        "job_id": report.job_id,
        "duration_ms": report.duration_ms,
        "halted": report.halted,
        "stages": [
            {
                "name": s.name,
                "applied": s.applied,
                "skipped": s.skipped,
                "error": s.error,
                "duration_ms": s.duration_ms,
                "warnings": [list(w) for w in s.warnings],
                "metrics": dict(s.metrics),
                "reason": s.reason,
            }
            for s in report.stages
        ],
    }


def _deserialise_report(data: dict[str, object]) -> Report:
    stages_raw = data.get("stages", [])
    stages: list[StageReport] = []
    for s in stages_raw:  # type: ignore[union-attr]
        if not isinstance(s, dict):
            continue
        warnings_raw = s.get("warnings", []) or []
        warnings = tuple(tuple(w) for w in warnings_raw if isinstance(w, list) and len(w) == 2)
        stages.append(
            StageReport(
                name=str(s.get("name", "")),
                applied=bool(s.get("applied", False)),
                skipped=bool(s.get("skipped", False)),
                error=s.get("error"),
                duration_ms=float(s.get("duration_ms", 0.0) or 0.0),
                warnings=warnings,
                metrics=dict(s.get("metrics", {}) or {}),
                reason=str(s.get("reason", "")),
            )
        )
    return Report(
        job_id=str(data.get("job_id", "")),
        stages=tuple(stages),
        duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
        halted=bool(data.get("halted", False)),
    )
