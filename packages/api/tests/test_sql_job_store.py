"""Tests for SqlJobStore — uses SQLite in-memory.

We exercise the same behavioural contract as ``InMemoryJobStore`` plus
SQL-specific edge cases (duplicate insert error, JSON round-trip of Report).
The same tests can be re-pointed at Postgres in CI by overriding the
DATABASE_URL fixture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from pps_api.db import Base, SqlJobStore, create_engine, create_session_factory
from pps_api.schemas import JobStatus
from pps_api.services.job_store import JobRecord
from pps_core.types import Report, StageReport
from sqlalchemy.ext.asyncio import AsyncSession


@pytest_asyncio.fixture
async def session_factory():
    """Build a fresh in-memory SQLite engine + session factory per test."""
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = create_session_factory(engine)

    @asynccontextmanager
    async def _ctx() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    yield _ctx
    await engine.dispose()


@pytest_asyncio.fixture
async def store(session_factory):
    return SqlJobStore(session_factory)


def _record(job_id: str = "abc123", status: JobStatus = JobStatus.queued) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        status=status,
        metadata={"user": "alice"},
    )


def _report() -> Report:
    return Report(
        job_id="abc123",
        stages=(
            StageReport(
                name="preflight",
                applied=True,
                duration_ms=42.0,
                warnings=(("info", "low contrast"),),
                metrics={"blur_score": 134.5},
            ),
            StageReport(
                name="real_estate",
                applied=True,
                duration_ms=1500.0,
                metrics={"scene": 1.0},
                reason="interior",
            ),
        ),
        duration_ms=1542.0,
    )


# ---------- create / get ----------


@pytest.mark.asyncio
class TestCreateGet:
    async def test_create_then_get(self, store):
        await store.create(_record())
        got = await store.get("abc123")
        assert got is not None
        assert got.job_id == "abc123"
        assert got.status == JobStatus.queued
        assert got.metadata == {"user": "alice"}

    async def test_get_missing_returns_none(self, store):
        assert await store.get("does-not-exist") is None

    async def test_create_duplicate_raises(self, store):
        await store.create(_record())
        with pytest.raises(ValueError):
            await store.create(_record())


# ---------- update ----------


@pytest.mark.asyncio
class TestUpdate:
    async def test_status_transition(self, store):
        await store.create(_record(status=JobStatus.queued))
        updated = await store.update("abc123", status=JobStatus.running)
        assert updated.status == JobStatus.running
        # Verify persisted
        got = await store.get("abc123")
        assert got.status == JobStatus.running  # type: ignore[union-attr]

    async def test_set_error_and_report(self, store):
        await store.create(_record())
        report = _report()
        updated = await store.update(
            "abc123",
            status=JobStatus.failed,
            error="Something blew up",
            report=report,
        )
        assert updated.error == "Something blew up"
        assert updated.report is not None
        assert updated.report.job_id == report.job_id
        assert len(updated.report.stages) == 2

    async def test_report_round_trip_preserves_warnings_and_metrics(self, store):
        await store.create(_record())
        original = _report()
        await store.update("abc123", report=original)
        got = await store.get("abc123")
        assert got is not None
        assert got.report is not None
        assert got.report.stages[0].warnings == (("info", "low contrast"),)
        assert got.report.stages[0].metrics == {"blur_score": 134.5}
        assert got.report.stages[1].reason == "interior"

    async def test_set_result_path(self, store):
        await store.create(_record())
        await store.update("abc123", result_path="/tmp/out/abc123.jpg")
        got = await store.get("abc123")
        assert got.result_path == "/tmp/out/abc123.jpg"  # type: ignore[union-attr]

    async def test_update_missing_raises_keyerror(self, store):
        with pytest.raises(KeyError):
            await store.update("nope", status=JobStatus.completed)


# ---------- list_recent ----------


@pytest.mark.asyncio
class TestListRecent:
    async def test_returns_empty_when_no_jobs(self, store):
        assert await store.list_recent() == []

    async def test_default_limit(self, store):
        for i in range(5):
            await store.create(_record(job_id=f"j{i}"))
        listed = await store.list_recent()
        assert len(listed) == 5

    async def test_explicit_limit(self, store):
        for i in range(10):
            await store.create(_record(job_id=f"j{i:02d}"))
        listed = await store.list_recent(limit=3)
        assert len(listed) == 3

    async def test_ordered_by_created_at_desc(self, store):
        # SQLite created_at uses microsecond resolution; insert sequentially.
        import asyncio

        for i in range(3):
            await store.create(_record(job_id=f"j{i}"))
            await asyncio.sleep(0.01)  # ensure distinct created_at
        listed = await store.list_recent()
        ids = [r.job_id for r in listed]
        # Most recent first
        assert ids == ["j2", "j1", "j0"]
