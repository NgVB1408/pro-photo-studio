"""Async SQLAlchemy engine + session factory.

Postgres in production via ``postgresql+asyncpg://...``; SQLite for tests
and single-process dev via ``sqlite+aiosqlite:///path``. Both work without
code changes.

Session lifecycle: caller acquires a session via ``async with get_session()``,
the session is committed on clean exit and rolled back on exception.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Build an async engine for the given URL."""
    if database_url.startswith("sqlite"):
        # SQLite needs aiosqlite + connect_args to enable JSON1 + foreign keys.
        if "+aiosqlite" not in database_url:
            database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def configure(database_url: str, *, echo: bool = False) -> None:
    """Initialise the module-global engine + session factory.

    Call this exactly once at application start. Subsequent calls replace
    the engine — useful for tests that rebuild the schema between cases.
    """
    global _engine, _session_factory
    _engine = create_engine(database_url, echo=echo)
    _session_factory = create_session_factory(_engine)
    logger.info("DB configured: %s", _redact(database_url))


def _redact(url: str) -> str:
    """Strip password from a database URL for safe logging."""
    if "@" not in url or "://" not in url:
        return url
    scheme_creds, rest = url.split("@", 1)
    if "://" in scheme_creds and ":" in scheme_creds.split("://", 1)[1]:
        scheme, creds = scheme_creds.split("://", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{rest}"
    return url


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding a session.

    Auto-commits on clean exit; rolls back on exception.

        async with get_session() as session:
            session.add(record)
    """
    if _session_factory is None:
        raise RuntimeError(
            "DB not configured. Call pps_api.db.engine.configure(database_url) first."
        )
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
