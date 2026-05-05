"""SQLAlchemy 2 ORM models — Postgres + SQLite compatible.

We keep models self-contained (no relationships across tables) for now
because the access patterns are simple: lookup by id, list recent. When
multi-tenant scoping arrives, add a foreign-key column on every row.

Type-hint policy: use ``Mapped[T]`` for column attributes so mypy can
verify ``record.field`` is the correct type without hitting SQLAlchemy
runtime objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Common declarative base for every ORM model."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobORM(Base):
    """Persisted job record. Mirrors ``pps_api.services.JobRecord``.

    The ``report`` and ``metadata`` columns store JSON blobs. SQLite uses
    its native JSON1 ext; Postgres uses JSONB if configured (not required
    for this size of data).
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    """Local filesystem path or S3 key (s3://bucket/key/...). None until completed."""

    report_json: Mapped[dict[str, Any] | None] = mapped_column("report", JSON, nullable=True)
    """Serialised ``Report`` dataclass — see services.job_store for shape."""

    job_metadata: Mapped[dict[str, str] | None] = mapped_column("metadata", JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("ix_jobs_created_at", "created_at"),
        Index("ix_jobs_status_updated_at", "status", "updated_at"),
    )


class APIKeyORM(Base):
    """Persisted API key record. Mirrors ``pps_api.security.APIKeyRecord``.

    The ``hash`` column stores the argon2id hash; the raw key never lands
    in the database. ``key_id`` is the public lookup key (first 16 chars
    of the body); ``suffix4`` is the human-display ending.
    """

    __tablename__ = "api_keys"

    key_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    env: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    hash: Mapped[str] = mapped_column(String(255), nullable=False)
    suffix4: Mapped[str] = mapped_column(String(4), nullable=False)
    scopes_json: Mapped[list[str]] = mapped_column("scopes", JSON, nullable=False, default=list)
    metadata_json: Mapped[dict[str, str] | None] = mapped_column("metadata", JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_api_keys_env_revoked", "env", "revoked_at"),)
