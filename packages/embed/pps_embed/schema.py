"""SQLAlchemy ORM schema — async Postgres metadata DB.

Tables:

* ``photos``       — every uploaded photo (id, sha1, dimensions, owner, …)
* ``algorithms``   — every parameter set we want to remember
* ``embeddings``   — pointer table linking photos/algos to their Qdrant points
* ``audit_log``    — every job run (dataset provenance, scores, durations)
* ``dataset_entries`` — provenance for each row pulled from FiveK / LSD / SUN
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # sha1 hex
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), default=None)
    owner: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    extra_meta: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, default=None)

    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="photo")


class Algorithm(Base):
    __tablename__ = "algorithms"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # sha1 hex
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)  # canonical JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"

    qdrant_point_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    collection: Mapped[str] = mapped_column(String(64), nullable=False)
    photo_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("photos.id", ondelete="CASCADE"), default=None
    )
    algorithm_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("algorithms.id", ondelete="CASCADE"), default=None
    )
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str | None] = mapped_column(String(120), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    photo: Mapped[Photo | None] = relationship(back_populates="embeddings")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    photo_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("photos.id"), default=None
    )
    algorithm_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("algorithms.id"), default=None
    )
    dataset_provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DatasetEntry(Base):
    __tablename__ = "dataset_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    repo_id: Mapped[str] = mapped_column(String(255), nullable=False)
    split: Mapped[str] = mapped_column(String(64), nullable=False, default="train")
    row_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    photo_id: Mapped[str | None] = mapped_column(String(40), default=None)
    license_tag: Mapped[str | None] = mapped_column(String(64), default=None)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
