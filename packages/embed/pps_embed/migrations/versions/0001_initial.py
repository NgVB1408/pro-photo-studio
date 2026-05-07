"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "photos",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("source", sa.String(255)),
        sa.Column("owner", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("metadata", sa.JSON),
    )
    op.create_table(
        "algorithms",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("params_json", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "embeddings",
        sa.Column("qdrant_point_id", sa.String(64), primary_key=True),
        sa.Column("collection", sa.String(64), nullable=False),
        sa.Column("photo_id", sa.String(40), sa.ForeignKey("photos.id", ondelete="CASCADE")),
        sa.Column("algorithm_id", sa.String(40), sa.ForeignKey("algorithms.id", ondelete="CASCADE")),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("model", sa.String(120)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(64), index=True),
        sa.Column("photo_id", sa.String(40), sa.ForeignKey("photos.id")),
        sa.Column("algorithm_id", sa.String(40), sa.ForeignKey("algorithms.id")),
        sa.Column("dataset_provenance", sa.JSON),
        sa.Column("scores", sa.JSON),
        sa.Column("duration_seconds", sa.Float),
        sa.Column("note", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "dataset_entries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("dataset", sa.String(64), nullable=False, index=True),
        sa.Column("repo_id", sa.String(255), nullable=False),
        sa.Column("split", sa.String(64), nullable=False, server_default="train"),
        sa.Column("row_idx", sa.Integer, nullable=False),
        sa.Column("photo_id", sa.String(40)),
        sa.Column("license_tag", sa.String(64)),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("dataset_entries")
    op.drop_table("audit_log")
    op.drop_table("embeddings")
    op.drop_table("algorithms")
    op.drop_table("photos")
