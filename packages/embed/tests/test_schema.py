"""Schema sanity — table names, foreign keys, basic in-memory CRUD on SQLite."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pps_embed.schema import (
    Algorithm,
    AuditLog,
    Base,
    DatasetEntry,
    Embedding,
    Photo,
)


def test_metadata_lists_all_tables():
    names = set(Base.metadata.tables)
    assert names == {
        "photos",
        "algorithms",
        "embeddings",
        "audit_log",
        "dataset_entries",
    }


def test_create_and_insert_in_sqlite():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as ses:
        p = Photo(id="a" * 40, width=1024, height=768, source="x.jpg")
        ses.add(p)
        a = Algorithm(id="b" * 40, name="villa_luxury", params_json='{"x":1}')
        ses.add(a)
        ses.commit()
        assert ses.scalar(sa.select(sa.func.count()).select_from(Photo)) == 1
        assert ses.scalar(sa.select(sa.func.count()).select_from(Algorithm)) == 1


def test_audit_log_holds_dataset_provenance_json():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as ses:
        log = AuditLog(
            job_id="job-1",
            dataset_provenance={"dataset": "fivek", "repo_id": "x/y", "split": "train"},
            scores={"psnr": 31.4, "ssim": 0.92},
            duration_seconds=12.3,
        )
        ses.add(log)
        ses.commit()
        loaded = ses.scalar(sa.select(AuditLog))
        assert loaded.dataset_provenance["dataset"] == "fivek"
        assert loaded.scores["psnr"] == 31.4


def test_dataset_entry_indexed_by_dataset():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as ses:
        ses.add_all(
            [
                DatasetEntry(dataset="fivek", repo_id="x/y", split="train", row_idx=0),
                DatasetEntry(dataset="lsd", repo_id="a/b", split="train", row_idx=0),
            ]
        )
        ses.commit()
        rows = ses.scalars(sa.select(DatasetEntry).where(DatasetEntry.dataset == "fivek")).all()
        assert len(rows) == 1
