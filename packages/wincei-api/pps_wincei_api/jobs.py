"""In-memory + file-backed job registry.

Persist job state ra JSON để service restart không mất.
Đơn giản — đủ cho service single-instance. Cần Redis nếu scale ngang.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import settings
from .schemas import JobInfo, JobStatus, JobType


class JobRegistry:
    """Thread-safe in-memory + file-backed job store."""

    def __init__(self, store_dir: Path):
        self._dir = store_dir
        self._lock = threading.RLock()
        self._cache: dict[str, JobInfo] = {}
        self._load_existing()

    def _path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    def _load_existing(self) -> None:
        for f in self._dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                job = JobInfo(**data)
                self._cache[job.job_id] = job
            except Exception:
                continue

    def _persist(self, job: JobInfo) -> None:
        self._path(job.job_id).write_text(
            job.model_dump_json(indent=2), encoding="utf-8"
        )

    def create(
        self,
        job_type: JobType,
        inputs: list[str],
        metadata: Optional[dict] = None,
    ) -> JobInfo:
        job = JobInfo(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            status=JobStatus.queued,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            inputs=inputs,
            metadata=metadata or {},
        )
        with self._lock:
            self._cache[job.job_id] = job
            self._persist(job)
        return job

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress_pct: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        outputs: Optional[list[str]] = None,
        metadata_merge: Optional[dict] = None,
    ) -> JobInfo:
        with self._lock:
            job = self._cache.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if status is not None:
                job.status = status
            if progress_pct is not None:
                job.progress_pct = progress_pct
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if outputs is not None:
                job.outputs = outputs
            if metadata_merge:
                job.metadata = {**job.metadata, **metadata_merge}
            job.updated_at = datetime.utcnow()
            self._persist(job)
            return job

    def get(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._cache.get(job_id)

    def list(self, limit: int = 50) -> list[JobInfo]:
        with self._lock:
            jobs = sorted(self._cache.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


registry = JobRegistry(settings.jobs_dir)
