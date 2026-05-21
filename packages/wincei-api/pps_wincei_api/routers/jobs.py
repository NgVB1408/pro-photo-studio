"""GET /api/v1/jobs/{id} + /jobs/{id}/download + /jobs (list)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..jobs import registry
from ..schemas import JobInfo

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("", response_model=list[JobInfo], summary="List recent jobs")
async def list_jobs(limit: int = 50) -> list[JobInfo]:
    return registry.list(limit=limit)


@router.get("/{job_id}", response_model=JobInfo, summary="Job status")
async def get_job(job_id: str) -> JobInfo:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@router.get("/{job_id}/download", summary="Download zip output của job")
async def download_job(job_id: str) -> FileResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    zip_path_str = job.metadata.get("zip")
    if not zip_path_str:
        raise HTTPException(409, f"Job {job_id} chưa có output. Status: {job.status}")
    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        raise HTTPException(410, f"Zip {zip_path.name} đã bị xoá")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"wincei_{job.job_type.value}_{job_id[:8]}.zip",
    )
