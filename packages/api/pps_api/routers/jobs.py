"""``/v1/jobs`` — submit, inspect, retrieve photo enhancement jobs.

Endpoints:

    POST   /v1/jobs                 multipart upload (image + JSON body)
    GET    /v1/jobs                 list recent jobs
    GET    /v1/jobs/{job_id}        status + report (when done)
    GET    /v1/jobs/{job_id}/result binary download of the final image

Implementation notes:

- The image is decoded once via cv2.imdecode and passed to the pipeline as
  a numpy array. We don't keep the original upload on disk — only the
  enhanced output is written.
- POST returns immediately with status=queued and the runner is dispatched
  via FastAPI's background tasks. Callers poll GET /v1/jobs/{id} or use the
  webhook delivery (Phase 2.3).
- Job IDs are UUID4 hex (32 chars). Tests can override via metadata.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pps_core.types import Job
from pydantic import ValidationError

from pps_api.schemas import JobCreate, JobOut, JobStatus, ReportOut, StageReportOut
from pps_api.security import maybe_require_api_key
from pps_api.services import (
    InMemoryJobStore,
    JobRecord,
    JobStore,
    run_pipeline_for_job,
)

router = APIRouter(
    prefix="/v1/jobs",
    tags=["jobs"],
    dependencies=[Depends(maybe_require_api_key)],
)


# -- Dependency injection -----------------------------------------------------
# A single in-memory store per process is created at import time. Tests can
# override via FastAPI's dependency_overrides.

_default_store: JobStore = InMemoryJobStore()
_default_output_dir = Path("./outputs")


def get_store() -> JobStore:
    return _default_store


def get_output_dir() -> Path:
    return _default_output_dir


def _to_job_out(record: JobRecord) -> JobOut:
    report_out = None
    if record.report is not None:
        report_out = ReportOut(
            job_id=record.report.job_id,
            duration_ms=record.report.duration_ms,
            halted=record.report.halted,
            stages=[
                StageReportOut(
                    name=s.name,
                    applied=s.applied,
                    skipped=s.skipped,
                    error=s.error,
                    duration_ms=s.duration_ms,
                    warnings=[(sev, msg) for sev, msg in s.warnings],
                    metrics=dict(s.metrics),
                    reason=s.reason,
                )
                for s in record.report.stages
            ],
        )
    result_url = (
        f"/v1/jobs/{record.job_id}/result"
        if record.status == JobStatus.completed and record.result_path
        else None
    )
    return JobOut(
        job_id=record.job_id,
        status=record.status,
        error=record.error,
        report=report_out,
        result_url=result_url,
    )


# -- Endpoints ----------------------------------------------------------------


@router.post(
    "",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a photo enhancement job",
)
async def create_job(
    bg: BackgroundTasks,
    image: Annotated[UploadFile, File(description="Image file (JPEG/PNG/WebP)")],
    body: Annotated[
        str,
        Form(
            description="JSON-encoded JobCreate body",
            examples=['{"stages": ["preflight"], "seed": 42}'],
        ),
    ],
    store: Annotated[JobStore, Depends(get_store)],
    output_dir: Annotated[Path, Depends(get_output_dir)],
) -> JobOut:
    """Create + dispatch a job. Returns 202 with status=queued."""
    try:
        spec = JobCreate.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=json.loads(exc.json()),
        ) from exc

    raw = await image.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image upload",
        )

    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image (unsupported format or corrupt data)",
        )

    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        stages=tuple(spec.stages),
        params=spec.params,
        seed=spec.seed,
        metadata=spec.metadata,
    )
    record = JobRecord(
        job_id=job_id,
        status=JobStatus.queued,
        metadata=dict(spec.metadata),
    )
    await store.create(record)

    bg.add_task(
        run_pipeline_for_job,
        job=job,
        image=decoded,
        store=store,
        output_dir=output_dir,
    )

    return _to_job_out(record)


@router.get("", response_model=list[JobOut], summary="List recent jobs")
async def list_jobs(
    store: Annotated[JobStore, Depends(get_store)],
    limit: int = 50,
) -> list[JobOut]:
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 500",
        )
    records = await store.list_recent(limit=limit)
    return [_to_job_out(r) for r in records]


@router.get("/{job_id}", response_model=JobOut, summary="Inspect a job")
async def get_job(
    job_id: str,
    store: Annotated[JobStore, Depends(get_store)],
) -> JobOut:
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return _to_job_out(record)


@router.get(
    "/{job_id}/result",
    summary="Download final image",
    response_model=None,
    responses={
        200: {"content": {"image/jpeg": {}, "image/png": {}}},
        404: {"description": "Job not found"},
        409: {"description": "Job not yet completed"},
    },
)
async def get_job_result(
    job_id: str,
    store: Annotated[JobStore, Depends(get_store)],
) -> FileResponse | JSONResponse:
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}"
        )
    if record.status != JobStatus.completed or not record.result_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is {record.status.value}; result not available",
        )
    return FileResponse(
        path=record.result_path,
        filename=f"{job_id}.jpg",
        media_type="image/jpeg",
    )
