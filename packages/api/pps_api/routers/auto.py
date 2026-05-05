"""``/v1/auto`` — one-shot autopilot endpoint.

The customer drops an image, the API picks the right stages, runs the full
multi-agent studio, and returns the rendered output + scorecard. No stage
toggles, no manual configuration. This is the friction-free path the web
portal defaults to.

Internally this just enqueues a job with ``stages=["auto_pilot"]`` so the
existing job-store + result-stream + webhook plumbing applies unchanged.
"""

from __future__ import annotations

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
    HTTPException,
    UploadFile,
    status,
)
from pps_core.types import Job

from pps_api.routers.jobs import _to_job_out, get_output_dir, get_store
from pps_api.schemas import JobOut, JobStatus
from pps_api.security import maybe_require_api_key
from pps_api.services import JobRecord, JobStore, run_pipeline_for_job

router = APIRouter(
    prefix="/v1/auto",
    tags=["auto"],
    dependencies=[Depends(maybe_require_api_key)],
)


@router.post(
    "",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="One-shot autopilot — classify, run, audit",
    description=(
        "Drop an image, get back a job ID. The pipeline auto-detects scene, "
        "runs the baseline enhancement, then routes the rendered output through "
        "the multi-agent studio (Vertical, Exposure, White Balance, Noise, Sky, "
        "Sharpness, Halo, Colour, Composition). The resulting StudioReport is "
        "attached to the stage report's `artifacts` map."
    ),
)
async def auto_enhance_endpoint(
    bg: BackgroundTasks,
    image: Annotated[UploadFile, File(description="Image (JPEG / PNG / WebP)")],
    store: Annotated[JobStore, Depends(get_store)],
    output_dir: Annotated[Path, Depends(get_output_dir)],
    scene: str | None = None,
    seed: int = 42,
    twilight: bool = False,
) -> JobOut:
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
    if scene is not None and scene not in ("interior", "exterior", "aerial"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scene must be 'interior', 'exterior', or 'aerial'",
        )

    job_id = uuid.uuid4().hex
    params: dict[str, object] = {"twilight": twilight}
    if scene is not None:
        params["scene"] = scene
    job = Job(
        job_id=job_id,
        stages=("auto_pilot",),
        params=params,
        seed=seed,
        metadata={"source": "v1-auto", "filename": image.filename or "upload.jpg"},
    )
    record = JobRecord(
        job_id=job_id,
        status=JobStatus.queued,
        metadata=dict(job.metadata),
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


__all__ = ["router"]
