"""POST /api/v1/hdr-fuse — Mertens bracket fusion."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ..config import settings
from ..jobs import registry
from ..schemas import JobType, ProcessResponse
from ..workers import run_hdr_fuse_job

router = APIRouter(prefix="/api/v1/hdr-fuse", tags=["hdr"])


@router.post("", response_model=ProcessResponse, summary="Mertens HDR fusion từ N ảnh bracket")
async def hdr_fuse(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="N ảnh bracket cùng cảnh (EXIF có ExposureBiasValue)"),
    align: bool = Form(True),
    contrast_weight: float = Form(1.0),
    saturation_weight: float = Form(1.0),
    exposure_weight: float = Form(1.0),
    gamma: float = Form(1.0),
    mock: bool = Form(False),
) -> ProcessResponse:
    if mock:
        return ProcessResponse(
            mode="sync", mock=True,
            eval={"fused": True, "groups": 1, "mock": True},
            output_url="/mock/hdr-output.jpg",
        )

    if not files:
        raise HTTPException(400, "Cần ≥2 ảnh bracket")

    upload_dir = settings.uploads_dir
    saved: list = []
    for f in files:
        if not f.filename:
            continue
        dst = upload_dir / f.filename
        dst.write_bytes(await f.read())
        saved.append(dst)

    job = registry.create(JobType.hdr_fuse, inputs=[str(p) for p in saved])
    params = dict(
        align=align,
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
        gamma=gamma,
    )
    background_tasks.add_task(run_hdr_fuse_job, job.job_id, saved, **params)
    return ProcessResponse(mode="async", job_id=job.job_id, job=job)
