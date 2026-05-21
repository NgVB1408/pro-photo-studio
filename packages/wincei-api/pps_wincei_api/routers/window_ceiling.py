"""POST /api/v1/window-ceiling — fix cửa sổ + trần."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ..config import settings
from ..jobs import registry
from ..schemas import JobType, ProcessResponse
from ..workers import run_window_ceiling_job

router = APIRouter(prefix="/api/v1/window-ceiling", tags=["wincei"])


@router.post("", response_model=ProcessResponse, summary="Fix cửa sổ blown + trần ám màu")
async def window_ceiling(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Ảnh BĐS (JPG/PNG)"),
    window_strength: Optional[float] = Form(None, ge=0, le=2),
    ceiling_strength: Optional[float] = Form(None, ge=0, le=1),
    include_lamps: Optional[bool] = Form(None),
    self_evaluate: bool = Form(True),
    mode: str = Form("async", description="'sync' (wait) or 'async' (job_id)"),
    mock: bool = Form(False),
) -> ProcessResponse:
    if mock:
        return ProcessResponse(
            mode="sync", mock=True,
            eval={"verdict": "pass", "overall_score": 0.92, "mock": True},
            output_url="/mock/window-ceiling-output.jpg",
        )

    if not file.filename:
        raise HTTPException(400, "Missing file")

    # Save uploaded file
    upload_dir = settings.uploads_dir
    src = upload_dir / file.filename
    src.write_bytes(await file.read())

    job = registry.create(JobType.window_ceiling, inputs=[str(src)])

    params = dict(
        window_strength=window_strength,
        ceiling_strength=ceiling_strength,
        include_lamps=include_lamps,
        self_evaluate=self_evaluate,
    )

    if mode == "sync":
        # Blocking sync — đợi xong rồi trả response
        await asyncio.get_event_loop().run_in_executor(
            None, run_window_ceiling_job, job.job_id, [src], *(), **params
        )
        updated = registry.get(job.job_id)
        return ProcessResponse(
            mode="sync", job_id=job.job_id, job=updated,
            output_url=updated.download_url if updated else None,
        )

    background_tasks.add_task(run_window_ceiling_job, job.job_id, [src], **params)
    return ProcessResponse(mode="async", job_id=job.job_id, job=job)
