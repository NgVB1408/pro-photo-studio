"""POST /api/v1/segment-masks — phân vùng + phào chỉ + AI eval."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ..config import settings
from ..jobs import registry
from ..schemas import JobType, ProcessResponse
from ..workers import run_segment_masks_job

router = APIRouter(prefix="/api/v1/segment-masks", tags=["segment"])


@router.post("", response_model=ProcessResponse, summary="Phân vùng wall/floor/ceiling/window/door/molding + AI eval")
async def segment_masks(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="1+ ảnh BĐS"),
    refine_edges: bool = Form(True, description="PyMatting closed-form refinement"),
    detect_molding: bool = Form(True, description="Phát hiện phào trần / chân tường / nẹp cửa"),
    include_lights: bool = Form(False),
    write_overlay: bool = Form(True),
    write_tiff: bool = Form(True),
    write_psd: bool = Form(False),
    mock: bool = Form(False),
) -> ProcessResponse:
    if mock:
        return ProcessResponse(
            mode="sync", mock=True,
            eval={
                "verdict": "pass",
                "overall_score": 0.88,
                "per_mask": {
                    "wall": {"coverage": 0.517, "verdict": "pass"},
                    "floor": {"coverage": 0.086, "verdict": "pass"},
                    "ceiling": {"coverage": 0.014, "verdict": "pass"},
                    "window": {"coverage": 0.0, "verdict": "no_target"},
                    "door": {"coverage": 0.114, "verdict": "pass"},
                    "opening": {"coverage": 0.124, "verdict": "pass"},
                    "crown": {"coverage": 0.0, "verdict": "no_target"},
                    "baseboard": {"coverage": 0.0, "verdict": "no_target"},
                    "casing": {"coverage": 0.013, "verdict": "pass"},
                },
                "mock": True,
            },
            output_url="/mock/masks-output.zip",
        )

    if not files:
        raise HTTPException(400, "Cần ≥1 ảnh")

    upload_dir = settings.uploads_dir
    saved: list = []
    for f in files:
        if not f.filename:
            continue
        dst = upload_dir / f.filename
        dst.write_bytes(await f.read())
        saved.append(dst)

    job = registry.create(JobType.segment_masks, inputs=[str(p) for p in saved])
    params = dict(
        refine_edges=refine_edges,
        detect_molding=detect_molding,
        include_lights=include_lights,
        write_overlay=write_overlay,
        write_tiff=write_tiff,
        write_psd=write_psd,
    )
    background_tasks.add_task(run_segment_masks_job, job.job_id, saved, **params)
    return ProcessResponse(mode="async", job_id=job.job_id, job=job)
