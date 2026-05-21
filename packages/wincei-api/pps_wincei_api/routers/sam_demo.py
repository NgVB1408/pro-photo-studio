"""POST /api/v1/sam-demo — SAM AutomaticMaskGenerator panoptic visualization.

Theo notebook official:
   facebookresearch/segment-anything → automatic_mask_generator_example.ipynb
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sam-demo", tags=["sam"])


@router.post(
    "",
    summary="SAM AutomaticMaskGenerator → panoptic colored overlay (notebook official style)",
    response_model=None,
)
async def sam_demo(
    file: UploadFile = File(...),
    points_per_side: int = Form(32),
    alpha: float = Form(0.35),
    border_thickness: int = Form(0),
    sam_checkpoint: str = Form("", description="SAM 1 .pth path (rỗng = auto-detect)"),
    output_format: str = Form("jpg", description="'jpg' overlay or 'json' report"),
    mock: bool = Form(False),
):
    if mock:
        return JSONResponse({
            "mode": "mock",
            "n_masks": 47,
            "model_type": "vit_b",
            "points_per_side": points_per_side,
            "alpha": alpha,
            "output_overlay": "/mock/sam_colored.jpg",
        })

    if not file.filename:
        raise HTTPException(400, "Missing file")
    raw = await file.read()
    nparr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")

    try:
        from pps_wincei_masks.sam_demo import sam_demo_pipeline
    except ImportError as exc:
        raise HTTPException(500, f"sam_demo module missing: {exc}")

    try:
        result = sam_demo_pipeline(
            img,
            checkpoint=Path(sam_checkpoint) if sam_checkpoint else None,
            points_per_side=points_per_side,
            alpha=alpha,
            border_thickness=border_thickness,
        )
    except RuntimeError as exc:
        raise HTTPException(424, str(exc))

    # Save outputs
    output_dir = settings.outputs_dir / "sam_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename).stem
    overlay_jpg = output_dir / f"{stem}_sam_colored.jpg"
    rgba_png = output_dir / f"{stem}_sam_rgba.png"
    cv2.imwrite(str(overlay_jpg), result.overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(rgba_png), result.overlay_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    report = {
        "source_file": file.filename,
        "n_masks": result.n_masks,
        "model_type": result.model_type,
        "points_per_side": points_per_side,
        "alpha": alpha,
        "output_overlay_jpg": str(overlay_jpg),
        "output_rgba_png": str(rgba_png),
        "method": "sam_automatic_mask_generator_official",
    }

    if output_format == "json":
        return JSONResponse(report)

    return FileResponse(
        path=overlay_jpg,
        media_type="image/jpeg",
        filename=f"{stem}_sam_colored.jpg",
        headers={
            "X-N-Masks": str(result.n_masks),
            "X-Model-Type": result.model_type,
        },
    )
