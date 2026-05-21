"""POST /api/v1/detect-regions — Real Estate Vision Analyzer.

Trả JSON bounding boxes normalized [0..1000] cho:
    - ceiling (1 bbox)
    - windows (N bbox per pane, type classification)
    - walls / floor / doors (optional)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..workers import _get_segmenter

router = APIRouter(prefix="/api/v1/detect-regions", tags=["regions"])


@router.post("", summary="Phân tích ảnh BĐS → JSON bounding boxes (ceiling + windows + walls/floor/doors)")
async def detect_regions(
    file: UploadFile = File(..., description="Ảnh BĐS (JPG/PNG)"),
    include_walls: bool = Form(True),
    include_floor: bool = Form(True),
    include_doors: bool = Form(True),
    mock: bool = Form(False),
) -> dict:
    if mock:
        return {
            "project_type": "Real Estate Segmentation",
            "image_size": {"width": 4608, "height": 3072},
            "camera_angle": "wide_angle",
            "detected_elements": {
                "ceiling": {
                    "coordinates": [0, 0, 280, 1000],
                    "confidence": 0.94,
                    "has_crown_molding": False,
                    "area_pct": 12.5,
                },
                "windows": [
                    {"id": 1, "coordinates": [180, 410, 720, 590], "type": "Casement Window", "confidence": 0.91, "area_pct": 4.3},
                ],
                "walls": {
                    "coordinates": [50, 0, 820, 1000],
                    "confidence": 0.96,
                    "area_pct": 51.7,
                },
                "floor": {
                    "coordinates": [780, 0, 1000, 1000],
                    "confidence": 0.95,
                    "area_pct": 8.6,
                },
                "doors": [
                    {"id": 1, "coordinates": [180, 410, 760, 590], "type": "Glass Door", "confidence": 0.89, "area_pct": 11.4},
                ],
            },
            "reasoning_steps": [
                "Step 1: SegFormer ADE20K inference → 9 class soft prob",
                "Step 2: camera_angle = wide_angle",
                "Step 3: no curtain detected",
                "Step 4: crown_molding = False",
                "Step 5: window panes detected = 1",
            ],
            "mock": True,
        }

    if not file.filename:
        raise HTTPException(400, "Missing file")

    from pps_wincei_masks.regions_json import analyze_image_to_json
    import cv2
    import numpy as np

    # Read upload into memory (no need to persist)
    raw = await file.read()
    nparr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không decode được ảnh")

    seg = _get_segmenter()
    result = analyze_image_to_json(
        img,
        segmenter=seg,
        include_walls=include_walls,
        include_floor=include_floor,
        include_doors=include_doors,
    )
    result["source_file"] = file.filename
    return result
