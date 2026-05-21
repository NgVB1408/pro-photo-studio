"""POST /api/v1/perfect-window — Cô lập cửa sổ, output PNG RGBA chỉ giữ window.

Pipeline v0.3.4:
    1. (Optional) VLM bds-brain → window bbox [ymin,xmin,ymax,xmax]
    2. Semantic fallback bbox(window∪door∪sky) nếu không có VLM
    3. SAM 2 predict_from_box (KHÔNG auto-scan toàn ảnh)
    4. Canny edge inside bbox → hard boundary chống tràn rèm/thạch cao
    5. Output PNG RGBA — chỉ vùng kính + khung, mọi thứ khác transparent

KHÔNG xử lý: ceiling, crown, baseboard, wall, floor (tạm dừng theo user request).
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
router = APIRouter(prefix="/api/v1/perfect-window", tags=["window"])


def _try_vlm_bbox(
    image_bgr: np.ndarray, vlm_model: str = "bds-brain",
) -> tuple[int, int, int, int] | None:
    """Try VLM để lấy window bbox. Return None nếu không khả dụng."""
    try:
        from pps_wincei_masks.vlm_client import OllamaVLM, check_ollama_available
    except ImportError:
        return None

    ok, models = check_ollama_available()
    if not ok or not any(vlm_model in m for m in models):
        log.info("VLM '%s' không khả dụng → semantic fallback bbox", vlm_model)
        return None

    vlm = OllamaVLM(
        model=vlm_model,
        endpoint="http://localhost:11434/api/chat",
        use_chat_api=True,
    )
    prompt = (
        "Quét ảnh nội thất này. Trả về DUY NHẤT bounding box ÔM SÁT vùng "
        "cửa sổ/cửa kính. Format JSON: "
        '{"window_bbox": [ymin, xmin, ymax, xmax]} '
        "với tọa độ pixel ảnh gốc. KHÔNG giải thích."
    )
    try:
        resp = vlm.query(image_bgr, prompt=prompt, max_side=1280)
    except RuntimeError as exc:
        log.warning("VLM bbox query fail: %s", exc)
        return None

    pts = resp.parsed_points
    if "window_bbox" in pts:
        bb = pts["window_bbox"]
        if isinstance(bb, list) and len(bb) >= 4:
            return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
    # Fallback: derive bbox từ window point cluster
    if "windows" in pts and pts["windows"]:
        ws = pts["windows"]
        xs = [p[0] for p in ws if len(p) >= 2]
        ys = [p[1] for p in ws if len(p) >= 2]
        if xs and ys:
            return (min(ys) - 50, min(xs) - 50, max(ys) + 50, max(xs) + 50)
    return None


@router.post(
    "",
    summary="VLM bbox + SAM 2 box prompt + Canny hard boundary → PNG RGBA chỉ window",
    response_model=None,
)
async def perfect_window(
    file: UploadFile = File(...),
    use_vlm: bool = Form(True, description="Thử Ollama bds-brain → window bbox"),
    vlm_model: str = Form("bds-brain"),
    sam_checkpoint: str = Form("", description="SAM 2 checkpoint .pt path"),
    bbox_padding_pct: float = Form(0.03, description="Padding ngoài bbox (% width)"),
    canny_low: int = Form(50),
    canny_high: int = Form(150),
    feather_px: int = Form(4, description="Gaussian feather alpha biên (px)"),
    output_format: str = Form("png", description="'png' file hoặc 'json' (bbox+metadata)"),
    mock: bool = Form(False),
):
    if mock:
        return JSONResponse({
            "mode": "mock",
            "bbox_ymin_xmin_ymax_xmax": [420, 1500, 2200, 3200],
            "method": "vlm_bbox",
            "sam_score": 0.94,
            "n_canny_edges": 18420,
            "window_coverage_pct": 12.5,
            "output_png": "/mock/window_opening.png",
            "mock": True,
        })

    if not file.filename:
        raise HTTPException(400, "Missing file")

    raw = await file.read()
    nparr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    h, w = img.shape[:2]

    # ━━ Step 1: VLM bbox (optional) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    vlm_bbox = None
    vlm_used = False
    if use_vlm:
        vlm_bbox = _try_vlm_bbox(img, vlm_model=vlm_model)
        vlm_used = vlm_bbox is not None

    # ━━ Step 2: Semantic fallback nếu cần ━━━━━━━━━━━━━━━━━━━━━━━━━━
    sem_result = None
    if vlm_bbox is None:
        from pps_wincei_masks.semantic import SemanticSegmenter
        # Load segmenter (cached static instance)
        from ..workers import _get_segmenter
        seg = _get_segmenter()
        sem_result = seg.segment(img)

    # ━━ Step 3: SAM 2 box prompt ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sam_engine = None
    try:
        from pps_wincei_masks.sam_engine import SAMEngine
        sam_ckpt = Path(sam_checkpoint) if sam_checkpoint else None
        sam_engine = SAMEngine(checkpoint=sam_ckpt)
    except Exception as exc:
        log.warning("SAM init fail: %s → semantic-only path", exc)

    # ━━ Step 4: Extract perfect window ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from pps_wincei_masks.perfect_window import extract_perfect_window
    try:
        result = extract_perfect_window(
            img,
            vlm_bbox=vlm_bbox,
            sem_result=sem_result,
            sam_engine=sam_engine,
            bbox_padding_pct=bbox_padding_pct,
            canny_low=canny_low,
            canny_high=canny_high,
            feather_px=feather_px,
        )
    except Exception as exc:
        raise HTTPException(500, f"Extract fail: {exc}")

    # ━━ Step 5: Output PNG RGBA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    output_dir = settings.outputs_dir / "perfect_window"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename).stem

    png_root = output_dir / "window_opening.png"
    png_named = output_dir / f"{stem}_window_opening.png"
    mask_png = output_dir / f"{stem}_window_mask.png"

    cv2.imwrite(str(png_root), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(png_named), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(mask_png), result.window_mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    coverage_pct = (result.window_mask > 128).mean() * 100

    report = {
        "source_file": file.filename,
        "image_size": {"width": w, "height": h},
        "method": result.method,
        "vlm_used": vlm_used,
        "vlm_model": vlm_model if vlm_used else None,
        "bbox_ymin_xmin_ymax_xmax": list(result.bbox),
        "sam_score": round(result.sam_score, 3),
        "n_canny_edges": result.n_canny_edges,
        "window_coverage_pct": round(coverage_pct, 2),
        "output_png": str(png_root),
        "output_named_png": str(png_named),
        "output_mask_png": str(mask_png),
    }

    if output_format == "json":
        return JSONResponse(report)

    return FileResponse(
        path=png_named,
        media_type="image/png",
        filename=f"{stem}_window_opening.png",
        headers={
            "X-Method": result.method,
            "X-SAM-Score": str(round(result.sam_score, 3)),
            "X-Window-Coverage-Pct": str(round(coverage_pct, 2)),
            "X-BBox": ",".join(str(x) for x in result.bbox),
        },
    )
