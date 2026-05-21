"""POST /api/v1/full-recovery-ceiling — Hybrid pipeline v0.3.0:

    1. Pre-process input (CLAHE auto + undistort optional)
    2. Ollama bds-brain CoT 2-step:
        Step A: shadow/đổ bóng/multi-tier/false-boundary analysis (text)
        Step B: 3-7 click points trên ceiling thực tế (JSON {points: [[x,y]...]})
    3. SAM 2 HIGH-RES config (points_per_side=64, iou=0.95, stability=0.96)
       predict_from_points (multi-point cùng lúc → 1 mask bám phào chỉ + tier)
    4. Co-segment WALL với cùng SAM session
    5. Overlap resolver — Sobel directional:
        vertical edge dominant → wall wins
        horizontal edge dominant → ceiling wins
       → triệt tiêu transom + leakage
    6. Export RGBA PNG: alpha = ceiling mask, các vùng khác transparent (0)
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
router = APIRouter(prefix="/api/v1/full-recovery-ceiling", tags=["recovery"])


@router.post(
    "",
    summary="VLM bds-brain CoT 2-step + SAM 2 high-grid + Sobel overlap resolver → ceiling RGBA PNG",
    response_model=None,
)
async def full_recovery_ceiling(
    file: UploadFile = File(...),
    vlm_model: str = Form("bds-brain", description="Ollama model (custom bds-brain hoặc qwen2.5vl:7b)"),
    sam_checkpoint: str = Form("", description="SAM 2 checkpoint .pt path (rỗng = ~/.cache/sam2/sam2_hiera_tiny.pt)"),
    target_class: str = Form("ceiling", description="ceiling | wall | floor | windows | doors"),
    apply_clahe: bool = Form(True, description="CLAHE preprocess cho ảnh khó (low contrast)"),
    apply_undistort: bool = Form(False, description="Undistort cho ảnh fisheye/ultra-wide"),
    apply_sobel_resolver: bool = Form(True, description="Sobel directional overlap resolver"),
    output_format: str = Form("png", description="'png' (RGBA file) hoặc 'json' (points + paths + reasoning)"),
    mock: bool = Form(False),
):
    if mock:
        return JSONResponse({
            "mode": "mock",
            "vlm_model": vlm_model,
            "vlm_reasoning": "Step 1 analysis: phòng khách modern, góc chụp eye-level, trần phẳng không phào, đèn chùm 3-bóng ở center. Shadow nhẹ ở góc trên trái. Không có ranh giới giả.",
            "points_detected": [[2310, 180], [1500, 220], [3200, 200], [800, 250], [3900, 230]],
            "sam_score": 0.94,
            "ceiling_cov_pct": 12.3,
            "overlap_resolved_pixels": 8421,
            "method": "vlm_sam2_cot_sobel_recovery",
            "mock": True,
        })

    if not file.filename:
        raise HTTPException(400, "Missing file")

    # ───────── Phase 0: Decode + preprocess ─────────
    raw = await file.read()
    nparr = np.frombuffer(raw, dtype=np.uint8)
    img_orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_orig is None:
        raise HTTPException(400, "Cannot decode image")
    h, w = img_orig.shape[:2]

    from pps_wincei_masks.preprocess import preprocess_for_vlm_sam
    img, prep_report = preprocess_for_vlm_sam(
        img_orig,
        auto_clahe=apply_clahe, auto_undistort=apply_undistort,
        force_clahe=apply_clahe, force_undistort=apply_undistort,
    )
    log.info("Preprocess: CLAHE=%s, undistort=%s", prep_report.clahe_applied, prep_report.undistort_applied)

    # ───────── Phase 1: VLM CoT 2-step ─────────
    try:
        from pps_wincei_masks.vlm_client import OllamaVLM, check_ollama_available
    except ImportError as exc:
        raise HTTPException(500, f"vlm_client module missing: {exc}")

    ok, available = check_ollama_available()
    if not ok:
        raise HTTPException(503, "Ollama không chạy. Khởi động: `ollama serve`")
    if not any(vlm_model in m for m in available):
        raise HTTPException(
            424,
            f"Model '{vlm_model}' chưa sẵn sàng. Có sẵn: {available}. "
            f"Run: ollama pull {vlm_model} hoặc bash scripts/setup_vlm_sam.sh",
        )

    vlm = OllamaVLM(
        model=vlm_model,
        endpoint="http://localhost:11434/api/chat",
        use_chat_api=True,
    )

    try:
        cot = vlm.query_chain_of_thought(img, max_side=1280, target_class=target_class)
    except RuntimeError as exc:
        raise HTTPException(503, f"VLM CoT fail: {exc}")

    ceiling_points = cot.parsed_points.get(target_class) or cot.parsed_points.get("ceiling")
    if not ceiling_points or not isinstance(ceiling_points[0], (list, tuple)):
        raise HTTPException(
            424,
            f"VLM không trả về points hợp lệ. Step 1 reasoning: {cot.reasoning[:300]} | "
            f"Step 2 raw: {cot.raw_text[:200]}"
        )

    log.info("VLM CoT: %d points for '%s' (reasoning %d chars, %dms)",
             len(ceiling_points), target_class, len(cot.reasoning), int(cot.elapsed_ms))

    # ───────── Phase 2: SAM 2 high-grid predict ─────────
    try:
        from pps_wincei_masks.sam_engine import SAMEngine
    except ImportError as exc:
        raise HTTPException(500, f"sam_engine module missing: {exc}")

    sam_ckpt = Path(sam_checkpoint) if sam_checkpoint else None
    try:
        sam = SAMEngine(checkpoint=sam_ckpt, high_res=True)
    except RuntimeError as exc:
        raise HTTPException(424, f"SAM init fail: {exc}")

    sam.set_image(img)
    pts = [(int(p[0]), int(p[1])) for p in ceiling_points]
    target_result = sam.predict_from_points(img, pts, multimask=True)
    target_mask = target_result.mask  # uint8 0/255
    target_cov = (target_mask > 128).mean() * 100

    # Co-segment WALL bằng heuristic 3-point sampling cho Sobel resolver
    wall_pts = [(w // 8, h // 2), (w * 7 // 8, h // 2), (w // 2, int(h * 0.6))]
    wall_result = sam.predict_from_points(img, wall_pts, multimask=True)
    wall_mask = wall_result.mask
    wall_cov = (wall_mask > 128).mean() * 100

    log.info("SAM2 high-grid: %s cov=%.2f%% (score %.2f), wall cov=%.2f%%",
             target_class, target_cov, target_result.score, wall_cov)

    # ───────── Phase 3: Sobel directional overlap resolver ─────────
    overlap_resolved_px = 0
    if apply_sobel_resolver and target_class == "ceiling":
        from pps_wincei_masks.overlap_resolver import resolve_ceiling_wall_overlap
        before = ((target_mask > 128) & (wall_mask > 128)).sum()
        target_mask, wall_mask = resolve_ceiling_wall_overlap(
            img, target_mask, wall_mask, direction_ratio=1.5,
        )
        after = ((target_mask > 128) & (wall_mask > 128)).sum()
        overlap_resolved_px = int(before - after)
        log.info("Sobel resolver: %d pixels reclassified (vertical→wall, horizontal→ceiling)",
                 overlap_resolved_px)

    final_cov = (target_mask > 128).mean() * 100

    # ───────── Phase 4: Export RGBA PNG ─────────
    output_dir = settings.outputs_dir / "ceiling_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename).stem

    # Trả về ảnh GỐC (img_orig, không preprocess) với alpha = ceiling mask
    bgra = cv2.cvtColor(img_orig, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = target_mask  # vùng != ceiling → alpha 0 → transparent

    png_path = output_dir / "ceiling_full_recovery.png"
    png_named = output_dir / f"{stem}_ceiling_full_recovery.png"
    cv2.imwrite(str(png_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(png_named), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # Mask grayscale standalone (cho Photoshop)
    mask_path = output_dir / f"{stem}_ceiling_mask.png"
    cv2.imwrite(str(mask_path), target_mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    report = {
        "source_file": file.filename,
        "image_size": {"width": w, "height": h},
        "target_class": target_class,
        "vlm_model": vlm_model,
        "vlm_reasoning": cot.reasoning,
        "vlm_step2_raw": cot.raw_text[:500],
        "vlm_elapsed_ms": round(cot.elapsed_ms, 0),
        "points_detected": pts,
        "n_points": len(pts),
        "sam_high_res_config": {
            "points_per_side": 64,
            "pred_iou_thresh": 0.95,
            "stability_score_thresh": 0.96,
        },
        "sam_score": round(target_result.score, 3),
        "wall_sam_score": round(wall_result.score, 3),
        "preprocess": {
            "clahe_applied": prep_report.clahe_applied,
            "undistort_applied": prep_report.undistort_applied,
        },
        "overlap_resolved_pixels": overlap_resolved_px,
        "ceiling_coverage_before_resolver_pct": round(target_cov, 2),
        "ceiling_coverage_after_resolver_pct": round(final_cov, 2),
        "method": "vlm_sam2_cot_sobel_recovery",
        "output_rgba_png": str(png_path),
        "output_named_png": str(png_named),
        "output_mask_png": str(mask_path),
    }

    if output_format == "json":
        return JSONResponse(report)

    # Default: PNG file response — caller download direct
    return FileResponse(
        path=png_path,
        media_type="image/png",
        filename="ceiling_full_recovery.png",
        headers={
            "X-Ceiling-Coverage-Pct": str(round(final_cov, 2)),
            "X-SAM-Score": str(round(target_result.score, 3)),
            "X-VLM-Points": str(len(pts)),
            "X-Overlap-Resolved-Px": str(overlap_resolved_px),
        },
    )
