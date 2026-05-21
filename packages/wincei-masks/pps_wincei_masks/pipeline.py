"""Orchestrator — 1 ảnh in, 1 folder masks out.

Phases:
    0. Load image (BGR)
    1. SegFormer semantic seg → 7 soft prob maps
    2. Build base masks (wall/floor/ceiling/window/door/light)
    3. (Optional) PyMatting refine biên
    4. Phào chỉ heuristic (crown/baseboard/casing)
    5. Export PNG + overlay + TIFF + optional PSD
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .ceiling_boost import boost_ceiling_mask
from .crf import crf_refine
from .evaluator import EvalReport, evaluate_masks
from .exporters import ExportResult, export_all_masks
from .molding import detect_moldings
from .overlap_resolver import resolve_all_overlaps
from .precision import refine_alpha_masks_precision
from .preprocess import preprocess_for_vlm_sam
from .refine import refine_alpha_masks
from .semantic import ADE20K_CLASSES, SemanticSegmenter
from .tta import multiscale_segment

log = logging.getLogger(__name__)

DEFAULT_CLASSES_TO_EXPORT = ["wall", "floor", "ceiling", "window", "door"]


@dataclass
class MaskExtractionResult:
    image_path: Path
    out_dir: Path
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    export: ExportResult | None = None
    evaluation: EvalReport | None = None
    timings: dict[str, float] = field(default_factory=dict)
    model_name: str = ""

    def report(self) -> str:
        lines = [
            f"📷 Input  : {self.image_path}",
            f"📁 Output : {self.out_dir}",
            f"🧠 Model  : {self.model_name}",
        ]
        for k, t in self.timings.items():
            lines.append(f"   ⏱️  {k:18s}  {t*1000:7.0f} ms")
        lines.append("─" * 40)
        if self.evaluation is not None:
            for k, m in self.masks.items():
                cov = (m > 128).mean() * 100
                ms = self.evaluation.per_mask.get(k)
                if ms:
                    lines.append(f"   {k:12s}  cov={cov:5.1f}%  score={ms.overall:.2f}  [{ms.verdict}]")
                else:
                    lines.append(f"   {k:12s}  cov={cov:5.1f}%")
            lines.append("─" * 40)
            lines.append(f"🤖 SELF-EVAL: {self.evaluation.verdict.upper()}  overall={self.evaluation.overall_score:.3f}")
            for rec in self.evaluation.recommendations[:8]:
                lines.append(f"   • {rec}")
        else:
            for k, m in self.masks.items():
                cov = (m > 128).mean() * 100
                lines.append(f"   {k:12s}  cov={cov:5.1f}%")
        return "\n".join(lines)


def extract_masks(
    image_path: Path | str,
    out_root: Path | str,
    *,
    segmenter: SemanticSegmenter | None = None,
    refine_edges: bool = True,
    detect_molding: bool = True,
    include_lights: bool = False,
    matting_max_side: int = 1600,
    write_overlay: bool = True,
    write_tiff: bool = True,
    write_psd: bool = False,
    self_evaluate: bool = True,
    save_qc_json: bool = True,
    precision_mode: bool = False,
    tta_scales: tuple[float, ...] = (0.75, 1.0, 1.5),
    use_crf: bool = True,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    polish_strength: float = 1.0,
    preprocess_clahe: bool = False,
    preprocess_undistort: bool = False,
    resolve_overlap: bool = True,
) -> MaskExtractionResult:
    """Extract masks from 1 ảnh → folder.

    Args:
        image_path: input JPG/PNG.
        out_root: parent folder cho masks/.
        segmenter: pre-loaded SemanticSegmenter (reuse cho batch).
        refine_edges: bật PyMatting (slower nhưng biên đẹp).
        detect_molding: detect phào chỉ 3 loại.
        include_lights: thêm lamp + light vào exports.
        matting_max_side: downscale cap cho PyMatting (RAM safety).
        write_overlay/tiff/psd: chọn output formats.

    Returns:
        MaskExtractionResult.
    """
    image_path = Path(image_path)
    out_root = Path(out_root)
    stem = image_path.stem

    img_orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_orig is None:
        raise IOError(f"Không đọc được: {image_path}")

    result = MaskExtractionResult(image_path=image_path, out_dir=out_root / stem)
    t0 = time.perf_counter()

    # Phase -1: Preprocess (CLAHE / undistort) — chỉ cho VLM/SAM input,
    # KHÔNG chỉnh sửa ảnh export cuối
    img = img_orig
    if preprocess_clahe or preprocess_undistort:
        img, prep_report = preprocess_for_vlm_sam(
            img_orig,
            auto_clahe=preprocess_clahe,
            auto_undistort=preprocess_undistort,
            force_clahe=preprocess_clahe,
            force_undistort=preprocess_undistort,
        )
        log.info("Preprocess: CLAHE=%s, undistort=%s, notes=%s",
                 prep_report.clahe_applied, prep_report.undistort_applied, prep_report.notes)

    if segmenter is None:
        segmenter = SemanticSegmenter()
    result.model_name = segmenter.model_name
    t1 = time.perf_counter()
    result.timings["load_segmenter"] = t1 - t0

    # Semantic inference (single or multi-scale TTA)
    if precision_mode:
        # flip = chỉ bật khi user thực sự muốn TTA (scales != (1.0,))
        use_flip = tta_scales != (1.0,)
        sem = multiscale_segment(segmenter, img, scales=tta_scales, flip=use_flip)
        result.model_name = sem.model_name
    else:
        sem = segmenter.segment(img)
    t2 = time.perf_counter()
    result.timings["semantic_inference"] = t2 - t1

    # CRF post-processing (refine biên trên softmax)
    if precision_mode and use_crf:
        try:
            refined_probs = crf_refine(img, sem.probs)
            sem.probs = refined_probs
            log.info("CRF refinement applied")
        except Exception as exc:
            log.warning("CRF skipped: %s", exc)
    t_crf = time.perf_counter()
    result.timings["crf_refine"] = t_crf - t2

    # Build soft masks
    soft_masks: dict[str, np.ndarray] = {}
    for name in DEFAULT_CLASSES_TO_EXPORT:
        ade_id = ADE20K_CLASSES[name]
        soft_masks[name] = sem.get_soft(ade_id)

    # Union window + sky (sky qua kính thường được model nhận diện)
    soft_masks["window"] = np.maximum(soft_masks["window"], sem.get_soft(ADE20K_CLASSES["sky"]))

    # 'opening' = anywhere có view ra ngoài: window ∪ door ∪ sky
    # Trường hợp double glass door (ảnh ngoài villa) thì ADE classify thành door
    # → nhân viên dùng 'opening' mask để chỉnh outdoor view 1 phát.
    soft_masks["opening"] = np.maximum.reduce([
        soft_masks["window"],
        soft_masks["door"],
        sem.get_soft(ADE20K_CLASSES["sky"]),
    ])

    # Lamp + light fixtures — used cho ceiling boost (anchor) + optional export
    lamp_soft = sem.get_soft(ADE20K_CLASSES["lamp"])
    light_soft = sem.get_soft(ADE20K_CLASSES["light"])
    if include_lights:
        soft_masks["light"] = np.maximum(lamp_soft, light_soft)

    # Refine biên
    if precision_mode:
        alpha_masks = refine_alpha_masks_precision(
            img, soft_masks,
            tile_size=tile_size, overlap=tile_overlap,
            unknown_band=12, polish_strength=polish_strength,
        )
    elif refine_edges:
        alpha_masks = refine_alpha_masks(
            img, soft_masks, use_matting=True, matting_max_side=matting_max_side
        )
    else:
        alpha_masks = {k: (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8) for k, v in soft_masks.items()}
    t3 = time.perf_counter()
    result.timings["edge_refine"] = t3 - t_crf

    # Ceiling boost — lamp anchor flood fill (modern interior)
    try:
        lamp_bin = (lamp_soft >= 0.4).astype(np.uint8) * 255
        light_bin = (light_soft >= 0.4).astype(np.uint8) * 255
        boosted_ceiling, boost_info = boost_ceiling_mask(
            img,
            alpha_masks["ceiling"],
            lamp_mask=lamp_bin,
            light_mask=light_bin,
            wall_mask=alpha_masks.get("wall"),
            floor_mask=alpha_masks.get("floor"),
            door_mask=alpha_masks.get("door"),
            window_mask=alpha_masks.get("window"),
            opening_mask=alpha_masks.get("opening"),
        )
        if boost_info.get("boosted"):
            alpha_masks["ceiling"] = boosted_ceiling
            log.info("Ceiling boost applied: %s, cov %.2f%% → %.2f%%",
                     boost_info["method"],
                     boost_info["original_cov_pct"],
                     boost_info["boosted_cov_pct"])
    except Exception as exc:
        log.warning("Ceiling boost fail (non-fatal): %s", exc)

    # Phào chỉ
    if detect_molding:
        mold = detect_moldings(
            img,
            wall_mask=alpha_masks["wall"],
            ceiling_mask=alpha_masks["ceiling"],
            floor_mask=alpha_masks["floor"],
            window_mask=alpha_masks["window"],
            door_mask=alpha_masks["door"],
        )
        alpha_masks["crown"] = mold.crown
        alpha_masks["baseboard"] = mold.baseboard
        alpha_masks["casing"] = mold.casing
    t4 = time.perf_counter()
    result.timings["molding_heuristic"] = t4 - t3

    # Phase 4.5: Overlap leakage resolver v0.3.2 — Sobel + 4 nguyên tắc
    if resolve_overlap:
        try:
            alpha_masks = resolve_all_overlaps(
                img, alpha_masks,
                ade_argmax_id=sem.argmax_id,  # ← furniture exclusion mask
            )
            log.info("v0.3.2 resolver: Sobel + exclusion + dynamic close + distance constrain")
        except Exception as exc:
            log.warning("Overlap resolver fail (non-fatal): %s", exc)

    result.masks = alpha_masks

    # Export
    export = export_all_masks(
        img,
        alpha_masks,
        out_root=out_root,
        stem=stem,
        write_overlay=write_overlay,
        write_tiff=write_tiff,
        write_psd=write_psd,
    )
    result.export = export
    t5 = time.perf_counter()
    result.timings["export"] = t5 - t4

    # Phase 5: AI Self-Evaluation (Supervisor)
    if self_evaluate:
        eval_report = evaluate_masks(img, alpha_masks)
        result.evaluation = eval_report
        if save_qc_json and export is not None:
            import json
            qc_path = export.out_dir / f"{stem}_qc_report.json"
            qc_path.write_text(
                json.dumps(eval_report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    t6 = time.perf_counter()
    result.timings["self_eval"] = t6 - t5
    result.timings["total"] = t6 - t0

    return result
