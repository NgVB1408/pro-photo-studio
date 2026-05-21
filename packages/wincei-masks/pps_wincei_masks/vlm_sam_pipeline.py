"""VLM + SAM2 pipeline — chuyên gia kiến trúc + cắt mượt sub-pixel.

Flow:
    1. Ollama VLM (Qwen2.5-VL / Llama 3.2 Vision) đọc ảnh → JSON click points
    2. Per class: SAM 2 (or SAM 1.0) nhận point → "loang mực" mask sub-pixel
    3. Compose dict[name -> mask uint8 255]
    4. Optional: phào chỉ heuristic + AI eval
    5. Export PNG/TIFF/PSD same như semantic pipeline

Strategy mapping VLM keys → mask names:
    ceiling          → ceiling
    walls            → wall
    floor            → floor
    windows          → window (multi-point single mask, all panes merged)
    doors            → door
    crown_molding    → crown
    baseboard        → baseboard
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .sam_engine import SAMEngine
from .vlm_client import OllamaVLM, VLMResponse

log = logging.getLogger(__name__)


VLM_TO_MASK_NAME = {
    "ceiling": "ceiling",
    "walls": "wall",
    "wall": "wall",
    "floor": "floor",
    "windows": "window",
    "window": "window",
    "doors": "door",
    "door": "door",
    "crown_molding": "crown",
    "baseboard": "baseboard",
}

# Multi-INSTANCE classes: mỗi point = 1 instance riêng, union để export 1 mask.
# Vd: 4 cửa sổ → 4 masks độc lập → union → window.png chứa tất cả 4.
MULTI_INSTANCE = {"window", "door", "crown", "baseboard"}

# Multi-POINT-PER-REGION: nhiều points GUIDING cùng 1 region (large surface area).
# Vd: ceiling 3-5 điểm rải ngang → 1 mask trần phủ toàn bộ.
# Khác với MULTI_INSTANCE: không union từng mask, mà gửi tất cả points làm prompt cùng lúc.
MULTI_POINT_PER_REGION = {"ceiling", "wall", "floor"}


@dataclass
class VLMSAMResult:
    image_path: Path
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    vlm_response: VLMResponse | None = None
    sam_scores: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def extract_masks_vlm_sam(
    image_path: Path | str,
    *,
    vlm: OllamaVLM | None = None,
    sam: SAMEngine | None = None,
    vlm_model: str = "qwen2.5vl:7b",
    sam_checkpoint: str | Path | None = None,
    min_sam_score: float = 0.5,
) -> VLMSAMResult:
    """Run VLM → click points → SAM masks.

    Args:
        image_path: input ảnh.
        vlm: pre-loaded OllamaVLM (None = tạo mới).
        sam: pre-loaded SAMEngine (None = tạo mới).
        vlm_model: Ollama model tag.
        sam_checkpoint: SAM checkpoint path.
        min_sam_score: bỏ qua mask SAM nếu score thấp.

    Returns:
        VLMSAMResult với masks + vlm response + scores.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Không đọc được: {image_path}")

    result = VLMSAMResult(image_path=image_path)
    t0 = time.perf_counter()

    if vlm is None:
        vlm = OllamaVLM(model=vlm_model)
    if sam is None:
        sam = SAMEngine(checkpoint=sam_checkpoint)

    # Phase 1: VLM query
    vlm_resp = vlm.query(img)
    result.vlm_response = vlm_resp
    log.info("VLM (%s) trả %d keys: %s", vlm.model, len(vlm_resp.parsed_points), list(vlm_resp.parsed_points))
    t1 = time.perf_counter()
    result.timings["vlm_query"] = t1 - t0

    # Phase 2: SAM mask per click point
    sam.set_image(img)
    h, w = img.shape[:2]
    masks: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}

    for vlm_key, points in vlm_resp.parsed_points.items():
        mask_name = VLM_TO_MASK_NAME.get(vlm_key)
        if mask_name is None:
            continue

        # Normalize: nếu chỉ 1 point đơn lẻ [x, y] → wrap thành [[x, y]]
        if isinstance(points[0], (int, float)) and len(points) == 2:
            points = [points]

        # Validate point list
        clean_pts = [(int(p[0]), int(p[1])) for p in points
                     if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not clean_pts:
            continue

        log.info("SAM '%s' với %d điểm: %s", mask_name, len(clean_pts), clean_pts[:5])

        if mask_name in MULTI_INSTANCE:
            # Mỗi point = 1 instance độc lập → union
            combined = np.zeros((h, w), dtype=np.uint8)
            best_score = 0.0
            for pt in clean_pts:
                r = sam.predict_from_point(img, pt)
                if r.score >= min_sam_score:
                    combined = np.maximum(combined, r.mask)
                    best_score = max(best_score, r.score)
            if combined.sum() > 0:
                masks[mask_name] = combined
                scores[mask_name] = best_score
        elif mask_name in MULTI_POINT_PER_REGION:
            # Tất cả points là PROMPT cho CÙNG 1 region lớn (ceiling/wall/floor)
            # SAM 2 "loang mực" phủ toàn bộ region qua multi-point guidance
            r = sam.predict_from_points(img, clean_pts, multimask=True)
            if r.score >= min_sam_score:
                masks[mask_name] = r.mask
                scores[mask_name] = r.score
            else:
                # Score thấp → thử lại từng point riêng + union (fallback)
                log.info("'%s' multi-point score thấp (%.2f), fallback per-point union",
                         mask_name, r.score)
                combined = np.zeros((h, w), dtype=np.uint8)
                best_score = 0.0
                for pt in clean_pts:
                    rr = sam.predict_from_point(img, pt)
                    if rr.score >= min_sam_score:
                        combined = np.maximum(combined, rr.mask)
                        best_score = max(best_score, rr.score)
                if combined.sum() > 0:
                    masks[mask_name] = combined
                    scores[mask_name] = best_score
        else:
            # Default: single point or first point
            r = sam.predict_from_point(img, clean_pts[0])
            if r.score >= min_sam_score:
                masks[mask_name] = r.mask
                scores[mask_name] = r.score

    t2 = time.perf_counter()
    result.timings["sam_segmentation"] = t2 - t1
    result.timings["total"] = t2 - t0
    result.masks = masks
    result.sam_scores = scores
    return result
