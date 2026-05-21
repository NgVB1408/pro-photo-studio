"""SAM Automatic Mask Generator demo visualizer.

Theo notebook chính thức của facebookresearch/segment-anything:
    notebooks/automatic_mask_generator_example.ipynb

Pipeline:
    1. SAM (vit_b/vit_l/vit_h) auto generate all masks toàn ảnh
    2. Sort by area descending
    3. Random color per mask + alpha 0.5
    4. Composite lên ảnh gốc → output panoptic-style visualization

Mục đích:
    - Demo rõ AI đang dùng SAM (đúng style notebook official)
    - Báo cáo khách: "Đây là kết quả SAM automatic — mỗi vùng = 1 instance"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SAMAutoResult:
    n_masks: int
    overlay_bgr: np.ndarray       # (H, W, 3) uint8 — original blended với colored masks
    overlay_rgba: np.ndarray      # (H, W, 4) uint8 BGRA — chỉ masks colored, ảnh gốc transparent
    masks_raw: list[dict]         # raw masks từ SAM (segmentation, area, bbox, ...)
    model_type: str               # 'vit_b' | 'vit_l' | 'vit_h'


def _rand_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Random BGR màu sáng vibrant (HSV space để tránh xám)."""
    h_deg = float(rng.uniform(0, 180))
    s = float(rng.uniform(160, 240))
    v = float(rng.uniform(180, 250))
    hsv = np.uint8([[[h_deg, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def visualize_sam_masks(
    image_bgr: np.ndarray,
    masks: list[dict],
    *,
    alpha: float = 0.35,
    seed: int = 42,
    min_area_pct: float = 0.0005,
    border_thickness: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose SAM masks lên ảnh — đúng style demo notebook chính thức.

    Args:
        image_bgr: ảnh gốc full-res.
        masks: list dict từ SamAutomaticMaskGenerator.generate().
        alpha: opacity màu mask (0.5 = demo default).
        seed: random color reproducibility.
        min_area_pct: bỏ qua mask quá nhỏ (default 0.05% image area).
        border_thickness: vẽ contour viền cho mỗi mask (0 = tắt).

    Returns:
        (overlay_bgr, overlay_rgba): blended ảnh + RGBA-only-masks
    """
    if not masks:
        h, w = image_bgr.shape[:2]
        return image_bgr.copy(), np.zeros((h, w, 4), dtype=np.uint8)

    h, w = image_bgr.shape[:2]
    total_area = h * w
    rng = np.random.default_rng(seed)

    # Sort descending area (lớn → nhỏ) để mask nhỏ vẽ ĐÈ lên mask lớn
    sorted_masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)

    # 2 outputs:
    # (a) overlay_bgr = ảnh gốc blend với màu mask
    # (b) overlay_rgba = chỉ màu masks, background transparent
    overlay_bgr = image_bgr.astype(np.float32).copy()
    rgba_layer = np.zeros((h, w, 4), dtype=np.float32)

    n_drawn = 0
    for ann in sorted_masks:
        seg = ann.get("segmentation")
        area = ann.get("area", 0)
        if seg is None or area < total_area * min_area_pct:
            continue
        # Boolean mask (H, W)
        if seg.dtype != bool:
            seg = seg.astype(bool)
        color_bgr = _rand_color(rng)

        # Blend overlay_bgr — chỉ pixel trong mask
        overlay_bgr[seg] = (
            overlay_bgr[seg] * (1 - alpha)
            + np.array(color_bgr, dtype=np.float32) * alpha
        )
        rgba_layer[seg, :3] = color_bgr
        rgba_layer[seg, 3] = 255

        # Optional border
        if border_thickness > 0:
            seg_u8 = seg.astype(np.uint8) * 255
            contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay_bgr, contours, -1,
                             (255, 255, 255), border_thickness)
        n_drawn += 1

    log.info("SAM visualize: %d/%d masks drawn (min_area=%.2f%% filtered out)",
             n_drawn, len(masks), min_area_pct * 100)

    overlay_bgr = np.clip(overlay_bgr, 0, 255).astype(np.uint8)
    overlay_rgba = np.clip(rgba_layer, 0, 255).astype(np.uint8)
    return overlay_bgr, overlay_rgba


def run_sam_auto_generator(
    image_bgr: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    model_type: str = "auto",  # 'vit_b' | 'vit_l' | 'vit_h' | 'auto'
    device: str = "auto",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 100,
) -> tuple[list[dict], str]:
    """Chạy SAM AutomaticMaskGenerator → list of mask dicts.

    Args:
        image_bgr: full-res ảnh.
        checkpoint: SAM 1 .pth path (auto-detect nếu None).
        model_type: 'auto' detect từ filename, hoặc explicit.
        device: 'auto' | 'cuda' | 'cpu'.
        points_per_side: grid density (32 = default, 64 = dày hơn).
        pred_iou_thresh: filter masks confidence.
        stability_score_thresh: filter masks ổn định.
        min_mask_region_area: drop blobs nhỏ.

    Returns:
        (masks, model_type_used)
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            f"segment_anything không cài. "
            f"Run: pip install 'git+https://github.com/facebookresearch/segment-anything.git'\n"
            f"({exc})"
        )

    # Resolve checkpoint
    if checkpoint is None:
        cache = Path.home() / ".cache" / "sam"
        for fname in ("sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"):
            cand = cache / fname
            if cand.exists():
                checkpoint = cand
                break
        if checkpoint is None:
            raise RuntimeError(
                "SAM 1 checkpoint missing. Run: bash scripts/download_sam.sh vit_b"
            )

    checkpoint = Path(checkpoint)
    if model_type == "auto":
        name = checkpoint.name.lower()
        if "vit_h" in name:
            model_type = "vit_h"
        elif "vit_l" in name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"

    # Pick device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    log.info("Loading SAM %s on %s (checkpoint=%s)", model_type, device, checkpoint)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)

    log.info("AutomaticMaskGenerator: points_per_side=%d, iou_thresh=%.2f, "
             "stability=%.2f, min_area=%d",
             points_per_side, pred_iou_thresh, stability_score_thresh,
             min_mask_region_area)
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = gen.generate(rgb)
    log.info("SAM AutomaticMaskGenerator → %d masks", len(masks))
    return masks, model_type


def sam_demo_pipeline(
    image_bgr: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    points_per_side: int = 32,
    alpha: float = 0.35,
    seed: int = 42,
    border_thickness: int = 0,
) -> SAMAutoResult:
    """Full pipeline: SAM auto → visualize → SAMAutoResult."""
    masks, model_type = run_sam_auto_generator(
        image_bgr,
        checkpoint=checkpoint,
        points_per_side=points_per_side,
    )
    overlay_bgr, overlay_rgba = visualize_sam_masks(
        image_bgr, masks, alpha=alpha, seed=seed, border_thickness=border_thickness,
    )
    return SAMAutoResult(
        n_masks=len(masks),
        overlay_bgr=overlay_bgr,
        overlay_rgba=overlay_rgba,
        masks_raw=masks,
        model_type=model_type,
    )
