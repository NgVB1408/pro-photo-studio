"""pps-wincei-masks — smart room segmentation + phào chỉ → Photoshop masks.

Pipeline:
    1. SegFormer-B3 ADE20K → semantic seg (wall/floor/ceiling/window/door/light/sky)
    2. PyMatting closed-form refinement (CPU, biên đẹp sub-pixel)
    3. Phào chỉ detection: seam-band + Canny + Hough line (crown/base/casing)
    4. Export: PNG mỗi class + multi-page TIFF + colored overlay JPG + optional PSD
"""

from .__version__ import __version__
from .semantic import SemanticSegmenter, SemanticResult
from .molding import detect_moldings, MoldingMasks
from .refine import refine_alpha_masks
from .exporters import export_all_masks, ExportResult
from .evaluator import evaluate_masks, EvalReport, MaskScore
from .pipeline import extract_masks, MaskExtractionResult
from .perfect_window import (
    extract_perfect_window, extract_perfect_window_from_path, PerfectWindowResult,
)
from .preprocess import preprocess_for_vlm_sam, apply_clahe, undistort_image
from .overlap_resolver import (
    resolve_all_overlaps,
    resolve_ceiling_wall_overlap,
    resolve_ceiling_floor_overlap,
)

__all__ = [
    "__version__",
    "SemanticSegmenter",
    "SemanticResult",
    "detect_moldings",
    "MoldingMasks",
    "refine_alpha_masks",
    "export_all_masks",
    "ExportResult",
    "evaluate_masks",
    "EvalReport",
    "MaskScore",
    "extract_masks",
    "MaskExtractionResult",
    "preprocess_for_vlm_sam",
    "apply_clahe",
    "undistort_image",
    "resolve_all_overlaps",
    "resolve_ceiling_wall_overlap",
    "resolve_ceiling_floor_overlap",
]
