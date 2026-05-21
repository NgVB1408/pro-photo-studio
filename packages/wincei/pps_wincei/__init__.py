"""pps-wincei — AI Window + Ceiling fixer.

Tool độc lập tách từ Pro Photo Studio, chuyên 2 vấn đề bất động sản:
  • Cửa sổ blown-highlight (trời cháy trắng) → AI HDR recovery
  • Trần ám màu (xanh/vàng) → AI segmentation + LAB neutralize

AI BẮT BUỘC: dùng rembg + ONNX Runtime cho semantic segmentation.
Không có heuristic fallback — nếu AI fail thì raise.

Usage CLI:
    pps-wincei input.jpg --out output.jpg [--debug]

Usage Python:
    from pps_wincei import process_image
    result = process_image("input.jpg", "output.jpg")
    print(result.report)
"""

from __future__ import annotations

from .pipeline import ProcessResult, process_image
from .detector import (
    AISegmenter,
    SegmentationResult,
    ensure_ai_available,
    detect_window_mask,
    detect_ceiling_mask,
    segment,
)
from .fixers import (
    fix_window_highlights,
    fix_ceiling_neutrality,
)
from .evaluator import SelfEvaluation, evaluate
from .runtime import RuntimeProfile, detect_runtime
from .context import SceneContext, classify_scene
from .tuner import FixerParams, tune
from .io_quality import ImageMeta, read_image, write_image
from .viewer import ComparisonItem, build_comparison_item, generate_html

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "ProcessResult",
    "process_image",
    "AISegmenter",
    "SegmentationResult",
    "ensure_ai_available",
    "detect_window_mask",
    "detect_ceiling_mask",
    "segment",
    "fix_window_highlights",
    "fix_ceiling_neutrality",
    "SelfEvaluation",
    "evaluate",
    "RuntimeProfile",
    "detect_runtime",
    "SceneContext",
    "classify_scene",
    "FixerParams",
    "tune",
    "ImageMeta",
    "read_image",
    "write_image",
    "ComparisonItem",
    "build_comparison_item",
    "generate_html",
]
