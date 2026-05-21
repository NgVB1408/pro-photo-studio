"""Pipeline orchestrator v0.2 — semantic segmentation + quality fixers + self-eval."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .context import SceneContext, classify_scene
from .detector import (
    DetectionDebug,
    SegmentationResult,
    detect_ceiling_mask,
    detect_window_mask,
    ensure_ai_available,
    segment,
)
from .evaluator import SelfEvaluation, evaluate
from .fixers import fix_ceiling_neutrality, fix_window_highlights
from .io_quality import ImageMeta, read_image, write_image
from .tuner import FixerParams, tune

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    width: int
    height: int
    duration_s: float
    runtime: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    tuning: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    window: dict = field(default_factory=dict)
    ceiling: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    debug_paths: dict = field(default_factory=dict)

    def report(self) -> str:
        verdict = self.evaluation.get("verdict", "?")
        score = self.evaluation.get("overall_score", 0.0)
        lines = [
            f"📷 Input  : {self.input_path}",
            f"💾 Output : {self.output_path}",
            f"📐 Size   : {self.width}×{self.height}",
            f"⏱️  Time   : {self.duration_s:.2f}s",
            f"{self.runtime.get('banner', '')}",
            "",
            "🔍 CONTEXT:",
            f"   {self.context.get('summary', '')}",
            f"   Cast={self.context.get('ceiling_cast_magnitude', 0):.1f}  "
            f"Clipped={self.context.get('window_clipped_pct', 0):.1f}%  "
            f"CCT={self.context.get('overall_color_temp_k', 0):.0f}K",
            "",
            "🎯 DECISIONS:",
        ]
        for r in self.tuning.get("reasoning", []):
            lines.append(f"   {r}")
        lines.append("")
        lines.append(
            f"🪟 WINDOW (mask {self.detection.get('window_pct', 0):.1f}%)"
            f"  applied={self.window.get('applied', False)}"
            f"  clipped {self.window.get('clipped_pct_before', 0):.1f}%→"
            f"{self.window.get('clipped_pct_after', 0):.1f}%"
        )
        lines.append(
            f"🏠 CEILING (mask {self.detection.get('ceiling_pct', 0):.1f}%)"
            f"  applied={self.ceiling.get('applied', False)}"
            f"  cast {self.ceiling.get('cast_magnitude_before', 0):.1f}→"
            f"{self.ceiling.get('cast_magnitude_after', 0):.1f}"
        )
        lines.append("")
        scope_ok = self.evaluation.get("scope_ok", True)
        scope_mean = self.evaluation.get("scope_delta_e", 0.0)
        scope_max = self.evaluation.get("scope_max_delta_e", 0.0)
        scope_icon = "✅" if scope_ok else "🚫"
        lines.append(
            f"🛡️  SCOPE: {scope_icon} wall/floor ΔE mean={scope_mean:.2f} p99={scope_max:.2f} "
            f"(giới hạn 2.0/5.0)"
        )
        lines.append(f"🤖 SELF-EVAL: {verdict.upper()}  score={score:.3f}/1.000")
        for finding in self.evaluation.get("findings", [])[:6]:
            lines.append(f"   • {finding}")
        return "\n".join(lines)


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    debug_dir: str | Path | None = None,
    window_strength: float | None = None,
    ceiling_strength: float | None = None,
    include_lamps: bool | None = None,
    expand_window_with_sky: bool | None = None,
    context_aware: bool = True,
    enable_clip: bool = True,
    self_evaluate: bool = True,
) -> ProcessResult:
    """Process 1 image end-to-end với context-aware semantic AI pipeline.

    Pipeline:
        Phase 0: Semantic segmentation (SegFormer ADE20K).
        Phase 1: Scene classification (CLIP zero-shot: room/lighting/mood/window/ceiling state).
        Phase 2: Adaptive tuner — context → fixer params + skip decisions.
        Phase 3: Apply fixers (skip vùng đã tốt).
        Phase 4: Self-evaluation (7 metrics).

    Args:
        window_strength / ceiling_strength: nếu None → tuner tự quyết.
                                            Nếu set → override tuner.
        context_aware: True → run scene classifier + tuner.
                       False → use base params (v0.2 behavior).
        enable_clip: True → use CLIP. False → heuristic-only classification.

    Returns:
        ProcessResult với context + tuning reasoning + metrics + verdict.
    """
    profile = ensure_ai_available()

    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Quality I/O: read with EXIF/ICC preserved
    img, img_meta = read_image(inp)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    debug = DetectionDebug()

    # Phase 0: Semantic segmentation
    seg: SegmentationResult = segment(img, debug=debug)

    # Phase 0.5: Initial mask (before context-aware tuning) — uses default flags
    expand_sky_initial = True if expand_window_with_sky is None else expand_window_with_sky
    include_lamps_initial = False if include_lamps is None else include_lamps
    win_mask = detect_window_mask(img, seg=seg, debug=debug, expand_with_sky=expand_sky_initial)
    ceil_mask = detect_ceiling_mask(img, seg=seg, debug=debug, include_lamps=include_lamps_initial)

    # Phase 1: Scene classification
    context: SceneContext
    if context_aware:
        context = classify_scene(
            img,
            seg=seg,
            window_mask=win_mask,
            ceiling_mask=ceil_mask,
            enable_clip=enable_clip,
        )
    else:
        context = SceneContext()

    # Phase 2: Adaptive tuning
    params: FixerParams = tune(context) if context_aware else FixerParams()

    # Manual overrides (user CLI > tuner decisions)
    if window_strength is not None:
        params.window_strength = window_strength
        params.skip_window = window_strength <= 0
        params.reasoning.append(f"⚙️  Override --window={window_strength} (skip tuner).")
    if ceiling_strength is not None:
        params.ceiling_strength = ceiling_strength
        params.skip_ceiling = ceiling_strength <= 0
        params.reasoning.append(f"⚙️  Override --ceiling={ceiling_strength} (skip tuner).")
    if include_lamps is not None:
        params.ceiling_include_lamps = include_lamps
    if expand_window_with_sky is not None:
        params.expand_window_with_sky = expand_window_with_sky

    # Re-derive masks if include_lamps / expand_with_sky changed
    if params.ceiling_include_lamps != include_lamps_initial or params.expand_window_with_sky != expand_sky_initial:
        win_mask = detect_window_mask(
            img, seg=seg, debug=debug, expand_with_sky=params.expand_window_with_sky
        )
        ceil_mask = detect_ceiling_mask(
            img, seg=seg, debug=debug, include_lamps=params.ceiling_include_lamps
        )

    # Phase 3: Apply fixers (sequential)
    if not params.skip_window and params.window_strength > 0:
        intermediate, win_metrics = fix_window_highlights(
            img,
            win_mask,
            strength=params.window_strength,
            knee=params.window_knee,
            target_ceiling=params.window_target_ceiling,
            chroma_recover=params.window_chroma_recover,
            guide_radius=params.window_guide_radius,
        )
    else:
        intermediate = img
        win_metrics = {"applied": False, "reason": "tuner_skip" if params.skip_window else "strength_0"}

    if not params.skip_ceiling and params.ceiling_strength > 0:
        final, ceil_metrics = fix_ceiling_neutrality(
            intermediate,
            ceil_mask,
            strength=params.ceiling_strength,
            luminance_equalize=params.ceiling_luminance_equalize,
            guide_radius=params.ceiling_guide_radius,
        )
    else:
        final = intermediate
        ceil_metrics = {"applied": False, "reason": "tuner_skip" if params.skip_ceiling else "strength_0"}

    duration = time.perf_counter() - t0

    # 4. Encode output — preserve EXIF + ICC + format-faithful
    write_image(final, out, img_meta, jpeg_quality=98)

    # 5. Self-evaluation
    evaluation_dict: dict = {}
    if self_evaluate:
        evaluation: SelfEvaluation = evaluate(
            before=img,
            after=final,
            window_mask=win_mask,
            ceiling_mask=ceil_mask,
            seg=seg,
        )
        evaluation_dict = asdict(evaluation)

    # 6. Debug dump
    debug_paths: dict[str, str] = {}
    if debug_dir is not None:
        dd = Path(debug_dir)
        dd.mkdir(parents=True, exist_ok=True)

        win_overlay = _make_overlay(img, win_mask, color=(255, 0, 255))
        ceil_overlay = _make_overlay(img, ceil_mask, color=(0, 255, 255))
        before_after = np.hstack([img, final])

        cv2.imwrite(str(dd / "window_mask.png"), win_mask)
        cv2.imwrite(str(dd / "window_overlay.jpg"), win_overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(dd / "ceiling_mask.png"), ceil_mask)
        cv2.imwrite(str(dd / "ceiling_overlay.jpg"), ceil_overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(dd / "before_after.jpg"), before_after, [cv2.IMWRITE_JPEG_QUALITY, 90])

        debug_paths = {
            "window_mask": str(dd / "window_mask.png"),
            "window_overlay": str(dd / "window_overlay.jpg"),
            "ceiling_mask": str(dd / "ceiling_mask.png"),
            "ceiling_overlay": str(dd / "ceiling_overlay.jpg"),
            "before_after": str(dd / "before_after.jpg"),
        }

    runtime_info = {
        "banner": profile.banner(),
        "use_gpu": profile.use_gpu,
        "cuda_device_name": profile.cuda_device_name,
        "cuda_vram_gb": profile.cuda_vram_gb,
        "onnx_providers": list(profile.onnx_providers),
    }

    # Context dict (include summary string for readable report)
    context_dict = asdict(context)
    context_dict["summary"] = context.summary() if context_aware else ""

    # Tuning dict (params + reasoning)
    tuning_dict = {
        "window_strength": params.window_strength,
        "window_knee": params.window_knee,
        "window_target_ceiling": params.window_target_ceiling,
        "window_guide_radius": params.window_guide_radius,
        "ceiling_strength": params.ceiling_strength,
        "ceiling_include_lamps": params.ceiling_include_lamps,
        "expand_window_with_sky": params.expand_window_with_sky,
        "skip_window": params.skip_window,
        "skip_ceiling": params.skip_ceiling,
        "reasoning": params.reasoning,
    }

    result = ProcessResult(
        input_path=str(inp),
        output_path=str(out),
        width=w,
        height=h,
        duration_s=duration,
        runtime=runtime_info,
        context=context_dict,
        tuning=tuning_dict,
        detection=asdict(debug),
        window=win_metrics,
        ceiling=ceil_metrics,
        evaluation=evaluation_dict,
        debug_paths=debug_paths,
    )

    if debug_dir is not None:
        sc = Path(debug_dir) / "scorecard.json"
        sc.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        result.debug_paths["scorecard"] = str(sc)

    return result


def _make_overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = img.copy()
    if mask.sum() == 0:
        return overlay
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(overlay)
    tint[:] = color
    blended = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
    return np.where(mask_3 > 32, blended, overlay)
