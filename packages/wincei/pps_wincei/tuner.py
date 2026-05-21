"""Adaptive tuner — SceneContext → FixerParams.

Logic:
  1. Decide window_strength theo window_clipped_pct + window_state + mood.
  2. Decide ceiling_strength theo cast magnitude + lighting + mood (preserve warmth nếu cozy/sunset).
  3. Adjust knee/target_ceiling theo blown level.
  4. Skip nếu KHÔNG cần fix (ảnh đã tốt).
  5. Output reasoning text để self-eval + user hiểu vì sao chọn vậy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .context import SceneContext

logger = logging.getLogger(__name__)


@dataclass
class FixerParams:
    """Per-image adapted fixer parameters.

    SCOPE PRINCIPLE (real-estate): tool ONLY edits window pixels + ceiling pixels.
    Wall / floor / furniture / view-through-window are untouched. Tighter defaults
    reflect this (subtle chroma, narrow guide radius, no luminance spread).
    """

    window_strength: float = 1.0
    window_knee: float = 0.55
    window_target_ceiling: float = 0.82
    window_chroma_recover: float = 1.04  # subtle
    window_guide_radius: int = 20        # tight

    ceiling_strength: float = 0.85
    ceiling_luminance_equalize: bool = False  # off by default — spreads via Gaussian
    ceiling_guide_radius: int = 16
    ceiling_include_lamps: bool = False
    expand_window_with_sky: bool = True

    reasoning: list[str] = field(default_factory=list)
    skip_window: bool = False
    skip_ceiling: bool = False


def tune(ctx: SceneContext, *, base: FixerParams | None = None) -> FixerParams:
    """Translate context → adapted params + reasoning.

    Args:
        ctx: SceneContext from `classify_scene`.
        base: Optional base params to start from (use to override defaults).

    Returns:
        FixerParams with reasoning explaining the choices.
    """
    p = base if base is not None else FixerParams()
    reasoning: list[str] = []

    # ── WINDOW DECISION ──────────────────────────────────────────────────────

    if ctx.is_exterior():
        p.skip_window = True
        p.window_strength = 0.0
        reasoning.append(
            f"⏭️  Window SKIP — ảnh exterior ({ctx.room_type}, conf={ctx.room_confidence:.0%}), "
            f"không có cửa sổ thực sự để fix."
        )
    elif not ctx.needs_window_fix():
        p.skip_window = True
        p.window_strength = 0.0
        reasoning.append(
            f"⏭️  Window SKIP — cửa sổ đã cân bằng "
            f"(clipped={ctx.window_clipped_pct:.1f}%, state={ctx.window_state} "
            f"conf={ctx.window_state_confidence:.0%}). Không cần can thiệp."
        )
    else:
        # Tune based on severity (guide_radius kept TIGHT — real-estate scope)
        if ctx.window_state == "blown_severe" or ctx.window_clipped_pct > 8:
            p.window_strength = 1.2
            p.window_knee = 0.48
            p.window_target_ceiling = 0.78
            p.window_chroma_recover = 1.08
            p.window_guide_radius = 24
            reasoning.append(
                f"🪟 Window AGGRESSIVE — blown {ctx.window_clipped_pct:.1f}% "
                f"({ctx.window_state}). knee=0.48, ceiling=0.78, scope tight."
            )
        elif ctx.window_state == "blown_mild" or ctx.window_clipped_pct > 2:
            p.window_strength = 1.0
            p.window_knee = 0.55
            p.window_target_ceiling = 0.82
            p.window_chroma_recover = 1.04
            p.window_guide_radius = 20
            reasoning.append(
                f"🪟 Window MODERATE — blown {ctx.window_clipped_pct:.1f}%. "
                f"Standard roll-off knee=0.55, scope tight."
            )
        else:
            p.window_strength = 0.7
            p.window_knee = 0.65
            p.window_target_ceiling = 0.88
            p.window_chroma_recover = 1.02
            p.window_guide_radius = 16
            reasoning.append(
                f"🪟 Window GENTLE — minor blown {ctx.window_clipped_pct:.1f}%. "
                f"Soft touch knee=0.65."
            )

        # Mood adjustment: scenic outdoor view through window → preserve more
        if ctx.extras.get("seg_sky_pct", 0) > 3.0:
            p.window_target_ceiling = min(0.92, p.window_target_ceiling + 0.05)
            reasoning.append(
                "🌤️  Có sky visible — nới target_ceiling +0.05 để giữ view ngoài trời."
            )

    # ── CEILING DECISION ─────────────────────────────────────────────────────

    if ctx.ceiling_state == "none" or ctx.extras.get("seg_ceiling_pct", 0) < 0.5:
        p.skip_ceiling = True
        p.ceiling_strength = 0.0
        reasoning.append("⏭️  Ceiling SKIP — không thấy trần trong ảnh.")
    elif not ctx.needs_ceiling_fix():
        p.skip_ceiling = True
        p.ceiling_strength = 0.0
        reasoning.append(
            f"⏭️  Ceiling SKIP — trần đã neutral "
            f"(cast={ctx.ceiling_cast_magnitude:.1f}, state={ctx.ceiling_state})."
        )
    else:
        # Default standard
        base_strength = 0.85

        # Mood/lighting adjustments
        if ctx.is_warm_mood():
            # Preserve warmth → moderate neutralize
            base_strength = 0.50
            reasoning.append(
                f"🔥 Mood ấm ({ctx.mood_style} / {ctx.lighting}) → "
                f"PRESERVE warmth, ceiling neutralize chỉ 0.50 (không ép trắng pure)."
            )
        elif ctx.lighting in {"evening_artificial", "low_light"}:
            base_strength = 0.65
            reasoning.append(
                f"🌙 Ánh sáng nhân tạo tối ({ctx.lighting}) → moderate 0.65."
            )
        elif ctx.lighting in {"daylight_natural", "overcast"}:
            base_strength = 0.85
            reasoning.append(
                f"☀️  Daylight tự nhiên → standard ceiling strength 0.85 about D65."
            )
        elif ctx.lighting == "mixed":
            base_strength = 0.75
            reasoning.append(
                f"⚖️  Mixed lighting → moderate-strong 0.75 (mixed thường cần balance)."
            )

        # Severity adjustment
        if ctx.ceiling_cast_magnitude > 15:
            base_strength = min(1.0, base_strength + 0.10)
            reasoning.append(
                f"🚨 Cast cực mạnh ({ctx.ceiling_cast_magnitude:.1f}) → tăng strength +0.10."
            )
        elif ctx.ceiling_cast_magnitude < 5:
            base_strength = max(0.30, base_strength - 0.15)
            reasoning.append(
                f"🤏 Cast nhẹ ({ctx.ceiling_cast_magnitude:.1f}) → giảm strength -0.15 (không over-correct)."
            )

        p.ceiling_strength = base_strength

        # Lamp inclusion: nếu mood industrial hoặc loft → có lamp/light fixture nổi bật trên trần
        if ctx.mood_style in {"industrial", "modern_minimal"}:
            p.ceiling_include_lamps = True
            reasoning.append(f"💡 Mood {ctx.mood_style} → include lamp fixtures vào ceiling mask.")

    p.reasoning = reasoning
    return p
