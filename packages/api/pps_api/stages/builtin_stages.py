"""Built-in pps_core stage adapters.

Each adapter wraps an existing ``pps_core.<module>`` function in the
``Stage`` protocol so the pipeline runner can invoke it. Stages are kept
intentionally thin: they pull params out of ``ctx``, call the underlying
function, and translate the result into a ``StageReport``.

The registration ``@register("name")`` happens at import time, so the
``app`` factory just imports this module to populate the global registry.
"""

from __future__ import annotations

import time

import numpy as np
from pps_core.pipeline import register
from pps_core.types import StageContext, StageReport


class _Preflight:
    name = "preflight"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        from pps_core.preflight import analyze_image

        rpt = analyze_image(image)
        warnings = tuple(("warn", w) for w in rpt.warnings)
        return image, StageReport(
            name=ctx.stage_name,
            applied=True,
            warnings=warnings,
            metrics={
                "blur_score": float(rpt.blur_score),
                "highlight_clip_pct": float(rpt.highlight_clip_pct),
                "shadow_clip_pct": float(rpt.shadow_clip_pct),
                "avg_brightness": float(rpt.avg_brightness),
            },
            reason=rpt.severity,
        )


class _RealEstate:
    name = "real_estate"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        from pps_core.realestate import enhance_realestate_full

        params = ctx.params
        out, rpt = enhance_realestate_full(
            image,
            sky_preset=params.get("sky_preset", "blue_clouds"),
            seed=ctx.stage_seed,
            enable_sky=bool(params.get("enable_sky", False)),
            use_ai_sky=bool(params.get("use_ai_sky", True)),
        )
        return out, StageReport(
            name=ctx.stage_name,
            applied=True,
            metrics={
                "scene": _scene_to_metric(rpt.scene.tag),
                "sky_replaced": float(rpt.sky_replaced),
                "windows_recovered": float(rpt.windows_recovered),
                "lawn_enhanced": float(rpt.lawn_enhanced),
                "vertical_rotated": float(rpt.vertical.rotated),
            },
            reason=rpt.scene.tag,
        )


class _Twilight:
    name = "twilight"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        from pps_core.twilight import transform_to_twilight

        params = ctx.params
        out, tw = transform_to_twilight(
            image,
            strength=float(params.get("strength", 0.85)),
            seed=ctx.stage_seed,
            use_ai_sky=bool(params.get("use_ai_sky", True)),
            glow_intensity=float(params.get("glow_intensity", 0.30)),
            warm_tone=bool(params.get("warm_tone", True)),
        )
        return out, StageReport(
            name=ctx.stage_name,
            applied=tw.applied,
            metrics={
                "sky_mask_pct": float(tw.sky_mask_pct),
                "glow_windows_pct": float(tw.glow_windows_pct),
            },
            reason=tw.reason,
        )


class _Perspective:
    name = "perspective"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        from pps_core.perspective import correct_upright

        out, rpt = correct_upright(image)
        return out, StageReport(
            name=ctx.stage_name,
            applied=rpt.applied,
            metrics={
                "skew": float(rpt.skew),
                "lines_used": float(rpt.lines_used),
                "angle_estimate_deg": float(rpt.angle_estimate_deg),
            },
            reason=rpt.reason or rpt.direction,
        )


class _Identity:
    """Trivial stage that returns the input unchanged. Useful for tests
    and as a sanity check that the pipeline plumbing is alive."""

    name = "identity"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        # Sleep a hair so timing measurements are non-zero on fast machines.
        time.sleep(0.001)
        return image, StageReport(
            name=ctx.stage_name,
            applied=True,
            metrics={"shape_h": float(image.shape[0]), "shape_w": float(image.shape[1])},
        )


class _EnhanceStudio:
    """Studio-grade 8-step enhance — punchier than ``real_estate`` alone.

    Designed to be chained AFTER ``real_estate`` for visibly stronger output
    (brightness lift, vibrant colours, detail enhancement). Tunable params:
        clahe_clip          default 3.5  (push to 5.0 for max contrast)
        highlight_recovery  default 0.40
        shadow_lift         default 0.45 (lift dim rooms)
        vibrance            default 0.30 (saturated, not over-cooked)
        unsharp_amount      default 0.55 (crisp detail)
        gamma               default 0.95 (slight brighten)
    """

    name = "enhance_studio"

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        from pps_core.enhance import EnhanceParams, enhance_studio

        p = ctx.params
        params = EnhanceParams(
            white_balance="auto",
            clahe_clip=float(p.get("clahe_clip", 3.5)),
            highlight_recovery=float(p.get("highlight_recovery", 0.40)),
            shadow_lift=float(p.get("shadow_lift", 0.45)),
            vibrance=float(p.get("vibrance", 0.30)),
            unsharp_amount=float(p.get("unsharp_amount", 0.55)),
            unsharp_sigma=float(p.get("unsharp_sigma", 1.5)),
            gamma=float(p.get("gamma", 0.95)),
        )
        out = enhance_studio(image, params=params)
        return out, StageReport(
            name=ctx.stage_name,
            applied=True,
            metrics={
                "clahe_clip": params.clahe_clip,
                "shadow_lift": params.shadow_lift,
                "vibrance": params.vibrance,
            },
        )


def _scene_to_metric(tag: str) -> float:
    """Map scene tag → numeric so it can ride in the metrics dict."""
    return {"interior": 1.0, "exterior": 2.0, "aerial": 3.0}.get(tag, 0.0)


# Register stage instances. Doing this at module-import time means simply
# importing ``pps_api.stages.builtin_stages`` populates the global registry
# the pipeline runner reads from.
register("preflight")(_Preflight())
register("real_estate")(_RealEstate())
register("twilight")(_Twilight())
register("perspective")(_Perspective())
register("identity")(_Identity())
register("enhance_studio")(_EnhanceStudio())
