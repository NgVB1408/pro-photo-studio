"""Cleanup specialist — SOP Phần 4 (Advanced Cleanup).

Object removal (small clutter), Sky/Grass enhancement, optional Screen
Replacement (TV blackout) and Fireplace overlay. Each sub-task runs
independently; analyze() builds masks once, apply() composites them.

Production note — seamless compositing techniques to evaluate when we move
beyond CPU baseline (these fix the "viền trắng / bóng ma" failures of naive
paste-on-paste):

* **Poisson Image Editing** (Pérez et al., SIGGRAPH 2003) — solves a Poisson
  PDE so a pasted patch adapts to the lighting of the host image. OpenCV
  exposes it as ``cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)``
  / ``MIXED_CLONE`` / ``MONOCHROME_TRANSFER``. Use for screen replacement,
  fireplace overlay, sky swaps, and any object-removal patch composite.
* **PatchMatch** (Barnes et al., SIGGRAPH 2009) — randomised
  nearest-neighbour patch search. Use for content-aware fill / object
  removal where we need to synthesise plausible texture from elsewhere in
  the image. ``opencv-contrib`` ships ``cv2.xphoto.inpaint`` plus there's
  the ``patchmatch`` Python wrapper for higher-quality inpaint.
* **Multi-Scale (Laplacian) Pyramid Blending** — already used in
  ``MicroContrastAgent._multi_band_texture`` for sharpening. The same
  pyramid-blend pattern (build LP for src + dst, blend per-level, reconstruct)
  is the right choice for sky swap edges.

These belong in v2 of the agent — for v1 we ship the simpler thresholded
masking path and let the Director flag halo / fringe issues.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseAgent
from .types import JobContext, StagePlan, StageReport


class CleanupAgent(BaseAgent):
    name = "cleanup"

    def _analyze(self, ctx: JobContext) -> StagePlan:
        img = ctx.image
        masks: dict[str, np.ndarray] = {}
        meta: dict = {}
        ops: list[dict] = []

        # Sky mask (only outdoor scenes)
        sky_mask, is_outdoor = self._detect_sky(img)
        meta["is_outdoor"] = is_outdoor
        if sky_mask is not None and float((sky_mask > 0).mean()) > 0.03:
            masks["sky"] = sky_mask
            ops.append({"op": "sky_fix"})

        # Lawn mask
        lawn_mask = self._detect_lawn(img)
        if lawn_mask is not None and float((lawn_mask > 0).mean()) > 0.02:
            masks["lawn"] = lawn_mask
            ops.append({"op": "enhance_lawn"})

        # TV / monitor candidates: large dark rectangles in interior scenes
        tv_mask = self._detect_dark_rectangles(img)
        if tv_mask is not None and float((tv_mask > 0).mean()) > 0.005:
            masks["tv"] = tv_mask
            ops.append({"op": "tv_blackout"})

        # Photographer / tripod reflection in mirrors — heuristic only:
        # high-contrast triangular shape near floor in bright wall regions. We
        # mark as "needs human attention" rather than blindly inpainting.
        meta["photog_reflection_check"] = "deferred_to_user"

        skip = not ops
        return StagePlan(
            name=self.name,
            operations=ops,
            masks=masks,
            metadata=meta,
            skip=skip,
            skip_reason="no_cleanup_targets" if skip else "",
        )

    # ------------------------------------------------------------------
    # detection helpers (try pps_core first, robust fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sky(img: np.ndarray) -> tuple[np.ndarray | None, bool]:
        try:
            from pps_core.realestate import detect_sky_mask, is_outdoor_scene

            outdoor, _ = is_outdoor_scene(img)
            if not outdoor:
                return None, False
            mask = detect_sky_mask(img)
            return mask, True
        except Exception:
            return None, False

    @staticmethod
    def _detect_lawn(img: np.ndarray) -> np.ndarray | None:
        try:
            from pps_core.realestate import detect_lawn_mask

            return detect_lawn_mask(img)
        except Exception:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            return green if green.any() else None

    @staticmethod
    def _detect_dark_rectangles(img: np.ndarray) -> np.ndarray | None:
        """Large dark connected components ~rectangular = candidate TV/monitor."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dark = (gray < 50).astype(np.uint8) * 255
        kern = np.ones((5, 5), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kern)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(dark)
        out = np.zeros_like(dark)
        min_area = (h * w) * 0.005
        max_area = (h * w) * 0.20
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < min_area or area > max_area:
                continue
            ar = ww / max(hh, 1)
            if 0.8 <= ar <= 2.4:  # screens roughly 16:9-ish
                out[lbl == i] = 255
        return out if out.any() else None

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        report = StageReport(name=self.name, metrics={})
        out = image
        for op in plan.operations:
            if op["op"] == "sky_fix":
                out, info = self._sky_fix(out, plan.masks["sky"])
                report.metrics["sky"] = info
            elif op["op"] == "enhance_lawn":
                out = self._lawn_enhance(out, plan.masks["lawn"])
                report.metrics["lawn"] = "applied"
            elif op["op"] == "tv_blackout":
                out = self._tv_blackout(out, plan.masks["tv"])
                report.metrics["tv"] = "applied"
        return out, report

    @staticmethod
    def _sky_fix(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        """Try replace_sky if available; else just lift sky vibrance + WB shift."""
        try:
            from pps_core.realestate import replace_sky

            out = replace_sky(image, sky_preset="blue_clouds")
            return out, {"method": "replace_sky"}
        except Exception:
            f = image.astype(np.float32)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            # Boost B, slightly cool — sky-only WB nudge
            f_sky = f.copy()
            f_sky[..., 0] = f_sky[..., 0] * 1.06
            f_sky[..., 2] = f_sky[..., 2] * 0.96
            out = f * (1 - alpha) + f_sky * alpha
            return np.clip(out, 0, 255).astype(np.uint8), {"method": "fallback_wb_shift"}

    @staticmethod
    def _lawn_enhance(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            from pps_core.realestate import enhance_lawn

            return enhance_lawn(image)
        except Exception:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            alpha = (mask.astype(np.float32) / 255.0)
            hsv[..., 1] = hsv[..., 1] + 25 * alpha
            hsv[..., 0] = np.clip(hsv[..., 0] - 2 * alpha, 0, 179)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _tv_blackout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            from pps_core import tv_blackout  # type: ignore

            if hasattr(tv_blackout, "blacken_tvs"):
                return tv_blackout.blacken_tvs(image)
        except Exception:
            pass
        out = image.copy()
        out[mask > 0] = (15, 15, 15)
        return out
