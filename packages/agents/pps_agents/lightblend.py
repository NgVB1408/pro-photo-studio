"""Light Blending specialist — SOP Phần 2.

Flash/Ambient blend, Window Pull, Halo Check, Shadow Integrity.

Production note — the bracket-merge / flash+ambient combine path needs robust
patch alignment when the camera or subjects shifted between captures. Plan to
adopt:

* **PatchMatch** (Barnes et al., SIGGRAPH 2009) for fast randomised
  nearest-neighbour patch search across the bracket — handles slight tilt /
  parallax that pure homography registration can't catch. Wrap via
  ``opencv-contrib`` xphoto APIs or a thin C++ binding.
* **Laplacian-pyramid blend** of the registered exposures — mid/macro bands
  carry the colour and shape, fine band carries the new sharper detail. This
  is what kills the white fringe at window edges in the current threshold-mask
  blend.

Current v1 implementation is a single-shot tone/highlight recovery, which is
deterministic and adequate for properly-exposed source frames; the bracket
fusion path is wired to ``pps_core.tone`` for now.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseAgent
from .types import JobContext, StagePlan, StageReport


class LightBlendAgent(BaseAgent):
    name = "lightblend"

    def _analyze(self, ctx: JobContext) -> StagePlan:
        img = ctx.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Dynamic range stats
        p1 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))
        clip_high = float((gray >= 250).mean())
        clip_low = float((gray <= 5).mean())

        # Blown windows mask (tries pps_core first, falls back to threshold)
        blown_ratio, blown_mask = self._detect_blown(img)

        ops: list[dict] = []
        masks: dict[str, np.ndarray] = {}

        # Highlight recovery: if >0.5% pixels at clip-high
        if clip_high > 0.005:
            recovery = float(min(0.6, 0.3 + 25 * clip_high))
            ops.append({"op": "highlight_recovery", "amount": recovery})

        # Window pull only when there's a real blown-window region
        if blown_ratio > 0.005 and blown_mask is not None:
            ops.append({"op": "window_pull", "blown_ratio": blown_ratio})
            masks["blown_windows"] = blown_mask

        # Shadow lift — strength scaled by % deep shadows
        shadow_ratio = float((gray <= 30).mean())
        if shadow_ratio > 0.02:
            lift = float(min(0.45, 0.25 + 4.0 * shadow_ratio))
            ops.append({"op": "shadow_lift", "amount": lift})

        skip = not ops
        return StagePlan(
            name=self.name,
            operations=ops,
            masks=masks,
            metadata={
                "p1": p1,
                "p99": p99,
                "clip_high": clip_high,
                "clip_low": clip_low,
                "blown_window_ratio": blown_ratio,
                "shadow_ratio": shadow_ratio,
            },
            skip=skip,
            skip_reason="dynamic_range_ok" if skip else "",
        )

    @staticmethod
    def _detect_blown(img: np.ndarray) -> tuple[float, np.ndarray | None]:
        try:
            from pps_core.realestate import detect_blown_windows

            mask = detect_blown_windows(img)
            if mask is None:
                return 0.0, None
            ratio = float((mask > 127).mean())
            return ratio, mask
        except Exception:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blown = (gray > 245).astype(np.uint8) * 255
            ratio = float((blown > 0).mean())
            return ratio, blown if ratio > 0 else None

    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        report = StageReport(name=self.name, metrics={})
        out = image
        for op in plan.operations:
            if op["op"] == "highlight_recovery":
                from pps_core.enhance import highlight_recovery

                out = highlight_recovery(out, amount=op["amount"])
                report.metrics["highlight_recovery"] = op["amount"]
            elif op["op"] == "window_pull":
                try:
                    from pps_core.realestate import window_pull

                    out, info = window_pull(out)
                    report.metrics["window_pull"] = (
                        info if isinstance(info, dict) else {"applied": True}
                    )
                except Exception as exc:
                    report.warnings.append(f"window_pull_failed: {exc!r}")
            elif op["op"] == "shadow_lift":
                from pps_core.enhance import shadow_lift

                out = shadow_lift(out, amount=op["amount"])
                report.metrics["shadow_lift"] = op["amount"]

        # Halo guard: if window mask exists, soft-edge feather around it to
        # avoid leaving harsh seams between blown areas and the recovered tone.
        blown = plan.masks.get("blown_windows")
        if blown is not None and blown.size:
            out = self._feather_halo(out, image, blown)
            report.metrics["halo_feather"] = True

        return out, report

    @staticmethod
    def _feather_halo(
        new: np.ndarray, original: np.ndarray, blown_mask: np.ndarray
    ) -> np.ndarray:
        """Around the blown-window edges, blend new toward original to soften
        any halo introduced by aggressive highlight recovery."""
        if blown_mask.shape[:2] != new.shape[:2]:
            return new
        edges = cv2.morphologyEx(blown_mask, cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8))
        feather = cv2.GaussianBlur(edges, (0, 0), sigmaX=6.0)
        feather = (feather.astype(np.float32) / max(feather.max(), 1)).clip(0, 1)
        feather = feather[..., None]
        out = new.astype(np.float32) * (1 - 0.4 * feather) + original.astype(np.float32) * (
            0.4 * feather
        )
        return np.clip(out, 0, 255).astype(np.uint8)
