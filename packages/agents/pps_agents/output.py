"""Output specialist — SOP Phần 5 (final + 8K export).

Shadow-targeted denoise, output sharpen for screen, upscale to ``target_long_edge``
(8K = 7680px), DPI tag. Real-ESRGAN if available, otherwise high-quality
Lanczos with mild post-sharpen.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseAgent
from .types import JobContext, StagePlan, StageReport


class OutputAgent(BaseAgent):
    name = "output"

    def _analyze(self, ctx: JobContext) -> StagePlan:
        img = ctx.image
        h, w = img.shape[:2]
        long_edge = max(h, w)
        upscale_factor = ctx.target_long_edge / max(long_edge, 1)
        upscale_factor = max(1.0, min(upscale_factor, 4.0))

        # Shadow noise estimate: std-dev in dark regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark = gray < 60
        shadow_noise = float(gray[dark].std()) if dark.any() else 0.0

        ops = []
        if shadow_noise > 9.0:
            ops.append({"op": "shadow_denoise", "strength": min(int(shadow_noise / 2), 10)})
        if upscale_factor > 1.05:
            ops.append({"op": "upscale", "factor": upscale_factor,
                        "target_long_edge": ctx.target_long_edge})
        ops.append({"op": "output_sharpen", "amount": 0.25, "sigma": 0.8})

        return StagePlan(
            name=self.name,
            operations=ops,
            metadata={
                "current_long_edge": long_edge,
                "target_long_edge": ctx.target_long_edge,
                "upscale_factor": float(upscale_factor),
                "shadow_noise_std": shadow_noise,
                "dpi": ctx.target_dpi,
            },
        )

    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        report = StageReport(name=self.name, metrics={})
        out = image
        for op in plan.operations:
            if op["op"] == "shadow_denoise":
                out = self._shadow_denoise(out, op["strength"])
                report.metrics["shadow_denoise_strength"] = op["strength"]
            elif op["op"] == "upscale":
                out, method = self._upscale(out, target_long_edge=op["target_long_edge"])
                report.metrics["upscale"] = {
                    "method": method,
                    "factor": op["factor"],
                    "target_long_edge": op["target_long_edge"],
                }
            elif op["op"] == "output_sharpen":
                out = self._output_sharpen(out, amount=op["amount"], sigma=op["sigma"])
                report.metrics["output_sharpen"] = op["amount"]
        return out, report

    @staticmethod
    def _shadow_denoise(image: np.ndarray, strength: int) -> np.ndarray:
        """Bilateral filter only in shadow regions (texture preserved elsewhere)."""
        if strength <= 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < 90).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10.0)[..., None]
        denoised = cv2.bilateralFilter(image, d=strength, sigmaColor=40, sigmaSpace=40)
        out = image.astype(np.float32) * (1 - mask) + denoised.astype(np.float32) * mask
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _upscale(
        image: np.ndarray, *, target_long_edge: int
    ) -> tuple[np.ndarray, str]:
        h, w = image.shape[:2]
        if max(h, w) >= target_long_edge:
            return image, "skip_already_large"
        # Try Real-ESRGAN if installed (lazy import).
        try:
            from pps_core.enhance import upscale_realesrgan

            scale = 2 if target_long_edge / max(h, w) <= 2.1 else 4
            out = upscale_realesrgan(image, scale=scale)
            # Resize down to exact target
            out = OutputAgent._resize_to_long_edge(out, target_long_edge)
            return out, f"realesrgan_x{scale}"
        except Exception:
            return (
                OutputAgent._resize_to_long_edge(image, target_long_edge),
                "lanczos",
            )

    @staticmethod
    def _resize_to_long_edge(image: np.ndarray, target_long_edge: int) -> np.ndarray:
        h, w = image.shape[:2]
        cur = max(h, w)
        if cur == target_long_edge:
            return image
        scale = target_long_edge / cur
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _output_sharpen(image: np.ndarray, *, amount: float, sigma: float) -> np.ndarray:
        if amount <= 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        out = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
