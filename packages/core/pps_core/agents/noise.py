"""Noise Specialist — quantifies grain and applies targeted denoise."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class NoiseAgent:
    name = "Noise Reduction Specialist"
    role = "Removes high-ISO grain while keeping fabric and stone texture intact."
    category = "noise"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Flat surfaces are clean",
        "No coloured speckle in shadows",
        "Texture preserved on fabric / stone / wood",
    )

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (5, 5), 1.2)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigma = 1.4826 * mad
        score = float(np.clip(10.0 - max(0.0, sigma - 1.0) * 0.7, 0.0, 10.0))
        status = "pass" if sigma < 1.5 else "warn" if sigma < 4.0 else "fail"
        items = tuple(
            AgentChecklistItem(
                label=l,
                status=status,
                detail=f"σ ≈ {sigma:.2f} (MAD = {mad:.2f})",
            )
            for l in self.CHECKLIST_LABELS
        )
        summary = (
            "Image is clean."
            if status == "pass"
            else "Visible grain — gentle denoise will help."
            if status == "warn"
            else "High-ISO grain dominant — apply NLM."
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={"sigma_est": sigma, "mad": mad},
        )

    def apply(
        self,
        image: np.ndarray,
        *,
        scene: str,
        evaluation: AgentEvaluation,
    ) -> tuple[np.ndarray, AgentApplyReport]:
        del scene
        t0 = time.perf_counter()
        sigma = float(evaluation.metrics.get("sigma_est", 0.0))
        if sigma < 1.5:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="No denoise needed.",
            )
        # Strength scales with detected sigma but capped to avoid plastic look.
        h_lum = float(min(7.0, max(2.5, sigma * 1.8)))
        h_color = float(min(7.0, max(2.0, sigma * 1.5)))
        out = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=h_lum,
            hColor=h_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return out, AgentApplyReport(
            applied=True,
            actions=("Non-local-means denoise on luma + chroma",),
            params={
                "sigma_before": sigma,
                "h_luminance": h_lum,
                "h_color": h_color,
            },
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary,
        )
