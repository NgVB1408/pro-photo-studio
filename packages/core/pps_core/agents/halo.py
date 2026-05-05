"""Halo Specialist — detects and dampens ringing around high-contrast edges."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class HaloAgent:
    name = "Halo Inspector"
    role = "Hunts down ringing artefacts around windows, frames, and silhouettes."
    category = "halo"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "No ringing on window frames",
        "No haloed silhouettes against the sky",
        "Edge transitions are smooth, not stair-stepped",
    )

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low = cv2.GaussianBlur(gray, (0, 0), sigmaX=4)
        high = gray - low
        edges = cv2.Canny(gray.astype(np.uint8), 80, 200)
        if edges.sum() == 0:
            return AgentEvaluation(
                score=9.5,
                checklist=tuple(
                    AgentChecklistItem(label=l, status="pass", detail="No strong edges to inspect")
                    for l in self.CHECKLIST_LABELS
                ),
                summary="No high-contrast edges to evaluate halo on.",
                metrics={"edge_pixels": 0.0, "overshoot_p95": 0.0},
            )
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        near = edge_dilated > 0
        overshoot = float(np.percentile(np.abs(high[near]), 95))

        if overshoot < 18:
            status = "pass"
        elif overshoot < 30:
            status = "warn"
        else:
            status = "fail"

        items = tuple(
            AgentChecklistItem(
                label=label,
                status=status,
                detail=f"95th-percentile edge overshoot = {overshoot:.1f}",
            )
            for label in self.CHECKLIST_LABELS
        )
        score = float(np.clip(10.0 - max(0.0, overshoot - 12.0) * 0.4, 0.0, 10.0))
        summary = (
            "No visible halo around high-contrast edges."
            if status == "pass"
            else "Slight ringing detected — soften the contrast operator."
            if status == "warn"
            else "Strong halo / ringing — switch to halo-free local detail."
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={
                "overshoot_p95": overshoot,
                "edge_pixels": float(edges.sum()),
            },
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
        overshoot = float(evaluation.metrics.get("overshoot_p95", 0.0))
        if overshoot < 22:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="No corrective action needed.",
            )
        # Bilateral smoothing only on the near-edge band reduces ringing without
        # softening the rest of the image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 80, 200), np.ones((3, 3), np.uint8))
        smoothed = cv2.bilateralFilter(image, d=7, sigmaColor=25, sigmaSpace=7)
        mask = edges > 0
        out = image.copy()
        out[mask] = smoothed[mask]
        return out, AgentApplyReport(
            applied=True,
            actions=("Edge-band bilateral smoothing to dampen ringing",),
            params={"overshoot_before": overshoot},
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary,
        )
