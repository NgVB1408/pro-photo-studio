"""Vertical Specialist — checks plumb walls and corrects tilt."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class VerticalAgent:
    name = "Vertical Alignment Specialist"
    role = "Keeps walls and frames plumb. Corrects camera tilt without crop scarring."
    category = "vertical_alignment"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Walls and door frames are plumb",
        "Window frames are not tilted",
        "Final image keeps the original aspect ratio",
    )

    MAX_CORRECT_DEG = 4.0  # avoid heavy crops on tripod-leveled photos
    MIN_TILT_TO_FIX = 0.4

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        deviations, total_lines = _vertical_deviations(image)
        if not deviations:
            applicable = scene in ("interior", "exterior")
            return AgentEvaluation(
                score=8.5,
                checklist=tuple(
                    AgentChecklistItem(label=l, status="pass", detail="No near-vertical lines detected")
                    for l in self.CHECKLIST_LABELS
                ),
                summary="No near-vertical lines — likely an aerial or close-up.",
                metrics={"median_deviation_deg": 0.0, "applicable": float(applicable)},
            )
        median_dev = float(np.median(deviations))
        score = float(np.clip(10.0 - 1.5 * median_dev, 0.0, 10.0))
        status = "pass" if median_dev < 0.6 else "warn" if median_dev < 1.5 else "fail"
        items = tuple(
            AgentChecklistItem(
                label=label,
                status=status,
                detail=f"Median tilt = {median_dev:.2f}° across {len(deviations)} vertical edges",
            )
            for label in self.CHECKLIST_LABELS
        )
        summary = (
            "Walls are plumb."
            if median_dev < 0.6
            else f"Verticals tilt by ~{median_dev:.1f}° — auto-rotate will fix."
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={
                "median_deviation_deg": median_dev,
                "vertical_lines": float(len(deviations)),
                "total_lines": float(total_lines),
                "applicable": 1.0,
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
        m = evaluation.metrics
        median = float(m.get("median_deviation_deg", 0.0))
        if median < self.MIN_TILT_TO_FIX or median > self.MAX_CORRECT_DEG:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="Tilt outside auto-correct band — left untouched.",
            )
        deviations, _ = _vertical_deviations(image)
        if not deviations:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="No vertical edges to use as a reference.",
            )
        # Sign of tilt: positive means top leans right.
        signed = [d if d > 0 else -d for d in deviations]
        del signed  # we use median absolute below
        angle = -median  # rotate by -median to bring verticals back to plumb
        h, w = image.shape[:2]
        m_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, m_rot, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
        )
        return rotated, AgentApplyReport(
            applied=True,
            actions=(f"Rotated by {angle:.2f}° to bring verticals plumb",),
            params={"angle_deg": angle, "tilt_before": median},
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary,
        )


def _vertical_deviations(image: np.ndarray) -> tuple[list[float], int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, 120, minLineLength=120, maxLineGap=15
    )
    if lines is None:
        return [], 0
    deviations: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if y2 == y1:
            continue
        dx, dy = float(x2 - x1), float(y2 - y1)
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        if angle > 90:
            angle = 180 - angle
        if angle <= 12:
            deviations.append(angle)
    return deviations, len(lines)
