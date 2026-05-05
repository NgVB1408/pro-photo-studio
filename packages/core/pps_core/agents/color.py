"""Colour Specialist — calibrates vibrance with skin-tone protection."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class ColorAgent:
    name = "Colour Specialist"
    role = "Tunes vibrance for the listing while protecting skin tones and warm woods."
    category = "color_richness"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Greens look fresh, not radioactive",
        "Wood / stone tones look natural",
        "No oversaturation in textiles",
    )

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        s_mean = float(sat.mean())
        s_p95 = float(np.percentile(sat, 95))

        if s_mean < 0.10:
            verdict = "warn"
            score = 5.0
            summary = "Image looks desaturated; vibrance can lift it."
        elif s_mean > 0.45 or s_p95 > 0.95:
            verdict = "fail"
            score = 4.0
            summary = "Oversaturated — pull vibrance back."
        elif 0.18 <= s_mean <= 0.32 and 0.55 <= s_p95 <= 0.85:
            verdict = "pass"
            score = 9.5
            summary = "Colour richness is in the professional sweet spot."
        else:
            verdict = "warn"
            score = 7.5
            summary = "Colour is acceptable but a touch flat or punchy."
        items = tuple(
            AgentChecklistItem(
                label=l,
                status=verdict,
                detail=f"Saturation mean = {s_mean:.2f}, p95 = {s_p95:.2f}",
            )
            for l in self.CHECKLIST_LABELS
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={"sat_mean": s_mean, "sat_p95": s_p95},
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
        s_mean = float(m.get("sat_mean", 0.25))
        s_p95 = float(m.get("sat_p95", 0.7))

        actions: list[str] = []
        params: dict[str, float] = {}
        out = image
        applied = False

        if s_mean < 0.16:
            boost = float(min(0.32, 0.4 - s_mean))
            out = _vibrance_skin_safe(out, boost=boost)
            actions.append(f"Skin-safe vibrance +{boost:.2f}")
            params["vibrance_boost"] = boost
            applied = True
        elif s_p95 > 0.92 or s_mean > 0.38:
            cut = float(min(0.30, max(0.0, s_mean - 0.30)))
            out = _vibrance_skin_safe(out, boost=-cut * 0.8)
            actions.append(f"Vibrance pulled back by {cut:.2f}")
            params["vibrance_pullback"] = cut
            applied = True

        return out, AgentApplyReport(
            applied=applied,
            actions=tuple(actions),
            params=params,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary if applied else "Colour already in target band.",
        )


def _vibrance_skin_safe(image: np.ndarray, *, boost: float) -> np.ndarray:
    """Adjust saturation, but protect skin-tone hue band (0–25, 170+ deg in HSV)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    skin_mask = (h <= 25) | (h >= 170)
    multiplier = np.where(
        skin_mask,
        np.float32(1.0 + boost * 0.3),
        np.float32(1.0 + boost),
    ).astype(np.float32)
    s = np.clip(s * multiplier, 0, 255).astype(np.float32)
    merged = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
