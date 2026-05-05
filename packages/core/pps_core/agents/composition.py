"""Composition Specialist — global tonal balance + listing readiness."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class CompositionAgent:
    name = "Composition Reviewer"
    role = "Final pass on global tone, contrast, and listing-thumbnail appeal."
    category = "composition"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Image reads well at thumbnail size",
        "Tonal balance is appealing (not dim, not washed-out)",
        "Contrast supports the focal point",
    )

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p10, p50, p90 = np.percentile(gray, [10, 50, 90])
        spread = float(p90 - p10)
        median = float(p50)

        score_median = 10.0 if 110 <= median <= 160 else max(0.0, 10.0 - abs(median - 135) * 0.08)
        score_spread = max(0.0, min(10.0, spread / 18.0))
        score = float(0.55 * score_median + 0.45 * score_spread)

        if median < 95:
            summary = "Image reads as dim — boost global exposure."
            status = "warn"
        elif median > 175:
            summary = "Image reads as washed-out — pull global brightness."
            status = "warn"
        elif spread < 90:
            summary = "Tonal spread is narrow; add micro-contrast."
            status = "warn"
        else:
            summary = "Composition is balanced and listing-ready."
            status = "pass"

        items = tuple(
            AgentChecklistItem(
                label=l,
                status=status,
                detail=f"P10={p10:.0f} · P50={median:.0f} · P90={p90:.0f} · Spread={spread:.0f}",
            )
            for l in self.CHECKLIST_LABELS
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={
                "p10": float(p10),
                "p50": median,
                "p90": float(p90),
                "spread": spread,
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
        median = float(m.get("p50", 128.0))
        spread = float(m.get("spread", 100.0))
        actions: list[str] = []
        params: dict[str, float] = {}
        out = image
        applied = False

        if spread < 100:
            out = _add_contrast(out, amount=0.18)
            actions.append("Mid-tone contrast boost")
            params["contrast_amount"] = 0.18
            applied = True
        # Composition agent runs *last*, so brightness nudges are subtle —
        # exposure agent already handled the heavy lifting.
        if median < 100 and not applied:
            out = cv2.add(out, np.full_like(out, 8, dtype=np.uint8))
            actions.append("Final +8 brightness lift")
            params["final_lift"] = 8.0
            applied = True

        return out, AgentApplyReport(
            applied=applied,
            actions=tuple(actions),
            params=params,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary if applied else "Composition already balanced.",
        )


def _add_contrast(image: np.ndarray, *, amount: float = 0.15) -> np.ndarray:
    img32 = image.astype(np.float32) / 255.0
    out = (img32 - 0.5) * (1.0 + amount) + 0.5
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
