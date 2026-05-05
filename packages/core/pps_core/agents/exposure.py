"""Exposure Specialist — recovers blown highlights and crushed shadows."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import (
    AgentApplyReport,
    AgentChecklistItem,
    AgentEvaluation,
)


class ExposureAgent:
    name = "Exposure Specialist"
    role = "Recovers blown skies, lifts shadow detail, and balances dynamic range."
    category = "exposure"

    # Public checklist — drives the scorecard on the web portal.
    CHECKLIST_LABELS: tuple[str, ...] = (
        "No blown highlights (>250)",
        "No crushed shadows (<5)",
        "Tonal range covers the full histogram",
        "Mid-tone exposure within professional range",
    )

    def __init__(self, *, blown_warn: float = 0.005, crushed_warn: float = 0.02) -> None:
        self.blown_warn = blown_warn
        self.crushed_warn = crushed_warn

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels = gray.size
        blown = float((gray >= 250).sum() / pixels)
        crushed = float((gray <= 5).sum() / pixels)
        p1, p99 = np.percentile(gray, [1, 99])
        median = float(np.median(gray))
        spread = float(p99 - p1)

        items = [
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[0],
                status="pass" if blown < self.blown_warn else "warn" if blown < 0.02 else "fail",
                detail=f"{blown * 100:.2f}% of pixels above 250",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[1],
                status="pass" if crushed < self.crushed_warn else "warn" if crushed < 0.05 else "fail",
                detail=f"{crushed * 100:.2f}% of pixels below 5",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[2],
                status="pass" if spread >= 200 else "warn" if spread >= 160 else "fail",
                detail=f"Histogram spread = {spread:.0f} (target ≥ 200)",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[3],
                status="pass" if 100 <= median <= 165 else "warn",
                detail=f"Median luminance = {median:.0f} (target 100–165)",
            ),
        ]

        # Score: 10 if all pass; deduct for each warn/fail.
        deductions = 0.0
        for it in items:
            if it.status == "warn":
                deductions += 1.2
            elif it.status == "fail":
                deductions += 2.5
        score = float(max(0.0, 10.0 - deductions))

        if blown >= 0.02:
            summary = "Aggressive highlight clipping — recover detail before sharpening."
        elif crushed >= 0.05:
            summary = "Heavy shadow clipping — open the shadows by ~30%."
        elif spread < 160:
            summary = "Tonal range is narrow; the image will look flat in print."
        else:
            summary = "Exposure is well-balanced for a hero photo."

        return AgentEvaluation(
            score=score,
            checklist=tuple(items),
            summary=summary,
            metrics={
                "blown_pct": blown,
                "crushed_pct": crushed,
                "p1": float(p1),
                "p99": float(p99),
                "spread": spread,
                "median": median,
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
        blown = float(m.get("blown_pct", 0.0))
        crushed = float(m.get("crushed_pct", 0.0))
        median = float(m.get("median", 128.0))

        actions: list[str] = []
        params: dict[str, float] = {}
        out = image
        applied = False

        # Highlight recovery — gamma compress on top decile then re-stretch.
        if blown >= self.blown_warn * 2:  # 1%+
            strength = float(min(0.6, 18.0 * blown))  # 0..0.6
            out = _highlight_recovery(out, strength=strength)
            actions.append(f"Recovered highlights with strength {strength:.2f}")
            params["highlight_strength"] = strength
            applied = True

        # Shadow lift — adaptive based on crushed percentage.
        if crushed >= self.crushed_warn * 1.5:
            lift = float(min(0.45, 8.0 * crushed))
            out = _shadow_lift(out, amount=lift)
            actions.append(f"Lifted shadows by {lift:.2f}")
            params["shadow_lift"] = lift
            applied = True

        # Global brightness nudge — bring median into target band.
        if median < 95:
            delta = float(min(20, 95 - median))
            out = cv2.add(out, np.full_like(out, int(delta), dtype=np.uint8))
            actions.append(f"Lifted global brightness by {delta:.0f}")
            params["global_lift"] = delta
            applied = True
        elif median > 175:
            delta = float(min(15, median - 175))
            out = cv2.subtract(out, np.full_like(out, int(delta), dtype=np.uint8))
            actions.append(f"Pulled global brightness down by {delta:.0f}")
            params["global_pull"] = delta
            applied = True

        notes = "Within target — no intervention needed." if not applied else evaluation.summary
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return out, AgentApplyReport(
            applied=applied,
            actions=tuple(actions),
            params=params,
            duration_ms=duration_ms,
            notes=notes,
        )


def _highlight_recovery(img: np.ndarray, *, strength: float) -> np.ndarray:
    """Compress the top decile of luminance without crushing mid-tones."""
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
    y = yuv[..., 0]
    threshold = np.percentile(y, 90)
    mask = y >= threshold
    if not mask.any():
        return img
    headroom = 255.0 - y[mask]
    yuv[..., 0][mask] = y[mask] + headroom * (-strength)  # pull bright pixels down
    yuv[..., 0] = np.clip(yuv[..., 0], 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def _shadow_lift(img: np.ndarray, *, amount: float) -> np.ndarray:
    """Lift the bottom 30% of luminance by ``amount`` without halos."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    threshold = np.percentile(l, 30)
    mask = l < threshold
    if not mask.any():
        return img
    lab[..., 0][mask] = l[mask] + (threshold - l[mask]) * amount
    lab[..., 0] = np.clip(lab[..., 0], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
