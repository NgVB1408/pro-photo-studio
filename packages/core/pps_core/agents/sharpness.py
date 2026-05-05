"""Sharpness Specialist — adds halo-free local detail."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class SharpnessAgent:
    name = "Sharpness Specialist"
    role = "Adds crisp local detail without producing halos around edges."
    category = "sharpness"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Edges feel crisp at viewing distance",
        "Detail is balanced (not soft, not over-cooked)",
        "Texture preserved on stone / wood / fabric",
    )

    def __init__(self, *, target_lapvar_at_1080p: float = 60.0) -> None:
        self.target = target_lapvar_at_1080p

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        lap_var = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var())
        target = self.target * (h * w / (1920 * 1080)) ** 0.45
        ratio = lap_var / max(target, 1e-3)

        crispness_status = "pass" if ratio >= 0.7 else "warn" if ratio >= 0.4 else "fail"
        balance_status = "pass" if 0.5 <= ratio <= 4.0 else "warn"
        texture_status = "pass" if 0.4 <= ratio <= 5.0 else "warn"
        items = [
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[0],
                status=crispness_status,
                detail=f"Laplacian variance = {lap_var:.0f}, target {target:.0f}",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[1],
                status=balance_status,
                detail=f"Sharpness ratio = {ratio:.2f} (sweet spot 0.7–2.5)",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[2],
                status=texture_status,
                detail="Texture intact" if ratio <= 5.0 else "Texture risks looking plasticky",
            ),
        ]
        score = float(np.clip(5.0 + 5.0 * np.tanh(np.log(max(ratio, 1e-3))), 0.0, 10.0))
        summary = (
            "Image is soft — local detail boost recommended."
            if ratio < 0.5
            else "Detail is overcooked — pull back sharpening."
            if ratio > 4.0
            else "Sharpness is in the professional range."
        )
        return AgentEvaluation(
            score=score,
            checklist=tuple(items),
            summary=summary,
            metrics={"lap_var": lap_var, "target": target, "ratio": float(ratio)},
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
        ratio = float(evaluation.metrics.get("ratio", 1.0))
        out = image
        actions: list[str] = []
        params: dict[str, float] = {}
        applied = False

        if ratio < 0.6:
            amount = float(min(0.55, 0.7 - ratio))
            out = _local_detail_enhance(out, amount=amount)
            actions.append(f"Halo-free local detail (guided filter), amount {amount:.2f}")
            params["amount"] = amount
            applied = True
        elif ratio > 3.5:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.6)
            actions.append("Mild Gaussian blur to back off over-sharpening")
            params["blur_sigma"] = 0.6
            applied = True

        return out, AgentApplyReport(
            applied=applied,
            actions=tuple(actions),
            params=params,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary if applied else "Sharpness already on target.",
        )


def _local_detail_enhance(image: np.ndarray, *, amount: float = 0.35) -> np.ndarray:
    """Halo-free detail boost via a guided-filter base/detail split."""
    img32 = image.astype(np.float32)
    base = cv2.bilateralFilter(image, d=9, sigmaColor=35, sigmaSpace=9).astype(np.float32)
    detail = img32 - base
    boosted = base + detail * (1.0 + amount)
    return np.clip(boosted, 0, 255).astype(np.uint8)
