"""Sky Specialist — checks the sky for banding / clipping; only active outdoors."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class SkyAgent:
    name = "Sky Specialist"
    role = "Audits the sky for banding, clipping, and noise; replaces or reseats when needed."
    category = "sky_quality"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "Sky is not blown / clipped",
        "Gradient is smooth (no banding)",
        "Sky tone matches the lighting on the building",
    )

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        if scene not in ("exterior", "aerial"):
            return AgentEvaluation(
                score=10.0,
                checklist=tuple(
                    AgentChecklistItem(label=l, status="pass", detail="No sky in this scene")
                    for l in self.CHECKLIST_LABELS
                ),
                summary="Interior shot — sky check skipped.",
                metrics={"applicable": 0.0},
            )
        h, _w = image.shape[:2]
        top = image[: max(1, h // 4)]
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(np.float32) / 255.0
        v = hsv[..., 2].astype(np.float32) / 255.0
        mask = (s < 0.45) & (v > 0.55)
        sky_ratio = float(mask.mean())
        if sky_ratio < 0.05:
            return AgentEvaluation(
                score=8.5,
                checklist=tuple(
                    AgentChecklistItem(
                        label=l,
                        status="pass",
                        detail="Sky region too small to evaluate",
                    )
                    for l in self.CHECKLIST_LABELS
                ),
                summary="Sky region too small to evaluate banding.",
                metrics={"sky_ratio": sky_ratio, "applicable": 1.0},
            )
        blue = top[..., 0][mask].astype(np.float32)
        sky_std = float(np.std(blue))
        sky_clip = float((blue >= 254).mean())

        if sky_clip > 0.05:
            verdict = "fail"
            summary = f"Sky is clipped on {sky_clip * 100:.1f}% of pixels — recover or replace."
        elif sky_std < 1.5:
            verdict = "warn"
            summary = "Posterising in the gradient — replace with a clean sky preset."
        elif sky_std > 35:
            verdict = "warn"
            summary = "Sky is noisy; apply targeted denoise."
        else:
            verdict = "pass"
            summary = "Sky looks clean and gradient-smooth."

        items = (
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[0],
                status="pass" if sky_clip <= 0.02 else "warn" if sky_clip <= 0.05 else "fail",
                detail=f"{sky_clip * 100:.1f}% of sky pixels at 254+",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[1],
                status="pass" if sky_std >= 1.5 else "warn",
                detail=f"Blue-channel std = {sky_std:.1f}",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[2],
                status=verdict,
                detail=summary,
            ),
        )
        score = float(
            np.clip(
                10.0 - 5.0 * sky_clip / 0.05 - max(0.0, 1.5 - sky_std) * 1.5
                - max(0.0, sky_std - 35.0) * 0.05,
                0.0,
                10.0,
            )
        )
        return AgentEvaluation(
            score=score,
            checklist=items,
            summary=summary,
            metrics={
                "sky_ratio": sky_ratio,
                "sky_std": sky_std,
                "sky_clip_pct": sky_clip,
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
        if m.get("applicable", 1.0) <= 0:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="Indoor scene — no sky to fix.",
            )
        sky_clip = float(m.get("sky_clip_pct", 0.0))
        sky_std = float(m.get("sky_std", 10.0))
        if sky_clip < 0.02 and sky_std >= 1.5:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="Sky is already clean.",
            )
        # Recover clipped sky by gentle compression of the top decile of luminance.
        out = _sky_recover(image, sky_clip=sky_clip)
        return out, AgentApplyReport(
            applied=True,
            actions=("Sky highlight compression",),
            params={"sky_clip_before": sky_clip, "sky_std_before": sky_std},
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=evaluation.summary,
        )


def _sky_recover(image: np.ndarray, *, sky_clip: float) -> np.ndarray:
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
    y = yuv[..., 0]
    threshold = np.percentile(y, 88)
    mask = y >= threshold
    if not mask.any():
        return image
    headroom = 255.0 - y[mask]
    pull = float(min(0.55, 8.0 * sky_clip + 0.18))
    yuv[..., 0][mask] = y[mask] - headroom * pull
    yuv[..., 0] = np.clip(yuv[..., 0], 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
