"""White Balance Specialist — neutralises colour casts."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import AgentApplyReport, AgentChecklistItem, AgentEvaluation


class WhiteBalanceAgent:
    name = "White Balance Specialist"
    role = "Removes colour casts so walls look neutral and woods look natural."
    category = "white_balance"

    CHECKLIST_LABELS: tuple[str, ...] = (
        "No warm cast (orange / red lift)",
        "No cool cast (blue / cyan lift)",
        "No green cast (mixed-light artefact)",
        "Mid-tones land on neutral grey",
    )

    def __init__(self, *, mild_threshold: float = 0.04, strong_threshold: float = 0.10) -> None:
        self.mild = mild_threshold
        self.strong = strong_threshold

    def evaluate(self, image: np.ndarray, *, scene: str) -> AgentEvaluation:
        del scene
        b_dev, g_dev, r_dev, cast = _channel_deviation(image)
        items = [
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[0],
                status=self._status(r_dev, cast),
                detail=f"R-channel deviation = {r_dev * 100:+.1f}%",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[1],
                status=self._status(b_dev, cast),
                detail=f"B-channel deviation = {b_dev * 100:+.1f}%",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[2],
                status=self._status(g_dev, cast),
                detail=f"G-channel deviation = {g_dev * 100:+.1f}%",
            ),
            AgentChecklistItem(
                label=self.CHECKLIST_LABELS[3],
                status="pass" if cast < self.mild else "warn" if cast < self.strong else "fail",
                detail=f"Overall cast magnitude = {cast:.3f}",
            ),
        ]
        deductions = sum(
            (1.2 if it.status == "warn" else 2.5 if it.status == "fail" else 0.0) for it in items
        )
        score = float(max(0.0, 10.0 - deductions))

        if cast < self.mild:
            summary = "White balance is neutral."
        elif r_dev > max(g_dev, b_dev) + 0.03:
            summary = "Warm cast — image leans orange/red."
        elif b_dev > max(r_dev, g_dev) + 0.03:
            summary = "Cool cast — image leans blue/cyan."
        elif g_dev > max(r_dev, b_dev) + 0.03:
            summary = "Green cast — typical of mixed fluorescent lighting."
        else:
            summary = "Mild colour cast that won't be obvious to most viewers."

        return AgentEvaluation(
            score=score,
            checklist=tuple(items),
            summary=summary,
            metrics={"cast_norm": cast, "b_dev": b_dev, "g_dev": g_dev, "r_dev": r_dev},
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
        cast = float(m.get("cast_norm", 0.0))
        if cast < self.mild:
            return image, AgentApplyReport(
                applied=False,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                notes="White balance already neutral.",
            )
        out = _gray_world_correct(image, strength=min(1.0, 0.6 + 4.0 * cast))
        b2, g2, r2, cast2 = _channel_deviation(out)
        return out, AgentApplyReport(
            applied=True,
            actions=("Robust gray-world correction on the 5–95 percentile band",),
            params={
                "before_cast": cast,
                "after_cast": cast2,
                "delta": cast - cast2,
            },
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes="Cast pulled from "
            f"{cast:.3f} → {cast2:.3f} (B={b2:+.2f}, G={g2:+.2f}, R={r2:+.2f}).",
        )

    def _status(self, dev: float, cast: float) -> str:
        if abs(dev) < 0.025 or cast < self.mild:
            return "pass"
        if abs(dev) < 0.06:
            return "warn"
        return "fail"


def _channel_deviation(image: np.ndarray) -> tuple[float, float, float, float]:
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y = yuv[..., 0]
    lo, hi = np.percentile(y, [40, 80])
    mask = (y >= lo) & (y <= hi)
    if mask.sum() < 1000:
        mask[:] = True
    pixels = image[mask].reshape(-1, 3)
    if pixels.size == 0:
        pixels = image.reshape(-1, 3)
    mean = pixels.mean(axis=0).astype(np.float64)
    grey = float(mean.mean())
    if grey < 1e-3:
        return 0.0, 0.0, 0.0, 0.0
    delta = mean / grey - 1.0
    b_dev, g_dev, r_dev = (float(d) for d in delta)
    return b_dev, g_dev, r_dev, float(np.linalg.norm(delta))


def _gray_world_correct(image: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """Robust per-channel gain so the 5–95% mid-tone block becomes neutral."""
    img32 = image.astype(np.float32)
    means = []
    for c in range(3):
        ch = img32[..., c]
        lo, hi = np.percentile(ch, [5, 95])
        mid = ch[(ch >= lo) & (ch <= hi)]
        means.append(float(mid.mean()) if mid.size else float(ch.mean()))
    target = float(np.mean(means))
    if target < 1.0:
        return image
    gains = [target / max(m, 1e-3) for m in means]
    # Damp by `strength` so we don't overcorrect on extreme inputs.
    gains = [1.0 + (g - 1.0) * strength for g in gains]
    out = img32.copy()
    for c, g in enumerate(gains):
        out[..., c] *= g
    return np.clip(out, 0, 255).astype(np.uint8)
