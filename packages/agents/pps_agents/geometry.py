"""Geometry & Lens specialist — SOP Phần 1.

Verticals 90°, horizontals, lens distortion (Brown-Conrady), chromatic
aberration on high-contrast edges.

Delegates to ``pps_core.realestate.correct_vertical`` (Hough vanishing point).
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseAgent
from .types import JobContext, StagePlan, StageReport


class GeometryAgent(BaseAgent):
    name = "geometry"

    def _analyze(self, ctx: JobContext) -> StagePlan:
        # Quick tilt + CA estimation. Heavy work happens in apply().
        img = ctx.image
        h, w = img.shape[:2]

        # Estimate tilt magnitude via gradient orientation histogram on
        # the longest verticals — we don't run the full upright algorithm
        # here (apply will). This is just for skip-or-not decision.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=int(min(h, w) * 0.18),
            maxLineGap=20,
        )
        tilt_score = 0.0
        n_vertical = 0
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                if abs(y2 - y1) < 5:
                    continue
                ang = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
                if ang < 25.0:
                    tilt_score += ang
                    n_vertical += 1
        tilt_avg = tilt_score / n_vertical if n_vertical else 0.0

        # Detect chromatic aberration on bright window edges:
        # measure per-channel Sobel response divergence at high-luminance edges.
        ca_strength = self._ca_strength(img)

        ops: list[dict] = []
        if n_vertical >= 4 and tilt_avg > 0.6:
            ops.append({"op": "upright_4point", "tilt_avg": float(tilt_avg)})
        if ca_strength > 0.6:
            ops.append({"op": "ca_correct", "strength": float(ca_strength)})

        skip = not ops
        return StagePlan(
            name=self.name,
            operations=ops,
            metadata={
                "tilt_avg_deg": float(tilt_avg),
                "n_vertical_lines": n_vertical,
                "ca_strength": float(ca_strength),
            },
            skip=skip,
            skip_reason="no_geometry_issues_detected" if skip else "",
        )

    @staticmethod
    def _ca_strength(img: np.ndarray) -> float:
        """Rough CA estimate: mean per-channel gradient divergence at bright edges.

        Returns ~0 for clean lens, >0.6 for visible fringing.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        if edges.sum() == 0:
            return 0.0
        b, g, r = cv2.split(img)
        sb = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        sg = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        sr = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
        e = edges > 0
        if not np.any(e):
            return 0.0
        # CA: r/b channels diverge from g at edge
        div = (np.abs(sr[e] - sg[e]) + np.abs(sb[e] - sg[e])).mean()
        return float(min(1.0, div / 60.0))

    def _apply(
        self, image: np.ndarray, plan: StagePlan
    ) -> tuple[np.ndarray, StageReport]:
        report = StageReport(name=self.name, metrics={})
        out = image
        for op in plan.operations:
            if op["op"] == "upright_4point":
                try:
                    from pps_core.realestate import correct_vertical

                    out, vr = correct_vertical(out)
                    report.metrics["upright"] = {
                        "applied": getattr(vr, "applied", False),
                        "skew": getattr(vr, "skew", 0.0),
                        "lines_used": getattr(vr, "lines_used", 0),
                    }
                except Exception as exc:
                    report.warnings.append(f"upright_failed: {exc!r}")
            elif op["op"] == "ca_correct":
                out = self._fix_chromatic_aberration(out)
                report.metrics["ca_correct"] = {
                    "strength": op["strength"],
                    "method": "channel_align_subpixel",
                }
        return out, report

    @staticmethod
    def _fix_chromatic_aberration(img: np.ndarray) -> np.ndarray:
        """Mild CA correction: scale R+B channels by 0.999/1.001 to align with G.

        Sub-pixel scale around image center reduces purple/green fringing on
        high-contrast edges (window frames typical case). Conservative — full
        CA correction needs lens profile.
        """
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        b, g, r = cv2.split(img)
        # Scale R toward center 0.999, B away 1.001
        m_r = cv2.getRotationMatrix2D((cx, cy), 0, 0.9995)
        m_b = cv2.getRotationMatrix2D((cx, cy), 0, 1.0005)
        r2 = cv2.warpAffine(r, m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b2 = cv2.warpAffine(b, m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.merge([b2, g, r2])
