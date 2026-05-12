"""Director / QC agent — final gate before user review.

Encodes the SOP "self-questions" the user defined and the standard SOP
checklist items into measurable, deterministic checks. Output is a
``DirectorReview`` with a numeric score per question, a verdict, and the
list of findings + recommendations the user reads.

Self-questions (mapped to scorers below):

  Q1 — Halo @ 200% zoom around windows  → ``q1_halo_window_corners``
  Q2 — Ceiling truly neutral (not bluish) → ``q2_ceiling_neutrality``
  Q3 — Move-in feel                       → ``q3_move_in_feel``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DirectorReview:
    verdict: str  # "pass" | "review" | "fail"
    overall_score: float  # 0..1
    question_scores: dict[str, float] = field(default_factory=dict)
    sop_scores: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "overall_score": round(self.overall_score, 3),
            "question_scores": {k: round(v, 3) for k, v in self.question_scores.items()},
            "sop_scores": {k: round(v, 3) for k, v in self.sop_scores.items()},
            "findings": self.findings,
            "recommendations": self.recommendations,
        }


class DirectorAgent:
    """Read-only QC. Does NOT modify the image — only inspects + reports."""

    name = "director"

    PASS_THRESHOLD = 0.78
    REVIEW_THRESHOLD = 0.55

    def review(
        self, original: np.ndarray, final: np.ndarray
    ) -> DirectorReview:
        q_scores = {
            "Q1_halo_window_corners": self.q1_halo_window_corners(final),
            "Q2_ceiling_neutrality": self.q2_ceiling_neutrality(final),
            "Q3_move_in_feel": self.q3_move_in_feel(final),
        }

        sop_scores = {
            "verticals_90deg": self.sop_verticals(final),
            "lens_distortion_residual": self.sop_lens_residual(final),
            "sharpness_uniformity": self.sop_sharpness(final),
            "shadow_noise": self.sop_shadow_noise(final),
            "consistency_vs_input": self.sop_consistency(original, final),
        }

        findings: list[str] = []
        recs: list[str] = []

        if q_scores["Q1_halo_window_corners"] < 0.7:
            findings.append("Có khả năng halo/lem ở rìa cửa sổ khi phóng 200%.")
            recs.append(
                "Giảm highlight_recovery xuống 0.25–0.30; tăng halo_feather sigma trên blown_window mask."
            )
        if q_scores["Q2_ceiling_neutrality"] < 0.7:
            findings.append("Trần/tường trắng còn ám (không thật neutral).")
            recs.append(
                "Bật indoor_pro.selective_wall_wb với strength 0.7; đo lại LAB AB của vùng V>200, S<25."
            )
        if q_scores["Q3_move_in_feel"] < 0.7:
            findings.append("Cảm giác 'muốn dời vào ở' yếu (thiếu dynamic range hoặc tone bệt).")
            recs.append(
                "Áp tone_coherency preset theo property_type; tăng vibrance 0.05; check shadow_lift."
            )

        if sop_scores["verticals_90deg"] < 0.7:
            findings.append("Đường dọc còn nghiêng > 1°.")
            recs.append("Re-run GeometryAgent với upright_4point ngưỡng tilt thấp hơn.")
        if sop_scores["sharpness_uniformity"] < 0.65:
            findings.append("Sharpness không đều — vùng tiền cảnh nhoè hoặc hậu cảnh quá gắt.")
            recs.append("Hạ saliency_sharpen amount xuống 0.4; tăng saliency_blur 71.")
        if sop_scores["shadow_noise"] < 0.65:
            findings.append("Noise vùng tối còn rõ.")
            recs.append("Tăng OutputAgent.shadow_denoise strength lên 7–10 trước khi upscale.")
        if sop_scores["consistency_vs_input"] < 0.55:
            findings.append("Khác xa ảnh gốc về tổng thể (PSNR thấp); nguy cơ over-process.")
            recs.append("Hạ texture macro band 0.18→0.10; reduce dehaze 0.5x.")

        weights_q = {"Q1_halo_window_corners": 0.30, "Q2_ceiling_neutrality": 0.25,
                     "Q3_move_in_feel": 0.45}
        weights_sop = {"verticals_90deg": 0.15, "lens_distortion_residual": 0.10,
                       "sharpness_uniformity": 0.20, "shadow_noise": 0.20,
                       "consistency_vs_input": 0.35}
        q_part = sum(q_scores[k] * w for k, w in weights_q.items())
        sop_part = sum(sop_scores[k] * w for k, w in weights_sop.items())
        overall = 0.55 * q_part + 0.45 * sop_part

        if overall >= self.PASS_THRESHOLD:
            verdict = "pass"
        elif overall >= self.REVIEW_THRESHOLD:
            verdict = "review"
        else:
            verdict = "fail"

        return DirectorReview(
            verdict=verdict,
            overall_score=overall,
            question_scores=q_scores,
            sop_scores=sop_scores,
            findings=findings,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Q1 — halo at window corners (200% zoom)
    # ------------------------------------------------------------------

    @staticmethod
    def q1_halo_window_corners(img: np.ndarray) -> float:
        """Look for bright haloing at edges of high-luminance windows.

        Halo signature: bright pixel → sudden mid-luminance ring → darker pixel.
        We compute the gradient at edges adjacent to V>240 regions and check for
        an over-brightened local mean inside a 7-pixel band immediately outside.
        Score = 1 - (halo_severity).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright = (gray > 240).astype(np.uint8)
        if bright.sum() < 200:
            return 1.0
        # Outer ring 2..7 px
        outer = cv2.dilate(bright, np.ones((15, 15), np.uint8)) - cv2.dilate(
            bright, np.ones((3, 3), np.uint8)
        )
        outer = outer > 0
        if outer.sum() < 100:
            return 1.0
        # Mean luminance in ring vs. in image overall
        ring_mean = float(gray[outer].mean())
        img_mean = float(gray.mean())
        # Halo = ring brighter than expected by >30 grey levels
        excess = max(0.0, ring_mean - img_mean - 30.0)
        severity = min(1.0, excess / 40.0)
        return float(1.0 - severity)

    # ------------------------------------------------------------------
    # Q2 — ceiling neutrality
    # ------------------------------------------------------------------

    @staticmethod
    def q2_ceiling_neutrality(img: np.ndarray) -> float:
        """Top 25% bright-low-sat region should have LAB (a, b) ~ (128, 128).

        Score = 1 if |a-128| + |b-128| <= 4, dropping linearly to 0 at delta=20.
        """
        h = img.shape[0]
        top = img[: int(h * 0.25)]
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        V = hsv[..., 2]
        S = hsv[..., 1]
        wall = (V >= 180) & (S <= 35)
        if wall.sum() < 200:
            return 0.85  # not enough sample → assume OK
        lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_mean = float(lab[..., 1][wall].mean())
        b_mean = float(lab[..., 2][wall].mean())
        delta = abs(a_mean - 128.0) + abs(b_mean - 128.0)
        score = 1.0 - max(0.0, (delta - 4.0) / 16.0)
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Q3 — move-in feel (composite proxy)
    # ------------------------------------------------------------------

    @staticmethod
    def q3_move_in_feel(img: np.ndarray) -> float:
        """Proxy for the subjective 'I want to move in now' feel:

          * dynamic range p1..p99 ~ 150-235  → 1.0, fall off outside
          * not clipped (clip_high < 0.5%, clip_low < 1%)
          * vibrance: HSV S median in [25, 90]
          * mid-tone contrast (std of L in midband) in [25, 55]

        Returns weighted score in [0, 1].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p1 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))
        dyn = p99 - p1
        clip_high = float((gray >= 250).mean())
        clip_low = float((gray <= 5).mean())

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_med = float(np.median(hsv[..., 1]))
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
        mid_mask = (L > 60) & (L < 200)
        mid_std = float(L[mid_mask].std()) if mid_mask.sum() else 0.0

        # Score components — each in [0, 1]
        dyn_score = 1.0 - min(1.0, abs(dyn - 200.0) / 100.0)
        clip_score = 1.0 - min(1.0, (clip_high * 50) + (clip_low * 30))
        vib_score = 1.0 - min(1.0, max(0.0, (25 - s_med)) / 25.0 + max(0.0, (s_med - 90)) / 90.0)
        mid_score = 1.0 - min(1.0, abs(mid_std - 40.0) / 30.0)

        return float(
            0.35 * dyn_score + 0.25 * clip_score + 0.20 * vib_score + 0.20 * mid_score
        )

    # ------------------------------------------------------------------
    # SOP scorers
    # ------------------------------------------------------------------

    @staticmethod
    def sop_verticals(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        h, w = gray.shape
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=int(min(h, w) * 0.18), maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            return 0.85
        verts = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if abs(y2 - y1) < 5:
                continue
            ang = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
            if ang < 20:
                verts.append(ang)
        if not verts:
            return 0.85
        avg = float(np.mean(verts))
        return float(np.clip(1.0 - avg / 5.0, 0.0, 1.0))

    @staticmethod
    def sop_lens_residual(img: np.ndarray) -> float:
        """Edge straightness deviation as a residual lens-distortion proxy.
        We sample horizontal/vertical strong edges and check linearity.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        # Hough straight line score: more long lines = lower distortion
        h, w = gray.shape
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=120,
            minLineLength=int(min(h, w) * 0.25), maxLineGap=10,
        )
        n = 0 if lines is None else len(lines)
        return float(np.clip(n / 25.0, 0.0, 1.0))

    @staticmethod
    def sop_sharpness(img: np.ndarray) -> float:
        """Variance of Laplacian — uniformity across 4 image quadrants."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        q = [
            gray[: h // 2, : w // 2],
            gray[: h // 2, w // 2 :],
            gray[h // 2 :, : w // 2],
            gray[h // 2 :, w // 2 :],
        ]
        sharps = [float(cv2.Laplacian(qi, cv2.CV_64F).var()) for qi in q]
        if max(sharps) < 1.0:
            return 0.0
        # Min/max ratio = uniformity; absolute min = enough sharpness
        ratio = min(sharps) / max(sharps)
        floor = min(1.0, min(sharps) / 80.0)
        return float(0.5 * ratio + 0.5 * floor)

    @staticmethod
    def sop_shadow_noise(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark = gray < 60
        if dark.sum() < 200:
            return 1.0
        std = float(gray[dark].std())
        return float(np.clip(1.0 - max(0.0, std - 5.0) / 12.0, 0.0, 1.0))

    @staticmethod
    def sop_consistency(orig: np.ndarray, final: np.ndarray) -> float:
        """High score = final tracks original well (no over-process)."""
        if orig.shape != final.shape:
            final_rs = cv2.resize(final, (orig.shape[1], orig.shape[0]))
        else:
            final_rs = final
        try:
            from pps_core.quality import psnr

            p = psnr(orig, final_rs)
        except Exception:
            mse = float(np.mean((orig.astype(np.float64) - final_rs.astype(np.float64)) ** 2))
            p = 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
        # 25 dB ≈ aggressive, 35 dB ≈ subtle
        return float(np.clip((p - 22) / 16.0, 0.0, 1.0))
