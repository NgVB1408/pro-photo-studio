"""AI Supervisor — chấm chất lượng masks tự động.

7 metrics + verdict:
    boundary_smoothness     — biên không zig-zag (gradient curvature score)
    coverage_sanity         — wall 30-70%, floor 5-30%, ceiling 5-25%, etc.
    no_orphan_blobs         — không có blob nhỏ rời (single connected component dominant)
    edge_alignment          — biên mask align với Canny edge của ảnh gốc
    hole_rate               — mask không có lỗ random bên trong
    soft_alpha_quality      — biên có gradient mượt (không cứng pixel-step)
    inter_class_overlap     — wall ∩ ceiling ≈ 0 (mask exclusive)

Verdict:
    pass    ≥ 0.85 → giao khách
    review  0.65-0.84 → mở Photoshop check trước
    fail    < 0.65 → re-run với params strict hơn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

EXPECTED_COVERAGE = {
    "wall":      (0.20, 0.75),
    "floor":     (0.03, 0.40),
    "ceiling":   (0.02, 0.30),
    "window":    (0.00, 0.40),
    "door":      (0.00, 0.30),
    "opening":   (0.00, 0.40),
    "crown":     (0.00, 0.05),
    "baseboard": (0.00, 0.05),
    "casing":    (0.00, 0.08),
    "light":     (0.00, 0.10),
}


@dataclass
class MaskScore:
    name: str
    coverage: float
    metrics: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    verdict: str = "?"  # pass / review / fail / no_target
    issues: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    per_mask: dict[str, MaskScore] = field(default_factory=dict)
    overall_score: float = 0.0
    verdict: str = "?"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "verdict": self.verdict,
            "recommendations": self.recommendations,
            "per_mask": {
                k: {
                    "coverage": round(v.coverage, 4),
                    "overall": round(v.overall, 3),
                    "verdict": v.verdict,
                    "issues": v.issues,
                    "metrics": {m: round(s, 3) for m, s in v.metrics.items()},
                }
                for k, v in self.per_mask.items()
            },
        }


def _coverage_sanity(name: str, cov: float) -> float:
    """Score 0..1 dựa vào coverage range expected cho class."""
    lo, hi = EXPECTED_COVERAGE.get(name, (0.0, 1.0))
    if lo <= cov <= hi:
        return 1.0
    if cov < lo:
        # under-coverage: phạt tăng dần
        deficit = (lo - cov) / max(lo, 0.01)
        return max(0.0, 1.0 - min(1.0, deficit))
    # over-coverage
    excess = (cov - hi) / max(hi, 0.01)
    return max(0.0, 1.0 - min(1.0, excess * 0.5))


def _boundary_smoothness(mask: np.ndarray) -> float:
    """Đo độ mượt biên: ratio (perimeter / sqrt(area))² so với hình tròn lý tưởng.

    Hình tròn → ratio = 4π. Zig-zag mask → ratio >> 4π.
    Score = clip(1 - normalized_excess, 0, 1).
    """
    binary = (mask > 128).astype(np.uint8)
    if binary.sum() < 100:
        return 1.0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
    perim = sum(cv2.arcLength(c, True) for c in contours)
    area = binary.sum()
    if area == 0:
        return 1.0
    isoperimetric = (perim * perim) / (4.0 * np.pi * area)
    # Soft normalize: 1.0 = perfect circle, ~5 = blob, ~30 = very rough
    score = np.clip(1.0 - (isoperimetric - 2.0) / 20.0, 0.0, 1.0)
    return float(score)


def _orphan_blob_score(mask: np.ndarray) -> float:
    """Score = ratio area của largest component / total area."""
    binary = (mask > 128).astype(np.uint8)
    if binary.sum() < 100:
        return 1.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 1.0
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if sizes.size == 0:
        return 1.0
    return float(sizes.max() / sizes.sum())


def _edge_alignment(image_bgr: np.ndarray, mask: np.ndarray, band_px: int = 8) -> float:
    """Boundary F-score xấp xỉ: tỷ lệ biên mask trùng với Canny edge gốc."""
    binary = (mask > 128).astype(np.uint8)
    if binary.sum() < 100:
        return 1.0
    mask_edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if mask_edge.sum() == 0:
        return 1.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1,) * 2)
    img_edges_d = cv2.dilate(img_edges, k, iterations=1)
    matched = cv2.bitwise_and(mask_edge, (img_edges_d > 0).astype(np.uint8))
    return float(matched.sum() / max(1, mask_edge.sum()))


def _hole_rate(mask: np.ndarray) -> float:
    """Score 0..1: 1 = không có lỗ; <1 = mask có hole inside."""
    binary = (mask > 128).astype(np.uint8)
    if binary.sum() < 100:
        return 1.0
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    hole_area = (filled - binary).sum()
    total = filled.sum()
    return float(1.0 - hole_area / max(1, total))


def _soft_alpha_quality(mask: np.ndarray) -> float:
    """Score dựa vào tỷ lệ pixel có alpha trung gian (0<a<255) — đại diện cho biên feather."""
    inter = ((mask > 0) & (mask < 255)).sum()
    total = (mask > 0).sum()
    if total == 0:
        return 1.0
    ratio = inter / total
    # Lý tưởng: 1-10% pixels là intermediate (biên feather)
    if 0.005 <= ratio <= 0.20:
        return 1.0
    if ratio < 0.005:
        return max(0.5, 1.0 - (0.005 - ratio) * 50)
    return max(0.5, 1.0 - (ratio - 0.20) * 2)


def _inter_class_overlap(masks: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Đo overlap (IoU) giữa wall/floor/ceiling — phải gần 0."""
    keys = ["wall", "floor", "ceiling"]
    pairs: dict[tuple[str, str], float] = {}
    bins = {k: (masks[k] > 128).astype(np.uint8) for k in keys if k in masks}
    for i, k1 in enumerate(bins):
        for k2 in list(bins.keys())[i + 1:]:
            inter = (bins[k1] & bins[k2]).sum()
            union = (bins[k1] | bins[k2]).sum()
            iou = inter / max(1, union)
            pairs[(k1, k2)] = float(iou)
    return pairs


def evaluate_masks(image_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> EvalReport:
    """Chấm tất cả masks. Trả EvalReport."""
    report = EvalReport()
    total_w = 0.0
    weighted_sum = 0.0
    recs: list[str] = []

    # Weight class quan trọng cho khách BĐS
    class_weights = {
        "wall": 1.5, "floor": 1.2, "ceiling": 1.0,
        "window": 1.3, "door": 1.0, "opening": 1.3,
        "crown": 0.5, "baseboard": 0.5, "casing": 0.7, "light": 0.3,
    }

    for name, mask in masks.items():
        score = MaskScore(name=name, coverage=float((mask > 128).mean()))

        # Skip empty
        if (mask > 0).sum() == 0:
            score.verdict = "no_target"
            score.overall = 0.0
            report.per_mask[name] = score
            continue

        m_cov = _coverage_sanity(name, score.coverage)
        m_smooth = _boundary_smoothness(mask)
        m_orphan = _orphan_blob_score(mask)
        m_edge = _edge_alignment(image_bgr, mask, band_px=8)
        m_hole = _hole_rate(mask)
        m_soft = _soft_alpha_quality(mask)

        score.metrics = {
            "coverage_sanity": m_cov,
            "boundary_smoothness": m_smooth,
            "no_orphan_blobs": m_orphan,
            "edge_alignment": m_edge,
            "hole_rate": m_hole,
            "soft_alpha_quality": m_soft,
        }

        # Weighted overall
        weights = {
            "coverage_sanity": 0.20,
            "boundary_smoothness": 0.15,
            "no_orphan_blobs": 0.15,
            "edge_alignment": 0.25,
            "hole_rate": 0.15,
            "soft_alpha_quality": 0.10,
        }
        score.overall = sum(score.metrics[k] * w for k, w in weights.items())

        # Issues
        if m_cov < 0.5:
            score.issues.append(f"coverage {score.coverage*100:.1f}% nằm ngoài expected range")
        if m_smooth < 0.5:
            score.issues.append("biên zig-zag → tăng refine_edges hoặc matting band")
        if m_orphan < 0.6:
            score.issues.append("có nhiều blob rời → relax close_px hoặc filter components")
        if m_edge < 0.4:
            score.issues.append("biên mask không align với edge ảnh → mất chi tiết")
        if m_hole < 0.85:
            score.issues.append("có lỗ trong mask → fill holes")

        # Verdict per mask
        if score.overall >= 0.85:
            score.verdict = "pass"
        elif score.overall >= 0.65:
            score.verdict = "review"
        else:
            score.verdict = "fail"

        report.per_mask[name] = score
        cw = class_weights.get(name, 1.0)
        weighted_sum += score.overall * cw
        total_w += cw

    # Inter-class overlap penalty (wall ∩ floor etc.)
    overlaps = _inter_class_overlap(masks)
    overlap_penalty = 0.0
    for (a, b), iou in overlaps.items():
        if iou > 0.05:
            overlap_penalty += iou
            recs.append(f"⚠️ {a} ∩ {b} IoU={iou:.2f} → mask không exclusive, kiểm Photoshop")

    report.overall_score = max(0.0, weighted_sum / max(1, total_w) - overlap_penalty)

    if report.overall_score >= 0.85:
        report.verdict = "pass"
    elif report.overall_score >= 0.65:
        report.verdict = "review"
    else:
        report.verdict = "fail"

    # Aggregate recommendations từ issues
    for name, score in report.per_mask.items():
        if score.issues:
            recs.append(f"[{name}] " + " / ".join(score.issues))

    report.recommendations = recs
    return report
