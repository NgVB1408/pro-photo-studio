"""Tests for the quality auditor (pps_core.qc)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from pps_core.qc import QCReport, audit


def _well_lit_interior(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 480, 720
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    # Wall on top, floor on bottom — gives realistic histogram spread.
    img[: h // 3] = (200, 195, 188)
    img[2 * h // 3 :] = (90, 85, 75)
    # Picture frame to give the sharpness scorer some edges.
    cv2.rectangle(img, (220, 180), (500, 320), (40, 40, 40), thickness=3)
    img = img + rng.integers(-3, 4, size=img.shape, dtype=np.int8).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _blown_image() -> np.ndarray:
    img = _well_lit_interior()
    img[:240, :, :] = 252  # 50% blown
    return img


def _crushed_image() -> np.ndarray:
    img = _well_lit_interior()
    img[240:, :, :] = 2  # 50% crushed
    return img


def _warm_cast_image() -> np.ndarray:
    img = _well_lit_interior().astype(np.int16)
    img[..., 2] = np.clip(img[..., 2] + 28, 0, 255)
    return img.astype(np.uint8)


# ---------- API surface ----------


class TestAuditFaçade:
    def test_audit_returns_qcreport(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        assert isinstance(rpt, QCReport)
        assert 0.0 <= rpt.overall <= 10.0
        assert rpt.grade in {"S", "A", "B", "C", "D"}
        assert len(rpt.categories) == 9
        # Every category appears with the expected name.
        names = {c.name for c in rpt.categories}
        assert "exposure" in names
        assert "white_balance" in names
        assert "sharpness" in names
        assert "sky_quality" in names

    def test_audit_rejects_non_uint8(self):
        with pytest.raises(ValueError, match="uint8"):
            audit(np.zeros((10, 10, 3), dtype=np.float32), scene="interior")

    def test_audit_rejects_grayscale(self):
        with pytest.raises(ValueError, match="BGR"):
            audit(np.zeros((10, 10), dtype=np.uint8), scene="interior")

    def test_audit_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            audit(np.zeros((0, 0, 3), dtype=np.uint8), scene="interior")


# ---------- Category-specific behaviour ----------


class TestExposureScoring:
    def test_well_lit_scores_high(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        exp = next(c for c in rpt.categories if c.name == "exposure")
        assert exp.score >= 7.5, f"expected ≥ 7.5, got {exp.score}: {exp.finding}"

    def test_blown_image_scores_low(self):
        rpt = audit(_blown_image(), scene="interior")
        exp = next(c for c in rpt.categories if c.name == "exposure")
        assert exp.score <= 5.5, f"blown image scored too high: {exp.score}"
        assert "blown" in exp.finding.lower()

    def test_crushed_image_scores_low(self):
        rpt = audit(_crushed_image(), scene="interior")
        exp = next(c for c in rpt.categories if c.name == "exposure")
        assert exp.score <= 6.5
        assert "shadow" in exp.finding.lower() or "crushed" in exp.finding.lower()


class TestWhiteBalanceScoring:
    def test_neutral_scores_high(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        wb = next(c for c in rpt.categories if c.name == "white_balance")
        assert wb.score >= 7.5

    def test_warm_cast_detected(self):
        rpt = audit(_warm_cast_image(), scene="interior")
        wb = next(c for c in rpt.categories if c.name == "white_balance")
        assert wb.score < 7.5
        assert "warm" in wb.finding.lower() or "red" in wb.finding.lower()


class TestSkyGate:
    def test_sky_skipped_for_interior(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        sky = next(c for c in rpt.categories if c.name == "sky_quality")
        assert sky.applicable is False
        # Skipped category should not pull down overall — weight should be 0.
        assert sky.weight == 0.0

    def test_sky_evaluated_for_exterior(self):
        # Build a synthetic exterior with a clean blue sky on top.
        h, w = 600, 900
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (60, 120, 180)  # ground / building
        img[: h // 3] = (210, 160, 90)  # sky band (BGR — light blue-ish)
        rpt = audit(img, scene="exterior")
        sky = next(c for c in rpt.categories if c.name == "sky_quality")
        assert sky.applicable is True


# ---------- Aggregation + grade ----------


class TestAggregation:
    def test_overall_is_weighted_mean(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        applicable = [c for c in rpt.categories if c.applicable and c.weight > 0]
        manual = sum(c.score * c.weight for c in applicable) / sum(c.weight for c in applicable)
        assert abs(rpt.overall - manual) < 0.05

    def test_recommendations_target_weakest(self):
        rpt = audit(_blown_image(), scene="interior")
        assert rpt.recommendations, "expected at least one recommendation for a blown image"
        # Each recommendation should match one of the weak categories.
        weak_names = {
            c.name.replace("_", " ").capitalize()
            for c in rpt.categories
            if c.applicable and c.score < 8.0
        }
        assert all(any(name in r for name in weak_names) for r in rpt.recommendations)

    def test_serialisation_round_trip(self):
        rpt = audit(_well_lit_interior(), scene="interior")
        d = rpt.as_dict()
        assert "overall" in d
        assert "grade" in d
        assert "categories" in d
        assert len(d["categories"]) == 9
        for c in d["categories"]:
            assert "name" in c
            assert "score" in c
            assert "applicable" in c
