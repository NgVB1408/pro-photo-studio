"""Tests for the auto-pilot — end-to-end scene-aware enhancement."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from pps_core.autopilot import AutoPilot, AutopilotReport, auto_enhance


def _interior(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 360, 540
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    img[: h // 3] = (200, 195, 188)
    img[2 * h // 3 :] = (90, 85, 75)
    cv2.rectangle(img, (180, 140), (380, 240), (40, 40, 40), thickness=3)
    return np.clip(
        img + rng.integers(-2, 3, size=img.shape, dtype=np.int8).astype(np.int16),
        0,
        255,
    ).astype(np.uint8)


def _exterior(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 480, 720
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 3] = (210, 160, 90)  # blue-ish sky
    img[h // 3 : 2 * h // 3] = (110, 140, 130)  # mid-ground
    img[2 * h // 3 :] = (60, 130, 70)  # grass
    img = img + rng.integers(-2, 3, size=img.shape, dtype=np.int8).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestAutoPilot:
    def test_returns_report_for_interior(self):
        out, report = auto_enhance(_interior(), scene="interior")
        assert isinstance(report, AutopilotReport)
        assert report.scene == "interior"
        assert out.shape == _interior().shape
        assert out.dtype == np.uint8
        assert "real_estate" in report.baseline_stages
        assert "enhance_studio" in report.baseline_stages
        # Studio review attached.
        assert report.studio.scene == "interior"
        assert len(report.studio.agents) == 9

    def test_returns_report_for_exterior(self):
        _, report = auto_enhance(_exterior(), scene="exterior")
        assert report.scene == "exterior"
        # Sky agent must be applicable for exterior scenes.
        sky_agent = next(a for a in report.studio.agents if a.role.startswith("Audits the sky"))
        assert sky_agent.before.metrics.get("applicable", 0.0) >= 1.0

    def test_scene_can_be_inferred(self):
        # Interior cue: top band is wall-like, no sky pattern.
        _, report = auto_enhance(_interior())
        assert report.scene in {"interior", "exterior", "aerial"}

    def test_serialisation(self):
        _, report = auto_enhance(_interior(), scene="interior")
        d = report.as_dict()
        assert d["scene"] == "interior"
        assert "studio" in d
        assert "agents" in d["studio"]
        assert "baseline_stages" in d
        assert isinstance(d["total_duration_ms"], float)

    def test_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            AutoPilot().run(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            AutoPilot().run(np.zeros((10, 10, 3), dtype=np.float32))

    def test_overall_grade_well_formed(self):
        _, report = auto_enhance(_interior(), scene="interior")
        assert report.grade in {"S", "A", "B", "C", "D"}
        assert 0.0 <= report.overall <= 10.0
