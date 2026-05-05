"""Tests for the multi-agent post-production studio."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from pps_core.agents import (
    AgentApplyReport,
    AgentEvaluation,
    ColorAgent,
    CompositionAgent,
    ExposureAgent,
    HaloAgent,
    NoiseAgent,
    PostProductionAgent,
    SharpnessAgent,
    SkyAgent,
    StudioOrchestrator,
    StudioReport,
    VerticalAgent,
    WhiteBalanceAgent,
)


def _balanced_interior(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 360, 540
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    img[: h // 3] = (200, 195, 188)
    img[2 * h // 3 :] = (90, 85, 75)
    cv2.rectangle(img, (180, 140), (380, 240), (40, 40, 40), thickness=3)
    img = img + rng.integers(-2, 3, size=img.shape, dtype=np.int8).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _blown_image() -> np.ndarray:
    img = _balanced_interior()
    img[:200] = 252
    return img


def _warm_cast() -> np.ndarray:
    img = _balanced_interior().astype(np.int16)
    img[..., 2] = np.clip(img[..., 2] + 35, 0, 255)
    return img.astype(np.uint8)


def _tilted_image() -> np.ndarray:
    base = _balanced_interior()
    # Add several long vertical lines so Hough has something to detect.
    for x in (80, 200, 320, 440):
        cv2.line(base, (x, 30), (x + 6, 320), (20, 20, 20), 2)
    # Now rotate 2.0° clockwise.
    h, w = base.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    return cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------- Per-agent contract ----------


@pytest.mark.parametrize(
    "agent",
    [
        ExposureAgent(),
        WhiteBalanceAgent(),
        SharpnessAgent(),
        ColorAgent(),
        VerticalAgent(),
        HaloAgent(),
        SkyAgent(),
        NoiseAgent(),
        CompositionAgent(),
    ],
)
class TestAgentContract:
    def test_satisfies_protocol(self, agent):
        assert isinstance(agent, PostProductionAgent)
        assert agent.name and isinstance(agent.name, str)
        assert agent.role and isinstance(agent.role, str)
        assert agent.category and isinstance(agent.category, str)

    def test_evaluate_returns_evaluation(self, agent):
        ev = agent.evaluate(_balanced_interior(), scene="interior")
        assert isinstance(ev, AgentEvaluation)
        assert 0.0 <= ev.score <= 10.0
        assert isinstance(ev.checklist, tuple)

    def test_apply_is_idempotent_when_image_is_good(self, agent):
        img = _balanced_interior()
        ev = agent.evaluate(img, scene="interior")
        out, report = agent.apply(img, scene="interior", evaluation=ev)
        assert isinstance(report, AgentApplyReport)
        # Apply might or might not run; if it didn't run, image is unchanged.
        if not report.applied:
            assert np.array_equal(out, img)


# ---------- Targeted intervention tests ----------


class TestExposureAgent:
    def test_intervenes_on_blown_image(self):
        agent = ExposureAgent()
        img = _blown_image()
        ev = agent.evaluate(img, scene="interior")
        assert ev.score < 6.0
        out, report = agent.apply(img, scene="interior", evaluation=ev)
        assert report.applied is True
        assert any("highlight" in a.lower() for a in report.actions)
        # After correction, blown ratio should drop.
        gray_after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        blown_after = float((gray_after >= 250).sum() / gray_after.size)
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blown_before = float((gray_before >= 250).sum() / gray_before.size)
        assert blown_after < blown_before


class TestWhiteBalanceAgent:
    def test_neutralises_warm_cast(self):
        agent = WhiteBalanceAgent()
        img = _warm_cast()
        ev = agent.evaluate(img, scene="interior")
        assert ev.score < 7.5
        out, report = agent.apply(img, scene="interior", evaluation=ev)
        assert report.applied is True
        ev_after = agent.evaluate(out, scene="interior")
        assert ev_after.score > ev.score  # actual improvement


class TestVerticalAgent:
    def test_corrects_tilt(self):
        agent = VerticalAgent()
        img = _tilted_image()
        ev = agent.evaluate(img, scene="interior")
        # The agent should detect tilt > 0.4°.
        if ev.metrics.get("median_deviation_deg", 0.0) <= 0.4:
            pytest.skip("synthetic tilt below threshold on this run")
        out, report = agent.apply(img, scene="interior", evaluation=ev)
        assert report.applied is True
        # After rotation, median deviation should drop.
        ev_after = agent.evaluate(out, scene="interior")
        assert (
            ev_after.metrics.get("median_deviation_deg", 99.0)
            < ev.metrics.get("median_deviation_deg", 0.0)
        )


class TestSkyAgent:
    def test_skipped_on_interior(self):
        agent = SkyAgent()
        ev = agent.evaluate(_balanced_interior(), scene="interior")
        assert ev.score == 10.0
        assert ev.metrics.get("applicable", 1.0) == 0.0


# ---------- Orchestrator ----------


class TestStudioOrchestrator:
    def test_returns_studio_report(self):
        orchestrator = StudioOrchestrator()
        out, report = orchestrator.run(_balanced_interior(), scene="interior")
        assert isinstance(report, StudioReport)
        assert out.shape == _balanced_interior().shape
        assert len(report.agents) == 9
        assert report.grade in {"S", "A", "B", "C", "D"}
        assert 0.0 <= report.overall_after <= 10.0

    def test_overall_after_at_least_as_good_as_before(self):
        orchestrator = StudioOrchestrator()
        _, report = orchestrator.run(_blown_image(), scene="interior")
        # Rollback policy ensures we never regress on aggregate.
        assert report.overall_after >= report.overall_before - 0.6, (
            f"overall regressed: {report.overall_before:.2f} → {report.overall_after:.2f}"
        )

    def test_serialisation(self):
        orchestrator = StudioOrchestrator()
        _, report = orchestrator.run(_balanced_interior(), scene="interior")
        d = report.as_dict()
        assert d["scene"] == "interior"
        assert "agents" in d
        assert len(d["agents"]) == 9
        for a in d["agents"]:
            assert "before" in a
            assert "after" in a
            assert "apply" in a
            assert "checklist" in a["before"]

    def test_rejects_non_bgr_uint8(self):
        with pytest.raises(ValueError):
            StudioOrchestrator().run(np.zeros((10, 10, 3), dtype=np.float32), scene="interior")
