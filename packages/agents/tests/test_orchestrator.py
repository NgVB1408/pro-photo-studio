"""End-to-end smoke: orchestrator runs all agents in parallel + Director gate."""

from __future__ import annotations

from pps_agents.orchestrator import DEFAULT_APPLY_ORDER, Orchestrator
from pps_agents.types import JobContext


def test_orchestrator_full_pipeline_runs(small_interior):
    ctx = JobContext(
        image=small_interior,
        target_long_edge=720,  # small to keep test fast
        target_dpi=300,
        property_type="villa_luxury",
        seed=7,
    )
    result = Orchestrator(max_workers=4).run(ctx)
    # Every default agent has a plan and a report.
    assert set(result.plans) >= set(DEFAULT_APPLY_ORDER)
    assert set(result.reports) == set(DEFAULT_APPLY_ORDER)
    # Director attached.
    assert result.director is not None
    assert result.director.verdict in {"pass", "review", "fail"}
    # Apply phase total ≤ analyze phase + apply phase + epsilon (parallel sanity)
    assert result.analyze_duration_s >= 0
    assert result.apply_duration_s >= 0
    # Output is a uint8 BGR image at the requested size.
    assert result.image.dtype == small_interior.dtype
    assert max(result.image.shape[:2]) == 720


def test_skip_propagates_to_report(small_interior):
    """Geometry agent should report a skip-or-applied state — never crash."""
    ctx = JobContext(image=small_interior, target_long_edge=540)
    result = Orchestrator(max_workers=2).run(ctx)
    geom = result.reports["geometry"]
    assert geom.name == "geometry"
    if geom.skipped:
        assert geom.skip_reason


def test_director_question_keys_stable(small_interior):
    ctx = JobContext(image=small_interior, target_long_edge=540)
    result = Orchestrator().run(ctx)
    keys = set(result.director.question_scores)
    assert keys == {
        "Q1_halo_window_corners",
        "Q2_ceiling_neutrality",
        "Q3_move_in_feel",
    }


def test_summary_serialisable(small_interior):
    """summary() must be JSON-friendly — no numpy types or non-stringable keys."""
    import json

    ctx = JobContext(image=small_interior, target_long_edge=540)
    result = Orchestrator().run(ctx)
    s = result.summary()
    json.dumps(s)  # raises if non-serialisable


def test_orchestrator_invokes_gene_provider(small_interior):
    """gene_providers callable must be invoked once per run with the input
    image, and its result must reach the agent's plan metadata."""
    calls: list[int] = []

    def fake_provider(image):
        calls.append(image.size)
        return [
            {
                "agent": "microcontrast",
                "texture": {"fine": 0.10, "mid": 0.08, "macro": 0.04},
                "dehaze_amount": 0.02,
            }
        ]

    ctx = JobContext(
        image=small_interior, target_long_edge=540, property_type="villa_luxury"
    )
    result = Orchestrator(
        max_workers=2, gene_providers={"microcontrast": fake_provider}
    ).run(ctx)
    assert len(calls) == 1
    assert calls[0] == small_interior.size
    micro_meta = result.plans["microcontrast"].metadata
    assert micro_meta["gene_count"] == 1
    assert micro_meta["gene_blend_weight"] > 0.0
    # Caller's ctx must NOT have been mutated.
    assert "genes_microcontrast" not in ctx.metadata


def test_orchestrator_gene_provider_failure_is_swallowed(small_interior):
    """A throwing gene_provider must NOT take down the pipeline."""

    def boom(_image):
        raise RuntimeError("oracle is down")

    ctx = JobContext(image=small_interior, target_long_edge=540)
    result = Orchestrator(
        max_workers=2, gene_providers={"microcontrast": boom}
    ).run(ctx)
    # Pipeline still completes with all stages reported.
    assert set(result.reports) == set(DEFAULT_APPLY_ORDER)
    # No genes were injected — agent ran on baseline only.
    assert result.plans["microcontrast"].metadata["gene_count"] == 0


def test_orchestrator_no_provider_means_baseline(small_interior):
    """Backwards-compat: omitting gene_providers leaves agents on baseline."""
    ctx = JobContext(image=small_interior, target_long_edge=540)
    result = Orchestrator(max_workers=2).run(ctx)
    assert result.plans["microcontrast"].metadata["gene_count"] == 0
    assert result.plans["microcontrast"].metadata["gene_blend_weight"] == 0.0
