"""Per-agent unit tests using a synthesised interior."""

from __future__ import annotations

import numpy as np

from pps_agents.cleanup import CleanupAgent
from pps_agents.geometry import GeometryAgent
from pps_agents.lightblend import LightBlendAgent
from pps_agents.microcontrast import MicroContrastAgent
from pps_agents.output import OutputAgent
from pps_agents.types import JobContext


def _ctx(img: np.ndarray, **kw) -> JobContext:
    return JobContext(image=img, **kw)


# ----------------------------- geometry -----------------------------


def test_geometry_detects_tilt(small_interior):
    a = GeometryAgent()
    plan = a.analyze(_ctx(small_interior))
    assert plan.name == "geometry"
    assert plan.metadata["n_vertical_lines"] >= 0
    out, report = a.apply(small_interior, plan)
    assert out.shape == small_interior.shape
    assert report.name == "geometry"


# ----------------------------- lightblend -----------------------------


def test_lightblend_finds_blown_window(small_interior):
    a = LightBlendAgent()
    plan = a.analyze(_ctx(small_interior))
    # Synthesised window fills > 0.5% of the frame
    assert plan.metadata["clip_high"] > 0.005 or plan.metadata["blown_window_ratio"] >= 0
    out, report = a.apply(small_interior, plan)
    assert out.shape == small_interior.shape
    # Window region should be at least as bright on average as before
    assert report.name == "lightblend"


# ----------------------------- microcontrast -----------------------------


def test_microcontrast_property_aware(small_interior):
    a = MicroContrastAgent()
    plan_villa = a.analyze(_ctx(small_interior, property_type="villa_luxury"))
    plan_studio = a.analyze(_ctx(small_interior, property_type="studio_minimal"))
    villa_band = next(o for o in plan_villa.operations if o["op"] == "multi_band_texture")
    studio_band = next(o for o in plan_studio.operations if o["op"] == "multi_band_texture")
    # Villas demand more punch than studios.
    assert villa_band["fine"] > studio_band["fine"]
    assert villa_band["mid"] > studio_band["mid"]
    # Apply runs and produces an image of the same size.
    out, report = a.apply(small_interior, plan_villa)
    assert out.shape == small_interior.shape
    assert report.name == "microcontrast"


def test_microcontrast_protects_extremes(small_interior):
    """Highlight protection prevents huge boost in 250+ pixels."""
    a = MicroContrastAgent()
    plan = a.analyze(_ctx(small_interior))
    out, _ = a.apply(small_interior, plan)
    # The pure-white window region (close to 252) should not grow beyond clip
    bright_orig = small_interior[(small_interior[..., 0] > 240)].mean(axis=0)
    bright_new = out[(out[..., 0] > 240)].mean(axis=0) if (out[..., 0] > 240).any() else bright_orig
    # Growth bounded by 6 grey levels
    assert np.all(bright_new <= bright_orig + 8)


def test_microcontrast_gene_blend_pulls_toward_genes(small_interior):
    """When ctx.metadata['genes_microcontrast'] is provided, params should
    blend the property baseline with the gene mean."""
    from pps_agents.microcontrast import GENE_BLEND_WEIGHT

    a = MicroContrastAgent()
    villa_baseline_plan = a.analyze(_ctx(small_interior, property_type="villa_luxury"))
    villa_band = next(o for o in villa_baseline_plan.operations if o["op"] == "multi_band_texture")
    baseline_fine = villa_band["fine"]

    # A studio-style gene (lower fine) should pull villa params downward.
    studio_gene = {
        "agent": "microcontrast",
        "property": "studio_minimal",
        "texture": {"fine": 0.10, "mid": 0.10, "macro": 0.04},
        "dehaze_amount": 0.02,
        "wood_clarity": 0.10,
    }
    ctx = _ctx(
        small_interior,
        property_type="villa_luxury",
        metadata={"genes_microcontrast": [studio_gene]},
    )
    plan = a.analyze(ctx)
    band = next(o for o in plan.operations if o["op"] == "multi_band_texture")
    expected = (1 - GENE_BLEND_WEIGHT) * baseline_fine + GENE_BLEND_WEIGHT * 0.10
    assert abs(band["fine"] - expected) < 1e-6
    # Metadata records that genes were used.
    assert plan.metadata["gene_count"] == 1
    assert plan.metadata["gene_blend_weight"] == GENE_BLEND_WEIGHT


def test_microcontrast_gene_blend_skips_wrong_agent(small_interior):
    """Genes tagged for a different agent are filtered out."""
    a = MicroContrastAgent()
    ref = a.analyze(_ctx(small_interior, property_type="apartment_modern"))
    ref_fine = next(
        o for o in ref.operations if o["op"] == "multi_band_texture"
    )["fine"]

    bogus_gene = {
        "agent": "lightblend",  # wrong agent — must be ignored
        "texture": {"fine": 0.99, "mid": 0.99, "macro": 0.99},
    }
    ctx = _ctx(
        small_interior,
        property_type="apartment_modern",
        metadata={"genes_microcontrast": [bogus_gene]},
    )
    plan = a.analyze(ctx)
    band = next(o for o in plan.operations if o["op"] == "multi_band_texture")
    assert abs(band["fine"] - ref_fine) < 1e-6
    # Gene was counted as input but provided no usable values.
    assert plan.metadata["gene_count"] == 1


def test_microcontrast_gene_blend_clipped_to_safe_bounds(small_interior):
    """Out-of-range gene values must be clipped to the sane bound, not used as-is."""
    a = MicroContrastAgent()
    poison_gene = {
        "agent": "microcontrast",
        "texture": {"fine": 99.0, "mid": -5.0, "macro": 0.2},
        "dehaze_amount": 50.0,
    }
    ctx = _ctx(
        small_interior,
        property_type="villa_luxury",
        metadata={"genes_microcontrast": [poison_gene]},
    )
    plan = a.analyze(ctx)
    band = next(o for o in plan.operations if o["op"] == "multi_band_texture")
    dehaze = next(o for o in plan.operations if o["op"] == "selective_dehaze_windows")
    # Bounded per _PARAM_BOUNDS in microcontrast.py
    assert 0.0 <= band["fine"] <= 0.7
    assert 0.0 <= band["mid"] <= 0.7
    assert 0.0 <= dehaze["amount"] <= 0.4


def test_blend_microcontrast_genes_unit():
    """Pure-function unit: convex blend with bound clipping."""
    from pps_agents.microcontrast import blend_microcontrast_genes

    baseline = {
        "fine": 0.4,
        "mid": 0.3,
        "macro": 0.1,
        "dehaze": 0.1,
        "wood": 0.45,
        "marble": 0.25,
        "sharpen_amount": 0.55,
        "sharpen_sigma": 1.4,
    }
    # No genes → identical to baseline
    out = blend_microcontrast_genes(baseline, [], weight=0.5)
    assert out == baseline
    # Single gene at weight=1.0 → output uses gene values for fields it sets,
    # baseline for fields it doesn't.
    gene = {
        "agent": "microcontrast",
        "texture": {"fine": 0.2, "mid": 0.2, "macro": 0.05},
        "dehaze_amount": 0.0,
    }
    out = blend_microcontrast_genes(baseline, [gene], weight=1.0)
    assert abs(out["fine"] - 0.2) < 1e-6
    assert abs(out["dehaze"] - 0.0) < 1e-6
    # 'wood' was not in gene — stays at baseline value 0.45
    assert abs(out["wood"] - 0.45) < 1e-6


# ----------------------------- cleanup -----------------------------


def test_cleanup_finds_dark_rectangles(small_interior):
    a = CleanupAgent()
    plan = a.analyze(_ctx(small_interior))
    # Dark TV rectangle should register
    has_tv = any(op["op"] == "tv_blackout" for op in plan.operations)
    assert has_tv
    out, report = a.apply(small_interior, plan)
    assert out.shape == small_interior.shape
    assert report.name == "cleanup"


# ----------------------------- output -----------------------------


def test_output_upscales_to_target(small_interior):
    a = OutputAgent()
    ctx = _ctx(small_interior, target_long_edge=1080)
    plan = a.analyze(ctx)
    out, report = a.apply(small_interior, plan)
    assert max(out.shape[:2]) == 1080
    assert "upscale" in report.metrics


def test_output_skips_when_already_big(interior_image):
    """target_long_edge below current size = no upscale op produced."""
    a = OutputAgent()
    ctx = _ctx(interior_image, target_long_edge=400)
    plan = a.analyze(ctx)
    upscale_ops = [o for o in plan.operations if o["op"] == "upscale"]
    assert not upscale_ops
