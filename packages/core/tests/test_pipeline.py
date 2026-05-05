"""Tests cho pps_core.pipeline — Pipeline.run + registry + determinism."""

from __future__ import annotations

import numpy as np
import pytest
from pps_core.pipeline import (
    _GLOBAL_REGISTRY,
    Pipeline,
    PipelineConfig,
    register,
    registry,
)
from pps_core.types import Job, StageContext, StageReport


@pytest.fixture(autouse=True)
def _reset_registry():
    """Save & restore the process-global registry around each test."""
    snapshot = dict(_GLOBAL_REGISTRY)
    _GLOBAL_REGISTRY.clear()
    yield
    _GLOBAL_REGISTRY.clear()
    _GLOBAL_REGISTRY.update(snapshot)


def _img(h=8, w=12, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ---------- registry / register ----------


class TestRegistry:
    def test_register_via_class(self):
        class Foo:
            name = "foo"

            def __call__(self, image, ctx):
                return image, StageReport(name="foo", applied=True)

        register("foo")(Foo())
        reg = registry()
        assert "foo" in reg
        assert reg["foo"].name == "foo"

    def test_register_via_callable(self):
        def bar(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True)

        register("bar")(bar)
        reg = registry()
        assert "bar" in reg

    def test_register_overwrites_silently(self):
        def v1(image, ctx):
            return image, StageReport(name="x", applied=True, metrics={"v": 1})

        def v2(image, ctx):
            return image, StageReport(name="x", applied=True, metrics={"v": 2})

        register("x")(v1)
        register("x")(v2)
        _out, rpt = registry()["x"](_img(), _ctx("x"))
        assert rpt.metrics["v"] == 2


# ---------- Pipeline.run ----------


class TestPipelineRun:
    def test_run_with_explicit_registry(self):
        def brighten(image, ctx):
            amt = ctx.params.get("amount", 1.1)
            return (
                np.clip(image.astype(np.float32) * amt, 0, 255).astype(np.uint8),
                StageReport(name=ctx.stage_name, applied=True, metrics={"amount": amt}),
            )

        p = Pipeline(registry={"brighten": _wrap("brighten", brighten)})
        job = Job(
            job_id="t1",
            stages=("brighten",),
            params={"brighten": {"amount": 1.5}},
        )
        out, report = p.run(job, _img(fill=100))
        assert out.dtype == np.uint8
        assert int(out.mean()) == 150
        assert report.job_id == "t1"
        assert report.stages[0].applied is True
        assert report.stages[0].metrics["amount"] == 1.5
        assert report.duration_ms > 0

    def test_unknown_stage_skipped_by_default(self):
        p = Pipeline(registry={})
        job = Job(job_id="t2", stages=("ghost",))
        _out, report = p.run(job, _img())
        assert report.stages[0].skipped is True
        assert "not registered" in report.stages[0].reason

    def test_unknown_stage_raises_when_configured(self):
        p = Pipeline(registry={}, config=PipelineConfig(skip_unknown_stages=False))
        job = Job(job_id="t3", stages=("ghost",))
        with pytest.raises(KeyError):
            p.run(job, _img())

    def test_stage_error_recorded_and_pipeline_continues(self):
        def boom(image, ctx):
            raise ValueError("intentional")

        def ok(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(
            registry={
                "boom": _wrap("boom", boom),
                "ok": _wrap("ok", ok),
            }
        )
        job = Job(job_id="t4", stages=("boom", "ok"))
        _out, report = p.run(job, _img())
        assert "ValueError: intentional" in report.stages[0].error
        assert report.stages[1].applied is True
        assert report.halted is False

    def test_halt_on_error(self):
        def boom(image, ctx):
            raise RuntimeError("die")

        def never(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(
            registry={"boom": _wrap("boom", boom), "never": _wrap("never", never)},
            config=PipelineConfig(halt_on_error=True),
        )
        job = Job(job_id="t5", stages=("boom", "never"))
        _out, report = p.run(job, _img())
        assert report.halted is True
        assert len(report.stages) == 1
        assert report.stages[0].error is not None

    def test_progress_callback_invoked(self):
        seen: list[tuple[int, int, str]] = []

        def cb(idx, total, rep):
            seen.append((idx, total, rep.name))

        def stage_a(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(registry={"a": _wrap("a", stage_a), "b": _wrap("b", stage_a)})
        job = Job(job_id="t6", stages=("a", "b"))
        p.run(job, _img(), progress=cb)
        assert seen == [(1, 2, "a"), (2, 2, "b")]

    def test_invalid_input_shape_raises(self):
        p = Pipeline(registry={})
        job = Job(job_id="t7", stages=())
        with pytest.raises(ValueError, match="expected"):
            p.run(job, np.zeros((4, 4), dtype=np.uint8))

    def test_stage_returning_non_image_skipped(self):
        def lies(image, ctx):
            return ("not an image", StageReport(name=ctx.stage_name, applied=True))

        p = Pipeline(registry={"liar": _wrap("liar", lies)})
        job = Job(job_id="t8", stages=("liar",))
        out, report = p.run(job, _img())
        assert report.stages[0].skipped is True
        assert "non-image" in report.stages[0].reason
        # Input should be preserved
        assert np.array_equal(out, _img())

    def test_seed_propagated_to_stage_context(self):
        captured_seeds: list[int | None] = []

        def grab(image, ctx):
            captured_seeds.append(ctx.stage_seed)
            return image, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(registry={"a": _wrap("a", grab), "b": _wrap("b", grab)})
        job = Job(job_id="t9", stages=("a", "b"), seed=42)
        p.run(job, _img())
        assert all(s is not None for s in captured_seeds)
        assert captured_seeds[0] != captured_seeds[1]  # different per stage

    def test_seed_none_propagated_as_none(self):
        captured: list[int | None] = []

        def grab(image, ctx):
            captured.append(ctx.stage_seed)
            return image, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(registry={"a": _wrap("a", grab)})
        job = Job(job_id="t10", stages=("a",), seed=None)
        p.run(job, _img())
        assert captured == [None]

    def test_runner_overrides_stage_duration(self):
        # Stage tries to lie about its duration; runner replaces it.
        def slow(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True, duration_ms=99999.0)

        p = Pipeline(registry={"a": _wrap("a", slow)})
        job = Job(job_id="t11", stages=("a",))
        _, report = p.run(job, _img())
        assert report.stages[0].duration_ms < 1000.0  # runner overrode

    def test_uses_global_registry_when_none(self):
        @register("global_stage")
        def gs(image, ctx):
            return image, StageReport(name=ctx.stage_name, applied=True, metrics={"k": 1.0})

        p = Pipeline()  # no registry passed → uses global
        job = Job(job_id="t12", stages=("global_stage",))
        _, report = p.run(job, _img())
        assert report.stages[0].applied is True


# ---------- Determinism ----------


class TestDeterminism:
    def test_same_seed_same_output(self):
        def noise(image, ctx):
            rng = np.random.default_rng(ctx.stage_seed)
            adj = rng.integers(-5, 5, image.shape, dtype=np.int16)
            out = np.clip(image.astype(np.int16) + adj, 0, 255).astype(np.uint8)
            return out, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(registry={"noise": _wrap("noise", noise)})
        job = Job(job_id="t-det", stages=("noise",), seed=12345)
        out_a, _ = p.run(job, _img(fill=120))
        out_b, _ = p.run(job, _img(fill=120))
        assert np.array_equal(out_a, out_b)

    def test_different_seed_different_output(self):
        def noise(image, ctx):
            rng = np.random.default_rng(ctx.stage_seed)
            adj = rng.integers(-5, 5, image.shape, dtype=np.int16)
            out = np.clip(image.astype(np.int16) + adj, 0, 255).astype(np.uint8)
            return out, StageReport(name=ctx.stage_name, applied=True)

        p = Pipeline(registry={"noise": _wrap("noise", noise)})
        job_a = Job(job_id="t-det1", stages=("noise",), seed=1)
        job_b = Job(job_id="t-det2", stages=("noise",), seed=2)
        out_a, _ = p.run(job_a, _img(fill=120))
        out_b, _ = p.run(job_b, _img(fill=120))
        assert not np.array_equal(out_a, out_b)


# ---------- helpers ----------


def _ctx(stage_name: str, *, params=None, seed=None):
    return StageContext(
        job=Job(job_id="ctx", stages=(stage_name,), seed=seed),
        stage_name=stage_name,
        stage_seed=seed,
        params=params or {},
    )


def _wrap(name: str, fn):
    """Make a plain function into a Stage-like object with .name attribute."""

    class _S:
        def __init__(self, n, f):
            self.name = n
            self._f = f

        def __call__(self, image, ctx):
            return self._f(image, ctx)

    return _S(name, fn)
