"""Tests cho pps_core.types — Job/Stage/Report dataclasses + seed derivation."""

from __future__ import annotations

import dataclasses

import pytest
from pps_core.types import (
    Job,
    Report,
    Severity,
    Stage,
    StageContext,
    StageReport,
    seed_for_stage,
)


class TestJob:
    def test_job_minimal(self):
        j = Job(job_id="abc", stages=("foo",))
        assert j.job_id == "abc"
        assert j.stages == ("foo",)
        assert j.params == {}
        assert j.seed is None
        assert j.metadata == {}

    def test_job_frozen(self):
        j = Job(job_id="abc", stages=("foo",))
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            j.job_id = "different"  # type: ignore[misc]

    def test_job_full(self):
        j = Job(
            job_id="x",
            stages=("a", "b"),
            params={"a": {"strength": 0.5}, "b": {"mode": "auto"}},
            seed=42,
            metadata={"user": "alice"},
        )
        assert j.params["a"]["strength"] == 0.5
        assert j.metadata["user"] == "alice"


class TestStageContext:
    def test_context_carries_stage_seed(self):
        job = Job(job_id="x", stages=("foo",), seed=100)
        ctx = StageContext(
            job=job,
            stage_name="foo",
            stage_seed=12345,
            params={"a": 1},
        )
        assert ctx.stage_seed == 12345
        assert ctx.params["a"] == 1
        assert ctx.job is job

    def test_context_frozen(self):
        job = Job(job_id="x", stages=("foo",))
        ctx = StageContext(job=job, stage_name="foo", stage_seed=None, params={})
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            ctx.stage_name = "bar"  # type: ignore[misc]


class TestStageReport:
    def test_report_defaults(self):
        r = StageReport(name="foo")
        assert r.name == "foo"
        assert r.applied is False
        assert r.skipped is False
        assert r.error is None
        assert r.duration_ms == 0.0
        assert r.warnings == ()
        assert r.metrics == {}

    def test_report_with_warnings(self):
        r = StageReport(
            name="foo",
            applied=True,
            warnings=(("info", "low contrast"), ("warn", "near clipping")),
        )
        assert len(r.warnings) == 2
        assert r.warnings[0] == ("info", "low contrast")

    def test_report_with_metrics(self):
        r = StageReport(name="foo", applied=True, metrics={"psnr": 38.4, "ssim": 0.95})
        assert r.metrics["psnr"] == pytest.approx(38.4)


class TestReport:
    def _make(self):
        return Report(
            job_id="abc",
            stages=(
                StageReport(name="a", applied=True, duration_ms=10.0),
                StageReport(name="b", skipped=True, reason="not needed"),
                StageReport(name="c", error="ValueError: bad input"),
                StageReport(name="d", applied=True, duration_ms=5.0),
            ),
            duration_ms=20.0,
        )

    def test_applied_stages(self):
        r = self._make()
        assert r.applied_stages == ("a", "d")

    def test_skipped_stages(self):
        r = self._make()
        assert r.skipped_stages == ("b",)

    def test_errored_stages(self):
        r = self._make()
        assert r.errored_stages == ("c",)

    def test_halted_default_false(self):
        r = self._make()
        assert r.halted is False


class TestSeedDerivation:
    def test_none_in_none_out(self):
        assert seed_for_stage(None, "foo") is None

    def test_deterministic(self):
        a = seed_for_stage(42, "sky_replace")
        b = seed_for_stage(42, "sky_replace")
        assert a == b
        assert a is not None
        assert 0 <= a < 2**32

    def test_different_stages_different_seeds(self):
        a = seed_for_stage(42, "sky_replace")
        b = seed_for_stage(42, "twilight")
        assert a != b

    def test_different_jobs_different_seeds(self):
        a = seed_for_stage(42, "sky_replace")
        b = seed_for_stage(43, "sky_replace")
        assert a != b

    def test_seed_is_uint32(self):
        for s in (0, 1, 42, 2**31, 2**31 - 1, 2**32 - 1):
            for name in ("a", "very_long_stage_name_that_is_unusual"):
                derived = seed_for_stage(s, name)
                assert derived is not None
                assert 0 <= derived < 2**32


class TestStageProtocol:
    def test_lambda_does_not_satisfy_protocol(self):
        # A bare lambda has no `name` attribute → fails runtime_checkable
        bad: object = lambda image, ctx: (image, StageReport(name="x"))  # noqa: E731
        assert not isinstance(bad, Stage)

    def test_class_with_name_satisfies_protocol(self):
        import numpy as np

        class Foo:
            name = "foo"

            def __call__(self, image, ctx):
                return image, StageReport(name=self.name, applied=True)

        f = Foo()
        assert isinstance(f, Stage)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        ctx = StageContext(
            job=Job(job_id="x", stages=("foo",)),
            stage_name="foo",
            stage_seed=None,
            params={},
        )
        out, rpt = f(img, ctx)
        assert out.shape == img.shape
        assert rpt.applied is True


def test_severity_is_loose_string():
    # Severity is intentionally just `str` for forward compat; nothing
    # enforces a particular value set at the type level.
    assert isinstance("info", Severity)
    assert isinstance("custom-severity", Severity)
