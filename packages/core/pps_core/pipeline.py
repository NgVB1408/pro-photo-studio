"""Pipeline runner — stage registry + ``Pipeline.run(job, image)``.

The runner is the architectural backbone: every entry point (CLI, web, API,
Celery worker) routes through ``Pipeline.run`` so behaviour is identical and
deterministic across surfaces.

Usage
-----

    from pps_core.pipeline import Pipeline, register
    from pps_core.types import Job, StageContext, StageReport
    import numpy as np

    @register("brighten")
    def brighten(img: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        amt = float(ctx.params.get("amount", 1.1))
        out = np.clip(img.astype(np.float32) * amt, 0, 255).astype(np.uint8)
        return out, StageReport(name=ctx.stage_name, applied=True,
                                metrics={"amount": amt})

    job = Job(job_id="abc", stages=("brighten",),
              params={"brighten": {"amount": 1.3}}, seed=42)
    final, report = Pipeline().run(job, image)

The default ``Pipeline()`` reads from a process-global registry populated by
``@register``. For tests, build a ``Pipeline(registry={"name": fn, ...})``
with only the stages you care about.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from .types import Job, Report, Stage, StageContext, StageReport, seed_for_stage

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "ProgressCallback",
    "register",
    "registry",
]

ProgressCallback = Callable[[int, int, StageReport], None]
"""Called after every stage. Args: (current_index_1based, total, report)."""


_GLOBAL_REGISTRY: dict[str, Stage] = {}


def register(name: str) -> Callable[[Stage], Stage]:
    """Decorator: register a stage by name in the process-global registry.

    A stage is anything matching the ``Stage`` protocol. Decorating with
    ``@register("foo")`` sets ``stage.name = "foo"`` and stores it.

    Re-registering the same name overwrites the previous entry — this is
    intentional so tests can override stages without import side-effects.
    """

    def _wrap(stage: Stage) -> Stage:
        # Allow plain callables — attach name attribute for Protocol compliance.
        if not hasattr(stage, "name") or getattr(stage, "name", None) != name:
            try:
                stage.name = name  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                # frozen object: wrap it
                wrapped = _NamedCallable(name, stage)
                _GLOBAL_REGISTRY[name] = wrapped
                return wrapped  # type: ignore[return-value]
        _GLOBAL_REGISTRY[name] = stage
        return stage

    return _wrap


def registry() -> Mapping[str, Stage]:
    """Read-only view of the global registry. Returns a copy snapshot."""
    return dict(_GLOBAL_REGISTRY)


class _NamedCallable:
    """Internal: wrap a callable that lacks a settable ``name`` attribute."""

    def __init__(self, name: str, fn: Callable[..., tuple[np.ndarray, StageReport]]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, image: np.ndarray, ctx: StageContext) -> tuple[np.ndarray, StageReport]:
        return self._fn(image, ctx)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runner behaviour knobs.

    Attributes:
        halt_on_error:
            If True, stop the pipeline after a stage raises. If False
            (default), record the error in that stage's report and continue
            — the next stage receives the *pre-error* image so partial
            damage doesn't propagate.
        skip_unknown_stages:
            If True (default), skip stage names not in the registry with a
            warning. If False, raise ``KeyError``.
        log_progress:
            If True, log INFO line per stage with timing.
    """

    halt_on_error: bool = False
    skip_unknown_stages: bool = True
    log_progress: bool = True


@dataclass
class Pipeline:
    """Stateless runner. Safe to share across threads — no instance state
    is mutated during ``run``."""

    registry: Mapping[str, Stage] | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def _resolve_stages(self, job: Job) -> Iterator[tuple[str, Stage | None]]:
        reg = self.registry if self.registry is not None else _GLOBAL_REGISTRY
        for name in job.stages:
            yield name, reg.get(name)

    def run(
        self,
        job: Job,
        image: np.ndarray,
        *,
        progress: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, Report]:
        """Run the requested stages on ``image`` for ``job``.

        Args:
            job: The job describing which stages to run with which params.
            image: Input BGR uint8 ndarray. The runner does NOT validate the
                shape — stages are expected to do that themselves.
            progress: Optional callback invoked after each stage with
                ``(index_1based, total, stage_report)``.

        Returns:
            ``(final_image, Report)``. ``final_image`` is the output of the
            last successful stage, or the original ``image`` if every stage
            was skipped/errored.
        """
        if image.ndim != 3:
            raise ValueError(f"Pipeline.run: expected (H, W, 3) BGR image, got shape {image.shape}")
        t0 = time.perf_counter()
        reports: list[StageReport] = []
        current = image
        halted = False
        total = len(job.stages)

        for idx, (name, stage) in enumerate(self._resolve_stages(job), start=1):
            if stage is None:
                report = StageReport(
                    name=name,
                    skipped=True,
                    reason="stage not registered",
                )
                reports.append(report)
                if self.config.log_progress:
                    logger.warning("[pipeline] %s/%s %s: not registered", idx, total, name)
                if progress is not None:
                    progress(idx, total, report)
                if self.config.skip_unknown_stages:
                    continue
                raise KeyError(f"Stage not registered: {name!r}")

            ctx = StageContext(
                job=job,
                stage_name=name,
                stage_seed=seed_for_stage(job.seed, name),
                params=dict(job.params.get(name, {})),
            )
            stage_t0 = time.perf_counter()
            try:
                next_image, report = stage(current, ctx)
                duration_ms = (time.perf_counter() - stage_t0) * 1000.0
                # Replace duration with our measurement (stages can pre-fill it
                # but the runner is authoritative).
                report = _replace_duration(report, duration_ms)
                if not _is_image_like(next_image):
                    report = _replace_skipped(
                        report,
                        skipped=True,
                        reason=(
                            f"stage returned non-image (type={type(next_image).__name__}); "
                            "input preserved"
                        ),
                    )
                else:
                    current = next_image
            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_t0) * 1000.0
                report = StageReport(
                    name=name,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                )
                logger.exception("[pipeline] %s/%s %s: error", idx, total, name)

            reports.append(report)
            if self.config.log_progress:
                logger.info(
                    "[pipeline] %s/%s %s: applied=%s skipped=%s err=%s %.1fms",
                    idx,
                    total,
                    name,
                    report.applied,
                    report.skipped,
                    "y" if report.error else "n",
                    report.duration_ms,
                )
            if progress is not None:
                progress(idx, total, report)
            if report.error and self.config.halt_on_error:
                halted = True
                break

        return current, Report(
            job_id=job.job_id,
            stages=tuple(reports),
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            halted=halted,
        )


def _is_image_like(x: object) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[2] in (3, 4)


def _replace_duration(r: StageReport, duration_ms: float) -> StageReport:
    return StageReport(
        name=r.name,
        applied=r.applied,
        skipped=r.skipped,
        error=r.error,
        duration_ms=duration_ms,
        warnings=r.warnings,
        metrics=r.metrics,
        reason=r.reason,
    )


def _replace_skipped(r: StageReport, *, skipped: bool, reason: str) -> StageReport:
    return StageReport(
        name=r.name,
        applied=False,
        skipped=skipped,
        error=r.error,
        duration_ms=r.duration_ms,
        warnings=r.warnings,
        metrics=r.metrics,
        reason=reason,
    )
