"""Sync pipeline dispatcher — bridges HTTP layer to pps_core.pipeline.

Runs the pipeline in a worker thread (so the FastAPI event loop is not
blocked) and updates the ``JobStore`` as the job progresses. Errors are
captured into ``JobRecord.error``; the status transitions are:

    queued → running → (completed | failed)

Phase 2.3 will introduce a Celery-backed runner with the same surface; the
current implementation is good enough for single-process tests and local
dev (≤ ~10 concurrent jobs).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
from pps_core.pipeline import Pipeline
from pps_core.types import Job

from pps_api.schemas import JobStatus

from .job_store import JobStore

logger = logging.getLogger(__name__)


async def run_pipeline_for_job(
    *,
    job: Job,
    image: np.ndarray,
    store: JobStore,
    output_dir: Path,
    pipeline: Pipeline | None = None,
) -> None:
    """Execute ``job`` on ``image`` and persist outcome via ``store``.

    Args:
        job: The pipeline job. Must be present in ``store`` (status=queued).
        image: BGR uint8 ndarray as returned by cv2.imread.
        store: Where to record progress.
        output_dir: Directory to write the final image.
        pipeline: Optional override (default uses the global registry).

    On success: writes ``output_dir/<job_id>.jpg`` and updates the record to
    ``status=completed`` with ``result_path`` and ``report`` set.

    On failure: updates to ``status=failed`` with ``error`` set. The
    function never raises — callers can fire-and-forget.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        await store.update(job.job_id, status=JobStatus.running)
        runner = pipeline or Pipeline()

        # Run the pipeline in a worker thread to avoid blocking the loop.
        loop = asyncio.get_running_loop()
        final_image, report = await loop.run_in_executor(None, runner.run, job, image)

        out_path = output_dir / f"{job.job_id}.jpg"
        # Use cv2 imwrite (BGR uint8). JPEG quality 92 = good default for
        # real-estate output; tweak via per-job param later if needed.
        ok = cv2.imwrite(str(out_path), final_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError(f"cv2.imwrite returned False for {out_path}")

        await store.update(
            job.job_id,
            status=JobStatus.completed,
            report=report,
            result_path=str(out_path),
        )
        logger.info("[runner] job=%s completed in %.1fms", job.job_id, report.duration_ms)
    except Exception as exc:
        logger.exception("[runner] job=%s failed", job.job_id)
        try:
            await store.update(
                job.job_id,
                status=JobStatus.failed,
                error=f"{type(exc).__name__}: {exc}",
            )
        except KeyError:
            # Store didn't have the job yet — log and move on.
            logger.error("[runner] couldn't update missing job=%s", job.job_id)
