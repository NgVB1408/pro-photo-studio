"""Shared streaming + retry helpers for HF dataset loaders."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

log = logging.getLogger(__name__)


def load_streaming(
    repo_id: str,
    *,
    split: str = "train",
    config: str | None = None,
    token: str | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Any:
    """Wrap ``datasets.load_dataset`` with streaming + retries.

    HF Hub occasionally throws transient 5xx during config probing. We retry
    with exponential backoff so a single hiccup doesn't kill a long training
    run. The result is an ``IterableDataset`` that yields rows lazily.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pps-data needs `datasets>=3.0`. Install with: pip install datasets"
        ) from exc

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "path": repo_id,
                "split": split,
                "streaming": True,
            }
            if config is not None:
                kwargs["name"] = config
            if token is not None:
                kwargs["token"] = token
            return load_dataset(**kwargs)
        except Exception as exc:  # noqa: BLE001 — datasets raises many shapes
            last_exc = exc
            log.warning(
                "load_dataset(%s, split=%s) attempt %d/%d failed: %s",
                repo_id,
                split,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


def take(stream: Any, n: int) -> Iterator[dict[str, Any]]:
    """Yield up to ``n`` rows from a streaming dataset."""
    count = 0
    for row in stream:
        if count >= n:
            return
        yield row
        count += 1
