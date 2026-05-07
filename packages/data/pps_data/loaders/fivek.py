"""MIT-Adobe FiveK loader.

5,000 RAW DNG photos retouched by 5 photographers (experts A-E). Each row
yields the original RAW + the 5 expert renditions. We stream from a public HF
mirror — we do **not** re-host. License is research-only.

Default mirror: ``logasja/mit-adobe-fivek`` (configs ``a``..``e`` per expert,
~1.4k downloads). Override via the ``mirror`` argument or ``PPS_FIVEK_REPO``
env variable. Some mirrors (e.g. ``yuukicammy/MIT-Adobe-FiveK``) expose all
experts as columns of one config — the ``config`` arg lets callers override.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from ._common import load_streaming

log = logging.getLogger(__name__)

ExpertId = Literal["a", "b", "c", "d", "e"]
FIVEK_EXPERTS: tuple[ExpertId, ...] = ("a", "b", "c", "d", "e")

DEFAULT_MIRROR = "logasja/mit-adobe-fivek"


def stream_fivek(
    expert: ExpertId = "c",
    *,
    split: str = "train",
    mirror: str | None = None,
    token: str | None = None,
    config: str | None = None,
) -> Any:
    """Stream FiveK pairs: (raw, expert_<expert>).

    Expert C is the canonical "Bible" target for retouch fine-tunes —
    consistent across the dataset, neither over- nor under-processed.

    Args:
        expert: which expert's edit to use as the target. Defaults to "c".
            For mirrors that expose one config per expert (the default
            ``logasja/mit-adobe-fivek`` does), this is also passed as the
            HF ``name`` (config) argument.
        split: HF split name. Most mirrors expose only "train".
        mirror: HF repo id. Defaults to ``DEFAULT_MIRROR`` or env
            ``PPS_FIVEK_REPO`` if set.
        token: HF token. Falls back to ``HF_TOKEN`` env if None.
        config: explicit HF config (``name=`` arg). When ``None`` we default to
            ``expert`` for the canonical ``logasja/...`` layout. Pass an empty
            string to suppress (mirrors with a single default config).

    Returns:
        ``datasets.IterableDataset`` — each row is a dict whose exact keys
        depend on the mirror but always includes a RAW image and the chosen
        expert's edit.
    """
    if expert not in FIVEK_EXPERTS:
        raise ValueError(f"expert must be one of {FIVEK_EXPERTS}, got {expert!r}")
    repo = mirror or os.environ.get("PPS_FIVEK_REPO", DEFAULT_MIRROR)
    tok = token or os.environ.get("HF_TOKEN")
    cfg: str | None
    if config is None:
        cfg = expert  # default layout: one config per expert
    elif config == "":
        cfg = None
    else:
        cfg = config
    log.info(
        "streaming FiveK from %s (expert=%s, config=%s, split=%s)",
        repo,
        expert,
        cfg,
        split,
    )
    return load_streaming(repo, split=split, config=cfg, token=tok)
