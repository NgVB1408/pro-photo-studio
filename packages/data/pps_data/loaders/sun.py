"""SUN Database loader — scene labels (kitchen / bedroom / living room ...).

Used by `pps_core.realestate.classify_scene` as a label-grounded reference
during eval, and as a stratification key when sampling fine-tune subsets.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ._common import load_streaming

log = logging.getLogger(__name__)

DEFAULT_MIRROR = "VicharVision/sun397"


def stream_sun(
    *,
    split: str = "train",
    config: str | None = None,
    mirror: str | None = None,
    token: str | None = None,
) -> Any:
    repo = mirror or os.environ.get("PPS_SUN_REPO", DEFAULT_MIRROR)
    tok = token or os.environ.get("HF_TOKEN")
    log.info("streaming SUN from %s (split=%s, config=%s)", repo, split, config)
    return load_streaming(repo, split=split, config=config, token=tok)
