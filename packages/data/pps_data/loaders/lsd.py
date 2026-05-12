"""LSD — Large-scale Scene De-scattering dataset loader.

Hazy / scattered outdoor scenes paired with clean ground truth. Useful when
training a real-estate exterior dehaze model. Default mirror points to a
public HF mirror — override via ``PPS_LSD_REPO`` env or ``mirror`` arg.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ._common import load_streaming

log = logging.getLogger(__name__)

DEFAULT_MIRROR = "fffiloni/LSD-Dataset"


def stream_lsd(
    *,
    split: str = "train",
    mirror: str | None = None,
    token: str | None = None,
) -> Any:
    repo = mirror or os.environ.get("PPS_LSD_REPO", DEFAULT_MIRROR)
    tok = token or os.environ.get("HF_TOKEN")
    log.info("streaming LSD from %s (split=%s)", repo, split)
    return load_streaming(repo, split=split, token=tok)
