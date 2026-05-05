"""Colab-trained model adapters.

This sub-package wraps weights produced by the user's training notebooks
(``training/notebooks/*.ipynb``) so they can plug into the agent roster
without disturbing the OpenCV-only baseline.

Each adapter loads its weights lazily on first use and exposes the same
``PostProductionAgent`` Protocol as the baseline agents — so the orchestrator
treats them interchangeably.

Currently registered:

  * :class:`ColabRealEstateAgent` — Real-estate fine-tune from
    ``Train-AI-DNG-JPG-BĐS.ipynb``. Augments the colour + sharpness specialists
    when its weights are present.
  * :class:`ColabQwenInstructionAgent` — Qwen-Image-Lightning instruction
    editor (``Qwen-Image-Lightning.ipynb``). Activated when the user supplies
    a natural-language prompt via the API.

Both adapters short-circuit to a no-op when their weights are not on disk.
This keeps tests + CI green without GPU credits and makes the production
deployment a one-line decision (mount the weights volume, or don't).
"""

from __future__ import annotations

from .adapter import (
    ColabAdapterStatus,
    ColabModelManifest,
    ColabQwenInstructionAgent,
    ColabRealEstateAgent,
    discover_manifests,
)

__all__ = [
    "ColabAdapterStatus",
    "ColabModelManifest",
    "ColabQwenInstructionAgent",
    "ColabRealEstateAgent",
    "discover_manifests",
]
