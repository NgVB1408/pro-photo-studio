"""Shared dataclasses + protocols for the agent pipeline.

Design: each specialist agent splits its work into ``analyze(ctx) -> StagePlan``
and ``apply(image, plan) -> (image_after, StageReport)``. All ``analyze`` calls
run concurrently on the original image (analysis is independent and CPU-bound,
OpenCV/numpy release the GIL). ``apply`` runs serially in deterministic order
(geometry → light → micro-contrast → cleanup → output) so each stage sees a
consistent pixel grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

PropertyType = Literal[
    "villa_luxury",
    "apartment_modern",
    "studio_minimal",
    "commercial_showroom",
    "twilight_cabin",
    "generic",
]

SceneTag = Literal["interior", "exterior", "aerial", "unknown"]


@dataclass
class JobContext:
    """Read-only snapshot every analyze() shares."""

    image: np.ndarray  # BGR uint8, original (post RAW decode)
    image_path: str | None = None
    seed: int | None = None
    target_long_edge: int = 7680  # 8K = 7680 px
    target_dpi: int = 300
    property_type: PropertyType = "generic"
    scene_tag: SceneTag = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class StagePlan:
    """Output of analyze(). Contains parameters + masks the apply() step needs.

    A plan with ``skip=True`` is a no-op at apply time — the agent decided this
    stage is unnecessary (e.g. no sky in the image, no halos to fix).
    """

    name: str
    operations: list[dict] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    analyze_duration_s: float = 0.0
    skip: bool = False
    skip_reason: str = ""


@dataclass
class StageReport:
    """Output of apply()."""

    name: str
    duration_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class _Timer:
    """Context manager timer used by agents."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "_Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
