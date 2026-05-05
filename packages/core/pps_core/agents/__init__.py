"""Post-production agent roster — a virtual photo studio.

Each agent module owns one quality dimension (exposure, white balance, sharpness,
verticals, halo, sky, noise, colour, composition). They share a common
Protocol so the orchestrator can route a photo through the whole crew without
knowing the internals of any one specialist.

Public surface
--------------
``StudioOrchestrator``
    Runs the full crew against an image and returns ``StudioReport`` —
    a structured artefact suitable for serialisation to JSON and rendering
    on the customer-facing web portal.

``PostProductionAgent``
    The Protocol every agent satisfies. Use it as a type hint when wiring
    custom agents (e.g. an ML-backed agent loaded from Colab weights).

``DEFAULT_ROSTER``
    The hand-picked default set, ordered for sensible dependencies (verticals
    first because all subsequent metrics depend on geometry being correct;
    halo last because it inspects the final pixels).

Importing this package does not load any heavy ML models. Each ML-backed
agent (when added in Phase 3) deferred-imports its weights on first use.
"""

from __future__ import annotations

from .base import (
    AgentApplyReport,
    AgentChecklistItem,
    AgentEvaluation,
    AgentReport,
    PostProductionAgent,
    StudioReport,
)
from .color import ColorAgent
from .composition import CompositionAgent
from .exposure import ExposureAgent
from .halo import HaloAgent
from .noise import NoiseAgent
from .orchestrator import DEFAULT_ROSTER, StudioOrchestrator
from .sharpness import SharpnessAgent
from .sky import SkyAgent
from .vertical import VerticalAgent
from .white_balance import WhiteBalanceAgent

__all__ = [
    "DEFAULT_ROSTER",
    "AgentApplyReport",
    "AgentChecklistItem",
    "AgentEvaluation",
    "AgentReport",
    "ColorAgent",
    "CompositionAgent",
    "ExposureAgent",
    "HaloAgent",
    "NoiseAgent",
    "PostProductionAgent",
    "SharpnessAgent",
    "SkyAgent",
    "StudioOrchestrator",
    "StudioReport",
    "VerticalAgent",
    "WhiteBalanceAgent",
]
