"""Multi-agent orchestration: 5 specialists ↔ SOP, Director QC, parallel execute."""

from .director import DirectorAgent, DirectorReview
from .orchestrator import Orchestrator, PipelineResult
from .types import (
    JobContext,
    PropertyType,
    StagePlan,
    StageReport,
)

__all__ = [
    "DirectorAgent",
    "DirectorReview",
    "JobContext",
    "Orchestrator",
    "PipelineResult",
    "PropertyType",
    "StagePlan",
    "StageReport",
]

__version__ = "0.1.0"
