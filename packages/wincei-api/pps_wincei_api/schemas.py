"""Pydantic models cho request/response."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobType(str, enum.Enum):
    window_ceiling = "window_ceiling"
    hdr_fuse = "hdr_fuse"
    segment_masks = "segment_masks"


class WindowCeilingRequest(BaseModel):
    """Params cho /window-ceiling. File upload qua multipart, params query string."""
    window_strength: Optional[float] = Field(None, ge=0, le=2, description="0..1.5 (None=AI tự quyết)")
    ceiling_strength: Optional[float] = Field(None, ge=0, le=1, description="0..1 (None=AI tự quyết)")
    include_lamps: Optional[bool] = None
    self_evaluate: bool = True
    mock: bool = Field(False, description="Trả stub trong 100ms cho integration test")


class HDRFuseRequest(BaseModel):
    align: bool = True
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = Field(1.0, description="Giảm xuống 0.4 để pull outdoor mạnh hơn")
    gamma: float = 1.0
    mock: bool = False


class SegmentMasksRequest(BaseModel):
    refine_edges: bool = True
    detect_molding: bool = True
    include_lights: bool = False
    write_overlay: bool = True
    write_tiff: bool = True
    write_psd: bool = False
    mock: bool = False


class JobInfo(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress_pct: float = 0.0
    message: str = ""
    error: Optional[str] = None
    inputs: list[str] = []
    outputs: list[str] = []
    metadata: dict[str, Any] = {}

    @property
    def download_url(self) -> str:
        return f"/api/v1/jobs/{self.job_id}/download"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "pps-wincei-api"
    version: str
    packages: dict[str, str]
    gpu_available: bool


class ProcessResponse(BaseModel):
    """Sync mode trả ngay; async mode trả job_id."""
    mode: str  # "sync" or "async"
    job_id: Optional[str] = None
    job: Optional[JobInfo] = None
    output_url: Optional[str] = None
    eval: Optional[dict[str, Any]] = None
    mock: bool = False
