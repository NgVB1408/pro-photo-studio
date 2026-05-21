"""GET /api/v1/health — health + version + GPU detect."""

from __future__ import annotations

from fastapi import APIRouter

from ..__version__ import __version__
from ..schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    packages = {}
    try:
        from pps_wincei import __version__ as v
        packages["pps_wincei"] = v
    except Exception:
        packages["pps_wincei"] = "missing"
    try:
        from pps_wincei_hdr import __version__ as v
        packages["pps_wincei_hdr"] = v
    except Exception:
        packages["pps_wincei_hdr"] = "missing"
    try:
        from pps_wincei_masks import __version__ as v
        packages["pps_wincei_masks"] = v
    except Exception:
        packages["pps_wincei_masks"] = "missing"

    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        pass

    return HealthResponse(
        version=__version__,
        packages=packages,
        gpu_available=gpu,
    )
