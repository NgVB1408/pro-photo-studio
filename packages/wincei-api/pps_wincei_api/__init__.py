"""pps-wincei-api — FastAPI REST service wrap 3 packages.

Endpoints:
    POST /api/v1/window-ceiling    fix cửa sổ + trần (pps-wincei)
    POST /api/v1/hdr-fuse          HDR Mertens fusion (pps-wincei-hdr)
    POST /api/v1/segment-masks     phân vùng + phào chỉ (pps-wincei-masks)
    GET  /api/v1/jobs/{id}         job status (async mode)
    GET  /api/v1/jobs/{id}/download  download zip output
    GET  /api/v1/health            health check
    GET  /docs                     Swagger UI
    GET  /redoc                    ReDoc
"""

from .__version__ import __version__

__all__ = ["__version__"]
