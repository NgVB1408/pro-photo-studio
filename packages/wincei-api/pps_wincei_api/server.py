"""Main FastAPI app + entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .__version__ import __version__
from .config import settings
from .routers import hdr, health, jobs, masks, recovery, regions, ui, window_ceiling

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="pps-wincei-api",
        description=(
            "REST API service cho Pro Photo Studio Wincei Stack.\n\n"
            "**Endpoints:**\n"
            "- `POST /api/v1/window-ceiling` — fix cửa sổ blown + trần ám màu\n"
            "- `POST /api/v1/hdr-fuse` — Mertens HDR bracket fusion\n"
            "- `POST /api/v1/segment-masks` — phân vùng AI + phào chỉ + QC verdict\n"
            "- `GET /api/v1/jobs/{id}` — job status\n"
            "- `GET /api/v1/jobs/{id}/download` — download zip output\n\n"
            "Mọi endpoint POST có flag `mock=true` để test integration nhanh."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(window_ceiling.router)
    app.include_router(hdr.router)
    app.include_router(masks.router)
    app.include_router(regions.router)
    app.include_router(recovery.router)
    app.include_router(jobs.router)
    app.include_router(ui.router)

    return app


app = create_app()


def run() -> None:
    """CLI entry point: pps-wincei-api."""
    import uvicorn

    uvicorn.run(
        "pps_wincei_api.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run()
