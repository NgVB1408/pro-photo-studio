"""FastAPI application factory and uvicorn entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pps_api import __version__
from pps_api.config import Settings, get_settings

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup + shutdown hooks."""
    settings = get_settings()
    logging.basicConfig(level=settings.pps_log_level)
    logger.info("Starting Pro Photo Studio API v%s (%s)", __version__, settings.pps_env)
    if settings.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.pps_env)
        except ImportError:
            logger.warning("sentry_sdk not installed; skipping Sentry init")
    yield
    logger.info("Shutting down Pro Photo Studio API")


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Routes are split across `pps_api.routers.*` and registered here.
    """
    settings = get_settings()
    app = FastAPI(
        title="Pro Photo Studio API",
        description="Production-grade real-estate photo enhancement.",
        version=__version__,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS — locked down in production by APP_ALLOWED_ORIGINS env (TBD).
    allowed_origins = ["*"] if not settings.is_production() else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
    async def health(s: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, env=s.pps_env)

    # Register built-in pipeline stages so the API has something to run on
    # day one. ML-backed stages register themselves only when pps_ai is
    # imported, which we leave to the deployment to opt into.
    from pps_api.routers import jobs as jobs_router
    from pps_api.stages import builtin_stages  # noqa: F401  (registration via import)

    app.include_router(jobs_router.router)

    return app


app = create_app()


def run() -> None:
    """uvicorn entry point — `pps-api` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "pps_api.main:app",
        host="0.0.0.0",  # noqa: S104 — bind all in containers
        port=8000,
        reload=not settings.is_production(),
        log_level=settings.pps_log_level.lower(),
    )


if __name__ == "__main__":
    run()
