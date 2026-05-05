"""Pro Photo Studio — API gateway + worker.

Submodules:
    main        FastAPI application factory + uvicorn entry point
    config      Pydantic Settings (loaded from env)
    routers     /v1/* endpoints
    tasks       Celery worker + job dispatcher
    db          SQLAlchemy 2 async + Alembic migrations
    storage     S3-compatible artifact storage
    integrations Slack, Dropbox, GCS, Stripe, Clerk
    security    Auth, rate limit, audit log, webhook signatures
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
