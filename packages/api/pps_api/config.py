"""Application configuration loaded from environment.

Single source of truth for all runtime parameters. Settings are read once at
process start; mutating an instance has no effect on already-imported modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic Settings — reads .env + environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    pps_env: Literal["development", "staging", "production"] = "development"
    pps_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    pps_secret_key: SecretStr = Field(
        default=SecretStr("changeme-min-32-chars-required-for-prod"),
        description="HMAC signing key — must be ≥32 chars in production.",
    )

    # Database
    database_url: str = "postgresql+asyncpg://pps:pps@localhost:5432/pps"

    # Redis (Celery broker + cache)
    redis_url: str = "redis://localhost:6379/0"

    # S3 storage
    s3_endpoint_url: str | None = None
    s3_bucket: str = "pps-dev"
    s3_access_key: SecretStr | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str = "auto"

    # HuggingFace (optional, for ML)
    hf_token: SecretStr | None = None
    hf_home: str = "./.hf_cache"

    # Stripe (optional)
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None

    # Clerk auth (optional)
    clerk_secret_key: SecretStr | None = None
    clerk_publishable_key: str | None = None

    # Webhooks
    slack_webhook_url: str | None = None
    sentry_dsn: str | None = None

    # CORS — comma-separated origins. Empty in production = lock everything down.
    # Example: "https://portal.propho.studio,https://www.propho.studio"
    pps_allowed_origins: str = ""

    # When true, /v1/jobs/* requires X-API-Key. Default true in production, false
    # elsewhere — tests and local dev should not need to mint a key on every run.
    pps_require_api_key: bool | None = None

    def is_production(self) -> bool:
        return self.pps_env == "production"

    def require_api_key(self) -> bool:
        """Resolve auth requirement.

        Explicit env override wins. Otherwise: production = required, others = open.
        """
        if self.pps_require_api_key is not None:
            return self.pps_require_api_key
        return self.is_production()

    def cors_origins(self) -> list[str]:
        """Resolve allowed CORS origins.

        Development → wildcard for ergonomics. Production → only the origins
        explicitly listed in ``PPS_ALLOWED_ORIGINS``. An empty value in
        production means no browser may call the API directly (server-to-server
        only via API key), which is fine when a Next.js portal proxies requests.
        """
        if not self.is_production():
            return ["*"]
        return [o.strip() for o in self.pps_allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    Use Pydantic dependency injection in FastAPI:
        @app.get("/foo")
        def foo(settings: Annotated[Settings, Depends(get_settings)]): ...
    """
    return Settings()
