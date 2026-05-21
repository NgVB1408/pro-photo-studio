"""Runtime config + storage paths."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    """Loaded từ env vars hoặc default."""

    storage_root: Path = Path(os.environ.get("WINCEI_STORAGE", "./wincei_storage")).resolve()
    max_upload_mb: int = int(os.environ.get("WINCEI_MAX_UPLOAD_MB", "200"))
    mock_default: bool = os.environ.get("WINCEI_MOCK_DEFAULT", "false").lower() == "true"
    api_key: str | None = os.environ.get("WINCEI_API_KEY")  # None = no auth
    cors_origins: list[str] = (
        os.environ.get("WINCEI_CORS", "*").split(",")
        if os.environ.get("WINCEI_CORS") else ["*"]
    )
    host: str = os.environ.get("WINCEI_HOST", "0.0.0.0")
    port: int = int(os.environ.get("WINCEI_PORT", "8000"))

    @property
    def uploads_dir(self) -> Path:
        d = self.storage_root / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def outputs_dir(self) -> Path:
        d = self.storage_root / "outputs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def jobs_dir(self) -> Path:
        d = self.storage_root / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
