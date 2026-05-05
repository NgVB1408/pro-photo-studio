"""Local filesystem storage — for tests and single-host deployments.

All keys map to ``<root>/<key>`` on disk. Path traversal protection is
enforced: keys containing ``..`` or absolute paths are rejected with
``ValueError`` before any filesystem call.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .base import StorageResult

logger = logging.getLogger(__name__)


class LocalFileStorage:
    """Filesystem-backed implementation of ``ObjectStorage``."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"Invalid storage key: {key!r}")
        path = (self._root / key).resolve()
        # Defence in depth: ensure resolved path is still under root.
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"Key escapes storage root: {key!r}") from exc
        return path

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> StorageResult:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # offload blocking write to default executor
        await asyncio.get_running_loop().run_in_executor(None, path.write_bytes, data)
        return StorageResult(
            key=key,
            size_bytes=len(data),
            content_type=content_type,
            url=None,
        )

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)

    async def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.is_file():
            return False
        await asyncio.get_running_loop().run_in_executor(None, path.unlink)
        return True

    async def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        # Local backend has no public URL — caller streams via the API.
        return None

    async def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()
