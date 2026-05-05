"""Storage Protocol + StorageResult value object.

Every backend (local FS, S3, R2, MinIO) implements ``ObjectStorage`` so
upstream code never knows where bytes physically live. Read paths return
either an inline byte payload OR a presigned URL the client can use to
download directly from the bucket — choosing whichever is cheaper for
the deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StorageResult:
    """Outcome of a successful PUT.

    Attributes:
        key: The storage key used to retrieve the object later. Format
            varies by backend: ``"<job_id>/<filename>"`` for local,
            ``"s3://bucket/<job_id>/<filename>"`` for S3.
        size_bytes: Size on disk / wire. Useful for billing + storage caps.
        content_type: MIME type stored with the object.
        url: Public or presigned URL if applicable, else None. Local
            backend always returns None — caller serves files via the API.
    """

    key: str
    size_bytes: int
    content_type: str
    url: str | None = None


@runtime_checkable
class ObjectStorage(Protocol):
    """Async object storage interface."""

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> StorageResult:
        """Upload ``data`` under ``key``. Overwrites existing objects."""
        ...

    async def get(self, key: str) -> bytes:
        """Fetch the full object as bytes. Raises ``FileNotFoundError`` on miss."""
        ...

    async def delete(self, key: str) -> bool:
        """Remove the object. Returns True if deleted, False if not present."""
        ...

    async def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        """Return a time-limited URL for direct download.

        Returns None for backends that don't support presigning (e.g. local
        filesystem) — caller should fall back to streaming via the API.
        """
        ...

    async def exists(self, key: str) -> bool: ...
