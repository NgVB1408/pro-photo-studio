"""S3-compatible storage (AWS S3, Cloudflare R2, MinIO, Wasabi).

All providers expose the same S3 v4 API; we use ``aiobotocore`` so the
blocking boto3 client can be awaited from FastAPI handlers. The same
client works for AWS S3, R2 (with ``endpoint_url=https://...r2.cloudflarestorage.com``),
and MinIO local dev (``endpoint_url=http://localhost:9000``).

Keys follow the convention ``<env>/<job_id>/<filename>`` so production
and staging artefacts cannot collide in a shared bucket.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import StorageResult

logger = logging.getLogger(__name__)


class S3Storage:
    """Async S3-compatible object storage."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "auto",
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"region_name": self._region}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        if self._access_key:
            kwargs["aws_access_key_id"] = self._access_key
        if self._secret_key:
            kwargs["aws_secret_access_key"] = self._secret_key
        return kwargs

    def _session(self):
        try:
            from aiobotocore.session import get_session
        except ImportError as exc:
            raise RuntimeError(
                "S3Storage requires aiobotocore. Install with: pip install aiobotocore"
            ) from exc
        return get_session()

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> StorageResult:
        session = self._session()
        async with session.create_client("s3", **self._client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        return StorageResult(
            key=f"s3://{self._bucket}/{key}",
            size_bytes=len(data),
            content_type=content_type,
            url=None,
        )

    async def get(self, key: str) -> bytes:
        session = self._session()
        async with session.create_client("s3", **self._client_kwargs()) as s3:
            try:
                resp = await s3.get_object(Bucket=self._bucket, Key=key)
            except Exception as exc:
                # botocore.exceptions.ClientError code NoSuchKey
                code = getattr(getattr(exc, "response", None), "get", lambda *_: None)("Error", {})
                if isinstance(code, dict) and code.get("Code") == "NoSuchKey":
                    raise FileNotFoundError(f"S3 key not found: {key}") from exc
                raise
            async with resp["Body"] as stream:
                return await stream.read()

    async def delete(self, key: str) -> bool:
        if not await self.exists(key):
            return False
        session = self._session()
        async with session.create_client("s3", **self._client_kwargs()) as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)
        return True

    async def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        session = self._session()
        async with session.create_client("s3", **self._client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )

    async def exists(self, key: str) -> bool:
        session = self._session()
        async with session.create_client("s3", **self._client_kwargs()) as s3:
            try:
                await s3.head_object(Bucket=self._bucket, Key=key)
                return True
            except Exception:
                return False
