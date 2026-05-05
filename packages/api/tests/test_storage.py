"""Tests for storage backends.

LocalFileStorage: full coverage with a temp directory.
S3Storage: focused tests on URL handling + path-traversal safety. Network
calls to actual S3 / MinIO are out-of-scope for CI; integration tests
will run against MinIO via docker-compose in Phase 2.3.5.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pps_api.storage import LocalFileStorage, ObjectStorage, S3Storage

# ---------- LocalFileStorage ----------


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path)


@pytest.mark.asyncio
class TestLocalPutGet:
    async def test_put_then_get(self, storage):
        result = await storage.put("a/b/c.jpg", b"hello", content_type="image/jpeg")
        assert result.key == "a/b/c.jpg"
        assert result.size_bytes == 5
        assert result.content_type == "image/jpeg"
        assert result.url is None
        got = await storage.get("a/b/c.jpg")
        assert got == b"hello"

    async def test_put_overwrites(self, storage):
        await storage.put("k.txt", b"first")
        await storage.put("k.txt", b"second")
        assert (await storage.get("k.txt")) == b"second"

    async def test_get_missing_raises_filenotfounderror(self, storage):
        with pytest.raises(FileNotFoundError):
            await storage.get("does-not-exist")

    async def test_exists(self, storage):
        assert await storage.exists("k.txt") is False
        await storage.put("k.txt", b"x")
        assert await storage.exists("k.txt") is True

    async def test_delete(self, storage):
        await storage.put("k.txt", b"x")
        assert await storage.delete("k.txt") is True
        assert await storage.exists("k.txt") is False
        assert await storage.delete("k.txt") is False  # second delete idempotent

    async def test_presigned_url_returns_none(self, storage):
        await storage.put("k.txt", b"x")
        assert await storage.presigned_url("k.txt") is None


class TestLocalPathTraversal:
    """Critical: keys must not escape the storage root."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalFileStorage:
        return LocalFileStorage(tmp_path)

    @pytest.mark.parametrize(
        "bad_key",
        [
            "",
            "/abs/path/file.jpg",
            "../escape.jpg",
            "../../etc/passwd",
            "valid/../../escape.jpg",
            "valid/../escape.jpg",
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_bad_keys(self, storage, bad_key):
        with pytest.raises(ValueError):
            await storage.put(bad_key, b"x")


# ---------- S3Storage (unit-level only — no network) ----------


class TestS3Construction:
    def test_protocol_compliance(self):
        # S3Storage must satisfy the ObjectStorage Protocol structurally.
        s3 = S3Storage(bucket="test", endpoint_url="http://localhost:9000")
        assert isinstance(s3, ObjectStorage)

    def test_client_kwargs_default_region(self):
        s3 = S3Storage(bucket="b")
        kwargs = s3._client_kwargs()
        assert kwargs["region_name"] == "auto"
        assert "endpoint_url" not in kwargs

    def test_client_kwargs_with_credentials(self):
        s3 = S3Storage(
            bucket="b",
            endpoint_url="http://minio:9000",
            access_key="ak",
            secret_key="sk",
            region="us-east-1",
        )
        kwargs = s3._client_kwargs()
        assert kwargs["region_name"] == "us-east-1"
        assert kwargs["endpoint_url"] == "http://minio:9000"
        assert kwargs["aws_access_key_id"] == "ak"
        assert kwargs["aws_secret_access_key"] == "sk"


class TestProtocolStructural:
    """LocalFileStorage and S3Storage both implement ObjectStorage."""

    def test_local_satisfies_protocol(self, tmp_path):
        s = LocalFileStorage(tmp_path)
        assert isinstance(s, ObjectStorage)
