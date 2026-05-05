"""Artifact storage — Protocol + LocalFileStorage + S3Storage.

Same interface for local-filesystem dev and S3-compatible production
(AWS S3, Cloudflare R2, MinIO, Wasabi). API layer is unchanged when
swapping backends.
"""

from __future__ import annotations

from .base import ObjectStorage, StorageResult
from .local import LocalFileStorage
from .s3 import S3Storage

__all__ = [
    "LocalFileStorage",
    "ObjectStorage",
    "S3Storage",
    "StorageResult",
]
