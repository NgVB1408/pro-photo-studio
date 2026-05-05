"""Shared fixtures for pps-api tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pps_api.main import create_app
from pps_api.routers.jobs import get_output_dir, get_store
from pps_api.services import InMemoryJobStore


@pytest.fixture
def output_dir() -> Iterator[Path]:
    """Per-test output directory — cleaned up automatically."""
    with TemporaryDirectory(prefix="pps-out-") as td:
        yield Path(td)


@pytest.fixture
def client(output_dir: Path) -> Iterator[TestClient]:
    """FastAPI test client with a fresh in-memory store and isolated output dir."""
    app = create_app()
    fresh_store = InMemoryJobStore()
    app.dependency_overrides[get_store] = lambda: fresh_store
    app.dependency_overrides[get_output_dir] = lambda: output_dir
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """64×96 BGR test image encoded as JPEG."""
    img = np.full((64, 96, 3), 128, dtype=np.uint8)
    # Add some structure so preflight doesn't trivially flag it
    img[16:48, 24:72] = (200, 180, 100)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()
