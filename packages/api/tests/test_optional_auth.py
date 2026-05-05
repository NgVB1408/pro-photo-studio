"""Tests for the conditional API key dependency on /v1/jobs.

The router uses ``maybe_require_api_key`` which checks
``Settings.require_api_key()``. We exercise both modes by overriding
``get_settings`` at the app level.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pps_api.config import Settings, get_settings
from pps_api.main import create_app
from pps_api.routers.jobs import get_output_dir, get_store
from pps_api.security import (
    InMemoryAPIKeyStore,
    generate_api_key,
)
from pps_api.security.api_key import get_api_key_store
from pps_api.services import InMemoryJobStore


def _build_client(settings: Settings, output_dir: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: InMemoryJobStore()
    app.dependency_overrides[get_output_dir] = lambda: output_dir
    return TestClient(app)


@pytest.fixture
def settings_dev(tmp_path: Path) -> Settings:
    return Settings(
        pps_env="development",
        pps_require_api_key=False,
        database_url="sqlite+aiosqlite:///:memory:",
    )


@pytest.fixture
def settings_locked(tmp_path: Path) -> Settings:
    return Settings(
        pps_env="staging",
        pps_require_api_key=True,
        database_url="sqlite+aiosqlite:///:memory:",
    )


class TestAuthDisabled:
    def test_list_jobs_no_key_required(self, settings_dev: Settings, tmp_path: Path):
        with _build_client(settings_dev, tmp_path) as client:
            resp = client.get("/v1/jobs")
            assert resp.status_code == 200
            assert resp.json() == []


class TestAuthEnabled:
    def test_list_jobs_rejects_missing_key(
        self, settings_locked: Settings, tmp_path: Path
    ):
        with _build_client(settings_locked, tmp_path) as client:
            resp = client.get("/v1/jobs")
            assert resp.status_code == 401
            assert "Missing X-API-Key" in resp.json()["detail"]

    def test_list_jobs_rejects_garbage_key(
        self, settings_locked: Settings, tmp_path: Path
    ):
        with _build_client(settings_locked, tmp_path) as client:
            resp = client.get("/v1/jobs", headers={"X-API-Key": "not-a-real-key"})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_list_jobs_accepts_valid_key(
        self, settings_locked: Settings, tmp_path: Path
    ):
        store = InMemoryAPIKeyStore()
        api_key, record = generate_api_key(env="staging", name="test")
        await store.create(record)

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings_locked
        app.dependency_overrides[get_store] = lambda: InMemoryJobStore()
        app.dependency_overrides[get_output_dir] = lambda: tmp_path
        app.dependency_overrides[get_api_key_store] = lambda: store

        with TestClient(app) as client:
            resp = client.get("/v1/jobs", headers={"X-API-Key": api_key.raw})
            assert resp.status_code == 200
            assert resp.json() == []
