"""Unit tests for pps_api.security.api_key."""

from __future__ import annotations

from datetime import datetime

import pytest
from pps_api.security import (
    APIKey,
    InMemoryAPIKeyStore,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from pps_api.security.api_key import _parse_raw_key

# ---------- generate_api_key ----------


class TestGenerate:
    def test_basic_generation(self):
        key, record = generate_api_key(name="ci", env="development", scopes=("jobs:read",))
        assert key.raw is not None
        assert key.raw.startswith("pps_development_")
        assert len(key.raw.split("_", 2)[2]) == 32
        assert key.name == "ci"
        assert key.env == "development"
        assert key.scopes == ("jobs:read",)
        assert record.hash != key.raw  # not plaintext
        assert record.suffix4 == key.raw[-4:]
        assert record.key_id == key.raw.split("_", 2)[2][:16]

    def test_two_calls_produce_different_keys(self):
        a, _ = generate_api_key(name="a", env="development")
        b, _ = generate_api_key(name="b", env="development")
        assert a.raw != b.raw
        assert a.key_id != b.key_id

    def test_invalid_env_rejected(self):
        with pytest.raises(ValueError):
            generate_api_key(name="x", env="prod")  # not in allowlist

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            generate_api_key(name="", env="development")


# ---------- hash + verify ----------


class TestHashAndVerify:
    def test_verify_correct_key(self):
        key, record = generate_api_key(name="t", env="development")
        assert verify_api_key(key.raw, against_hash=record.hash) is True  # type: ignore[arg-type]

    def test_verify_wrong_key(self):
        _, record = generate_api_key(name="t", env="development")
        assert verify_api_key("pps_development_" + "X" * 32, against_hash=record.hash) is False

    def test_verify_corrupted_hash(self):
        # Truly malformed hash returns False, never raises.
        assert verify_api_key("anything", against_hash="not-a-real-hash") is False

    def test_hash_is_argon2id(self):
        h = hash_api_key("pps_development_" + "a" * 32)
        assert h.startswith("$argon2id$")


# ---------- _parse_raw_key ----------


class TestParseRawKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("pps_development_" + "a" * 32, ("development", "a" * 32)),
            ("pps_production_" + "0" * 32, ("production", "0" * 32)),
            ("pps_staging_" + "Z" * 32, ("staging", "Z" * 32)),
        ],
    )
    def test_valid_keys(self, raw, expected):
        assert _parse_raw_key(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "ppsdevelopmentaaaa",
            "pps_dev_" + "a" * 32,  # bad env
            "pps_development_" + "a" * 31,  # short
            "pps_development_" + "a" * 33,  # long
            "wrong_development_" + "a" * 32,  # wrong prefix
            "pps_development_" + "a" * 31 + "!",  # non-base62
        ],
    )
    def test_invalid_keys(self, raw):
        assert _parse_raw_key(raw) is None


# ---------- InMemoryAPIKeyStore ----------


class TestStore:
    @pytest.fixture
    def store(self):
        return InMemoryAPIKeyStore()

    @pytest.fixture
    def key_and_record(self):
        return generate_api_key(name="store-test", env="development")

    @pytest.mark.asyncio
    async def test_create_and_get(self, store, key_and_record):
        _, record = key_and_record
        await store.create(record)
        got = await store.get(record.key_id)
        assert got is not None
        assert got.key_id == record.key_id

    @pytest.mark.asyncio
    async def test_create_duplicate_raises(self, store, key_and_record):
        _, record = key_and_record
        await store.create(record)
        with pytest.raises(ValueError):
            await store.create(record)

    @pytest.mark.asyncio
    async def test_revoke(self, store, key_and_record):
        _, record = key_and_record
        await store.create(record)
        revoked = await store.revoke(record.key_id)
        assert revoked is not None
        assert revoked.revoked_at is not None
        assert isinstance(revoked.revoked_at, datetime)

    @pytest.mark.asyncio
    async def test_revoke_unknown_returns_none(self, store):
        result = await store.revoke("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_touch_last_used(self, store, key_and_record):
        _, record = key_and_record
        await store.create(record)
        assert (await store.get(record.key_id)).last_used_at is None  # type: ignore[union-attr]
        await store.touch_last_used(record.key_id)
        got = await store.get(record.key_id)
        assert got is not None
        assert got.last_used_at is not None

    @pytest.mark.asyncio
    async def test_list_active(self, store):
        _, r1 = generate_api_key(name="dev1", env="development")
        _, r2 = generate_api_key(name="dev2", env="development")
        _, r3 = generate_api_key(name="prod1", env="production")
        await store.create(r1)
        await store.create(r2)
        await store.create(r3)
        await store.revoke(r1.key_id)

        all_active = await store.list_active()
        assert len(all_active) == 2
        prod_only = await store.list_active(env="production")
        assert len(prod_only) == 1
        assert prod_only[0].name == "prod1"


# ---------- require_api_key (FastAPI dependency, integration via TestClient) ----------


class TestRequireAPIKey:
    """Use the existing TestClient fixture in conftest to drive the full FastAPI stack."""

    @pytest.fixture
    def authed_client(self, client):
        """Override the API key store + register a tiny protected test route."""
        from fastapi import Depends
        from pps_api.security.api_key import (
            get_api_key_store,
            require_api_key,
        )

        store = InMemoryAPIKeyStore()
        client.app.dependency_overrides[get_api_key_store] = lambda: store

        # Idempotent route registration — pytest fixtures may be invoked many
        # times with the same app; only add the route once.
        if not any(getattr(r, "path", None) == "/v1/_test_protected" for r in client.app.routes):

            @client.app.get("/v1/_test_protected")
            async def protected(key: APIKey = Depends(require_api_key)):  # noqa: B008
                return {"key_name": key.name}

        return client, store

    def test_no_header_returns_401(self, authed_client):
        client, _ = authed_client
        resp = client.get("/v1/_test_protected")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_malformed_header_returns_401(self, authed_client):
        client, _ = authed_client
        resp = client.get("/v1/_test_protected", headers={"X-API-Key": "garbage"})
        assert resp.status_code == 401

    def test_unknown_key_returns_401(self, authed_client):
        client, _ = authed_client
        fake = "pps_development_" + "z" * 32
        resp = client.get("/v1/_test_protected", headers={"X-API-Key": fake})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_authenticates(self, authed_client):

        client, store = authed_client
        key, record = generate_api_key(name="ci-test", env="development", scopes=("read",))
        await store.create(record)
        resp = client.get("/v1/_test_protected", headers={"X-API-Key": key.raw})
        assert resp.status_code == 200
        assert resp.json()["key_name"] == "ci-test"

    @pytest.mark.asyncio
    async def test_revoked_key_returns_403(self, authed_client):
        client, store = authed_client
        key, record = generate_api_key(name="revoked", env="development")
        await store.create(record)
        await store.revoke(record.key_id)
        resp = client.get("/v1/_test_protected", headers={"X-API-Key": key.raw})
        assert resp.status_code == 403
        assert "revoked" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_wrong_env_returns_403(self, authed_client):
        client, store = authed_client
        # Create a production key — request will use it but our header will
        # be parsed with whatever env is in the prefix; if record env mismatches
        # the request env, 403.
        prod_key, prod_record = generate_api_key(name="prod-only", env="production")
        # Inject record but with different env in the persisted form to trigger mismatch.
        # We achieve this by creating a development-prefix raw key but storing
        # under a 'production' record (simulates DB drift / leaked test key).
        from dataclasses import replace

        dev_record = replace(prod_record, env="development")
        await store.create(dev_record)
        resp = client.get("/v1/_test_protected", headers={"X-API-Key": prod_key.raw})
        assert resp.status_code == 403
        assert "env" in resp.json()["detail"].lower()
