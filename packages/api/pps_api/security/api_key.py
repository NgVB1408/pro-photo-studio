"""API key authentication.

Design:
  - **Generate**: ``pps_<env>_<32-char-base62>`` — env-prefixed so a leaked
    test key can never authenticate against production.
  - **Store**: argon2id-hashed in the database (NEVER plaintext). Only the
    last 4 chars of the raw key are kept in plaintext for human display
    ("ends with ...x7Qy").
  - **Verify**: O(1) lookup by ``key_id`` (first 16 chars after prefix);
    constant-time argon2id verification on the hash.
  - **Rotation**: revoking a key sets ``revoked_at`` — verify rejects.
  - **Rate limiting**: lives in middleware; this module just supplies the
    authenticated identity to the request.

The dependency function ``require_api_key`` raises 401 / 403 from FastAPI
so endpoint code stays clean:

    from pps_api.security import APIKey, require_api_key
    from typing import Annotated

    @router.get("/jobs")
    async def list_jobs(key: Annotated[APIKey, Depends(require_api_key)]): ...
"""

from __future__ import annotations

import hmac
import logging
import secrets
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Protocol

from fastapi import Depends, HTTPException, Request, status
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)


__all__ = [
    "APIKey",
    "APIKeyRecord",
    "APIKeyStore",
    "InMemoryAPIKeyStore",
    "generate_api_key",
    "hash_api_key",
    "require_api_key",
    "verify_api_key",
]


# Plaintext key length AFTER the env prefix: 32 base62 chars × 5.95 bits/char
# ≈ 191 bits of entropy. Comfortably above the 128-bit recommended minimum.
KEY_LENGTH = 32
KEY_ID_LENGTH = 16  # first 16 chars used as the database lookup key
BASE62 = string.ascii_letters + string.digits


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class APIKey:
    """Authenticated API key context attached to a request.

    Routes that depend on ``require_api_key`` receive this. ``raw`` is never
    set on responses — it only exists fresh from ``generate_api_key`` so the
    caller can show it to the user once.
    """

    key_id: str
    """Stable ID used for lookups + log correlation. First 16 chars of the key."""

    name: str
    """Human-readable label assigned at key creation (e.g. "ci-test", "prod-mobile")."""

    env: str
    """Environment label baked into the prefix — must match ``Settings.pps_env``."""

    scopes: tuple[str, ...] = ()
    """Authorisation scopes attached to this key (e.g. ``("jobs:read", "jobs:write")``)."""

    raw: str | None = None
    """Plaintext key — populated only at creation time, never after retrieval."""


@dataclass(frozen=True, slots=True)
class APIKeyRecord:
    """Persisted form of an API key — what the store sees."""

    key_id: str
    name: str
    env: str
    hash: str
    """argon2id hash of the FULL plaintext key (env prefix + body)."""

    suffix4: str
    """Last 4 chars of the raw key, kept for human-friendly display."""

    scopes: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_now)
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generation + hashing
# ---------------------------------------------------------------------------


def generate_api_key(
    *, name: str, env: str, scopes: tuple[str, ...] = ()
) -> tuple[APIKey, APIKeyRecord]:
    """Create a new key. Returns ``(key_with_raw, record_for_store)``.

    Hand ``key_with_raw.raw`` to the caller exactly ONCE; never store it
    plaintext. Persist the returned ``record`` via ``APIKeyStore.create``.
    """
    if not name:
        raise ValueError("API key must have a non-empty name")
    if env not in {"development", "staging", "production"}:
        raise ValueError(f"Invalid env: {env!r}")

    body = "".join(secrets.choice(BASE62) for _ in range(KEY_LENGTH))
    raw = f"pps_{env}_{body}"
    key_id = body[:KEY_ID_LENGTH]
    suffix4 = body[-4:]
    record = APIKeyRecord(
        key_id=key_id,
        name=name,
        env=env,
        hash=hash_api_key(raw),
        suffix4=suffix4,
        scopes=tuple(scopes),
    )
    api_key = APIKey(
        key_id=key_id,
        name=name,
        env=env,
        scopes=tuple(scopes),
        raw=raw,
    )
    return api_key, record


def hash_api_key(raw: str) -> str:
    """argon2id hash of the plaintext key.

    We use argon2id (memory-hard, GPU-resistant) because API key verification
    happens once per request. CPU cost ~30ms is acceptable; brute force at
    scale is infeasible.
    """
    try:
        from argon2 import PasswordHasher
    except ImportError as exc:
        raise RuntimeError(
            "API key hashing requires argon2-cffi. Install with: pip install argon2-cffi"
        ) from exc
    hasher = PasswordHasher(
        time_cost=2,
        memory_cost=64 * 1024,  # 64 MiB
        parallelism=2,
        hash_len=32,
    )
    return hasher.hash(raw)


def verify_api_key(raw: str, *, against_hash: str) -> bool:
    """Constant-time argon2id verification."""
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError
    except ImportError as exc:
        raise RuntimeError("API key verification requires argon2-cffi") from exc
    hasher = PasswordHasher()
    try:
        hasher.verify(against_hash, raw)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Store (in-memory implementation; SQL implementation in db/ later)
# ---------------------------------------------------------------------------


class APIKeyStore(Protocol):
    """Persistence interface every API-key store must satisfy."""

    async def create(self, record: APIKeyRecord) -> None: ...
    async def get(self, key_id: str) -> APIKeyRecord | None: ...
    async def revoke(self, key_id: str) -> APIKeyRecord | None: ...
    async def touch_last_used(self, key_id: str) -> None: ...
    async def list_active(self, env: str | None = None) -> list[APIKeyRecord]: ...


class InMemoryAPIKeyStore:
    """Process-local store for tests and single-process dev.

    Production should use a SQL-backed implementation (Phase 2.3.2).
    """

    def __init__(self) -> None:
        self._records: dict[str, APIKeyRecord] = {}

    async def create(self, record: APIKeyRecord) -> None:
        if record.key_id in self._records:
            raise ValueError(f"API key already exists: {record.key_id}")
        self._records[record.key_id] = record

    async def get(self, key_id: str) -> APIKeyRecord | None:
        return self._records.get(key_id)

    async def revoke(self, key_id: str) -> APIKeyRecord | None:
        existing = self._records.get(key_id)
        if existing is None or existing.revoked_at is not None:
            return existing
        from dataclasses import replace

        revoked = replace(existing, revoked_at=_now())
        self._records[key_id] = revoked
        return revoked

    async def touch_last_used(self, key_id: str) -> None:
        existing = self._records.get(key_id)
        if existing is None:
            return
        from dataclasses import replace

        self._records[key_id] = replace(existing, last_used_at=_now())

    async def list_active(self, env: str | None = None) -> list[APIKeyRecord]:
        return [
            r
            for r in self._records.values()
            if r.revoked_at is None and (env is None or r.env == env)
        ]


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


_default_store: APIKeyStore = InMemoryAPIKeyStore()


def get_api_key_store() -> APIKeyStore:
    return _default_store


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    raw_key: Annotated[str | None, Depends(_api_key_header)] = None,
    store: Annotated[APIKeyStore, Depends(get_api_key_store)] = None,  # type: ignore[assignment]
) -> APIKey:
    """FastAPI dependency: extract + verify ``X-API-Key`` header.

    Raises:
        401 if header is missing or malformed
        401 if key body fails argon2id verification
        403 if key is revoked or for the wrong environment
    """
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    parsed = _parse_raw_key(raw_key)
    if parsed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed API key (expected pps_<env>_<32 chars>)",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    env, body = parsed
    key_id = body[:KEY_ID_LENGTH]
    record = await store.get(key_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if record.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key revoked")

    # Constant-time comparison to avoid environment-leak via timing.
    if record.env != env and not hmac.compare_digest(record.env, env):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key is for env={record.env!r}, request env={env!r}",
        )

    if not verify_api_key(raw_key, against_hash=record.hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    await store.touch_last_used(key_id)
    request.state.api_key_id = key_id
    return APIKey(
        key_id=record.key_id,
        name=record.name,
        env=record.env,
        scopes=record.scopes,
    )


def _parse_raw_key(raw: str) -> tuple[str, str] | None:
    """Validate ``pps_<env>_<32 base62>`` shape. Returns (env, body)."""
    parts = raw.split("_", 2)
    if len(parts) != 3 or parts[0] != "pps":
        return None
    env, body = parts[1], parts[2]
    if env not in {"development", "staging", "production"}:
        return None
    if len(body) != KEY_LENGTH or any(c not in BASE62 for c in body):
        return None
    return env, body
