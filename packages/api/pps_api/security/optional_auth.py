"""Conditionally-enforced API key auth.

The ``maybe_require_api_key`` dependency consults ``Settings.require_api_key()``.
When auth is disabled (default for development + tests) it returns ``None``
without touching the request. When enabled, it delegates to ``require_api_key``
and either returns the resolved ``APIKey`` or raises 401 / 403.

Routers that should be protected in production but open in dev wire it up like:

    @router.post("/jobs", dependencies=[Depends(maybe_require_api_key)])
    async def create_job(...): ...

This keeps existing test fixtures (no API key headers) green while still
hardening prod deployments by default.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security.api_key import APIKeyHeader

from pps_api.config import Settings, get_settings

from .api_key import APIKey, APIKeyStore, get_api_key_store, require_api_key

_optional_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def maybe_require_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    raw_key: Annotated[str | None, Depends(_optional_header)] = None,
    store: Annotated[APIKeyStore, Depends(get_api_key_store)] = None,  # type: ignore[assignment]
) -> APIKey | None:
    if not settings.require_api_key():
        return None
    return await require_api_key(request=request, raw_key=raw_key, store=store)


__all__ = ["maybe_require_api_key"]
