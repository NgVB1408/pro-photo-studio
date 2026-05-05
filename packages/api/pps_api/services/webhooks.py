"""Webhook delivery — HMAC-signed POST with exponential backoff retry.

When a job reaches a terminal state (``completed`` or ``failed``), the
runner enqueues a webhook delivery if the customer registered a callback
URL. Each delivery:

1. Builds the JSON payload (job_id, status, report summary, result_url).
2. Computes ``X-PPS-Signature`` = HMAC-SHA256 of the body using the
   per-customer shared secret. The customer verifies this on receipt to
   reject spoofed deliveries.
3. POSTs with timeout 10s. 2xx = delivered.
4. On 4xx (other than 408/429) we give up — caller's URL is wrong.
5. On 5xx, 408, 429, or network error we retry with exponential backoff
   (5 attempts, 2s/8s/30s/120s/300s).
6. Every attempt is logged via ``WebhookAttempt`` so customers can debug
   delivery in the dashboard.

The module is async and uses ``httpx.AsyncClient``. Tests use mock
transport so no real network calls are made.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


__all__ = [
    "DeliveryOutcome",
    "WebhookAttempt",
    "WebhookDeliverer",
    "WebhookDelivery",
    "compute_signature",
    "verify_signature",
]


# Retry schedule in seconds. Total budget ≈ 8 min. Calibrated for typical
# customer endpoints that may have brief outages but recover quickly.
RETRY_DELAYS_S = (2, 8, 30, 120, 300)
TIMEOUT_S = 10.0
SIGNATURE_HEADER = "X-PPS-Signature"
TIMESTAMP_HEADER = "X-PPS-Timestamp"
EVENT_HEADER = "X-PPS-Event"


class DeliveryOutcome(StrEnum):
    delivered = "delivered"
    """2xx response within retry budget."""

    failed_4xx = "failed_4xx"
    """Caller URL rejected the payload (gave up early)."""

    failed_exhausted = "failed_exhausted"
    """All retries exhausted with 5xx / network errors."""


@dataclass(frozen=True, slots=True)
class WebhookAttempt:
    """One delivery attempt — recorded for the customer's audit log."""

    attempt_no: int  # 1-based
    sent_at: datetime
    status_code: int | None
    """HTTP status if a response was received; None on transport error."""

    response_snippet: str
    """First 256 chars of the response body — for debugging."""

    error: str | None = None
    """Transport-level error message (timeout, DNS failure, etc.) when present."""

    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class WebhookDelivery:
    """Outcome of a complete delivery (with retries)."""

    url: str
    event: str
    outcome: DeliveryOutcome
    attempts: tuple[WebhookAttempt, ...]


def compute_signature(body: bytes, secret: str, *, timestamp: str) -> str:
    """HMAC-SHA256 of ``timestamp + "." + body``.

    Including the timestamp in the signing string defends against replay:
    customer rejects deliveries where ``X-PPS-Timestamp`` is older than 5
    minutes.
    """
    msg = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(
    body: bytes,
    secret: str,
    *,
    timestamp: str,
    signature: str,
    max_age_seconds: int = 300,
) -> bool:
    """Verify an incoming webhook (used by SDK / customer code).

    Returns False if signature mismatches or ``timestamp`` is outside the
    allowed window. Constant-time comparison.
    """
    try:
        ts_int = int(timestamp)
    except (TypeError, ValueError):
        return False
    now = int(datetime.now(UTC).timestamp())
    if abs(now - ts_int) > max_age_seconds:
        return False
    expected = compute_signature(body, secret, timestamp=timestamp)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# WebhookDeliverer
# ---------------------------------------------------------------------------


@dataclass
class WebhookDeliverer:
    """Async webhook sender. One instance per process is fine.

    The sender is stateless apart from the optional ``transport`` used in
    tests (e.g. ``httpx.MockTransport``). In production leave it None and
    a fresh ``httpx.AsyncClient`` is built per delivery.
    """

    sleep_fn: Callable[[float], Awaitable[None]] = field(default_factory=lambda: asyncio.sleep)
    transport: httpx.BaseTransport | None = None

    async def deliver(
        self,
        *,
        url: str,
        event: str,
        payload: dict[str, Any],
        secret: str,
        retry_delays: tuple[int, ...] = RETRY_DELAYS_S,
    ) -> WebhookDelivery:
        """Send ``payload`` to ``url`` with retries until delivered or exhausted."""
        body = _serialise(payload)
        attempts: list[WebhookAttempt] = []
        max_attempts = 1 + len(retry_delays)

        for i in range(max_attempts):
            timestamp = str(int(datetime.now(UTC).timestamp()))
            sig = compute_signature(body, secret, timestamp=timestamp)
            attempt_no = i + 1
            t0 = datetime.now(UTC)

            status_code, snippet, err = await self._post_once(
                url=url,
                body=body,
                signature=sig,
                timestamp=timestamp,
                event=event,
            )
            duration_ms = (datetime.now(UTC) - t0).total_seconds() * 1000.0
            attempt = WebhookAttempt(
                attempt_no=attempt_no,
                sent_at=t0,
                status_code=status_code,
                response_snippet=snippet,
                error=err,
                duration_ms=duration_ms,
            )
            attempts.append(attempt)

            # 2xx → done.
            if status_code is not None and 200 <= status_code < 300:
                logger.info(
                    "webhook delivered url=%s event=%s attempt=%d code=%d",
                    url,
                    event,
                    attempt_no,
                    status_code,
                )
                return WebhookDelivery(
                    url=url,
                    event=event,
                    outcome=DeliveryOutcome.delivered,
                    attempts=tuple(attempts),
                )

            # 4xx (other than 408 timeout, 429 too many) → give up early.
            if (
                status_code is not None
                and 400 <= status_code < 500
                and status_code not in (408, 429)
            ):
                logger.warning(
                    "webhook 4xx, giving up url=%s code=%d body=%s",
                    url,
                    status_code,
                    snippet[:100],
                )
                return WebhookDelivery(
                    url=url,
                    event=event,
                    outcome=DeliveryOutcome.failed_4xx,
                    attempts=tuple(attempts),
                )

            # Otherwise: 5xx / 408 / 429 / network error → retry if budget remains.
            if i < len(retry_delays):
                delay = retry_delays[i]
                logger.info(
                    "webhook retry url=%s attempt=%d/%d in %ds (code=%s err=%s)",
                    url,
                    attempt_no,
                    max_attempts,
                    delay,
                    status_code,
                    err,
                )
                await self.sleep_fn(delay)

        logger.error(
            "webhook exhausted url=%s event=%s attempts=%d",
            url,
            event,
            len(attempts),
        )
        return WebhookDelivery(
            url=url,
            event=event,
            outcome=DeliveryOutcome.failed_exhausted,
            attempts=tuple(attempts),
        )

    async def _post_once(
        self,
        *,
        url: str,
        body: bytes,
        signature: str,
        timestamp: str,
        event: str,
    ) -> tuple[int | None, str, str | None]:
        """Single POST attempt. Returns (status_code, snippet, transport_error)."""
        headers = {
            "Content-Type": "application/json",
            SIGNATURE_HEADER: signature,
            TIMESTAMP_HEADER: timestamp,
            EVENT_HEADER: event,
            "User-Agent": "pps-webhooks/1.0",
        }
        transport_kw = {"transport": self.transport} if self.transport else {}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S, **transport_kw) as client:
                resp = await client.post(url, content=body, headers=headers)
            return resp.status_code, resp.text[:256], None
        except httpx.HTTPError as exc:
            return None, "", f"{type(exc).__name__}: {exc}"


def _serialise(payload: dict[str, Any]) -> bytes:
    """Stable JSON serialisation — sorted keys so signatures are reproducible."""
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
