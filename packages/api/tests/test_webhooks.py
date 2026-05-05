"""Tests for webhook delivery — signature, retry, outcome handling."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from pps_api.services.webhooks import (
    DeliveryOutcome,
    WebhookDeliverer,
    compute_signature,
    verify_signature,
)

# ---------- Signature ----------


class TestSignature:
    def test_compute_signature_format(self):
        sig = compute_signature(b'{"a":1}', "secret", timestamp="1700000000")
        assert sig.startswith("sha256=")
        assert len(sig) == 7 + 64  # sha256= + 64 hex chars

    def test_signature_is_deterministic(self):
        a = compute_signature(b"x", "k", timestamp="1")
        b = compute_signature(b"x", "k", timestamp="1")
        assert a == b

    def test_signature_changes_with_body(self):
        a = compute_signature(b"x", "k", timestamp="1")
        b = compute_signature(b"y", "k", timestamp="1")
        assert a != b

    def test_signature_changes_with_timestamp(self):
        a = compute_signature(b"x", "k", timestamp="1")
        b = compute_signature(b"x", "k", timestamp="2")
        assert a != b

    def test_verify_valid(self):
        ts = str(int(datetime.now(UTC).timestamp()))
        sig = compute_signature(b"hello", "k", timestamp=ts)
        assert verify_signature(b"hello", "k", timestamp=ts, signature=sig) is True

    def test_verify_wrong_secret(self):
        ts = str(int(datetime.now(UTC).timestamp()))
        sig = compute_signature(b"hello", "k", timestamp=ts)
        assert verify_signature(b"hello", "different", timestamp=ts, signature=sig) is False

    def test_verify_wrong_body(self):
        ts = str(int(datetime.now(UTC).timestamp()))
        sig = compute_signature(b"hello", "k", timestamp=ts)
        assert verify_signature(b"different", "k", timestamp=ts, signature=sig) is False

    def test_verify_old_timestamp_rejected(self):
        # 10 minutes ago — outside 5-minute default window
        old_ts = str(int(datetime.now(UTC).timestamp()) - 600)
        sig = compute_signature(b"hello", "k", timestamp=old_ts)
        assert verify_signature(b"hello", "k", timestamp=old_ts, signature=sig) is False

    def test_verify_malformed_timestamp(self):
        assert verify_signature(b"x", "k", timestamp="not-a-number", signature="sha256=00") is False


# ---------- Delivery happy path ----------


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_2xx_delivers_first_attempt():
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request.content,
            }
        )
        return httpx.Response(200, content=b'{"ok":true}')

    deliverer = WebhookDeliverer(transport=_mock_transport(handler))
    result = await deliverer.deliver(
        url="https://customer.example/hook",
        event="job.completed",
        payload={"job_id": "abc", "status": "completed"},
        secret="topsecret",
    )

    assert result.outcome is DeliveryOutcome.delivered
    assert len(result.attempts) == 1
    assert result.attempts[0].status_code == 200
    assert len(received) == 1
    sent = received[0]
    assert sent["headers"]["x-pps-event"] == "job.completed"
    assert sent["headers"]["x-pps-signature"].startswith("sha256=")
    assert sent["headers"]["x-pps-timestamp"]
    payload = json.loads(sent["body"])
    assert payload["job_id"] == "abc"


@pytest.mark.asyncio
async def test_5xx_then_2xx_succeeds_on_retry():
    state = {"attempts": 0}
    delays_seen: list[float] = []

    async def fake_sleep(s: float) -> None:
        delays_seen.append(s)

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] < 3:
            return httpx.Response(503, content=b"down")
        return httpx.Response(200, content=b"ok")

    deliverer = WebhookDeliverer(
        sleep_fn=fake_sleep,
        transport=_mock_transport(handler),
    )
    result = await deliverer.deliver(
        url="https://x.example/h",
        event="job.completed",
        payload={"a": 1},
        secret="k",
        retry_delays=(0.01, 0.01, 0.01, 0.01, 0.01),
    )
    assert result.outcome is DeliveryOutcome.delivered
    assert len(result.attempts) == 3
    assert result.attempts[0].status_code == 503
    assert result.attempts[1].status_code == 503
    assert result.attempts[2].status_code == 200
    assert delays_seen == [0.01, 0.01]  # sleep before attempts 2 + 3


@pytest.mark.asyncio
async def test_4xx_gives_up_immediately():
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        return httpx.Response(404, content=b"not found")

    deliverer = WebhookDeliverer(transport=_mock_transport(handler))
    result = await deliverer.deliver(
        url="https://x.example/h",
        event="job.completed",
        payload={"a": 1},
        secret="k",
    )
    assert result.outcome is DeliveryOutcome.failed_4xx
    assert len(result.attempts) == 1
    assert state["attempts"] == 1


@pytest.mark.asyncio
async def test_408_429_keep_retrying():
    """408 (request timeout) and 429 (too many requests) are retryable."""
    state = {"attempts": 0}

    async def fake_sleep(s: float) -> None:
        pass

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] == 1:
            return httpx.Response(429, content=b"slow down")
        if state["attempts"] == 2:
            return httpx.Response(408, content=b"timeout")
        return httpx.Response(200, content=b"ok")

    deliverer = WebhookDeliverer(
        sleep_fn=fake_sleep,
        transport=_mock_transport(handler),
    )
    result = await deliverer.deliver(
        url="https://x.example/h",
        event="x",
        payload={"a": 1},
        secret="k",
        retry_delays=(0.01,) * 5,
    )
    assert result.outcome is DeliveryOutcome.delivered
    assert len(result.attempts) == 3


@pytest.mark.asyncio
async def test_all_5xx_exhausts_budget():
    async def fake_sleep(s: float) -> None:
        pass

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"down")

    deliverer = WebhookDeliverer(
        sleep_fn=fake_sleep,
        transport=_mock_transport(handler),
    )
    result = await deliverer.deliver(
        url="https://x.example/h",
        event="x",
        payload={"a": 1},
        secret="k",
        retry_delays=(0.01, 0.01),
    )
    assert result.outcome is DeliveryOutcome.failed_exhausted
    assert len(result.attempts) == 3  # initial + 2 retries
    assert all(a.status_code == 500 for a in result.attempts)


@pytest.mark.asyncio
async def test_signature_is_verifiable_by_recipient():
    """Round-trip: sign with deliverer, verify with recipient logic."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        captured["sig"] = request.headers["x-pps-signature"]
        captured["ts"] = request.headers["x-pps-timestamp"]
        return httpx.Response(200)

    deliverer = WebhookDeliverer(transport=_mock_transport(handler))
    await deliverer.deliver(
        url="https://x.example/h",
        event="job.completed",
        payload={"job_id": "abc"},
        secret="shared-secret",
    )
    assert (
        verify_signature(
            captured["body"],  # type: ignore[arg-type]
            "shared-secret",
            timestamp=captured["ts"],  # type: ignore[arg-type]
            signature=captured["sig"],  # type: ignore[arg-type]
        )
        is True
    )
