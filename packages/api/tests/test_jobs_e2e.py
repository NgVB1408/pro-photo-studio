"""End-to-end tests for the job lifecycle.

These tests use FastAPI's ``TestClient`` which runs the app in-process,
including BackgroundTasks. The pipeline runs synchronously inside the
request handler's lifecycle so by the time the test polls
``GET /v1/jobs/{id}``, the job has reached terminal state.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pps_api.schemas import JobStatus


def _submit(client: TestClient, image: bytes, body: dict) -> str:
    """POST a job, return its id."""
    resp = client.post(
        "/v1/jobs",
        files={"image": ("input.jpg", image, "image/jpeg")},
        data={"body": json.dumps(body)},
    )
    assert resp.status_code == 202, resp.text
    payload = resp.json()
    assert payload["status"] in ("queued", "running", "completed")
    return payload["job_id"]


# ---------- happy paths ----------


class TestSubmitAndPoll:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["env"]

    def test_identity_stage_completes(self, client: TestClient, sample_jpeg_bytes: bytes):
        job_id = _submit(client, sample_jpeg_bytes, {"stages": ["identity"]})

        resp = client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == JobStatus.completed.value
        assert body["report"] is not None
        assert body["report"]["stages"][0]["name"] == "identity"
        assert body["report"]["stages"][0]["applied"] is True
        assert body["result_url"] == f"/v1/jobs/{job_id}/result"

    def test_result_download(self, client: TestClient, sample_jpeg_bytes: bytes):
        job_id = _submit(client, sample_jpeg_bytes, {"stages": ["identity"]})
        resp = client.get(f"/v1/jobs/{job_id}/result")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/jpeg")
        assert len(resp.content) > 100

    def test_no_stages_returns_input_unchanged(self, client: TestClient, sample_jpeg_bytes: bytes):
        job_id = _submit(client, sample_jpeg_bytes, {"stages": []})
        resp = client.get(f"/v1/jobs/{job_id}")
        body = resp.json()
        assert body["status"] == JobStatus.completed.value
        assert body["report"]["stages"] == []

    def test_seed_propagated(self, client: TestClient, sample_jpeg_bytes: bytes):
        job_id_1 = _submit(client, sample_jpeg_bytes, {"stages": ["identity"], "seed": 42})
        job_id_2 = _submit(client, sample_jpeg_bytes, {"stages": ["identity"], "seed": 42})

        # Both should complete; result downloads should be byte-identical.
        r1 = client.get(f"/v1/jobs/{job_id_1}/result").content
        r2 = client.get(f"/v1/jobs/{job_id_2}/result").content
        assert r1 == r2

    def test_real_estate_stage_runs(self, client: TestClient, sample_jpeg_bytes: bytes):
        # Real-estate pipeline is heavier but still pure-CV; should complete.
        job_id = _submit(
            client,
            sample_jpeg_bytes,
            {"stages": ["real_estate"], "params": {"real_estate": {"enable_sky": False}}},
        )
        resp = client.get(f"/v1/jobs/{job_id}")
        body = resp.json()
        assert body["status"] == JobStatus.completed.value


# ---------- list / inspect ----------


class TestListAndInspect:
    def test_list_recent_returns_submitted_jobs(self, client: TestClient, sample_jpeg_bytes: bytes):
        ids = [_submit(client, sample_jpeg_bytes, {"stages": ["identity"]}) for _ in range(3)]
        resp = client.get("/v1/jobs?limit=10")
        assert resp.status_code == 200
        body = resp.json()
        returned_ids = [j["job_id"] for j in body]
        for i in ids:
            assert i in returned_ids

    def test_list_limit_validation(self, client: TestClient):
        assert client.get("/v1/jobs?limit=0").status_code == 400
        assert client.get("/v1/jobs?limit=501").status_code == 400


# ---------- error cases ----------


class TestErrorCases:
    def test_unknown_job_404(self, client: TestClient):
        assert client.get("/v1/jobs/does-not-exist").status_code == 404
        assert client.get("/v1/jobs/does-not-exist/result").status_code == 404

    def test_corrupt_image_400(self, client: TestClient):
        resp = client.post(
            "/v1/jobs",
            files={"image": ("input.jpg", b"not-an-image", "image/jpeg")},
            data={"body": json.dumps({"stages": ["identity"]})},
        )
        assert resp.status_code == 400
        assert "decode" in resp.json()["detail"].lower()

    def test_empty_image_400(self, client: TestClient):
        resp = client.post(
            "/v1/jobs",
            files={"image": ("input.jpg", b"", "image/jpeg")},
            data={"body": json.dumps({"stages": ["identity"]})},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_malformed_body_400(self, client: TestClient, sample_jpeg_bytes: bytes):
        resp = client.post(
            "/v1/jobs",
            files={"image": ("input.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"body": "{not valid json"},
        )
        assert resp.status_code == 400

    def test_unknown_stage_skipped_not_failed(self, client: TestClient, sample_jpeg_bytes: bytes):
        job_id = _submit(client, sample_jpeg_bytes, {"stages": ["nope_does_not_exist"]})
        resp = client.get(f"/v1/jobs/{job_id}")
        body = resp.json()
        assert body["status"] == JobStatus.completed.value
        assert body["report"]["stages"][0]["skipped"] is True
        assert "not registered" in body["report"]["stages"][0]["reason"]

    def test_negative_seed_400(self, client: TestClient, sample_jpeg_bytes: bytes):
        resp = client.post(
            "/v1/jobs",
            files={"image": ("input.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"body": json.dumps({"stages": ["identity"], "seed": -1})},
        )
        assert resp.status_code == 400


# ---------- result not yet ready ----------


def test_result_409_when_not_completed(client: TestClient):
    """If a job is failed (no result_path), result endpoint returns 409."""
    import asyncio

    from pps_api.routers.jobs import get_store
    from pps_api.services import JobRecord

    store = client.app.dependency_overrides[get_store]()  # type: ignore[union-attr]

    async def _setup() -> None:
        await store.create(JobRecord(job_id="fail-1", status=JobStatus.failed, error="boom"))

    asyncio.run(_setup())
    resp = client.get("/v1/jobs/fail-1/result")
    assert resp.status_code == 409
