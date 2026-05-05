# pps-api

FastAPI gateway + Celery worker + Postgres + S3 for Pro Photo Studio.

This package is the public HTTP surface for the platform. It depends on
`pps-core` for the actual pipeline work and exposes job submission,
inspection, and result retrieval over REST.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/docs` | Swagger UI (non-prod only) |
| POST | `/v1/jobs` | Submit a job (multipart: image + JSON body) |
| GET | `/v1/jobs` | List recent jobs |
| GET | `/v1/jobs/{id}` | Status + per-stage report |
| GET | `/v1/jobs/{id}/result` | Download final image |

## Quick start (dev)

```bash
uv pip install -e packages/core
uv pip install -e packages/api
uv run pps-api  # http://localhost:8000/docs
```

## Built-in stages

Adapters in `pps_api/stages/builtin_stages.py` register the following
pipeline stages on import:

- `preflight` — blur/exposure/dimension QC
- `real_estate` — scene-aware RE pipeline (sky/lawn/window/vertical/classify)
- `twilight` — Day → Sunset transform
- `perspective` — Hough VP upright correction
- `identity` — no-op (sanity check)

ML-backed stages live in `pps-ai` and register only when that package is
imported by the deployment.

## Configuration

All settings come from environment variables. See `.env.example` at the
repo root for the full list.

## Tests

```bash
uv pip install -e packages/api pytest httpx
uv run pytest packages/api/tests
```

E2E tests use FastAPI's `TestClient` — no external services required.
