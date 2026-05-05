# Deployment

## Local development stack

One command brings up Postgres + Redis + MinIO + the FastAPI app:

```bash
docker compose -f deploy/docker-compose.dev.yml up --build
```

Endpoints once everything is healthy:

| Service | URL | Credentials |
|---|---|---|
| API + Swagger | http://localhost:8000/docs | (none in dev) |
| API health | http://localhost:8000/health | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| MinIO S3 API | http://localhost:9000 | (use S3 SDK) |
| Postgres | `localhost:5432` | `pps` / `pps` / db `pps` |
| Redis | `localhost:6379` | — |

The bucket `pps-dev` is created automatically by the `minio-init` one-shot
service. Drop into a clean state with:

```bash
docker compose -f deploy/docker-compose.dev.yml down -v
```

## Production image

The production image is `deploy/docker/Dockerfile.api`. Highlights:

- Multi-stage build — `python:3.11-slim-bookworm` base, ~140 MB final image
- Non-root user `pps:1001`
- gunicorn + uvicorn workers (4 by default; tune with `WORKERS=`)
- HEALTHCHECK hits `/health` every 30 s
- `libgl1 + libglib2.0-0` installed for OpenCV runtime

Build standalone:

```bash
docker build \
    -t ghcr.io/ngvb1408/pps-api:dev \
    -f deploy/docker/Dockerfile.api \
    .
```

Run with prod-grade env vars:

```bash
docker run --rm -p 8000:8000 \
    -e PPS_ENV=production \
    -e DATABASE_URL=postgresql+asyncpg://user:pass@host/db \
    -e REDIS_URL=redis://host:6379/0 \
    -e S3_BUCKET=pps-prod \
    -e S3_REGION=auto \
    -e PPS_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
    ghcr.io/ngvb1408/pps-api:dev
```

## Smoke test against the running stack

```bash
# Health
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","env":"development"}

# Submit a no-op job (identity stage; multipart form)
curl -X POST http://localhost:8000/v1/jobs \
     -F 'image=@samples/test.jpg' \
     -F 'body={"stages":["identity"],"seed":42}'
```

## Coming next (Phase 6)

- `deploy/k8s/` — Kubernetes manifests (Deployment + Service + HPA)
- `deploy/runpod/` — RunPod serverless template for GPU worker pool
- `deploy/terraform/` — R2 bucket + Cloudflare DNS module
