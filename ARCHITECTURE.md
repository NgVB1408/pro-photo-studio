# Architecture

## Goals

1. **One pipeline, many entry points** — Web, desktop, CLI, and public API all
   converge on the same deterministic stage graph. No drift between surfaces.
2. **CPU-fast path, GPU-best path** — All features have a CPU OpenCV implementation.
   GPU ML is opt-in for higher quality. A single job description routes to the right
   workers.
3. **Production from day one** — Auth, rate limiting, audit logs, observability,
   and billing are first-class, not afterthoughts.
4. **Open-source core, optional commercial weights** — Pipeline is Apache 2.0.
   Proprietary fine-tuned LoRAs (e.g. real-estate-specific staging) ship as a
   separate add-on.

## High-level flow

```
                  ┌────────────────────────────────────────────────────┐
   Customer ─────►│  Web Portal (Next.js)  /  Desktop (PySide6)        │
                  │  /  Public REST API   /  Slack/email integration   │
                  └────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────────────────┐
                  │  FastAPI Gateway: auth + rate limit + webhooks     │
                  └────────────────────┬───────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────────────┐
                  │  Celery Job Queue (Redis broker)                   │
                  │   ┌── CPU workers (preflight, OpenCV pipeline)     │
                  │   └── GPU workers (Qwen / SUPIR / SAM2 / SD3.5)    │
                  └────────────────────┬───────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────────────┐
                  │  Pipeline Engine — deterministic stage graph       │
                  └────────────────────┬───────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────────────┐
                  │  Postgres (jobs, billing, audit) + S3 (artifacts)  │
                  └────────────────────────────────────────────────────┘
```

## Pipeline stage graph

Stages are functions `(image, ctx) -> (image, report)`. The runner picks stages
based on the job description; each stage is independently testable.

```
0  Preflight QC          (blur, exposure, dimension, focus uniformity)
1  RAW decode            (rawpy / libraw if DNG/RAF/CR2/NEF/ARW)
2  Lens correction       (Brown-Conrady)
3  Perspective upright   (Hough vanishing point)
4  Bracket fuse          (Mertens + deghost + LAB normalize)
5  Scene classify        (interior/exterior/aerial)
6  Object removal        (SAM 2 click-mask + LaMa or AI inpaint)
7  Color enhance         (RE pipeline: WB, CLAHE, highlight/shadow, vibrance)
8  Window pull           (pseudo-HDR or diffusion refine)
9  Sky replace           (ControlNet LoRA, procedural fallback)
10 Lawn enhance          (HSV selective)
11 Virtual staging       (SD3.5 + IPAdapter, opt-in)
12 Twilight transform    (sky composite + warm tone, opt-in)
13 Tone coherency        (batch LAB anchor)
14 Selective sharpen     (saliency-aware)
15 AI upscale            (SUPIR / Real-ESRGAN ncnn)
16 Auto privacy          (face/plate blur, GDPR audit log)
17 Output encode         (JPEG / PNG / WebP, EXIF preserve, color profile)
```

## Package boundaries

| Package | Responsibility | Key deps |
|---|---|---|
| `pps_core` | OpenCV pipeline, deterministic stages | numpy, opencv-python, pillow, rawpy |
| `pps_ai` | ML inference wrappers | torch, diffusers, transformers, segment-anything-2 |
| `pps_api` | FastAPI + Celery + DB + storage | fastapi, celery, sqlalchemy, boto3 |
| `pps_desktop` | PySide6 client | PySide6, pps_core (direct) |
| `pps_web` | Next.js + React 19 | next, react, tailwind, shadcn/ui |

`pps_core` has zero ML dependencies. Adding ML to a stage means writing
`pps_ai.<name>(image)` and the stage selecting between `cv` / `ai` based on the
job description.

## Job lifecycle

```
[POST /v1/jobs] ── persisted as {status: queued, ...}
       │
       ▼
[Celery dispatcher] ── routes to cpu_pool or gpu_pool based on stages requested
       │
       ▼
[Worker.run(job)] ── streams progress via Redis pub-sub → SSE → client
       │
       ▼
[Pipeline.execute] ── per-stage timing + report
       │
       ▼
[S3 upload] ── presigned URL written to job.result_url
       │
       ▼
[Webhook delivery] ── POST to job.callback_url with signed payload
       │
       ▼
[Postgres update] ── status: completed, artifacts: {...}
```

## Determinism

Every job carries a `seed: int | None`. Set it for reproducible output. The
pipeline propagates the seed to every stage that uses RNG (twilight noise,
diffusion sampling, etc.). Same input + same seed + same git SHA ⇒ same output.

## Storage layout

```
s3://pro-photo-studio-<env>/
  uploads/<job_id>/<filename>           original input
  artifacts/<job_id>/<stage>/<filename> intermediate (debug only, retention 7d)
  results/<job_id>/<filename>           final output (retention per plan)
  reports/<job_id>.json                 stage-by-stage timings + warnings
```

## Observability

- **Sentry** — exception capture in API, worker, web
- **OpenTelemetry** — tracing spans across HTTP → Celery → ML → S3
- **Prometheus** — `/metrics` from API + worker (job count, p99, GPU util)
- **Grafana Cloud** — dashboards + alerts (latency, error rate, queue depth)
- **Audit log** — every job logged to Postgres `audit_log` with user, IP,
  duration, GDPR-relevant flags

## Deployment

- **Local dev:** `docker compose -f deploy/docker-compose.dev.yml up`
  (api + worker + redis + postgres + minio + web)
- **Staging/prod:** Kubernetes (deploy/k8s/) with HPA on CPU + GPU pools
- **GPU pool:** RunPod serverless template, scales 0→N on queue depth
- **Storage:** Cloudflare R2 (S3-compatible, no egress fees)

## Future-proofing

- Stage graph is data, not code. Adding a new stage = drop a Python file that
  implements the protocol. Order is configured per-job, not hardcoded.
- ML models are wrapped behind a uniform `Predictor` protocol. Swapping
  Qwen-Image-Lightning → Flux Kontext or SD3.5 → SD4 is one file.
- API versioned (`/v1`, `/v2`); no breaking changes within a major version.
