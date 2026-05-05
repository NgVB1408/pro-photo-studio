# Roadmap

This roadmap tracks the 6-week delivery plan for Pro Photo Studio v2 — a
flagship real-estate photo enhancement platform consolidating three
predecessor codebases (`watermark-toolkit`, `imagen-ai`, `Edit-image`) and
integrating SOTA 2026 ML models.

## Phase 0 — Repo bootstrap ✅ done

- [x] Monorepo layout (`packages/{core,ai,api,desktop,web}`)
- [x] License Apache 2.0
- [x] Top-level `README`, `ARCHITECTURE`, `CONTRIBUTING`, `SECURITY`
- [x] `.gitignore`, `.gitattributes` (Git LFS), `.env.example`
- [x] Workspace `pyproject.toml` with ruff + mypy + pytest config
- [x] Pre-commit hooks (`ruff`, `mypy`, `gitleaks`, custom secret scan)
- [x] CI pipeline (`ci.yml`) — lint, format, test, secret-scan
- [x] Custom `scripts/check_no_secrets.py` to block known token patterns

## Phase 1 — Core port (1 week)

- [x] Copy 39 modules `watermark-toolkit/src` → `packages/core/pps_core`
- [x] Rename `watermark_toolkit` → `pps_core` in 26 files (sources + tests)
- [x] All 172 tests pass in new namespace (1 skip pre-existing)
- [x] `pps-core` package metadata + extras (`raw`, `sky-ai`, `dropbox`, `ui`)
- [ ] Type hints strict pass on stable subset (twilight, tone_coherency,
      perspective, hdr) — partial; rest tracked in follow-up
- [ ] `bracket_group.py` port from imagen-ai (auto-detect bracket sets)
- [ ] `pps_core.types` dataclasses for Job/Stage/Report

## Phase 2 — API gateway (1 week)

- [x] `pps-api` package skeleton (FastAPI + Settings)
- [x] `/health` endpoint + lifespan + CORS + Sentry hook
- [ ] Auth middleware (Clerk or Better-Auth)
- [ ] Celery worker + job model
- [ ] Postgres schema + Alembic baseline migration
- [ ] S3 storage abstraction (boto3 + MinIO local)
- [ ] Webhook delivery (Slack, email, custom URL)
- [ ] Public REST: `/v1/jobs`, `/v1/upload`, `/v1/result/{id}`
- [ ] OpenAPI docs auto-gen

## Phase 3 — ML inference (1.5 weeks)

- [x] `pps-ai` package skeleton + Predictor protocol
- [x] `QwenEditor` stub (lazy load + local/remote mode dispatch)
- [ ] Wire `Qwen-Edit-2509` for image-to-image (replaces stub)
- [ ] `SUPIR` upscale wrapper x2/x4 + tile splitting
- [ ] `SAM 2` click-mask service
- [ ] `LaMa Cleaner` object removal
- [ ] `ControlNet` sky LoRA (train or pull from HF)
- [ ] CPU fallback path for every ML module

## Phase 4 — Web portal (1 week)

- [ ] Next.js 15 + React 19 + Tailwind v4 + shadcn/ui
- [ ] Pages: `/` `/upload` `/jobs` `/result/[id]` `/billing` `/api-keys`
- [ ] Clerk auth flow
- [ ] Job streaming progress (SSE)
- [ ] Before/after slider component
- [ ] Dark mode

## Phase 5 — Training integration (3 days)

- [x] 9 Drive notebooks downloaded for inspection
- [x] `Qwen-Image-Lightning` source identified → `lightx2v/Qwen-Image-Lightning`
- [x] `Qwen-Edit-2509-Multiple-angles` source identified → `dx8152/...`
- [x] **Security**: 3 leaked tokens flagged for revocation
- [ ] Move notebooks into `training/notebooks/` (only after token revocation
      and scrubbing — gitleaks gates the commit)
- [ ] `training/scripts/finetune_re.py` headless runner
- [ ] Document training data format in `training/README.md`
- [ ] Optional: GCS dataset migration to public S3 (`aifusionproject-image-enhancement`)

## Phase 6 — Deploy infra (3 days)

- [ ] `Dockerfile` per package
- [ ] `docker-compose.yml` for local dev (Postgres + Redis + MinIO + API + worker + web)
- [ ] K8s manifest (port from Edit-image, update tags)
- [ ] RunPod serverless template with GPU worker
- [ ] Terraform module (R2 storage + Cloudflare DNS)

## Phase 7 — Hardening + launch (3 days)

- [ ] Sentry SDK setup all services
- [ ] OpenTelemetry tracing FastAPI → Celery → ML
- [ ] Load test 100 concurrent jobs
- [ ] GDPR compliance (delete-on-request, audit log, EXIF strip option)
- [ ] Security audit (secret scan, OWASP top 10)
- [ ] Stripe products + pricing page
- [ ] Beta launch with 10 users

## Tracking

Phase 0 completed in **session 1**. Subsequent phases tracked via GitHub
Project board (TBD) and milestones.
