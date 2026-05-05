# Pro Photo Studio

[![CI](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml)
[![Release](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/release.yml/badge.svg)](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/release.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

> Production-grade real-estate photo enhancement, automated end-to-end.
> Drop a photo, get back a listing-ready render with a 0–10 scorecard from a
> nine-specialist post-production studio.

## What you get

| | |
| --- | --- |
| **Auto-pilot** | One endpoint. The pipeline auto-detects scene, runs the baseline pipeline, and routes the rendered output through 9 specialists with scored checklists. No knobs to tune. |
| **Multi-agent studio** | Vertical · Exposure · White Balance · Noise · Sky · Sharpness · Halo · Colour · Composition. Every specialist has a public checklist and rolls back any change that lowers a score. |
| **Real-estate baseline** | Sky replace, window pull, lawn boost, perspective correction, scene classify (interior/exterior/aerial), tone coherency for a whole listing. |
| **Production backbone** | FastAPI + SQLAlchemy 2 (async) + S3 storage + signed webhook delivery + argon2id-hashed API keys. |
| **Customer portal** | Next.js 15 + Tailwind v4. Drag-drop upload, live polling, before/after slider, scorecard view, demo gallery. |
| **Deploy story** | One-host docker-compose + Caddy TLS, OR pull pre-built `ghcr.io` images and roll forward. |
| **Optional ML** | Drop fine-tuned LoRA weights from your Colab notebooks into `packages/ai/_models/colab/` — they slot into the same agent Protocol. |

## Quick start (local)

The repository ships a one-command bootstrap. It writes a fresh `.env`,
mints a dev API key, builds the images, and brings up the full stack.

```bash
git clone https://github.com/NgVB1408/pro-photo-studio
cd pro-photo-studio
python scripts/bootstrap_dev.py
```

When it returns:

| | |
| --- | --- |
| Web portal | <http://localhost:3001> |
| API + Swagger | <http://localhost:8000/docs> |
| Demo gallery | <http://localhost:3001/demo> |
| MinIO console | <http://localhost:9001> (`minioadmin` / `minioadmin`) |

Stop the stack with `docker compose -f deploy/docker-compose.dev.yml down`.

## Quick start (production)

Build images on tag push (CI handles it via `.github/workflows/release.yml`),
then on the host:

```bash
# /etc/pps/.env   ← copy from .env.example, fill in real values
cd /opt/pps
docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env up -d
```

Caddy auto-provisions Let's Encrypt certs for `PPS_DOMAIN` and `API_DOMAIN`.

Detailed operational runbook: [`RUNBOOK.md`](RUNBOOK.md).

## How auto-pilot works

```
                                         ┌──────────────────────────┐
            POST /v1/auto                │  AutoPilot.run()         │
   ┌─────────────────────────────────┐   │                           │
   │ image (multipart, optional      │   │  classify_scene → tag     │
   │ scene/seed query params)        │──▶│  baseline pipeline        │
   └─────────────────────────────────┘   │   • perspective           │
                                         │   • real_estate           │
                                         │   • enhance_studio        │
                                         │  StudioOrchestrator       │
                                         │   • 9 specialists         │
                                         │   • per-agent checklist   │
                                         │   • rollback on regress   │
                                         └────────────┬──────────────┘
                                                      │
                                                      ▼
                                  StudioReport (overall 0–10, grade S/A/B/C/D,
                                  per-category checklists, before-after deltas)
```

The customer-facing scorecard surfaces every specialist's checklist verbatim:

```
Exposure Specialist             9.4 / 10  ↑ +1.8
  ✓ No blown highlights (>250)   0.04% above 250
  ✓ No crushed shadows (<5)      0.27% below 5
  ✓ Tonal range full             P1=12 · P99=235 · spread 223
  ✓ Mid-tone in range            P50=132
```

## Repository layout

```
pro-photo-studio/
├── packages/
│   ├── core/   pps_core — OpenCV + numpy pipeline + agents/* + autopilot.py + qc.py
│   ├── api/    pps_api  — FastAPI gateway, SQLAlchemy stores, webhooks, auth
│   ├── web/    @pps/web — Next.js 15 customer portal
│   ├── ai/     pps_ai   — ML inference adapters (Qwen, SUPIR, SAM2, Colab models)
│   └── desktop/ pps_desktop — PySide6 thick client (legacy port)
├── deploy/
│   ├── docker/{Dockerfile.api, Dockerfile.web}
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   └── caddy/Caddyfile
├── training/   notebooks + LoRA fine-tune scripts (Colab)
├── docs/       investor brief, feature matrix, manual demo guide
├── scripts/    bootstrap_dev.py, secret scanners, ad-hoc tooling
└── tests/      tests live next to each package (`packages/<pkg>/tests/`)
```

## Testing

```bash
# Python tests
uv run pytest packages/core/tests packages/api/tests

# Web build + typecheck
pnpm --filter @pps/web typecheck
pnpm --filter @pps/web build
```

CI runs the full matrix on every PR (`Linux × Windows × Python 3.11/3.12`)
plus the web build and gitleaks secret scan.

## Architecture deep-dives

| Document | What it covers |
| --- | --- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Pipeline contract, stage Protocol, deterministic seed handling. |
| [`RUNBOOK.md`](RUNBOOK.md) | Production deploy, rotating keys, DR, capacity planning. |
| [`docs/INVESTOR_BRIEF.md`](docs/INVESTOR_BRIEF.md) | One-page progress checklist for stakeholders. |
| [`docs/FEATURE_MATRIX_FOR_INVESTORS.md`](docs/FEATURE_MATRIX_FOR_INVESTORS.md) | 22-row competitor comparison vs AutoEnhance + Manuka. |
| [`docs/MANUAL_ML_DEMO_GUIDE.md`](docs/MANUAL_ML_DEMO_GUIDE.md) | Running the Drive notebooks for hero photo demos. |
| [`SECURITY.md`](SECURITY.md) | Disclosure policy + threat model. |

## Status

| Phase | What | Status |
| --- | --- | --- |
| 0 | Repo bootstrap, CI, secret scanning | ✅ |
| 1 | Core OpenCV pipeline migration (227 tests passing) | ✅ |
| 2 | API gateway + auth + DB + storage + webhooks (92 tests passing) | ✅ |
| 3 | ML inference wiring (Qwen, SUPIR, SAM 2) | 🔄 (Colab adapter Protocol ready, weights pending GPU credits) |
| 4 | Customer web portal + auto-pilot + scorecard | ✅ |
| 5 | Training notebook integration | 🔄 |
| 6 | K8s + RunPod GPU autoscale | 📋 |
| 7 | Sentry, OTEL, Stripe, beta launch | 📋 |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All changes ship behind tests.
The agent roster is the easiest extension surface — copy any
`packages/core/pps_core/agents/<role>.py` for a working template.

## Security

Report vulnerabilities via [`SECURITY.md`](SECURITY.md). The codebase blocks
common secret patterns in CI; never commit `.env`.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Optional ML backends carry their own
licenses (Qwen-Image: Apache 2.0; SD3.5: Stability AI Community; SAM 2:
Apache 2.0; SUPIR: Apache 2.0).
