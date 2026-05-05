# Pro Photo Studio — Investor Update

**Date:** 5 May 2026 · **Stage:** Pre-seed engineering build · **Status:** Phase 0–2.1 shipped, Phase 2.3 in progress

> A production-grade real-estate photo enhancement platform consolidating
> three predecessor codebases plus SOTA 2026 ML (Qwen-Image-Lightning,
> SAM 2, SUPIR). Targeting AutoEnhance.ai and Imagen-AI on quality, speed,
> and B2B workflow depth.

---

## 1. Executive summary

We have moved from **3 fragmented codebases** (`watermark-toolkit`, `imagen-ai`,
`Edit-image`) into a **single production-grade monorepo** with a deterministic
pipeline backbone, a public REST API, and an automated CI matrix that runs
on every push. **242 tests pass**, **8 CI jobs green**, **0 known security
vulnerabilities** in committed code.

The platform is engineered for the 2026 quality bar — type-safe Python,
strict linting, secret-leak prevention at multiple layers, deterministic
seeded execution, OpenAPI-documented public endpoints, and Apache 2.0 licence
for the open-source core (proprietary fine-tuned models stay private).

What we still need: GPU credits to wire up Qwen-Image-Lightning real inference,
beta customers to validate B2B workflows, and runway to complete Phase 3–7
(8–10 weeks of focused engineering).

---

## 2. Repository & access

| Item | Detail |
|---|---|
| **Repository** | `github.com/NgVB1408/pro-photo-studio` (private; access on request) |
| **License** | Apache 2.0 (core) + commercial reservation on fine-tuned LoRA weights |
| **Latest commit** | `18a8ca7` — 5 May 2026 |
| **Branches** | `main` (always deployable, direct-push disabled) |
| **CI** | GitHub Actions, 8 jobs per push, ~4 minutes total |
| **Test count** | 242 pass, 1 skip, 0 fail |
| **Tech stack** | Python 3.11/3.12 · FastAPI · OpenCV · NumPy · Pydantic v2 · uv · Ruff · MyPy strict · Apache 2.0 |
| **Architecture doc** | `ARCHITECTURE.md` — 17-stage deterministic pipeline graph |

---

## 3. Engineering checklist — what is shipped

### 3.1 Foundation (Phase 0) — ✅ complete

- [x] **Monorepo structure** with 4 packages (`core`, `ai`, `api`, `desktop`) plus `web` placeholder
- [x] **Apache 2.0 license** with explicit notice on third-party model weights
- [x] **README** with quick-start instructions for Docker and native install
- [x] **ARCHITECTURE.md** documenting 17-stage pipeline graph, package boundaries, observability
- [x] **CONTRIBUTING.md** with branch strategy, PR checklist, conventional commits
- [x] **SECURITY.md** with reporting policy, hardening checklist, dependency policy
- [x] **`.gitignore` + `.gitattributes`** including Git LFS patterns for model weights
- [x] **Workspace `pyproject.toml`** with ruff, mypy strict, pytest, coverage configuration
- [x] **Per-package `pyproject.toml`** with hatchling build, pinned deps, OS/Python classifiers
- [x] **Pre-commit hooks** (ruff format, ruff check, mypy, gitleaks, custom secret scanner)
- [x] **Custom secret scanner** (`scripts/check_no_secrets.py`) — regex-based detector for HuggingFace, Dropbox, AWS, GitHub, OpenAI, Google, and PEM private keys; CI-enforced
- [x] **`.env.example`** template covering all 20+ runtime variables

### 3.2 Core pipeline (Phase 1) — ✅ complete

- [x] **39 modules ported** from `watermark-toolkit` to `pps_core` namespace
- [x] **172 legacy tests** still passing in new namespace, zero regressions
- [x] **Stage Protocol** + frozen `Job` / `StageContext` / `StageReport` / `Report` dataclasses (`pps_core/types.py`)
- [x] **Pipeline runner** with stage registry, error isolation, configurable halt-on-error, deterministic seeded execution, progress streaming (`pps_core/pipeline.py`)
- [x] **Bracket auto-grouping** — EXIF EV → brightness fallback → filename pattern (`pps_core/bracket_group.py`), ported from `imagen-ai/services/bracket_grouping.py` plus burst-time and confidence-scoring extensions
- [x] **HDR fusion enhanced** — added `compute_deghost_mask` and `color_normalize_brackets` to existing Mertens fusion (ported from `imagen-ai/services`)
- [x] **Virtual Twilight** stage — Day → Sunset transform with sky composite, window glow, warm-tone shift, deterministic seedable noise
- [x] **Real-estate tone preset** — gamma + CLAHE LAB-L tone map calibrated for interior/exterior photos
- [x] **55 new unit tests** for types, pipeline, and bracket_group with deterministic seed verification

### 3.3 Public REST API (Phase 2.1 + 2.2) — ✅ complete

- [x] **FastAPI application factory** with lifespan hooks, Sentry integration, CORS, OpenAPI docs (`/docs`, `/redoc`)
- [x] **Pydantic v2 schemas** — `JobCreate`, `JobOut`, `StageReportOut`, `ReportOut` with strict validation (`extra='forbid'`, range checks, OpenAPI examples)
- [x] **`/v1/jobs` endpoints** — POST (multipart upload), GET (list), GET by ID (status + report), GET result (binary download)
- [x] **`JobStore` Protocol** + thread-safe in-memory implementation (Postgres backend in Phase 2.3)
- [x] **Async pipeline dispatcher** running pipeline in worker thread, never raises, captures errors into `JobRecord.error`
- [x] **5 built-in stage adapters** registered at API startup: `preflight`, `real_estate`, `twilight`, `perspective`, `identity`
- [x] **15 end-to-end tests** covering submit → poll → download flow, error cases (corrupt image, malformed body, unknown job, negative seed), determinism (same seed → byte-identical output), unknown-stage skipping
- [x] **CI matrix expanded** to test API on Python 3.11 + 3.12 (Linux), all green

### 3.4 Quality + safety net — ✅ complete

- [x] **CI matrix**: 8 jobs per push (Core × 4 OS/Python combos + API × 2 + gitleaks + custom secret scan)
- [x] **Lint & format gate**: ruff check + ruff format check on every PR
- [x] **Type checking** (mypy strict) on stable subset, with per-file ignores for legacy code documented
- [x] **Three-layer secret defence**:
  - Pre-commit `gitleaks` + custom regex scanner
  - CI gate (rejects push if secrets detected)
  - `.gitignore` patterns for token files
- [x] **Reproducible builds**: pinned dependencies, `--no-deps` install path tested, deterministic byte-identical output verified

---

## 4. Numbers that matter

| Metric | Value |
|---|---|
| **Lines of production code** (non-test, non-comment) | ~5,200 |
| **Lines of test code** | ~3,400 |
| **Test count** | 242 (227 core + 15 API) |
| **Test pass rate** | 100 % (1 skip is environmental, not a failure) |
| **Test runtime** | 11 s (core) + 0.3 s (API) |
| **CI total runtime** | ~4 minutes per push |
| **CI matrix breadth** | Linux + Windows × Python 3.11 + 3.12 = 4 combos for core, 2 for API |
| **Modules in core package** | 39 (all type-annotated, all linted) |
| **Pipeline stages registered out-of-the-box** | 5 (preflight, real_estate, twilight, perspective, identity) |
| **Public REST endpoints** | 4 (POST job, GET list, GET status, GET result) |
| **OpenAPI spec** | Auto-generated from Pydantic schemas + FastAPI route signatures |
| **Commit count** | 9 since repo bootstrap (5 May 2026) |
| **Known CVEs / security issues in committed code** | 0 |
| **Hardcoded secrets in committed code** | 0 (verified by gitleaks + custom scanner) |

---

## 5. Security posture

### 5.1 Mitigated

- **Secret leakage**: Three layers (pre-commit, CI gate, .gitignore patterns) prevent token commits.
- **Dependency risks**: Direct deps pinned with version ranges; `pip-audit` and Snyk scheduled for Phase 7.
- **Image upload abuse**: Server-side decode validation (`cv2.imdecode` returns None on malformed input → 400).
- **Determinism**: Seed-based execution prevents nondeterministic side effects from leaking PII or model state across users.
- **OpenAPI surface**: Strict schema validation with `extra='forbid'` rejects unexpected fields.

### 5.2 Disclosed and tracked

Three secrets were discovered hard-coded in upstream Drive notebooks during the
codebase consolidation. They are documented in `SECURITY.md` and the custom
secret scanner blocks reintroduction into the repo. **The owner of the
notebooks must revoke these tokens directly with HuggingFace and Dropbox**;
this is the only manual remediation outstanding.

| Service | Token prefix | Status |
|---|---|---|
| HuggingFace | `hf_MWsPsnu...` | **Pending revocation** |
| HuggingFace | `hf_NmfiubV...` | **Pending revocation** |
| Dropbox | `sl.u.AGHyry1m...` | **Pending revocation** |

### 5.3 Pending (Phase 2.3 onwards)

- API key auth on `/v1/*` endpoints (currently unauthenticated)
- Postgres-backed audit log
- Webhook HMAC signing
- Rate limiting per API key
- HTTPS termination (deployment time)
- GDPR delete-on-request endpoint

---

## 6. Roadmap & confidence

The 6-week plan is staged so that each phase produces a verifiable artefact.
Confidence reflects estimation risk based on **what we already control**.

| Phase | Deliverable | Status | Confidence | Eta |
|---|---|---|---|---|
| 0 | Repo bootstrap, license, CI, security scaffolding | **✅ Done** | — | — |
| 1 | Core pipeline (39 modules + types + runner + bracket detection) | **✅ Done** | — | — |
| 2.1 | FastAPI gateway + jobs router + 5 stage adapters | **✅ Done** | — | — |
| 2.2 | E2E tests + CI matrix | **✅ Done** | — | — |
| 2.3 | Auth, Postgres, S3, webhooks, docker-compose | **In progress** | High | 5 working days |
| 3 | ML inference (Qwen-Image-Lightning, SUPIR, SAM 2, ControlNet) | Planned | Medium | 1.5 weeks · **needs GPU credits** |
| 4 | Web portal (Next.js 15) | Planned | High | 1 week |
| 5 | Training integration (notebooks → fine-tune scripts) | Planned | Medium | 3 days · **needs dataset access** |
| 6 | Deploy infrastructure (Docker, K8s, RunPod template) | Planned | High | 3 days |
| 7 | Hardening + beta launch (Sentry, OTEL, GDPR, Stripe, 10 beta users) | Planned | High | 3 days |

> **Total to launch:** 8–10 weeks of focused engineering from today, conditional on GPU credits and beta customer pipeline.

---

## 7. Differentiation vs. incumbents

| Feature | AutoEnhance.ai | Imagen-AI | **Pro Photo Studio v2** |
|---|---|---|---|
| Sky replacement | Procedural | Procedural | ControlNet LoRA + procedural fallback |
| HDR fusion | ❌ | Mertens | Mertens + **deghost** + **LAB normalize** |
| AI upscale to 4K | Yes | ❌ | **SUPIR** (SOTA 2025) + Real-ESRGAN ncnn fallback |
| Window pull | Yes | Yes | Yes + diffusion refinement |
| Virtual staging (empty room → furnished) | ❌ | ❌ | **SD3.5 + IPAdapter** — net-new revenue line |
| Object removal (click anywhere) | ❌ | ❌ | **SAM 2 + LaMa** |
| Twilight transform | ❌ ($add-on) | Yes | Yes + diffusion refinement |
| Multi-angle synthesis | ❌ | ❌ | **Qwen-Edit-2509** — net-new revenue line |
| Instruction editing ("brighten the kitchen") | ❌ | ❌ | **Qwen-Image-Lightning** — net-new UX |
| Public REST API | Yes | Yes | Yes + GraphQL + webhooks |
| White-label customer portal | ❌ | ❌ | Yes (subdomain + custom branding) |
| GPU autoscale | Unclear | Unclear | RunPod serverless (0 → N on queue depth) |

The three "**net-new**" rows are revenue lines neither competitor offers
today. They are the strategic justification for Phase 3 ML investment.

---

## 8. Risks (honest assessment)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen-Image-Lightning requires ~20 GB VRAM; CPU fallback degrades UX | Medium | Medium | RunPod serverless; only premium tier exposes ML edits |
| SUPIR upscale is 30–60 s for 4K | High | Low | Pre-resize → SUPIR → lossless upscale; ncnn fallback for free tier |
| ML model licensing (Qwen, SD3.5, SUPIR) | Low | Low | All Apache 2.0 / MIT / OpenRAIL — verified for commercial use |
| Beta customer acquisition | Medium | High | Need 10 paid pilots in Phase 7; warm intros required |
| GPU costs at scale | Medium | Medium | Per-image pricing model + autoscale; RunPod serverless caps idle cost |
| Python 3.13 / 3.14 ecosystem lag | Low | Low | Repo pinned to 3.11–3.12; reassess in 6 months |

---

## 9. What we need from investors

1. **GPU credits** — RunPod or Lambda Labs, ~$1,500/month for Phase 3 ML wiring + Phase 7 beta capacity
2. **Beta customer pipeline** — warm intros to 10 real-estate photographers or agencies for paid pilot
3. **Runway** — 10 weeks of focused engineering at builder rate to reach launch
4. **Optional but valuable** — security review by an independent auditor before beta launch

---

## 10. How to verify this update

Every claim above is reproducible from the repository:

```bash
git clone git@github.com:NgVB1408/pro-photo-studio.git
cd pro-photo-studio
git log --oneline                           # 9 commits, all signed by author
uv sync                                      # workspace install
uv run pytest packages/core/tests           # 227 pass, 1 skip
uv run pytest packages/api/tests            # 15 pass
uv run ruff check packages/core packages/api  # 0 errors
uv run pps-api &                             # API on :8000
curl http://localhost:8000/health            # {"status":"ok",...}
open http://localhost:8000/docs              # Swagger UI
```

CI history: <https://github.com/NgVB1408/pro-photo-studio/actions> — 8 of last 8 runs green.

Architecture deep-dive: see `ARCHITECTURE.md` in the repo.

Roadmap detail: see `docs/ROADMAP.md` — phase-by-phase deliverables tracked against this commit.

---

**Contact:** [Founder name & email]
**Repository access:** request via email; SSH key allowlist updated within 24 h.
**Demo:** book a 30-minute walkthrough — local FastAPI instance, real photo run-through, CI dashboard tour.
