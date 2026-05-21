# Pro Photo Studio

[![CI](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml)
[![Release](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/release.yml/badge.svg)](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/release.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

> **Hệ thống nâng cao ảnh bất động sản cấp production, tự động end-to-end.**
> Đẩy 1 ảnh vào, nhận lại render listing-ready cùng scorecard 0–10 đi qua đội
> 5 chuyên gia (Geometry · LightBlend · MicroContrast · Cleanup · Output) +
> Director QC trước khi chạm tay người dùng cuối.

> Phiên bản tiếng Anh đầy đủ: [`README.en.md`](README.en.md).

---

## 📸 Showcase — Ảnh BĐS thật (test case DSC01527)

> Ảnh Sony A7M4 6K thực tế từ khách hàng — phòng khách modern, ngược sáng,
> trần thạch cao giật cấp, cửa kính double glass door.
> Pipeline đã verified **PASS verdict 0.852** trên ảnh này.

### So sánh BEFORE / AFTER

| 🟢 ẢNH GỐC (HDR fused từ Sony AEB ±3EV) | 🔵 OUTPUT (Smart Segmentation 9 masks + AI eval) |
| :---: | :---: |
| ![BEFORE — DSC01527](docs/showcase/real_estate/DSC01527_before.jpg) | ![AFTER — overlay color-coded](docs/showcase/real_estate/DSC01527_overlay.jpg) |
| *4608×3072 Sony A7M4 · EV 0 reference · Outdoor view recovered từ -3EV bracket* | *Pink=casing, Chartreuse=opening (door+window+sky), tints=wall/floor/ceiling* |

### Phân tách từng class — RGBA masks (Photoshop-ready)

| Wall (51.1%) | Opening (11.6%) | Floor (8.5%) |
| :---: | :---: | :---: |
| ![Wall mask](docs/showcase/real_estate/DSC01527_wall_mask.png) | ![Opening mask](docs/showcase/real_estate/DSC01527_opening_mask.png) | ![Floor mask](docs/showcase/real_estate/DSC01527_floor_mask.png) |
| *Tường tách rõ tới viền cửa, sofa loại trừ chính xác* | *Glass door + outdoor view trong suốt* | *Vân sàn gỗ giữa phòng* |

### Full Recovery Ceiling — RGBA transparent output

![ceiling_full_recovery](docs/showcase/real_estate/DSC01527_ceiling_recovery.png)

> Output cuối từ `/api/v1/full-recovery-ceiling`: PNG RGBA với vùng ngoài ceiling
> trong suốt hoàn toàn. Mọi vùng khác (sofa, sàn, tường) đều `alpha=0`.

### Scorecard — AI Supervisor verdict

7-metric quality gate (`pps_wincei_masks.evaluator`):

| Class | Coverage | Score | Verdict |
| --- | ---: | ---: | :---: |
| `wall` | 51.07% | 0.85 | ✅ pass |
| `floor` | 8.54% | 0.93 | ✅ pass |
| `ceiling` | 1.87% | 0.72 | ⚠️ review |
| `door` | 11.59% | 0.89 | ✅ pass |
| `opening` | 11.59% | 0.89 | ✅ pass |
| `window` | 0.00% | 0.95 | ✅ no_target |
| `crown` (phào trần) | 0.00% | 0.00 | ⚠️ no_target |
| `baseboard` (phào chân) | 0.06% | 0.82 | ⚠️ review |
| `casing` (nẹp cửa) | 2.34% | 0.61 | ❌ fail |
| **Overall** | — | **0.852** | **✅ PASS** |

Floor edge alignment **0.99** (near-perfect). Wall biên tới viền cửa sắc nét,
sofa loại trừ chính xác. Casing fail do hole_rate trong vùng pattern sofa —
retoucher xử trong 30 giây Photoshop.

---

## Kiến trúc Multi-Agent

```
                                       ┌─────────────────────────────────┐
   POST /v1/auto                       │  Orchestrator.run(JobContext)   │
   ┌────────────────────────────┐      │                                  │
   │ image (JPG/PNG/RAW)        │      │  Phase 0: gene_provider?         │
   │ + property_type            │─────▶│    └─ EmbedStore.fetch_genes()  │
   │ + target_long_edge         │      │       (top-K ảnh đẹp tương tự)  │
   │ + seed                     │      │                                  │
   └────────────────────────────┘      │  Phase 1: ANALYZE (parallel)    │
                                       │   ├─ GeometryAgent              │
                                       │   ├─ LightBlendAgent            │
                                       │   ├─ MicroContrastAgent (gene)  │
                                       │   ├─ CleanupAgent               │
                                       │   └─ OutputAgent                │
                                       │                                  │
                                       │  Phase 2: APPLY (deterministic) │
                                       │   geometry → light → micro      │
                                       │   → cleanup → output            │
                                       │                                  │
                                       │  Phase 3: Director QC           │
                                       │   ├─ Q1 halo @ 200%             │
                                       │   ├─ Q2 ceiling neutrality      │
                                       │   ├─ Q3 move-in feel            │
                                       │   └─ 5 SOP scorers              │
                                       └────────────┬─────────────────────┘
                                                    ▼
                                            PipelineResult
                                            (image, plans, reports,
                                             director: PASS/REVIEW/FAIL)
                                                    │
                                                    ▼
                                              USER REVIEW (final gate)
```

**Vì sao parallel analyze + serial apply:** mọi `analyze()` chỉ đọc ảnh gốc,
CPU-bound (OpenCV/numpy thả GIL) → chạy đa luồng được. `apply()` thì phải tuần
tự để mỗi stage thấy pixel grid nhất quán → kết quả deterministic theo seed.

**Phase 0 (gene retrieval):** khi `EmbedStore` đã có dữ liệu, orchestrator query
top-3 ảnh tương tự → lấy params từ ảnh đẹp đã được duyệt → MicroContrastAgent
blend với baseline (weight 0.4). Đây là cơ chế "lấy gene của ảnh đẹp" để
pipeline tự cải thiện theo thời gian.

---

## Quick start — local

Repo có script bootstrap 1-lệnh: tạo `.env`, mint API key dev, build images,
khởi động full stack.

```powershell
git clone https://github.com/NgVB1408/pro-photo-studio
cd pro-photo-studio
python scripts/bootstrap_dev.py
```

Khi script trả về:

| Endpoint | URL |
| --- | --- |
| Web portal | <http://localhost:3001> |
| API + Swagger | <http://localhost:8000/docs> |
| Demo gallery | <http://localhost:3001/demo> |
| MinIO console | <http://localhost:9001> (`minioadmin` / `minioadmin`) |

Dừng stack: `docker compose -f deploy/docker-compose.dev.yml down`.

---

## Quick start — production

CI build images khi push tag (xem `.github/workflows/release.yml`), sau đó trên
host:

```bash
# /etc/pps/.env  ← copy từ .env.example, điền giá trị thật
cd /opt/pps
docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env up -d
```

Caddy tự issue Let's Encrypt cho `PPS_DOMAIN` + `API_DOMAIN`. Runbook chi tiết:
[`RUNBOOK.md`](RUNBOOK.md).

---

## Cấu trúc repo

```
pro-photo-studio/
├── packages/
│   ├── core/      pps_core    — OpenCV + numpy pipeline + autopilot + qc
│   ├── api/       pps_api     — FastAPI + SQLAlchemy 2 async + webhooks + auth
│   ├── web/       @pps/web    — Next.js 15 customer portal (TypeScript)
│   ├── ai/        pps_ai      — ML adapters (Qwen, SUPIR, SAM 2, LoRA Colab)
│   ├── desktop/   pps_desktop — PySide6 thick client (legacy port)
│   ├── agents/    pps_agents  — 5 chuyên gia + Director QC + Orchestrator
│   ├── data/      pps_data    — HF datasets streaming (FiveK / LSD / SUN)
│   └── embed/     pps_embed   — Qdrant vector store + Postgres metadata
├── training/      LoRA fine-tune scripts + configs + evaluate
├── deploy/
│   ├── docker/{Dockerfile.api, Dockerfile.web, qdrant/docker-compose.yml}
│   ├── docker-compose.{dev,prod}.yml
│   ├── caddy/Caddyfile
│   └── hf_space/  Hugging Face Spaces (Gradio CPU demo)
├── docs/          architecture, runbook, investor brief, showcase/
├── scripts/       bootstrap_dev, generate_showcase, discover_repos…
└── .github/workflows/  ci, release, weekly-discovery, hf-space-deploy
```

---

## Phase A–D — Dataset / Vector / Training / Automation

| Phase | Module | Trạng thái | Tests |
| --- | --- | --- | --- |
| **A — Data** | `packages/data/pps_data/` — HF datasets streaming + FiftyOne views + i18n | ✅ Live verified với mirror `logasja/mit-adobe-fivek` | 16/16 |
| **B — Embed** | `packages/embed/pps_embed/` — Qdrant async + Postgres + Alembic + gene fetch | ✅ `migrate --check` offline OK | 27/27 |
| **C — Training** | `training/` — LoRA Qwen-Image-Edit + evaluate (PSNR/SSIM/LPIPS) | 🚦 Code ready, GẤATE bởi `SECURITY.md` token revoke | 5/5 (dry-run) |
| **D — Automation** | `.github/workflows/weekly-discovery.yml`, HF Spaces deploy | ✅ Workflow YAML valid | 3/3 (script) |

**Test tổng** sau A1–A4 wiring: **75/75 pass** (24 agents + 27 embed + 16 data
+ 5 training + 3 scripts).

```powershell
# Chạy từng suite riêng (tránh conftest collision)
.venv-agents\Scripts\python.exe -m pytest packages\agents\tests -ra
.venv-agents\Scripts\python.exe -m pytest packages\embed\tests -ra
.venv-agents\Scripts\python.exe -m pytest packages\data\tests -ra
.venv-agents\Scripts\python.exe -m pytest training\tests -ra
.venv-agents\Scripts\python.exe -m pytest scripts\tests -ra
```

---

## CLI cheatsheet

```bash
# Render lại showcase pack (mặc định: synthetic interior)
python scripts/generate_showcase.py
python scripts/generate_showcase.py --input fixtures/villa.jpg --scene villa-real

# Stream FiveK (cần HF_TOKEN read-scope)
$env:HF_TOKEN = "hf_xxx_read_scope"
python -m pps_data sample fivek --n 5 --out fixtures/fivek/

# Index ảnh đẹp + algorithm gene vào Qdrant (cần QDRANT_URL)
python -m pps_embed index-photo fixtures/villa.jpg
python -m pps_embed index-algo configs/microcontrast_villa.json --name villa-luxury
python -m pps_embed query fixtures/new_villa.jpg -k 5

# Validate Alembic migrations offline (CI gate)
python -m pps_embed migrate --check
# Apply lên DB thật
python -m pps_embed migrate

# Discovery cron locally (dry-run, dùng fixture, không gọi GitHub)
python scripts/discover_repos.py --dry-run --out discovery_dryrun.md

# Fine-tune Qwen-Image-Edit dry-run (gated, GPU required cho real run)
python training/finetune_qwen_edit.py --config training/configs/fivek_lora.yaml --dry-run
```

---

## Bảo mật

Báo lỗ hổng qua [`SECURITY.md`](SECURITY.md). Ghi nhớ:

- **Không commit `.env`** — pre-commit hook + gitleaks CI block các pattern bí mật.
- **Token leak gate**: 3 token cũ (HF×2 + Dropbox) PHẢI revoke trước khi chạy
  Phase C training thật. `training/finetune_qwen_edit.py` không tự bỏ qua gate.
- **FiveK research-only**: trained weights từ FiveK chỉ ở HF Private repos.

---

## Tài liệu kiến trúc sâu

| Tài liệu | Nội dung |
| --- | --- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Pipeline contract, stage Protocol, deterministic seed handling |
| [`RUNBOOK.md`](RUNBOOK.md) | Production deploy, key rotation, DR, capacity planning |
| [`SECURITY.md`](SECURITY.md) | Disclosure policy + threat model + token revoke gate |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Workflow, code style, testing requirements |
| [`docs/INVESTOR_BRIEF.md`](docs/INVESTOR_BRIEF.md) | One-page progress checklist cho stakeholders |
| [`docs/FEATURE_MATRIX_FOR_INVESTORS.md`](docs/FEATURE_MATRIX_FOR_INVESTORS.md) | So sánh 22-row vs AutoEnhance + Manuka |
| [`docs/MANUAL_ML_DEMO_GUIDE.md`](docs/MANUAL_ML_DEMO_GUIDE.md) | Chạy notebook Drive cho hero photo demo |
| [`packages/data/LICENSES.md`](packages/data/LICENSES.md) | License gate cho FiveK / LSD / SUN |

---

## Roadmap kỹ thuật v2 (đã đặt comment hook trong code)

Các kỹ thuật blending đẳng cấp Adobe Research / Stability AI đã được map vào
đúng module sẽ triển khai (xem docstring đầu mỗi file):

- **Poisson Image Editing** (Pérez et al. SIGGRAPH 2003) →
  [`packages/agents/pps_agents/cleanup.py`](packages/agents/pps_agents/cleanup.py) —
  `cv2.seamlessClone` cho object removal / sky swap / fireplace overlay.
- **PatchMatch** (Barnes et al. SIGGRAPH 2009) →
  [`packages/agents/pps_agents/lightblend.py`](packages/agents/pps_agents/lightblend.py) —
  bracket alignment khi camera/subject lệch giữa các exposure.
- **Multi-Scale Laplacian Pyramid Blend** → đã chạy live trong
  [`packages/agents/pps_agents/microcontrast.py`](packages/agents/pps_agents/microcontrast.py)
  (`_multi_band_texture` với 3 band σ=1.2/4.0/10.0).
- **Cross-Attention Control** (Hertz et al. Prompt-to-Prompt) +
  **Semantic Consistency Loss** (CLIP/DINOv2) →
  [`training/finetune_qwen_edit.py`](training/finetune_qwen_edit.py) — mở khóa
  sau khi gate token revoke đóng.

---

## License

Apache 2.0 — xem [`LICENSE`](LICENSE). ML backend tùy chọn có license riêng:

| Backend | License |
| --- | --- |
| Qwen-Image | Apache 2.0 |
| SD3.5 | Stability AI Community |
| SAM 2 | Apache 2.0 |
| SUPIR | Apache 2.0 |
| MIT-Adobe FiveK dataset | Research-only (xem `packages/data/LICENSES.md`) |
