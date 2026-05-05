# Pro Photo Studio

[![CI](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/NgVB1408/pro-photo-studio/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> Production-grade real-estate photo enhancement, 2026 edition.
> Beats AutoEnhance.ai + Imagen-AI on quality, speed, and cost.

## What it does

Drop in real-estate photos (single shot, bracketed HDR sets, or DNG/RAW), get back
publication-ready output:

- **Sky replacement** — ControlNet LoRA + procedural fallback for offline mode
- **HDR fusion** — Mertens with deghost + LAB color normalization across exposures
- **Window pull** — recover blown highlights via diffusion or pseudo-HDR
- **Virtual staging** — empty room → furnished, via SD3.5 + IPAdapter
- **Twilight transform** — daytime exterior → sunset/golden hour
- **AI upscale** — SUPIR (SOTA 2025) with Real-ESRGAN ncnn fallback
- **Object removal** — click-anywhere via SAM 2 + LaMa
- **Instruction editing** — natural-language edits via Qwen-Image-Lightning
  ("brighten the kitchen", "remove the photographer's reflection")
- **Multi-angle synthesis** — generate alternative views via Qwen-Edit-2509
- **Auto privacy** — face/plate blur with GDPR audit log
- **Batch tone coherency** — LAB anchor lock for "one shoot, one mood"
- **Perspective correct** — Hough-vanishing-point upright
- **Lens correct** — Brown-Conrady model
- **Photographer removal** — heuristic + AI inpaint for mirror reflections

## Quick start

### Local dev (Docker)

```bash
git clone https://github.com/NgVB1408/pro-photo-studio
cd pro-photo-studio
docker compose -f deploy/docker-compose.dev.yml up
# Web UI:    http://localhost:3000
# API docs:  http://localhost:8000/docs
# MinIO:     http://localhost:9001 (admin / admin)
```

### Native (no Docker)

```bash
# Python 3.11
uv sync                          # install all package deps
pnpm install                     # web + tooling
pnpm dev                         # runs api + worker + web in parallel
```

### Desktop client

```bash
uv run --package=pps-desktop python -m pps_desktop.main
```

### CLI

```bash
uv run pps enhance ./photos/ --out ./enhanced/ --preset real_estate
uv run pps job submit --instruction "brighten the bathroom"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). High-level:

```
Web (Next.js) ─┐
Desktop (Qt) ──┼─► API (FastAPI) ─► Celery ─► Pipeline (CPU OpenCV + GPU ML) ─► S3
CLI (Click)  ──┘                              ↑
                                              ML pack: Qwen-Image, SUPIR, SAM2, ControlNet
```

## Repository layout

```
packages/
  core/      # OpenCV pipeline (ported from watermark-toolkit)
  ai/        # ML inference (Qwen, SUPIR, SAM2, ControlNet)
  api/       # FastAPI + Celery + Postgres + S3
  desktop/   # PySide6 client
  web/       # Next.js B2B portal
training/    # Notebooks + LoRA fine-tune scripts
deploy/      # Docker + K8s + Terraform
docs/
tests/
```

## Status

Active development — Phase 0 bootstrap complete, Phase 1 (core port) in progress.
See [docs/ROADMAP.md](docs/ROADMAP.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

If you discover a security issue, please follow [SECURITY.md](SECURITY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE). Some optional ML backends pull weights under
their own licenses (Qwen-Image: Apache 2.0; SD3.5: Stability AI Community License;
SAM 2: Apache 2.0; SUPIR: Apache 2.0).
