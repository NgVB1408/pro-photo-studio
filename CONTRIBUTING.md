# Contributing

Thanks for your interest. This guide covers how to get a working dev environment
and the standards we hold PRs to.

## Environment

- **Python 3.11** (not 3.12, 3.13, or 3.14 — torch / basicsr / iopaint don't
  build on Windows for those yet)
- **Node 20+** for the web package
- **Docker 25+** (optional, for running the full stack locally)
- **uv** for Python package management — install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`
- **pnpm 9+** for Node — install via `npm install -g pnpm`

## Getting started

```bash
git clone https://github.com/NgVB1408/pro-photo-studio
cd pro-photo-studio
uv sync                 # all Python packages
pnpm install            # web + tooling
pre-commit install      # hooks: ruff, mypy, biome, secret scan
```

## Day-to-day

```bash
# Run tests for one package
uv run --package=pps-core pytest

# Run all tests
pnpm test

# Lint everything
pnpm lint

# Format
pnpm format

# Run dev stack (api + worker + web)
pnpm dev
```

## Branch strategy

- `main` is always deployable. Direct push is disabled.
- Feature branches: `feat/<short-name>` (e.g. `feat/sam2-click-mask`)
- Fix branches: `fix/<issue-number>` (e.g. `fix/123`)
- Open PRs early as draft; mark ready for review when CI is green.

## Commit messages

Conventional Commits. Examples:

```
feat(core): add saliency-aware sharpening
fix(api): handle empty bracket sets
chore(deps): bump diffusers to 0.32.1
docs(readme): clarify GPU requirements
```

## Pull request checklist

- [ ] Title follows Conventional Commits
- [ ] Tests added for new behavior; existing tests updated for changed behavior
- [ ] No new lint or mypy errors
- [ ] If touching pipeline output, attach before/after on at least 3 sample
      images (interior + exterior + aerial)
- [ ] If adding a model weight, update `docs/MODELS.md` with license + source
- [ ] If touching API surface, update `docs/api/openapi.json` (auto-regen)
- [ ] No secrets, tokens, or personally identifiable data committed (CI checks)

## Code style

- **Python:** ruff + mypy strict. No type ignores without a comment explaining
  why. Prefer `dataclass(frozen=True)` for value objects.
- **TypeScript:** biome (lint + format). Strict mode enabled. No `any`.
- **Tests:** pytest for Python, vitest for TS. Use real images (small fixtures)
  rather than mocking image arrays — mocked tests have caught zero real bugs.

## Project conventions

- **Stage functions** in `pps_core` follow signature `(image, ctx) -> (image, report)`
  and never raise on bad input — they emit a `report.skip(reason=...)` instead.
- **ML wrappers** in `pps_ai` always have a CPU fallback or a clean `RuntimeError`
  with install hint when the GPU path is unavailable.
- **API endpoints** version-prefixed (`/v1/...`); breaking changes require a new
  major version.
- **Migrations** via Alembic; never edit existing migrations after merge.

## Releasing

- Tags `v<major>.<minor>.<patch>` trigger Docker image build + push to GHCR
  and Helm chart publish.
- Changelog via `git-cliff`; do not edit `CHANGELOG.md` manually.
