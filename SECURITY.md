# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities. Instead:

1. Email **security@pro-photo-studio.dev** (or DM the maintainer privately).
2. Include reproduction steps, affected version, and impact.
3. We aim to acknowledge within 48h and triage within 5 business days.

## Scope

In scope:
- Code in `packages/` and `deploy/`
- Public deployments under `*.pro-photo-studio.dev`
- Container images published to GHCR
- Public REST API at `api.pro-photo-studio.dev`

Out of scope:
- Third-party model weights downloaded at runtime (HuggingFace, etc.) —
  report to the upstream model author.
- Self-hosted deployments running custom forks.

## Secret handling — non-negotiable

- **Never** commit API keys, tokens, passwords, OAuth secrets, or private
  certificates to this repository. Pre-commit hooks and CI scan for common
  leak patterns.
- **All secrets** flow through `.env` files (gitignored) in development and
  Kubernetes Secrets / cloud secret managers in production.
- Secrets must be **referenced by name** in code, never hardcoded:

  ```python
  import os
  HF_TOKEN = os.environ["HF_TOKEN"]   # OK
  HF_TOKEN = "hf_..."                 # NEVER
  ```

## Token revocation history

Tokens that were exposed during early development of upstream notebooks and
**must not** be reintroduced anywhere in this repository:

| Service | Token prefix | Status | Action required |
|---|---|---|---|
| HuggingFace | `hf_MWsPsnu...` | Leaked in shared Drive notebook (Qwen-Image-Lightning) | **REVOKE** at https://huggingface.co/settings/tokens |
| HuggingFace | `hf_NmfiubV...` | Leaked in shared Drive notebook (Train-AI-DNG-JPG-BĐS) | **REVOKE** at https://huggingface.co/settings/tokens |
| Dropbox | `sl.u.AGHyry1m...` | Leaked in shared Drive notebook (Untitled0) | **REVOKE** at https://www.dropbox.com/account/security |

After revocation, generate fresh tokens and store them in environment variables
only — never in notebooks or source files committed to a public repository.

## Production hardening checklist

Before any production deployment, the following must be verified:

- [ ] All secrets loaded from environment / cloud secret manager
- [ ] HTTPS enforced (no plain HTTP listener in K8s ingress)
- [ ] Rate limiting on all public endpoints (per-API-key + per-IP)
- [ ] CORS allowlist matches deployed frontend domain only
- [ ] CSP, HSTS, X-Frame-Options headers present
- [ ] Input validation on all uploads (file type, size, dimension limits)
- [ ] EXIF auto-stripped from public output unless explicitly opted in
- [ ] GDPR delete-on-request endpoint working (`DELETE /v1/users/me`)
- [ ] Audit log retention 90 days minimum
- [ ] Container images run as non-root user
- [ ] No privileged Kubernetes pods
- [ ] Postgres + Redis network-isolated to API/worker pods
- [ ] Stripe webhooks signature-verified
- [ ] OAuth callback URLs locked to production domain only
- [ ] Sentry DSN restricted to project; `sentry-cli` keys rotated quarterly
- [ ] Dependabot + Snyk + `pip-audit` clean on `main`

## Cryptography

- **TLS:** Cloudflare-terminated TLS 1.3 in front of all public endpoints.
- **Password hashing:** argon2id (via `argon2-cffi`) for any local user
  records. Most production deployments delegate auth to Clerk / Better-Auth.
- **API keys:** issued as `pps_<env>_<32-char-base62>`, hashed (argon2id)
  before storage in Postgres.
- **Job artifact URLs:** S3 presigned, default TTL 1h.
- **Webhook signatures:** HMAC-SHA256 with per-endpoint shared secret.

## Dependency policy

- Pin direct dependencies with version ranges in `pyproject.toml` /
  `package.json`. Lockfiles (`uv.lock`, `pnpm-lock.yaml`) committed.
- `pip-audit`, `npm audit`, and Snyk run in CI on every PR.
- High/critical findings block merge until patched or risk-accepted in writing.
