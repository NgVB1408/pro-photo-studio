# Pro Photo Studio — operational runbook

This document is for whoever is on call. It assumes the production stack is
running on a Linux host with Docker Engine + the Compose v2 plugin.

---

## TL;DR — what runs where

| Component | Role | Image | Port |
| --- | --- | --- | --- |
| `caddy`     | TLS termination + reverse proxy | `caddy:2-alpine` | 80, 443 |
| `web`       | Next.js customer portal         | `ghcr.io/<org>/pps-web:<tag>` | 3001 (internal) |
| `api`       | FastAPI gateway + worker        | `ghcr.io/<org>/pps-api:<tag>` | 8000 (internal) |
| `postgres`  | jobs / api-keys / billing       | `postgres:16-alpine`           | 5432 (internal) |
| `redis`     | job broker + cache              | `redis:7-alpine`               | 6379 (internal) |

There is no separate worker process today — the API runs CPU stages in
background tasks. GPU stages will move to a Celery worker pool in Phase 6.

---

## First-time deploy

1. Provision a host with Docker + Compose v2.
2. Point DNS A records for `portal.<your-domain>` and `api.<your-domain>` at
   the host. Caddy will provision Let's Encrypt certs on first request.
3. Drop a `.env` file at `/etc/pps/.env` populated from `.env.example`. At
   minimum:
   - `PPS_ENV=production`
   - `PPS_DOMAIN`, `API_DOMAIN`
   - `PPS_SECRET_KEY` (≥32 random chars)
   - `POSTGRES_PASSWORD`, `REDIS_PASSWORD`
   - `S3_*` credentials (Cloudflare R2, AWS S3, or Wasabi)
   - `PPS_API_KEY` — mint with `pps-keys create` after first boot
4. Pull and start:

   ```bash
   cd /opt/pps
   docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env pull
   docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env up -d
   ```

5. Verify:

   ```bash
   curl -fsS https://api.<domain>/health
   curl -fsS https://portal.<domain>/
   ```

6. Mint a real API key (replaces the bootstrap one) and update `/etc/pps/.env`.

---

## Health & smoke tests

The Caddy reverse proxy hits `/health` (api) and `/` (web) every 30 s. Manually:

```bash
docker compose -f deploy/docker-compose.prod.yml ps
docker compose -f deploy/docker-compose.prod.yml logs --tail 200 api
docker compose -f deploy/docker-compose.prod.yml logs --tail 200 web
```

End-to-end test (auto-pilot path):

```bash
curl -F "image=@sample.jpg" -H "X-API-Key: $PPS_API_KEY" \
     https://api.<domain>/v1/auto
```

You should get a 202 with `{"job_id": "...", "status": "queued"}`. Poll
`/v1/jobs/<id>` until status is `completed`, then download the result.

---

## Common operations

### Rotate the API key
```bash
docker compose exec api pps-keys revoke <old-key-id>
docker compose exec api pps-keys create --env production --name "web-portal"
# update /etc/pps/.env with the new value, then:
docker compose up -d web
```

### Rolling deploy a new version
```bash
export PPS_IMAGE_TAG=v0.5.0
docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env pull
docker compose -f deploy/docker-compose.prod.yml --env-file /etc/pps/.env up -d
```

`up -d` only recreates services whose image SHA changed. Postgres + Redis
volumes survive untouched.

### Database backup
```bash
docker compose exec postgres pg_dump -U pps pps | \
  gzip > /backup/pps-$(date +%Y%m%d-%H%M).sql.gz
```

Restore:
```bash
gunzip -c /backup/pps-YYYYMMDD-HHMM.sql.gz | \
  docker compose exec -T postgres psql -U pps pps
```

### Clear the job queue (emergency)
```bash
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" FLUSHDB
```

This drops in-flight tasks. Customers receive `failed` webhooks; their
deliveries can be retried via the dashboard.

---

## Troubleshooting

**Symptom**: API returns 502 from Caddy.
- `docker compose ps` — confirm `api` is healthy. If not, `logs api` and
  look for DB-connection errors. Most often a stale `DATABASE_URL` or a
  Postgres container that didn't finish initializing.

**Symptom**: Web portal loads but uploads fail with 401.
- `PPS_REQUIRE_API_KEY=true` in production *and* the web container's
  `PPS_API_KEY` is wrong. Check that the env values in
  `/etc/pps/.env` match the key hash in the `api_keys` Postgres table.

**Symptom**: Long jobs time out in the browser.
- Browser POST timeout is 60 s on the proxy. Anything longer flips to
  webhook delivery automatically — the front-end polls and shows progress.
  If the webhook never fires, check `docker compose logs api | grep webhook`.

**Symptom**: GPU stage errors with "Colab weights not on disk".
- That's the expected behaviour when the model volume is unmounted.
  Mount the LoRA weights at `/app/packages/ai/pps_ai/_models/colab/<name>/`
  and restart the api container.

**Symptom**: Image build fails on `next build`.
- Usually env validation. Make sure `PPS_API_URL` and `PPS_API_KEY` are set
  during build (they don't have to be the real prod values). The
  `Dockerfile.web` injects placeholders — if you've custom-built it, port
  the same trick over.

---

## Disaster recovery

1. Spin a fresh Linux host.
2. Copy `/etc/pps/.env` and the Postgres backup to it.
3. `docker compose -f deploy/docker-compose.prod.yml ... up -d postgres redis caddy`
4. Restore the DB (see above).
5. `docker compose ... up -d api web`.
6. Update DNS to point at the new host. Caddy gets new certs automatically.

Total downtime budget: < 30 minutes assuming hourly DB backups.

---

## Capacity planning

| Metric | Comfortable | Page on-call |
| --- | --- | --- |
| API queue depth (`/v1/jobs?status=queued`) | < 50 | > 500 sustained 5 min |
| Job p50 duration | < 10 s | > 30 s |
| Job p95 duration | < 30 s | > 90 s |
| 5xx rate | < 0.1 % | > 1 % |

Scale-up steps:
- Vertical: bump `API_WORKERS` in `.env`, recreate the `api` service.
- Horizontal: run multiple `api` replicas behind Caddy (Caddyfile already
  load-balances). Postgres becomes the bottleneck before the API does;
  consider read replicas or PgBouncer at ~500 RPS.
