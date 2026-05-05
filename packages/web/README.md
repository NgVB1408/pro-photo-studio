# @pps/web — Pro Photo Studio customer portal

Next.js 15 (App Router) + React 19 + Tailwind v4 customer-facing web app.
Talks to the FastAPI backend in `packages/api` via server-side fetch so the
API key never reaches the browser bundle.

## Pages

| Path | Purpose |
| --- | --- |
| `/` | Landing — pitch, feature matrix, CTA |
| `/upload` | Drag-and-drop a photo, pick stages, submit |
| `/jobs` | Recent jobs (most-recent first) |
| `/jobs/[id]` | Status, stage report, before/after slider, download |
| `/api/*` | Internal route handlers — proxy uploads + image bytes |

## Local dev

```bash
cp .env.example .env.local
# Edit PPS_API_URL + PPS_API_KEY (use the dev key from `pps-api keys create`).

pnpm install
pnpm dev      # → http://localhost:3000
```

The web app talks to the API via server-side fetch — start the API alongside:

```bash
cd packages/api
uvicorn pps_api.main:app --reload
```

## Production build

```bash
pnpm build
pnpm start
```

## Architecture notes

- **API key isolation.** All API calls happen in route handlers (`app/api/*`)
  and server components. The browser never sees `PPS_API_KEY`.
- **Polling, not WebSockets.** Job status uses 1.5s client-side polling.
  Sufficient for typical 5–60s pipelines and survives flaky connections.
  We can swap to SSE in Phase 5 once jobs run multi-minute.
- **No auth provider yet.** Phase 4.1 will integrate Clerk/Better-Auth.
  For now the deployment runs behind a single shared API key — fine for
  beta + investor demos.
- **Tailwind v4.** Uses the new `@import "tailwindcss"` directive and the
  `@tailwindcss/postcss` plugin. No `tailwind.config.ts` required.
