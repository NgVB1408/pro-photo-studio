"""Main FastAPI app + entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .__version__ import __version__
from .config import settings
from .routers import hdr, health, jobs, masks, recovery, regions, window_ceiling

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="pps-wincei-api",
        description=(
            "REST API service cho Pro Photo Studio Wincei Stack.\n\n"
            "**Endpoints:**\n"
            "- `POST /api/v1/window-ceiling` — fix cửa sổ blown + trần ám màu\n"
            "- `POST /api/v1/hdr-fuse` — Mertens HDR bracket fusion\n"
            "- `POST /api/v1/segment-masks` — phân vùng AI + phào chỉ + QC verdict\n"
            "- `GET /api/v1/jobs/{id}` — job status\n"
            "- `GET /api/v1/jobs/{id}/download` — download zip output\n\n"
            "Mọi endpoint POST có flag `mock=true` để test integration nhanh."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(window_ceiling.router)
    app.include_router(hdr.router)
    app.include_router(masks.router)
    app.include_router(regions.router)
    app.include_router(recovery.router)
    app.include_router(jobs.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def landing() -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>pps-wincei-api v{__version__}</title>
<style>
body {{ font-family: sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #222 }}
h1 {{ border-bottom: 2px solid #4a90e2; padding-bottom: 8px }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px }}
a {{ color: #4a90e2 }}
.box {{ background: #f9f9f9; border-left: 4px solid #4a90e2; padding: 12px 16px; margin: 16px 0 }}
</style></head>
<body>
<h1>🪟🏠 pps-wincei-api <small>v{__version__}</small></h1>
<p>REST service cho 3 module: <b>wincei</b> (window+ceiling fix), <b>wincei-hdr</b> (bracket fusion),
<b>wincei-masks</b> (smart segmentation + AI QC).</p>

<div class="box">
<b>📖 Documentation:</b> <a href="/docs">Swagger UI</a> · <a href="/redoc">ReDoc</a>
</div>

<h2>Quick test (mock mode)</h2>
<pre><code>curl -X POST http://localhost:{settings.port}/api/v1/segment-masks \\
  -F "files=@foto.jpg" \\
  -F "mock=true"</code></pre>

<h2>Endpoints</h2>
<ul>
<li><code>POST /api/v1/window-ceiling</code> — fix cửa sổ + trần</li>
<li><code>POST /api/v1/hdr-fuse</code> — HDR bracket fusion</li>
<li><code>POST /api/v1/segment-masks</code> — phân vùng + phào chỉ</li>
<li><code>GET  /api/v1/jobs/{{id}}</code> — job status</li>
<li><code>GET  /api/v1/jobs/{{id}}/download</code> — download zip</li>
<li><code>GET  /api/v1/health</code> — health check</li>
</ul>
</body></html>
"""

    return app


app = create_app()


def run() -> None:
    """CLI entry point: pps-wincei-api."""
    import uvicorn

    uvicorn.run(
        "pps_wincei_api.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run()
