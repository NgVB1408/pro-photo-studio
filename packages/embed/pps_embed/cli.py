"""CLI: pps-embed <command>."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import cv2
import typer

from . import messages_vi as m
from .store import EmbedStore

app = typer.Typer(help=m.CLI_HELP_APP, no_args_is_help=True, add_completion=False)


def _store(qdrant_url: str | None, qdrant_key: str | None) -> EmbedStore:
    url = qdrant_url or os.environ.get("QDRANT_URL")
    if not url:
        typer.echo(m.ERR_QDRANT_NO_URL, err=True)
        raise typer.Exit(2)
    key = qdrant_key or os.environ.get("QDRANT_API_KEY")
    return EmbedStore(url=url, api_key=key)


@app.command(help=m.CLI_HELP_INDEX)
def index_photo(
    image: Path = typer.Argument(..., help=m.ARG_IMAGE),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help=m.ARG_QDRANT_URL),
    qdrant_key: str | None = typer.Option(None, "--qdrant-key", help=m.ARG_QDRANT_KEY),
) -> None:
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        typer.echo(m.ERR_NOT_AN_IMAGE.format(path=image), err=True)
        raise typer.Exit(1)
    store = _store(qdrant_url, qdrant_key)

    async def go() -> str:
        await store.ensure_collections()
        pid = await store.upsert_photo(img, payload={"source": str(image)})
        await store.close()
        return pid

    pid = asyncio.run(go())
    typer.echo(m.INFO_INDEXED.format(pid=pid))


@app.command(help=m.CLI_HELP_QUERY)
def query(
    image: Path = typer.Argument(..., help=m.ARG_IMAGE),
    k: int = typer.Option(5, "-k", help=m.ARG_K),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help=m.ARG_QDRANT_URL),
    qdrant_key: str | None = typer.Option(None, "--qdrant-key", help=m.ARG_QDRANT_KEY),
) -> None:
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        typer.echo(m.ERR_NOT_AN_IMAGE.format(path=image), err=True)
        raise typer.Exit(1)
    store = _store(qdrant_url, qdrant_key)

    async def go() -> list:
        await store.ensure_collections()
        hits = await store.query_similar_photos(img, k=k)
        await store.close()
        return hits

    hits = asyncio.run(go())
    typer.echo(m.INFO_QUERY_HEADER.format(k=k))
    for h in hits:
        typer.echo(f"  score={h.score:.4f}  id={h.point_id}  payload={h.payload}")


@app.command("index-algo", help=m.CLI_HELP_INDEX_ALGO)
def index_algo(
    params_path: Path = typer.Argument(..., help=m.ARG_PARAMS_JSON),
    name: str = typer.Option("unnamed", help=m.ARG_NAME),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url"),
    qdrant_key: str | None = typer.Option(None, "--qdrant-key"),
) -> None:
    params = json.loads(params_path.read_text(encoding="utf-8"))
    store = _store(qdrant_url, qdrant_key)

    async def go() -> str:
        await store.ensure_collections()
        aid = await store.upsert_algorithm(params, name=name)
        await store.close()
        return aid

    aid = asyncio.run(go())
    typer.echo(m.INFO_INDEXED_ALGO.format(aid=aid))


@app.command("migrate", help=m.CLI_HELP_MIGRATE)
def migrate(
    check: bool = typer.Option(
        False,
        "--check",
        help=m.ARG_MIGRATE_CHECK,
    ),
    db_url_opt: str | None = typer.Option(
        None,
        "--db-url",
        help=m.ARG_DB_URL,
    ),
) -> None:
    """Apply Alembic migrations to the metadata DB.

    With ``--check`` the command runs entirely offline: it loads the Alembic
    config + script directory and verifies all revisions parse, without ever
    opening a connection. Useful for CI gating before infrastructure is up.
    """
    try:
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory
    except ImportError as e:  # pragma: no cover
        typer.echo("alembic missing: pip install alembic", err=True)
        raise typer.Exit(2) from e

    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))

    if check:
        # Offline parse: enumerate revisions, ensure heads/bases resolve.
        try:
            script = ScriptDirectory.from_config(cfg)
            heads = script.get_heads()
            revs = list(script.walk_revisions())
        except Exception as exc:  # noqa: BLE001
            typer.echo(m.ERR_MIGRATE_CHECK_FAILED.format(err=exc), err=True)
            raise typer.Exit(1) from exc
        typer.echo(m.INFO_MIGRATE_CHECK.format(n=len(revs), heads=", ".join(heads)))
        return

    db_url = db_url_opt or os.environ.get("DATABASE_URL")
    if not db_url:
        typer.echo(m.ERR_NO_DB_URL, err=True)
        raise typer.Exit(2)
    cfg.set_main_option("sqlalchemy.url", db_url.replace("+asyncpg", ""))
    command.upgrade(cfg, "head")
    typer.echo(m.INFO_MIGRATE_DONE)


def main() -> None:
    app()


if __name__ == "__main__":
    app()
