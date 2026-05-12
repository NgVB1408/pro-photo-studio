"""CLI: pps-data <command> — Vietnamese-helped Typer app."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import typer

from . import messages_vi as m
from .loaders.fivek import FIVEK_EXPERTS, stream_fivek
from .loaders.lsd import stream_lsd
from .loaders.sun import stream_sun
from .loaders._common import take

app = typer.Typer(help=m.CLI_HELP_APP, no_args_is_help=True, add_completion=False)

_DATASETS = {"fivek", "lsd", "sun"}


@app.command(help=m.CLI_HELP_SAMPLE)
def sample(
    dataset: str = typer.Argument(..., help=m.ARG_DATASET),
    n: int = typer.Option(5, "-n", "--n", help=m.ARG_N),
    out: Path = typer.Option(Path("fixtures"), "-o", "--out", help=m.ARG_OUT),
    expert: str = typer.Option("c", help=m.ARG_EXPERT),
    split: str = typer.Option("train", help=m.ARG_SPLIT),
    mirror: str | None = typer.Option(None, help=m.ARG_MIRROR),
) -> None:
    if dataset not in _DATASETS:
        typer.echo(m.ERR_DATASET_UNKNOWN.format(name=dataset, options=", ".join(_DATASETS)),
                   err=True)
        raise typer.Exit(2)

    if not os.environ.get("HF_TOKEN"):
        # Some public datasets work anonymous, but warn loudly
        typer.echo(f"warning: {m.ERR_NO_HF_TOKEN}", err=True)

    if dataset == "fivek":
        if expert not in FIVEK_EXPERTS:
            typer.echo(f"expert must be one of {FIVEK_EXPERTS}", err=True)
            raise typer.Exit(2)
        stream = stream_fivek(expert=expert, split=split, mirror=mirror)
    elif dataset == "lsd":
        stream = stream_lsd(split=split, mirror=mirror)
    else:
        stream = stream_sun(split=split, mirror=mirror)

    out = out / dataset
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, row in enumerate(take(stream, n)):
        saved += _save_row(row, out, i)
    if saved == 0:
        typer.echo(m.INFO_NO_SAMPLES, err=True)
        raise typer.Exit(1)
    typer.echo(m.INFO_SAMPLE_DONE.format(n=saved, out=out))


@app.command("list", help=m.CLI_HELP_LIST)
def list_cmd() -> None:
    from .loaders.fivek import DEFAULT_MIRROR as FIVEK
    from .loaders.lsd import DEFAULT_MIRROR as LSD
    from .loaders.sun import DEFAULT_MIRROR as SUN

    typer.echo("Datasets hỗ trợ:")
    for name, default, env in (
        ("fivek", FIVEK, "PPS_FIVEK_REPO"),
        ("lsd", LSD, "PPS_LSD_REPO"),
        ("sun", SUN, "PPS_SUN_REPO"),
    ):
        active = os.environ.get(env, default)
        marker = "*" if active != default else " "
        typer.echo(f"  {marker} {name:6s}  {active}  (default {default})")


@app.command(help=m.CLI_HELP_INSPECT)
def inspect(
    dataset: str = typer.Argument(..., help=m.ARG_DATASET),
    n: int = typer.Option(50, "-n", "--n", help=m.ARG_N),
    name: str = typer.Option("pps_inspect", help="FiftyOne dataset name"),
    expert: str = typer.Option("c", help=m.ARG_EXPERT),
) -> None:
    try:
        from .fiftyone_views import register_sampled_view
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    if dataset == "fivek":
        stream = stream_fivek(expert=expert)
    elif dataset == "lsd":
        stream = stream_lsd()
    else:
        stream = stream_sun()
    rows = list(take(stream, n))
    register_sampled_view(rows, name=name)
    typer.echo(m.INFO_INSPECT_OPEN)
    try:
        import fiftyone as fo

        session = fo.launch_app()
        session.wait()
    except Exception as e:  # pragma: no cover
        typer.echo(f"FiftyOne app failed to launch: {e}", err=True)


def _save_row(row: dict[str, Any], out: Path, idx: int) -> int:
    """Best-effort: save any image-like fields in the row to disk."""
    saved = 0
    for key, value in row.items():
        img = _to_pil(value)
        if img is None:
            continue
        ext = "jpg" if img.mode in {"RGB", "L"} else "png"
        path = out / f"{idx:05d}_{key}.{ext}"
        try:
            img.convert("RGB").save(path, quality=92)
            saved += 1
        except Exception as e:  # noqa: BLE001
            logging.warning("could not save %s: %s", path, e)
    return saved


def _to_pil(obj: Any):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    if obj is None:
        return None
    if hasattr(obj, "save") and hasattr(obj, "convert"):  # PIL Image
        return obj
    if isinstance(obj, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(obj))
        except Exception:
            return None
    if isinstance(obj, dict):
        if "bytes" in obj and obj["bytes"]:
            try:
                return Image.open(io.BytesIO(obj["bytes"]))
            except Exception:
                return None
        if "path" in obj and obj["path"]:
            try:
                return Image.open(obj["path"])
            except Exception:
                return None
    if isinstance(obj, str) and len(obj) < 4096:
        try:
            return Image.open(obj)
        except Exception:
            return None
    return None


def main() -> None:  # entry point for backwards compat
    app()


if __name__ == "__main__":
    app()
