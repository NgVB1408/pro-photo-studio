"""CLI smoke — Typer test runner, no network."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pps_data.cli import app

runner = CliRunner()


def test_list_runs():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "fivek" in result.stdout
    assert "lsd" in result.stdout
    assert "sun" in result.stdout


def test_unknown_dataset_returns_error():
    result = runner.invoke(app, ["sample", "imagenet", "-n", "1"])
    assert result.exit_code == 2


def test_sample_writes_files(patch_load_streaming, tmp_path: Path):
    patch_load_streaming()
    result = runner.invoke(
        app, ["sample", "fivek", "-n", "3", "--out", str(tmp_path), "--expert", "c"]
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "fivek"
    saved = sorted(p.name for p in out.iterdir())
    # Each row has 6 image fields; we asked for 3 rows → 18 files maximum
    assert len(saved) >= 3
    assert any("input_image" in n for n in saved)
    assert any("expert_c" in n for n in saved)


def test_sample_invalid_expert(patch_load_streaming, tmp_path):
    patch_load_streaming()
    result = runner.invoke(
        app, ["sample", "fivek", "--expert", "z", "--out", str(tmp_path)]
    )
    assert result.exit_code == 2
