"""Smoke for finetune_qwen_edit.py --dry-run (no real model load)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import finetune_qwen_edit as ft  # type: ignore[import-not-found]


@pytest.fixture
def patched_stream(monkeypatch):
    """Replace pps_data.stream_fivek with an in-memory generator."""

    def fake(*args, **kwargs):
        return iter([{"input_image": b"x", "expert_c": b"y"}])

    monkeypatch.setattr("pps_data.stream_fivek", fake)
    return fake


def test_dry_run_succeeds(tmp_path: Path, patched_stream, capsys) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset: { name: fivek, expert: c, split: train }\n"
        "model: { base: lightx2v/Qwen-Image-Lightning, task: image-to-image }\n"
        "lora: { rank: 8 }\n"
        "train: { learning_rate: 1.0e-4, max_steps: 10 }\n"
        "output: { hf_private_repo: null }\n",
        encoding="utf-8",
    )
    rc = ft.main(["--config", str(cfg), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry_run_ok" in out
    assert "lightx2v/Qwen-Image-Lightning" in out


def test_real_run_requires_output_repo(tmp_path: Path, patched_stream) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dataset: {}\nmodel: {}\nlora: {}\ntrain: {}\noutput: {}\n", encoding="utf-8"
    )
    rc = ft.main(["--config", str(cfg)])
    assert rc == 2
