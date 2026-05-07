"""Smoke: evaluate on a few synthetic pairs — no GPU, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# evaluate.py lives next to this file's parent, not on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # type: ignore[import-not-found]


@pytest.fixture
def synth_pairs(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for i in range(3):
        inp = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        tgt = np.clip(inp.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:03d}_input.jpg"), inp)
        cv2.imwrite(str(tmp_path / f"{i:03d}_target.jpg"), tgt)
    return tmp_path


def test_iter_pairs_from_dir(synth_pairs: Path) -> None:
    pairs = list(evaluate.iter_pairs_from_dir(synth_pairs))
    assert len(pairs) == 3


def test_evaluate_pairs_returns_metrics(synth_pairs: Path) -> None:
    pairs = (evaluate._pair_from_paths(p) for p in evaluate.iter_pairs_from_dir(synth_pairs))
    summary = evaluate.evaluate_pairs(pairs, max_pairs=3)
    assert summary.n == 3
    # Synthetic +5 noise → PSNR around 35 dB, SSIM > 0.9
    assert summary.psnr_mean > 25.0
    assert summary.ssim_mean > 0.6


def test_cli_writes_report(synth_pairs: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "report.json"
    rc = evaluate.main(
        ["--source", "dir", "--input-dir", str(synth_pairs), "--n", "3",
         "--out", str(out), "--checkpoint", "synth"]
    )
    assert rc == 0
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert data["n"] == 3
    assert data["checkpoint"] == "synth"
