"""Smoke for scripts/discover_repos.py — dry-run only (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discover_repos as dr  # type: ignore[import-not-found]


def test_dry_run_writes_report(tmp_path: Path):
    out = tmp_path / "report.md"
    rc = dr.main(["--dry-run", "--out", str(out)])
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert body.startswith("# Weekly OSS discovery")
    assert "github" in body.lower()
    assert "example/super-resolution" in body


def test_render_report_handles_no_hits():
    body = dr.render_report([])
    assert "no new candidates" in body


def test_hit_dataclass_round_trip():
    h = dr.Hit(source="github", name="a/b", url="https://x", stars=5, tags=["x", "y"])
    assert h.source == "github"
    assert h.tags == ["x", "y"]
