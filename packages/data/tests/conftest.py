"""Mocked HF endpoint — no network, no real datasets dep needed at runtime."""

from __future__ import annotations

import io
from typing import Any, Iterator

import pytest


class FakeIterableDataset:
    """Mimics ``datasets.IterableDataset`` enough for our loaders + CLI."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


def _png_bytes(color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_fivek_rows() -> list[dict[str, Any]]:
    return [
        {
            "input_image": _png_bytes((20, 20, 20)),
            "expert_a": _png_bytes((100, 100, 100)),
            "expert_b": _png_bytes((110, 110, 110)),
            "expert_c": _png_bytes((130, 130, 130)),
            "expert_d": _png_bytes((140, 140, 140)),
            "expert_e": _png_bytes((150, 150, 150)),
        }
        for _ in range(8)
    ]


@pytest.fixture
def patch_load_streaming(monkeypatch, fake_fivek_rows):
    def _factory(rows=None):
        captured = {"calls": []}

        def fake(*args, **kwargs):
            captured["calls"].append({"args": args, "kwargs": kwargs})
            return FakeIterableDataset(rows or fake_fivek_rows)

        monkeypatch.setattr("pps_data.loaders._common.load_streaming", fake)
        # Also patch the symbols imported into each loader namespace.
        for module in ("pps_data.loaders.fivek", "pps_data.loaders.lsd",
                       "pps_data.loaders.sun"):
            monkeypatch.setattr(f"{module}.load_streaming", fake)
        return captured

    return _factory
