"""Streaming loader smoke tests — mocked HF, no network."""

from __future__ import annotations

import pytest

from pps_data import FIVEK_EXPERTS, stream_fivek, stream_lsd, stream_sun


def test_stream_fivek_default_expert(patch_load_streaming):
    captured = patch_load_streaming()
    ds = stream_fivek()
    rows = list(ds)
    assert len(rows) == 8
    assert "expert_c" in rows[0]
    # Verify our wrapper passed streaming kwargs.
    assert captured["calls"], "load_streaming was not called"


def test_stream_fivek_expert_validation():
    with pytest.raises(ValueError):
        stream_fivek(expert="z")  # type: ignore[arg-type]


@pytest.mark.parametrize("expert", FIVEK_EXPERTS)
def test_stream_fivek_all_experts(patch_load_streaming, expert):
    patch_load_streaming()
    ds = stream_fivek(expert=expert)
    assert len(list(ds)) == 8


def test_stream_lsd(patch_load_streaming):
    patch_load_streaming(rows=[{"image": b"\x00", "clean": b"\x00"}])
    ds = stream_lsd()
    assert next(iter(ds)) == {"image": b"\x00", "clean": b"\x00"}


def test_stream_sun(patch_load_streaming):
    patch_load_streaming(rows=[{"image": b"x", "label": "kitchen"}])
    ds = stream_sun()
    rows = list(ds)
    assert rows[0]["label"] == "kitchen"


def test_mirror_override_via_env(monkeypatch, patch_load_streaming):
    captured = patch_load_streaming()
    monkeypatch.setenv("PPS_FIVEK_REPO", "myorg/private-fivek")
    stream_fivek()
    repo = captured["calls"][0]["args"][0]
    assert repo == "myorg/private-fivek"


def test_token_falls_back_to_env(monkeypatch, patch_load_streaming):
    captured = patch_load_streaming()
    monkeypatch.setenv("HF_TOKEN", "hf_test_xxx")
    stream_fivek()
    kwargs = captured["calls"][0]["kwargs"]
    assert kwargs["token"] == "hf_test_xxx"


def test_explicit_token_arg_overrides_env(monkeypatch, patch_load_streaming):
    captured = patch_load_streaming()
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    stream_fivek(token="hf_explicit")
    kwargs = captured["calls"][0]["kwargs"]
    assert kwargs["token"] == "hf_explicit"
