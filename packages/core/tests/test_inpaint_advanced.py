import sys
import types

import pytest
from pps_core.inpaint import SUPPORTED_LAMA_MODELS, resolve_device


def test_supported_models_include_main_ones():
    assert "lama" in SUPPORTED_LAMA_MODELS
    assert "mat" in SUPPORTED_LAMA_MODELS
    assert "migan" in SUPPORTED_LAMA_MODELS


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_no_torch(monkeypatch):
    # giả lập không có torch
    monkeypatch.setitem(sys.modules, "torch", None)
    # ImportError sẽ xảy ra trong resolve_device
    # nhưng setitem với None làm import bị raise NameError -> dùng cách khác:
    real_import = __import__

    def fake_import(name, *args, **kw):
        if name == "torch":
            raise ImportError("simulated")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_auto_cuda_available(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_mps_available(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == "mps"


def test_invalid_lama_model_raises():
    from pps_core.inpaint import _load_lama

    with pytest.raises(ValueError):
        _load_lama("cpu", model_name="nonexistent")
