"""Shared fixtures: synthetic photos for embedding tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def small_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)


@pytest.fixture
def small_image_b() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)


@pytest.fixture
def algo_params_villa() -> dict:
    return {
        "agent": "microcontrast",
        "property": "villa_luxury",
        "texture": {"fine": 0.45, "mid": 0.40, "macro": 0.18},
        "dehaze_amount": 0.18,
        "wood_clarity": 0.45,
    }


@pytest.fixture
def algo_params_studio() -> dict:
    return {
        "agent": "microcontrast",
        "property": "studio_minimal",
        "texture": {"fine": 0.25, "mid": 0.20, "macro": 0.08},
        "dehaze_amount": 0.05,
        "wood_clarity": 0.30,
    }
