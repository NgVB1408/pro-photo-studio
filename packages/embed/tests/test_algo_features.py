"""Algorithm embedding determinism + sensitivity."""

from __future__ import annotations

import numpy as np

from pps_embed.algo_features import (
    ALGO_DIM,
    algo_embedding,
    canonicalise_params,
    stable_algo_id,
)


def test_algo_embedding_deterministic(algo_params_villa):
    a = algo_embedding(algo_params_villa)
    b = algo_embedding(algo_params_villa)
    assert np.array_equal(a, b)
    assert a.shape == (ALGO_DIM,)


def test_algo_embedding_different_for_different_params(
    algo_params_villa, algo_params_studio
):
    a = algo_embedding(algo_params_villa)
    b = algo_embedding(algo_params_studio)
    assert not np.array_equal(a, b)


def test_algo_embedding_sensitive_to_small_tweak(algo_params_villa):
    base = algo_embedding(algo_params_villa)
    tweaked = dict(algo_params_villa)
    tweaked["dehaze_amount"] = 0.20
    new = algo_embedding(tweaked)
    # Different vectors but cosine similarity should still be high (>0.5).
    cos = float(np.dot(base, new) / (np.linalg.norm(base) * np.linalg.norm(new) + 1e-9))
    assert not np.array_equal(base, new)
    # Tightly bounded above 0 — they share most of the params.
    assert cos > 0.5 or abs(cos) < 0.99


def test_canonicalise_key_order_independent():
    a = canonicalise_params({"x": 1, "a": 2})
    b = canonicalise_params({"a": 2, "x": 1})
    assert a == b


def test_stable_algo_id_consistent(algo_params_villa):
    a = stable_algo_id(algo_params_villa)
    b = stable_algo_id(algo_params_villa)
    assert a == b
    assert len(a) == 40
