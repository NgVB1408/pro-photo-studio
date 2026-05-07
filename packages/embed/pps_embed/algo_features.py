"""Algorithm / parameter-set embedding.

When the orchestrator picks a winning combination of pipeline parameters
(e.g. ``MicroContrastAgent`` profile that scored well on a villa photo), we
want to index those parameters in Qdrant so a future similar photo can
retrieve them. Two needs:

1. **Stability** — the same JSON params must always map to the same vector.
2. **Sensible distance** — small param tweaks should give close vectors.

We canonicalise params (sort keys, deterministic floats), hash to seed a
``GaussianRandomProjection`` from sklearn, then project a key-frequency
histogram into the chosen dimensionality (default 256). Same params → same
seed → same vector.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

ALGO_DIM = 256
_TOPHASH_BUCKETS = 1024


def canonicalise_params(params: dict[str, Any]) -> str:
    """Deterministic JSON: sorted keys, fixed-precision floats, no whitespace."""
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [round(float(x), 6) for x in o.flatten()]
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"unserialisable: {type(o)!r}")


def algo_embedding(params: dict[str, Any], *, dim: int = ALGO_DIM) -> np.ndarray:
    """Project a parameter dictionary into a fixed-dim deterministic vector.

    Process:
    1. Canonicalise → SHA-256 → 32 bytes seed.
    2. Build a 1024-bucket key-presence-with-value histogram (each
       canonicalised ``key=value`` token hashes into a bucket whose count is
       the float value or 1.0 for non-numeric).
    3. Project that 1024-d sparse vector down to ``dim`` via Gaussian RP
       seeded by the SHA-256, so identical params always yield the same
       projection matrix.
    """
    canon = canonicalise_params(params)
    digest = hashlib.sha256(canon.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")

    sparse = np.zeros(_TOPHASH_BUCKETS, dtype=np.float32)
    for token, value in _tokens(params):
        h = int(hashlib.sha1(token.encode()).hexdigest()[:8], 16)
        idx = h % _TOPHASH_BUCKETS
        sparse[idx] += float(value)
    if (n := float(np.linalg.norm(sparse))) > 0:
        sparse /= n

    rp = GaussianRandomProjection(n_components=dim, random_state=seed)
    rp.fit(np.zeros((1, _TOPHASH_BUCKETS)))  # required to set up matrix
    out = rp.transform(sparse.reshape(1, -1)).flatten().astype(np.float32)
    if (n := float(np.linalg.norm(out))) > 0:
        out /= n
    return out


def _tokens(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Flatten a nested dict into ``(path, value)`` pairs."""
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_tokens(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_tokens(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float, np.floating, np.integer)):
        out.append((prefix, float(obj)))
    elif isinstance(obj, bool):
        out.append((prefix, 1.0 if obj else 0.0))
    elif obj is None:
        out.append((prefix, 0.0))
    else:
        # string / arbitrary — bucket by string identity
        out.append((f"{prefix}={obj!s}", 1.0))
    return out


def stable_algo_id(params: dict[str, Any], *, namespace: str = "pps_algo") -> str:
    canon = canonicalise_params(params)
    return hashlib.sha1((namespace + "|" + canon).encode()).hexdigest()
