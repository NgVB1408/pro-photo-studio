"""Photo embedding — pHash ⊕ LAB histogram ⊕ saliency stats.

Default 176-d vector, fully deterministic, CPU-only:

* **pHash 64d** — perceptual hash. Two visually similar photos have small
  Hamming distance.
* **LAB histogram 96d** — 32 bins per channel (L, a, b) normalised to sum 1.
  Captures palette / colour distribution invariant to layout.
* **Saliency stats 16d** — first 4 moments + 4 quartiles of saliency-mean
  per quadrant. Captures where the visual interest lives.

Optional ``+512d`` from OpenCLIP if installed (extras=["clip"]). Disabled by
default to keep the core dependency footprint tiny.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable

import cv2
import numpy as np

log = logging.getLogger(__name__)

PHASH_DIM = 64
LAB_HIST_DIM = 32 * 3  # 96
SALIENCY_DIM = 16
PHOTO_DIM = PHASH_DIM + LAB_HIST_DIM + SALIENCY_DIM  # 176


# ----------------------------------------------------------------------
# pHash 8x8 DCT
# ----------------------------------------------------------------------


def phash_64(img: np.ndarray) -> np.ndarray:
    """Perceptual hash → 64 floats {0.0, 1.0} (use as bits)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8]
    median = float(np.median(block[1:].flatten()))  # exclude DC term
    bits = (block.flatten() > median).astype(np.float32)
    return bits  # 64-d


# ----------------------------------------------------------------------
# LAB histogram
# ----------------------------------------------------------------------


def lab_histogram_96(img: np.ndarray, bins: int = 32) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out = np.zeros(bins * 3, dtype=np.float32)
    for c in range(3):
        h, _ = np.histogram(lab[..., c], bins=bins, range=(0, 256), density=False)
        s = float(h.sum())
        if s > 0:
            out[c * bins : (c + 1) * bins] = h.astype(np.float32) / s
    return out  # 96-d, each 32-block sums to 1


# ----------------------------------------------------------------------
# Saliency stats per-quadrant
# ----------------------------------------------------------------------


def saliency_stats_16(img: np.ndarray) -> np.ndarray:
    """4 quadrants × 4 stats (mean, std, min, max) of saliency map."""
    sal = _compute_saliency(img)
    h, w = sal.shape
    quads = [
        sal[: h // 2, : w // 2],
        sal[: h // 2, w // 2 :],
        sal[h // 2 :, : w // 2],
        sal[h // 2 :, w // 2 :],
    ]
    out = []
    for q in quads:
        if q.size == 0:
            out.extend([0.0, 0.0, 0.0, 0.0])
            continue
        out.extend([float(q.mean()), float(q.std()), float(q.min()), float(q.max())])
    return np.array(out, dtype=np.float32)


def _compute_saliency(img: np.ndarray) -> np.ndarray:
    """Try pps_core's saliency; fall back to gradient-based."""
    try:
        from pps_core.saliency_sharpen import compute_saliency

        return compute_saliency(img).astype(np.float32)
    except Exception:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx**2 + gy**2)
        if g.max() > 0:
            g = g / g.max()
        return g


# ----------------------------------------------------------------------
# Public combined embedding
# ----------------------------------------------------------------------


def photo_embedding(img: np.ndarray, *, with_clip: bool = False) -> np.ndarray:
    """Return concatenated photo embedding — float32, length ``PHOTO_DIM``
    (or ``PHOTO_DIM + 512`` when ``with_clip=True``).
    """
    parts: list[np.ndarray] = [
        phash_64(img),
        lab_histogram_96(img),
        saliency_stats_16(img),
    ]
    if with_clip:
        parts.append(_clip_embedding(img))
    return np.concatenate(parts).astype(np.float32)


def _clip_embedding(img: np.ndarray) -> np.ndarray:  # pragma: no cover — optional dep
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP not installed. Run: pip install 'pps-embed[clip]'"
        ) from exc

    model, _, preprocess = _clip_lazy_load(open_clip)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil = Image.fromarray(rgb)
    with torch.no_grad():
        feats = model.encode_image(preprocess(pil).unsqueeze(0))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().squeeze(0).astype(np.float32)


_CLIP_CACHE: dict = {}


def _clip_lazy_load(open_clip):  # pragma: no cover
    if "model" not in _CLIP_CACHE:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _CLIP_CACHE.update({"model": model.eval(), "preprocess": preprocess})
    return _CLIP_CACHE["model"], None, _CLIP_CACHE["preprocess"]


# ----------------------------------------------------------------------
# Stable id helper
# ----------------------------------------------------------------------


def stable_photo_id(img: np.ndarray, *, namespace: str = "pps") -> str:
    """SHA-1 of pHash + raw shape, hex-encoded — stable across reruns."""
    h = hashlib.sha1()
    h.update(namespace.encode())
    h.update(np.asarray(img.shape, dtype=np.int64).tobytes())
    h.update(phash_64(img).tobytes())
    return h.hexdigest()


def normalise(vec: np.ndarray | Iterable[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else (v / n).astype(np.float32)
