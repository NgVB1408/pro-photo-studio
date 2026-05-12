"""Photo embedding determinism + dimensions."""

from __future__ import annotations

import numpy as np

from pps_embed.photo_features import (
    LAB_HIST_DIM,
    PHASH_DIM,
    PHOTO_DIM,
    SALIENCY_DIM,
    lab_histogram_96,
    phash_64,
    photo_embedding,
    stable_photo_id,
)


def test_phash_dim(small_image):
    assert phash_64(small_image).shape == (PHASH_DIM,)


def test_lab_histogram_dim(small_image):
    h = lab_histogram_96(small_image)
    assert h.shape == (LAB_HIST_DIM,)
    # Each 32-block sums approximately 1.
    for i in range(0, LAB_HIST_DIM, 32):
        assert abs(h[i : i + 32].sum() - 1.0) < 1e-4


def test_photo_embedding_dim(small_image):
    v = photo_embedding(small_image)
    assert v.shape == (PHOTO_DIM,)
    assert v.dtype == np.float32


def test_photo_embedding_deterministic(small_image):
    a = photo_embedding(small_image)
    b = photo_embedding(small_image)
    assert np.array_equal(a, b)


def test_two_different_photos_have_different_embeddings(small_image, small_image_b):
    a = photo_embedding(small_image)
    b = photo_embedding(small_image_b)
    assert not np.array_equal(a, b)


def test_stable_photo_id_consistent(small_image):
    a = stable_photo_id(small_image)
    b = stable_photo_id(small_image)
    assert a == b
    assert len(a) == 40  # sha-1 hex


def test_embedding_components_breakdown():
    assert PHOTO_DIM == PHASH_DIM + LAB_HIST_DIM + SALIENCY_DIM == 176
