"""Shared fixtures: synthesise a real-estate-ish interior for pipeline tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def _interior_synthetic(h: int = 720, w: int = 1080) -> np.ndarray:
    """Build an image that has the photographic features the agents look for:

    - bright window region (clipping highlights, blown out)
    - mid-grey wall + ceiling
    - warm wood-toned floor (hue band)
    - dark sofa region (deep shadows)
    - a tilted vertical line (wall edge) so geometry has work to do
    """
    img = np.full((h, w, 3), 145, dtype=np.uint8)  # mid grey

    # Ceiling: cool white wall — slightly bluish to test Q2 (neutrality).
    img[: h // 4, :] = (200, 198, 192)

    # Floor: wood (BGR ≈ warm brown).
    img[int(h * 0.65) :, :] = (60, 95, 145)
    # Some grain pattern
    rng = np.random.default_rng(seed=42)
    grain = rng.normal(0, 6, (h - int(h * 0.65), w, 3)).astype(np.int16)
    img[int(h * 0.65) :, :] = np.clip(img[int(h * 0.65) :, :].astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Window: bright rectangle near right
    cv2.rectangle(img, (int(w * 0.55), int(h * 0.18)),
                  (int(w * 0.92), int(h * 0.55)), (252, 252, 252), -1)

    # Dark sofa at left
    cv2.rectangle(img, (int(w * 0.05), int(h * 0.55)),
                  (int(w * 0.40), int(h * 0.78)), (28, 24, 22), -1)

    # Tilted vertical (a wall corner line) — 3° off vertical
    cv2.line(img, (int(w * 0.25), 5), (int(w * 0.25) - 35, h - 5),
             (90, 88, 86), thickness=3)
    cv2.line(img, (int(w * 0.50), 5), (int(w * 0.50) - 25, h - 5),
             (88, 85, 82), thickness=2)

    # TV: dark rectangle on wall
    cv2.rectangle(img, (int(w * 0.10), int(h * 0.30)),
                  (int(w * 0.32), int(h * 0.50)), (12, 12, 12), -1)
    return img


@pytest.fixture
def interior_image() -> np.ndarray:
    return _interior_synthetic()


@pytest.fixture
def small_interior() -> np.ndarray:
    return _interior_synthetic(360, 540)
