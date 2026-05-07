"""Director QC tests — the 3 self-questions and SOP scorers."""

from __future__ import annotations

import cv2
import numpy as np

from pps_agents.director import DirectorAgent


def _flat(h=200, w=300, color=(170, 170, 170)) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def test_q1_clean_image_has_no_halo():
    img = _flat()
    d = DirectorAgent()
    score = d.q1_halo_window_corners(img)
    # No bright pixels at all → can't have halo
    assert score == 1.0


def test_q1_halo_detected_around_bright_region():
    img = _flat(color=(80, 80, 80))
    # Add a halo: bright square + brighter ring around it.
    cv2.rectangle(img, (60, 40), (140, 120), (252, 252, 252), -1)
    # Ring 6 px outside (artificially bright) — synthetic halo
    cv2.rectangle(img, (54, 34), (146, 126), (200, 200, 200), 6)
    d = DirectorAgent()
    score = d.q1_halo_window_corners(img)
    assert score < 0.95


def test_q2_neutral_ceiling():
    img = _flat(color=(240, 240, 240))  # neutral white
    d = DirectorAgent()
    s = d.q2_ceiling_neutrality(img)
    assert s > 0.9


def test_q2_blue_tinted_ceiling_penalised():
    img = _flat(color=(245, 235, 220))  # blue-ish white in BGR (B>R)
    d = DirectorAgent()
    s = d.q2_ceiling_neutrality(img)
    assert s < 0.85


def test_q3_move_in_feel_in_range():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 230, size=(400, 600, 3), dtype=np.uint8)
    d = DirectorAgent()
    s = d.q3_move_in_feel(base)
    assert 0.0 <= s <= 1.0


def test_review_returns_verdict_and_scores(small_interior):
    d = DirectorAgent()
    review = d.review(small_interior, small_interior)
    assert review.verdict in {"pass", "review", "fail"}
    assert set(review.question_scores) == {
        "Q1_halo_window_corners",
        "Q2_ceiling_neutrality",
        "Q3_move_in_feel",
    }
    assert 0 <= review.overall_score <= 1
