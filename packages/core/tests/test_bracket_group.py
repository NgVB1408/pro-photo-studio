"""Tests cho pps_core.bracket_group — auto-detect bracket sets."""

from __future__ import annotations

from pathlib import Path

from pps_core.bracket_group import (
    BracketGroup,
    PhotoSample,
    _filename_pattern_match,
    group_brackets,
)


def _sample(
    path_str: str, *, ts: float | None = 0.0, ev: float | None = None, brt: float | None = None
):
    return PhotoSample(
        path=Path(path_str),
        capture_ts=ts,
        exposure_bias_ev=ev,
        brightness=brt,
    )


# ---------- group_brackets ----------


class TestGroupBrackets:
    def test_empty_returns_empty(self):
        assert group_brackets([]) == []

    def test_single_photo_becomes_singleton_group(self):
        groups = group_brackets([_sample("a.jpg", ts=100.0, ev=0.0)])
        assert len(groups) == 1
        assert groups[0].size == 1
        assert groups[0].reference.name == "a.jpg"
        assert groups[0].brackets == ()
        assert groups[0].confidence == 1.0
        assert "single" in groups[0].reason

    def test_exif_ev_bracket_detected(self):
        samples = [
            _sample("a.jpg", ts=100.0, ev=-2.0, brt=0.30),
            _sample("b.jpg", ts=101.0, ev=0.0, brt=0.55),
            _sample("c.jpg", ts=102.0, ev=+2.0, brt=0.80),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 1
        g = groups[0]
        assert g.size == 3
        assert g.confidence == 1.0
        assert g.reference.name == "b.jpg"  # median EV
        assert set(p.name for p in g.brackets) == {"a.jpg", "c.jpg"}
        assert "exif" in g.reason

    def test_brightness_fallback_when_ev_missing(self):
        # No EXIF EV — brightness varies enough to trigger fallback.
        samples = [
            _sample("u.jpg", ts=100.0, brt=0.20),
            _sample("m.jpg", ts=100.5, brt=0.50),
            _sample("o.jpg", ts=101.0, brt=0.85),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 1
        assert groups[0].size == 3
        assert groups[0].confidence == 0.6
        assert groups[0].reference.name == "m.jpg"
        assert "brightness" in groups[0].reason

    def test_burst_window_separates_groups(self):
        samples = [
            _sample("a.jpg", ts=100.0, ev=-2.0),
            _sample("b.jpg", ts=101.0, ev=0.0),
            _sample("c.jpg", ts=102.0, ev=+2.0),
            # 30s gap → new burst
            _sample("d.jpg", ts=132.0, ev=-2.0),
            _sample("e.jpg", ts=133.0, ev=0.0),
            _sample("f.jpg", ts=134.0, ev=+2.0),
        ]
        groups = group_brackets(samples, burst_window_s=6.0)
        assert len(groups) == 2
        assert groups[0].size == 3
        assert groups[1].size == 3

    def test_filename_only_fallback_low_confidence(self):
        # All same brightness, no EV → should fall to filename pattern.
        samples = [
            _sample("IMG_0001.jpg", ts=100.0, brt=0.50),
            _sample("IMG_0002.jpg", ts=100.5, brt=0.50),
            _sample("IMG_0003.jpg", ts=101.0, brt=0.50),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 1
        assert groups[0].size == 3
        assert groups[0].confidence == 0.4
        assert "filename" in groups[0].reason

    def test_no_signals_emit_singletons(self):
        # Same brightness, different filename prefixes, no EV — cluster but
        # cannot bracket → singleton groups.
        samples = [
            _sample("alpha.jpg", ts=100.0, brt=0.50),
            _sample("bravo.jpg", ts=101.0, brt=0.50),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 2
        for g in groups:
            assert g.size == 1
            assert "no bracket signal" in g.reason

    def test_mixed_some_groups_some_singletons(self):
        samples = [
            _sample("a1.jpg", ts=100.0, ev=-2.0),
            _sample("a2.jpg", ts=101.0, ev=0.0),
            _sample("a3.jpg", ts=102.0, ev=+2.0),
            # 60s gap, single shot
            _sample("solo.jpg", ts=162.0, ev=0.0),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 2
        assert groups[0].size == 3
        assert groups[1].size == 1
        assert groups[1].reference.name == "solo.jpg"

    def test_min_ev_spread_threshold(self):
        # 0.5 EV spread is below default 1.0 — should NOT EXIF-group.
        # Brightness might still rescue if spread is large enough; here it's
        # not, so falls to no-signal singletons.
        samples = [
            _sample("x.jpg", ts=100.0, ev=-0.3, brt=0.50),
            _sample("y.jpg", ts=101.0, ev=0.0, brt=0.51),
            _sample("z.jpg", ts=102.0, ev=+0.2, brt=0.52),
        ]
        groups = group_brackets(samples)
        assert len(groups) == 3
        for g in groups:
            assert g.size == 1

    def test_custom_burst_window(self):
        samples = [
            _sample("a.jpg", ts=100.0, ev=-2.0),
            _sample("b.jpg", ts=110.0, ev=0.0),  # 10s gap
            _sample("c.jpg", ts=120.0, ev=+2.0),  # 10s gap
        ]
        # default 6s window → 3 separate clusters of 1 each
        default_groups = group_brackets(samples)
        assert len(default_groups) == 3
        # 15s window → all in one bracket
        wide_groups = group_brackets(samples, burst_window_s=15.0)
        assert len(wide_groups) == 1
        assert wide_groups[0].size == 3

    def test_no_timestamps_groups_by_filename(self):
        samples = [
            _sample("IMG_001.jpg", ts=None, ev=-2.0),
            _sample("IMG_002.jpg", ts=None, ev=0.0),
            _sample("IMG_003.jpg", ts=None, ev=+2.0),
            _sample("DSC_001.jpg", ts=None, ev=0.0),
        ]
        groups = group_brackets(samples)
        # IMG_* should bracket together; DSC_* alone.
        sizes = sorted(g.size for g in groups)
        assert sizes == [1, 3]


# ---------- _filename_pattern_match ----------


class TestFilenamePatternMatch:
    def test_matches_img_counter(self):
        cluster = [
            _sample("IMG_0001.jpg"),
            _sample("IMG_0002.jpg"),
            _sample("IMG_0003.jpg"),
        ]
        assert _filename_pattern_match(cluster) is True

    def test_matches_dsc_counter(self):
        cluster = [_sample("DSC00010.jpg"), _sample("DSC00011.jpg")]
        assert _filename_pattern_match(cluster) is True

    def test_rejects_different_prefixes(self):
        cluster = [_sample("IMG_0001.jpg"), _sample("DSC_0001.jpg")]
        assert _filename_pattern_match(cluster) is False

    def test_rejects_no_counter(self):
        cluster = [_sample("photo_kitchen.jpg"), _sample("photo_bath.jpg")]
        # both lack trailing digits → regex fails for both
        assert _filename_pattern_match(cluster) is False

    def test_rejects_single(self):
        assert _filename_pattern_match([_sample("a.jpg")]) is False


# ---------- BracketGroup ----------


class TestBracketGroup:
    def test_members_property(self):
        g = BracketGroup(
            reference=Path("ref.jpg"),
            brackets=(Path("u.jpg"), Path("o.jpg")),
            confidence=1.0,
        )
        assert g.members == (Path("ref.jpg"), Path("u.jpg"), Path("o.jpg"))

    def test_size_property(self):
        g = BracketGroup(
            reference=Path("r.jpg"),
            brackets=(Path("a.jpg"), Path("b.jpg")),
        )
        assert g.size == 3
