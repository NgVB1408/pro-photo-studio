"""Auto-detect bracketed exposure sets in a folder of photos.

Real-estate photographers commonly shoot 3–7 brackets per scene (e.g. -2 EV,
0 EV, +2 EV) for HDR fusion. This module groups input files into bracket
sets so the batch runner can fuse each set into one HDR output.

Three signals are combined, in priority order:

1. **EXIF ``ExposureBiasValue``** — when ≥ 2 photos share the same scene
   and have different EV biases, they are treated as a bracket set.
2. **Burst time window** — photos taken within 6 seconds of each other
   (configurable) are candidates. Real estate photographers typically fire
   3 brackets in < 3 s with a tripod or hand-bracket pose.
3. **Filename pattern** — common camera naming conventions group brackets
   under a stem with a counter suffix (``IMG_0001`` … ``IMG_0003``,
   ``DSC00010_-2`` … ``DSC00010_+2``). Used as a tiebreaker.

Brightness fallback (computing the V-channel mean of each candidate) catches
sets where EXIF was stripped but exposure clearly varies.

Port: ``imagen-ai/backend/services/bracket_grouping.py`` (PIL EXIF) plus
   filename + time-window heuristics.

Returns ``list[BracketGroup]`` ordered by capture time. Single photos
(no bracket peers) are returned as 1-element groups.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "BracketGroup",
    "PhotoSample",
    "group_brackets",
    "read_exposure_metadata",
]


@dataclass(frozen=True, slots=True)
class PhotoSample:
    """One photo file with the metadata we need for grouping."""

    path: Path
    capture_ts: float | None
    """Unix timestamp from EXIF DateTimeOriginal, ``None`` if unreadable."""

    exposure_bias_ev: float | None
    """EV bias from EXIF, ``None`` if unreadable."""

    brightness: float | None
    """Mean V-channel brightness as a 0..1 float. Used as fallback when
    EXIF EV is missing. ``None`` if image cannot be read."""

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True, slots=True)
class BracketGroup:
    """A set of photos to be fused into one HDR output."""

    reference: Path
    """The "0 EV" / median-EV photo, used as the geometric and color reference."""

    brackets: tuple[Path, ...] = field(default_factory=tuple)
    """Other photos in the bracket set, in order of EV (under → over)."""

    confidence: float = 1.0
    """0..1 — how strongly we believe these photos are an actual bracket set.
    EXIF-driven groups score 1.0; brightness-only groups score 0.6;
    filename-only groups score 0.4."""

    reason: str = ""
    """Short human-readable explanation of why these were grouped."""

    @property
    def members(self) -> tuple[Path, ...]:
        return (self.reference, *self.brackets)

    @property
    def size(self) -> int:
        return 1 + len(self.brackets)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def group_brackets(
    photos: Iterable[Path | PhotoSample],
    *,
    burst_window_s: float = 6.0,
    min_ev_spread: float = 1.0,
    min_brightness_spread: float = 0.18,
) -> list[BracketGroup]:
    """Group ``photos`` into bracket sets.

    Args:
        photos: Files to group. Either ``Path`` (we'll read EXIF) or pre-built
            ``PhotoSample`` instances (useful for tests or when EXIF was
            already extracted upstream).
        burst_window_s: Maximum seconds between consecutive captures for them
            to be considered the same burst. Default 6s.
        min_ev_spread: Minimum EV difference within a candidate group for the
            EXIF-driven path to accept it. Default 1.0 EV.
        min_brightness_spread: Brightness range (max-min in 0..1) within a
            candidate group for the brightness fallback to accept. Default
            0.18 (≈ 1.5 EV equivalent).

    Returns:
        ``list[BracketGroup]`` ordered by capture time of the reference photo.
        Photos not assigned to any multi-member group come back as
        single-member groups (size=1).
    """
    samples: list[PhotoSample] = []
    for p in photos:
        if isinstance(p, PhotoSample):
            samples.append(p)
        else:
            samples.append(_sample_for_path(p))

    if not samples:
        return []

    samples.sort(
        key=lambda s: (s.capture_ts if s.capture_ts is not None else 0.0, s.path.as_posix())
    )

    # Step 1: split into burst clusters by time gap.
    clusters: list[list[PhotoSample]] = []
    current: list[PhotoSample] = []
    last_ts: float | None = None
    for s in samples:
        if (
            last_ts is not None
            and s.capture_ts is not None
            and (s.capture_ts - last_ts) > burst_window_s
        ):
            if current:
                clusters.append(current)
            current = []
        current.append(s)
        last_ts = s.capture_ts if s.capture_ts is not None else last_ts
    if current:
        clusters.append(current)

    # Photos with no timestamp form their own clusters by filename stem.
    if any(s.capture_ts is None for s in samples):
        clusters = _refine_clusters_by_filename(clusters)

    groups: list[BracketGroup] = []
    for cluster in clusters:
        if len(cluster) < 2:
            groups.append(BracketGroup(reference=cluster[0].path, reason="single (no peers)"))
            continue

        # Try EXIF EV first.
        evs = [s.exposure_bias_ev for s in cluster]
        if all(ev is not None for ev in evs):
            spread = max(evs) - min(evs)  # type: ignore[type-var]
            if spread >= min_ev_spread:
                groups.append(_build_group(cluster, by="exif", confidence=1.0))
                continue

        # Brightness fallback.
        brights = [s.brightness for s in cluster]
        if all(b is not None for b in brights):
            spread = max(brights) - min(brights)  # type: ignore[type-var]
            if spread >= min_brightness_spread:
                groups.append(_build_group(cluster, by="brightness", confidence=0.6))
                continue

        # Filename hint (e.g. IMG_0001/0002/0003 with same prefix and a
        # numeric suffix) — weakest signal.
        if _filename_pattern_match(cluster):
            groups.append(_build_group(cluster, by="filename", confidence=0.4))
            continue

        # Couldn't bracket — emit as singletons.
        for s in cluster:
            groups.append(BracketGroup(reference=s.path, reason="cluster but no bracket signal"))

    return groups


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_for_path(path: Path) -> PhotoSample:
    """Read EXIF + brightness for one file. Robust against missing fields."""
    capture_ts, ev = read_exposure_metadata(path)
    brightness = _read_brightness(path)
    return PhotoSample(
        path=path,
        capture_ts=capture_ts,
        exposure_bias_ev=ev,
        brightness=brightness,
    )


def read_exposure_metadata(path: Path) -> tuple[float | None, float | None]:
    """Return ``(unix_timestamp, exposure_bias_ev)`` from EXIF.

    Both fields may be ``None`` if the EXIF block is missing or unreadable.
    Implementation uses Pillow because it's already a dependency; rawpy
    would be needed for non-JPEG sources but real-estate workflows almost
    always shoot to JPEG.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return None, None

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except (OSError, ValueError):
        return None, None
    if not exif:
        return None, None

    raw = {TAGS.get(tag, tag): value for tag, value in exif.items()}
    ts = _parse_exif_datetime(raw.get("DateTimeOriginal") or raw.get("DateTime"))
    ev = _parse_rational(raw.get("ExposureBiasValue"))
    return ts, ev


def _parse_exif_datetime(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    # EXIF DateTime is "YYYY:MM:DD HH:MM:SS"
    import datetime as _dt

    try:
        dt = _dt.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        return None


def _parse_rational(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        try:
            return float(num) / float(den) if den else 0.0
        except (TypeError, ZeroDivisionError):
            return None
    # PIL sometimes returns a custom Fraction-like object with numerator/denominator
    num = getattr(value, "numerator", None)
    den = getattr(value, "denominator", None)
    if num is not None and den:
        try:
            return float(num) / float(den)
        except (TypeError, ZeroDivisionError):
            return None
    return None


def _read_brightness(path: Path) -> float | None:
    """Mean V-channel brightness of the image, scaled to 0..1.

    Returns ``None`` if the image cannot be read. Used only as a fallback
    when EXIF EV is missing.
    """
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(hsv[..., 2].mean()) / 255.0
    except (ImportError, OSError):
        return None


# ---------------------------------------------------------------------------
# Filename-pattern fallback
# ---------------------------------------------------------------------------


_FILENAME_BRACKET_RE = re.compile(
    r"^(?P<prefix>.+?)[_-]?(?P<num>\d{1,5})$",
)


def _filename_pattern_match(cluster: list[PhotoSample]) -> bool:
    """True iff every sample shares a common prefix and differs only by a
    trailing counter or signed EV suffix."""
    if len(cluster) < 2:
        return False
    prefixes: set[str] = set()
    for s in cluster:
        m = _FILENAME_BRACKET_RE.match(s.stem)
        if not m:
            return False
        prefixes.add(m.group("prefix").rstrip("_-"))
    return len(prefixes) == 1


def _refine_clusters_by_filename(clusters: list[list[PhotoSample]]) -> list[list[PhotoSample]]:
    """When timestamps are missing, regroup by filename prefix so files like
    ``IMG_0001/0002/0003`` end up in the same cluster regardless of mtime."""
    refined: list[list[PhotoSample]] = []
    for cluster in clusters:
        with_ts = [s for s in cluster if s.capture_ts is not None]
        without_ts = [s for s in cluster if s.capture_ts is None]
        if with_ts:
            refined.append(with_ts)
        if not without_ts:
            continue
        by_prefix: dict[str, list[PhotoSample]] = {}
        for s in without_ts:
            m = _FILENAME_BRACKET_RE.match(s.stem)
            key = m.group("prefix").rstrip("_-") if m else s.stem
            by_prefix.setdefault(key, []).append(s)
        refined.extend(by_prefix.values())
    return refined


# ---------------------------------------------------------------------------
# Group assembly
# ---------------------------------------------------------------------------


def _build_group(cluster: list[PhotoSample], *, by: str, confidence: float) -> BracketGroup:
    """Pick the reference photo (median EV / median brightness / middle index)
    and assemble a ``BracketGroup``."""
    if by == "exif":
        ordered = sorted(cluster, key=lambda s: s.exposure_bias_ev or 0.0)
    elif by == "brightness":
        ordered = sorted(cluster, key=lambda s: s.brightness or 0.0)
    else:
        ordered = list(cluster)

    median_idx = len(ordered) // 2
    reference = ordered[median_idx]
    brackets = tuple(s.path for i, s in enumerate(ordered) if i != median_idx)
    return BracketGroup(
        reference=reference.path,
        brackets=brackets,
        confidence=confidence,
        reason=f"by={by}, n={len(cluster)}",
    )
