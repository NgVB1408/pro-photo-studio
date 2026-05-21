"""Auto-detect bracket groups từ EXIF.

Sony A7M4 AEB chuẩn:
    - DateTimeOriginal giống nhau (camera ghi cùng giây cho cả triplet)
    - Sequential filename (DSC01628, 01629, 01630)
    - ExposureBiasValue khác nhau: thường [-3, 0, +3] hoặc [-2, 0, +2]

Grouping rule:
    1. Sort theo (DateTimeOriginal, filename)
    2. Group nếu: DT cách nhau ≤ time_tolerance_s VÀ EV khác nhau
    3. Min group size = 2, max = 7 (Sony hỗ trợ tới 9 shot bracket)
    4. Mỗi group phải có 1 shot EV gần 0 nhất → làm "reference"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

log = logging.getLogger(__name__)


@dataclass
class BracketShot:
    path: Path
    datetime_original: datetime | None
    ev: float | None
    iso: int | None
    shutter: float | None
    fnumber: float | None

    @property
    def is_reference(self) -> bool:
        return self.ev is not None and abs(self.ev) < 0.5


@dataclass
class BracketGroup:
    shots: list[BracketShot] = field(default_factory=list)

    @property
    def reference(self) -> BracketShot:
        """Shot gần EV=0 nhất → dùng làm filename + EXIF gốc."""
        return min(self.shots, key=lambda s: abs(s.ev) if s.ev is not None else 99)

    @property
    def ev_range(self) -> tuple[float, float]:
        evs = [s.ev for s in self.shots if s.ev is not None]
        return (min(evs), max(evs)) if evs else (0.0, 0.0)

    @property
    def signature(self) -> str:
        ref = self.reference
        return f"{ref.path.stem}_ev[{self.ev_range[0]:+.1f}..{self.ev_range[1]:+.1f}]_n={len(self.shots)}"


def _parse_exif(path: Path) -> BracketShot:
    try:
        with Image.open(path) as img:
            exif = img._getexif() or {}
    except Exception as exc:
        log.warning("EXIF read fail %s: %s", path.name, exc)
        return BracketShot(path, None, None, None, None, None)

    tagged = {TAGS.get(k, k): v for k, v in exif.items()}

    dt = None
    if dt_str := tagged.get("DateTimeOriginal"):
        try:
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass

    def _num(v) -> float | None:
        if v is None:
            return None
        if isinstance(v, tuple) and len(v) == 2:
            return v[0] / v[1] if v[1] else None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return BracketShot(
        path=path,
        datetime_original=dt,
        ev=_num(tagged.get("ExposureBiasValue")),
        iso=tagged.get("ISOSpeedRatings"),
        shutter=_num(tagged.get("ExposureTime")),
        fnumber=_num(tagged.get("FNumber")),
    )


def detect_brackets(
    files: list[Path],
    *,
    time_tolerance_s: float = 3.0,
    min_group: int = 2,
    max_group: int = 7,
) -> tuple[list[BracketGroup], list[Path]]:
    """Group files thành brackets + return singletons không match.

    Returns:
        (groups, singletons): groups là brackets ≥ min_group; singletons là ảnh single-shot.
    """
    shots = [_parse_exif(p) for p in files]
    shots.sort(key=lambda s: (s.datetime_original or datetime.min, s.path.name))

    groups: list[BracketGroup] = []
    current: list[BracketShot] = []

    def _flush():
        if min_group <= len(current) <= max_group:
            evs = [s.ev for s in current if s.ev is not None]
            if len(set(evs)) >= 2:  # must have variation in EV
                groups.append(BracketGroup(shots=list(current)))
                return True
        return False

    for shot in shots:
        if not current:
            current = [shot]
            continue

        prev = current[-1]
        same_window = (
            shot.datetime_original is not None
            and prev.datetime_original is not None
            and abs((shot.datetime_original - prev.datetime_original).total_seconds())
            <= time_tolerance_s
        )
        ev_distinct = shot.ev is not None and shot.ev not in [
            s.ev for s in current if s.ev is not None
        ]

        if same_window and ev_distinct and len(current) < max_group:
            current.append(shot)
        else:
            _flush() or _push_singletons(current, groups)
            current = [shot]

    _flush() or _push_singletons(current, groups)

    # Collect singletons (not grouped) — files that ended up alone
    grouped_paths = {s.path for g in groups for s in g.shots}
    singletons = [p for p in files if p not in grouped_paths]

    return groups, singletons


def _push_singletons(buf: list[BracketShot], groups: list[BracketGroup]) -> bool:
    """Mark buffer as singletons (do nothing — caller collects via diff)."""
    return False
