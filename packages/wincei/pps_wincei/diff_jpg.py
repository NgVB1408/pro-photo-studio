"""Side-by-side comparison JPG generator — fool-proof viewer.

Each output JPG shows:
  +-----------------+-----------------+
  |   BEFORE        |    AFTER        |
  |   (left)        |    (right)      |
  +-----------------+-----------------+
  | diff heatmap (magenta = change)   |
  +-----------------------------------+

Or simpler stack mode: just BEFORE | AFTER, no heatmap.

User opens in any image viewer → sees diff immediately.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _make_label(text: str, width: int, height: int = 48) -> np.ndarray:
    """Create a label bar JPG."""
    bar = np.full((height, width, 3), 25, dtype=np.uint8)  # dark gray
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, text, (16, 32), font, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return bar


def _make_diff_heatmap(before: np.ndarray, after: np.ndarray, *, amp: float = 6.0) -> np.ndarray:
    """Magenta-tinted heatmap showing |after - before| magnitude.
    amp = visual amplification (× factor) to make subtle changes visible.
    """
    diff = cv2.absdiff(before, after).astype(np.float32).mean(axis=2)
    diff = np.clip(diff * amp, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
    return heatmap


def write_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    out_path: str | Path,
    *,
    name: str = "",
    max_width: int = 3840,
    include_diff: bool = True,
    quality: int = 92,
) -> Path:
    """Generate fool-proof comparison JPG: BEFORE | AFTER (+ optional diff heatmap below).

    Args:
        before, after: BGR uint8 same shape.
        out_path: target JPG.
        name: filename label (top).
        max_width: cap output width (downsize if needed).
        include_diff: include diff heatmap row below.
        quality: JPEG quality 1-100.

    Returns:
        out_path (Path).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert before.shape == after.shape, "before/after must match shape"
    h, w = before.shape[:2]

    # Downsize jointly if combined output too wide (w*2 + separator)
    pair_w = w * 2 + 4
    if pair_w > max_width:
        scale = max_width / pair_w
        new_w = int(w * scale)
        new_h = int(h * scale)
        before = cv2.resize(before, (new_w, new_h), interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Separator strip (4px vertical white line)
    sep_v = np.full((h, 4, 3), 200, dtype=np.uint8)

    # Stack horizontally: BEFORE | sep | AFTER
    side_by_side = np.hstack([before, sep_v, after])

    # Top label bar
    label_bar = _make_label(f"{name}    BEFORE  |  AFTER", side_by_side.shape[1])
    panels = [label_bar, side_by_side]

    if include_diff:
        # Diff heatmap stretched across full width
        heat = _make_diff_heatmap(before, after, amp=6.0)
        heat_full = np.hstack([heat, sep_v, heat])
        heat_label = _make_label(
            "DIFF HEATMAP  (magenta=window/ceiling changes, dark=no change, x6 amp)",
            heat_full.shape[1],
            height=36,
        )
        sep_h = np.full((4, side_by_side.shape[1], 3), 200, dtype=np.uint8)
        panels += [sep_h, heat_label, heat_full]

    composite = np.vstack(panels)

    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out_path


def write_index_html(diff_jpgs: list[tuple[str, Path]], out_html: str | Path) -> Path:
    """Simple HTML index — 1 ảnh diff per row, no JS magic, just <img>.

    Args:
        diff_jpgs: list of (label, jpg_path) tuples.
        out_html: output HTML path.
    """
    out_html = Path(out_html)
    out_dir = out_html.parent
    rows = []
    for label, jpg in diff_jpgs:
        try:
            rel = jpg.relative_to(out_dir)
        except ValueError:
            rel = jpg
        rows.append(f"""
<div class="row">
  <h3>{label}</h3>
  <img src="{rel.as_posix()}" alt="{label}">
</div>""")
    body = "\n".join(rows)
    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>WinCei comparison</title>
<style>
body {{ margin: 0; padding: 24px; background: #0f1419; color: #e6edf3; font-family: sans-serif; }}
.row {{ margin-bottom: 32px; }}
.row h3 {{ margin: 0 0 8px; font-size: 16px; }}
.row img {{ display: block; width: 100%; max-width: 1800px; border-radius: 6px; }}
</style>
</head><body>
<h1>🪟🏠 Window + Ceiling — Diff comparison ({len(diff_jpgs)} ảnh)</h1>
<p>Mỗi ảnh: BEFORE | AFTER trên cùng, heatmap diff (magenta=vùng đã chỉnh) dưới.</p>
{body}
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html
