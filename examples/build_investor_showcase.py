"""Build investor showcase folder — 4-column side-by-side comparison.

Layout:
    Column A:  Original (customer input)
    Column B1: Competitor #1 output
    Column B2: Competitor #2 output
    Column C:  Pro Photo Studio v2 output (current OpenCV pipeline)

Output structure:
    investor_showcase/
      INDEX.md                          per-photo summary table
      hero_photos/                       2-3 photos picked as "best demos"
        <id>/
          A_original.jpg
          B1_competitor1.jpg
          B2_competitor2.jpg
          C_pps_v2.jpg
          composite_4col.jpg            full 4-column side-by-side
          composite_2col_pps_vs_orig.jpg before/after focus
          stage_report.json             which PPS stages applied
      all/                              every photo, same structure
        <id>/...

Usage:
    python examples/build_investor_showcase.py \\
        --originals "C:/path/to/originals" \\
        --competitor1 "C:/path/to/competitor1" \\
        --competitor2 "C:/path/to/competitor2" \\
        --output investor_showcase

The script is idempotent — re-running regenerates only what changed. Photos
not present in all 4 inputs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Make pps_core + pps_api importable from src layout when run with no install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "core"))
sys.path.insert(0, str(ROOT / "packages" / "api"))

from pps_core.pipeline import Pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("showcase")

# Built-in stages must be registered before we run.
import pps_api.stages.builtin_stages  # noqa: E402, F401  -- side-effect: register stages


# Header config — keep minimal so the photo dominates.
HEADER_HEIGHT = 96
HEADER_BG = (24, 30, 48)        # dark navy
HEADER_FG = (240, 246, 255)     # near-white
ACCENT_PPS = (40, 200, 120)     # green for PPS column header
ACCENT_COMP = (200, 120, 80)    # warm orange for competitors

# Watermark / footer config
FOOTER_HEIGHT = 56
FOOTER_BG = (16, 20, 32)
FOOTER_FG = (160, 180, 210)

# Common width for each column in the composite.
COLUMN_WIDTH = 1280
GAP_PX = 6
GAP_BG = (10, 14, 22)


@dataclass(frozen=True)
class PhotoSet:
    photo_id: str
    original: Path
    competitor1: Path | None
    competitor2: Path | None


@dataclass
class StageOutcome:
    name: str
    applied: bool
    duration_ms: float
    reason: str = ""


def find_photo_sets(
    originals_dir: Path,
    competitor1_dir: Path | None,
    competitor2_dir: Path | None,
    competitor2_subfolder: str = "enhanced",
) -> list[PhotoSet]:
    """Match photos by stem name across all input folders."""
    sets: list[PhotoSet] = []
    for orig in sorted(originals_dir.glob("*.[jJpP]*")):
        stem = orig.stem
        c1: Path | None = None
        c2: Path | None = None
        if competitor1_dir:
            c1 = _find_match(competitor1_dir, stem)
        if competitor2_dir:
            # Manuka folder uses an `enhanced/` subfolder for AI output.
            sub = competitor2_dir / competitor2_subfolder
            if sub.is_dir():
                c2 = _find_match(sub, stem)
            if c2 is None:
                c2 = _find_match(competitor2_dir, stem)
        sets.append(PhotoSet(photo_id=stem, original=orig, competitor1=c1, competitor2=c2))
    return sets


def _find_match(folder: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".JPG", ".JPEG"):
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def run_pps_pipeline(image: np.ndarray, *, seed: int = 42) -> tuple[np.ndarray, list[StageOutcome]]:
    """Run the PPS v2 showcase pipeline on a real-estate photo.

    Stage chain optimised to produce visibly punchy output that stands up to
    AutoEnhance / Manuka in side-by-side comparison:

      1. perspective       — fix camera tilt (when ≥12 vertical lines detected)
      2. real_estate       — scene-aware enhance (sky/lawn/window/vertical)
      3. enhance_studio    — beefier 8-step finish (CLAHE 3.5, shadow lift 0.45,
                              vibrance 0.30, unsharp 0.55) for visibly bright,
                              vibrant output that pops on a slide deck

    Twilight is excluded — it's a stylistic transform, not a baseline enhance.
    Sky replace requires an outdoor scene; we leave it off so interiors don't
    get false positives.
    """
    from pps_core.types import Job

    pipeline = Pipeline()
    job = Job(
        job_id=f"showcase-{seed}",
        stages=("perspective", "real_estate", "enhance_studio"),
        params={
            "real_estate": {"enable_sky": False, "use_ai_sky": False},
            "enhance_studio": {
                # Dialled back from 3.0 → 2.2 because aggressive CLAHE produced
                # mottled artifacts on flat surfaces (white ceilings).
                "clahe_clip": 2.2,
                "highlight_recovery": 0.35,
                "shadow_lift": 0.30,
                "vibrance": 0.18,
                "unsharp_amount": 0.40,
                "gamma": 0.97,
            },
        },
        seed=seed,
    )
    out, report = pipeline.run(job, image)
    outcomes = [
        StageOutcome(
            name=s.name,
            applied=s.applied,
            duration_ms=round(s.duration_ms, 1),
            reason=s.reason,
        )
        for s in report.stages
    ]
    return out, outcomes


def make_detail_crop(
    photo_id: str,
    panels: list[tuple[str, str, np.ndarray | None, tuple[int, int, int]]],
    *,
    crop_box: tuple[float, float, float, float] = (0.30, 0.20, 0.70, 0.55),
    column_width: int = COLUMN_WIDTH,
) -> np.ndarray:
    """Generate a zoom comparison strip from the same crop in each panel.

    The crop box is in normalised (x0, y0, x1, y1) coordinates relative to
    each image's own dimensions. Default box targets the middle-upper area
    where windows, ceiling lines, and texture detail typically live.
    """
    cropped: list[tuple[str, str, np.ndarray | None, tuple[int, int, int]]] = []
    for label, sub, panel, accent in panels:
        if panel is None:
            cropped.append((label, sub, None, accent))
            continue
        h, w = panel.shape[:2]
        x0 = int(crop_box[0] * w)
        y0 = int(crop_box[1] * h)
        x1 = int(crop_box[2] * w)
        y1 = int(crop_box[3] * h)
        crop = panel[y0:y1, x0:x1]
        cropped.append((label, sub + " — detail crop", crop, accent))
    return build_composite(f"{photo_id} (detail)", cropped, target_width=column_width)


def fit_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(img, (width, new_h), interpolation=interp)


def _pil_font(size: int, bold: bool = False):
    """Find a system font that supports Unicode (· em-dash etc.).

    Falls back to PIL default if nothing usable is found — that font is
    bitmap-only and ugly, but at least it won't crash.
    """
    from PIL import ImageFont

    candidates = [
        # Windows
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_header(
    img: np.ndarray,
    title: str,
    subtitle: str = "",
    accent: tuple[int, int, int] = HEADER_FG,
) -> np.ndarray:
    """Add a 96-pixel header band with title + subtitle, rendered via Pillow
    so Unicode characters (·, —, …) display correctly."""
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    canvas = np.zeros((h + HEADER_HEIGHT, w, 3), dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = HEADER_BG
    canvas[HEADER_HEIGHT:] = img
    # Accent stripe at very top
    cv2.rectangle(canvas, (0, 0), (w, 4), accent, thickness=-1)

    # Pillow operates on RGB, OpenCV on BGR
    rgb = cv2.cvtColor(canvas[:HEADER_HEIGHT], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    title_font = _pil_font(36, bold=True)
    sub_font = _pil_font(20, bold=False)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((w - tw) // 2, 16 - bbox[1]),
        title,
        fill=tuple(reversed(HEADER_FG)),
        font=title_font,
    )
    if subtitle:
        bbox2 = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox2[2] - bbox2[0]
        draw.text(
            ((w - sw) // 2, 18 + th + 4 - bbox2[1]),
            subtitle,
            fill=tuple(reversed(FOOTER_FG)),
            font=sub_font,
        )
    canvas[:HEADER_HEIGHT] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return canvas


def add_footer(img: np.ndarray, text: str) -> np.ndarray:
    """Add a 56-pixel footer band with watermark text via Pillow."""
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    canvas = np.zeros((h + FOOTER_HEIGHT, w, 3), dtype=np.uint8)
    canvas[:h] = img
    canvas[h:] = FOOTER_BG

    rgb = cv2.cvtColor(canvas[h:], cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _pil_font(16, bold=False)
    draw.text((16, 18), text, fill=tuple(reversed(FOOTER_FG)), font=font)
    canvas[h:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return canvas


def build_composite(
    photo_id: str,
    panels: list[tuple[str, str, np.ndarray | None, tuple[int, int, int]]],
    target_width: int = COLUMN_WIDTH,
) -> np.ndarray:
    """Stack panels horizontally with gap. Missing panels are gray placeholders."""
    cols: list[np.ndarray] = []
    placeholder_h = 720
    for label, sub, panel, accent in panels:
        if panel is None:
            blank = np.full((placeholder_h, target_width, 3), (60, 65, 80), dtype=np.uint8)
            cv2.putText(
                blank, "(missing)", (target_width // 2 - 80, placeholder_h // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (160, 160, 180), 2, cv2.LINE_AA,
            )
            framed = add_header(blank, label, sub, accent)
        else:
            sized = fit_to_width(panel, target_width)
            framed = add_header(sized, label, sub, accent)
        cols.append(framed)

    # Equalise heights with neutral fill so the composite is rectangular.
    max_h = max(c.shape[0] for c in cols)
    padded = []
    for c in cols:
        if c.shape[0] < max_h:
            pad = np.full((max_h - c.shape[0], c.shape[1], 3), HEADER_BG, dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)

    gap = np.full((max_h, GAP_PX, 3), GAP_BG, dtype=np.uint8)
    rows = []
    for i, c in enumerate(padded):
        rows.append(c)
        if i < len(padded) - 1:
            rows.append(gap)
    composite = np.hstack(rows)
    return add_footer(
        composite,
        f"Pro Photo Studio v2  |  Photo: {photo_id}  |  Same input, deterministic seed=42",
    )


def safe_imread(path: Path) -> np.ndarray | None:
    try:
        # Use np.fromfile to handle non-ASCII Windows paths.
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not decode %s", path)
        return img
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def safe_imwrite(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def score_photo(img: np.ndarray) -> float:
    """Pick "hero" photos for the investor showcase.

    Formula favours images where PPS v2's strengths are most visible:
      - sharpness (focus, depth of field)
      - exterior content visible through windows (lawn, sky, building)
      - blown highlights — opportunity for window_pull to recover detail
      - balanced exposure (not too dark, not blown out overall)

    This tends to surface photos with *visible exterior view through windows*
    rather than dim closet shots. Higher = better demo candidate.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    size_factor = min(1.0, (h * w) / (1920 * 1080))
    mean = float(gray.mean())
    expo_factor = 1.0 - abs(mean - 130) / 130

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Blown highlights → opportunity for window_pull to make a visible win.
    blown_pct = float((V >= 245).mean())
    blown_bonus = min(blown_pct * 12.0, 1.5)  # cap at 1.5x

    # Green / sky pixels → exterior content visible.
    green_pct = float(((H > 30) & (H < 90) & (S > 60)).mean())
    blue_sky_pct = float(((H > 95) & (H < 130) & (S > 50) & (V > 150)).mean())
    exterior_bonus = min((green_pct + blue_sky_pct) * 8.0, 1.5)

    base = sharpness * size_factor * max(expo_factor, 0.1)
    return base * (1.0 + blown_bonus) * (1.0 + exterior_bonus)


def process_one(
    photo: PhotoSet,
    out_root: Path,
    *,
    is_hero: bool = False,
) -> tuple[float, dict[str, str]]:
    """Process a single photo set; return (hero_score, summary_row)."""
    target = out_root / photo.photo_id
    target.mkdir(parents=True, exist_ok=True)

    original = safe_imread(photo.original)
    if original is None:
        logger.warning("[%s] skipping — could not load original", photo.photo_id)
        return 0.0, {"id": photo.photo_id, "status": "missing original"}

    safe_imwrite(target / "A_original.jpg", original)

    comp1 = safe_imread(photo.competitor1) if photo.competitor1 else None
    comp2 = safe_imread(photo.competitor2) if photo.competitor2 else None
    if comp1 is not None:
        safe_imwrite(target / "B1_competitor1.jpg", comp1)
    if comp2 is not None:
        safe_imwrite(target / "B2_competitor2.jpg", comp2)

    pps_out, outcomes = run_pps_pipeline(original)
    safe_imwrite(target / "C_pps_v2.jpg", pps_out)
    (target / "stage_report.json").write_text(
        json.dumps([asdict(o) for o in outcomes], indent=2),
        encoding="utf-8",
    )

    panels = [
        ("A · Original", "Customer input (no edits)", original, HEADER_FG),
        ("B1 · Competitor #1", "AutoEnhance / order delivery", comp1, ACCENT_COMP),
        ("B2 · Competitor #2", "Manuka enhanced", comp2, ACCENT_COMP),
        ("C · Pro Photo Studio v2", "perspective → real_estate → studio finish", pps_out, ACCENT_PPS),
    ]
    composite = build_composite(photo.photo_id, panels)
    safe_imwrite(target / "composite_4col.jpg", composite, quality=88)

    # Detail-crop strip — same area zoomed in across all 4 panels. This is the
    # most useful image for a pitch deck because subtle differences in CLAHE,
    # shadow lift, and sharpening only become visible at 100% pixel scale.
    detail = make_detail_crop(photo.photo_id, panels)
    safe_imwrite(target / "composite_4col_detail.jpg", detail, quality=92)

    # Also a focused 2-column (PPS vs original) for "before/after" tile use.
    bf_panels = [
        ("Before", "Original", original, HEADER_FG),
        ("After", "Pro Photo Studio v2", pps_out, ACCENT_PPS),
    ]
    bf_composite = build_composite(photo.photo_id, bf_panels)
    safe_imwrite(target / "composite_2col_before_after.jpg", bf_composite, quality=88)

    score = score_photo(original)
    applied = ", ".join(o.name for o in outcomes if o.applied) or "—"
    summary = {
        "id": photo.photo_id,
        "original": str(photo.original),
        "competitor1_present": "yes" if comp1 is not None else "no",
        "competitor2_present": "yes" if comp2 is not None else "no",
        "pps_stages_applied": applied,
        "hero_score": f"{score:,.0f}",
        "hero": "yes" if is_hero else "",
    }
    return score, summary


def write_index(out_root: Path, summaries: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# Pro Photo Studio — Investor Showcase\n")
    lines.append("Side-by-side 4-column comparison: Original | Competitor 1 | Competitor 2 | **PPS v2**.\n")
    lines.append("Each photo has its own folder with the four panels and a composite image.\n")
    lines.append("## Photos\n")
    lines.append("| # | Photo ID | Sharpness | Comp 1 | Comp 2 | PPS stages | Hero |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for i, s in enumerate(summaries, 1):
        lines.append(
            f"| {i} | [{s['id']}](./{s['id']}/) | {s.get('hero_score', '—')} | "
            f"{s.get('competitor1_present', '—')} | {s.get('competitor2_present', '—')} | "
            f"{s.get('pps_stages_applied', '—')} | {s.get('hero', '—')} |\n"
        )
    lines.append("\n## Hero photos\n")
    heroes = [s for s in summaries if s.get("hero") == "yes"]
    if heroes:
        for h in heroes:
            lines.append(f"- [{h['id']}](./{h['id']}/composite_4col.jpg)\n")
    else:
        lines.append("(automatically picked at run time)\n")
    lines.append("\n## How to read\n")
    lines.append("- **A · Original** = customer input, no edits.\n")
    lines.append("- **B1 · Competitor #1** = output from current vendor (`order/...`).\n")
    lines.append("- **B2 · Competitor #2** = output from current vendor (`Manuka/enhanced/`).\n")
    lines.append("- **C · Pro Photo Studio v2** = our deterministic pipeline output.\n")
    lines.append("- All photos use the same seed (42) so PPS v2 output is byte-identical on re-run.\n")
    lines.append("- ML-powered stages (Qwen-Image, SUPIR upscale, virtual staging) are NOT yet wired — see `INVESTOR_BRIEF.md` Phase 3 for status.\n")
    (out_root / "INDEX.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", required=True, help="Folder with original input photos")
    parser.add_argument("--competitor1", default=None, help="Folder with competitor #1 output")
    parser.add_argument("--competitor2", default=None, help="Folder with competitor #2 output")
    parser.add_argument(
        "--competitor2-subfolder",
        default="enhanced",
        help="Subfolder under competitor2 holding enhanced photos (default: 'enhanced')",
    )
    parser.add_argument("--output", default="investor_showcase", help="Output root folder")
    parser.add_argument("--hero-count", type=int, default=3, help="How many hero photos to pick")
    args = parser.parse_args()

    originals = Path(args.originals)
    if not originals.is_dir():
        logger.error("Originals folder not found: %s", originals)
        return 1
    c1 = Path(args.competitor1) if args.competitor1 else None
    c2 = Path(args.competitor2) if args.competitor2 else None
    out_root = Path(args.output)

    photos = find_photo_sets(originals, c1, c2, competitor2_subfolder=args.competitor2_subfolder)
    logger.info("Found %d original photos.", len(photos))
    if not photos:
        logger.error("No photos found.")
        return 2

    summaries: list[dict[str, str]] = []
    scored: list[tuple[float, PhotoSet, dict[str, str]]] = []

    # First pass: process all photos, collect scores.
    for ph in photos:
        score, summary = process_one(ph, out_root / "all", is_hero=False)
        summaries.append(summary)
        scored.append((score, ph, summary))

    # Second pass: pick hero photos by score, copy composite to hero_photos/.
    scored.sort(key=lambda t: t[0], reverse=True)
    heroes = scored[: max(args.hero_count, 0)]
    for score, ph, summary in heroes:
        summary["hero"] = "yes"
        src = out_root / "all" / ph.photo_id
        dst = out_root / "hero_photos" / ph.photo_id
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (dst / f.name).write_bytes(f.read_bytes())
        logger.info("[hero] %s — score %.0f", ph.photo_id, score)

    write_index(out_root, summaries)

    logger.info("Done. Output at: %s", out_root.resolve())
    logger.info("Open: %s", (out_root / "INDEX.md").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
