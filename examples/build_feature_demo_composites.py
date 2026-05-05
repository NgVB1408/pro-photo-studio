"""Build slide-ready before/after composites from manual ML demo outputs.

When the user finishes the Colab notebook runs in MANUAL_ML_DEMO_GUIDE.md,
they drop the output JPEGs into folders matching this convention:

    investor_showcase/feature_demos/
      virtual_staging/
        <id>_before_*.jpg
        <id>_after_*.jpg
      multi_angle/
        <id>_original.jpg
        <id>_angle_left.jpg
        <id>_angle_right.jpg
        <id>_angle_top.jpg
      instruction_edit/
        <id>_original.jpg
        <id>_<task>.jpg            (e.g. brighten_kitchen, no_reflection)
      upscale_supir/
        <id>_original.jpg
        <id>_2x_supir.jpg

This script walks each feature folder, pairs originals with their results,
and writes 2-/N-column composites with branded headers ready to paste into
a slide deck.

Usage:
    python examples/build_feature_demo_composites.py
    python examples/build_feature_demo_composites.py --root path/to/feature_demos
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

# Reuse header / composite helpers from the showcase script.
from build_investor_showcase import (  # noqa: E402
    ACCENT_COMP,
    ACCENT_PPS,
    HEADER_FG,
    add_footer,
    build_composite,
    safe_imread,
    safe_imwrite,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature-demos")


@dataclass(frozen=True)
class Pairing:
    """One before/after (or original/N-variant) pairing."""

    photo_id: str
    feature: str
    before: tuple[str, str, Path]
    afters: list[tuple[str, str, Path]]


def _split_id_and_label(stem: str) -> tuple[str, str]:
    """Split a filename stem like ``5044_after_furnished`` into
    ``("5044", "after_furnished")``. Falls back to ("", stem) if no match."""
    match = re.match(r"^(?P<id>[A-Za-z0-9]+?)_(?P<label>.+)$", stem)
    if not match:
        return "", stem
    return match.group("id"), match.group("label")


def collect_virtual_staging(folder: Path) -> list[Pairing]:
    pairings: list[Pairing] = []
    befores = sorted(folder.glob("*before*.jpg")) + sorted(folder.glob("*before*.jpeg"))
    for b in befores:
        pid, _ = _split_id_and_label(b.stem)
        afters = [
            f
            for f in sorted(folder.glob(f"{pid}_after*.jp*g"))
            if f != b
        ]
        if not afters:
            continue
        pairings.append(
            Pairing(
                photo_id=pid,
                feature="virtual_staging",
                before=("Before · Empty Room", "Customer input", b),
                afters=[
                    ("After · Furnished by PPS v2", "SD3.5 + IPAdapter", a)
                    for a in afters
                ],
            )
        )
    return pairings


def collect_multi_angle(folder: Path) -> list[Pairing]:
    pairings: list[Pairing] = []
    originals = sorted(folder.glob("*original*.jpg")) + sorted(folder.glob("*original*.jpeg"))
    for orig in originals:
        pid, _ = _split_id_and_label(orig.stem)
        angles = [
            f
            for f in sorted(folder.glob(f"{pid}_angle_*.jp*g"))
            if f != orig
        ]
        if not angles:
            continue
        afters: list[tuple[str, str, Path]] = []
        for a in angles:
            _, label = _split_id_and_label(a.stem)
            display = label.replace("angle_", "").replace("_", " ").title()
            afters.append((f"Angle · {display}", "Qwen-Edit-2509", a))
        pairings.append(
            Pairing(
                photo_id=pid,
                feature="multi_angle",
                before=("Original Angle", "Customer input", orig),
                afters=afters,
            )
        )
    return pairings


def collect_instruction_edit(folder: Path) -> list[Pairing]:
    pairings: list[Pairing] = []
    originals = sorted(folder.glob("*original*.jpg")) + sorted(folder.glob("*original*.jpeg"))
    for orig in originals:
        pid, _ = _split_id_and_label(orig.stem)
        edits = [
            f
            for f in sorted(folder.glob(f"{pid}_*.jp*g"))
            if f != orig and "original" not in f.stem
        ]
        if not edits:
            continue
        afters: list[tuple[str, str, Path]] = []
        for e in edits:
            _, label = _split_id_and_label(e.stem)
            instruction = label.replace("_", " ")
            afters.append((f'"{instruction}"', "Qwen-Image-Lightning", e))
        pairings.append(
            Pairing(
                photo_id=pid,
                feature="instruction_edit",
                before=("Original", "Customer input", orig),
                afters=afters,
            )
        )
    return pairings


def collect_upscale(folder: Path) -> list[Pairing]:
    pairings: list[Pairing] = []
    originals = sorted(folder.glob("*original*.jpg")) + sorted(folder.glob("*original*.jpeg"))
    for orig in originals:
        pid, _ = _split_id_and_label(orig.stem)
        upscaled = [
            f
            for f in sorted(folder.glob(f"{pid}_*.jp*g"))
            if f != orig and ("supir" in f.stem.lower() or "2x" in f.stem.lower() or "4x" in f.stem.lower())
        ]
        if not upscaled:
            continue
        pairings.append(
            Pairing(
                photo_id=pid,
                feature="upscale_supir",
                before=("Original (1x)", "Customer input", orig),
                afters=[
                    ("Upscaled · SUPIR", "SOTA 2025 restoration", a)
                    for a in upscaled
                ],
            )
        )
    return pairings


COLLECTORS = {
    "virtual_staging": collect_virtual_staging,
    "multi_angle": collect_multi_angle,
    "instruction_edit": collect_instruction_edit,
    "upscale_supir": collect_upscale,
}


def render_pairing(pairing: Pairing, *, target_dir: Path) -> Path | None:
    """Render and save the slide-ready composite for one pairing."""
    panels: list[tuple[str, str, np.ndarray | None, tuple[int, int, int]]] = []
    before_label, before_sub, before_path = pairing.before
    before_img = safe_imread(before_path)
    if before_img is None:
        logger.warning("[skip] %s: cannot read %s", pairing.photo_id, before_path)
        return None
    panels.append((before_label, before_sub, before_img, HEADER_FG))

    for label, sub, path in pairing.afters:
        img = safe_imread(path)
        if img is None:
            logger.warning("[skip pair] %s: cannot read %s", pairing.photo_id, path)
            continue
        panels.append((label, sub, img, ACCENT_PPS))

    if len(panels) < 2:
        return None

    composite = build_composite(f"{pairing.photo_id} · {pairing.feature}", panels)
    composite = add_footer(
        composite,
        f"Pro Photo Studio v2  |  Feature: {pairing.feature.replace('_', ' ').title()}  "
        f"|  Photo: {pairing.photo_id}",
    )
    out = target_dir / f"{pairing.photo_id}_{pairing.feature}_composite.jpg"
    safe_imwrite(out, composite, quality=92)
    logger.info("[ok] %s · %s → %s", pairing.feature, pairing.photo_id, out.name)
    return out


def write_index(root: Path, all_pairings: dict[str, list[Pairing]]) -> None:
    lines: list[str] = []
    lines.append("# Feature Demo Composites — Slide Ready\n\n")
    lines.append("Generated from manual Colab notebook output by `build_feature_demo_composites.py`.\n\n")
    lines.append("Drag any of these JPEGs into a slide deck. They are 1280×N per panel,\n")
    lines.append("composited side-by-side with branded headers.\n\n")
    for feature, pairings in all_pairings.items():
        if not pairings:
            continue
        title = feature.replace("_", " ").title()
        lines.append(f"## {title}\n\n")
        for p in pairings:
            jpg = f"{feature}/{p.photo_id}_{p.feature}_composite.jpg"
            lines.append(f"- **{p.photo_id}** — [{jpg}]({jpg})\n")
        lines.append("\n")
    if not any(all_pairings.values()):
        lines.append("_No demos found. Run notebooks per `docs/MANUAL_ML_DEMO_GUIDE.md` and drop output into the folder._\n")
    (root / "INDEX.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="investor_showcase/feature_demos",
        help="Root folder containing virtual_staging/, multi_angle/, etc.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        logger.warning("Root not found, creating: %s", root)
        root.mkdir(parents=True, exist_ok=True)
        for feature in COLLECTORS:
            (root / feature).mkdir(exist_ok=True)
        logger.info("Empty folder structure created. Drop demo outputs and re-run.")
        return 0

    all_pairings: dict[str, list[Pairing]] = {}
    for feature, collector in COLLECTORS.items():
        folder = root / feature
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
            all_pairings[feature] = []
            continue
        pairings = collector(folder)
        all_pairings[feature] = pairings
        for p in pairings:
            render_pairing(p, target_dir=folder)

    write_index(root, all_pairings)
    total = sum(len(v) for v in all_pairings.values())
    logger.info("Done. %d composite(s) generated. Index: %s", total, (root / "INDEX.md").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
