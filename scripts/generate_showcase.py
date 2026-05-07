"""Render a showcase pack: before / after / side-by-side / diff heatmap / detail crops.

Defaults to the synthetic interior fixture used by the agent unit tests so the
script works offline with zero external assets. Pass ``--input <path>`` to
render a real photo.

Output: ``docs/showcase/<scene>/{before,after,compare,diff,crop_*}.jpg``
        + ``scorecard.json`` with the Director's verdict.

Usage:

    python scripts/generate_showcase.py
    python scripts/generate_showcase.py --input fixtures/villa.jpg \\
        --property villa_luxury --scene villa-luxury --long-edge 1280
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agents"))

from pps_agents.orchestrator import Orchestrator  # noqa: E402
from pps_agents.types import JobContext  # noqa: E402

# Same generator used in agents/tests/conftest.py — keeps the demo
# reproducible without adding any external image to the repo.
def _synthetic_interior(h: int = 720, w: int = 1080) -> np.ndarray:
    img = np.full((h, w, 3), 145, dtype=np.uint8)
    img[: h // 4, :] = (200, 198, 192)
    img[int(h * 0.65):, :] = (60, 95, 145)
    rng = np.random.default_rng(seed=42)
    grain = rng.normal(0, 6, (h - int(h * 0.65), w, 3)).astype(np.int16)
    img[int(h * 0.65):, :] = np.clip(
        img[int(h * 0.65):, :].astype(np.int16) + grain, 0, 255
    ).astype(np.uint8)
    cv2.rectangle(img, (int(w * 0.55), int(h * 0.18)),
                  (int(w * 0.92), int(h * 0.55)), (252, 252, 252), -1)
    cv2.rectangle(img, (int(w * 0.05), int(h * 0.55)),
                  (int(w * 0.40), int(h * 0.78)), (28, 24, 22), -1)
    cv2.line(img, (int(w * 0.25), 5), (int(w * 0.25) - 35, h - 5),
             (90, 88, 86), thickness=3)
    cv2.line(img, (int(w * 0.50), 5), (int(w * 0.50) - 25, h - 5),
             (88, 85, 82), thickness=2)
    cv2.rectangle(img, (int(w * 0.10), int(h * 0.30)),
                  (int(w * 0.32), int(h * 0.50)), (12, 12, 12), -1)
    return img


# ROIs in (x0_frac, y0_frac, x1_frac, y1_frac) for each detail crop.
DETAIL_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Q1 halo check — the bright window corner where halos appear at 200% zoom
    "window_corner": (0.50, 0.15, 0.95, 0.55),
    # Q2 ceiling neutrality check — top quarter of the frame
    "ceiling": (0.05, 0.00, 0.50, 0.22),
    # Wood floor — texture pop from MicroContrastAgent
    "floor_wood": (0.00, 0.65, 0.55, 1.00),
    # Sofa shadow — Q3 move-in feel + shadow integrity
    "sofa_shadow": (0.00, 0.50, 0.45, 0.82),
}


def _label(image: np.ndarray, text: str, *, color=(255, 255, 255)) -> np.ndarray:
    """Stamp a label band at the top of the image."""
    img = image.copy()
    h, w = img.shape[:2]
    band = max(28, h // 24)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, band), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(
        img, text, (12, int(band * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, band / 50.0),
        color, 1, cv2.LINE_AA,
    )
    return img


def _hstack_with_gap(a: np.ndarray, b: np.ndarray, *, gap: int = 6) -> np.ndarray:
    sep = np.full((a.shape[0], gap, 3), 32, dtype=np.uint8)
    return np.hstack([a, sep, b])


def _vstack_with_gap(a: np.ndarray, b: np.ndarray, *, gap: int = 6) -> np.ndarray:
    sep = np.full((gap, a.shape[1], 3), 32, dtype=np.uint8)
    return np.vstack([a, sep, b])


def _crop(image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = roi
    x0p, y0p = int(x0 * w), int(y0 * h)
    x1p, y1p = int(x1 * w), int(y1 * h)
    return image[y0p:y1p, x0p:x1p].copy()


def _diff_heatmap(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """|after - before| on luminance, COLORMAP_MAGMA for visibility."""
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    diff = cv2.absdiff(after, before)
    mag = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mag = np.clip(mag.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_MAGMA)


def _safe_summary(result: Any) -> dict[str, Any]:
    s = result.summary()

    def coerce(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [coerce(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if is_dataclass(o):
            return coerce(asdict(o))
        return o

    return coerce(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=None,
                   help="Optional source image; default = synthetic interior")
    p.add_argument("--scene", default="synthetic_villa",
                   help="Output sub-directory name under docs/showcase/")
    p.add_argument("--property", default="villa_luxury",
                   help="property_type passed to JobContext")
    p.add_argument("--long-edge", type=int, default=1280,
                   help="target_long_edge for OutputAgent upscale")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    # 1. Load source image
    if args.input is not None:
        if not args.input.exists():
            print(f"input not found: {args.input}", file=sys.stderr)
            return 2
        before = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
        if before is None:
            print(f"could not decode: {args.input}", file=sys.stderr)
            return 2
    else:
        before = _synthetic_interior()

    # 2. Run pipeline
    ctx = JobContext(
        image=before,
        target_long_edge=args.long_edge,
        property_type=args.property,
        seed=7,
    )
    result = Orchestrator(max_workers=args.max_workers).run(ctx)
    after = result.image

    # 3. Output dir
    out_dir = ROOT / "docs" / "showcase" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save raw before/after at matching size for visual comparison
    if before.shape != after.shape:
        before_match = cv2.resize(
            before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA
        )
    else:
        before_match = before
    cv2.imwrite(str(out_dir / "before.jpg"), before_match,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / "after.jpg"), after,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 5. Side-by-side comparison
    compare = _hstack_with_gap(
        _label(before_match, "BEFORE — input gốc"),
        _label(after, "AFTER — pipeline output"),
    )
    cv2.imwrite(str(out_dir / "compare.jpg"), compare,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 6. Diff heatmap
    heatmap = _diff_heatmap(before_match, after)
    diff_panel = _hstack_with_gap(
        _label(after, "AFTER"),
        _label(heatmap, "DIFF |after - before| (magma)"),
    )
    cv2.imwrite(str(out_dir / "diff.jpg"), diff_panel,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 7. Detail crops (window/ceiling/floor/sofa). 3-column panel:
    # BEFORE | AFTER | DIFF heatmap, all at 200% zoom for legibility.
    for name, roi in DETAIL_REGIONS.items():
        crop_b = _crop(before_match, roi)
        crop_a = _crop(after, roi)
        crop_b = cv2.resize(crop_b, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
        crop_a = cv2.resize(crop_a, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
        crop_d = _diff_heatmap(crop_b, crop_a)
        panel = _hstack_with_gap(
            _hstack_with_gap(
                _label(crop_b, f"{name} — BEFORE"),
                _label(crop_a, f"{name} — AFTER"),
            ),
            _label(crop_d, f"{name} — DIFF (200%)"),
        )
        cv2.imwrite(str(out_dir / f"crop_{name}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 8. Scorecard JSON
    scorecard = _safe_summary(result)
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"OK: wrote {len(list(out_dir.glob('*')))} files to {out_dir}")
    print(f"verdict: {scorecard.get('verdict')}, "
          f"score: {scorecard.get('overall_score')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
