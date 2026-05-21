"""CLI: pps-sam-demo — chạy SAM AutomaticMaskGenerator + visualize panoptic style.

Theo notebook chính thức:
   facebookresearch/segment-anything → notebooks/automatic_mask_generator_example.ipynb

Usage:
    pps-sam-demo foto.jpg
    pps-sam-demo foto.jpg --out colored.jpg --points-per-side 64 --border 2
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pps-sam-demo",
        description="SAM Automatic Mask Generator — panoptic visualization (official demo style)",
    )
    p.add_argument("input", type=Path)
    p.add_argument("--out", "-o", type=Path, default=Path("./sam_colored.jpg"),
                   help="Output blended JPG (default ./sam_colored.jpg)")
    p.add_argument("--out-rgba", type=Path, default=None,
                   help="Optional RGBA PNG (chỉ colored masks, transparent background)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="SAM 1 checkpoint .pth (auto-detect nếu None)")
    p.add_argument("--points-per-side", type=int, default=32,
                   help="Grid density (default 32 official, 64 dày hơn).")
    p.add_argument("--pred-iou-thresh", type=float, default=0.88)
    p.add_argument("--stability-score", type=float, default=0.95)
    p.add_argument("--min-mask-area", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="Mask color opacity (default 0.35 official).")
    p.add_argument("--border", type=int, default=0,
                   help="Border thickness (px) — 0 = no border.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-report", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        print(f"❌ Không tìm thấy: {args.input}", file=sys.stderr)
        return 1

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Cannot decode: {args.input}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    print(f"📷 Input: {args.input.name}  {w}×{h}")

    from .sam_demo import sam_demo_pipeline

    print(f"🔍 SAM AutomaticMaskGenerator (points_per_side={args.points_per_side})...")
    t0 = time.perf_counter()
    try:
        result = sam_demo_pipeline(
            img,
            checkpoint=args.checkpoint,
            points_per_side=args.points_per_side,
            alpha=args.alpha,
            seed=args.seed,
            border_thickness=args.border,
        )
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - t0
    print(f"   ✓ Generated {result.n_masks} masks  model={result.model_type}  ({elapsed:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), result.overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"💾 Overlay JPG : {args.out}")

    if args.out_rgba:
        args.out_rgba.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_rgba), result.overlay_rgba,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"💾 RGBA PNG    : {args.out_rgba}")

    if args.save_report:
        report = {
            "source": str(args.input),
            "model_type": result.model_type,
            "n_masks": result.n_masks,
            "points_per_side": args.points_per_side,
            "alpha": args.alpha,
            "elapsed_sec": round(elapsed, 2),
            "output_overlay": str(args.out),
            "output_rgba": str(args.out_rgba) if args.out_rgba else None,
        }
        args.save_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"📋 Report      : {args.save_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
