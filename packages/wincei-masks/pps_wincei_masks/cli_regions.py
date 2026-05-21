"""CLI: pps-wincei-regions — phân tích ảnh BĐS → JSON bounding boxes.

Usage:
    pps-wincei-regions foto.jpg                        # in JSON ra stdout
    pps-wincei-regions foto.jpg --out result.json
    pps-wincei-regions foto.jpg --pretty               # pretty print
    pps-wincei-regions foto.jpg --no-walls --no-floor  # chỉ ceiling + windows
    pps-wincei-regions ./folder/                       # batch — out: <stem>_regions.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .regions_json import analyze_file_to_json
from .semantic import SemanticSegmenter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pps-wincei-regions",
        description="Real Estate Vision Analyzer — JSON bounding box [0..1000] cho ceiling + windows + walls + floor + doors.",
    )
    parser.add_argument("--version", action="version", version=f"pps-wincei-masks {__version__}")
    parser.add_argument("input", type=Path, nargs="+", help="Ảnh hoặc folder.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON file. Nếu input là folder hoặc nhiều ảnh, dùng --out-dir.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Folder lưu <stem>_regions.json (cho batch mode).")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON indented.")
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--no-doors", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve input list
    files: list[Path] = []
    for arg in args.input:
        p = arg.resolve()
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXT)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"⚠️  Bỏ qua (không tồn tại): {p}", file=sys.stderr)

    if not files:
        print("❌ Không có ảnh input", file=sys.stderr)
        return 1

    segmenter = SemanticSegmenter(model_name=args.model, device=args.device)

    if len(files) > 1 and args.out_dir is None:
        args.out_dir = Path("./outputs/regions").resolve()
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    indent = 2 if args.pretty else None
    n_ok = n_err = 0

    for i, src in enumerate(files, 1):
        t0 = time.perf_counter()
        if args.verbose:
            print(f"[{i}/{len(files)}] {src.name}", file=sys.stderr)
        try:
            result = analyze_file_to_json(
                src,
                segmenter=segmenter,
                include_walls=not args.no_walls,
                include_floor=not args.no_floor,
                include_doors=not args.no_doors,
            )
            result["source_file"] = src.name
            result["processing_ms"] = round((time.perf_counter() - t0) * 1000)

            txt = json.dumps(result, indent=indent, ensure_ascii=False)

            if args.out_dir:
                out_path = args.out_dir / f"{src.stem}_regions.json"
                out_path.write_text(txt, encoding="utf-8")
                if args.verbose:
                    print(f"   → {out_path}", file=sys.stderr)
            elif args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(txt, encoding="utf-8")
            else:
                # stdout
                print(txt)
            n_ok += 1
        except Exception as exc:
            print(f"❌ {src.name}: {exc}", file=sys.stderr)
            n_err += 1

    if len(files) > 1:
        print(f"\n📊 {n_ok} ok, {n_err} fail (out: {args.out_dir})", file=sys.stderr)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
