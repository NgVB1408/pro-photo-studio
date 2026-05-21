"""CLI cho pps-wincei.

Usage:
    pps-wincei input.jpg                          # output → input.wincei.jpg
    pps-wincei input.jpg --out result.jpg
    pps-wincei input.jpg --debug --debug-dir ./out
    pps-wincei batch *.jpg --out-dir ./fixed
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so emoji + Vietnamese in reports render.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import __version__
from .pipeline import process_image


def _auto_output(src: Path, out_dir: Path | None = None) -> Path:
    """Default output keeps SAME filename. Defaults to ./outputs/<filename>.

    User explicit --out or --out-dir overrides. KHÔNG add suffix .wincei nữa —
    real-estate workflow needs filename preservation cho khách reference.
    """
    if out_dir is None:
        out_dir = Path("./outputs")
    return out_dir / src.name


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _process_one(
    src: Path,
    dst: Path,
    debug_dir: Path | None,
    window_strength: float | None,
    ceiling_strength: float | None,
    include_lamps: bool | None,
    expand_window_with_sky: bool | None,
    context_aware: bool,
    enable_clip: bool,
    self_evaluate: bool,
) -> int:
    try:
        result = process_image(
            src,
            dst,
            debug_dir=debug_dir,
            window_strength=window_strength,
            ceiling_strength=ceiling_strength,
            include_lamps=include_lamps,
            expand_window_with_sky=expand_window_with_sky,
            context_aware=context_aware,
            enable_clip=enable_clip,
            self_evaluate=self_evaluate,
        )
    except Exception as exc:
        print(f"❌ {src.name}: {exc}", file=sys.stderr)
        return 1

    print(result.report())
    print("─" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pps-wincei",
        description="AI Window + Ceiling Fixer — chỉnh cửa sổ blown + trần ám màu.",
    )
    parser.add_argument("--version", action="version", version=f"pps-wincei {__version__}")
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Đường dẫn ảnh input (.jpg/.png). Có thể nhiều ảnh cho batch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file (chỉ khi 1 input). Mặc định: ./outputs/<tên-file-input>.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory — file giữ TÊN GỐC. Mặc định: ./outputs/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump mask, overlay, before/after vào debug dir.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Debug output dir (default: <out>.debug/).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        metavar="0..1.5",
        help="Override cường độ window (None = AI tự quyết theo context).",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=None,
        metavar="0..1",
        help="Override cường độ ceiling (None = AI tự quyết).",
    )
    parser.add_argument(
        "--include-lamps",
        action="store_true",
        default=None,
        help="Force include lamps vào mask trần (AI tự quyết nếu không set).",
    )
    parser.add_argument(
        "--no-sky-as-window",
        action="store_true",
        help="Force KHÔNG gộp sky vào mask cửa sổ.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Tắt context-aware mode (dùng base params v0.2).",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Tắt CLIP scene classifier (heuristic-only context).",
    )
    parser.add_argument(
        "--no-self-eval",
        action="store_true",
        help="Tắt AI tự chấm điểm output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose log.")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.input:
        parser.error("Cần ít nhất 1 input file.")

    exit_code = 0
    multi = len(args.input) > 1

    for src in args.input:
        if not src.exists():
            print(f"❌ Không tìm thấy: {src}", file=sys.stderr)
            exit_code = 1
            continue

        # Resolve output — keep input filename, NEVER auto-rename.
        if multi:
            out_dir = args.out_dir if args.out_dir is not None else Path("./outputs")
            dst = out_dir / src.name
        else:
            if args.out is not None:
                dst = args.out
            elif args.out_dir is not None:
                dst = args.out_dir / src.name
            else:
                dst = _auto_output(src)
        if dst.resolve() == src.resolve():
            print(
                f"⛔ Output ghi đè input: {src.name}. Dùng --out hoặc --out-dir để chỉ định nơi khác.",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        # Resolve debug dir
        debug_dir = None
        if args.debug:
            if args.debug_dir is not None:
                debug_dir = args.debug_dir / src.stem if multi else args.debug_dir
            else:
                debug_dir = dst.with_suffix("").with_name(f"{dst.stem}.debug")

        code = _process_one(
            src,
            dst,
            debug_dir=debug_dir,
            window_strength=args.window,
            ceiling_strength=args.ceiling,
            include_lamps=args.include_lamps,
            expand_window_with_sky=False if args.no_sky_as_window else None,
            context_aware=not args.no_context,
            enable_clip=not args.no_clip,
            self_evaluate=not args.no_self_eval,
        )
        if code != 0:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
