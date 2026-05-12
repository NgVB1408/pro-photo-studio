"""CLI entry point: pps-agents process input.jpg out.jpg [options]."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .orchestrator import Orchestrator
from .types import JobContext


def cmd_process(args: argparse.Namespace) -> int:
    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: cannot read {args.input}", file=sys.stderr)
        return 1

    ctx = JobContext(
        image=img,
        image_path=str(args.input),
        seed=args.seed,
        target_long_edge=args.long_edge,
        target_dpi=args.dpi,
        property_type=args.property,
    )

    orch = Orchestrator(max_workers=args.workers)
    result = orch.run(ctx)
    summary = result.summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    verdict = result.director.verdict if result.director else "no_review"
    if not args.yes:
        try:
            ans = input(
                f"\nDirector verdict: {verdict}.  Save output to "
                f"{args.output}? [y/N]: "
            ).strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted by user", file=sys.stderr)
            return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output), result.image)
    if not ok:
        print(f"error: failed to write {args.output}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}  ({result.image.shape[1]}x{result.image.shape[0]})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote report {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pps-agents",
        description="Multi-agent SOP pipeline for Pro Photo Studio.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("process", help="Run the pipeline on a single image")
    pp.add_argument("input", type=Path)
    pp.add_argument("output", type=Path)
    pp.add_argument(
        "--property",
        choices=[
            "villa_luxury",
            "apartment_modern",
            "studio_minimal",
            "commercial_showroom",
            "twilight_cabin",
            "generic",
        ],
        default="generic",
    )
    pp.add_argument("--long-edge", type=int, default=7680, dest="long_edge")
    pp.add_argument("--dpi", type=int, default=300)
    pp.add_argument("--workers", type=int, default=5)
    pp.add_argument("--seed", type=int, default=None)
    pp.add_argument("--report", type=Path, default=None,
                    help="Write JSON QC report alongside output")
    pp.add_argument("-y", "--yes", action="store_true",
                    help="Skip user confirmation prompt")
    pp.set_defaults(func=cmd_process)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
