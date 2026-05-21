"""CLI cho pps-wincei-hdr — batch HDR fusion từ folder Sony AEB.

Workflow:
    1. Scan --inputs (default ./inputs).
    2. Detect brackets bằng EXIF (DateTimeOriginal + ExposureBiasValue).
    3. Mỗi group: AlignMTB → MergeMertens → Output JPG q98.
    4. Output filename = reference shot (EV≈0) → giữ tên gốc khách.
    5. (Optional) pipe vào pps-wincei window+ceiling fix.

Usage:
    pps-wincei-hdr                            # ./inputs → ./outputs
    pps-wincei-hdr --inputs /raw --outputs /fused
    pps-wincei-hdr --no-align                 # tripod input
    pps-wincei-hdr --chain-wincei             # auto run wincei sau fusion
    pps-wincei-hdr --keep-singletons          # copy ảnh không thuộc bracket
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .bracket_detect import detect_brackets
from .fusion import align_brackets, fuse_mertens
from .io_meta import write_jpg_with_meta

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

log = logging.getLogger("pps_wincei_hdr.cli")

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _scan(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pps-wincei-hdr",
        description="HDR bracket fusion (Mertens) — Sony AEB → 1 LDR JPG.",
    )
    parser.add_argument("--version", action="version", version=f"pps-wincei-hdr {__version__}")
    parser.add_argument("--inputs", type=Path, default=Path("./inputs"))
    parser.add_argument("--outputs", type=Path, default=Path("./outputs"))
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--no-align", action="store_true",
                        help="Tắt AlignMTB (chỉ khi tripod chắc chắn).")
    parser.add_argument("--time-tolerance", type=float, default=3.0,
                        help="Giây giữa các shot trong 1 bracket (default 3).")
    parser.add_argument("--min-group", type=int, default=2)
    parser.add_argument("--max-group", type=int, default=7)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--saturation-weight", type=float, default=1.0)
    parser.add_argument("--exposure-weight", type=float, default=1.0,
                        help="Giảm xuống (0.3-0.5) để pull outdoor highlight mạnh hơn.")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--quality", type=int, default=98)
    parser.add_argument("--keep-singletons", action="store_true",
                        help="Copy ảnh single-shot vào outputs (không bracket).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in groups detect, không fuse.")
    parser.add_argument("--chain-wincei", action="store_true",
                        help="Sau khi fuse → chạy pps-wincei window+ceiling fix.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Chỉ process N groups đầu (0 = all).")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args.inputs = args.inputs.resolve()
    args.outputs = args.outputs.resolve()

    if not args.inputs.is_dir():
        print(f"❌ Inputs không tồn tại: {args.inputs}", file=sys.stderr)
        return 2

    args.outputs.mkdir(parents=True, exist_ok=True)

    files = _scan(args.inputs, not args.no_recursive)
    if not files:
        print(f"❌ Không tìm thấy ảnh trong {args.inputs}", file=sys.stderr)
        return 1

    print(f"📂 Inputs : {args.inputs} ({len(files)} files)")
    print(f"📁 Outputs: {args.outputs}")
    print("─" * 70)
    print("🔍 Detecting brackets via EXIF...")

    groups, singletons = detect_brackets(
        files,
        time_tolerance_s=args.time_tolerance,
        min_group=args.min_group,
        max_group=args.max_group,
    )

    print(f"   📦 Brackets : {len(groups)}")
    print(f"   📷 Singletons: {len(singletons)}")
    if not groups:
        print("⚠️  Không detect được bracket nào. Check EXIF DateTimeOriginal + ExposureBiasValue.")
        if args.keep_singletons and singletons:
            for p in singletons:
                shutil.copy2(p, args.outputs / p.name)
            print(f"   ✓ Copied {len(singletons)} singletons.")
        return 0

    if args.limit > 0:
        groups = groups[: args.limit]

    if args.dry_run:
        for i, g in enumerate(groups, 1):
            ref = g.reference
            print(f"[{i:3d}] → {ref.path.name}  EV{g.ev_range[0]:+.1f}..{g.ev_range[1]:+.1f}  n={len(g.shots)}")
            for s in g.shots:
                print(f"        {s.path.name}  EV={s.ev:+.1f}  T={s.shutter}")
        return 0

    n_ok = n_err = 0
    t0 = time.perf_counter()
    chain_targets: list[Path] = []

    for i, g in enumerate(groups, 1):
        ref = g.reference
        ref_rel = ref.path.relative_to(args.inputs) if args.inputs in ref.path.parents else ref.path.name
        dst = args.outputs / ref.path.name
        t_g0 = time.perf_counter()

        print(f"[{i}/{len(groups)}] 🔄 {ref_rel}  ({len(g.shots)} shots, EV{g.ev_range[0]:+.1f}..{g.ev_range[1]:+.1f})", flush=True)

        try:
            import cv2
            images = []
            for s in g.shots:
                img = cv2.imread(str(s.path), cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError(f"không đọc được {s.path}")
                images.append(img)

            images = align_brackets(images, enabled=not args.no_align)
            fused = fuse_mertens(
                images,
                contrast_weight=args.contrast_weight,
                saturation_weight=args.saturation_weight,
                exposure_weight=args.exposure_weight,
                gamma=args.gamma,
            )

            comment = (
                f"pps-wincei-hdr v{__version__} | Mertens fusion | "
                f"n={len(g.shots)} | EV[{g.ev_range[0]:+.1f},{g.ev_range[1]:+.1f}] | "
                f"ref={ref.path.name}"
            )
            write_jpg_with_meta(
                fused, dst,
                reference_path=ref.path,
                quality=args.quality,
                add_user_comment=comment,
            )
            n_ok += 1
            chain_targets.append(dst)
            dur = time.perf_counter() - t_g0
            print(f"           ✓ fused → {dst.name}  ({dur:.1f}s)")
        except Exception as exc:
            print(f"           ❌ ERROR: {exc}", file=sys.stderr)
            n_err += 1

    if args.keep_singletons and singletons:
        for p in singletons:
            target = args.outputs / p.name
            if not target.exists():
                shutil.copy2(p, target)
        print(f"   ⏩ Copied {len(singletons)} singletons (no bracket).")

    total = time.perf_counter() - t0
    print("─" * 70)
    print(
        f"📊 SUMMARY  groups={len(groups)}  ✅ok={n_ok}  ❌err={n_err}  "
        f"⏱️ {total:.1f}s ({total/max(1,n_ok):.1f}s/group)"
    )

    if args.chain_wincei and chain_targets:
        print("─" * 70)
        print("🪟 Chain → pps-wincei window+ceiling fix…")
        try:
            from pps_wincei.folder_cli import main as wincei_main
            # Re-run wincei in same outputs dir, sub-folder _wincei
            wincei_out = args.outputs / "_wincei"
            wincei_out.mkdir(exist_ok=True)
            wincei_main([
                "--inputs", str(args.outputs),
                "--outputs", str(wincei_out),
                "--no-recursive",
            ])
        except ImportError:
            print("⚠️  pps-wincei not installed. pip install pps-wincei", file=sys.stderr)

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
