"""Folder-mode CLI: inputs/ → outputs/ + comparison.html viewer.

Workflow:
    1. Mày drop ảnh vào --inputs folder (default: ./inputs).
    2. CLI process từng ảnh → --outputs folder (default: ./outputs).
    3. Sinh comparison.html với slider before/after, mở browser xem.

Usage:
    pps-wincei-folder                          # default ./inputs → ./outputs
    pps-wincei-folder --inputs /path/to/raw --outputs /path/to/fixed
    pps-wincei-folder --no-html                # skip comparison viewer
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .diff_jpg import write_index_html, write_side_by_side
from .io_quality import SUPPORTED_EXT, read_image
from .pipeline import process_image
from .viewer import ComparisonItem, build_comparison_item, generate_html

# Force UTF-8 stdout/stderr on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _scan_images(folder: Path, recursive: bool = True) -> list[Path]:
    """Recursively scan folder for supported image files."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pps-wincei-folder",
        description="Batch process folder ảnh BĐS → comparison HTML viewer.",
    )
    parser.add_argument("--version", action="version", version=f"pps-wincei {__version__}")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("./inputs"),
        help="Folder chứa ảnh input (default: ./inputs).",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("./outputs"),
        help="Folder lưu ảnh đã fix (default: ./outputs).",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Folder dump mask/overlay debug (optional).",
    )
    parser.add_argument(
        "--viewer",
        type=Path,
        default=None,
        help="Output HTML viewer path (default: <outputs>/comparison.html).",
    )
    parser.add_argument("--no-html", action="store_true", help="Tắt comparison viewer.")
    parser.add_argument(
        "--diff-jpg",
        action="store_true",
        default=True,
        help="Xuất side-by-side JPG cho từng ảnh (mở bằng image viewer thông thường).",
    )
    parser.add_argument(
        "--no-diff-jpg",
        action="store_false",
        dest="diff_jpg",
        help="Tắt diff JPG output.",
    )
    parser.add_argument(
        "--diff-dir",
        type=Path,
        default=None,
        help="Folder lưu diff JPG (default: <outputs>/diff/).",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Không scan subfolder.")
    parser.add_argument(
        "--thumb-side",
        type=int,
        default=1280,
        help="Kích thước thumbnail trong viewer (px, default 1280). Giảm nếu HTML quá lớn.",
    )
    parser.add_argument(
        "--external-thumb",
        action="store_true",
        help="Lưu thumbnail ra file .thumb/ thay vì embed base64 (giảm HTML size cho >100 ảnh).",
    )
    parser.add_argument(
        "--window", type=float, default=None,
        help="Override window strength (None = AI tự quyết).",
    )
    parser.add_argument(
        "--ceiling", type=float, default=None,
        help="Override ceiling strength (None = AI tự quyết).",
    )
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ process N ảnh đầu (0 = all).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua ảnh đã có output (resume mode).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args.inputs = args.inputs.resolve()
    args.outputs = args.outputs.resolve()

    if not args.inputs.exists() or not args.inputs.is_dir():
        print(f"❌ Inputs folder không tồn tại: {args.inputs}", file=sys.stderr)
        return 2

    args.outputs.mkdir(parents=True, exist_ok=True)

    images = _scan_images(args.inputs, recursive=not args.no_recursive)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"❌ Không tìm thấy ảnh nào trong {args.inputs}", file=sys.stderr)
        return 1

    print(f"📂 Inputs : {args.inputs} ({len(images)} ảnh)")
    print(f"📁 Outputs: {args.outputs}")
    print("─" * 70)

    items: list[ComparisonItem] = []
    n_pass = n_review = n_fail = n_scope = n_no_target = n_error = n_skip = 0
    t_total_start = time.perf_counter()

    thumb_dir = None
    if args.external_thumb and not args.no_html:
        thumb_dir = args.outputs / ".thumb"
        thumb_dir.mkdir(parents=True, exist_ok=True)

    diff_dir = args.diff_dir or (args.outputs / "diff")
    if args.diff_jpg:
        diff_dir.mkdir(parents=True, exist_ok=True)
    diff_records: list[tuple[str, Path]] = []

    for i, src in enumerate(images, 1):
        rel = src.relative_to(args.inputs)
        dst = args.outputs / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and dst.exists():
            print(f"[{i}/{len(images)}] ⏭️  {rel} (đã có, skip)")
            n_skip += 1
            continue

        print(f"[{i}/{len(images)}] 🔄 {rel}", flush=True)
        t0 = time.perf_counter()

        try:
            result = process_image(
                src,
                dst,
                debug_dir=(args.debug_dir / rel.stem) if args.debug_dir else None,
                window_strength=args.window,
                ceiling_strength=args.ceiling,
                context_aware=not args.no_context,
                enable_clip=not args.no_clip,
                self_evaluate=True,
            )
        except Exception as exc:
            print(f"           ❌ ERROR: {exc}", file=sys.stderr)
            n_error += 1
            continue

        verdict = result.evaluation.get("verdict", "?")
        score = result.evaluation.get("overall_score", 0.0)
        scope = result.evaluation.get("scope_delta_e", 0.0)
        dur = time.perf_counter() - t0
        print(
            f"           ✓ {verdict.upper():15s} score={score:.2f}  "
            f"scope ΔE={scope:.2f}  ({dur:.1f}s)"
        )
        if verdict == "pass": n_pass += 1
        elif verdict == "review": n_review += 1
        elif verdict == "scope_violation": n_scope += 1
        elif verdict == "no_target": n_no_target += 1
        else: n_fail += 1

        # Generate diff JPG + html (load images once, reuse)
        before_bgr_for_viewer = None
        after_bgr_for_viewer = None
        if args.diff_jpg or not args.no_html:
            try:
                before_bgr_for_viewer, _ = read_image(src)
                after_bgr_for_viewer, _ = read_image(dst)
            except Exception as exc:
                print(f"           ⚠️  re-read for viewer failed: {exc}", file=sys.stderr)

        if args.diff_jpg and before_bgr_for_viewer is not None and after_bgr_for_viewer is not None:
            try:
                diff_path = diff_dir / f"{src.stem}_diff.jpg"
                write_side_by_side(
                    before_bgr_for_viewer,
                    after_bgr_for_viewer,
                    diff_path,
                    name=src.name,
                    max_width=3200,
                    include_diff=True,
                )
                label = f"{src.name}  [{verdict.upper()}]  score={score:.2f}  scope ΔE={scope:.2f}"
                diff_records.append((label, diff_path))
                print(f"           💾 diff JPG: {diff_path.relative_to(args.outputs)}")
            except Exception as exc:
                print(f"           ⚠️  diff JPG failed: {exc}", file=sys.stderr)

        if not args.no_html:
            try:
                before_bgr = before_bgr_for_viewer
                after_bgr = after_bgr_for_viewer
                if before_bgr is None or after_bgr is None:
                    raise RuntimeError("re-read failed earlier")
                item = build_comparison_item(
                    name=str(rel),
                    before_bgr=before_bgr,
                    after_bgr=after_bgr,
                    process_result=result,
                    embed_thumbnails=not args.external_thumb,
                    thumb_side=args.thumb_side,
                )
                if args.external_thumb and thumb_dir is not None:
                    # Write thumbs as files and link relatively
                    from .io_quality import make_thumbnail
                    import cv2
                    bt = make_thumbnail(before_bgr, max_side=args.thumb_side)
                    at = make_thumbnail(after_bgr, max_side=args.thumb_side)
                    bn = thumb_dir / f"{i:04d}_before.jpg"
                    an = thumb_dir / f"{i:04d}_after.jpg"
                    cv2.imwrite(str(bn), bt, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    cv2.imwrite(str(an), at, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    item.before_url = f".thumb/{bn.name}"
                    item.after_url = f".thumb/{an.name}"
                items.append(item)
            except Exception as exc:
                print(f"           ⚠️  Viewer item build failed: {exc}", file=sys.stderr)

    total_dur = time.perf_counter() - t_total_start
    print("─" * 70)
    print(
        f"📊 SUMMARY  total={len(images)}  "
        f"✅pass={n_pass}  ⚠️review={n_review}  ❌fail={n_fail}  "
        f"🚫scope={n_scope}  ⏭️notarget={n_no_target}  "
        f"❗error={n_error}  ⏩skip={n_skip}  "
        f"⏱️ {total_dur:.1f}s ({total_dur/max(1,len(images)-n_skip):.1f}s/ảnh)"
    )

    if not args.no_html and items:
        viewer_path = args.viewer or (args.outputs / "comparison.html")
        subtitle = (
            f"{args.inputs.name} → {args.outputs.name} · "
            f"{len(items)} ảnh · {total_dur:.0f}s"
        )
        generate_html(items, viewer_path, title="Window + Ceiling AI", subtitle=subtitle)
        print(f"🌐 Slider viewer  : {viewer_path}")
        print(f"                    Open: file:///{viewer_path.as_posix()}")

    if args.diff_jpg and diff_records:
        diff_index = diff_dir / "index.html"
        write_index_html(diff_records, diff_index)
        print(f"🖼  Diff JPG folder: {diff_dir}")
        print(f"   Index HTML     : file:///{diff_index.as_posix()}")
        print(f"   ↑ Mỗi JPG mở bằng app xem ảnh thường — BEFORE | AFTER + heatmap.")

    return 0 if n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
