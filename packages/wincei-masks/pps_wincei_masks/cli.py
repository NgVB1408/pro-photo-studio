"""CLI cho pps-wincei-masks.

Usage:
    pps-wincei-masks foto.jpg                         # masks → ./outputs/<stem>/
    pps-wincei-masks foto.jpg --out-dir ./psmasks
    pps-wincei-masks --inputs ./inputs --outputs ./outputs   # batch folder
    pps-wincei-masks foto.jpg --no-refine             # skip PyMatting (faster)
    pps-wincei-masks foto.jpg --no-molding            # chỉ semantic
    pps-wincei-masks foto.jpg --psd                   # xuất PSD multi-layer
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .pipeline import extract_masks
from .semantic import SemanticSegmenter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

log = logging.getLogger("pps_wincei_masks.cli")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _scan(folder: Path, recursive: bool) -> list[Path]:
    pat = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pat) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pps-wincei-masks",
        description="Smart room segmentation + phào chỉ → Photoshop-ready masks.",
    )
    parser.add_argument("--version", action="version", version=f"pps-wincei-masks {__version__}")

    parser.add_argument("input", type=Path, nargs="*",
                        help="1+ ảnh input. Bỏ trống nếu dùng --inputs folder mode.")
    parser.add_argument("--inputs", type=Path, default=None,
                        help="Folder batch mode (override input args).")
    parser.add_argument("--outputs", "--out-dir", dest="outputs", type=Path,
                        default=Path("./outputs/masks"),
                        help="Output folder. Tạo subfolder <stem>/ cho mỗi ảnh.")
    parser.add_argument("--no-recursive", action="store_true")

    parser.add_argument("--no-refine", action="store_true",
                        help="Tắt PyMatting (nhanh hơn, biên cứng hơn).")
    parser.add_argument("--no-molding", action="store_true",
                        help="Tắt phào chỉ detection.")
    parser.add_argument("--include-lights", action="store_true",
                        help="Thêm mask đèn (lamp + light fixture).")
    parser.add_argument("--matting-max-side", type=int, default=1600,
                        help="PyMatting downscale cap (default 1600, lớn hơn tốn RAM).")
    parser.add_argument("--precision", action="store_true",
                        help="PRECISION MODE: B5 + multi-scale TTA + CRF + tile matting full-res + edge polish + AI gate.")
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="Tile size cho precision matting (default 1024).")
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--polish-strength", type=float, default=1.0,
                        help="Bilateral edge polish strength 0..1.")
    parser.add_argument("--no-tta", action="store_true",
                        help="Tắt multi-scale TTA trong precision mode.")
    parser.add_argument("--no-crf", action="store_true",
                        help="Tắt DenseCRF refinement trong precision mode.")
    parser.add_argument("--engine", choices=["semantic", "vlm-sam2"], default="semantic",
                        help="Engine: 'semantic' (SegFormer ADE20K) hoặc 'vlm-sam2' (Ollama VLM + SAM 2).")
    parser.add_argument("--preprocess-clahe", action="store_true",
                        help="CLAHE histogram equalization cho ảnh khó (low contrast).")
    parser.add_argument("--preprocess-undistort", action="store_true",
                        help="Undistort fisheye/ultra-wide trước khi segment.")
    parser.add_argument("--no-overlap-resolve", action="store_true",
                        help="Tắt Sobel-based overlap leakage resolver.")
    parser.add_argument("--vlm-model", type=str, default="qwen2.5vl:7b",
                        help="Ollama VLM model tag (default qwen2.5vl:7b).")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="SAM 2 checkpoint path (.pt file).")
    parser.add_argument("--no-self-eval", action="store_true",
                        help="Tắt AI supervisor.")
    parser.add_argument("--retry-on-fail", action="store_true",
                        help="Re-run với matting_max_side cao hơn nếu eval=fail.")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-tiff", action="store_true")
    parser.add_argument("--psd", action="store_true",
                        help="Thử xuất PSD multi-layer (cần pytoshop).")

    parser.add_argument("--model", type=str, default=None,
                        help="HF model id (default auto-pick theo VRAM).")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu' (default auto).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve input list
    files: list[Path] = []
    if args.inputs is not None:
        args.inputs = args.inputs.resolve()
        if not args.inputs.is_dir():
            print(f"❌ Inputs không tồn tại: {args.inputs}", file=sys.stderr)
            return 2
        files = _scan(args.inputs, not args.no_recursive)
    else:
        files = [p.resolve() for p in args.input if p.exists()]

    if not files:
        print("❌ Không có ảnh input. Dùng `pps-wincei-masks foto.jpg` hoặc `--inputs ./folder`",
              file=sys.stderr)
        return 1

    if args.limit > 0:
        files = files[: args.limit]

    args.outputs = args.outputs.resolve()
    args.outputs.mkdir(parents=True, exist_ok=True)

    print(f"📂 Inputs : {len(files)} ảnh")
    print(f"📁 Outputs: {args.outputs}")
    print("─" * 70)

    # Precision mode overrides
    if args.precision:
        if args.model is None:
            args.model = "nvidia/segformer-b5-finetuned-ade-640-640"
        args.matting_max_side = max(args.matting_max_side, 2400)
        print("🎯 PRECISION MODE")
        print(f"   model       : {args.model}")
        print(f"   TTA         : {'ON (3 scales + flip)' if not args.no_tta else 'OFF'}")
        print(f"   CRF refine  : {'ON' if not args.no_crf else 'OFF'}")
        print(f"   Tile matting: {args.tile_size}x{args.tile_size} overlap={args.tile_overlap}")
        print(f"   Edge polish : strength={args.polish_strength}")
        print(f"   AI eval gate: ON (retry if fail)")

    # Engine VLM+SAM2 nhánh riêng
    if args.engine == "vlm-sam2":
        from .vlm_client import OllamaVLM, check_ollama_available
        from .sam_engine import SAMEngine
        from .vlm_sam_pipeline import extract_masks_vlm_sam
        from .exporters import export_all_masks
        from .evaluator import evaluate_masks
        import cv2, json

        ok, models = check_ollama_available()
        if not ok:
            print("❌ Ollama không chạy. Cài: https://ollama.com/download và `ollama serve`", file=sys.stderr)
            return 3
        if args.vlm_model not in models:
            print(f"⚠️  Model '{args.vlm_model}' chưa pull. Chạy: ollama pull {args.vlm_model}", file=sys.stderr)
            print(f"   Models đã có: {', '.join(models)}", file=sys.stderr)

        vlm = OllamaVLM(model=args.vlm_model)
        sam = SAMEngine(checkpoint=args.sam_checkpoint)

        n_ok = n_err = 0
        for i, src in enumerate(files, 1):
            print(f"[{i}/{len(files)}] 🔄 {src.name} (vlm-sam2)", flush=True)
            try:
                r = extract_masks_vlm_sam(src, vlm=vlm, sam=sam,
                                          sam_checkpoint=args.sam_checkpoint)
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                stem = src.stem
                ex = export_all_masks(
                    img, r.masks, out_root=args.outputs, stem=stem,
                    write_overlay=not args.no_overlay,
                    write_tiff=not args.no_tiff,
                    write_psd=args.psd,
                )
                ev = evaluate_masks(img, r.masks)
                # Save QC + VLM info
                qc_path = ex.out_dir / f"{stem}_vlm_qc.json"
                qc_path.write_text(json.dumps({
                    "engine": "vlm-sam2",
                    "vlm_model": vlm.model,
                    "vlm_response_ms": r.vlm_response.elapsed_ms if r.vlm_response else None,
                    "vlm_points": r.vlm_response.parsed_points if r.vlm_response else {},
                    "sam_scores": r.sam_scores,
                    "evaluation": ev.to_dict(),
                    "timings": r.timings,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"           ✓ {len(r.masks)} masks  [{ev.verdict.upper():7s}] score={ev.overall_score:.2f}  ({r.timings.get('total',0):.1f}s)  → {ex.out_dir.name}/")
                n_ok += 1
            except Exception as exc:
                print(f"           ❌ ERROR: {exc}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                n_err += 1

        print("─" * 70)
        print(f"📊 vlm-sam2  ✅{n_ok}  ❌{n_err}")
        return 0 if n_err == 0 else 1

    # Load segmenter once (default semantic engine)
    segmenter = SemanticSegmenter(model_name=args.model, device=args.device)

    n_ok = n_err = n_skip = 0
    t_total = time.perf_counter()

    for i, src in enumerate(files, 1):
        stem = src.stem
        target_dir = args.outputs / stem
        if args.skip_existing and target_dir.exists() and any(target_dir.iterdir()):
            print(f"[{i}/{len(files)}] ⏭️  {src.name} (đã có)")
            n_skip += 1
            continue

        print(f"[{i}/{len(files)}] 🔄 {src.name}", flush=True)
        try:
            result = extract_masks(
                src,
                args.outputs,
                segmenter=segmenter,
                refine_edges=not args.no_refine,
                detect_molding=not args.no_molding,
                include_lights=args.include_lights,
                matting_max_side=args.matting_max_side,
                write_overlay=not args.no_overlay,
                write_tiff=not args.no_tiff,
                write_psd=args.psd,
                self_evaluate=not args.no_self_eval,
                precision_mode=args.precision,
                tta_scales=(0.75, 1.0, 1.5) if not args.no_tta else (1.0,),
                use_crf=not args.no_crf,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                polish_strength=args.polish_strength,
                preprocess_clahe=args.preprocess_clahe,
                preprocess_undistort=args.preprocess_undistort,
                resolve_overlap=not args.no_overlap_resolve,
            )
            # Retry on fail (precision mode)
            if (args.retry_on_fail or args.precision) and result.evaluation \
               and result.evaluation.verdict == "fail":
                print(f"           ⚠️  verdict=fail, retry với matting_max_side x1.5")
                result = extract_masks(
                    src,
                    args.outputs,
                    segmenter=segmenter,
                    refine_edges=True,
                    detect_molding=not args.no_molding,
                    include_lights=args.include_lights,
                    matting_max_side=int(args.matting_max_side * 1.5),
                    write_overlay=not args.no_overlay,
                    write_tiff=not args.no_tiff,
                    write_psd=args.psd,
                    self_evaluate=True,
                )
            n_ok += 1
            verdict = result.evaluation.verdict if result.evaluation else "?"
            score = result.evaluation.overall_score if result.evaluation else 0.0
            print(
                f"           ✓ {len(result.masks)} masks  "
                f"[{verdict.upper():7s}] score={score:.2f}  "
                f"({result.timings['total']:.1f}s)  → {result.out_dir.name}/"
            )
            if args.verbose:
                print(result.report())
        except Exception as exc:
            print(f"           ❌ ERROR: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            n_err += 1

    total = time.perf_counter() - t_total
    print("─" * 70)
    n_proc = max(1, n_ok)
    print(
        f"📊 SUMMARY  total={len(files)}  ✅ok={n_ok}  ❌err={n_err}  "
        f"⏭️skip={n_skip}  ⏱️ {total:.1f}s ({total/n_proc:.1f}s/ảnh)"
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
