#!/usr/bin/env python
"""Standalone CLI: ảnh BĐS → ceiling_full_recovery.png (RGBA transparent).

Workflow:
    python scripts/full_recovery_ceiling.py DSC01527.jpg [--out ceiling_full_recovery.png]

Pipeline:
    1. Preprocess (CLAHE auto + undistort optional)
    2. Ollama bds-brain CoT 2-step
    3. SAM 2 high-grid predict_from_points
    4. Sobel directional overlap resolver
    5. Export RGBA PNG with transparent non-ceiling areas
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full Recovery Ceiling — VLM (bds-brain) + SAM 2 high-grid → RGBA PNG"
    )
    parser.add_argument("input", type=Path, help="Ảnh BĐS input (JPG/PNG)")
    parser.add_argument("--out", "-o", type=Path, default=Path("./ceiling_full_recovery.png"),
                        help="Output RGBA PNG (default: ./ceiling_full_recovery.png)")
    parser.add_argument("--vlm-model", default="bds-brain",
                        help="Ollama model (default: bds-brain)")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="SAM 2 checkpoint .pt (default: ~/.cache/sam2/sam2_hiera_tiny.pt)")
    parser.add_argument("--target", choices=["ceiling", "wall", "floor", "windows", "doors"],
                        default="ceiling", help="Class cần recovery")
    parser.add_argument("--no-clahe", action="store_true", help="Tắt CLAHE preprocess")
    parser.add_argument("--undistort", action="store_true", help="Bật undistort cho fisheye")
    parser.add_argument("--no-sobel", action="store_true", help="Tắt Sobel overlap resolver")
    parser.add_argument("--save-report", type=Path, default=None,
                        help="Lưu JSON report với reasoning + metrics")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        print(f"❌ Không tìm thấy: {args.input}", file=sys.stderr)
        return 1

    img_orig = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img_orig is None:
        print(f"❌ Không decode được: {args.input}", file=sys.stderr)
        return 1
    h, w = img_orig.shape[:2]
    print(f"📷 Input  : {args.input.name}  {w}×{h}")

    # Phase 0: Preprocess
    from pps_wincei_masks.preprocess import preprocess_for_vlm_sam
    img, prep = preprocess_for_vlm_sam(
        img_orig,
        auto_clahe=not args.no_clahe, auto_undistort=args.undistort,
        force_clahe=not args.no_clahe, force_undistort=args.undistort,
    )
    print(f"🛠  Preprocess: CLAHE={prep.clahe_applied}, undistort={prep.undistort_applied}")

    # Phase 1: VLM CoT
    from pps_wincei_masks.vlm_client import OllamaVLM, check_ollama_available
    ok, available = check_ollama_available()
    if not ok:
        print("❌ Ollama không chạy. Khởi động: ollama serve", file=sys.stderr)
        return 2
    if not any(args.vlm_model in m for m in available):
        print(f"❌ Model '{args.vlm_model}' chưa pull. Run: ollama pull {args.vlm_model}", file=sys.stderr)
        return 3

    vlm = OllamaVLM(
        model=args.vlm_model,
        endpoint="http://localhost:11434/api/chat",
        use_chat_api=True,
    )
    print(f"🧠 VLM CoT 2-step ({args.vlm_model})...")
    cot = vlm.query_chain_of_thought(img, max_side=1280, target_class=args.target)
    pts_raw = cot.parsed_points.get(args.target) or cot.parsed_points.get("ceiling") or []
    if not pts_raw or not isinstance(pts_raw[0], (list, tuple)):
        print(f"❌ VLM không trả points hợp lệ. Reasoning: {cot.reasoning[:200]}", file=sys.stderr)
        return 4

    pts = [(int(p[0]), int(p[1])) for p in pts_raw]
    print(f"   ✓ {len(pts)} điểm trên {args.target}: {pts[:5]}{'...' if len(pts) > 5 else ''}")
    if args.verbose:
        print(f"   Reasoning preview: {cot.reasoning[:300]}")

    # Phase 2: SAM 2 high-grid
    from pps_wincei_masks.sam_engine import SAMEngine
    print(f"🎯 SAM 2 high-grid predict ({len(pts)} points → multi-point prompt)...")
    sam = SAMEngine(checkpoint=args.sam_checkpoint, high_res=True)
    sam.set_image(img)
    result = sam.predict_from_points(img, pts, multimask=True)
    mask = result.mask
    cov_before = (mask > 128).mean() * 100
    print(f"   ✓ Mask cov={cov_before:.2f}%  score={result.score:.3f}")

    # Phase 3: Sobel resolver
    if not args.no_sobel and args.target == "ceiling":
        from pps_wincei_masks.overlap_resolver import resolve_ceiling_wall_overlap
        print(f"🧭 Sobel directional overlap resolver...")
        # Co-segment wall
        wall_pts = [(w // 8, h // 2), (w * 7 // 8, h // 2), (w // 2, int(h * 0.6))]
        wall_result = sam.predict_from_points(img, wall_pts, multimask=True)
        before_overlap = ((mask > 128) & (wall_result.mask > 128)).sum()
        mask, _wall = resolve_ceiling_wall_overlap(img, mask, wall_result.mask, direction_ratio=1.5)
        after_overlap = ((mask > 128) & (_wall > 128)).sum()
        cleaned = int(before_overlap - after_overlap)
        print(f"   ✓ Reclassified {cleaned} pixels overlap (vertical→wall, horizontal→ceiling)")

    cov_after = (mask > 128).mean() * 100

    # Phase 4: Export RGBA PNG (transparent ngoài ceiling)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(img_orig, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask  # alpha = mask, vùng khác → 0 (transparent)
    cv2.imwrite(str(args.out), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # Save raw mask alongside (PS friendly)
    mask_path = args.out.parent / f"{args.out.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    print(f"💾 Output  : {args.out}  ({(args.out.stat().st_size / 1024):.0f}KB)")
    print(f"💾 Mask    : {mask_path}")
    print(f"📊 Final cov: {cov_after:.2f}%  (before resolver: {cov_before:.2f}%)")
    print(f"🏁 Method  : vlm_sam2_cot_sobel_recovery")

    if args.save_report:
        report = {
            "source": str(args.input),
            "output_rgba": str(args.out),
            "output_mask": str(mask_path),
            "target_class": args.target,
            "vlm_model": args.vlm_model,
            "vlm_reasoning": cot.reasoning,
            "vlm_step2_raw": cot.raw_text[:1000],
            "vlm_elapsed_ms": round(cot.elapsed_ms),
            "points_detected": pts,
            "sam_score": round(result.score, 3),
            "preprocess": {
                "clahe_applied": prep.clahe_applied,
                "undistort_applied": prep.undistort_applied,
            },
            "coverage_before_resolver_pct": round(cov_before, 2),
            "coverage_after_resolver_pct": round(cov_after, 2),
            "method": "vlm_sam2_cot_sobel_recovery",
        }
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📋 Report  : {args.save_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
