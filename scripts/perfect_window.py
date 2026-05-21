#!/usr/bin/env python
"""Standalone CLI: ảnh BĐS → window_opening.png (RGBA, chỉ giữ cửa sổ).

Usage:
    python scripts/perfect_window.py foto.jpg
    python scripts/perfect_window.py foto.jpg --out window_opening.png --no-vlm
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
    p = argparse.ArgumentParser(
        description="Perfect Window v0.3.4 — VLM bbox + SAM box + Canny boundary → PNG RGBA"
    )
    p.add_argument("input", type=Path)
    p.add_argument("--out", "-o", type=Path, default=Path("./window_opening.png"))
    p.add_argument("--no-vlm", action="store_true", help="Skip Ollama, dùng semantic fallback")
    p.add_argument("--vlm-model", default="bds-brain")
    p.add_argument("--sam-checkpoint", type=Path, default=None)
    p.add_argument("--padding-pct", type=float, default=0.03)
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument("--feather", type=int, default=4)
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
    print(f"📷 Input  : {args.input.name}  {w}×{h}")

    # VLM bbox
    vlm_bbox = None
    if not args.no_vlm:
        from pps_wincei_masks.vlm_client import OllamaVLM, check_ollama_available
        ok, models = check_ollama_available()
        if ok and any(args.vlm_model in m for m in models):
            print(f"🧠 VLM: {args.vlm_model} → query window bbox...")
            vlm = OllamaVLM(
                model=args.vlm_model,
                endpoint="http://localhost:11434/api/chat",
                use_chat_api=True,
            )
            prompt = (
                "Quét ảnh nội thất. Trả JSON DUY NHẤT: "
                '{"window_bbox": [ymin, xmin, ymax, xmax]} '
                "ôm sát vùng cửa sổ/cửa kính. Pixel coords ảnh gốc."
            )
            try:
                resp = vlm.query(img, prompt=prompt, max_side=1280)
                pts = resp.parsed_points
                if "window_bbox" in pts and len(pts["window_bbox"]) >= 4:
                    bb = pts["window_bbox"]
                    vlm_bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                    print(f"   ✓ VLM bbox: {vlm_bbox}")
            except RuntimeError as exc:
                print(f"   ⚠️  VLM fail: {exc}, semantic fallback")

    # Semantic fallback nếu cần
    sem_result = None
    if vlm_bbox is None:
        from pps_wincei_masks.semantic import SemanticSegmenter
        print("🎯 SegFormer semantic (fallback bbox)...")
        seg = SemanticSegmenter()
        sem_result = seg.segment(img)

    # SAM 2 (best effort)
    sam_engine = None
    try:
        from pps_wincei_masks.sam_engine import SAMEngine
        sam_engine = SAMEngine(checkpoint=args.sam_checkpoint)
    except Exception as exc:
        print(f"   ⚠️  SAM init fail: {exc}")

    # Extract
    from pps_wincei_masks.perfect_window import extract_perfect_window
    print("🪟 Extract perfect window...")
    result = extract_perfect_window(
        img,
        vlm_bbox=vlm_bbox, sem_result=sem_result, sam_engine=sam_engine,
        bbox_padding_pct=args.padding_pct,
        canny_low=args.canny_low, canny_high=args.canny_high,
        feather_px=args.feather,
    )
    cov_pct = (result.window_mask > 128).mean() * 100
    print(f"   ✓ Method: {result.method}  SAM score: {result.sam_score:.3f}")
    print(f"   ✓ bbox: {result.bbox}  Canny edges: {result.n_canny_edges}")
    print(f"   ✓ Coverage: {cov_pct:.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    mask_path = args.out.parent / f"{args.out.stem}_mask.png"
    cv2.imwrite(str(mask_path), result.window_mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"💾 Output RGBA : {args.out}")
    print(f"💾 Mask        : {mask_path}")

    if args.save_report:
        report = {
            "source": str(args.input),
            "method": result.method,
            "bbox_ymin_xmin_ymax_xmax": list(result.bbox),
            "sam_score": round(result.sam_score, 3),
            "n_canny_edges": result.n_canny_edges,
            "coverage_pct": round(cov_pct, 2),
            "vlm_used": vlm_bbox is not None,
            "output_rgba": str(args.out),
            "output_mask": str(mask_path),
        }
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📋 Report      : {args.save_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
