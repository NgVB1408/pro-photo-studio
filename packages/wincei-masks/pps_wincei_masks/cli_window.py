"""CLI: pps-wincei-window — geometric BBOX crop ONLY.

Usage:
    pps-wincei-window foto.jpg                          # → ./window_opening.png
    pps-wincei-window foto.jpg --out cropped.png
    pps-wincei-window foto.jpg --bbox 420 1500 2200 3200  # explicit bbox
    pps-wincei-window foto.jpg --no-vlm                 # semantic fallback only
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
        prog="pps-wincei-window",
        description="Perfect Window v0.3.4 — GEOMETRIC BBOX CROP → PNG RGBA (no manipulation)",
    )
    p.add_argument("input", type=Path)
    p.add_argument("--out", "-o", type=Path, default=Path("./window_opening.png"))
    p.add_argument("--bbox", type=int, nargs=4, metavar=("YMIN", "XMIN", "YMAX", "XMAX"),
                   default=None, help="Explicit bbox (skip VLM + semantic).")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--vlm-model", default="bds-brain")
    p.add_argument("--padding-pct", type=float, default=0.0,
                   help="Padding ngoài bbox theo %% width (default 0 — strict).")
    p.add_argument("--zoom", action="store_true",
                   help="ZOOM & CROP strategy (CLAHE + SAM trên crop) cho window nhỏ.")
    p.add_argument("--zoom-clahe-clip", type=float, default=3.0)
    p.add_argument("--zoom-padding-pct", type=float, default=0.05,
                   help="Padding zoom mode (default 5%% width).")
    p.add_argument("--sam-checkpoint", type=Path, default=None)
    p.add_argument("--auto-zoom-threshold-pct", type=float, default=3.0,
                   help="Auto trigger --zoom nếu bbox area < này %% of image (default 3).")
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

    # === Resolve bbox ===
    bbox = None
    method = "unknown"

    # Priority 1: explicit --bbox arg
    if args.bbox is not None:
        bbox = tuple(args.bbox)
        method = "explicit_arg"
        print(f"📦 Bbox (explicit): {bbox}")

    # Priority 2: VLM
    if bbox is None and not args.no_vlm:
        try:
            from .vlm_client import OllamaVLM, check_ollama_available
            ok, models = check_ollama_available()
            if ok and any(args.vlm_model in m for m in models):
                vlm = OllamaVLM(
                    model=args.vlm_model,
                    endpoint="http://localhost:11434/api/chat",
                    use_chat_api=True,
                )
                prompt = (
                    'Trả JSON DUY NHẤT: {"window_bbox": [ymin, xmin, ymax, xmax]} '
                    "ôm sát cửa sổ/cửa kính, pixel coords ảnh gốc."
                )
                resp = vlm.query(img, prompt=prompt, max_side=1280)
                bb = resp.parsed_points.get("window_bbox")
                if bb and len(bb) >= 4:
                    bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                    method = "vlm_bbox"
                    print(f"🧠 VLM bbox: {bbox}")
        except Exception as exc:
            print(f"   ⚠️  VLM fail: {exc}", file=sys.stderr)

    # Priority 3: semantic fallback
    sem_result = None
    if bbox is None:
        from .semantic import SemanticSegmenter
        print("🎯 SegFormer semantic fallback...")
        sem_result = SemanticSegmenter().segment(img)
        method = "semantic_fallback"

    # === Decide mode: zoom-crop hay simple bbox ===
    use_zoom = args.zoom

    # Auto-trigger zoom nếu bbox khả dụng + nhỏ
    if not use_zoom and bbox is not None:
        bb_h = bbox[2] - bbox[0]
        bb_w = bbox[3] - bbox[1]
        bb_area_pct = 100.0 * bb_h * bb_w / (h * w)
        if bb_area_pct < args.auto_zoom_threshold_pct:
            use_zoom = True
            print(f"🔍 Auto-zoom triggered: bbox {bb_area_pct:.2f}% < threshold "
                  f"{args.auto_zoom_threshold_pct}%")

    # === Bbox resolve cho zoom mode ===
    if use_zoom and bbox is None:
        if sem_result is None:
            from .semantic import SemanticSegmenter
            sem_result = SemanticSegmenter().segment(img)
        from .perfect_window import _ade_opening_bbox
        bbox = _ade_opening_bbox(sem_result, (h, w))
        if bbox is None:
            print("❌ Cannot find window bbox — neither VLM nor semantic", file=sys.stderr)
            return 2

    # === ZOOM & CROP STRATEGY ===
    if use_zoom and bbox is not None:
        print("🔍 ZOOM & CROP strategy (CLAHE + SAM trên enhanced crop)...")
        sam_engine = None
        try:
            from .sam_engine import SAMEngine
            sam_engine = SAMEngine(checkpoint=args.sam_checkpoint)
        except Exception as exc:
            print(f"   ⚠️  SAM init fail: {exc} — GrabCut fallback")

        from .perfect_window import zoom_crop_window
        zoom_result = zoom_crop_window(
            img, bbox,
            clahe_clip_limit=args.zoom_clahe_clip,
            sam_engine=sam_engine,
            bbox_padding_pct=args.zoom_padding_pct,
            feather_px=3,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), zoom_result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        mask_path = args.out.parent / f"{args.out.stem}_mask.png"
        cv2.imwrite(str(mask_path), zoom_result.window_mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        cov_pct = (zoom_result.window_mask > 128).mean() * 100
        print(f"💾 RGBA  : {args.out}")
        print(f"💾 Mask  : {mask_path}")
        print(f"📦 BBox  : {zoom_result.bbox}  method={zoom_result.method}")
        print(f"🎨 CLAHE : {zoom_result.clahe_applied}")
        print(f"📐 Crop  : {zoom_result.crop_size[1]}×{zoom_result.crop_size[0]} px")
        print(f"📊 Cov   : {cov_pct:.2f}%  SAM score={zoom_result.sam_score:.3f}")

        if args.save_report:
            args.save_report.write_text(json.dumps({
                "source": str(args.input),
                "strategy": "zoom_crop",
                "method": zoom_result.method,
                "bbox_ymin_xmin_ymax_xmax": list(zoom_result.bbox),
                "clahe_applied": zoom_result.clahe_applied,
                "crop_size": list(zoom_result.crop_size),
                "coverage_pct": round(cov_pct, 2),
                "sam_score": round(zoom_result.sam_score, 3),
                "output_rgba": str(args.out),
                "output_mask": str(mask_path),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"📋 Report: {args.save_report}")
        return 0

    # === SIMPLE BBOX CROP (default) ===
    from .perfect_window import extract_perfect_window
    print("✂️  BBOX crop...")
    result = extract_perfect_window(
        img,
        vlm_bbox=bbox,
        sem_result=sem_result,
        bbox_padding_pct=args.padding_pct,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), result.rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"💾 RGBA  : {args.out}")
    print(f"📦 BBox  : {result.bbox}  method={result.method}")

    if args.save_report:
        args.save_report.write_text(json.dumps({
            "source": str(args.input),
            "strategy": "simple_bbox",
            "method": result.method,
            "bbox_ymin_xmin_ymax_xmax": list(result.bbox),
            "output_rgba": str(args.out),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📋 Report: {args.save_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
