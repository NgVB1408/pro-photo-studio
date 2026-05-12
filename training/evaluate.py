"""Evaluation: PSNR / SSIM / LPIPS on a held-out split.

Run on synthetic data, FiveK held-out, or any directory of (input, target)
pairs. LPIPS is optional (`pip install lpips` + torch). PSNR + SSIM use
`pps_core.quality`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger("pps.evaluate")


@dataclass
class EvalSummary:
    n: int = 0
    psnr_mean: float = 0.0
    ssim_mean: float = 0.0
    lpips_mean: float | None = None
    failures: int = 0
    per_pair: list[dict] = field(default_factory=list)


def iter_pairs_from_dir(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield ``(input, target)`` from ``root/<n>_input.*`` + ``root/<n>_target.*``."""
    for inp in sorted(root.glob("*_input.*")):
        stem = inp.stem.replace("_input", "")
        candidates = list(root.glob(f"{stem}_target.*"))
        if candidates:
            yield inp, candidates[0]


def iter_pairs_from_fivek(n: int = 50, expert: str = "c") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(input, target)`` BGR uint8 pairs from FiveK stream."""
    from pps_data import stream_fivek

    ds = stream_fivek(expert=expert)
    yielded = 0
    for row in ds:
        if yielded >= n:
            return
        inp = _to_bgr(row.get("input_image"))
        tgt = _to_bgr(row.get(f"expert_{expert}"))
        if inp is None or tgt is None:
            continue
        # Resize target to input shape (some mirrors store different sizes)
        if tgt.shape != inp.shape:
            tgt = cv2.resize(tgt, (inp.shape[1], inp.shape[0]))
        yield inp, tgt
        yielded += 1


def _to_bgr(obj) -> np.ndarray | None:
    """Best-effort cast of HF cell to BGR uint8 ndarray."""
    if obj is None:
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    img = None
    if hasattr(obj, "convert"):
        img = obj
    elif isinstance(obj, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(obj))
    elif isinstance(obj, dict):
        if obj.get("bytes"):
            import io
            img = Image.open(io.BytesIO(obj["bytes"]))
        elif obj.get("path"):
            img = Image.open(obj["path"])
    if img is None:
        return None
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def evaluate_pairs(
    pairs: Iterator[tuple[np.ndarray, np.ndarray]],
    *,
    with_lpips: bool = False,
    max_pairs: int | None = None,
) -> EvalSummary:
    """Compute PSNR/SSIM/(LPIPS) over an iterable of (input, target) pairs.

    PSNR / SSIM via ``pps_core.quality``. LPIPS is loaded lazily because it
    pulls torch + a network the size of VGG.
    """
    try:
        from pps_core.quality import psnr, ssim
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pps-core not installed") from exc

    lpips_fn = None
    if with_lpips:
        lpips_fn = _load_lpips()

    summary = EvalSummary()
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    for i, (inp, tgt) in enumerate(pairs):
        if max_pairs is not None and i >= max_pairs:
            break
        try:
            p = float(psnr(inp, tgt))
            s = float(ssim(inp, tgt))
            l_ = float(lpips_fn(inp, tgt)) if lpips_fn else None
            psnrs.append(p)
            ssims.append(s)
            if l_ is not None:
                lpipss.append(l_)
            summary.per_pair.append({
                "idx": i, "psnr": round(p, 3), "ssim": round(s, 5),
                "lpips": round(l_, 5) if l_ is not None else None,
            })
        except Exception as e:
            log.warning("pair %d failed: %s", i, e)
            summary.failures += 1
    summary.n = len(psnrs)
    summary.psnr_mean = float(np.mean(psnrs)) if psnrs else 0.0
    summary.ssim_mean = float(np.mean(ssims)) if ssims else 0.0
    summary.lpips_mean = float(np.mean(lpipss)) if lpipss else None
    return summary


def _load_lpips():  # pragma: no cover — optional dep
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError("LPIPS missing: pip install lpips torch") from exc
    model = lpips.LPIPS(net="alex").eval()

    def _score(a: np.ndarray, b: np.ndarray) -> float:
        ta = _to_torch(a)
        tb = _to_torch(b)
        with torch.no_grad():
            return float(model(ta, tb).item())

    return _score


def _to_torch(img: np.ndarray):  # pragma: no cover
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSNR/SSIM/LPIPS evaluation")
    parser.add_argument("--input-dir", type=Path, help="dir of *_input.* + *_target.*")
    parser.add_argument("--source", choices=["dir", "fivek"], default="dir")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--expert", default="c")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("training/reports/eval.json"))
    parser.add_argument("--checkpoint", default="base", help="(metadata only — used to label the run)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.source == "dir":
        if not args.input_dir or not args.input_dir.exists():
            print("error: --input-dir is required for source=dir", file=sys.stderr)
            return 2
        pair_iter = (_pair_from_paths(p) for p in iter_pairs_from_dir(args.input_dir))
    else:
        pair_iter = iter_pairs_from_fivek(n=args.n, expert=args.expert)

    summary = evaluate_pairs(pair_iter, with_lpips=args.lpips, max_pairs=args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": args.checkpoint,
        "source": args.source,
        **asdict(summary),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_pair"}, indent=2))
    print(f"wrote {args.out} with {summary.n} pairs")
    return 0


def _pair_from_paths(pair: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
    a = cv2.imread(str(pair[0]))
    b = cv2.imread(str(pair[1]))
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return a, b


if __name__ == "__main__":
    raise SystemExit(main())
