"""Batch processor — xử lý cả thư mục/glob, parallel + progress + idempotent."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class BatchResult:
    success: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.success) + len(self.skipped) + len(self.failed)

    def summary(self) -> str:
        return (
            f"OK={len(self.success)}  "
            f"skipped={len(self.skipped)}  "
            f"failed={len(self.failed)}  "
            f"total={self.total}"
        )


def find_images(
    source: str | Path,
    *,
    recursive: bool = False,
    exts: set[str] = IMAGE_EXTS,
) -> list[Path]:
    """Liệt kê file ảnh từ folder hoặc glob pattern."""
    src = Path(str(source)).expanduser()
    files: list[Path] = []
    if src.is_dir():
        pattern = "**/*" if recursive else "*"
        for p in src.glob(pattern):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    elif "*" in str(source) or "?" in str(source):
        # glob pattern
        base = src.parent if src.parent.exists() else Path(".")
        for p in base.glob(src.name):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    elif src.is_file():
        files = [src]
    files.sort()
    return files


def _process_one(
    job: tuple[Path, Path, dict],
) -> tuple[Path, str | None]:
    """Worker — phải import lại trong subprocess (Windows spawn)."""
    in_path, out_path, kwargs = job
    try:
        from .detect import auto_mask
        from .inpaint import InpaintBackend, inpaint
        from .mask import (
            build_mask_from_boxes,
            build_mask_from_color,
            build_mask_from_image,
            combine_masks,
            dilate_mask,
        )
        from .utils import read_image, write_image

        img = read_image(in_path)
        masks = []
        if kwargs.get("auto"):
            masks.append(
                auto_mask(
                    img,
                    strategy=kwargs.get("auto_strategy", "auto"),
                    border_only=kwargs.get("border_only", True),
                    dilate_iters=0,  # dilate sau cùng
                )
            )
        if kwargs.get("boxes"):
            masks.append(build_mask_from_boxes(img, kwargs["boxes"]))
        if kwargs.get("mask_path"):
            masks.append(build_mask_from_image(img, kwargs["mask_path"]))
        if kwargs.get("color_lower"):
            masks.append(
                build_mask_from_color(
                    img,
                    lower=tuple(kwargs["color_lower"]),
                    upper=tuple(kwargs.get("color_upper", (255, 255, 255))),
                )
            )
        if not masks:
            return in_path, "không có mask source"

        mask = combine_masks(masks)
        dilate_iters = int(kwargs.get("dilate", 2))
        if dilate_iters > 0:
            mask = dilate_mask(mask, iterations=dilate_iters)

        result = inpaint(
            img,
            mask,
            backend=InpaintBackend(kwargs.get("backend", "opencv")),
            opencv_method=kwargs.get("opencv_method", "telea"),
            opencv_radius=int(kwargs.get("opencv_radius", 3)),
            lama_device=kwargs.get("lama_device", "cpu"),
            hd_strategy=kwargs.get("hd_strategy", "crop"),
        )
        write_image(out_path, result, quality=int(kwargs.get("quality", 95)))
        return in_path, None
    except Exception as exc:
        return in_path, f"{type(exc).__name__}: {exc}"


def run_batch(
    files: Iterable[Path],
    out_dir: str | Path,
    *,
    workers: int | None = None,
    skip_existing: bool = True,
    suffix: str = "_clean",
    out_format: str | None = None,
    progress: bool = True,
    **inpaint_kwargs,
) -> BatchResult:
    """Chạy inpaint trên nhiều file song song.

    workers: None -> dùng os.cpu_count() // 2 (cân bằng I/O & CPU).
    out_format: None giữ định dạng gốc; hoặc '.jpg', '.png', '.webp'.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(files)

    if not files:
        logger.warning("Không có file ảnh nào để xử lý")
        return BatchResult()

    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    result = BatchResult()

    jobs: list[tuple[Path, Path, dict]] = []
    for f in files:
        ext = (out_format or f.suffix).lower()
        if not ext.startswith("."):
            ext = "." + ext
        out_path = out_dir / f"{f.stem}{suffix}{ext}"
        if skip_existing and out_path.exists():
            result.skipped.append(f)
            continue
        jobs.append((f, out_path, inpaint_kwargs))

    if not jobs:
        logger.info("Tất cả output đã tồn tại — không có việc mới")
        return result

    bar = tqdm(total=len(jobs), disable=not progress, desc="batch", unit="img")
    if workers <= 1:
        for job in jobs:
            in_path, err = _process_one(job)
            if err:
                result.failed.append((in_path, err))
            else:
                result.success.append(in_path)
            bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, j): j[0] for j in jobs}
            for fut in as_completed(futures):
                in_path, err = fut.result()
                if err:
                    result.failed.append((in_path, err))
                else:
                    result.success.append(in_path)
                bar.update(1)
    bar.close()
    logger.info("Batch xong: %s", result.summary())
    return result


def call_per_file(
    files: Iterable[Path],
    fn: Callable[[Path], None],
    *,
    progress: bool = True,
) -> None:
    """Helper sequential cho debug — không dùng multiprocessing."""
    files = list(files)
    bar = tqdm(total=len(files), disable=not progress)
    for f in files:
        try:
            fn(f)
        except Exception as exc:
            logger.error("%s: %s", f, exc)
        finally:
            bar.update(1)
    bar.close()
