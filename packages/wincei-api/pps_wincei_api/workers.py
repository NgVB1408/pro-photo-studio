"""Background workers — run heavy AI tasks async via threadpool.

Mỗi job tạo subfolder riêng dưới settings.outputs_dir/<job_id>/.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path

from .config import settings
from .jobs import registry
from .schemas import JobStatus

log = logging.getLogger(__name__)

# Lazy segmenter — load 1 lần share giữa jobs
_segmenter = None


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from pps_wincei_masks import SemanticSegmenter
        _segmenter = SemanticSegmenter()
    return _segmenter


def _zip_dir(src_dir: Path, out_zip: Path) -> Path:
    """Zip toàn bộ folder thành out_zip."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(
        base_name=str(out_zip.with_suffix("")),
        format="zip",
        root_dir=str(src_dir),
    )
    return Path(archive)


def run_window_ceiling_job(job_id: str, inputs: list[Path], **params) -> None:
    """Worker thread for window+ceiling fix."""
    try:
        from pps_wincei import process_image
        registry.update(job_id, status=JobStatus.running, message="Processing window+ceiling...", progress_pct=10)

        job_out = settings.outputs_dir / job_id
        job_out.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []

        for i, src in enumerate(inputs, 1):
            dst = job_out / src.name
            result = process_image(src, dst, **params)
            outputs.append(str(dst))
            pct = 10 + int(80 * i / max(1, len(inputs)))
            registry.update(
                job_id, progress_pct=pct,
                message=f"{i}/{len(inputs)} done"
            )

        zip_path = _zip_dir(job_out, settings.outputs_dir / f"{job_id}.zip")
        registry.update(
            job_id,
            status=JobStatus.completed,
            progress_pct=100,
            outputs=outputs,
            message=f"Done. Zip: {zip_path.name}",
            metadata_merge={"zip": str(zip_path)},
        )
    except Exception as exc:
        log.exception("window_ceiling job %s failed", job_id)
        registry.update(job_id, status=JobStatus.failed, error=f"{exc}\n{traceback.format_exc()[:1000]}")


def run_hdr_fuse_job(job_id: str, inputs: list[Path], **params) -> None:
    try:
        from pps_wincei_hdr import detect_brackets, align_brackets, fuse_mertens
        from pps_wincei_hdr.io_meta import write_jpg_with_meta
        import cv2

        registry.update(job_id, status=JobStatus.running, message="Detecting brackets...", progress_pct=5)

        groups, singletons = detect_brackets(inputs)
        if not groups:
            registry.update(
                job_id, status=JobStatus.failed,
                error="Không detect được bracket trong inputs. Kiểm EXIF DateTimeOriginal + ExposureBiasValue.",
            )
            return

        job_out = settings.outputs_dir / job_id
        job_out.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []

        for i, g in enumerate(groups, 1):
            images = [cv2.imread(str(s.path), cv2.IMREAD_COLOR) for s in g.shots]
            images = align_brackets(images, enabled=params.get("align", True))
            fused = fuse_mertens(
                images,
                contrast_weight=params.get("contrast_weight", 1.0),
                saturation_weight=params.get("saturation_weight", 1.0),
                exposure_weight=params.get("exposure_weight", 1.0),
                gamma=params.get("gamma", 1.0),
            )
            ref = g.reference
            dst = job_out / ref.path.name
            write_jpg_with_meta(fused, dst, reference_path=ref.path, quality=98)
            outputs.append(str(dst))

            pct = 5 + int(85 * i / max(1, len(groups)))
            registry.update(job_id, progress_pct=pct, message=f"{i}/{len(groups)} fused")

        zip_path = _zip_dir(job_out, settings.outputs_dir / f"{job_id}.zip")
        registry.update(
            job_id, status=JobStatus.completed, progress_pct=100,
            outputs=outputs, message=f"Done {len(groups)} fused. Zip: {zip_path.name}",
            metadata_merge={"zip": str(zip_path), "n_groups": len(groups), "n_singletons": len(singletons)},
        )
    except Exception as exc:
        log.exception("hdr_fuse job %s failed", job_id)
        registry.update(job_id, status=JobStatus.failed, error=f"{exc}\n{traceback.format_exc()[:1000]}")


def run_segment_masks_job(job_id: str, inputs: list[Path], **params) -> None:
    try:
        from pps_wincei_masks import extract_masks

        seg = _get_segmenter()
        registry.update(job_id, status=JobStatus.running, message="Running SegFormer...", progress_pct=5)

        job_out = settings.outputs_dir / job_id
        job_out.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        verdicts: list[str] = []

        for i, src in enumerate(inputs, 1):
            result = extract_masks(
                src, job_out,
                segmenter=seg,
                refine_edges=params.get("refine_edges", True),
                detect_molding=params.get("detect_molding", True),
                include_lights=params.get("include_lights", False),
                write_overlay=params.get("write_overlay", True),
                write_tiff=params.get("write_tiff", True),
                write_psd=params.get("write_psd", False),
                self_evaluate=True,
            )
            if result.export and result.export.out_dir:
                outputs.append(str(result.export.out_dir))
            verdicts.append(result.evaluation.verdict if result.evaluation else "?")

            pct = 5 + int(90 * i / max(1, len(inputs)))
            registry.update(
                job_id, progress_pct=pct,
                message=f"{i}/{len(inputs)} done. verdicts={verdicts.count('pass')}p/{verdicts.count('review')}r/{verdicts.count('fail')}f",
            )

        zip_path = _zip_dir(job_out, settings.outputs_dir / f"{job_id}.zip")
        registry.update(
            job_id, status=JobStatus.completed, progress_pct=100,
            outputs=outputs,
            message=f"Done. pass={verdicts.count('pass')} review={verdicts.count('review')} fail={verdicts.count('fail')}",
            metadata_merge={
                "zip": str(zip_path),
                "verdicts": verdicts,
                "pass_count": verdicts.count("pass"),
                "review_count": verdicts.count("review"),
                "fail_count": verdicts.count("fail"),
            },
        )
    except Exception as exc:
        log.exception("segment_masks job %s failed", job_id)
        registry.update(job_id, status=JobStatus.failed, error=f"{exc}\n{traceback.format_exc()[:1000]}")
