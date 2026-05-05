"""Batch processing worker — chạy trên QThread, không block UI.

Signals:
- progress(current, total, filename) — update progress bar + log
- file_done(filename, success, duration_ms)
- finished(stats)  — dict {ok, fail, total_seconds}
- error(message)
- cancelled()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """1 job batch processing."""
    input_dir: Path
    output_dir: Path
    output_format: str = "jpg"           # jpg | png | webp
    output_quality: int = 95
    output_size_label: str = "Giữ nguyên"  # ref OUTPUT_SIZE_CHOICES
    # Toggles
    denoise: bool = False
    keep_size: bool = True               # tương đương "Lấy chuẩn size gốc"
    detail_recovery: bool = False        # "Phục Hồi Chi Tiết"
    color_enhance: bool = True
    enhance_preset: str = "real_estate"  # studio | real_estate | portrait | product | outdoor
    realestate_pipeline: bool = True     # Auto vertical / window pull / lawn / classify (luôn bật)
    enable_sky_replace: bool = False     # Replace sky chỉ khi user chọn preset
    sky_preset: str = "blue_clouds"      # nếu enable_sky_replace
    sky_source: str = "auto"             # procedural | real_photo | auto
    # ===== AI V5 features (Autoenhance parity) =====
    perspective_correct: bool = False    # Adobe Upright
    lens_correct: bool = False           # Brown-Conrady distortion fix
    auto_privacy: bool = False           # blur faces + plates
    tv_blackout: bool = False            # detect TV → blacken
    fire_fireplace: bool = False         # composite fire
    photog_removal: bool = False         # mirror reflection inpaint
    ai_inpaint: bool = False             # LaMa for any user-marked region
    ai_upscale_scale: int = 0            # 0=off, 2 hoặc 4
    ai_upscale_model: str = "RealESRGAN_x4plus"
    # ===== Pro v2 features =====
    seed: int | None = None              # determinism cho sky pick + random ops
    tone_preset: str = "neutral"         # neutral | warm | cool | auto (locked batch-wide)
    tone_strength: float = 0.5
    selective_sharpen: bool = False      # saliency-based sharpen subject only
    auto_hdr_merge: bool = True          # auto group brackets -> 1 HDR output
    review_contact_sheet: bool = True    # before/after sheet for QA
    write_processing_report: bool = True # CSV + desktop log friendly report
    # ===== Pro v3 features =====
    preflight_check: bool = True         # blur/exposure/dimension QC trước khi xử lý
    use_ai_sky: bool = True              # rembg-based sky segmentation (fallback heuristic)
    accept_raw: bool = True              # nhận RAW input qua rawpy nếu cài
    # ===== Pro v4 features (port từ imagen-ai + Edit-image) =====
    virtual_twilight: bool = False       # Day → Sunset/Golden Hour (opt-in)
    twilight_strength: float = 0.85
    hdr_deghost: bool = True             # Bracket fuse: skip ghost pixel
    hdr_color_normalize: bool = True     # Bracket fuse: LAB-match cross frames
    # Recursive
    recursive: bool = False
    skip_existing: bool = True


@dataclass
class BatchStats:
    ok: int = 0
    fail: int = 0
    skipped: int = 0
    preflight_warnings: int = 0
    preflight_fails: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    total_seconds: float = 0.0
    report_path: Path | None = None
    review_sheet_path: Path | None = None

    @property
    def total(self) -> int:
        return self.ok + self.fail + self.skipped

    def summary(self) -> str:
        lines = [
            f"✅ Thành công: {self.ok}",
            f"❌ Thất bại : {self.fail}",
            f"⊘  Bỏ qua  : {self.skipped} (đã tồn tại)",
            f"⚠ Preflight: {self.preflight_warnings} warn, {self.preflight_fails} fail",
            f"⏱  Tổng     : {self.total_seconds:.1f}s",
        ]
        if self.total > 0:
            lines.append(f"   Trung bình: {self.total_seconds/self.total:.2f}s/ảnh")
        if self.failed_files:
            lines.append("\nFile thất bại (10 đầu):")
            for fname, err in self.failed_files[:10]:
                lines.append(f"  ✗ {fname}: {err}")
        if self.report_path:
            lines.append(f"\nReport: {self.report_path}")
        if self.review_sheet_path:
            lines.append(f"Review trước/sau: {self.review_sheet_path}")
        return "\n".join(lines)


class BracketSignature(NamedTuple):
    path: Path
    shape: tuple[int, int]
    brightness: float
    thumb: np.ndarray
    capture_ts: float | None         # EXIF DateTimeOriginal as epoch (None = unknown)
    exposure_bias: float | None      # EXIF ExposureBiasValue (EV) (None = unknown)


class ProcessingTask(NamedTuple):
    reference: Path
    brackets: tuple[Path, ...]

    @property
    def label(self) -> str:
        if not self.brackets:
            return self.reference.name
        names = ", ".join([self.reference.name, *(p.name for p in self.brackets)])
        return f"HDR({len(self.brackets) + 1}) {names}"


class BatchWorker(QThread):
    """QThread worker — chạy job trong background, emit signals progress."""

    progress = Signal(int, int, str)  # current, total, filename
    file_done = Signal(str, bool, float)  # filename, success, duration_ms
    finished_with_stats = Signal(object)  # BatchStats
    error_occurred = Signal(str)
    log = Signal(str)  # generic log message

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    RAW_EXTS = {
        ".dng", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
        ".raf", ".rw2", ".orf", ".pef", ".srw", ".crw", ".kdc", ".dcr",
        ".mrw", ".rwl", ".x3f", ".3fr", ".iiq", ".fff",
    }

    def __init__(self, job: BatchJob, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel_requested = False
        self._report_rows: list[dict[str, str | float | int]] = []
        self._review_pairs: list[tuple[Path, Path, str]] = []
        self._batch_anchor = None  # BatchAnchor | None — set trong _fit_batch_anchor

    def cancel(self):
        self._cancel_requested = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def _find_images(self) -> list[Path]:
        """Trả list ảnh để xử lý.

        Ưu tiên `_job._explicit_files` (do FlexInputPicker mode=files set),
        nếu không có thì quét `input_dir`.
        """
        explicit = getattr(self._job, "_explicit_files", None)
        if explicit:
            return [Path(p) for p in explicit if Path(p).is_file()]

        d = self._job.input_dir
        if not d.is_dir():
            raise FileNotFoundError(f"Thư mục input không tồn tại: {d}")
        pattern = "**/*" if self._job.recursive else "*"
        accepted = set(self.IMAGE_EXTS)
        if self._job.accept_raw:
            accepted |= self.RAW_EXTS
        files = []
        for p in d.glob(pattern):
            if p.is_file() and p.suffix.lower() in accepted:
                files.append(p)
        files.sort()
        return files

    def _process_one(
        self,
        img_path: Path,
        *,
        bracket_paths: tuple[Path, ...] = (),
        output_stem: str | None = None,
    ) -> tuple[bool, str | None, float, Path | None, dict]:
        """Xử lý 1 ảnh. Trả (success, err_msg, duration_ms, out_path, info)."""
        from pps_core.utils import read_image, write_image

        info: dict = {
            "scene": "",
            "actions": [],
            "preflight": "",
            "preflight_severity": "ok",
        }
        t0 = time.perf_counter()
        try:
            img = read_image(img_path)
            j = self._job

            # === Stage 0: Preflight QC ===
            if j.preflight_check:
                from pps_core.preflight import analyze_image
                rpt = analyze_image(img)
                info["preflight"] = rpt.csv_summary()
                info["preflight_severity"] = rpt.severity
                if rpt.severity in ("warn", "fail"):
                    icon = "⚠" if rpt.severity == "warn" else "❗"
                    self.log.emit(
                        f"  {icon} {img_path.name}: {rpt.csv_summary()}"
                        + (f" → {rpt.suggested_action}" if rpt.suggested_action else "")
                    )

            bracket_imgs = self._load_bracket_images(img, bracket_paths)

            if bracket_imgs:
                from pps_core.hdr import align_brackets, fuse_brackets
                hdr_inputs = align_brackets([img, *bracket_imgs], method="mtb")
                img = fuse_brackets(
                    hdr_inputs,
                    deghost=j.hdr_deghost,
                    color_normalize=j.hdr_color_normalize,
                )
                tag_parts = [f"hdr_fuse({len(hdr_inputs)})"]
                if j.hdr_deghost:
                    tag_parts.append("deghost")
                if j.hdr_color_normalize:
                    tag_parts.append("color_norm")
                info["actions"].append("+".join(tag_parts))

            # === Stage 1: Geometric correction (làm trước enhance để clean view) ===
            if j.lens_correct:
                from pps_core.lens import auto_correct_lens
                img, _ = auto_correct_lens(img, image_path=img_path)
            if j.perspective_correct:
                from pps_core.perspective import correct_upright
                img, _ = correct_upright(img)

            # === Stage 2: Object removal (cần làm trước color enhance) ===
            if j.photog_removal:
                from pps_core.photog_removal import remove_photographer
                img, _ = remove_photographer(img, use_ai_inpaint=True)
            if j.tv_blackout:
                from pps_core.tv_blackout import tv_blackout
                img, _ = tv_blackout(img)

            # === Stage 3: Compositing add ===
            if j.fire_fireplace:
                from pps_core.fire_fireplace import fire_in_fireplace
                img, _ = fire_in_fireplace(img)

            # === Stage 4: Color/exposure pipeline ===
            if j.realestate_pipeline:
                from pps_core.realestate import enhance_realestate_full
                img, re_report = enhance_realestate_full(
                    img,
                    sky_preset=j.sky_preset, seed=j.seed,
                    brackets=bracket_imgs or None,
                    enable_sky=j.enable_sky_replace,
                    use_ai_sky=j.use_ai_sky,
                )
                try:
                    info["scene"] = re_report.scene.tag
                    parts = []
                    if re_report.sky_replaced:
                        parts.append(f"sky:{re_report.sky_preset_used}")
                    if re_report.windows_recovered:
                        parts.append("window_pull")
                    if re_report.lawn_enhanced:
                        parts.append("lawn")
                    if re_report.vertical.rotated:
                        parts.append(f"vert:{re_report.vertical.angle_deg:.1f}°")
                    info["actions"].extend(parts)
                except AttributeError:
                    pass
                # Vẫn áp generic preset cho ảnh ở scene "unknown" để không bị thiếu enhance
                if j.color_enhance and getattr(re_report, "scene", None) and re_report.scene.tag == "unknown":
                    from pps_core.enhance import enhance_preset
                    img = enhance_preset(img, j.enhance_preset)
                    info["actions"].append(f"preset:{j.enhance_preset}")
            elif j.color_enhance:
                from pps_core.enhance import enhance_preset
                img = enhance_preset(img, j.enhance_preset)
                info["actions"].append(f"preset:{j.enhance_preset}")

            if j.detail_recovery:
                from pps_core.hdr import recover_blown_windows, pseudo_hdr_single
                if bracket_imgs:
                    img, _ = recover_blown_windows(
                        img, mode="bracket", brackets=bracket_imgs,
                        align=True, strength=0.85,
                    )
                else:
                    img = pseudo_hdr_single(img)

            # === Stage 4.5: Tone coherency (lock batch-wide) ===
            if j.tone_preset == "auto_batch" and self._batch_anchor is not None:
                img = self._batch_anchor.apply(img, strength=j.tone_strength)
                info["actions"].append("tone:auto_batch")
            elif j.tone_preset and j.tone_preset not in ("neutral", "auto_batch"):
                from pps_core.tone_coherency import TonePreset
                tone = TonePreset(name=j.tone_preset, strength=j.tone_strength)
                img = tone.apply(img)
                info["actions"].append(f"tone:{j.tone_preset}")

            # === Stage 4.6: Selective sharpening (saliency-based) ===
            if j.selective_sharpen:
                from pps_core.saliency_sharpen import saliency_sharpen
                img = saliency_sharpen(img, sharp_amount=0.6, bg_smooth=0.3)

            # === Stage 4.7: Virtual Twilight (Day → Sunset, opt-in) ===
            if j.virtual_twilight:
                from pps_core.twilight import transform_to_twilight
                img, tw = transform_to_twilight(
                    img,
                    strength=j.twilight_strength,
                    seed=j.seed,
                    use_ai_sky=j.use_ai_sky,
                )
                if tw.applied:
                    info["actions"].append(
                        f"twilight(sky={tw.sky_mask_pct:.0f}%,"
                        f"glow={tw.glow_windows_pct:.0f}%)"
                    )
                else:
                    info["actions"].append(f"twilight:skip({tw.reason})")

            if j.denoise:
                from pps_core.enhance import denoise as denoise_fn
                img = denoise_fn(img, strength=5)

            # === Stage 5: Privacy (face/plate blur — sau enhance để mask đúng tone) ===
            if j.auto_privacy:
                from pps_core.auto_privacy import auto_privacy
                img, _ = auto_privacy(img)

            # === Stage 6: AI Upscale (cuối cùng để tránh upscale noise) ===
            if j.ai_upscale_scale in (2, 4):
                from pps_core.upscale import upscale_ai_safe
                img = upscale_ai_safe(img, scale=j.ai_upscale_scale,
                                       model_name=j.ai_upscale_model)

            # === Stage 7: Output size (downscale only) ===
            img = _resize_for_output(img, j.output_size_label)

            # === Write ===
            out_name = self._build_output_name(img_path, output_stem=output_stem)
            out_path = j.output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if j.skip_existing and out_path.exists():
                return False, "skipped", (time.perf_counter() - t0) * 1000, out_path, info

            write_image(out_path, img, quality=j.output_quality,
                          exif_source=img_path)
            return True, None, (time.perf_counter() - t0) * 1000, out_path, info

        except Exception as exc:  # noqa: BLE001
            logger.exception("Process %s fail", img_path)
            return False, f"{type(exc).__name__}: {exc}", (time.perf_counter() - t0) * 1000, None, info

    def _load_bracket_images(self, ref_img: np.ndarray, paths: tuple[Path, ...]) -> list[np.ndarray]:
        if not paths:
            return []
        from pps_core.utils import read_image

        h, w = ref_img.shape[:2]
        imgs: list[np.ndarray] = []
        for p in paths:
            b = read_image(p)
            if b.shape[:2] != (h, w):
                b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
            imgs.append(b)
        return imgs

    def _build_output_name(self, img_path: Path, *, output_stem: str | None = None) -> str:
        ext = self._job.output_format.lower().lstrip(".")
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            ext = "jpg"
        ext_dot = "." + ("jpg" if ext == "jpeg" else ext)
        return f"{output_stem or img_path.stem}{ext_dot}"

    def run(self):
        stats = BatchStats()
        t_start = time.perf_counter()

        try:
            files = self._find_images()
            if not files:
                self.error_occurred.emit(
                    f"Không tìm thấy ảnh nào trong: {self._job.input_dir}"
                )
                return
            tasks = self._build_processing_tasks(files)
            self._job.output_dir.mkdir(parents=True, exist_ok=True)
            hdr_groups = sum(1 for t in tasks if t.brackets)
            self.log.emit(
                f"Tìm thấy {len(files)} ảnh → {len(tasks)} output "
                f"({hdr_groups} nhóm HDR bracket)."
            )
            logger.info(
                "Batch start: files=%d tasks=%d hdr_groups=%d output=%s",
                len(files), len(tasks), hdr_groups, self._job.output_dir,
            )
            if self._job.tone_preset == "auto_batch":
                self._fit_batch_anchor(tasks)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Lỗi quét folder: {exc}")
            return

        for i, task in enumerate(tasks):
            if self._cancel_requested:
                self.log.emit(f"⚠ Đã huỷ bởi user (xử lý {i}/{len(tasks)})")
                break
            output_stem = f"{task.reference.stem}_hdr" if task.brackets else None
            self.progress.emit(i + 1, len(tasks), task.label)
            success, err, dur_ms, out_path, file_info = self._process_one(
                task.reference,
                bracket_paths=task.brackets,
                output_stem=output_stem,
            )
            self.file_done.emit(task.label, success, dur_ms)
            self._record_report_row(task, success, err, dur_ms, out_path, file_info)
            severity = (file_info or {}).get("preflight_severity", "ok")
            if severity == "warn":
                stats.preflight_warnings += 1
            elif severity == "fail":
                stats.preflight_fails += 1
            if success:
                stats.ok += 1
                if out_path:
                    scene_tag = (file_info or {}).get("scene", "") or ""
                    self._review_pairs.append((task.reference, out_path, task.label, scene_tag))
                self.log.emit(
                    f"  ✓ {task.label} ({dur_ms/1000:.1f}s)"
                )
            elif err == "skipped":
                stats.skipped += 1
                self.log.emit(f"  ⊘ {task.label}: bỏ qua (đã tồn tại)")
            else:
                stats.fail += 1
                stats.failed_files.append((task.label, err or "unknown"))
                self.log.emit(f"  ✗ {task.label}: {err}")

        stats.total_seconds = time.perf_counter() - t_start
        if self._job.write_processing_report:
            stats.report_path = self._write_processing_report(stats)
        if self._job.review_contact_sheet:
            stats.review_sheet_path = self._write_review_contact_sheet()
        logger.info("Batch finished: %s", stats.summary().replace("\n", " | "))
        self.finished_with_stats.emit(stats)

    def _fit_batch_anchor(self, tasks: list[ProcessingTask]) -> None:
        """Sample reference của mỗi task, fit BatchAnchor LAB median.

        Cap 24 samples đều khắp batch để giữ pre-pass < 5s ngay cả với
        batch 200 ảnh. Reference là ảnh ev=0 đã chọn trong _build_processing_tasks
        nên anchor không bị skew bởi over/under bracket.
        """
        from pps_core.tone_coherency import BatchToneFitter

        refs = [t.reference for t in tasks]
        if len(refs) > 24:
            step = max(1, len(refs) // 24)
            refs = refs[::step][:24]
        fitter = BatchToneFitter(sample_short_edge=384)
        sampled = 0
        for p in refs:
            if self._cancel_requested:
                break
            if fitter.add_from_path(p):
                sampled += 1
        anchor = fitter.fit_anchor()
        if anchor is None:
            self.log.emit(
                "⚠ Tone auto_batch: không sample được ảnh nào — fallback neutral."
            )
            return
        self._batch_anchor = anchor
        self.log.emit(
            "🎭 Tone anchor (auto_batch, n="
            f"{anchor.samples}): L={anchor.lab_median[0]:.1f} "
            f"a={anchor.lab_median[1]:.1f} b={anchor.lab_median[2]:.1f} "
            f"hue={anchor.hue_mean:.1f}"
        )

    def _build_processing_tasks(self, files: list[Path]) -> list[ProcessingTask]:
        if not self._job.auto_hdr_merge or len(files) < 2:
            return [ProcessingTask(f, ()) for f in files]

        sigs = [self._signature(p) for p in files]
        tasks: list[ProcessingTask] = []
        i = 0
        while i < len(files):
            best: list[BracketSignature] = [sigs[i]]
            for j in range(i + 1, min(i + 5, len(files))):
                if self._can_be_same_scene(sigs[i], sigs[j]):
                    best.append(sigs[j])
                else:
                    break

            group = self._pick_hdr_group(best)
            if len(group) >= 2:
                ref = min(group, key=lambda s: abs(s.brightness - 128.0))
                brackets = tuple(s.path for s in group if s.path != ref.path)
                tasks.append(ProcessingTask(ref.path, brackets))
                i += len(group)
            else:
                tasks.append(ProcessingTask(files[i], ()))
                i += 1
        return tasks

    def _signature(self, path: Path) -> BracketSignature:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {path}")
        h, w = img.shape[:2]
        small = cv2.resize(img, (96, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        thumb = cv2.equalizeHist(gray).astype(np.float32)
        brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]))
        ts, ev = _read_exif_capture_meta(path)
        return BracketSignature(path, (h, w), brightness, thumb, ts, ev)

    def _can_be_same_scene(self, a: BracketSignature, b: BracketSignature) -> bool:
        if a.shape != b.shape:
            return False
        # EXIF strong hint: cùng burst (≤3s) thì auto coi là cùng scene, bỏ qua corr.
        if a.capture_ts is not None and b.capture_ts is not None:
            if abs(a.capture_ts - b.capture_ts) > 6.0:
                return False
            if abs(a.capture_ts - b.capture_ts) <= 3.0:
                return True
        va = a.thumb.reshape(-1)
        vb = b.thumb.reshape(-1)
        if float(np.std(va)) < 1e-3 or float(np.std(vb)) < 1e-3:
            return False
        corr = float(np.corrcoef(va, vb)[0, 1])
        return corr >= 0.93

    def _pick_hdr_group(self, sigs: list[BracketSignature]) -> list[BracketSignature]:
        if len(sigs) < 2:
            return []
        # 1. EXIF-driven: nếu cluster có ExposureBias khác nhau ≥ 1 EV → bracket chắc chắn
        ev_vals = [s.exposure_bias for s in sigs if s.exposure_bias is not None]
        if len(ev_vals) >= 2 and max(ev_vals) - min(ev_vals) >= 1.0:
            # Lấy tối đa 5 ảnh có EV phân tán nhất, ưu tiên gần ảnh đầu (cùng burst)
            for n in (5, 4, 3, 2):
                if len(sigs) >= n:
                    cand = sigs[:n]
                    cand_ev = [s.exposure_bias for s in cand if s.exposure_bias is not None]
                    if len(cand_ev) >= 2 and max(cand_ev) - min(cand_ev) >= 1.0:
                        return cand
        # 2. Brightness fallback: spread ≥18 V-units thì coi là bracket.
        for n in (5, 4, 3, 2):
            if len(sigs) >= n:
                cand = sigs[:n]
                spread = max(s.brightness for s in cand) - min(s.brightness for s in cand)
                if spread >= 18.0:
                    return cand
        return []

    def _record_report_row(
        self,
        task: ProcessingTask,
        success: bool,
        err: str | None,
        dur_ms: float,
        out_path: Path | None,
        info: dict | None = None,
    ) -> None:
        info = info or {}
        actions = info.get("actions") or []
        self._report_rows.append({
            "status": "ok" if success else ("skipped" if err == "skipped" else "fail"),
            "reference": str(task.reference),
            "brackets": " | ".join(str(p) for p in task.brackets),
            "output": str(out_path or ""),
            "preflight": info.get("preflight_severity", "ok"),
            "preflight_msg": info.get("preflight", ""),
            "scene": info.get("scene", ""),
            "actions": " + ".join(actions),
            "duration_seconds": round(dur_ms / 1000.0, 3),
            "message": err or "",
        })

    def _write_processing_report(self, stats: BatchStats) -> Path | None:
        try:
            import csv
            from datetime import datetime
            report = self._job.output_dir / "processing_report.csv"
            j = self._job
            settings_summary = (
                f"preset={j.enhance_preset} | re_pipeline={j.realestate_pipeline} | "
                f"sky_replace={j.enable_sky_replace}({j.sky_preset if j.enable_sky_replace else 'off'}) | "
                f"ai_sky={j.use_ai_sky} | window=auto | lens={j.lens_correct} | "
                f"upright={j.perspective_correct} | hdr_merge={j.auto_hdr_merge} | "
                f"raw_input={j.accept_raw} | preflight={j.preflight_check} | "
                f"upscale={j.ai_upscale_scale}x | tone={j.tone_preset}({j.tone_strength:.2f})"
            )
            with report.open("w", newline="", encoding="utf-8-sig") as f:
                f.write(f"# Pro Photo Studio batch report\n")
                f.write(f"# generated: {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"# settings: {settings_summary}\n")
                fieldnames = [
                    "status", "reference", "brackets", "output",
                    "preflight", "preflight_msg", "scene", "actions",
                    "duration_seconds", "message",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self._report_rows)
                w.writerow({
                    "status": "summary",
                    "reference": f"ok={stats.ok} fail={stats.fail} skipped={stats.skipped}",
                    "duration_seconds": round(stats.total_seconds, 3),
                    "actions": settings_summary,
                })
            return report
        except Exception as exc:  # noqa: BLE001
            logger.warning("Write processing report fail: %s", exc)
            return None

    def _write_review_contact_sheet(self) -> Path | None:
        if not self._review_pairs:
            return None
        try:
            from datetime import datetime
            from PIL import Image, ImageDraw, ImageFont

            review_dir = self._job.output_dir / "_review"
            review_dir.mkdir(parents=True, exist_ok=True)
            sheet_path = review_dir / "before_after_contact_sheet.jpg"

            # Page-size A4-ish layout: 6 rows / page, 60 max → up to 10 pages.
            font = _load_review_font(15)
            font_small = _load_review_font(12)
            font_header = _load_review_font(20)

            thumb_w = 540
            label_h = 38
            gap = 22
            margin = 28
            header_h = 92

            j = self._job
            header_text = (
                f"Pro Photo Studio — Batch review ({len(self._review_pairs)} ảnh)\n"
                f"Output: {j.output_dir}\n"
                f"Preset: {j.enhance_preset} | RE pipeline: "
                f"{'ON' if j.realestate_pipeline else 'off'} | "
                f"Sky replace: {j.sky_preset if j.enable_sky_replace else 'off'} | "
                f"HDR merge: {'ON' if j.auto_hdr_merge else 'off'} | "
                f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            row_imgs: list[Image.Image] = []
            for before_path, after_path, label, scene_tag in self._review_pairs[:60]:
                try:
                    before = Image.open(before_path).convert("RGB")
                    after = Image.open(after_path).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("contact sheet skip %s: %s", before_path, exc)
                    continue
                b = _pil_fit_width(before, thumb_w)
                a = _pil_fit_width(after, thumb_w)
                row_h = max(b.height, a.height) + label_h
                row = Image.new("RGB", (thumb_w * 2 + 24, row_h), (255, 255, 255))
                row.paste(b, (0, label_h))
                row.paste(a, (thumb_w + 24, label_h))

                d = ImageDraw.Draw(row)
                # Header bar BEFORE
                d.rectangle([(0, 0), (thumb_w, label_h - 2)], fill=(40, 60, 88))
                d.text((10, 8), f"BEFORE  {label[:62]}", fill=(255, 255, 255), font=font)
                if scene_tag:
                    badge = f" [{scene_tag.upper()}] "
                    d.text((thumb_w - 110, 10), badge, fill=(252, 211, 77), font=font_small)
                # Header bar AFTER
                d.rectangle([(thumb_w + 24, 0), (thumb_w * 2 + 24, label_h - 2)],
                            fill=(20, 100, 60))
                d.text((thumb_w + 34, 8),
                       f"AFTER   {after_path.name[:60]}",
                       fill=(255, 255, 255), font=font)
                row_imgs.append(row)

            if not row_imgs:
                return None

            width = thumb_w * 2 + 24 + margin * 2
            content_h = sum(r.height for r in row_imgs) + gap * (len(row_imgs) - 1)
            height = header_h + content_h + margin * 2

            sheet = Image.new("RGB", (width, height), (244, 246, 250))
            d = ImageDraw.Draw(sheet)
            # Header background
            d.rectangle([(0, 0), (width, header_h)], fill=(15, 22, 36))
            for i, line in enumerate(header_text.splitlines()):
                d.text((margin, 12 + i * 18), line,
                       fill=(255, 255, 255) if i == 0 else (170, 200, 240),
                       font=font_header if i == 0 else font_small)

            y = header_h + margin
            for row in row_imgs:
                sheet.paste(row, (margin, y))
                y += row.height + gap
            sheet.save(sheet_path, quality=92, optimize=True)
            return sheet_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Write review contact sheet fail: %s", exc)
            return None


# =====================================================================
# Helpers
# =====================================================================

OUTPUT_SIZE_MAP: dict[str, int | None] = {
    "Giữ nguyên (chất lượng tối đa)": None,
    "6K (cạnh dài 6000)": 6000,
    "4K (cạnh dài 3840)": 3840,
    "Full HD (cạnh dài 1920)": 1920,
    # Aliases
    "Giữ nguyên": None,
    "6K": 6000,
    "4K": 3840,
    "Full HD": 1920,
}


def _resize_for_output(bgr: np.ndarray, label: str) -> np.ndarray:
    target = OUTPUT_SIZE_MAP.get(label)
    if target is None:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= target:
        return bgr
    scale = target / longest
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _pil_fit_width(img, width: int):
    w, h = img.size
    if w == width:
        return img
    scale = width / max(1, w)
    return img.resize((width, int(round(h * scale))), resample=1)


_EXIF_TAG_DATETIME = 36867       # DateTimeOriginal
_EXIF_TAG_EXPOSURE_BIAS = 37380  # ExposureBiasValue


_EXIF_IFD_POINTER = 34665


def _read_exif_capture_meta(path: Path) -> tuple[float | None, float | None]:
    """Trả (epoch_ts, exposure_bias_ev). Bất kỳ field nào thiếu → None.

    Dùng cho bracket detection: cùng burst (≤ vài giây) + EV khác → bracket
    chắc chắn, không cần fallback brightness analysis.

    Note: DateTimeOriginal + ExposureBiasValue nằm trong Exif sub-IFD, không
    phải IFD0. Phải call `exif.get_ifd(_EXIF_IFD_POINTER)` để lấy.
    """
    try:
        from datetime import datetime
        from PIL import Image
    except ImportError:
        return None, None
    ts: float | None = None
    ev: float | None = None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None, None
            sub = exif.get_ifd(_EXIF_IFD_POINTER) or {}
            dt_raw = sub.get(_EXIF_TAG_DATETIME) or exif.get(_EXIF_TAG_DATETIME)
            if isinstance(dt_raw, bytes):
                dt_raw = dt_raw.decode("ascii", "ignore")
            if isinstance(dt_raw, str):
                try:
                    ts = datetime.strptime(dt_raw.strip("\x00 ").strip(), "%Y:%m:%d %H:%M:%S").timestamp()
                except ValueError:
                    ts = None
            ev_raw = sub.get(_EXIF_TAG_EXPOSURE_BIAS)
            if ev_raw is None:
                ev_raw = exif.get(_EXIF_TAG_EXPOSURE_BIAS)
            if ev_raw is not None:
                try:
                    ev = float(ev_raw)
                except (TypeError, ValueError):
                    ev = None
    except Exception:  # noqa: BLE001
        return None, None
    return ts, ev


def _load_review_font(size: int):
    """Cố load font TrueType có sẵn trên Win/Linux để text contact sheet
    rõ ràng. Fallback default bitmap nếu không có font hệ thống."""
    from PIL import ImageFont
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()
