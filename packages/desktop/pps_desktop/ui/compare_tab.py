"""Compare Tab — Single-image before/after viewer giống Lightroom / AutoHDR.

Tải 1 ảnh, chạy full pipeline (cùng settings với Auto tab + AI panel),
hiện before/after qua **vertical drag splitter**: kéo thanh dọc để tỉ lệ
before/after thay đổi real-time.

3 chế độ view (toggle bằng 3 nút):
  - "Split"      : drag splitter ngang
  - "Side-by-side": 2 ảnh cạnh nhau, kích thước bằng nhau
  - "Toggle"     : click giữ chuột để xem original

Status panel dưới cùng:
  - Preflight severity + msg
  - Scene tag + actions taken
  - Pipeline duration

Pipeline chạy trên QThread con để không freeze UI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QThread
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush,
    QFontMetrics, QCursor,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGroupBox, QSizePolicy, QMessageBox, QButtonGroup, QFrame, QTextEdit,
    QProgressBar,
)


logger = logging.getLogger(__name__)


ViewMode = Literal["split", "side", "toggle"]


@dataclass
class CompareResult:
    """Output gói gọn từ pipeline run cho 1 ảnh."""
    before: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))
    after: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))
    preflight_severity: str = "ok"
    preflight_msg: str = ""
    scene: str = ""
    actions: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str = ""


# =====================================================================
# Custom widget: Before/After viewer with draggable split line
# =====================================================================

class BeforeAfterView(QWidget):
    """Vẽ 2 pixmap chồng nhau, dùng split_x để xác định nơi ngắt.

    - mode="split":   trái = before, phải = after, drag handle
    - mode="side":    cạnh nhau, equal width
    - mode="toggle":  hiện after, click giữ chuột → hiện before
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._mode: ViewMode = "split"
        self._split_ratio: float = 0.5
        self._dragging = False
        self._toggle_pressed = False
        self._processing: bool = False  # True → vẽ overlay "Đang xử lý" lên AFTER
        self.setMinimumHeight(360)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #0d1117;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.SizeHorCursor)

    def set_processing(self, on: bool) -> None:
        """Khi True, vẽ overlay tối + label 'Đang xử lý' lên AFTER side."""
        self._processing = on
        self.update()

    # ----- Public API -----

    def set_images(self, before_bgr: np.ndarray, after_bgr: np.ndarray) -> None:
        """Cập nhật 2 ảnh. Tự convert BGR→QPixmap."""
        self._before = self._bgr_to_pixmap(before_bgr) if before_bgr.size else None
        self._after = self._bgr_to_pixmap(after_bgr) if after_bgr.size else None
        self.update()

    def clear(self) -> None:
        self._before = None
        self._after = None
        self.update()

    def set_mode(self, mode: ViewMode) -> None:
        self._mode = mode
        if mode == "split":
            self.setCursor(Qt.SizeHorCursor)
        elif mode == "toggle":
            self.setCursor(Qt.PointingHandCursor)
        else:  # side
            self.setCursor(Qt.ArrowCursor)
        self.update()

    # ----- Internals -----

    @staticmethod
    def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_c = np.ascontiguousarray(rgb)
        h, w = rgb_c.shape[:2]
        qimg = QImage(rgb_c.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _fit_pixmap(self, pix: QPixmap, target: QRect) -> tuple[QPixmap, QRect]:
        """Scale pixmap vừa target rect, trả (scaled, dest_rect_centered)."""
        if pix.isNull() or target.isEmpty():
            return pix, target
        scaled = pix.scaled(
            target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        return scaled, QRect(x, y, scaled.width(), scaled.height())

    def _placeholder(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor(13, 17, 23))
        painter.setPen(QColor(100, 116, 139))
        painter.setFont(QFont("Segoe UI", 13))
        painter.drawText(
            self.rect(), Qt.AlignCenter,
            "Chưa có ảnh — bấm '📂 Mở ảnh' để bắt đầu so sánh trước/sau",
        )

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(13, 17, 23))

        if self._before is None or self._after is None:
            self._placeholder(painter)
            return

        if self._mode == "side":
            self._paint_side(painter)
        elif self._mode == "toggle":
            self._paint_toggle(painter)
        else:
            self._paint_split(painter)

        if self._processing:
            self._paint_processing_overlay(painter)

    def _paint_processing_overlay(self, painter: QPainter) -> None:
        """Vẽ overlay nửa bên phải (AFTER side) báo 'Đang xử lý'."""
        target = self.rect().adjusted(8, 8, -8, -8)
        if self._mode == "split":
            split_x = int(target.width() * self._split_ratio)
            overlay = QRect(target.x() + split_x, target.y(),
                            target.width() - split_x, target.height())
        elif self._mode == "side":
            half_w = (target.width() - 12) // 2
            overlay = QRect(target.x() + half_w + 12, target.y(),
                            half_w, target.height())
        else:  # toggle
            overlay = target

        painter.fillRect(overlay, QColor(0, 0, 0, 140))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
        painter.drawText(overlay, Qt.AlignCenter,
                         "⏳ Đang xử lý...\nChờ pipeline hoàn tất")

    def _paint_split(self, painter: QPainter) -> None:
        # Fit cả 2 cùng vào full rect (assume same shape — apply on `after`)
        target = self.rect().adjusted(8, 8, -8, -8)
        before_scaled, dest = self._fit_pixmap(self._before, target)
        after_scaled, _ = self._fit_pixmap(self._after, target)

        split_x = int(dest.width() * self._split_ratio)
        # Draw before (left part)
        if split_x > 0:
            src_left = QRect(0, 0, split_x, before_scaled.height())
            dst_left = QRect(dest.x(), dest.y(), split_x, dest.height())
            painter.drawPixmap(dst_left, before_scaled, src_left)
        # Draw after (right part)
        if split_x < dest.width():
            src_right = QRect(split_x, 0, after_scaled.width() - split_x, after_scaled.height())
            dst_right = QRect(dest.x() + split_x, dest.y(),
                               dest.width() - split_x, dest.height())
            painter.drawPixmap(dst_right, after_scaled, src_right)

        # Vertical handle line + drag pill
        line_x = dest.x() + split_x
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.drawLine(line_x, dest.y(), line_x, dest.y() + dest.height())

        pill_w, pill_h = 36, 36
        pill_rect = QRect(line_x - pill_w // 2, dest.y() + dest.height() // 2 - pill_h // 2,
                          pill_w, pill_h)
        painter.setBrush(QBrush(QColor(20, 30, 48, 220)))
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.drawRoundedRect(pill_rect, 18, 18)
        painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        painter.drawText(pill_rect, Qt.AlignCenter, "⇔")

        # Labels
        self._draw_label(painter, dest.adjusted(8, 8, -8, -8), Qt.AlignTop | Qt.AlignLeft,
                         "BEFORE", QColor(60, 90, 130))
        self._draw_label(painter, dest.adjusted(8, 8, -8, -8), Qt.AlignTop | Qt.AlignRight,
                         "AFTER", QColor(40, 130, 80))

    def _paint_side(self, painter: QPainter) -> None:
        full = self.rect().adjusted(8, 8, -8, -8)
        gap = 12
        half_w = (full.width() - gap) // 2
        left_target = QRect(full.x(), full.y(), half_w, full.height())
        right_target = QRect(full.x() + half_w + gap, full.y(), half_w, full.height())

        b_pix, b_dest = self._fit_pixmap(self._before, left_target)
        a_pix, a_dest = self._fit_pixmap(self._after, right_target)
        painter.drawPixmap(b_dest, b_pix)
        painter.drawPixmap(a_dest, a_pix)

        self._draw_label(painter, b_dest.adjusted(6, 6, -6, -6), Qt.AlignTop | Qt.AlignLeft,
                         "BEFORE", QColor(60, 90, 130))
        self._draw_label(painter, a_dest.adjusted(6, 6, -6, -6), Qt.AlignTop | Qt.AlignLeft,
                         "AFTER", QColor(40, 130, 80))

    def _paint_toggle(self, painter: QPainter) -> None:
        target = self.rect().adjusted(8, 8, -8, -8)
        # Đang giữ chuột → hiện before; thả ra → after
        pix = self._before if self._toggle_pressed else self._after
        scaled, dest = self._fit_pixmap(pix, target)
        painter.drawPixmap(dest, scaled)
        label = "BEFORE (đang xem)" if self._toggle_pressed else "AFTER (giữ chuột để xem BEFORE)"
        bg = QColor(60, 90, 130) if self._toggle_pressed else QColor(40, 130, 80)
        self._draw_label(painter, dest.adjusted(6, 6, -6, -6), Qt.AlignTop | Qt.AlignLeft,
                         label, bg)

    @staticmethod
    def _draw_label(painter: QPainter, target: QRect, align, text: str, bg: QColor) -> None:
        font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        pad_x, pad_y = 10, 6
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        box_w = text_w + pad_x * 2
        box_h = text_h + pad_y * 2

        if align & Qt.AlignRight:
            x = target.x() + target.width() - box_w
        else:
            x = target.x()
        y = target.y()

        rect = QRect(x, y, box_w, box_h)
        painter.fillRect(rect, bg)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect, Qt.AlignCenter, text)

    # ----- Mouse interaction -----

    def mousePressEvent(self, ev) -> None:
        if self._before is None:
            return
        if self._mode == "split":
            self._dragging = True
            self._update_split(ev.position().x())
        elif self._mode == "toggle":
            self._toggle_pressed = True
            self.update()

    def mouseMoveEvent(self, ev) -> None:
        if self._dragging and self._mode == "split":
            self._update_split(ev.position().x())

    def mouseReleaseEvent(self, _ev) -> None:
        self._dragging = False
        if self._toggle_pressed:
            self._toggle_pressed = False
            self.update()

    def _update_split(self, mouse_x: float) -> None:
        # Recompute fitted dest rect xác định ratio chính xác
        target = self.rect().adjusted(8, 8, -8, -8)
        if self._before is None:
            return
        scaled, dest = self._fit_pixmap(self._before, target)
        local = mouse_x - dest.x()
        ratio = max(0.02, min(0.98, local / max(1, dest.width())))
        self._split_ratio = ratio
        self.update()


# =====================================================================
# Pipeline runner — chạy 1 ảnh trên thread con
# =====================================================================

class _CompareWorker(QThread):
    finished_with_result = Signal(object)        # CompareResult
    stage_progress = Signal(int, int, str)       # current, total, stage name
    stage_log = Signal(str)                       # live log line per stage

    def __init__(self, img_path: Path, job_factory_fn, parent=None):
        super().__init__(parent)
        self._path = img_path
        self._job_factory = job_factory_fn

    def _emit(self, current: int, total: int, name: str, log: str | None = None) -> None:
        self.stage_progress.emit(current, total, name)
        if log:
            self.stage_log.emit(log)

    def run(self) -> None:
        result = CompareResult()
        try:
            from pps_core.utils import read_image
            from pps_core.preflight import analyze_image

            t0 = time.perf_counter()
            self._emit(0, 10, "Đọc ảnh", f"📂 Đọc {self._path.name}")
            before = read_image(self._path)
            result.before = before.copy()

            job = self._job_factory()

            if job.preflight_check:
                self._emit(1, 10, "Pre-flight QC")
                rpt = analyze_image(before)
                result.preflight_severity = rpt.severity
                result.preflight_msg = rpt.csv_summary()
                result.actions.append(f"preflight:{rpt.severity}")
                self.stage_log.emit(f"🩺 Pre-flight: {rpt.severity} — {rpt.csv_summary()}")

            img = before.copy()

            # Geometric
            if job.lens_correct:
                self._emit(2, 10, "Lens correction")
                from pps_core.lens import auto_correct_lens
                img, _ = auto_correct_lens(img, image_path=self._path)
                result.actions.append("lens_correct")
                self.stage_log.emit("📷 Lens correction (Brown-Conrady)")
            if job.perspective_correct:
                self._emit(3, 10, "Perspective upright")
                from pps_core.perspective import correct_upright
                img, _ = correct_upright(img)
                result.actions.append("upright")
                self.stage_log.emit("⊟ Perspective upright (Adobe Upright)")

            # Object removal
            if job.photog_removal:
                self._emit(4, 10, "Photog removal")
                from pps_core.photog_removal import remove_photographer
                img, _ = remove_photographer(img, use_ai_inpaint=True)
                result.actions.append("photog_removal")
                self.stage_log.emit("👤 Photographer removal (mirror inpaint)")
            if job.tv_blackout:
                self._emit(4, 10, "TV blackout")
                from pps_core.tv_blackout import tv_blackout
                img, _ = tv_blackout(img)
                result.actions.append("tv_blackout")
                self.stage_log.emit("📺 TV blackout")

            if job.fire_fireplace:
                self._emit(4, 10, "Fire in fireplace")
                from pps_core.fire_fireplace import fire_in_fireplace
                img, _ = fire_in_fireplace(img)
                result.actions.append("fire_fireplace")
                self.stage_log.emit("🔥 Fire composite vào fireplace")

            # Color/exposure
            if job.realestate_pipeline:
                self._emit(5, 10, "RE pipeline (scene+wb+clahe+window+lawn)")
                from pps_core.realestate import enhance_realestate_full
                img, re_report = enhance_realestate_full(
                    img,
                    sky_preset=job.sky_preset,
                    seed=job.seed,
                    enable_sky=job.enable_sky_replace,
                    use_ai_sky=job.use_ai_sky,
                )
                result.scene = re_report.scene.tag
                result.actions.append(f"scene:{re_report.scene.tag}")
                self.stage_log.emit(f"🏡 Scene classify: {re_report.scene.tag}")
                if re_report.scene.tag == "interior":
                    result.actions.append("indoor_color (wb+clarity+shadow_lift+vibrance)")
                    self.stage_log.emit("  → Indoor color: wb+clarity+shadow_lift+vibrance")
                else:
                    result.actions.append("studio_color (wb+clahe+highlight+shadow+vibrance+detail)")
                    self.stage_log.emit("  → Studio color: wb+clahe+highlight+shadow+vibrance+detail")
                if re_report.sky_replaced:
                    result.actions.append(f"sky:{re_report.sky_preset_used}")
                    self.stage_log.emit(f"☁ Sky replaced: {re_report.sky_preset_used}")
                if re_report.windows_recovered:
                    result.actions.append("window_pull")
                    self.stage_log.emit("🌅 Window pull")
                if re_report.lawn_enhanced:
                    result.actions.append("lawn")
                    self.stage_log.emit("🌿 Lawn enhance")
                if re_report.vertical.rotated:
                    result.actions.append(f"vert:{re_report.vertical.angle_deg:.1f}°")
                    self.stage_log.emit(f"⊟ Vertical correct: {re_report.vertical.angle_deg:.1f}°")
                else:
                    result.actions.append("vert: tilt OK (skip)")
            elif job.color_enhance:
                self._emit(5, 10, "Color preset")
                from pps_core.enhance import enhance_preset
                img = enhance_preset(img, job.enhance_preset)
                result.actions.append(f"preset:{job.enhance_preset}")
                self.stage_log.emit(f"🎨 Preset: {job.enhance_preset}")

            if job.detail_recovery:
                self._emit(6, 10, "Detail recovery")
                from pps_core.hdr import pseudo_hdr_single
                img = pseudo_hdr_single(img)
                result.actions.append("detail_recovery (pseudo_hdr)")
                self.stage_log.emit("✨ Detail recovery (pseudo_HDR)")

            # Tone — auto_batch chỉ có nghĩa khi batch ≥ 2 ảnh; single → log skip
            if job.tone_preset == "auto_batch":
                result.actions.append("tone:auto_batch (single image → skip; chỉ apply trong batch)")
                self.stage_log.emit("🎭 Tone auto_batch: skip (single image, dùng warm/cool/neutral)")
            elif job.tone_preset and job.tone_preset != "neutral":
                self._emit(7, 10, f"Tone {job.tone_preset}")
                from pps_core.tone_coherency import TonePreset
                img = TonePreset(name=job.tone_preset, strength=job.tone_strength).apply(img)
                result.actions.append(f"tone:{job.tone_preset}")
                self.stage_log.emit(f"🎭 Tone: {job.tone_preset}")

            if job.selective_sharpen:
                self._emit(8, 10, "Selective sharpen")
                from pps_core.saliency_sharpen import saliency_sharpen
                img = saliency_sharpen(img, sharp_amount=0.6, bg_smooth=0.3)
                result.actions.append("selective_sharpen (saliency-aware)")
                self.stage_log.emit("🔬 Selective sharpening (saliency-aware)")

            if getattr(job, "virtual_twilight", False):
                self._emit(8, 10, "Virtual twilight (Day→Sunset)")
                from pps_core.twilight import transform_to_twilight
                img, tw = transform_to_twilight(
                    img,
                    strength=getattr(job, "twilight_strength", 0.85),
                    seed=job.seed,
                    use_ai_sky=job.use_ai_sky,
                )
                if tw.applied:
                    result.actions.append(
                        f"twilight (sky={tw.sky_mask_pct:.0f}%,"
                        f" glow={tw.glow_windows_pct:.0f}%)"
                    )
                    self.stage_log.emit(
                        f"🌇 Virtual Twilight: sky composite {tw.sky_mask_pct:.0f}% +"
                        f" warm glow {tw.glow_windows_pct:.0f}%"
                    )
                else:
                    result.actions.append(f"twilight skip ({tw.reason})")
                    self.stage_log.emit(f"🌇 Twilight skip: {tw.reason}")

            if job.denoise:
                self._emit(8, 10, "Denoise")
                from pps_core.enhance import denoise as denoise_fn
                img = denoise_fn(img, strength=5)
                result.actions.append("denoise (bilateral)")
                self.stage_log.emit("🔇 Denoise (bilateral)")

            if job.auto_privacy:
                self._emit(9, 10, "Auto privacy")
                from pps_core.auto_privacy import auto_privacy
                img, _ = auto_privacy(img)
                result.actions.append("auto_privacy (faces+plates blur)")
                self.stage_log.emit("🛡 Auto privacy (faces+plates blur)")

            if job.ai_upscale_scale in (2, 4):
                self._emit(10, 10, f"AI Upscale x{job.ai_upscale_scale}")
                self.stage_log.emit(
                    f"🎯 AI Upscale x{job.ai_upscale_scale} (Real-ESRGAN) — có thể mất vài giây..."
                )
                from pps_core.upscale import upscale_ai_safe
                img = upscale_ai_safe(img, scale=job.ai_upscale_scale,
                                       model_name=job.ai_upscale_model)
                result.actions.append(f"upscale:x{job.ai_upscale_scale} (Real-ESRGAN)")

            # Match before shape so split-view shows aligned comparison.
            if img.shape != before.shape:
                img = cv2.resize(img, (before.shape[1], before.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)

            result.after = img
            result.duration_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:  # noqa: BLE001
            logger.exception("Compare worker fail")
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished_with_result.emit(result)


# =====================================================================
# Tab widget
# =====================================================================

class CompareTab(QWidget):
    """Tab so sánh 1 ảnh — load + run pipeline + before/after viewer.

    Signal `request_job_factory` không dùng — thay vào đó MainWindow truyền
    callback `set_job_factory(fn)` trả BatchJob đã apply AI panel settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_path: Path | None = None
        self._result: CompareResult | None = None
        self._worker: _CompareWorker | None = None
        self._job_factory_fn = None
        self._build_ui()

    # ----- Public API -----

    def set_job_factory(self, fn) -> None:
        """fn() → BatchJob — gọi mỗi lần Run, lấy snapshot settings hiện tại."""
        self._job_factory_fn = fn

    # ----- UI -----

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)

        # ===== Top toolbar =====
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.load_btn = QPushButton("📂 Mở ảnh")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.clicked.connect(self._on_load)
        toolbar.addWidget(self.load_btn)

        self.run_btn = QPushButton("⚡ Chạy pipeline")
        self.run_btn.setObjectName("primary_btn")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        toolbar.addWidget(self.run_btn)

        self.save_btn = QPushButton("💾 Lưu AFTER")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(self.save_btn)

        toolbar.addSpacing(20)

        # View mode buttons
        mode_label = QLabel("Chế độ xem:")
        mode_label.setStyleSheet("color: #94a3b8;")
        toolbar.addWidget(mode_label)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for label, mode in (("⇔ Split", "split"), ("⊞ Side", "side"), ("👁 Toggle", "toggle")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            btn.setMinimumWidth(86)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, m=mode: self._set_mode(m))
            self.mode_group.addButton(btn)
            toolbar.addWidget(btn)
            if mode == "split":
                btn.setChecked(True)

        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        # ===== Progress bar (chỉ hiện khi pipeline đang chạy) =====
        self.progress_row = QHBoxLayout()
        self.progress_row.setSpacing(10)
        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet(
            "color: #93c5fd; font-weight: 600; font-size: 12px;"
        )
        self.progress_row.addWidget(self.stage_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setFormat("%v / %m stages")
        self.progress_row.addWidget(self.progress_bar, 2)
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(self.progress_row)
        self.progress_widget.setVisible(False)
        outer.addWidget(self.progress_widget)

        # ===== Main viewer =====
        self.view = BeforeAfterView()
        outer.addWidget(self.view, 1)

        # ===== Status panel =====
        status_group = QGroupBox("📋 Kết quả pipeline")
        status_group.setMaximumHeight(220)
        sl = QHBoxLayout(status_group)
        sl.setContentsMargins(14, 22, 14, 14)
        sl.setSpacing(20)

        # Left col: preflight + scene
        left = QVBoxLayout()
        left.setSpacing(6)

        self.preflight_label = QLabel("Preflight: —")
        self.preflight_label.setStyleSheet("font-weight: 600;")
        left.addWidget(self.preflight_label)

        self.preflight_msg = QLabel("")
        self.preflight_msg.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.preflight_msg.setWordWrap(True)
        left.addWidget(self.preflight_msg)

        self.scene_label = QLabel("Scene: —")
        self.scene_label.setStyleSheet("font-weight: 600;")
        left.addWidget(self.scene_label)

        self.duration_label = QLabel("Duration: —")
        self.duration_label.setStyleSheet("color: #94a3b8;")
        left.addWidget(self.duration_label)

        left.addStretch()
        sl.addLayout(left, 1)

        # Right col: actions log
        right = QVBoxLayout()
        right.setSpacing(6)
        right_lbl = QLabel("Pipeline actions:")
        right_lbl.setStyleSheet("font-weight: 600;")
        right.addWidget(right_lbl)
        self.actions_text = QTextEdit()
        self.actions_text.setReadOnly(True)
        self.actions_text.setMaximumHeight(120)
        self.actions_text.setStyleSheet(
            "background: #0d1117; border: 1px solid #30363d; "
            "color: #c9d1d9; font-family: Consolas, monospace; font-size: 11px;"
        )
        right.addWidget(self.actions_text)
        sl.addLayout(right, 2)

        outer.addWidget(status_group)

    # ----- Slots -----

    def _on_load(self):
        from pps_core.utils import RAW_EXTS

        raw_glob = " ".join(f"*{e}" for e in sorted(RAW_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh để so sánh trước/sau", str(Path.home()),
            "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff "
            f"{raw_glob})"
            ";;JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff)"
            f";;RAW ({raw_glob});;Tất cả (*.*)",
        )
        if not path:
            return
        self._current_path = Path(path)
        try:
            from pps_core.utils import read_image
            img = read_image(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Lỗi đọc ảnh", str(exc))
            return
        # Hiện ngay before (after = before tạm — chưa run pipeline)
        self.view.set_images(img, img)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.preflight_label.setText("Preflight: chưa chạy")
        self.preflight_msg.setText("Bấm '⚡ Chạy pipeline' để xử lý.")
        self.scene_label.setText(f"📷 {self._current_path.name} — {img.shape[1]}×{img.shape[0]}")
        self.duration_label.setText("Duration: —")
        self.actions_text.setPlainText("")

    def _on_run(self):
        if self._current_path is None:
            return
        if self._job_factory_fn is None:
            QMessageBox.warning(self, "Chưa wired",
                                "MainWindow chưa cấu hình job factory.")
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Đang xử lý...")
        self.save_btn.setEnabled(False)
        # Show progress + reset live log
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Khởi động pipeline…")
        self.actions_text.clear()
        # Overlay AFTER side để user biết đang chờ kết quả thật
        self.view.set_processing(True)

        self._worker = _CompareWorker(self._current_path, self._job_factory_fn, self)
        self._worker.stage_progress.connect(self._on_stage_progress)
        self._worker.stage_log.connect(self._on_stage_log)
        self._worker.finished_with_result.connect(self._on_run_done)
        self._worker.start()

    def _on_stage_progress(self, current: int, total: int, name: str):
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(current)
        self.stage_label.setText(f"Stage {current}/{total} — {name}")

    def _on_stage_log(self, line: str):
        self.actions_text.append(line)
        # Auto-scroll
        scroll = self.actions_text.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _on_run_done(self, result: CompareResult):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("⚡ Chạy pipeline")
        # Hoàn tất progress bar + tắt overlay
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.stage_label.setText(f"✅ Hoàn tất ({result.duration_ms / 1000:.2f}s)")
        self.view.set_processing(False)
        self._result = result

        if result.error:
            self.progress_widget.setVisible(True)
            self.stage_label.setText("❌ Pipeline FAILED")
            QMessageBox.warning(self, "Pipeline lỗi", result.error)
            self.preflight_label.setText("Pipeline FAILED")
            self.preflight_msg.setText(result.error)
            return

        self.view.set_images(result.before, result.after)
        self.save_btn.setEnabled(True)

        sev_color = {
            "ok": "#22c55e", "info": "#0ea5e9",
            "warn": "#eab308", "fail": "#ef4444",
        }.get(result.preflight_severity, "#94a3b8")
        self.preflight_label.setText(
            f"Preflight: <b style='color: {sev_color}'>"
            f"{result.preflight_severity.upper()}</b>"
        )
        self.preflight_label.setTextFormat(Qt.RichText)
        self.preflight_msg.setText(result.preflight_msg or "(không có warning)")

        scene_str = result.scene if result.scene else "—"
        self.scene_label.setText(f"Scene: <b>{scene_str}</b>")
        self.scene_label.setTextFormat(Qt.RichText)
        self.duration_label.setText(f"Duration: {result.duration_ms / 1000:.2f}s")

        # Append summary actions list ở cuối live log (đã có log per-stage rồi)
        if result.actions:
            self.actions_text.append("─" * 40)
            self.actions_text.append("Tổng kết:")
            for a in result.actions:
                self.actions_text.append(f"  • {a}")
            scroll = self.actions_text.verticalScrollBar()
            scroll.setValue(scroll.maximum())

    def _on_save(self):
        if self._result is None or self._result.after.size == 0:
            return
        from pps_core.utils import write_image
        default_name = (
            f"{self._current_path.stem}_after.jpg"
            if self._current_path else "after.jpg"
        )
        default_dir = str(self._current_path.parent if self._current_path else Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh AFTER",
            str(Path(default_dir) / default_name),
            "JPEG (*.jpg);;PNG (*.png);;WebP (*.webp)",
        )
        if not path:
            return
        try:
            write_image(
                Path(path), self._result.after, quality=95,
                exif_source=str(self._current_path) if self._current_path else None,
            )
            QMessageBox.information(self, "Đã lưu", f"✅ {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Lỗi lưu", str(exc))

    def _set_mode(self, mode: ViewMode):
        self.view.set_mode(mode)
