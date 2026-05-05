"""MainWindow — Pro Photo Studio Tool desktop app.

Layout:
- Top: License bar
- Center: Tab Auto / Manual / Settings
- Bottom: Progress + log
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStatusBar,
    QMessageBox, QSplitter, QScrollArea,
)

from ..license import client as lic
from ..workers.batch_worker import BatchWorker, BatchJob, BatchStats
from .license_widget import LicenseBar, ActivateDialog
from .auto_tab import AutoTab
from .manual_tab import ManualTab
from .compare_tab import CompareTab
from .progress_widget import ProgressWidget
from .ai_settings_panel import AISettingsPanel, AISettings

logger = logging.getLogger(__name__)

APP_NAME = "Pro Photo Studio"
APP_VERSION = "0.1.0-mvp"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._license: lic.License | None = None
        self._worker: BatchWorker | None = None
        self._stats_running = {"ok": 0, "fail": 0, "total": 0}

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1440, 880)
        self.resize(1680, 1000)

        # Load license trước khi build UI
        self._license = lic.load_license()
        if self._license is None or not lic.is_machine_match(self._license):
            self._prompt_activate()
            if self._license is None:
                # User cancel → close app
                import sys
                sys.exit(0)

        self._build_ui()

    def _prompt_activate(self):
        dialog = ActivateDialog(self)
        if dialog.exec():
            self._license = dialog.license
            QMessageBox.information(
                self, "Kích hoạt thành công",
                f"✅ Đã kích hoạt cho gói {self._license.tier_label()}\n"
                f"Khách hàng: {self._license.customer_name}\n"
                f"Hạn: {self._license.expiry_label()}",
            )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # === License bar ===
        self.license_bar = LicenseBar(self._license)
        self.license_bar.logout_requested.connect(self._on_logout)
        layout.addWidget(self.license_bar)

        # === Body: AI panel (left) + tabs (right) ===
        body = QHBoxLayout()
        body.setSpacing(12)

        # AI Settings sidebar (LEFT — narrow vertical panel để ảnh chiếm hết phải)
        self.ai_panel = AISettingsPanel()
        self.ai_panel.settings_changed.connect(self._on_ai_settings_changed)
        self.ai_panel.max_preset_requested.connect(self._on_max_preset)
        self.ai_panel.light_preset_requested.connect(self._on_light_preset)

        ai_scroll = QScrollArea()
        ai_scroll.setWidgetResizable(True)
        ai_scroll.setWidget(self.ai_panel)
        ai_scroll.setMinimumWidth(290)
        ai_scroll.setMaximumWidth(340)
        ai_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ai_scroll.setFrameShape(QScrollArea.NoFrame)
        body.addWidget(ai_scroll, 0)

        # Tabs (RIGHT — chiếm phần còn lại để ảnh compare to)
        self.tabs = QTabWidget()
        self.auto_tab = AutoTab()
        self.auto_tab.start_requested.connect(self._on_start_batch)
        self.tabs.addTab(self.auto_tab, "⚡ TỰ ĐỘNG")

        self.compare_tab = CompareTab()
        self.compare_tab.set_job_factory(self._build_compare_job)
        self.tabs.addTab(self.compare_tab, "🔀 SO SÁNH TRƯỚC / SAU")

        self.manual_tab = ManualTab()
        self.tabs.addTab(self.manual_tab, "🎚 THỦ CÔNG (Pro Develop)")

        # Hide progress widget khi user chuyển sang tab Compare/Manual (chỉ
        # relevant cho batch). Tab TỰ ĐỘNG sẽ giữ nó visible.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        body.addWidget(self.tabs, 1)

        layout.addLayout(body, 1)
        self._ai_settings = self.ai_panel.settings()

        # === Progress widget ===
        self.progress = ProgressWidget()
        self.progress.cancel_requested.connect(self._on_cancel_batch)
        layout.addWidget(self.progress)

        # === Status bar ===
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(f"{APP_NAME} v{APP_VERSION} · Đã kích hoạt · "
                       f"Sẵn sàng xử lý")

    def _on_logout(self):
        reply = QMessageBox.question(
            self, "Đăng xuất license",
            "Bạn có chắc muốn đăng xuất license khỏi máy này?\n"
            "(Có thể kích hoạt lại bằng key cũ.)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            lic.deactivate()
            QMessageBox.information(
                self, "Đã đăng xuất", "License đã xoá khỏi máy. "
                "Khởi động lại app để kích hoạt key mới.",
            )
            self.close()

    def _on_ai_settings_changed(self, settings: AISettings):
        self._ai_settings = settings

    def _on_tab_changed(self, idx: int):
        """Ẩn batch progress widget trên tab Compare/Manual để ảnh có thêm chỗ."""
        # Index 0 = TỰ ĐỘNG (auto), 1 = SO SÁNH, 2 = THỦ CÔNG
        is_auto = (idx == 0)
        running = self._worker is not None and self._worker.isRunning()
        self.progress.setVisible(is_auto or running)

    def _on_max_preset(self):
        """Bật mọi enhancement có lợi cho chất lượng (auto_tab side)."""
        at = self.auto_tab
        at.cb_denoise.setChecked(True)
        at.cb_detail.setChecked(True)
        at.cb_realestate.setChecked(True)
        at.cb_preflight.setChecked(True)
        at.cb_ai_sky.setChecked(True)
        at.cb_raw.setChecked(True)
        self.statusBar().showMessage(
            "🎯 Tối đa chi tiết: Detail + Denoise + Sharp + Upscale x2 + Tone auto",
            6000,
        )

    def _on_light_preset(self):
        at = self.auto_tab
        at.cb_denoise.setChecked(False)
        at.cb_detail.setChecked(False)
        at.cb_realestate.setChecked(True)
        at.cb_preflight.setChecked(True)
        at.cb_ai_sky.setChecked(True)
        at.cb_raw.setChecked(True)
        self.statusBar().showMessage("⚖ Cân bằng: pipeline cốt lõi, không upscale", 4000)

    def _build_compare_job(self) -> BatchJob:
        """Snapshot Auto tab + AI panel settings vào BatchJob để CompareTab dùng.

        Dummy input/output dirs vì compare chạy in-memory trên 1 ảnh, không
        ghi qua batch path.
        """
        # Lấy current state từ auto_tab (toggles + format) + AI panel
        at = self.auto_tab
        size_label = at.size_combo.currentText()
        if at.cb_keep_size.isChecked():
            size_label = "Giữ nguyên (chất lượng tối đa)"
        job = BatchJob(
            input_dir=Path("."),
            output_dir=Path("."),
            output_format=at.format_combo.currentText(),
            output_quality=at.quality_slider.value(),
            output_size_label=size_label,
            denoise=at.cb_denoise.isChecked(),
            keep_size=at.cb_keep_size.isChecked(),
            detail_recovery=at.cb_detail.isChecked(),
            color_enhance=at.cb_color_enhance.isChecked(),
            enhance_preset=at.preset_combo.currentText(),
            realestate_pipeline=at.cb_realestate.isChecked(),
            enable_sky_replace=False,
            auto_hdr_merge=False,           # single-image, không gom bracket
            review_contact_sheet=False,
            write_processing_report=False,
            preflight_check=at.cb_preflight.isChecked(),
            use_ai_sky=at.cb_ai_sky.isChecked(),
            accept_raw=at.cb_raw.isChecked(),
            recursive=False,
            skip_existing=False,
        )
        # AI panel sẽ overlay sky preset + perspective/lens/tone/...
        self.ai_panel.apply_to_job(job)
        return job

    def _on_start_batch(self, job: BatchJob):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(
                self, "Đang chạy",
                "Đang có job khác chạy. Vui lòng đợi hoặc huỷ trước.",
            )
            return

        # Apply AI panel settings vào job
        self.ai_panel.apply_to_job(job)

        self._stats_running = {"ok": 0, "fail": 0, "total": 0}
        # Quick scan để biết total
        self.progress.start(0)  # sẽ update khi worker emit progress
        self.auto_tab.set_running(True)

        self._worker = BatchWorker(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.log.connect(self.progress.append_log)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished_with_stats.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, filename: str):
        self.progress.update_progress(current, total, filename)
        self._stats_running["total"] = total

    def _on_file_done(self, filename: str, success: bool, dur_ms: float):
        if success:
            self._stats_running["ok"] += 1
        else:
            self._stats_running["fail"] += 1
        self.progress.update_stats(
            self._stats_running["ok"],
            self._stats_running["fail"],
            self._stats_running["total"],
        )

    def _on_error(self, msg: str):
        self.progress.append_log(f"❌ LỖI: {msg}")
        QMessageBox.warning(self, "Lỗi xử lý", msg)
        self.auto_tab.set_running(False)

    def _on_finished(self, stats: BatchStats):
        self.progress.finish(stats.summary())
        self.auto_tab.set_running(False)
        self.statusBar().showMessage(
            f"Đã xử lý {stats.ok} ảnh trong {stats.total_seconds:.1f}s "
            f"({stats.fail} lỗi)", 10_000,
        )

    def _on_cancel_batch(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.progress.append_log("⚠ Đã yêu cầu huỷ — đợi job hiện tại xong...")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "Đang xử lý",
                "Job đang chạy. Đóng app sẽ huỷ. Tiếp tục?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)
        event.accept()
