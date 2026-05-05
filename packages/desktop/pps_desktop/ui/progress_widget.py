"""Progress widget — hiển thị tiến độ batch + log live + cancel button."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPlainTextEdit, QPushButton,
)


class ProgressWidget(QWidget):
    """Progress bar + log + cancel button cho batch."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header với current file + cancel
        header = QHBoxLayout()
        self.current_label = QLabel("Sẵn sàng")
        self.current_label.setStyleSheet(
            "color: #93c5fd; font-weight: bold; font-size: 13px;"
        )
        header.addWidget(self.current_label, 1)

        self.cancel_btn = QPushButton("✕ Huỷ")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        header.addWidget(self.cancel_btn)
        layout.addLayout(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%  (%v / %m ảnh)")
        layout.addWidget(self.progress_bar)

        # Stats label (OK / Fail / Còn)
        self.stats_label = QLabel("Đã xong: 0 | Lỗi: 0 | Còn: 0")
        self.stats_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.stats_label)

        # Log area — compact mặc định, expand khi batch chạy nhiều log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)  # giới hạn để không OOM
        self.log.setMinimumHeight(90)
        self.log.setMaximumHeight(220)
        layout.addWidget(self.log, 0)

    def _on_cancel(self):
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("⏸ Đang huỷ...")
        self.cancel_requested.emit()

    def reset(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        self.current_label.setText("Sẵn sàng")
        self.stats_label.setText("Đã xong: 0 | Lỗi: 0 | Còn: 0")
        self.log.clear()
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("✕ Huỷ")

    def start(self, total: int):
        self.reset()
        self.progress_bar.setRange(0, total)
        self.cancel_btn.setVisible(True)
        self.append_log(f"▶ Bắt đầu xử lý {total} ảnh\n" + "─" * 50)

    def update_progress(self, current: int, total: int, filename: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.current_label.setText(f"Đang xử lý: {filename} ({current}/{total})")

    def update_stats(self, ok: int, fail: int, total: int):
        remaining = max(0, total - ok - fail)
        self.stats_label.setText(
            f"Đã xong: <span style='color:#10b981'>{ok}</span> | "
            f"Lỗi: <span style='color:#ef4444'>{fail}</span> | "
            f"Còn: {remaining}"
        )

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        # Auto-scroll
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def finish(self, summary: str):
        self.cancel_btn.setVisible(False)
        self.current_label.setText("✅ Hoàn tất")
        self.append_log("\n" + "═" * 50)
        self.append_log(summary)
