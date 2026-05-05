"""Tab Thủ công — Pro Develop sliders Lightroom-style.

Single-image preview + slider tone curve / WB / detail / lens.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QFileDialog, QComboBox, QScrollArea, QSpinBox,
)


class LabeledSlider(QWidget):
    """Slider có label + value display."""

    value_changed = Signal(float)

    def __init__(self, label: str, minimum: float = -1.0, maximum: float = 1.0,
                 default: float = 0.0, decimals: int = 2, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._decimals = decimals
        self._scale = 100  # internal int scale

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.label = QLabel(label)
        self.label.setMinimumWidth(180)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum * self._scale), int(maximum * self._scale))
        self.slider.setValue(int(default * self._scale))
        self.slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self.slider, 1)

        self.value_label = QLabel(f"{default:+.{decimals}f}")
        self.value_label.setMinimumWidth(50)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.value_label)

        self.reset_btn = QPushButton("↺")
        self.reset_btn.setFixedSize(24, 24)
        self.reset_btn.setToolTip("Reset về 0")
        self._default = default
        self.reset_btn.clicked.connect(lambda: self.set_value(self._default))
        layout.addWidget(self.reset_btn)

    def _on_changed(self, v: int):
        f = v / self._scale
        self.value_label.setText(f"{f:+.{self._decimals}f}")
        self.value_changed.emit(f)

    def value(self) -> float:
        return self.slider.value() / self._scale

    def set_value(self, v: float):
        self.slider.setValue(int(v * self._scale))


class ManualTab(QWidget):
    """Tab Thủ công — preview 1 ảnh, chỉnh slider real-time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: np.ndarray | None = None
        self._current_path: Path | None = None
        self._preview_pixmap: QPixmap | None = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ====== LEFT: Preview ======
        left_col = QVBoxLayout()

        load_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Mở ảnh để chỉnh")
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.clicked.connect(self._on_load_image)
        load_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("💾 Lưu ảnh đã chỉnh")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_image)
        load_layout.addWidget(self.save_btn)

        load_layout.addStretch()
        left_col.addLayout(load_layout)

        self.preview_label = QLabel("Chưa có ảnh — bấm 'Mở ảnh' để bắt đầu")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet(
            "background-color: #0f1419; border: 1px solid #3a4154; "
            "border-radius: 6px; color: #64748b;"
        )
        left_col.addWidget(self.preview_label, 1)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        left_col.addWidget(self.info_label)

        main_layout.addLayout(left_col, 2)

        # ====== RIGHT: Sliders ======
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setSpacing(12)

        # --- Group: WB ---
        wb_group = QGroupBox("🌡 Cân bằng trắng")
        wb_layout = QVBoxLayout(wb_group)
        self.s_temp = LabeledSlider("Temperature", -1.0, 1.0, 0.0)
        self.s_tint = LabeledSlider("Tint", -1.0, 1.0, 0.0)
        wb_layout.addWidget(self.s_temp)
        wb_layout.addWidget(self.s_tint)
        sliders_layout.addWidget(wb_group)

        # --- Group: Tone curve ---
        tone_group = QGroupBox("🎚 Đường cong tông màu")
        tone_layout = QVBoxLayout(tone_group)
        self.s_exposure = LabeledSlider("Phơi sáng", -1.0, 1.0, 0.0)
        self.s_contrast = LabeledSlider("Tương phản", -1.0, 1.0, 0.0)
        self.s_highlights = LabeledSlider("Vùng sáng", -1.0, 1.0, 0.0)
        self.s_shadows = LabeledSlider("Vùng tối", -1.0, 1.0, 0.0)
        self.s_whites = LabeledSlider("Trắng", -1.0, 1.0, 0.0)
        self.s_blacks = LabeledSlider("Đen", -1.0, 1.0, 0.0)
        for s in (self.s_exposure, self.s_contrast, self.s_highlights,
                   self.s_shadows, self.s_whites, self.s_blacks):
            tone_layout.addWidget(s)
        sliders_layout.addWidget(tone_group)

        # --- Group: Detail ---
        det_group = QGroupBox("✨ Chi tiết")
        det_layout = QVBoxLayout(det_group)
        self.s_texture = LabeledSlider("Texture", -1.0, 1.0, 0.0)
        self.s_clarity = LabeledSlider("Clarity", -1.0, 1.0, 0.0)
        self.s_dehaze = LabeledSlider("Dehaze", 0.0, 1.0, 0.0)
        for s in (self.s_texture, self.s_clarity, self.s_dehaze):
            det_layout.addWidget(s)
        sliders_layout.addWidget(det_group)

        # --- Group: Lens ---
        lens_group = QGroupBox("📸 Hiệu chỉnh ống kính")
        lens_layout = QVBoxLayout(lens_group)
        self.lens_combo = QComboBox()
        self.lens_combo.addItem("(Không áp lens)", "")
        try:
            from pps_core.lens import list_lens_profiles
            for p in list_lens_profiles():
                self.lens_combo.addItem(f"{p['name']}", p['id'])
        except Exception:
            pass
        lens_layout.addWidget(self.lens_combo)
        self.s_lens_intensity = LabeledSlider("Cường độ", 0.0, 1.0, 1.0)
        lens_layout.addWidget(self.s_lens_intensity)
        sliders_layout.addWidget(lens_group)

        # --- Apply button ---
        self.apply_btn = QPushButton("🎚 Áp dụng & Xem trước")
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_preview)
        sliders_layout.addWidget(self.apply_btn)

        # Reset all
        reset_btn = QPushButton("↺ Reset tất cả slider")
        reset_btn.clicked.connect(self._reset_all)
        sliders_layout.addWidget(reset_btn)

        sliders_layout.addStretch()
        scroll.setWidget(sliders_widget)
        main_layout.addWidget(scroll)

    def _on_load_image(self):
        from pps_core.utils import read_image
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh để chỉnh", str(Path.home()),
            "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Lỗi đọc ảnh", str(exc))
            return
        self._current_image = img
        self._current_path = Path(path)
        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.info_label.setText(
            f"📷 {self._current_path.name} — {img.shape[1]}×{img.shape[0]}"
        )
        self._show_image(img)

    def _show_image(self, bgr: np.ndarray):
        """Hiện ảnh BGR vào preview_label."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Scale fit preview area
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        # Force contiguous để Qt không corrupt
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_c.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Sao chép để tránh ref tới numpy buffer bị GC
        qimg = qimg.copy()
        pixmap = QPixmap.fromImage(qimg)
        self._preview_pixmap = pixmap
        self.preview_label.setPixmap(pixmap)

    def _apply_preview(self):
        if self._current_image is None:
            return
        from pps_core.tone import ToneParams, apply_tone_full
        from pps_core.lens import auto_correct_lens

        img = self._current_image.copy()

        # 1. Lens correction trước
        profile_id = self.lens_combo.currentData()
        if profile_id:
            img, _ = auto_correct_lens(
                img, profile_id=profile_id,
                intensity=self.s_lens_intensity.value(),
            )

        # 2. Apply tone full
        params = ToneParams(
            temp_shift=self.s_temp.value(),
            tint_shift=self.s_tint.value(),
            exposure=self.s_exposure.value(),
            contrast=self.s_contrast.value(),
            highlights=self.s_highlights.value(),
            lights=self.s_highlights.value() * 0.5,
            shadows=self.s_shadows.value(),
            darks=self.s_shadows.value() * 0.5,
            whites=self.s_whites.value(),
            blacks=self.s_blacks.value(),
            texture=self.s_texture.value(),
            clarity=self.s_clarity.value(),
            dehaze=self.s_dehaze.value(),
        )
        img = apply_tone_full(img, params)
        self._show_image(img)
        self._current_processed = img

    def _on_save_image(self):
        if self._current_image is None:
            return
        from pps_core.utils import write_image
        if not hasattr(self, "_current_processed"):
            self._apply_preview()
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh đã chỉnh",
            str(self._current_path.parent / f"{self._current_path.stem}_edited.jpg"),
            "JPEG (*.jpg);;PNG (*.png);;WebP (*.webp)",
        )
        if not path:
            return
        try:
            write_image(path, self._current_processed, quality=95,
                          exif_source=str(self._current_path))
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Đã lưu",
                f"✅ Đã lưu: {path}",
            )
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Lỗi lưu", str(exc))

    def _reset_all(self):
        for s in (self.s_temp, self.s_tint, self.s_exposure, self.s_contrast,
                   self.s_highlights, self.s_shadows, self.s_whites,
                   self.s_blacks, self.s_texture, self.s_clarity,
                   self.s_dehaze, self.s_lens_intensity):
            s.set_value(0.0 if s != self.s_lens_intensity else 1.0)
        self.lens_combo.setCurrentIndex(0)
        if self._current_image is not None:
            self._show_image(self._current_image)
