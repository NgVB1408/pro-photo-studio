"""Tab Tự động — folder pickers + minimal output controls + big run button.

UI design: clean, spacious, pro. Tất cả AI feature đã chuyển sang sidebar phải.
Tab này chỉ giữ:
  - Input/Output folders
  - Output format + quality + size
  - Recursive + skip-existing toggles
  - Big run button

V5 toggles (Sky, Perspective, Lens, Privacy, TV, Fire, Photog) ở AI panel phải.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QStandardPaths
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QComboBox, QSlider, QGroupBox, QFileDialog,
    QSizePolicy, QSpacerItem,
)


def _default_browse_dir() -> str:
    """Tìm folder mặc định cho file dialog.

    Claude Terminal override USERPROFILE/HOME → QStandardPaths sai. Ưu tiên đọc
    Windows registry HKEY_CURRENT_USER User Shell Folders để lấy Pictures
    chính xác. Nếu path không tồn tại, fallback sang `C:\\Users\\<USERNAME>`.
    """
    import os
    import sys

    # Path 1 (Windows): đọc registry — chính xác nhất
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
            )
            for reg_name in ("My Pictures", "Personal", "Desktop"):
                try:
                    val, _ = winreg.QueryValueEx(key, reg_name)
                    expanded = os.path.expandvars(val)
                    if expanded and Path(expanded).is_dir():
                        return expanded
                except FileNotFoundError:
                    continue
        except Exception:
            pass

    # Path 2: build path từ USERNAME (env vẫn đúng dù USERPROFILE bị override)
    username = os.environ.get("USERNAME") or os.environ.get("USER")
    if username and sys.platform == "win32":
        for sub in ("Pictures", "Documents", "Desktop", ""):
            cand = Path(f"C:/Users/{username}") / sub if sub else Path(f"C:/Users/{username}")
            if cand.is_dir():
                return str(cand)

    # Path 3: fallback Qt standard paths (last resort)
    for loc in (
        QStandardPaths.PicturesLocation,
        QStandardPaths.DocumentsLocation,
        QStandardPaths.DesktopLocation,
    ):
        paths = QStandardPaths.standardLocations(loc)
        for p in paths:
            if p and Path(p).is_dir():
                return p

    # Path 4: hardcoded fallback chain
    for c in ("C:/Users", "C:/", "/"):
        if Path(c).is_dir():
            return c
    return ""

from ..workers.batch_worker import BatchJob, OUTPUT_SIZE_MAP


class FolderPicker(QWidget):
    """LineEdit + nút CHỌN cho folder picker (compat — chỉ folder)."""

    path_changed = Signal(str)

    def __init__(self, placeholder: str = "Chọn thư mục...", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.line = QLineEdit()
        self.line.setPlaceholderText(placeholder)
        self.line.setMinimumHeight(40)
        self.line.textChanged.connect(self.path_changed.emit)
        layout.addWidget(self.line, 1)

        self.btn = QPushButton("CHỌN")
        self.btn.setObjectName("folder_picker_btn")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setMinimumHeight(40)
        self.btn.setMinimumWidth(140)
        self.btn.setToolTip("Mở dialog chọn thư mục")
        self.btn.clicked.connect(self._pick)
        layout.addWidget(self.btn)

    def _pick(self):
        current = self.line.text().strip()
        if not current or not Path(current).is_dir():
            current = _default_browse_dir()
        d = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục", current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if d:
            self.line.setText(d)

    def path(self) -> str:
        return self.line.text().strip()

    def set_path(self, p: str):
        self.line.setText(p)


class FlexInputPicker(QWidget):
    """Chọn FOLDER hoặc nhiều FILE riêng lẻ.

    Mode = "folder" (default): chọn 1 thư mục, xử lý tất cả ảnh trong đó.
    Mode = "files": chọn N file riêng lẻ (multi-select dialog).
    """

    selection_changed = Signal()  # emit khi user thay đổi paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode: str = "folder"
        self._paths: list[Path] = []  # khi mode=files, lưu list file
        self._folder: str = ""        # khi mode=folder, lưu path

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        self.rb_folder = QRadioButton("📁  Cả thư mục")
        self.rb_folder.setChecked(True)
        self.rb_files = QRadioButton("📷  Chọn từng file (nhiều file OK)")
        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.rb_folder)
        self.btn_group.addButton(self.rb_files)
        self.rb_folder.toggled.connect(self._on_mode_change)
        self.rb_files.toggled.connect(self._on_mode_change)
        mode_row.addWidget(self.rb_folder)
        mode_row.addWidget(self.rb_files)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.line = QLineEdit()
        self.line.setPlaceholderText("Chưa chọn — bấm nút bên phải")
        self.line.setMinimumHeight(40)
        self.line.setReadOnly(True)
        input_row.addWidget(self.line, 1)
        self.btn = QPushButton("CHỌN")
        self.btn.setObjectName("folder_picker_btn")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setMinimumHeight(40)
        self.btn.setMinimumWidth(140)
        self.btn.clicked.connect(self._pick)
        input_row.addWidget(self.btn)
        outer.addLayout(input_row)

    def _on_mode_change(self):
        if self.rb_folder.isChecked():
            self._mode = "folder"
            self.line.setPlaceholderText("Chọn thư mục chứa ảnh — bấm nút bên phải")
        else:
            self._mode = "files"
            self.line.setPlaceholderText("Chọn 1 hoặc nhiều file ảnh — bấm nút bên phải")
        self.line.setText("")
        self._paths = []
        self._folder = ""
        self.selection_changed.emit()

    def _pick(self):
        if self._mode == "folder":
            current = self._folder or _default_browse_dir()
            d = QFileDialog.getExistingDirectory(
                self, "Chọn thư mục ảnh", current,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
            )
            if d:
                self._folder = d
                self.line.setText(d)
                self.selection_changed.emit()
        else:
            current = (self._paths[0].parent if self._paths else Path(_default_browse_dir())).as_posix()
            files, _ = QFileDialog.getOpenFileNames(
                self, "Chọn 1 hoặc nhiều ảnh", current,
                "Ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff "
                "*.dng *.cr2 *.cr3 *.nef *.nrw *.arw *.raf *.rw2 *.orf *.pef *.srw)"
                ";;JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff)"
                ";;RAW (*.dng *.cr2 *.cr3 *.nef *.arw *.raf *.rw2 *.orf)"
                ";;Tất cả (*.*)",
            )
            if files:
                self._paths = [Path(f) for f in files]
                self.line.setText(f"{len(self._paths)} file đã chọn — {self._paths[0].name}{' …' if len(self._paths)>1 else ''}")
                self.selection_changed.emit()

    def mode(self) -> str:
        return self._mode

    def folder(self) -> str:
        return self._folder

    def files(self) -> list[Path]:
        return list(self._paths)

    def first_path(self) -> Path | None:
        """Trả 1 path bất kỳ (folder hoặc parent của file đầu) để derive output default."""
        if self._mode == "folder" and self._folder:
            return Path(self._folder)
        if self._paths:
            return self._paths[0].parent
        return None


class OptionalFolderPicker(QWidget):
    """Output picker — checkbox 'Cùng folder input' + line/button khi unchecked."""

    path_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.cb_same = QCheckBox(
            "📂  Cùng thư mục input + hậu tố `_enhanced` (tự động, không cần chọn)"
        )
        self.cb_same.setChecked(True)
        self.cb_same.setMinimumHeight(28)
        self.cb_same.toggled.connect(self._on_toggle)
        outer.addWidget(self.cb_same)

        # Custom path row (hidden by default)
        custom_row = QWidget()
        cr_layout = QHBoxLayout(custom_row)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_layout.setSpacing(8)
        self.line = QLineEdit()
        self.line.setPlaceholderText("Chọn thư mục xuất riêng — bấm nút bên phải")
        self.line.setMinimumHeight(40)
        self.line.setReadOnly(True)
        cr_layout.addWidget(self.line, 1)
        self.btn = QPushButton("CHỌN")
        self.btn.setObjectName("folder_picker_btn")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setMinimumHeight(40)
        self.btn.setMinimumWidth(140)
        self.btn.clicked.connect(self._pick)
        cr_layout.addWidget(self.btn)
        outer.addWidget(custom_row)
        self._custom_row = custom_row
        self._custom_row.setVisible(False)

    def _on_toggle(self, checked: bool):
        self._custom_row.setVisible(not checked)
        self.path_changed.emit()

    def _pick(self):
        current = self.line.text().strip() or _default_browse_dir()
        d = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục xuất", current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if d:
            self.line.setText(d)
            self.path_changed.emit()

    def use_input_folder(self) -> bool:
        return self.cb_same.isChecked()

    def custom_path(self) -> str:
        return self.line.text().strip()

    def resolve(self, input_first_path: Path | None) -> Path | None:
        """Resolve final output path dựa trên checkbox + input."""
        if self.use_input_folder():
            if input_first_path is None:
                return None
            return input_first_path / "enhanced"
        custom = self.custom_path()
        if custom and Path(custom).parent.exists():
            return Path(custom)
        return None


class AutoTab(QWidget):
    """Tab Tự động — single click batch processing."""

    start_requested = Signal(BatchJob)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # ===== Group 1: Input + Output (linh hoạt) =====
        io_group = QGroupBox("📂  ẢNH ĐẦU VÀO + ĐẦU RA")
        io_group.setMinimumHeight(340)
        io_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        io_layout = QVBoxLayout(io_group)
        io_layout.setContentsMargins(20, 36, 20, 20)
        io_layout.setSpacing(10)

        # Input section
        in_label = QLabel("🖼  Ảnh gốc (chọn thư mục HOẶC từng file):")
        in_label.setStyleSheet("font-weight: 700; color: #93c5fd; font-size: 13px;")
        io_layout.addWidget(in_label)
        self.input_picker = FlexInputPicker()
        self.input_picker.selection_changed.connect(self._on_input_changed)
        io_layout.addWidget(self.input_picker)

        io_layout.addSpacing(8)

        # Output section
        out_label = QLabel("💾  Nơi lưu ảnh đã xử lý:")
        out_label.setStyleSheet("font-weight: 700; color: #93c5fd; font-size: 13px;")
        io_layout.addWidget(out_label)
        self.output_picker = OptionalFolderPicker()
        io_layout.addWidget(self.output_picker)

        layout.addWidget(io_group)

        # ===== Group 2: Output format =====
        format_group = QGroupBox("⚙  ĐỊNH DẠNG ĐẦU RA")
        format_group.setMinimumHeight(160)
        format_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        fmt_layout = QGridLayout(format_group)
        fmt_layout.setContentsMargins(20, 32, 20, 18)
        fmt_layout.setHorizontalSpacing(16)
        fmt_layout.setVerticalSpacing(14)
        fmt_layout.setColumnStretch(1, 1)
        fmt_layout.setColumnStretch(3, 2)

        fmt_label = QLabel("Định dạng:")
        fmt_label.setMinimumWidth(80)
        fmt_layout.addWidget(fmt_label, 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png", "webp"])
        self.format_combo.setMinimumHeight(36)
        self.format_combo.setMinimumWidth(120)
        fmt_layout.addWidget(self.format_combo, 0, 1)

        qual_label = QLabel("Chất lượng:")
        fmt_layout.addWidget(qual_label, 0, 2)
        qual_box = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(60, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.setTickInterval(5)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_slider.setMinimumHeight(28)
        self.quality_label = QLabel("95")
        self.quality_label.setMinimumWidth(40)
        self.quality_label.setStyleSheet("font-weight: 700; color: #93c5fd; font-size: 14px;")
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(str(v))
        )
        qual_box.addWidget(self.quality_slider, 1)
        qual_box.addWidget(self.quality_label)
        fmt_layout.addLayout(qual_box, 0, 3)

        size_label = QLabel("Kích thước:")
        fmt_layout.addWidget(size_label, 1, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(OUTPUT_SIZE_MAP.keys())[:4])
        self.size_combo.setMinimumHeight(36)
        fmt_layout.addWidget(self.size_combo, 1, 1, 1, 3)

        layout.addWidget(format_group)

        # ===== Group 3: Batch options (recursive, skip, denoise) =====
        opt_group = QGroupBox("🔧  TUỲ CHỌN BATCH")
        opt_group.setMinimumHeight(400)
        opt_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        opt_layout = QGridLayout(opt_group)
        opt_layout.setContentsMargins(20, 32, 20, 18)
        opt_layout.setHorizontalSpacing(20)
        opt_layout.setVerticalSpacing(14)

        self.cb_recursive = QCheckBox("📂  Quét cả thư mục con (subfolders)")
        self.cb_recursive.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_recursive, 0, 0)

        self.cb_skip_existing = QCheckBox("⏭  Bỏ qua file đã tồn tại")
        self.cb_skip_existing.setChecked(True)
        self.cb_skip_existing.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_skip_existing, 0, 1)

        self.cb_denoise = QCheckBox("🔇  Khử Noise (bilateral filter)")
        # OFF default — combine với sharpen sẽ tạo plastic look
        self.cb_denoise.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_denoise, 1, 0)

        self.cb_keep_size = QCheckBox("📐  Giữ size gốc (override 'Kích thước' trên)")
        self.cb_keep_size.setChecked(True)
        self.cb_keep_size.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_keep_size, 1, 1)

        # HDR Bracket mode — Off / Auto / 3 / 5 / 7 exposures
        hdr_row = QWidget()
        hdr_l = QHBoxLayout(hdr_row)
        hdr_l.setContentsMargins(0, 0, 0, 0)
        hdr_l.setSpacing(8)
        hdr_lbl = QLabel("🌆  HDR Bracket:")
        hdr_lbl.setStyleSheet("font-weight: 600;")
        hdr_l.addWidget(hdr_lbl)
        self.combo_bracket = QComboBox()
        self.combo_bracket.addItems(["Tắt (ảnh đơn)", "Tự động", "3 ảnh/cảnh", "5 ảnh/cảnh", "7 ảnh/cảnh"])
        self.combo_bracket.setCurrentIndex(1)  # Auto default
        self.combo_bracket.setMinimumHeight(36)
        self.combo_bracket.setMinimumWidth(160)
        self.combo_bracket.setToolTip(
            "Off: xử lý từng ảnh độc lập\n"
            "Auto: tự gom 2-5 ảnh cùng burst (EV khác nhau) → fuse Mertens HDR\n"
            "3/5/7: ép gom đúng N ảnh/scene (giống realtyedit chọn 3/5/7 exp)"
        )
        hdr_l.addWidget(self.combo_bracket, 1)
        opt_layout.addWidget(hdr_row, 2, 0, 1, 2)

        # Vertical alignment toggle (giống "Tích gióng/không gióng thẳng")
        self.cb_vertical = QCheckBox("📐  Gióng thẳng đường dọc (vertical alignment)")
        self.cb_vertical.setChecked(True)
        self.cb_vertical.setToolTip(
            "Tự kéo thẳng đường dọc (cột, khung cửa, tường) khi ảnh nghiêng phối cảnh.\n"
            "Bỏ tick nếu muốn giữ nguyên góc chụp gốc."
        )
        self.cb_vertical.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_vertical, 3, 0)

        # Sky preset combo — clone từ AI panel, ngay tay
        sky_row = QWidget()
        sky_l = QHBoxLayout(sky_row)
        sky_l.setContentsMargins(0, 0, 0, 0)
        sky_l.setSpacing(8)
        sky_lbl = QLabel("☁  Thay trời:")
        sky_lbl.setStyleSheet("font-weight: 600;")
        sky_l.addWidget(sky_lbl)
        self.combo_sky = QComboBox()
        # (label, preset_key)
        self._sky_options = [
            ("Tắt", ""),
            ("Blue Sky + Clouds", "blue_clouds"),
            ("Blue Clear", "blue_clear"),
            ("Sunset Warm", "sunset_warm"),
            ("Golden Hour", "golden_hour"),
            ("Dramatic Storm", "dramatic_storm"),
            ("Overcast Soft", "overcast_soft"),
            ("Twilight Blue", "twilight_blue"),
        ]
        for label, _ in self._sky_options:
            self.combo_sky.addItem(label)
        self.combo_sky.setCurrentIndex(0)  # Off default — sky thay đổi nội dung, để user opt-in
        self.combo_sky.setMinimumHeight(36)
        self.combo_sky.setMinimumWidth(160)
        self.combo_sky.setToolTip(
            "Thay trời cho ảnh ngoại thất (giống realtyedit chọn sky).\n"
            "AI sky segmentation tự detect vùng trời + composite preset mới."
        )
        sky_l.addWidget(self.combo_sky, 1)
        opt_layout.addWidget(sky_row, 3, 1)

        self.cb_realestate = QCheckBox(
            "🏡  Pipeline BĐS đầy đủ (auto window pull / lawn / classify)"
        )
        self.cb_realestate.setChecked(True)
        self.cb_realestate.setToolTip(
            "Bật flow Real Estate: tự classify scene → window pull "
            "indoor → enhance lawn exterior. Vertical + Sky tách thành option riêng ở trên."
        )
        self.cb_realestate.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_realestate, 4, 0, 1, 2)

        self.cb_review = QCheckBox("🧾  Report CSV + contact sheet trước/sau")
        self.cb_review.setChecked(True)
        self.cb_review.setToolTip(
            "Sau khi batch xong, app tự xuất:\n"
            " • processing_report.csv — scene, action, time, settings\n"
            " • _review/before_after_contact_sheet.jpg — review nhanh"
        )
        self.cb_review.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_review, 5, 0)

        self.cb_preflight = QCheckBox(
            "🩺  Pre-flight QC (blur / exposure / dimension)"
        )
        self.cb_preflight.setChecked(True)
        self.cb_preflight.setToolTip(
            "Trước khi xử lý, app phân tích ảnh và cảnh báo nếu blur, cháy ≥18%, "
            "hoặc resolution thấp. Vẫn xử lý — chỉ ghi vào CSV để photographer "
            "biết retake."
        )
        self.cb_preflight.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_preflight, 5, 1)

        self.cb_ai_sky = QCheckBox(
            "🤖  AI Sky segmentation (rembg, fallback heuristic)"
        )
        self.cb_ai_sky.setChecked(True)
        self.cb_ai_sky.setToolTip(
            "Dùng rembg + onnxruntime để segment sky chính xác hơn HSV thuần "
            "(xử lý cây/skyline phức tạp). Cần `pip install -e .[sky-ai]`. "
            "Nếu chưa cài → tự fallback HSV heuristic."
        )
        self.cb_ai_sky.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_ai_sky, 6, 0)

        self.cb_raw = QCheckBox(
            "📸  Nhận RAW input (.dng/.cr2/.nef/.arw/.raf/...)"
        )
        self.cb_raw.setChecked(True)
        self.cb_raw.setToolTip(
            "Bật để worker scan + demosaic file RAW (cần `pip install rawpy`). "
            "Ảnh JPG/PNG vẫn xử lý bình thường khi tắt option này."
        )
        self.cb_raw.setMinimumHeight(28)
        opt_layout.addWidget(self.cb_raw, 6, 1)

        # Backward compat — code phía dưới còn ref cb_auto_hdr (chỉ on/off)
        # Map từ combo_bracket
        self.cb_auto_hdr = QCheckBox()
        self.cb_auto_hdr.setChecked(True)
        self.cb_auto_hdr.hide()

        # Hidden — controlled bởi AI panel + sidebar
        # Đặt default cho compatibility
        self.cb_color_enhance = QCheckBox()
        self.cb_color_enhance.setChecked(True)
        self.cb_color_enhance.hide()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["studio", "real_estate", "portrait", "product", "outdoor"])
        self.preset_combo.setCurrentText("real_estate")
        self.preset_combo.hide()
        self.cb_detail = QCheckBox()
        # OFF default — pseudo_HDR gây "HDR look" giả, halo, crushed highlights
        self.cb_detail.hide()

        layout.addWidget(opt_group)

        layout.addStretch(1)

        # ===== Big run button =====
        self.run_btn = QPushButton("⚡  XỬ LÝ HÀNG LOẠT")
        self.run_btn.setObjectName("primary_btn")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(58)
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

    def _on_input_changed(self):
        # Nếu user chưa override output folder, không cần làm gì — resolve tại runtime
        pass

    def _on_run_clicked(self):
        from PySide6.QtWidgets import QMessageBox

        # Resolve input — folder OR list of files
        if self.input_picker.mode() == "folder":
            in_dir_str = self.input_picker.folder()
            if not in_dir_str or not Path(in_dir_str).is_dir():
                QMessageBox.warning(self, "Thiếu input",
                    "Vui lòng chọn thư mục ảnh gốc (hoặc switch sang chế độ 'Chọn từng file').")
                return
            in_dir = Path(in_dir_str)
            input_files = None
        else:
            files = self.input_picker.files()
            if not files:
                QMessageBox.warning(self, "Thiếu input",
                    "Vui lòng chọn ít nhất 1 file ảnh.")
                return
            in_dir = files[0].parent  # dummy folder
            input_files = files

        # Resolve output (optional, default = input/enhanced)
        first = self.input_picker.first_path()
        out_dir = self.output_picker.resolve(first)
        if out_dir is None:
            QMessageBox.warning(self, "Output không hợp lệ",
                "Bỏ tick 'Cùng thư mục input' rồi chọn folder xuất, "
                "hoặc đảm bảo folder input đã chọn.")
            return

        size_label = self.size_combo.currentText()
        if self.cb_keep_size.isChecked():
            size_label = "Giữ nguyên (chất lượng tối đa)"

        # Resolve HDR bracket mode
        bracket_idx = self.combo_bracket.currentIndex()
        # 0=Off, 1=Auto, 2=3 exp, 3=5 exp, 4=7 exp
        if bracket_idx == 0:
            auto_hdr = False
            bracket_size = None
        elif bracket_idx == 1:
            auto_hdr = True
            bracket_size = None
        else:
            auto_hdr = True
            bracket_size = {2: 3, 3: 5, 4: 7}[bracket_idx]

        # Resolve Sky preset
        sky_idx = self.combo_sky.currentIndex()
        sky_label, sky_key = self._sky_options[sky_idx]
        enable_sky = bool(sky_key)
        sky_preset = sky_key or "blue_clouds"

        # AI panel sẽ override các option này khi MainWindow gọi apply_to_job()
        job = BatchJob(
            input_dir=in_dir,
            output_dir=out_dir,
            output_format=self.format_combo.currentText(),
            output_quality=self.quality_slider.value(),
            output_size_label=size_label,
            denoise=self.cb_denoise.isChecked(),
            keep_size=self.cb_keep_size.isChecked(),
            detail_recovery=self.cb_detail.isChecked(),
            color_enhance=self.cb_color_enhance.isChecked(),
            enhance_preset=self.preset_combo.currentText(),
            realestate_pipeline=self.cb_realestate.isChecked(),
            enable_sky_replace=enable_sky,
            sky_preset=sky_preset,
            vertical_align=self.cb_vertical.isChecked(),
            auto_hdr_merge=auto_hdr,
            hdr_bracket_size=bracket_size,
            review_contact_sheet=self.cb_review.isChecked(),
            write_processing_report=self.cb_review.isChecked(),
            preflight_check=self.cb_preflight.isChecked(),
            use_ai_sky=self.cb_ai_sky.isChecked(),
            accept_raw=self.cb_raw.isChecked(),
            recursive=self.cb_recursive.isChecked(),
            skip_existing=self.cb_skip_existing.isChecked(),
        )
        # Attach explicit file list (nếu mode=files)
        if input_files is not None:
            job._explicit_files = input_files  # type: ignore[attr-defined]
        self.start_requested.emit(job)

    def set_running(self, running: bool):
        """Disable inputs khi đang xử lý."""
        self.run_btn.setEnabled(not running)
        self.input_picker.setEnabled(not running)
        self.output_picker.setEnabled(not running)
        self.format_combo.setEnabled(not running)
        self.quality_slider.setEnabled(not running)
        self.size_combo.setEnabled(not running)
        self.cb_auto_hdr.setEnabled(not running)
        self.cb_review.setEnabled(not running)
        self.cb_realestate.setEnabled(not running)
        self.cb_preflight.setEnabled(not running)
        self.cb_ai_sky.setEnabled(not running)
        self.cb_raw.setEnabled(not running)
        if running:
            self.run_btn.setText("⏳  ĐANG XỬ LÝ...")
        else:
            self.run_btn.setText("⚡  XỬ LÝ HÀNG LOẠT")
