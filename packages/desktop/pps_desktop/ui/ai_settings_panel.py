"""AI Settings Panel — sidebar giống Autoenhance V5.

Replicate UX:
┌─────────────────────────────┐
│ ✨ Adjust enhancement       │
├─────────────────────────────┤
│ 🤖 AI Version       V5 ▼    │
│ ☁ Sky Replacement  Off ▼   │
│ ⊟ Perspective      On ▼    │
│ 🛡 Auto Privacy     Off ▼   │
│ 📷 Lens Correction Off ▼   │
│ 🌅 Window Pull     Auto ▼  │
│ 📺 TV Blackout      Off ▼   │
│ 🔥 Fire Fireplace  Off ▼   │
│ 👤 Photog Removal  Off ▼   │
│ 🎯 AI Upscale      Off ▼   │
└─────────────────────────────┘

Mỗi row = QHBoxLayout với icon + label + dropdown (QComboBox).
Phát signal `settings_changed` mỗi khi user thay đổi → MainWindow apply vào BatchJob.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox,
    QPushButton, QSizePolicy,
)


@dataclass
class AISettings:
    """Snapshot tất cả AI toggle để apply vào BatchJob.

    Default = "Cân bằng tự nhiên" (giống AutoEnhance.ai output thật):
      - Geometric correct (lens + perspective + vertical) ✓
      - Real Estate color pipeline (scene-aware wb+clarity+shadow_lift) ✓
      - Window pull cho cửa sổ cháy ✓
      - KHÔNG detail_recovery (pseudo_HDR gây HDR look giả)
      - KHÔNG selective_sharpen (làm texture phẳng kiểu plastic)
      - KHÔNG denoise (artifact khi combine với sharpen)
      - Tone auto_batch trong batch (subtle nudge, không bias)

    User muốn HDR mạnh tay thì bật thủ công các toggle.
    """
    ai_version: str = "V5"
    sky_replacement: str = "Off"
    perspective_correct: bool = True
    auto_privacy: bool = False
    lens_correct: bool = True
    window_pull: str = "Auto (Windows with Skies)"
    tv_blackout: bool = False
    fire_fireplace: bool = False
    photog_removal: bool = False
    ai_upscale_scale: int = 0
    enhance_preset: str = "real_estate"
    tone_preset: str = "auto_batch"
    selective_sharpen: bool = False     # OFF — gây over-sharp/flat
    virtual_twilight: bool = False      # Day → Sunset transform (opt-in)
    twilight_strength: float = 0.85
    hdr_deghost: bool = True             # Bracket fuse: skip ghost pixels
    hdr_color_normalize: bool = True     # Bracket fuse: LAB-match cross frames
    seed: int | None = None


SKY_OPTIONS = [
    ("Off", ""),
    ("Blue Sky + Clouds", "blue_clouds"),
    ("Blue Clear", "blue_clear"),
    ("Sunset Warm", "sunset_warm"),
    ("Golden Hour", "golden_hour"),
    ("Dramatic Storm", "dramatic_storm"),
    ("Overcast Soft", "overcast_soft"),
    ("Twilight Blue", "twilight_blue"),
]


TONE_OPTIONS = [
    ("Neutral", "neutral"),
    ("Warm", "warm"),
    ("Cool", "cool"),
    ("Real Estate (gamma+CLAHE)", "real_estate"),
    ("Auto (batch coherency)", "auto_batch"),
]


class _Row(QWidget):
    """1 hàng setting: icon + label + sub-label + dropdown."""
    changed = Signal(str)  # emit selected value

    def __init__(self, icon: str, title: str, sub: str, options: list[str], default_idx: int = 0, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Icon
        icon_lbl = QLabel(icon)
        icon_lbl.setFixedWidth(28)
        f = QFont()
        f.setPointSize(14)
        icon_lbl.setFont(f)
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        # Title + sub stacked
        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet("font-weight: 600; color: #e0e6ed;")
        self.sub_lbl = QLabel(sub)
        self.sub_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        text_box.addWidget(self.title_lbl)
        text_box.addWidget(self.sub_lbl)
        layout.addLayout(text_box, 1)

        # Dropdown
        self.combo = QComboBox()
        self.combo.addItems(options)
        self.combo.setCurrentIndex(default_idx)
        self.combo.setMinimumWidth(140)
        self.combo.currentTextChanged.connect(self._on_changed)
        layout.addWidget(self.combo)

    def _on_changed(self, value: str):
        # Update sub label to show selected
        self.sub_lbl.setText(value)
        self.changed.emit(value)

    def value(self) -> str:
        return self.combo.currentText()


class AISettingsPanel(QWidget):
    """Side panel với tất cả AI toggle."""
    settings_changed = Signal(object)   # AISettings
    max_preset_requested = Signal()     # User bấm "Tối đa chi tiết"
    light_preset_requested = Signal()   # Reset về preset nhẹ (mặc định)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = AISettings()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header = QLabel("✨ ADJUST ENHANCEMENT")
        header.setStyleSheet("font-weight: bold; color: #93c5fd; "
                             "font-size: 13px; padding: 6px 8px;")
        outer.addWidget(header)

        # Quick preset buttons (Max / Nhẹ)
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(8, 0, 8, 4)
        preset_row.setSpacing(6)
        self.max_btn = QPushButton("🎯 TỐI ĐA CHI TIẾT")
        self.max_btn.setObjectName("primary_btn")
        self.max_btn.setToolTip(
            "Bật mọi enhancement có lợi cho chất lượng:\n"
            "• Detail recovery + Denoise + Selective sharpen\n"
            "• AI Upscale x2\n"
            "• Tone Auto (batch coherency)\n"
            "Sky/Privacy/TV/Fire/Photog vẫn để user tự chọn (vì thay đổi nội dung)."
        )
        self.max_btn.setMinimumHeight(34)
        f = QFont()
        f.setBold(True)
        self.max_btn.setFont(f)
        self.max_btn.clicked.connect(self._on_max_clicked)
        preset_row.addWidget(self.max_btn, 2)

        self.light_btn = QPushButton("⚖ Cân bằng")
        self.light_btn.setMinimumHeight(34)
        self.light_btn.setToolTip("Quay về cấu hình mặc định nhanh (RE pipeline + perspective + lens).")
        self.light_btn.clicked.connect(self._on_light_clicked)
        preset_row.addWidget(self.light_btn, 1)
        outer.addLayout(preset_row)

        group = QGroupBox()
        group.setObjectName("ai_settings_group")
        gl = QVBoxLayout(group)
        gl.setSpacing(2)
        gl.setContentsMargins(4, 8, 4, 8)

        # Row: AI Version
        self.ai_ver = _Row("🤖", "AI Version", "V5 Latest", ["V5 Latest"], 0)
        gl.addWidget(self.ai_ver)

        # Sky Replacement
        sky_choices = [name for name, _ in SKY_OPTIONS]
        self.sky = _Row("☁", "Sky Replacement", "Off",
                         sky_choices, 0)
        self.sky.changed.connect(lambda v: self._update("sky_replacement", v))
        gl.addWidget(self.sky)

        # Perspective Correction
        self.persp = _Row("⊟", "Perspective Correction", "On", ["On", "Off"], 0)
        self.persp.changed.connect(lambda v: self._update("perspective_correct", v == "On"))
        gl.addWidget(self.persp)

        # Auto Privacy
        self.priv = _Row("🛡", "Auto Privacy", "Off", ["Off", "Faces only", "Faces + plates"], 0)
        self.priv.changed.connect(lambda v: self._update("auto_privacy", v != "Off"))
        gl.addWidget(self.priv)

        # Lens Correction
        self.lens = _Row("📷", "Lens Correction", "On", ["On", "Off"], 0)
        self.lens.changed.connect(lambda v: self._update("lens_correct", v == "On"))
        gl.addWidget(self.lens)

        # Window Pull
        self.window = _Row("🌅", "Window Pull", "Auto",
                            ["Off", "Auto (Windows with Skies)"], 1)
        self.window.changed.connect(lambda v: self._update("window_pull", v))
        gl.addWidget(self.window)

        # TV Blackout
        self.tv = _Row("📺", "TV Blackout", "Off", ["Off", "Auto detect"], 0)
        self.tv.changed.connect(lambda v: self._update("tv_blackout", v != "Off"))
        gl.addWidget(self.tv)

        # Fire in Fireplace
        self.fire = _Row("🔥", "Fire in Fireplace", "Off", ["Off", "Auto detect"], 0)
        self.fire.changed.connect(lambda v: self._update("fire_fireplace", v != "Off"))
        gl.addWidget(self.fire)

        # Photographer Removal
        self.photog = _Row("👤", "Photographer Removal", "Off", ["Off", "Auto"], 0)
        self.photog.changed.connect(lambda v: self._update("photog_removal", v != "Off"))
        gl.addWidget(self.photog)

        # AI Upscale (Real-ESRGAN) — default Off vì chậm trên CPU (~30-60s/ảnh)
        self.upscale = _Row("🎯", "AI Upscale (Real-ESRGAN)",
                             "Off", ["Off", "x2", "x4"], 0)
        self.upscale.changed.connect(self._on_upscale_changed)
        gl.addWidget(self.upscale)

        # Preset (cho enhance màu)
        self.preset = _Row("🎨", "Color Preset", "Real Estate",
                            ["studio", "real_estate", "portrait", "product", "outdoor"], 1)
        self.preset.changed.connect(lambda v: self._update("enhance_preset", v))
        gl.addWidget(self.preset)

        # ── Pro v2 features (default = MAX) ─────────
        # Batch tone coherency — default Auto batch coherency
        tone_labels = [name for name, _ in TONE_OPTIONS]
        self._tone_label_to_value = dict(TONE_OPTIONS)
        # default index = "Auto (batch coherency)"
        tone_default_idx = next(
            (i for i, (_, v) in enumerate(TONE_OPTIONS) if v == "auto_batch"), 0
        )
        self.tone = _Row("🎭", "Batch Tone Coherency",
                          tone_labels[tone_default_idx],
                          tone_labels, tone_default_idx)
        self.tone.changed.connect(self._on_tone_changed)
        gl.addWidget(self.tone)

        # Saliency sharpen — default Off (gây over-sharp/flat khi auto)
        self.sharp = _Row("🔬", "Selective Sharpening", "Off",
                           ["Off", "On (subject only)"], 0)
        self.sharp.changed.connect(lambda v: self._update("selective_sharpen", v != "Off"))
        gl.addWidget(self.sharp)

        # Virtual Twilight — Day → Sunset/Golden Hour (opt-in, killer feature)
        self.twilight = _Row("🌇", "Virtual Twilight", "Off",
                              ["Off", "Soft (0.5)", "Medium (0.7)", "Strong (0.85)"], 0)
        self.twilight.changed.connect(self._on_twilight_changed)
        gl.addWidget(self.twilight)

        # HDR Bracket Quality (deghost + color normalize) — chỉ ảnh hưởng khi
        # batch có bracket sets. Default ON vì cải thiện chất lượng merge mà
        # không tăng rủi ro cho ảnh đơn (toggle bỏ qua khi không có brackets).
        self.hdr_quality = _Row("📷", "HDR Bracket Quality", "On (deghost+norm)",
                                 ["Off", "On (deghost+norm)"], 1)
        self.hdr_quality.changed.connect(self._on_hdr_quality_changed)
        gl.addWidget(self.hdr_quality)

        # Seed determinism
        from PySide6.QtWidgets import QLineEdit
        seed_row = QWidget()
        seed_l = QHBoxLayout(seed_row)
        seed_l.setContentsMargins(8, 6, 8, 6)
        seed_lbl = QLabel("🎲")
        seed_lbl.setFixedWidth(28)
        f2 = QFont()
        f2.setPointSize(14)
        seed_lbl.setFont(f2)
        seed_lbl.setAlignment(Qt.AlignCenter)
        seed_l.addWidget(seed_lbl)
        seed_text_box = QVBoxLayout()
        seed_text_box.setSpacing(0)
        seed_t = QLabel("Seed (deterministic)")
        seed_t.setStyleSheet("font-weight: 600; color: #e0e6ed;")
        seed_s = QLabel("Empty = random | int = same output mỗi lần")
        seed_s.setStyleSheet("color: #94a3b8; font-size: 11px;")
        seed_text_box.addWidget(seed_t)
        seed_text_box.addWidget(seed_s)
        seed_l.addLayout(seed_text_box, 1)
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("None")
        self.seed_input.setMinimumWidth(140)
        self.seed_input.textChanged.connect(self._on_seed_changed)
        seed_l.addWidget(self.seed_input)
        gl.addWidget(seed_row)

        outer.addWidget(group)

        # Bottom: Save preset + Reset
        bottom = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save as default")
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        bottom.addWidget(self.save_btn)
        bottom.addWidget(self.reset_btn)
        outer.addLayout(bottom)

        outer.addStretch()

        # Init signal — emit current settings
        self._emit()

    def _on_upscale_changed(self, v: str):
        self._settings.ai_upscale_scale = {"Off": 0, "x2": 2, "x4": 4}.get(v, 0)
        self._emit()

    def _on_tone_changed(self, label: str):
        self._settings.tone_preset = self._tone_label_to_value.get(label, "neutral")
        self._emit()

    def _on_twilight_changed(self, label: str):
        if label == "Off":
            self._settings.virtual_twilight = False
            self._settings.twilight_strength = 0.85
        else:
            self._settings.virtual_twilight = True
            self._settings.twilight_strength = {
                "Soft (0.5)": 0.5,
                "Medium (0.7)": 0.7,
                "Strong (0.85)": 0.85,
            }.get(label, 0.85)
        self._emit()

    def _on_hdr_quality_changed(self, label: str):
        on = label != "Off"
        self._settings.hdr_deghost = on
        self._settings.hdr_color_normalize = on
        self._emit()

    def _on_max_clicked(self):
        """Flip mọi 'detail enhancer' về ON. Không động sky/privacy/tv/fire/photog
        vì những cái đó thay đổi nội dung — user phải tự chọn."""
        # Selective sharpen
        self.sharp.combo.setCurrentIndex(1)  # "On (subject only)"
        # AI Upscale x2 (x4 quá nặng cho default)
        self.upscale.combo.setCurrentIndex(1)  # "x2"
        # Tone auto batch
        for i in range(self.tone.combo.count()):
            if self.tone.combo.itemText(i).lower().startswith("auto"):
                self.tone.combo.setCurrentIndex(i)
                break
        # Window pull đảm bảo Auto
        self.window.combo.setCurrentIndex(1)
        # Perspective + Lens On
        self.persp.combo.setCurrentIndex(0)
        self.lens.combo.setCurrentIndex(0)
        # Color preset → real_estate (vẫn giữ nếu user đã chọn khác)
        # NOT touched here.
        self.max_preset_requested.emit()

    def _on_light_clicked(self):
        """Reset về defaults nhẹ — chỉ giữ pipeline cốt lõi."""
        self.sharp.combo.setCurrentIndex(0)   # Off
        self.upscale.combo.setCurrentIndex(0)  # Off
        self.tone.combo.setCurrentIndex(0)     # Neutral
        self.window.combo.setCurrentIndex(1)   # Auto
        self.persp.combo.setCurrentIndex(0)    # On
        self.lens.combo.setCurrentIndex(0)     # On
        self.light_preset_requested.emit()

    def _on_seed_changed(self, text: str):
        try:
            self._settings.seed = int(text) if text.strip() else None
        except ValueError:
            self._settings.seed = None
        self._emit()

    def _update(self, attr: str, value):
        setattr(self._settings, attr, value)
        self._emit()

    def _emit(self):
        self.settings_changed.emit(self._settings)

    def _on_reset(self):
        """Reset về MAX defaults (= dataclass AISettings defaults)."""
        self._settings = AISettings()
        self.ai_ver.combo.setCurrentIndex(0)
        self.sky.combo.setCurrentIndex(0)        # Sky Off (content-changing)
        self.persp.combo.setCurrentIndex(0)      # On
        self.priv.combo.setCurrentIndex(0)       # Off (sensitive)
        self.lens.combo.setCurrentIndex(0)       # On
        self.window.combo.setCurrentIndex(1)     # Auto
        self.tv.combo.setCurrentIndex(0)         # Off
        self.fire.combo.setCurrentIndex(0)       # Off
        self.photog.combo.setCurrentIndex(0)     # Off
        self.upscale.combo.setCurrentIndex(0)    # Off (chậm trên CPU)
        self.preset.combo.setCurrentIndex(1)     # real_estate
        # Tone → Auto (batch coherency)
        for i in range(self.tone.combo.count()):
            if self.tone.combo.itemText(i).lower().startswith("auto"):
                self.tone.combo.setCurrentIndex(i)
                break
        self.sharp.combo.setCurrentIndex(1)      # On (subject only)
        self.twilight.combo.setCurrentIndex(0)   # Off (opt-in)
        self.hdr_quality.combo.setCurrentIndex(1)  # On
        self._emit()

    def settings(self) -> AISettings:
        return self._settings

    def apply_to_job(self, job) -> None:
        """Áp settings vào BatchJob.

        Chính sách: Real-estate pipeline luôn ON theo default của BatchJob /
        toggle bên auto_tab. AI panel chỉ điều chỉnh các sub-feature:
          - Sky preset: bật `enable_sky_replace` khi user chọn preset, off khi "Off"
          - Window pull: nếu user tắt RE pipeline thì fallback `detail_recovery`
        """
        s = self._settings
        sky_map = dict(SKY_OPTIONS)
        sky_pre = sky_map.get(s.sky_replacement, "")
        if sky_pre:
            job.enable_sky_replace = True
            job.sky_preset = sky_pre
            # Nếu user chọn sky preset thì coi như họ chắc chắn muốn RE pipeline
            job.realestate_pipeline = True
        else:
            job.enable_sky_replace = False
            # KHÔNG override realestate_pipeline ở đây — auto_tab/BatchJob default đã ON
        # Window Pull dropdown: khi user tắt cả RE pipeline thì còn fallback detail_recovery
        if s.window_pull.startswith("Auto") and not job.realestate_pipeline:
            job.detail_recovery = True
        job.perspective_correct = s.perspective_correct
        job.lens_correct = s.lens_correct
        job.auto_privacy = s.auto_privacy
        job.tv_blackout = s.tv_blackout
        job.fire_fireplace = s.fire_fireplace
        job.photog_removal = s.photog_removal
        job.ai_upscale_scale = s.ai_upscale_scale
        job.enhance_preset = s.enhance_preset
        job.color_enhance = True
        # Pro v2 fields
        job.tone_preset = s.tone_preset
        job.selective_sharpen = s.selective_sharpen
        # Pro v3 features (port từ imagen-ai + Edit-image)
        job.virtual_twilight = s.virtual_twilight
        job.twilight_strength = s.twilight_strength
        job.hdr_deghost = s.hdr_deghost
        job.hdr_color_normalize = s.hdr_color_normalize
        job.seed = s.seed
