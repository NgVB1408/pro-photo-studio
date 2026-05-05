"""License bar — hiển thị info khách + nút logout. Gọi activate_dialog nếu chưa có."""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QDialog, QVBoxLayout,
    QLineEdit, QDialogButtonBox, QMessageBox,
)

from ..license import client as lic


class ActivateDialog(QDialog):
    """Dialog yêu cầu user nhập license key."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kích hoạt license")
        self.setModal(True)
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("🔑 Nhập License Key")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #93c5fd;")
        layout.addWidget(title)

        info = QLabel(
            "Nhập key đã nhận từ người bán. Format: "
            "<code>XXXXXX-XXXXXX-XXXXXX-XXXXXX</code>"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #94a3b8;")
        layout.addWidget(info)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Ví dụ: PROABC-DEFGHI-JKLMNO-PQR123")
        self.key_input.setMaxLength(27)
        self.key_input.textChanged.connect(self._normalize)
        layout.addWidget(self.key_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Tên/Studio (hiển thị trong app)")
        layout.addWidget(self.name_input)

        # Demo hint
        demo_hint = QLabel(
            "💡 <i>Demo: dùng <code>" + lic.DEMO_PRO_KEY + "</code></i>"
        )
        demo_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(demo_hint)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.button(QDialogButtonBox.Ok).setText("Kích hoạt")
        btn_box.button(QDialogButtonBox.Cancel).setText("Huỷ")
        btn_box.accepted.connect(self._on_activate)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._activated_license: lic.License | None = None

    def _normalize(self, text: str):
        """Auto-format key khi gõ."""
        norm = lic.normalize_key(text)
        if norm != text:
            cursor = self.key_input.cursorPosition()
            self.key_input.blockSignals(True)
            self.key_input.setText(norm)
            self.key_input.setCursorPosition(min(cursor, len(norm)))
            self.key_input.blockSignals(False)

    def _on_activate(self):
        key = lic.normalize_key(self.key_input.text())
        name = self.name_input.text().strip() or "Khách hàng VIP"
        try:
            self._activated_license = lic.activate(key, customer_name=name)
            self.accept()
        except lic.LicenseError as exc:
            QMessageBox.warning(self, "License không hợp lệ", str(exc))

    @property
    def license(self) -> lic.License | None:
        return self._activated_license


class LicenseBar(QWidget):
    """Top bar hiển thị license info + nút logout."""

    logout_requested = Signal()

    def __init__(self, license: lic.License, parent=None):
        super().__init__(parent)
        self._license = license
        self.setObjectName("license_bar")
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(20)

        # Customer info
        cust_layout = QVBoxLayout()
        cust_layout.setSpacing(2)
        cust_label = QLabel(f"👤 Khách hàng: <b>{self._license.customer_name}</b>")
        cust_layout.addWidget(cust_label)

        key_label = QLabel(f"🔑 License Key: {self._license.key}")
        key_label.setObjectName("license_key_label")
        cust_layout.addWidget(key_label)

        layout.addLayout(cust_layout)

        # Tier + expiry
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        tier_label = QLabel(f"💎 Gói: <b>{self._license.tier_label()}</b>")
        tier_label.setObjectName("license_tier_label")
        info_layout.addWidget(tier_label)

        expiry_label = QLabel(f"⏳ Hạn: {self._license.expiry_label()}")
        info_layout.addWidget(expiry_label)
        layout.addLayout(info_layout)

        layout.addStretch()

        # Logout button
        self.logout_btn = QPushButton("ĐĂNG XUẤT")
        self.logout_btn.setObjectName("logout_btn")
        self.logout_btn.setCursor(Qt.PointingHandCursor)
        self.logout_btn.clicked.connect(self.logout_requested.emit)
        layout.addWidget(self.logout_btn)

    def update_license(self, license: lic.License):
        self._license = license
        # Rebuild để update labels
        for i in reversed(range(self.layout().count())):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().setParent(None)
        self._build_ui()
