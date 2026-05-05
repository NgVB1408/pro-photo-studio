"""Pro Photo Studio Tool — desktop entry point.

Chạy:
    python -m desktop
hoặc:
    python desktop/main.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path để import pps_core và desktop modules
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Also add project root for desktop modules
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def _setup_logging():
    log_file = (
        Path(sys.executable).resolve().parent / "desktop.log"
        if getattr(sys, "frozen", False)
        else ROOT / "desktop.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )
    logging.info("Desktop log: %s", log_file)


def _resource_dir() -> Path:
    """Trả thư mục chứa data files — works cả dev mode + PyInstaller frozen."""
    if getattr(sys, "frozen", False):
        # PyInstaller one-folder: data ở _internal/ next to exe
        # PyInstaller one-file: data extract vào sys._MEIPASS
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent / "_internal"
    return Path(__file__).resolve().parent.parent  # dev mode → project root


def _load_stylesheet() -> str:
    """Load styles.qss — try cả frozen path lẫn dev path."""
    candidates = [
        _resource_dir() / "desktop" / "ui" / "styles.qss",  # frozen one-folder
        Path(__file__).resolve().parent / "ui" / "styles.qss",  # dev mode
    ]
    for p in candidates:
        if p.is_file():
            logging.info("Stylesheet loaded: %s", p)
            return p.read_text(encoding="utf-8")
    logging.warning("styles.qss không tìm thấy — dùng theme mặc định")
    return ""


def main():
    _setup_logging()

    # Init telemetry (no-op nếu không có DSN)
    try:
        from pps_desktop.telemetry import init_sentry
        init_sentry()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Telemetry init fail: %s", exc)

    # High DPI cho Windows
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Pro Photo Studio")
    app.setOrganizationName("WatermarkToolkit")
    app.setStyle("Fusion")  # consistent across platforms

    # Stylesheet
    qss = _load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    # Icon (nếu có)
    icon_path = Path(__file__).resolve().parent / "assets" / "icon.ico"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Import sau khi app + stylesheet đã setup
    from pps_desktop.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
