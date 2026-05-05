"""Telemetry / crash reporting via Sentry (optional).

Init khi app start. Nếu không có DSN env → no-op.

Set DSN qua build (PyInstaller --add-data hoặc env var):
    SENTRY_DSN=https://xxx@oXXX.ingest.sentry.io/yyy

Hoặc trong code (KHÔNG khuyến nghị commit DSN public):
    init_sentry(dsn="https://...")

Privacy: gắn user_id = machine_id (anonymous), KHÔNG send PII.
"""
from __future__ import annotations

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

# DSN — set qua env hoặc bake-in khi build
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
APP_VERSION = "0.1.0"


def init_sentry(dsn: str = "", environment: str = "production") -> bool:
    """Init Sentry. Trả True nếu init thành công."""
    dsn = dsn or SENTRY_DSN
    if not dsn:
        logger.info("Sentry DSN không set — telemetry tắt")
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.info("sentry-sdk chưa cài — telemetry tắt")
        return False

    try:
        # Lấy machine_id để làm user_id (anonymous)
        from .license.machine_id import get_machine_id
        user_id = get_machine_id()
    except Exception:
        user_id = "unknown"

    sentry_sdk.init(
        dsn=dsn,
        release=f"photo-studio@{APP_VERSION}",
        environment=environment,
        # Performance traces (10% sample để không waste quota)
        traces_sample_rate=0.1,
        # KHÔNG ship default integrations để giảm noise
        default_integrations=False,
        send_default_pii=False,  # privacy
        attach_stacktrace=True,
        max_breadcrumbs=30,
    )
    sentry_sdk.set_user({"id": user_id})
    sentry_sdk.set_tag("os", platform.system())
    sentry_sdk.set_tag("os_version", platform.version())
    sentry_sdk.set_tag("python_version", sys.version.split()[0])
    logger.info("Sentry initialized — user_id=%s", user_id[:12])
    return True


def capture_exception(exc: Exception):
    """Manual capture exception."""
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except ImportError:
        pass


def capture_message(msg: str, level: str = "info"):
    try:
        import sentry_sdk
        sentry_sdk.capture_message(msg, level=level)
    except ImportError:
        pass


def add_breadcrumb(category: str, message: str, level: str = "info", **data):
    """Track user action — hiển thị trong stack trace."""
    try:
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category=category, message=message, level=level, data=data,
        )
    except ImportError:
        pass
