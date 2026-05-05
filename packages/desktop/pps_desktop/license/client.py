"""License client — online-first với offline fallback.

Workflow:
1. activate(key) → POST /activate {key, machine_id}
   → Server return signed token + license info
   → Lưu vào %APPDATA%/AutoHDR/license.json
2. Mỗi lần mở app: load token, verify signature offline với public key
3. Mỗi 7 ngày: heartbeat() → POST /heartbeat → refresh token
4. Offline > 30 ngày: token expire → require online activate lại

Fallback offline-only mode (không có internet):
- DEV_MODE=1 (env var): chấp nhận DEMO keys không cần server
- Nếu server không reachable: dùng cached token đến khi expire

PRODUCTION endpoints (set qua env LICENSE_SERVER_URL):
  https://photostudio-license.<account>.workers.dev
hoặc:
  https://license.photostudio.vn (custom domain)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from .machine_id import get_machine_id

logger = logging.getLogger(__name__)


# ====== Config ======

# License server URL — set qua env LICENSE_SERVER_URL hoặc hardcode khi build
DEFAULT_SERVER_URL = os.environ.get(
    "LICENSE_SERVER_URL",
    "https://photostudio-license.workers.dev",
)
SERVER_TIMEOUT = float(os.environ.get("LICENSE_SERVER_TIMEOUT", "10"))

# DEV mode — bypass server, chấp nhận DEMO keys
DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"

# Offline grace period — sau ngần ngày này không heartbeat → require online
OFFLINE_GRACE_DAYS = 30
HEARTBEAT_INTERVAL_DAYS = 7

# MVP fallback secret (offline DEMO mode chỉ)
_DEMO_SECRET = b"watermark-toolkit-mvp-secret-CHANGE-ME-IN-PROD-2026"


LicenseTier = Literal["standard", "pro", "studio", "trial"]

_KEY_PATTERN = re.compile(r"^[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}$")


# ====== Models ======

@dataclass
class License:
    key: str
    customer_id: str = "VIP-00000"
    customer_name: str = "Khách hàng VIP"
    tier: LicenseTier = "pro"
    machine_id: str = ""
    issued_at: str = ""
    expires_at: str | None = None  # None = lifetime
    max_machines: int = 3
    last_heartbeat: str = ""
    server_token: str = ""  # base64 signature từ server (cho offline verify)
    server_payload: str = ""  # JSON payload signed
    valid_until: str = ""  # token expiry (offline grace period)

    @property
    def is_lifetime(self) -> bool:
        return self.expires_at is None

    @property
    def is_expired(self) -> bool:
        if self.is_lifetime:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False

    def expiry_label(self) -> str:
        if self.is_lifetime:
            return "Vĩnh viễn (Lifetime)"
        if self.is_expired:
            return f"Đã hết hạn ({self.expires_at})"
        return f"Còn đến: {self.expires_at}"

    def tier_label(self) -> str:
        labels = {
            "trial": "Dùng thử",
            "standard": "Standard",
            "pro": "Pro",
            "studio": "Studio",
        }
        return labels.get(self.tier, self.tier)


class LicenseError(Exception):
    pass


# ====== Storage ======

def _config_dir() -> Path:
    if os.environ.get("WATERMARK_TOOLKIT_LICENSE_DIR"):
        d = Path(os.environ["WATERMARK_TOOLKIT_LICENSE_DIR"])
    elif os.name == "nt":
        d = Path(os.environ.get("APPDATA", str(Path.home()))) / "AutoHDR"
    elif os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        d = Path.home() / "Library" / "Application Support" / "AutoHDR"
    else:
        d = Path.home() / ".config" / "AutoHDR"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _license_path() -> Path:
    return _config_dir() / "license.json"


# ====== Format helpers ======

def normalize_key(raw: str) -> str:
    s = "".join(c for c in raw.upper() if c.isalnum() or c == "-")
    if len(s) == 24 and "-" not in s:
        s = f"{s[:6]}-{s[6:12]}-{s[12:18]}-{s[18:24]}"
    return s


def is_valid_key_format(key: str) -> bool:
    return bool(_KEY_PATTERN.match(key))


# ====== Demo / DEV mode (offline fallback) ======

DEMO_TRIAL_KEY = "TRIAL1-DEMO00-2026A0-MVP001"
DEMO_PRO_KEY = "PROABC-DEFGHI-JKLMNO-PQR123"
DEMO_STUDIO_KEY = "USTUDIO-MVP123-2026AB-DEMO00"


def _verify_demo_key_offline(key: str) -> bool:
    """DEV mode hoặc DEMO keys — accept không cần server."""
    if not is_valid_key_format(key):
        return False
    if key in (DEMO_TRIAL_KEY, DEMO_PRO_KEY, DEMO_STUDIO_KEY):
        return True
    if not DEV_MODE:
        return False
    parts = key.split("-")
    if len(parts) != 4:
        return False
    for p in parts:
        if len(set(p)) < 2:
            return False
    return True


def _activate_demo(key: str, customer_name: str) -> License:
    """Activate offline cho DEMO keys hoặc DEV mode."""
    first_char = key[0]
    tier_map = {"S": "standard", "P": "pro", "U": "studio", "T": "trial"}
    tier: LicenseTier = tier_map.get(first_char, "pro")  # type: ignore[assignment]
    now = datetime.now(timezone.utc)
    expires_at = (
        (now + timedelta(days=14)).isoformat() if tier == "trial" else None
    )
    return License(
        key=key,
        customer_id="DEMO-" + key[-6:],
        customer_name=customer_name,
        tier=tier,
        machine_id=get_machine_id(),
        issued_at=now.isoformat(),
        expires_at=expires_at,
        max_machines={"trial": 1, "standard": 1, "pro": 3, "studio": 10}[tier],
        last_heartbeat=now.isoformat(),
        server_token="DEMO-OFFLINE",
        valid_until=(now + timedelta(days=OFFLINE_GRACE_DAYS)).isoformat(),
    )


# ====== HTTP client ======

def _post_server(endpoint: str, payload: dict) -> dict:
    """POST tới license server. Raise LicenseError nếu fail."""
    try:
        import requests
    except ImportError:
        raise LicenseError(
            "Thiếu thư viện 'requests'. Cài: pip install requests"
        )

    url = f"{DEFAULT_SERVER_URL.rstrip('/')}{endpoint}"
    try:
        resp = requests.post(
            url, json=payload,
            timeout=SERVER_TIMEOUT,
            headers={"User-Agent": "PhotoStudio-Client/0.1"},
        )
    except requests.exceptions.Timeout:
        raise LicenseError(
            "Server không phản hồi (timeout). Kiểm tra mạng + thử lại."
        )
    except requests.exceptions.ConnectionError:
        raise LicenseError(
            "Không kết nối được server license. Kiểm tra mạng."
        )
    except Exception as exc:  # noqa: BLE001
        raise LicenseError(f"Lỗi kết nối: {exc}")

    try:
        data = resp.json()
    except Exception:
        raise LicenseError(f"Server trả response không hợp lệ ({resp.status_code})")

    if resp.status_code >= 400:
        msg = data.get("error", f"HTTP {resp.status_code}")
        raise LicenseError(msg)
    return data


# ====== Public API ======

def activate(key: str, customer_name: str = "Khách hàng VIP",
             customer_email: str = "") -> License:
    """Kích hoạt license — gọi server, fallback offline cho DEMO keys/DEV."""
    key_norm = normalize_key(key)
    if not is_valid_key_format(key_norm):
        raise LicenseError(
            "License key sai định dạng. Phải dạng XXXXXX-XXXXXX-XXXXXX-XXXXXX "
            "(chỉ chữ HOA và số)."
        )

    machine_id = get_machine_id()

    # Try DEMO/DEV mode first (no network required)
    if DEV_MODE or key_norm in (DEMO_TRIAL_KEY, DEMO_PRO_KEY, DEMO_STUDIO_KEY):
        if _verify_demo_key_offline(key_norm):
            license = _activate_demo(key_norm, customer_name)
            save_license(license)
            logger.info("Activated DEMO/DEV license: tier=%s", license.tier)
            return license

    # Online activate qua server
    try:
        data = _post_server("/activate", {
            "key": key_norm,
            "machine_id": machine_id,
            "customer_email": customer_email,
        })
    except LicenseError:
        # Re-raise — không silent fallback ở production
        raise

    # Parse response
    lic_info = data.get("license", {})
    token = data.get("token", {})
    payload = token.get("payload", {})
    signature = token.get("signature", "")

    now = datetime.now(timezone.utc)
    license = License(
        key=lic_info.get("key", key_norm),
        customer_id=lic_info.get("customer_id", "VIP-00000"),
        customer_name=lic_info.get("customer_name", customer_name),
        tier=lic_info.get("tier", "pro"),
        machine_id=machine_id,
        issued_at=lic_info.get("issued_at", now.isoformat()),
        expires_at=lic_info.get("expires_at"),
        max_machines=lic_info.get("max_machines", 3),
        last_heartbeat=now.isoformat(),
        server_token=signature,
        server_payload=json.dumps(payload),
        valid_until=payload.get(
            "valid_until",
            (now + timedelta(days=OFFLINE_GRACE_DAYS)).isoformat(),
        ),
    )
    save_license(license)
    logger.info("Activated license online: tier=%s, customer=%s",
                license.tier, license.customer_id)
    return license


def heartbeat(license: License) -> License:
    """Renew token bằng cách gọi /heartbeat. Fallback giữ nguyên nếu offline."""
    if license.server_token == "DEMO-OFFLINE":
        license.last_heartbeat = datetime.now(timezone.utc).isoformat()
        save_license(license)
        return license

    try:
        data = _post_server("/heartbeat", {
            "key": license.key,
            "machine_id": license.machine_id,
        })
        token = data.get("token", {})
        signature = token.get("signature", "")
        payload = token.get("payload", {})
        license.server_token = signature
        license.server_payload = json.dumps(payload)
        license.valid_until = payload.get(
            "valid_until", license.valid_until,
        )
        license.last_heartbeat = datetime.now(timezone.utc).isoformat()
        save_license(license)
        logger.info("Heartbeat OK")
    except LicenseError as exc:
        logger.warning("Heartbeat fail (offline mode active): %s", exc)
    return license


def deactivate_remote(license: License) -> bool:
    """Gọi server /deactivate để free machine slot."""
    if license.server_token == "DEMO-OFFLINE":
        return True
    try:
        _post_server("/deactivate", {
            "key": license.key,
            "machine_id": license.machine_id,
        })
        return True
    except LicenseError as exc:
        logger.warning("Deactivate fail (vẫn xoá local): %s", exc)
        return False


def deactivate() -> bool:
    """Xoá license local + gọi server deactivate (best-effort)."""
    license = load_license()
    if license:
        deactivate_remote(license)
    path = _license_path()
    if path.is_file():
        path.unlink()
        return True
    return False


def save_license(license: License) -> Path:
    path = _license_path()
    path.write_text(
        json.dumps(asdict(license), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_license() -> License | None:
    path = _license_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Strip new fields nếu có (compat với data cũ)
        valid_keys = {f.name for f in License.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_keys}
        return License(**data)
    except Exception as exc:
        logger.warning("Không đọc được license: %s", exc)
        return None


def is_machine_match(license: License) -> bool:
    return license.machine_id == get_machine_id()


def needs_online_check(license: License) -> bool:
    """True nếu token sắp hết hạn hoặc quá HEARTBEAT_INTERVAL_DAYS."""
    if not license.last_heartbeat:
        return True
    if license.valid_until:
        try:
            valid = datetime.fromisoformat(license.valid_until)
            # Cần online check khi còn < 7 ngày
            if (valid - datetime.now(timezone.utc)) < timedelta(days=7):
                return True
        except Exception:
            return True
    try:
        last = datetime.fromisoformat(license.last_heartbeat)
        return (
            datetime.now(timezone.utc) - last
        ) > timedelta(days=HEARTBEAT_INTERVAL_DAYS)
    except Exception:
        return True


def is_token_valid_offline(license: License) -> bool:
    """Check signature + valid_until offline."""
    if not license.valid_until:
        return False
    try:
        valid_until = datetime.fromisoformat(license.valid_until)
        if datetime.now(timezone.utc) > valid_until:
            return False
    except Exception:
        return False
    return is_machine_match(license)


# Compat aliases
def verify_key_offline(key: str, machine_id: str) -> bool:
    """Compat với code cũ — chỉ check format."""
    return is_valid_key_format(normalize_key(key))
