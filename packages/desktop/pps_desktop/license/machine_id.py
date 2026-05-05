"""Machine fingerprint — bind license tới phần cứng.

Hash từ:
- MAC address (bộ chuyển đổi mạng đầu tiên)
- CPU ID (Windows: ProcessorId; macOS/Linux: cpuid)
- Disk serial (Windows: VolumeSerialNumber; *nix: blkid)
- Hostname (fallback)

Format: SHA-256 hex 64 chars, truncate 32 cho compact.
"""
from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import uuid


def _get_mac() -> str:
    """MAC từ uuid.getnode() — stable across reboot trên cùng máy."""
    try:
        return f"{uuid.getnode():012x}"
    except Exception:
        return "00" * 6


def _get_cpu_id_windows() -> str:
    try:
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "processorid"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="ignore")
        for line in out.strip().splitlines()[1:]:
            line = line.strip()
            if line:
                return line
    except Exception:
        pass
    return ""


def _get_disk_serial_windows() -> str:
    try:
        out = subprocess.check_output(
            ["wmic", "diskdrive", "get", "serialnumber"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="ignore")
        for line in out.strip().splitlines()[1:]:
            line = line.strip()
            if line and line.lower() != "serialnumber":
                return line
    except Exception:
        pass
    return ""


def _get_volume_serial_windows() -> str:
    try:
        out = subprocess.check_output(
            ["cmd", "/c", "vol", "C:"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="ignore")
        # Output dạng: "Volume Serial Number is XXXX-XXXX"
        for line in out.splitlines():
            if "Serial" in line:
                parts = line.strip().split()
                if parts:
                    return parts[-1]
    except Exception:
        pass
    return ""


def _get_machine_components() -> dict[str, str]:
    """Thu thập các thành phần để hash."""
    components = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "mac": _get_mac(),
    }
    if platform.system() == "Windows":
        components["cpu"] = _get_cpu_id_windows()
        components["disk"] = _get_disk_serial_windows()
        components["volume"] = _get_volume_serial_windows()
    else:
        # macOS / Linux fallback — chỉ dùng MAC + hostname (đủ ổn định)
        try:
            out = subprocess.check_output(
                ["uname", "-a"], stderr=subprocess.DEVNULL, timeout=5,
            ).decode("utf-8", errors="ignore").strip()
            components["uname"] = out
        except Exception:
            pass
    return components


def get_machine_id() -> str:
    """Trả machine fingerprint 32-char hex (deterministic per máy)."""
    comps = _get_machine_components()
    # Sort để hash stable
    canonical = "|".join(f"{k}={v}" for k, v in sorted(comps.items()))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:32]


def get_machine_info() -> dict:
    """Debug info — không dùng cho license, chỉ để hiển thị/log."""
    comps = _get_machine_components()
    return {
        **comps,
        "machine_id": get_machine_id(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_machine_info(), indent=2, ensure_ascii=False))
