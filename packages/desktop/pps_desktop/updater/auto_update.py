"""Auto-updater — pull release info từ GitHub Releases / license server.

Workflow:
1. Khi mở app, check version mới qua POST /version (hoặc GitHub API)
2. Nếu có version mới → hiện dialog hỏi user update
3. Tải installer mới về %TEMP%
4. Đóng app + chạy installer (Inno Setup tự update)

Format version: SemVer (0.1.0)
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Current version — sync với pyproject.toml + setup.iss
CURRENT_VERSION = "0.1.0"

# GitHub repo cho releases
GITHUB_REPO = os.environ.get(
    "PHOTOSTUDIO_GITHUB_REPO",
    "NgVB1408/watermark-toolkit",
)

# Version check endpoint — license server hoặc GitHub API
VERSION_CHECK_URL = os.environ.get(
    "PHOTOSTUDIO_VERSION_URL",
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
)


@dataclass
class UpdateInfo:
    available: bool
    current: str
    latest: str
    download_url: str = ""
    release_notes: str = ""
    release_url: str = ""
    is_critical: bool = False  # nếu current < min_supported


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse SemVer 'x.y.z' → tuple(int)."""
    v = v.lstrip("v").split("-")[0]
    parts = v.split(".")
    try:
        return tuple(int(x) for x in parts[:3])
    except ValueError:
        return (0, 0, 0)


def _is_newer(a: str, b: str) -> bool:
    """True nếu a > b (newer)."""
    return _parse_version(a) > _parse_version(b)


def check_for_updates(timeout: float = 8.0) -> UpdateInfo:
    """Check version mới. Trả UpdateInfo. KHÔNG raise nếu offline."""
    try:
        import requests
    except ImportError:
        logger.warning("requests chưa cài — bỏ qua update check")
        return UpdateInfo(False, CURRENT_VERSION, CURRENT_VERSION)

    try:
        resp = requests.get(
            VERSION_CHECK_URL, timeout=timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code != 200:
            logger.warning("Update check HTTP %d", resp.status_code)
            return UpdateInfo(False, CURRENT_VERSION, CURRENT_VERSION)
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Update check fail: %s", exc)
        return UpdateInfo(False, CURRENT_VERSION, CURRENT_VERSION)

    # GitHub API format
    if "tag_name" in data:
        latest = data["tag_name"]
        notes = data.get("body", "")
        url = data.get("html_url", "")
        # Tìm asset .exe nếu có
        download = ""
        for asset in data.get("assets", []):
            name = asset.get("name", "").lower()
            if name.endswith(".exe") or "setup" in name:
                download = asset.get("browser_download_url", "")
                break
    else:
        # License server /version format
        latest = data.get("latest", CURRENT_VERSION)
        notes = data.get("release_notes", "")
        url = data.get("release_notes_url", "")
        download = data.get("download_url", "")

    available = _is_newer(latest, CURRENT_VERSION)
    return UpdateInfo(
        available=available,
        current=CURRENT_VERSION,
        latest=latest,
        download_url=download,
        release_notes=notes,
        release_url=url,
    )


def download_installer(url: str, dest_dir: Path | None = None,
                        progress_callback=None) -> Path:
    """Tải installer .exe về %TEMP% (hoặc dest_dir).

    progress_callback(downloaded, total): callable, nếu None thì im lặng.

    Returns: Path tới file đã tải.
    """
    import tempfile
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests chưa cài")

    if dest_dir is None:
        dest_dir = Path(tempfile.gettempdir()) / "PhotoStudio_Update"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1] or "setup.exe"
    dst = dest_dir / name

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
    logger.info("Đã tải %s (%.1f MB)", dst, dst.stat().st_size / 1024 / 1024)
    return dst


def launch_installer(installer_path: Path, silent: bool = False) -> bool:
    """Chạy installer mới, đóng app hiện tại.

    silent=True: chạy /VERYSILENT (Inno Setup) — auto update không UI.
    """
    if not installer_path.is_file():
        logger.error("Installer không tồn tại: %s", installer_path)
        return False

    args = [str(installer_path)]
    if silent:
        # Inno Setup silent flags
        args += ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"]

    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(args, shell=False, close_fds=True,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                              if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0)
        else:
            import subprocess
            subprocess.Popen(args, close_fds=True)
        logger.info("Launched installer: %s", installer_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Launch installer fail: %s", exc)
        return False


def perform_update(update_info: UpdateInfo, *, silent: bool = False,
                    progress_callback=None) -> bool:
    """Workflow đầy đủ: download + launch installer + exit app.

    Caller nên gọi sau khi user confirm và app đã save state.
    """
    if not update_info.available or not update_info.download_url:
        return False
    try:
        installer = download_installer(
            update_info.download_url,
            progress_callback=progress_callback,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Download fail: %s", exc)
        return False
    return launch_installer(installer, silent=silent)
