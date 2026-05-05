"""Sky library với ảnh trời thật — pro-grade replacement.

Thay vì depend vào Unsplash API (URL không ổn định, đổi format), tool dùng
**user-managed library**: user drop ảnh trời JPG/PNG vào folder, tool tự
quét và pick ngẫu nhiên theo category.

Workflow cho photographer:
1. Sưu tầm 50-200 ảnh trời chất lượng cao của riêng bạn (chụp + tải CC0)
2. Tổ chức theo subfolder category trong:
       ~/.pps_core/sky_library/
       ├── blue_clouds/
       ├── blue_clear/
       ├── sunset_warm/
       ├── golden_hour/
       ├── dramatic_storm/
       ├── overcast_soft/
       └── twilight_blue/
3. Tool tự quét + cache index. Khi `replace_sky(..., sky_source="real_photo")`
   → random pick từ category tương ứng.
4. Nếu category không có ảnh → fallback procedural (sky_lib.generate_sky).

Custom directory: set env var ``WATERMARK_TOOLKIT_SKY_DIR`` hoặc gọi
``set_sky_library_dir(path)`` trước.

CLI helper:
    watermark sky-library --list                 # liệt kê
    watermark sky-library --add file.jpg --category blue_clouds
    watermark sky-library --download-samples     # download tập demo nhỏ

Format:
- Ảnh JPG/PNG/WebP, cạnh dài ≥ 1920px (sẽ auto-resize khi composite)
- Tên file: bất kỳ, tool dùng tên file làm sky id
- Subfolder = category. Ảnh ở root → category "uncategorized"
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Categories chuẩn — match với preset của sky_lib v2
STANDARD_CATEGORIES: tuple[str, ...] = (
    "blue_clouds",
    "blue_clear",
    "sunset_warm",
    "golden_hour",
    "dramatic_storm",
    "overcast_soft",
    "twilight_blue",
)

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_LIBRARY_DIR_OVERRIDE: Path | None = None
_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, list[Path]] | None = None


def set_sky_library_dir(path: str | Path) -> None:
    """Set thư mục sky library tuỳ chỉnh (override env var)."""
    global _LIBRARY_DIR_OVERRIDE, _INDEX_CACHE
    _LIBRARY_DIR_OVERRIDE = Path(path).expanduser()
    _INDEX_CACHE = None  # invalidate index


def get_sky_library_dir() -> Path:
    """Trả thư mục sky library hiện tại, tạo nếu chưa có."""
    if _LIBRARY_DIR_OVERRIDE is not None:
        d = _LIBRARY_DIR_OVERRIDE
    else:
        env = os.environ.get("WATERMARK_TOOLKIT_SKY_DIR")
        d = Path(env).expanduser() if env else Path.home() / ".pps_core" / "sky_library"
    d.mkdir(parents=True, exist_ok=True)
    # Tạo subfolder category để user có template
    for cat in STANDARD_CATEGORIES:
        (d / cat).mkdir(exist_ok=True)
    return d


def _build_index() -> dict[str, list[Path]]:
    """Quét library dir và build index {category: [paths]}."""
    root = get_sky_library_dir()
    index: dict[str, list[Path]] = {}
    # Subfolder = category
    for sub in root.iterdir():
        if sub.is_dir():
            cat = sub.name
            files = [
                f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in _VALID_EXTENSIONS
            ]
            if files:
                index[cat] = sorted(files)
    # Ảnh ở root = uncategorized
    root_files = [
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() in _VALID_EXTENSIONS
    ]
    if root_files:
        index["uncategorized"] = sorted(root_files)
    return index


def _get_index(refresh: bool = False) -> dict[str, list[Path]]:
    """Lazy-load + cache index. ``refresh=True`` để force rebuild."""
    global _INDEX_CACHE
    with _INDEX_LOCK:
        if _INDEX_CACHE is None or refresh:
            _INDEX_CACHE = _build_index()
        return _INDEX_CACHE


def list_categories() -> list[str]:
    """Liệt kê category có ảnh."""
    return sorted(_get_index().keys())


def list_skies(category: str | None = None) -> list[Path]:
    """Liệt kê path các sky theo category. None = tất cả."""
    idx = _get_index()
    if category:
        return list(idx.get(category, []))
    return [p for files in idx.values() for p in files]


def total_skies() -> int:
    return sum(len(v) for v in _get_index().values())


def add_sky(
    path: str | Path,
    category: str = "uncategorized",
    *,
    copy: bool = True,
) -> Path:
    """Thêm 1 file ảnh vào library.

    Args:
        path: file ảnh nguồn.
        category: category đích (tạo subfolder nếu chưa có).
        copy: True = copy file (an toàn). False = move file.
    Returns:
        Path mới trong library.
    """
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")
    if src.suffix.lower() not in _VALID_EXTENSIONS:
        raise ValueError(f"Định dạng không hỗ trợ: {src.suffix}")
    lib_dir = get_sky_library_dir() / category
    lib_dir.mkdir(parents=True, exist_ok=True)
    dst = lib_dir / src.name
    # Tránh ghi đè: thêm suffix (1), (2)...
    if dst.exists():
        i = 1
        while True:
            candidate = lib_dir / f"{src.stem}_{i}{src.suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        src.rename(dst)
    logger.info("Đã thêm sky: %s → %s", src, dst)
    _get_index(refresh=True)
    return dst


def remove_sky(path: str | Path) -> bool:
    """Xoá 1 sky khỏi library. Trả True nếu thành công."""
    p = Path(path).expanduser()
    if not p.is_file():
        return False
    p.unlink()
    _get_index(refresh=True)
    logger.info("Đã xoá sky: %s", p)
    return True


def load_sky(path: str | Path) -> np.ndarray | None:
    """Load 1 sky bằng path. BGR uint8 hoặc None nếu fail."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Không decode được sky: %s", p)
    return img


def random_sky(
    category: str | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    """Random pick 1 sky từ category. Trả (img, info_dict)."""
    candidates = list_skies(category)
    if not candidates:
        return None, None
    rng = np.random.default_rng(seed)
    picked = candidates[int(rng.integers(0, len(candidates)))]
    img = load_sky(picked)
    if img is None:
        return None, None
    return img, {
        "id": picked.stem,
        "path": str(picked),
        "category": picked.parent.name
        if picked.parent != get_sky_library_dir()
        else "uncategorized",
    }


def load_sky_by_id(sky_id: str) -> np.ndarray | None:
    """Load sky bằng filename stem (id). Tìm xuyên qua mọi category."""
    for path in list_skies():
        if path.stem == sky_id:
            return load_sky(path)
    return None


def stats() -> dict:
    """Thống kê thư viện."""
    idx = _get_index()
    return {
        "library_dir": str(get_sky_library_dir()),
        "total_count": sum(len(v) for v in idx.values()),
        "by_category": {cat: len(files) for cat, files in idx.items()},
    }


def refresh_index() -> dict[str, list[Path]]:
    """Force rebuild index — gọi sau khi user copy ảnh thủ công vào folder."""
    return _get_index(refresh=True)


# =====================================================================
# Sample download — best-effort từ Pexels/Pixabay/etc, fallback procedural
# =====================================================================

# Pexels CC0 photo URLs (verified working). User có thể chạy
# `watermark sky-library --download-samples` để có 1 set demo nhỏ.
_SAMPLE_PEXELS_URLS: dict[str, list[str]] = {
    "blue_clouds": [
        "https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/1118873/pexels-photo-1118873.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/1431822/pexels-photo-1431822.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "blue_clear": [
        "https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/45848/pexels-photo-45848.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "sunset_warm": [
        "https://images.pexels.com/photos/87611/sun-fireball-solar-flare-sunlight-87611.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/2884866/pexels-photo-2884866.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/189349/pexels-photo-189349.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "golden_hour": [
        "https://images.pexels.com/photos/1416530/pexels-photo-1416530.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "dramatic_storm": [
        "https://images.pexels.com/photos/1118869/pexels-photo-1118869.jpeg?auto=compress&cs=tinysrgb&w=2400",
        "https://images.pexels.com/photos/53435/clouds-storm-cumulus-cloud-53435.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "overcast_soft": [
        "https://images.pexels.com/photos/96622/pexels-photo-96622.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
    "twilight_blue": [
        "https://images.pexels.com/photos/1118874/pexels-photo-1118874.jpeg?auto=compress&cs=tinysrgb&w=2400",
    ],
}


def download_samples(timeout: float = 30.0) -> dict[str, int]:
    """Download tập sky demo nhỏ (~14 ảnh) từ Pexels CC0 vào library.

    Trả {category: count_added}. Fail silently — KHÔNG block nếu offline.
    """
    try:
        import requests
    except ImportError:
        logger.error("requests chưa cài — pip install requests")
        return {}

    results: dict[str, int] = {}
    for cat, urls in _SAMPLE_PEXELS_URLS.items():
        cat_dir = get_sky_library_dir() / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for url in urls:
            # Tên file từ URL
            name = url.split("/")[-1].split("?")[0]
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                name = f"sky_{cat}_{added}.jpg"
            dst = cat_dir / name
            if dst.exists():
                added += 1
                continue
            try:
                resp = requests.get(
                    url, timeout=timeout, stream=True, headers={"User-Agent": "Mozilla/5.0"}
                )
                if resp.status_code != 200:
                    logger.warning("Pexels %d cho %s", resp.status_code, url)
                    continue
                tmp = dst.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.rename(dst)
                added += 1
                logger.info("Tải %s (%.0f KB)", dst.name, dst.stat().st_size / 1024)
            except Exception as exc:
                logger.warning("Tải %s thất bại: %s", url, exc)
        results[cat] = added
    refresh_index()
    return results
