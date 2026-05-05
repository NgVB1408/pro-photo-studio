from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, *, max_len: int = 80) -> str:
    """Chuẩn hoá tên file: bỏ ký tự nguy hiểm, giới hạn độ dài."""
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._") or "file"
    return cleaned[:max_len]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


RAW_EXTS = {
    ".dng", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".raf", ".rw2", ".orf", ".pef", ".srw", ".crw", ".kdc", ".dcr",
    ".mrw", ".rwl", ".x3f", ".3fr", ".iiq", ".fff",
}


def read_image(path: str | Path) -> np.ndarray:
    """Đọc ảnh — JPG/PNG/WebP/TIFF qua OpenCV, RAW (.dng/.cr2/.nef/.arw...)
    qua rawpy demosaic. Giữ alpha nếu có.

    Trả: BGR/BGRA uint8.

    Raises:
        FileNotFoundError nếu path không tồn tại
        ImportError nếu RAW nhưng `rawpy` không cài (gợi ý cài [raw] extra)
        ValueError nếu decode fail
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {p}")

    if p.suffix.lower() in RAW_EXTS:
        return _read_raw(p)

    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không đọc được ảnh (định dạng không hỗ trợ?): {p}")
    if img.ndim == 2:  # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _read_raw(p: Path) -> np.ndarray:
    """Demosaic RAW file → BGR uint8.

    Dùng default postprocess: AHD demosaic, no-WB-override (preserve camera
    WB), gamma sRGB curve, 8-bit output. Để pipeline downstream xử lý nâng
    cao tonemap / exposure như JPG bình thường.

    Lưu ý: 8-bit cho compatibility — dynamic range thật của RAW (12-14bit)
    sẽ mất; trong tương lai có thể đổi sang `output_bps=16` rồi extend
    pipeline lên uint16. Hiện tại pipeline OpenCV chủ yếu uint8 → giữ 8-bit
    để tránh đổi rộng.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError(
            f"Đọc RAW {p.suffix.upper()} cần `rawpy`. Cài: "
            f"`pip install -e .[raw]` hoặc `pip install rawpy`."
        ) from exc

    try:
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                gamma=(2.222, 4.5),  # sRGB
                output_color=rawpy.ColorSpace.sRGB,
            )
    except (rawpy.LibRawError, RuntimeError, OSError) as exc:
        raise ValueError(f"Decode RAW fail ({p.name}): {exc}") from exc

    # rawpy trả RGB → convert BGR cho OpenCV pipeline
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.info("RAW loaded: %s → %dx%d BGR uint8", p.name, bgr.shape[1], bgr.shape[0])
    return bgr


def write_image(
    path: str | Path,
    image: np.ndarray,
    *,
    quality: int = 95,
    exif_source: str | Path | None = None,
) -> Path:
    """Lưu ảnh, hỗ trợ tên Unicode (Windows) qua imencode + write bytes.

    exif_source: nếu set + format hỗ trợ (jpg/jpeg/webp), copy EXIF từ ảnh nguồn.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, max(1, min(100, quality))]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Không encode được ảnh sang {ext}")
    p.write_bytes(buf.tobytes())

    if exif_source and ext in {".jpg", ".jpeg", ".webp"}:
        try:
            _copy_exif(exif_source, p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không copy được EXIF từ %s: %s", exif_source, exc)
    return p


def _copy_exif(src: str | Path, dst: str | Path) -> None:
    """Copy EXIF từ src sang dst dùng PIL (không cần piexif)."""
    from PIL import Image

    src_path, dst_path = Path(src), Path(dst)
    src_img = Image.open(src_path)
    exif = src_img.info.get("exif")
    if not exif:
        return
    dst_img = Image.open(dst_path)
    dst_img.save(dst_path, exif=exif, quality="keep" if dst_path.suffix.lower() in {".jpg", ".jpeg"} else None)


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        logger.info("%s: %.3fs", label, dt)
