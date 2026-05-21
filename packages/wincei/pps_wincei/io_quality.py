"""Quality I/O — read/write images preserving EXIF, ICC profile, resolution, format.

Real-estate constraint:
    - JPG → JPG quality 98, EXIF preserved, ICC preserved.
    - PNG → PNG with optimization.
    - TIFF → 16-bit if input is 16-bit, LZW compression, EXIF preserved.

OpenCV is fast for computation but loses EXIF/ICC.
We use OpenCV for pipeline math, then Pillow to write metadata back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ImageMeta:
    """Metadata carried alongside pixel data — preserved across pipeline."""

    format: str  # "JPEG" / "PNG" / "TIFF" / "WEBP"
    mode: str  # "RGB" / "RGBA" / "I;16"
    width: int
    height: int
    is_16bit: bool
    exif_bytes: bytes | None = None
    icc_profile: bytes | None = None
    dpi: tuple[float, float] | None = None
    orientation: int = 1


def read_image(path: str | Path) -> tuple[np.ndarray, ImageMeta]:
    """Read image as BGR uint8 ndarray + ImageMeta (EXIF/ICC preserved).

    Returns:
        (bgr_array, meta)

    Notes:
        - 16-bit TIFF read as 16-bit then DOWNCAST to 8-bit for pipeline math
          (re-upcast on write if input was 16-bit).
        - EXIF orientation applied to pixel array so pipeline sees upright image.
        - Pillow used for metadata extraction; OpenCV for raw pixel decode (faster).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # Use Pillow for metadata + orientation correction
    with Image.open(p) as pim:
        format_name = pim.format or "JPEG"
        mode = pim.mode
        exif_bytes = pim.info.get("exif")
        icc_profile = pim.info.get("icc_profile")
        dpi = pim.info.get("dpi")
        orientation = 1
        try:
            exif_dict = pim.getexif()
            orientation = int(exif_dict.get(0x0112, 1))
        except Exception:
            pass
        is_16bit = pim.mode in {"I;16", "I;16B", "I;16L"} or (
            format_name == "TIFF" and "16" in pim.mode
        )
        w, h = pim.size

    # Read pixels via OpenCV (fast)
    bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        # Fallback to Pillow for non-cv-supported formats
        with Image.open(p) as pim:
            pim = pim.convert("RGB")
            arr = np.array(pim)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Apply EXIF orientation to pixels so pipeline sees upright image
    bgr = _apply_orientation(bgr, orientation)
    # Update meta after orientation rotation
    h2, w2 = bgr.shape[:2]

    meta = ImageMeta(
        format=format_name,
        mode=mode,
        width=w2,
        height=h2,
        is_16bit=is_16bit,
        exif_bytes=exif_bytes,
        icc_profile=icc_profile,
        dpi=dpi,
        orientation=1,  # reset to 1 since we baked it into pixels
    )
    return bgr, meta


def _apply_orientation(bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation tag to pixel array (baked in)."""
    if orientation == 1:
        return bgr
    if orientation == 2:
        return cv2.flip(bgr, 1)
    if orientation == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(bgr, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def write_image(
    bgr: np.ndarray,
    path: str | Path,
    meta: ImageMeta,
    *,
    jpeg_quality: int = 98,
    png_compress: int = 6,
    tiff_compression: str = "tiff_lzw",
) -> None:
    """Write BGR uint8 array to path with metadata preserved.

    Format derived from `meta.format` (input format-faithful by default), unless
    `path` extension overrides.

    Args:
        jpeg_quality: 1-100. Default 98 = visually lossless.
        png_compress: 0-9. Default 6 = balance.
        tiff_compression: 'tiff_lzw' (lossless) | 'tiff_deflate' | 'raw'.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    # Convert BGR → RGB for Pillow
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pim = Image.fromarray(rgb, mode="RGB")

    save_kwargs: dict = {}
    if meta.exif_bytes:
        save_kwargs["exif"] = meta.exif_bytes
    if meta.icc_profile:
        save_kwargs["icc_profile"] = meta.icc_profile
    if meta.dpi:
        save_kwargs["dpi"] = meta.dpi

    if ext in (".jpg", ".jpeg"):
        save_kwargs.update(
            quality=jpeg_quality,
            subsampling=0,  # 4:4:4 chroma — max quality
            optimize=True,
            progressive=False,
        )
        pim.save(p, "JPEG", **save_kwargs)
    elif ext == ".png":
        save_kwargs["compress_level"] = png_compress
        pim.save(p, "PNG", **save_kwargs)
    elif ext in (".tif", ".tiff"):
        save_kwargs["compression"] = tiff_compression
        pim.save(p, "TIFF", **save_kwargs)
    elif ext == ".webp":
        save_kwargs.update(quality=98, method=6, lossless=False)
        pim.save(p, "WEBP", **save_kwargs)
    else:
        raise ValueError(f"Định dạng output không hỗ trợ: {ext}")

    pim.close()


def make_thumbnail(
    bgr: np.ndarray, max_side: int = 1024
) -> np.ndarray:
    """Resize to max side for HTML comparison viewer (keeps aspect)."""
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
