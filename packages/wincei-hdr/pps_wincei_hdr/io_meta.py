"""EXIF + ICC preservation cho output JPG.

Sau Mertens fusion → output np.ndarray, mất hết metadata.
Hàm này: lấy EXIF + ICC từ reference shot (EV≈0), inject vào output JPG.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def write_jpg_with_meta(
    image_bgr: np.ndarray,
    out_path: Path,
    *,
    reference_path: Path,
    quality: int = 98,
    add_user_comment: str | None = None,
) -> None:
    """Lưu BGR uint8 ra JPG, preserve EXIF + ICC từ reference_path.

    Args:
        image_bgr: ảnh fused BGR uint8.
        out_path: target JPG.
        reference_path: ảnh source để hút EXIF (thường shot EV=0).
        quality: JPEG quality (98 = chuẩn ngành).
        add_user_comment: gắn vào UserComment EXIF (audit trail).
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Pull EXIF + ICC từ reference
    exif_bytes = b""
    icc = b""
    try:
        with Image.open(reference_path) as ref:
            exif_bytes = ref.info.get("exif", b"")
            icc = ref.info.get("icc_profile", b"")
    except Exception as exc:
        log.warning("Reference EXIF read fail: %s", exc)

    # Inject UserComment via piexif (optional audit trail)
    if add_user_comment and exif_bytes:
        try:
            import piexif

            exif_dict = piexif.load(exif_bytes)
            comment = ("ASCII\x00\x00\x00" + add_user_comment).encode("utf-8")
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment
            exif_bytes = piexif.dump(exif_dict)
        except Exception as exc:
            log.warning("UserComment inject fail: %s", exc)

    save_kwargs = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,  # 4:4:4 — chuẩn ngành BĐS
        "optimize": True,
    }
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    if icc:
        save_kwargs["icc_profile"] = icc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, **save_kwargs)
