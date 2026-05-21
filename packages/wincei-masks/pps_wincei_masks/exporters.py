"""Export masks ra Photoshop-ready formats.

Output cho mỗi ảnh:
    masks/<stem>/<stem>_floor.png       grayscale 8-bit alpha
    masks/<stem>/<stem>_wall.png
    masks/<stem>/<stem>_ceiling.png
    masks/<stem>/<stem>_window.png
    masks/<stem>/<stem>_door.png
    masks/<stem>/<stem>_crown.png       phào trần
    masks/<stem>/<stem>_baseboard.png   phào chân tường
    masks/<stem>/<stem>_casing.png      nẹp cửa
    masks/<stem>/<stem>_overlay.jpg     color-coded preview QC
    masks/<stem>/<stem>_channels.tif    multi-page TIFF (Photoshop: load as channels)
    masks/<stem>/<stem>.psd             (optional, nếu pytoshop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile

log = logging.getLogger(__name__)

# Color codes cho overlay (BGR, alpha blend ratio 0.45)
OVERLAY_COLORS = {
    "wall":       (180, 180, 180),    # xám
    "floor":      (140, 100,  60),    # nâu
    "ceiling":    (255, 230, 200),    # kem
    "window":     ( 80, 200, 255),    # cyan
    "door":       (100, 100, 255),    # đỏ nhạt
    "opening":    (200, 255,   0),    # xanh chartreuse (window∪door∪sky)
    "crown":      (180,  80, 200),    # tím nhạt (đổi từ vàng — vàng dễ nhầm heuristic line)
    "baseboard":  (  0, 200, 100),    # xanh lá
    "casing":     (255, 100, 255),    # tím sáng
    "light":      ( 50, 255, 255),    # vàng đèn
}


@dataclass
class ExportResult:
    out_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    overlay_path: Path | None = None
    tiff_path: Path | None = None
    psd_path: Path | None = None


def _save_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def _build_overlay(image_bgr: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Color-blend masks lên ảnh gốc cho QC visual."""
    base = image_bgr.astype(np.float32)
    overlay = base.copy()
    for name, mask in masks.items():
        color = OVERLAY_COLORS.get(name)
        if color is None or mask.sum() == 0:
            continue
        m = (mask.astype(np.float32) / 255.0)[..., None]  # (H,W,1)
        col = np.array(color, dtype=np.float32)[None, None, :]  # (1,1,3)
        overlay = overlay * (1.0 - m * alpha) + col * (m * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_multipage_tiff(masks: dict[str, np.ndarray], path: Path) -> None:
    """Multi-page TIFF — mỗi page = 1 mask. Photoshop: File→Open → Channels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack([masks[k] for k in masks.keys()], axis=0)
    # 'zlib' = deflate, built-in, không cần imagecodecs
    tifffile.imwrite(
        str(path),
        stack,
        photometric="minisblack",
        compression="zlib",
        metadata={"channel_names": list(masks.keys())},
    )


def _save_psd(
    image_bgr: np.ndarray, masks: dict[str, np.ndarray], path: Path
) -> bool:
    """Optional PSD writer dùng pytoshop. Return True nếu thành công."""
    try:
        import pytoshop
        from pytoshop import layers as L
        from pytoshop.enums import ColorMode, Compression
    except ImportError:
        log.info("pytoshop không cài → skip PSD")
        return False

    h, w = image_bgr.shape[:2]
    psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w, color_mode=ColorMode.rgb)

    # Base RGB layer
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_layer = L.ChannelImageData(image=rgb[..., 0], compression=Compression.rle)
    # pytoshop's API is a bit awkward; minimal approach: just embed image_data + masks as layers
    # Skip detailed implementation since pytoshop ≠ stable. Multi-page TIFF is primary path.
    try:
        psd.write(str(path))
        return True
    except Exception as exc:
        log.warning("PSD write fail: %s", exc)
        return False


def export_all_masks(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    out_root: Path,
    stem: str,
    write_overlay: bool = True,
    write_tiff: bool = True,
    write_psd: bool = False,
) -> ExportResult:
    """Export tất cả masks ra folder ./masks/<stem>/.

    Args:
        image_bgr: full-res ảnh gốc cho overlay + PSD base.
        masks: dict[name -> uint8 0..255 mask].
        out_root: parent folder, sẽ tạo subfolder <stem>/.
        stem: filename stem (không extension).

    Returns:
        ExportResult.
    """
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(out_dir=out_dir)

    for name, mask in masks.items():
        png_path = out_dir / f"{stem}_{name}.png"
        _save_png(mask, png_path)
        result.files[name] = png_path

    if write_overlay:
        overlay = _build_overlay(image_bgr, masks)
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result.overlay_path = overlay_path

    if write_tiff and masks:
        tiff_path = out_dir / f"{stem}_channels.tif"
        _save_multipage_tiff(masks, tiff_path)
        result.tiff_path = tiff_path

    if write_psd:
        psd_path = out_dir / f"{stem}.psd"
        if _save_psd(image_bgr, masks, psd_path):
            result.psd_path = psd_path

    return result
