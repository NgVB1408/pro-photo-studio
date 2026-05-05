"""Lens correction — sửa barrel distortion, chromatic aberration, vignetting.

Pro RE photographer dùng wide-angle 14-24mm thường có:
- **Barrel distortion**: đường thẳng cong ra (ảnh wide)
- **Chromatic aberration**: viền tím/xanh ở high-contrast edges
- **Vignetting**: rìa tối hơn trung tâm

Approach:
- Đọc EXIF lens model + focal length
- Áp profile distortion preset cho lens phổ biến (Canon RF 14-35, Sony 16-35,
  Sigma 14-24, etc.) — Brown-Conrady model
- Nếu không nhận diện được lens → cho user manual sliders
- Chromatic aberration: detect & correct R-channel, B-channel offset từ G

Reference: Brown-Conrady distortion model
    x_d = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r²+2x²) + 2*p2*x*y
    y_d = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r²+2y²)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LensProfile:
    """Distortion params (Brown-Conrady), normalized to image diag = 1.

    Negative k1/k2 = barrel (cong ra), positive = pincushion (cong vào).
    """
    name: str
    focal_min_mm: float = 14.0
    focal_max_mm: float = 35.0
    k1: float = 0.0       # radial 2nd order
    k2: float = 0.0       # radial 4th order
    k3: float = 0.0       # radial 6th order
    p1: float = 0.0       # tangential 1
    p2: float = 0.0       # tangential 2
    vignette_amount: float = 0.0  # -1..0 (vignette dark)
    vignette_midpoint: float = 0.5  # 0..1


# Curated profiles cho lens phổ biến chụp BĐS
# Số liệu xấp xỉ — pro nên hiệu chỉnh thêm nếu cần.
_LENS_PROFILES: dict[str, LensProfile] = {
    # Canon
    "canon_rf_14_35": LensProfile(
        name="Canon RF 14-35mm f/4L IS USM", focal_min_mm=14, focal_max_mm=35,
        k1=-0.18, k2=0.04, vignette_amount=-0.35,
    ),
    "canon_rf_15_35": LensProfile(
        name="Canon RF 15-35mm f/2.8L IS USM", focal_min_mm=15, focal_max_mm=35,
        k1=-0.15, k2=0.03, vignette_amount=-0.30,
    ),
    "canon_ef_16_35_iii": LensProfile(
        name="Canon EF 16-35mm f/2.8L III USM", focal_min_mm=16, focal_max_mm=35,
        k1=-0.12, k2=0.02, vignette_amount=-0.28,
    ),
    # Sony
    "sony_fe_14_24": LensProfile(
        name="Sony FE 14-24mm f/2.8 GM", focal_min_mm=14, focal_max_mm=24,
        k1=-0.16, k2=0.05, vignette_amount=-0.32,
    ),
    "sony_fe_16_35": LensProfile(
        name="Sony FE 16-35mm f/2.8 GM", focal_min_mm=16, focal_max_mm=35,
        k1=-0.13, k2=0.03, vignette_amount=-0.30,
    ),
    # Nikon
    "nikon_z_14_24": LensProfile(
        name="Nikon Z 14-24mm f/2.8 S", focal_min_mm=14, focal_max_mm=24,
        k1=-0.17, k2=0.04, vignette_amount=-0.34,
    ),
    "nikon_z_14_30": LensProfile(
        name="Nikon Z 14-30mm f/4 S", focal_min_mm=14, focal_max_mm=30,
        k1=-0.15, k2=0.04, vignette_amount=-0.32,
    ),
    # Sigma
    "sigma_art_14_24": LensProfile(
        name="Sigma 14-24mm f/2.8 DG HSM Art", focal_min_mm=14, focal_max_mm=24,
        k1=-0.18, k2=0.05, vignette_amount=-0.36,
    ),
    "sigma_art_24_70": LensProfile(
        name="Sigma 24-70mm f/2.8 DG OS HSM Art", focal_min_mm=24, focal_max_mm=70,
        k1=-0.05, k2=0.01, vignette_amount=-0.20,
    ),
    # Tamron
    "tamron_17_28": LensProfile(
        name="Tamron 17-28mm f/2.8 Di III RXD", focal_min_mm=17, focal_max_mm=28,
        k1=-0.13, k2=0.03, vignette_amount=-0.30,
    ),
    # Sample/test - distortionless reference
    "rectilinear_50": LensProfile(
        name="Rectilinear 50mm reference (no distortion)",
        focal_min_mm=50, focal_max_mm=85,
    ),
}


def list_lens_profiles() -> list[dict]:
    """Liệt kê tất cả profile có sẵn."""
    return [
        {"id": k, "name": v.name,
         "focal_range": f"{v.focal_min_mm:.0f}-{v.focal_max_mm:.0f}mm",
         "k1": v.k1, "vignette": v.vignette_amount}
        for k, v in _LENS_PROFILES.items()
    ]


def get_profile(profile_id: str) -> LensProfile | None:
    return _LENS_PROFILES.get(profile_id)


def correct_distortion(
    img: np.ndarray,
    profile: LensProfile,
    *,
    intensity: float = 1.0,
) -> np.ndarray:
    """Áp Brown-Conrady distortion correction.

    intensity: 0..1, mức áp profile (1.0 = full).
    """
    h, w = img.shape[:2]
    # Tâm ảnh
    cx, cy = w / 2.0, h / 2.0
    # Normalize bằng diagonal/2 (tỉ lệ chuẩn)
    norm = np.sqrt(w * w + h * h) / 2.0

    # Build undistortion map
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - cx) / norm
    y = (yy - cy) / norm
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion factor (Brown-Conrady forward — to undistort, áp inverse)
    # Để undistort: tìm coords nguồn cho mỗi target (x, y).
    # Approach simple: forward distortion với negative k1/k2 (vì lens barrel có
    # k1 < 0, để undistort cần k1 > 0 trong correction map)
    k1 = -profile.k1 * intensity
    k2 = -profile.k2 * intensity
    k3 = -profile.k3 * intensity
    p1 = -profile.p1 * intensity
    p2 = -profile.p2 * intensity

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Map về pixel coords
    map_x = (x_d * norm + cx).astype(np.float32)
    map_y = (y_d * norm + cy).astype(np.float32)

    out = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out


def correct_vignetting(
    img: np.ndarray,
    profile: LensProfile,
    *,
    intensity: float = 1.0,
) -> np.ndarray:
    """Sửa vignetting (rìa tối). intensity 0..1."""
    if abs(profile.vignette_amount) < 1e-3:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    norm = np.sqrt(w * w + h * h) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / norm
    midpoint = profile.vignette_midpoint
    # Anti-vignette gain: max ở rìa, =1 ở midpoint
    # Profile vignette_amount âm = lens darken rìa → ta lift để correct
    correction_amount = -profile.vignette_amount * intensity
    gain = 1.0 + correction_amount * np.clip(
        (r - midpoint) / (1.0 - midpoint), 0, 1,
    ) ** 2
    gain = gain[..., None]
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def correct_chromatic_aberration(
    img: np.ndarray,
    *,
    radial_amount: float = 0.001,
) -> np.ndarray:
    """CA correction — scale R/B channels nhẹ để giảm fringing.

    radial_amount: 0..0.005, mức scale (0.001 = 0.1% thường đủ).
    """
    if abs(radial_amount) < 1e-5:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Scale R về center 1+amount, B về 1-amount
    # Build separate maps cho R và B
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    # Map cho R: pull về center (scale > 1 means sample from outside)
    map_x_r = cx + dx * (1.0 - radial_amount)
    map_y_r = cy + dy * (1.0 - radial_amount)
    # Map cho B: ngược lại
    map_x_b = cx + dx * (1.0 + radial_amount)
    map_y_b = cy + dy * (1.0 + radial_amount)

    b, g, r = cv2.split(img)
    r_corr = cv2.remap(
        r, map_x_r.astype(np.float32), map_y_r.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    b_corr = cv2.remap(
        b, map_x_b.astype(np.float32), map_y_b.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.merge([b_corr, g, r_corr])


def detect_lens_from_exif(image_path: str | Path) -> dict:
    """Đọc EXIF + tìm lens model. Trả {lens_model, focal_mm, profile_id}.

    Cần Pillow. Nếu không có Pillow hoặc EXIF không có info → trả empty.
    """
    info: dict = {"lens_model": None, "focal_mm": None, "profile_id": None}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return info

    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
    except Exception:  # noqa: BLE001
        return info

    tag_map = {TAGS.get(k, k): v for k, v in exif.items()}
    lens_model = tag_map.get("LensModel") or tag_map.get("LensMake") or ""
    focal = tag_map.get("FocalLength")
    if isinstance(focal, tuple):
        focal = focal[0] / focal[1] if focal[1] else None

    info["lens_model"] = str(lens_model) if lens_model else None
    info["focal_mm"] = float(focal) if focal else None

    # Map lens_model string → profile_id
    if lens_model:
        lower = str(lens_model).lower()
        if "rf" in lower and "14-35" in lower:
            info["profile_id"] = "canon_rf_14_35"
        elif "rf" in lower and "15-35" in lower:
            info["profile_id"] = "canon_rf_15_35"
        elif "ef" in lower and "16-35" in lower:
            info["profile_id"] = "canon_ef_16_35_iii"
        elif "fe 14-24" in lower or "14-24mm gm" in lower:
            info["profile_id"] = "sony_fe_14_24"
        elif "fe 16-35" in lower:
            info["profile_id"] = "sony_fe_16_35"
        elif "z 14-24" in lower or "nikkor z 14-24" in lower:
            info["profile_id"] = "nikon_z_14_24"
        elif "z 14-30" in lower or "nikkor z 14-30" in lower:
            info["profile_id"] = "nikon_z_14_30"
        elif "sigma" in lower and "14-24" in lower:
            info["profile_id"] = "sigma_art_14_24"
        elif "sigma" in lower and "24-70" in lower:
            info["profile_id"] = "sigma_art_24_70"
        elif "tamron" in lower and ("17-28" in lower):
            info["profile_id"] = "tamron_17_28"
    return info


def auto_correct_lens(
    img: np.ndarray,
    *,
    image_path: str | Path | None = None,
    profile_id: str | None = None,
    intensity: float = 1.0,
    correct_chromatic: bool = True,
) -> tuple[np.ndarray, dict]:
    """Auto-correct lens: detect EXIF → áp profile + CA correction.

    Args:
        image_path: nếu có, đọc EXIF từ file.
        profile_id: chỉ định trực tiếp profile (override EXIF).
        intensity: 0..1 mức áp.
        correct_chromatic: True = áp CA correction (subtle).

    Returns:
        (corrected BGR, info dict).
    """
    info: dict = {"applied": False}
    pid = profile_id
    if not pid and image_path:
        exif_info = detect_lens_from_exif(image_path)
        info.update(exif_info)
        pid = exif_info.get("profile_id")

    if not pid:
        info["reason"] = "Không nhận diện được lens — bỏ qua correction"
        return img.copy(), info

    profile = get_profile(pid)
    if profile is None:
        info["reason"] = f"Profile {pid} không tồn tại"
        return img.copy(), info

    out = correct_distortion(img, profile, intensity=intensity)
    out = correct_vignetting(out, profile, intensity=intensity)
    if correct_chromatic:
        out = correct_chromatic_aberration(out, radial_amount=0.0008)
    info.update({
        "applied": True,
        "profile_id": pid,
        "profile_name": profile.name,
        "k1": profile.k1,
        "vignette": profile.vignette_amount,
    })
    return out, info
