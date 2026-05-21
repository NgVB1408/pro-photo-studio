"""Image pre-processing cho ảnh khó (HDR/fisheye) trước khi gửi VLM/SAM.

2 kỹ thuật:
    [A] CLAHE — cân bằng vùng tối (trần ám) + vùng sáng (cửa sổ blown)
    [B] Undistort — bẻ thẳng fisheye/ultra-wide lens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    clahe_applied: bool = False
    undistort_applied: bool = False
    original_low_contrast: bool = False
    detected_fisheye: bool = False
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def apply_clahe(
    image_bgr: np.ndarray,
    *,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
    apply_on_lab: bool = True,
) -> np.ndarray:
    """CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Áp dụng trên kênh L của LAB (giữ màu, chỉ chỉnh luminance).
    Vùng tối (trần ám) sáng lên, vùng sáng (cửa sổ blown) compress.
    """
    if apply_on_lab:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    # Fallback: gray (loses color contrast)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_eq = clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)


def _detect_low_contrast(image_bgr: np.ndarray, *, threshold: float = 40.0) -> bool:
    """Auto-detect ảnh thiếu contrast (cần CLAHE).

    Threshold: std(L) < 40 → áp CLAHE.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_std = lab[:, :, 0].std()
    return l_std < threshold


def _detect_fisheye(image_bgr: np.ndarray) -> bool:
    """Heuristic detect fisheye qua đường biên: nếu EXIF có lens fisheye thì True.

    Fallback: detect circular dark vignette ở 4 góc (typical fisheye signature).
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Check 4 corner brightness vs center
    corner_size = min(h, w) // 8
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    corner_mean = np.mean([c.mean() for c in corners])
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_mean = center.mean()
    # If corners much darker than center → likely vignette → fisheye/ultra-wide
    return corner_mean < center_mean * 0.55


def undistort_image(
    image_bgr: np.ndarray,
    *,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    fisheye: bool = False,
) -> np.ndarray:
    """Bẻ thẳng ảnh fisheye/ultra-wide qua cv2.undistort.

    Args:
        image_bgr: input ảnh.
        camera_matrix: (3,3) intrinsic K (default = generic estimate từ ảnh).
        dist_coeffs: (4,) hoặc (5,) distortion coefficients.
        fisheye: True nếu dùng cv2.fisheye.undistortImage.

    Returns:
        Undistorted BGR uint8.
    """
    h, w = image_bgr.shape[:2]

    if camera_matrix is None:
        # Generic estimate: assume 24mm equivalent (typical real estate wide-angle)
        # Focal length ~ image width / (2 * tan(half_fov))
        # For 90° FoV → f ≈ w / 2
        f = w / 2.0
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1.0],
        ], dtype=np.float64)

    if dist_coeffs is None:
        # Generic mild barrel distortion (k1=-0.15)
        dist_coeffs = np.array([-0.15, 0.05, 0, 0, 0], dtype=np.float64)

    if fisheye:
        # Fisheye model needs (k1, k2, k3, k4)
        k = dist_coeffs[:4] if len(dist_coeffs) >= 4 else np.array([-0.15, 0.05, 0, 0])
        new_K, _ = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, k, (w, h), np.eye(3), balance=0.5,
        )
        return cv2.fisheye.undistortImage(image_bgr, camera_matrix, k, Knew=new_K)

    return cv2.undistort(image_bgr, camera_matrix, dist_coeffs)


def preprocess_for_vlm_sam(
    image_bgr: np.ndarray,
    *,
    auto_clahe: bool = True,
    auto_undistort: bool = False,
    force_clahe: bool = False,
    force_undistort: bool = False,
    fisheye_mode: bool = False,
) -> tuple[np.ndarray, PreprocessReport]:
    """One-shot preprocess: CLAHE + undistort khi cần thiết.

    Args:
        image_bgr: input.
        auto_clahe: True = detect + apply nếu low contrast.
        auto_undistort: True = detect + apply nếu fisheye signature.
        force_clahe / force_undistort: force apply bất kể detect.
        fisheye_mode: dùng cv2.fisheye thay vì standard undistort.

    Returns:
        (processed_image, report)
    """
    report = PreprocessReport()
    out = image_bgr.copy()

    # CLAHE
    low_contrast = _detect_low_contrast(out)
    report.original_low_contrast = low_contrast
    if force_clahe or (auto_clahe and low_contrast):
        out = apply_clahe(out, clip_limit=2.5)
        report.clahe_applied = True
        report.notes.append(f"CLAHE applied (low_contrast={low_contrast})")

    # Undistort
    fisheye_detected = _detect_fisheye(out)
    report.detected_fisheye = fisheye_detected
    if force_undistort or (auto_undistort and fisheye_detected):
        out = undistort_image(out, fisheye=fisheye_mode)
        report.undistort_applied = True
        report.notes.append(f"Undistort applied (fisheye={fisheye_detected})")

    return out, report
