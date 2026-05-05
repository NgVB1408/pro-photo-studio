"""Pre-flight QC: phân tích ảnh đầu vào trước khi chạy enhance pipeline.

Mục tiêu: phát hiện ảnh không cứu được (quá blur / quá tối / quá nhỏ /
EXIF báo ISO cao + shutter dài) → flag retake để photographer biết chụp lại
trước khi tốn thời gian batch enhance.

Triết lý: KHÔNG block — vẫn xử lý bình thường — nhưng emit warning vào log
+ ghi cột severity vào processing_report.csv để photographer review nhanh.

Đo lường:
    - blur_score:    cv2.Laplacian variance (>500 = sharp, <100 = blur nặng)
    - exposure:      % pixel cháy (V≥250) + % pixel chìm (V≤5)
    - dimension:     min(h, w) — < 1080 = thấp, < 720 = cảnh báo
    - color_cast:    a*/b* mean trong LAB (lệch >18 = WB lệch nặng)
    - focus_uniform: blur_score chênh lệch giữa 9 ô grid → soft focus

Severity:
    "ok"   — không có warning
    "info" — minor (vd ảnh hơi tối)
    "warn" — nên xem lại (blur nhẹ / exposure clipping ≥10%)
    "fail" — gần như không cứu được — đề xuất retake
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Severity = Literal["ok", "info", "warn", "fail"]


@dataclass
class PreflightReport:
    severity: Severity = "ok"
    warnings: list[str] = field(default_factory=list)
    blur_score: float = 0.0
    blur_uniform: float = 0.0
    highlight_clip_pct: float = 0.0
    shadow_clip_pct: float = 0.0
    avg_brightness: float = 0.0
    color_cast: float = 0.0
    width: int = 0
    height: int = 0
    suggested_action: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        # Round floats để CSV gọn
        for k in (
            "blur_score",
            "blur_uniform",
            "highlight_clip_pct",
            "shadow_clip_pct",
            "avg_brightness",
            "color_cast",
        ):
            d[k] = round(float(d[k]), 2)
        return d

    def csv_summary(self) -> str:
        if not self.warnings:
            return "ok"
        return f"{self.severity}: {'; '.join(self.warnings)}"


# ===== Thresholds (tune để khớp với sensor DSLR/mirrorless điển hình) =====

BLUR_FAIL = 60.0  # < 60 var = motion blur / out-of-focus rõ rệt
BLUR_WARN = 150.0  # < 150 var = soft, có thể chấp nhận listing nhỏ
BLUR_OK = 400.0  # ≥ 400 var = sharp

CLIP_HIGHLIGHT_FAIL = 18.0  # ≥18% pixel cháy → quá nhiều, recover hạn chế
CLIP_HIGHLIGHT_WARN = 8.0
CLIP_SHADOW_FAIL = 25.0
CLIP_SHADOW_WARN = 12.0

BRIGHTNESS_DARK_FAIL = 35.0  # mean V < 35 → quá tối
BRIGHTNESS_DARK_WARN = 60.0
BRIGHTNESS_OVER_WARN = 220.0

DIMENSION_FAIL = 720  # short side < 720 = MLS reject
DIMENSION_WARN = 1080

COLOR_CAST_WARN = 14.0  # |a*| or |b*| mean offset từ neutral 128
COLOR_CAST_FAIL = 22.0


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian — chuẩn đo focus/blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _grid_blur_uniform(gray: np.ndarray, grid: int = 3) -> float:
    """Trả std/mean của blur_score trên grid 3x3 — soft-focus có std thấp,
    motion blur 1 phần ảnh có std cao.
    Score = std / mean. < 0.15 = uniform sharp/blur, > 0.6 = lệch rõ.
    """
    h, w = gray.shape[:2]
    cells: list[float] = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = h * i // grid, h * (i + 1) // grid
            x0, x1 = w * j // grid, w * (j + 1) // grid
            tile = gray[y0:y1, x0:x1]
            if tile.size > 0:
                cells.append(_laplacian_var(tile))
    if not cells:
        return 0.0
    arr = np.asarray(cells, dtype=np.float64)
    if arr.mean() < 1e-3:
        return 0.0
    return float(arr.std() / arr.mean())


def analyze_image(img: np.ndarray) -> PreflightReport:
    """Phân tích numpy BGR uint8 → PreflightReport."""
    if img is None or img.size == 0:
        rpt = PreflightReport(severity="fail")
        rpt.warnings.append("Ảnh rỗng / không decode được")
        rpt.suggested_action = "retake"
        return rpt

    h, w = img.shape[:2]
    rpt = PreflightReport(width=w, height=h)
    short_side = min(h, w)

    # Downscale để measure blur/exposure nhanh — không thay đổi thực tế
    work = img
    if short_side > 1500:
        scale = 1500.0 / short_side
        work = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if work.ndim == 2:
        gray = work
        bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    elif work.shape[2] == 4:
        bgr = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        bgr = work
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    rpt.blur_score = _laplacian_var(gray)
    rpt.blur_uniform = _grid_blur_uniform(gray)

    # Brightness + clipping qua channel V của HSV
    v_chan = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    total = v_chan.size
    rpt.avg_brightness = float(v_chan.mean())
    rpt.highlight_clip_pct = float((v_chan >= 250).sum()) / total * 100.0
    rpt.shadow_clip_pct = float((v_chan <= 5).sum()) / total * 100.0

    # WB / color cast — LAB a*,b* mean offset từ 128
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    rpt.color_cast = float(
        max(abs(float(lab[..., 1].mean()) - 128.0), abs(float(lab[..., 2].mean()) - 128.0))
    )

    # ===== Decide severity + warnings =====
    severities: list[Severity] = []

    if rpt.blur_score < BLUR_FAIL:
        rpt.warnings.append(f"Blur nặng (Laplacian var={rpt.blur_score:.0f})")
        severities.append("fail")
    elif rpt.blur_score < BLUR_WARN:
        rpt.warnings.append(f"Soft focus (Laplacian var={rpt.blur_score:.0f})")
        severities.append("warn")

    if rpt.blur_uniform > 0.6 and rpt.blur_score >= BLUR_FAIL:
        rpt.warnings.append("Focus không đều (1 phần ảnh blur, 1 phần sharp)")
        severities.append("info")

    if rpt.highlight_clip_pct >= CLIP_HIGHLIGHT_FAIL:
        rpt.warnings.append(f"Cháy {rpt.highlight_clip_pct:.1f}% pixel — bracket exposure cần")
        severities.append("fail")
    elif rpt.highlight_clip_pct >= CLIP_HIGHLIGHT_WARN:
        rpt.warnings.append(f"Highlight clipping {rpt.highlight_clip_pct:.1f}%")
        severities.append("warn")

    if rpt.shadow_clip_pct >= CLIP_SHADOW_FAIL:
        rpt.warnings.append(f"Chìm {rpt.shadow_clip_pct:.1f}% pixel — bù sáng / fill flash")
        severities.append("fail")
    elif rpt.shadow_clip_pct >= CLIP_SHADOW_WARN:
        rpt.warnings.append(f"Shadow clipping {rpt.shadow_clip_pct:.1f}%")
        severities.append("warn")

    if rpt.avg_brightness < BRIGHTNESS_DARK_FAIL:
        rpt.warnings.append(f"Quá tối (avg V={rpt.avg_brightness:.0f}/255)")
        severities.append("fail")
    elif rpt.avg_brightness < BRIGHTNESS_DARK_WARN:
        rpt.warnings.append(f"Hơi tối (avg V={rpt.avg_brightness:.0f}/255)")
        severities.append("info")
    elif rpt.avg_brightness > BRIGHTNESS_OVER_WARN:
        rpt.warnings.append(f"Quá sáng (avg V={rpt.avg_brightness:.0f}/255)")
        severities.append("warn")

    if rpt.color_cast >= COLOR_CAST_FAIL:
        rpt.warnings.append(f"WB lệch nặng (LAB cast={rpt.color_cast:.1f})")
        severities.append("warn")
    elif rpt.color_cast >= COLOR_CAST_WARN:
        rpt.warnings.append(f"WB hơi lệch (LAB cast={rpt.color_cast:.1f})")
        severities.append("info")

    if short_side < DIMENSION_FAIL:
        rpt.warnings.append(f"Quá nhỏ ({w}×{h}, MLS yêu cầu ≥1024px)")
        severities.append("fail")
    elif short_side < DIMENSION_WARN:
        rpt.warnings.append(f"Resolution thấp ({w}×{h})")
        severities.append("info")

    # Combine: severity = max( severities )
    rank = {"ok": 0, "info": 1, "warn": 2, "fail": 3}
    if severities:
        rpt.severity = max(severities, key=lambda s: rank[s])

    # Suggest action
    if rpt.severity == "fail":
        if any("Blur" in w_msg for w_msg in rpt.warnings):
            rpt.suggested_action = "retake (blur)"
        elif any("Cháy" in w_msg or "Chìm" in w_msg for w_msg in rpt.warnings):
            rpt.suggested_action = "bracket / retake (exposure)"
        elif any("tối" in w_msg.lower() for w_msg in rpt.warnings):
            rpt.suggested_action = "retake (under-exposed)"
        elif any("nhỏ" in w_msg for w_msg in rpt.warnings):
            rpt.suggested_action = "retake (resolution)"
        else:
            rpt.suggested_action = "retake"
    elif rpt.severity == "warn":
        rpt.suggested_action = "review carefully"
    elif rpt.severity == "info":
        rpt.suggested_action = "ok (minor)"

    return rpt


def analyze_file(path: str | Path) -> PreflightReport:
    """Phân tích 1 file ảnh → PreflightReport."""
    from .utils import read_image

    p = Path(path)
    try:
        img = read_image(p)
    except Exception as exc:
        rpt = PreflightReport(severity="fail")
        rpt.warnings.append(f"Đọc ảnh fail: {exc}")
        rpt.suggested_action = "skip (cannot decode)"
        return rpt
    return analyze_image(img)
