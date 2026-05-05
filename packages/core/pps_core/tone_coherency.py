"""Tone coherency — đảm bảo TẤT CẢ ảnh trong batch có cùng "voice".

Hai chế độ:

1. Static preset (`TonePreset`): bias cố định warm/cool/neutral.
2. Dynamic batch anchor (`BatchToneFitter` → `BatchAnchor`): học LAB median
   của cả batch rồi nudge từng ảnh về anchor → ảnh nào lệch nhiều bị kéo
   nhiều, ảnh đã đúng tone không thay đổi. Mode này dùng khi user chọn
   `tone_preset == "auto_batch"`.

Kết quả: 30 ảnh shoot 1 căn nhà → cùng tone, feel "1 ngày 1 không khí",
không bị mỗi ảnh 1 tone do scene-adaptive WB ngẫu nhiên.

API:
    cohesion = TonePreset("warm", strength=0.5)
    cohesion.apply(img) -> processed BGR

    fitter = BatchToneFitter()
    for p in paths: fitter.add_from_path(p)
    anchor = fitter.fit_anchor()
    img = anchor.apply(img, strength=0.6)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PresetName = Literal["neutral", "warm", "cool", "auto", "auto_batch", "real_estate"]


def tone_map_real_estate(
    img: np.ndarray,
    *,
    gamma: float = 0.92,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Gamma + CLAHE LAB-L tone map calibrated cho ảnh BĐS interior/exterior.

    Port từ imagen-ai/backend/services/tone_map.py.
    Parameters of Edit-image/hdr_processor.py "real_estate" Reinhard inspired.

    Khác `enhance.py:enhance_studio` (8-step heavier): đây là tone map nhẹ
    chỉ làm 2 việc — slight gamma brighten + adaptive local contrast trên
    luminance — giữ chroma nguyên. Phù hợp dùng làm post-tone preset trong
    batch khi không muốn áp full enhance pipeline.

    Args:
        img: BGR uint8.
        gamma: <1 = brighten (default 0.92 → midtone +6%).
        clahe_clip: CLAHE contrast limit. Cao = local contrast mạnh hơn.
        clahe_grid: tile grid; (8,8) cho ảnh phổ thông, (16,16) cho 4K+.
    """
    if img.dtype != np.uint8:
        raise ValueError("tone_map_real_estate cần uint8 input")
    f = (img.astype(np.float32) / 255.0)
    # Gamma slight brighten
    f = np.power(f, gamma)
    bright = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    # CLAHE on L
    lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


@dataclass
class TonePreset:
    """Batch-wide tone preset — locked params apply tới mọi ảnh trong batch."""

    name: PresetName = "neutral"
    strength: float = 0.5         # 0..1, bias strength

    # Computed at first call (cached)
    _wb_scales: tuple[float, float, float] | None = None
    _color_shift: tuple[float, float, float] | None = None  # B, G, R adjust

    def __post_init__(self):
        # Pre-compute color bias theo preset
        if self.name == "warm":
            # Bias warm: tăng R nhẹ, giảm B nhẹ. Strength 0.5 = +5% R, -5% B
            shift = self.strength * 0.10
            self._color_shift = (-shift, 0.0, +shift)
        elif self.name == "cool":
            shift = self.strength * 0.10
            self._color_shift = (+shift, 0.0, -shift)
        else:
            # neutral hoặc auto
            self._color_shift = (0.0, 0.0, 0.0)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply tone preset đến 1 ảnh.

        Steps:
        1. Adaptive WB neutralize (bỏ cast cá nhân, dùng cùng algorithm)
        2. Apply locked color shift theo preset (warm/cool/neutral/real_estate)
        """
        from .enhance import auto_white_balance
        # Step 1: per-image neutralization với strict auto WB
        neutral = auto_white_balance(img, method="auto")

        # real_estate: dedicated tone-map (port từ imagen-ai/Edit-image)
        if self.name == "real_estate":
            # Strength 0..1 modulates clahe_clip + gamma intensity
            gamma = 1.0 - 0.10 * float(np.clip(self.strength, 0, 1))
            clip = 1.5 + 1.5 * float(np.clip(self.strength, 0, 1))
            return tone_map_real_estate(neutral, gamma=gamma, clahe_clip=clip)

        # Step 2: apply batch-locked color shift
        if all(s == 0.0 for s in self._color_shift):
            return neutral

        f = neutral.astype(np.float32)
        b_shift, g_shift, r_shift = self._color_shift
        # Apply chỉ trên midtones (mask) để tránh break shadows/highlights
        # Midtones = pixel có luminance 50-200
        gray = cv2.cvtColor(neutral, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mid_mask = ((gray > 50) & (gray < 200)).astype(np.float32)[..., None]
        f[..., 0] = f[..., 0] * (1 + b_shift * mid_mask[..., 0])
        f[..., 1] = f[..., 1] * (1 + g_shift * mid_mask[..., 0])
        f[..., 2] = f[..., 2] * (1 + r_shift * mid_mask[..., 0])
        return np.clip(f, 0, 255).astype(np.uint8)


def detect_scene_tone(img: np.ndarray) -> PresetName:
    """Detect scene tone từ ảnh đầu — auto chọn preset cho batch."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0]
    S = hsv[..., 1]
    # Mean hue weighted by saturation
    weights = S.astype(np.float32) + 1
    mean_h = float(np.average(H, weights=weights))

    # Hue OpenCV 0-180.  Warm = ~15-25 (orange/yellow), cool = ~95-115 (blue)
    # 0/180 = red, 60 = yellow, 90 = green, 120 = cyan, 150 = blue/violet
    if mean_h < 35 or mean_h > 165:
        return "warm"
    elif 90 < mean_h < 130:
        return "cool"
    return "neutral"


# =====================================================================
# Dynamic batch anchor
# =====================================================================

@dataclass(frozen=True)
class BatchAnchor:
    """Tone anchor đã fit từ tập ảnh batch.

    `lab_median` ở không gian OpenCV uint8 (L,a,b ∈ [0,255]) — match output
    của `cv2.cvtColor(BGR2LAB)` để áp trực tiếp không cần convert.

    Apply chỉ động vào chrominance (a, b). L (luminance) không touch để
    không thay đổi exposure.
    """

    lab_median: tuple[float, float, float]
    hue_mean: float
    samples: int

    def apply(
        self,
        img: np.ndarray,
        *,
        strength: float = 0.6,
        max_shift: float = 14.0,
    ) -> np.ndarray:
        """Nudge image's color cast về anchor.

        Args:
            img: BGR uint8.
            strength: 0..1, fraction of (anchor − current) gap đóng lại.
            max_shift: clip shift magnitude (uint8 LAB units; 14 ≈ 7 a*/b*).

        Trả ảnh BGR uint8. Nếu anchor không có sample, trả ảnh gốc.
        """
        if self.samples <= 0 or img is None or img.size == 0:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        midtone = (L > 50) & (L < 220)
        if int(midtone.sum()) < 100:
            return img
        cur_a = float(np.median(lab[..., 1][midtone]))
        cur_b = float(np.median(lab[..., 2][midtone]))
        delta_a = float(np.clip((self.lab_median[1] - cur_a) * strength, -max_shift, max_shift))
        delta_b = float(np.clip((self.lab_median[2] - cur_b) * strength, -max_shift, max_shift))
        if abs(delta_a) < 0.4 and abs(delta_b) < 0.4:
            return img
        # Feathered weight: chỉ shift midtone, soft transition để không seam.
        weight = midtone.astype(np.float32)
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=8.0)
        lab[..., 1] = np.clip(lab[..., 1] + delta_a * weight, 0, 255)
        lab[..., 2] = np.clip(lab[..., 2] + delta_b * weight, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


class BatchToneFitter:
    """Aggregate per-image tone stats để compute shared anchor.

    Chỉ giữ median (3 floats per image) — memory O(N), không O(N·pixels).
    Sau khi gọi `fit_anchor`, instance vẫn dùng được — `add` thêm sample
    sẽ cập nhật anchor lần `fit_anchor` kế.

    Để tăng tốc batch lớn (>50 ảnh), gọi với short edge `sample_short_edge`
    hoặc tự sub-sample paths trước khi gọi `add_from_path`.
    """

    def __init__(self, sample_short_edge: int = 512):
        self._sample_short_edge = max(64, int(sample_short_edge))
        self._L_meds: list[float] = []
        self._a_meds: list[float] = []
        self._b_meds: list[float] = []
        self._hues: list[float] = []

    @property
    def samples(self) -> int:
        return len(self._a_meds)

    def add(self, img: np.ndarray) -> bool:
        """Sample 1 ảnh BGR. Trả True nếu ghi nhận được median."""
        if img is None or img.size == 0:
            return False
        small = self._downsample(img)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        mask = (L > 50) & (L < 220)
        if int(mask.sum()) < 50:
            return False
        self._L_meds.append(float(np.median(L[mask])))
        self._a_meds.append(float(np.median(lab[..., 1][mask])))
        self._b_meds.append(float(np.median(lab[..., 2][mask])))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.float32)
        S = hsv[..., 1].astype(np.float32) + 1.0
        self._hues.append(float(np.average(H, weights=S)))
        return True

    def add_from_path(self, path) -> bool:
        try:
            from .utils import read_image
            img = read_image(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("BatchToneFitter skip %s: %s", path, exc)
            return False
        return self.add(img)

    def add_many(self, images: Iterable[np.ndarray]) -> int:
        n = 0
        for im in images:
            if self.add(im):
                n += 1
        return n

    def fit_anchor(self) -> BatchAnchor | None:
        """Trả `BatchAnchor` (median của các median). None nếu không có sample."""
        if not self._a_meds:
            return None
        return BatchAnchor(
            lab_median=(
                float(np.median(self._L_meds)),
                float(np.median(self._a_meds)),
                float(np.median(self._b_meds)),
            ),
            hue_mean=float(np.median(self._hues)) if self._hues else 0.0,
            samples=len(self._a_meds),
        )

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        short = min(h, w)
        if short <= self._sample_short_edge:
            return img
        scale = self._sample_short_edge / short
        return cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
