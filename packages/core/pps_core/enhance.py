"""Image enhancement — replace Autoenhance.ai-style services.

Hai tier:
1. **Lightweight pipeline** (default, no ML deps): white balance, exposure,
   highlight recovery, shadow lift, vibrance, sharpening, denoise. CPU-only,
   ~0.3s cho ảnh 4K. Phù hợp real estate, e-commerce, portrait casual.

2. **ML upscaler** (optional, lazy import torch):
   - Real-ESRGAN: super-resolution + denoise (general)
   - GFPGAN: face restoration

Workflow điển hình real estate:
    raw photo  →  enhance.preset_real_estate()  →  polished photo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WhiteBalanceMethod = Literal["gray_world", "white_patch", "auto", "off"]
EnhancePreset = Literal[
    "real_estate", "portrait", "product", "outdoor", "auto", "custom",
]


@dataclass
class EnhanceParams:
    white_balance: WhiteBalanceMethod = "gray_world"
    clahe_clip: float = 2.0          # CLAHE clip limit (1=mild, 4=strong)
    clahe_tile: int = 8               # CLAHE tile grid
    highlight_recovery: float = 0.3   # 0=off, 1=aggressive
    shadow_lift: float = 0.3          # 0=off, 1=strong lift
    vibrance: float = 0.25            # 0=off, 1=very saturated
    saturation_boost: float = 0.0     # uniform boost (use vibrance instead usually)
    unsharp_amount: float = 0.6       # 0=off, 1=strong sharpen
    unsharp_sigma: float = 1.5
    denoise_strength: int = 0         # 0=off, 3-10 = bilateral filter d
    gamma: float = 1.0                # 1.0 = không đổi


PRESETS: dict[str, EnhanceParams] = {
    "studio": EnhanceParams(  # high-quality, slower (~2s/4K)
        white_balance="gray_world",
        clahe_clip=2.0, clahe_tile=12,
        highlight_recovery=0.5,
        shadow_lift=0.5,
        vibrance=0.3,
        unsharp_amount=0.6, unsharp_sigma=1.0,
        denoise_strength=0,
        gamma=0.92,
    ),
    "real_estate": EnhanceParams(
        white_balance="auto",
        clahe_clip=1.2, clahe_tile=8,
        highlight_recovery=0.4,
        shadow_lift=0.35,
        vibrance=0.18,
        unsharp_amount=0.35, unsharp_sigma=1.2,
        denoise_strength=0,
        gamma=0.97,
    ),
    "portrait": EnhanceParams(
        white_balance="gray_world",
        clahe_clip=1.5, clahe_tile=8,
        highlight_recovery=0.2,
        shadow_lift=0.3,
        vibrance=0.15,
        unsharp_amount=0.4, unsharp_sigma=1.0,
        denoise_strength=5,  # smooth skin
    ),
    "product": EnhanceParams(
        white_balance="white_patch",
        clahe_clip=1.5,
        highlight_recovery=0.2,
        shadow_lift=0.2,
        vibrance=0.3,
        unsharp_amount=0.7, unsharp_sigma=1.0,
    ),
    "outdoor": EnhanceParams(
        white_balance="gray_world",
        clahe_clip=2.5,
        highlight_recovery=0.5,
        shadow_lift=0.4,
        vibrance=0.35,
        unsharp_amount=0.6,
    ),
}


def auto_white_balance(img: np.ndarray, method: WhiteBalanceMethod = "gray_world") -> np.ndarray:
    """Loại bỏ color cast (vd: ám vàng đèn ấm, ám xanh ngày mây).

    "auto" = adaptive gray-world. Đo magnitude cast; nếu ảnh đã neutral
    (deviation < 3%) thì skip; nếu cast nhẹ → partial correct; cast mạnh →
    full correct. Tránh over-correction trên ảnh BĐS đã trắng (gây ám hồng/xanh).
    """
    if method == "off":
        return img
    if method == "auto":
        b, g, r = cv2.split(img.astype(np.float32))
        # Skip pixel sáng > 245 và tối < 10 — tránh skew bởi blown highlights / pure black
        m = (img.max(axis=2) < 245) & (img.min(axis=2) > 10)
        if m.sum() < 100:
            return img
        b_m, g_m, r_m = b[m].mean(), g[m].mean(), r[m].mean()
        avg = (b_m + g_m + r_m) / 3.0
        if avg < 1.0:
            return img
        # Magnitude của cast = max relative deviation từ trung bình
        dev = max(abs(b_m - avg), abs(g_m - avg), abs(r_m - avg)) / avg
        # Strength: dev<0.03 → skip, 0.03-0.08 ramp 0..1, >0.08 = full
        if dev < 0.03:
            return img
        strength = float(np.clip((dev - 0.03) / 0.05, 0.0, 1.0))
        sb = avg / max(b_m, 1e-6)
        sg = avg / max(g_m, 1e-6)
        sr = avg / max(r_m, 1e-6)
        # Blend toward identity scale (1.0) theo strength
        sb = 1.0 + (sb - 1.0) * strength
        sg = 1.0 + (sg - 1.0) * strength
        sr = 1.0 + (sr - 1.0) * strength
        out = cv2.merge([b * sb, g * sg, r * sr])
    elif method == "gray_world":
        # Mean of each channel ≈ same neutral gray
        b, g, r = cv2.split(img.astype(np.float32))
        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        avg = (b_mean + g_mean + r_mean) / 3.0
        b = b * (avg / max(b_mean, 1e-6))
        g = g * (avg / max(g_mean, 1e-6))
        r = r * (avg / max(r_mean, 1e-6))
        out = cv2.merge([b, g, r])
    elif method == "white_patch":
        # Brightest pixel = pure white
        out = img.astype(np.float32)
        for c in range(3):
            ch = out[..., c]
            top = np.percentile(ch, 99)
            if top > 1:
                out[..., c] = ch * (255.0 / top)
    else:
        raise ValueError(f"WB method không hợp lệ: {method}")
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, *, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE trên kênh L của LAB — boost contrast local mà không over-saturate."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def highlight_recovery(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Pull down highlights (cứu sky/cửa sổ bị blown out)."""
    if amount <= 0:
        return img
    f = img.astype(np.float32) / 255.0
    # luminance
    lum = 0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0]
    # mask highlights (lum > 0.7)
    mask = np.clip((lum - 0.7) / 0.3, 0, 1)[..., None]
    # tone curve để giảm highlights: y = x - amount * mask * (x - 0.7)
    f = f - amount * mask * np.clip(f - 0.7, 0, None)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def shadow_lift(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Push up shadows (lift detail trong vùng tối)."""
    if amount <= 0:
        return img
    f = img.astype(np.float32) / 255.0
    lum = 0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0]
    # mask shadows (lum < 0.3)
    mask = np.clip((0.3 - lum) / 0.3, 0, 1)[..., None]
    # tăng pixel theo mask: y = x + amount * mask * (0.3 - x)
    f = f + amount * mask * np.clip(0.3 - f, 0, None)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def vibrance(img: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """Tăng saturation cho pixel có sat thấp (giữ skin tones, tăng nature/walls)."""
    if amount <= 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    # Vibrance: tăng nhiều cho s thấp, ít cho s cao
    boost = (255 - s) / 255.0 * amount
    s = s * (1 + boost) + boost * 30  # tăng tuyệt đối + tương đối
    hsv[..., 1] = np.clip(s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def saturation_boost(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Tăng sat đồng đều (dùng vibrance thay nếu có thể)."""
    if amount == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + amount), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp_mask(img: np.ndarray, *, sigma: float = 1.5, amount: float = 0.5) -> np.ndarray:
    """Sharpen bằng unsharp mask. amount=0 = off."""
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, *, strength: int = 5) -> np.ndarray:
    """Bilateral filter — giảm noise giữ edge."""
    if strength <= 0:
        return img
    return cv2.bilateralFilter(img, d=strength, sigmaColor=50, sigmaSpace=50)


def multi_scale_retinex(
    img: np.ndarray, *, scales: tuple[int, ...] = (15, 80, 250),
) -> np.ndarray:
    """Multi-Scale Retinex — classic color/exposure normalization.
    Simulates human visual system, excellent for indoor real estate photos
    with mixed lighting (natural + artificial).
    """
    img_f = img.astype(np.float32) + 1.0
    log_img = np.log(img_f)
    msr = np.zeros_like(log_img)
    for sigma in scales:
        # Gaussian blur ~ surround function
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
        msr += log_img - np.log(blurred + 1.0)
    msr /= len(scales)
    # Re-stretch to 0-255
    for c in range(3):
        ch = msr[..., c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi - lo > 1e-6:
            msr[..., c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
    return msr.astype(np.uint8)


def guided_filter(
    img: np.ndarray, guide: np.ndarray | None = None,
    *, radius: int = 8, eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving smoothing (He et al. 2010). Pure numpy, fast.
    Dùng để smooth noise nhưng giữ edge, hoặc tách base/detail layer.
    """
    if guide is None:
        guide = img
    img_f = img.astype(np.float32) / 255.0
    guide_f = guide.astype(np.float32) / 255.0

    if img_f.ndim == 2:
        return _guided_filter_single(img_f, guide_f, radius, eps) * 255.0

    out = np.zeros_like(img_f)
    for c in range(img_f.shape[2]):
        g = guide_f[..., c] if guide_f.ndim == 3 else guide_f
        out[..., c] = _guided_filter_single(img_f[..., c], g, radius, eps)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _guided_filter_single(p: np.ndarray, I: np.ndarray, r: int, eps: float) -> np.ndarray:
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    return mean_a * I + mean_b


def tonemap_reinhard(
    img: np.ndarray, *, key: float = 0.18, white_point: float = 0.95,
) -> np.ndarray:
    """Reinhard global tone mapping — HDR-style với roll-off highlights."""
    f = img.astype(np.float32) / 255.0
    # Luminance
    lum = 0.2126 * f[..., 2] + 0.7152 * f[..., 1] + 0.0722 * f[..., 0]
    log_lum = np.log(lum + 1e-6)
    avg_lum = np.exp(log_lum.mean())
    # Scale luminance
    scaled = key / avg_lum * lum
    # Reinhard with white point
    L_white = white_point * scaled.max()
    new_lum = (scaled * (1 + scaled / (L_white ** 2))) / (1 + scaled)
    # Apply tone curve preserving color ratios
    ratio = (new_lum / (lum + 1e-6))[..., None]
    out = f * ratio
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def s_curve_tone(
    img: np.ndarray, *, shadows: float = 0.0, highlights: float = 0.0,
    contrast: float = 0.15,
) -> np.ndarray:
    """S-curve tone mapping với crushable shadows + roll-off highlights.
    contrast 0..1 = strength của S-curve.
    """
    f = img.astype(np.float32) / 255.0
    # S-curve qua sigmoid
    if contrast > 0:
        # Sigmoid centered at 0.5
        steepness = 4 + contrast * 8
        f = 1.0 / (1.0 + np.exp(-steepness * (f - 0.5)))
        # Re-stretch to [0,1]
        f0 = 1.0 / (1.0 + np.exp(steepness * 0.5))
        f1 = 1.0 / (1.0 + np.exp(-steepness * 0.5))
        f = (f - f0) / (f1 - f0)
    # Lift shadows + crush blacks
    if shadows > 0:
        f = f + shadows * 0.15 * np.clip(0.3 - f, 0, None) / 0.3
    if shadows < 0:
        f = f + shadows * 0.15 * np.clip(0.3 - f, 0, None) / 0.3
    # Roll off highlights
    if highlights < 0:
        h_mask = np.clip((f - 0.7) / 0.3, 0, 1)
        f = f + highlights * 0.2 * h_mask
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def local_detail_enhance(
    img: np.ndarray, *, strength: float = 0.5, radius: int = 8,
) -> np.ndarray:
    """Local detail enhancement via guided filter base/detail decomposition.
    Tách ảnh = base (smooth) + detail (high freq), boost detail, recombine.
    Tránh halo của naive unsharp.
    """
    if strength <= 0:
        return img
    base = guided_filter(img, radius=radius, eps=1e-2)
    detail = img.astype(np.float32) - base.astype(np.float32)
    enhanced = base.astype(np.float32) + detail * (1 + strength)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction. gamma<1 = brighter, >1 = darker."""
    if gamma == 1.0:
        return img
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def enhance(img: np.ndarray, params: EnhanceParams | None = None) -> np.ndarray:
    """Apply full enhancement pipeline với params tuỳ chỉnh."""
    if params is None:
        params = EnhanceParams()

    out = auto_white_balance(img, params.white_balance)
    if params.clahe_clip > 0:
        out = apply_clahe(out, clip=params.clahe_clip, tile=params.clahe_tile)
    out = highlight_recovery(out, params.highlight_recovery)
    out = shadow_lift(out, params.shadow_lift)
    out = vibrance(out, params.vibrance)
    if params.saturation_boost != 0:
        out = saturation_boost(out, params.saturation_boost)
    out = denoise(out, strength=params.denoise_strength)
    out = unsharp_mask(out, sigma=params.unsharp_sigma, amount=params.unsharp_amount)
    out = gamma_correct(out, params.gamma)
    return out


def preset(name: EnhancePreset = "real_estate") -> EnhanceParams:
    """Lấy preset params theo tên scene."""
    if name == "auto":
        return PRESETS["real_estate"]  # default
    if name not in PRESETS:
        raise ValueError(f"Preset không hỗ trợ: {name}. Có: {list(PRESETS)}")
    return PRESETS[name]


def enhance_preset(img: np.ndarray, preset_name: EnhancePreset = "real_estate") -> np.ndarray:
    """Shortcut: enhance với preset có sẵn."""
    if preset_name == "studio":
        return enhance_studio(img)
    return enhance(img, preset(preset_name))


def enhance_studio(
    img: np.ndarray,
    params: EnhanceParams | None = None,
) -> np.ndarray:
    """Studio-grade pipeline (chất lượng cao nhất):
      1. WB robust (percentile 5–95%) — không bị skew bởi sky/dark
      2. CLAHE LAB L
      3. Highlight recovery + shadow lift
      4. Vibrance giữ skin tones
      5. Local detail enhance halo-free (guided filter base/detail decomposition)
      6. Mild final unsharp + gamma

    Nếu `params` được cấp, các giá trị clahe_clip/highlight_recovery/shadow_lift/
    vibrance/unsharp_amount/gamma sẽ override default — phục vụ Web UI slider.
    `clahe_tile` / `unsharp_sigma` / `denoise_strength` cũng được tôn trọng.
    """
    p = params if params is not None else PRESETS["studio"]

    if p.white_balance == "off":
        out = img.copy()
    elif p.white_balance == "white_patch":
        out = auto_white_balance(img, "white_patch")
    else:  # "gray_world" hoặc bất kỳ → robust gray-world
        out = _wb_robust(img)

    if p.clahe_clip > 0:
        tile = p.clahe_tile if p.clahe_tile and p.clahe_tile > 0 else 10
        out = apply_clahe(out, clip=p.clahe_clip, tile=tile)
    out = highlight_recovery(out, amount=p.highlight_recovery)
    out = shadow_lift(out, amount=p.shadow_lift)
    out = _vibrance_skin_safe(out, amount=max(0.0, p.vibrance))
    if p.denoise_strength and p.denoise_strength > 0:
        out = denoise(out, strength=int(p.denoise_strength))
    # Detail nâng cao halo-free; strength scale theo unsharp_amount
    detail_strength = float(np.clip(p.unsharp_amount * 1.0, 0.0, 1.0))
    if detail_strength > 0:
        out = local_detail_enhance(out, strength=detail_strength, radius=5)
    # Mild final sharpen
    out = unsharp_mask(
        out,
        sigma=max(0.6, p.unsharp_sigma * 0.7),
        amount=max(0.0, p.unsharp_amount * 0.4),
    )
    out = gamma_correct(out, gamma=p.gamma if p.gamma > 0 else 0.93)
    return out


def _wb_robust(img: np.ndarray) -> np.ndarray:
    """Gray world WB nhưng dùng percentile 5-95% pixel — tránh skew bởi
    extreme bright/dark."""
    f = img.astype(np.float32)
    means = []
    for c in range(3):
        ch = f[..., c]
        lo, hi = np.percentile(ch, [5, 95])
        valid = (ch >= lo) & (ch <= hi)
        means.append(ch[valid].mean() if valid.any() else ch.mean())
    avg = sum(means) / 3.0
    out = f.copy()
    for c in range(3):
        out[..., c] = f[..., c] * (avg / max(means[c], 1e-6))
    return np.clip(out, 0, 255).astype(np.uint8)


def _vibrance_skin_safe(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Vibrance giữ skin tones (Hue 0-30 in HSV space)."""
    if amount <= 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_chan = hsv[..., 0]
    s = hsv[..., 1]

    # Skin hue mask (in OpenCV HSV: H ~ 0-25 cho skin)
    skin_mask = ((h_chan <= 25) | (h_chan >= 170)).astype(np.float32)
    skin_factor = 1.0 - 0.7 * skin_mask  # giảm 70% effect cho skin

    boost = (255 - s) / 255.0 * amount * skin_factor
    s = s * (1 + boost) + boost * 30
    hsv[..., 1] = np.clip(s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ======================================================================
# ML upscaler (optional) — Real-ESRGAN / GFPGAN
# ======================================================================

_realesrgan_model = None
_gfpgan_model = None


def upscale_realesrgan(
    img: np.ndarray, *, scale: int = 2, device: str = "auto",
) -> np.ndarray:
    """Real-ESRGAN super-resolution. Cần `pip install realesrgan basicsr`."""
    global _realesrgan_model
    try:
        from realesrgan import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Real-ESRGAN cần: pip install realesrgan basicsr (cần torch)"
        ) from exc

    if _realesrgan_model is None:
        from .inpaint import resolve_device
        actual_device = resolve_device(device)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
            scale=scale,
        )
        _realesrgan_model = RealESRGANer(
            scale=scale,
            model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth",
            model=model,
            half=actual_device == "cuda",
            device=actual_device,
        )
    output, _ = _realesrgan_model.enhance(img, outscale=scale)
    return output


def restore_face_gfpgan(
    img: np.ndarray, *, device: str = "auto",
) -> np.ndarray:
    """GFPGAN face restoration. Cần `pip install gfpgan`."""
    global _gfpgan_model
    try:
        from gfpgan import GFPGANer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("GFPGAN cần: pip install gfpgan (cần torch)") from exc

    if _gfpgan_model is None:
        from .inpaint import resolve_device
        actual_device = resolve_device(device)
        _gfpgan_model = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=1, arch="clean", channel_multiplier=2,
            device=actual_device,
        )
    _, _, restored = _gfpgan_model.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True,
    )
    return restored if restored is not None else img
