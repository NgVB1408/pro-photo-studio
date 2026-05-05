"""AI Upscale — Real-ESRGAN x2/x4 qua spandrel + torch.

Yêu cầu: torch + spandrel (đã cài). Model lazy-download lần đầu.

CPU: ~30-60s cho ảnh 4K x2. GPU CUDA: ~2-5s.

Output: ảnh sắc nét gấp 2x/4x, KHÔNG vỡ pixel, KHÔNG blur.

Usage:
    from pps_core.upscale import upscale_ai
    out = upscale_ai(img_4k, scale=2)   # 4K → 8K
    out = upscale_ai(img_2k, scale=4)   # 2K → 8K (16 megapixels)

Memory-aware tile processing:
- Ảnh nhỏ (<1500px) chạy trực tiếp
- Ảnh lớn split thành tile 512×512 với overlap 16px → ghép lại seamless
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ----- Model registry -----
# Real-ESRGAN x4plus: general photo upscaler, GAN-trained, sharper than EDSR
# Real-ESRGAN x2plus: x2 variant
# realesr-general-x4v3: smaller (4.7 MB), faster, slightly less detail
MODEL_URLS: dict[str, tuple[str, int]] = {
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        4,
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        2,
    ),
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        4,
    ),
}

_MODEL_CACHE: dict[str, object] = {}
_LOAD_LOCK = threading.Lock()


def get_models_dir() -> Path:
    """Thư mục lưu model (lazy download)."""
    env = os.environ.get("WATERMARK_TOOLKIT_MODELS_DIR")
    d = Path(env).expanduser() if env else Path.home() / ".pps_core" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_model(name: str) -> Path:
    """Download model nếu chưa có. Return path."""
    if name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_URLS)}")
    url, _ = MODEL_URLS[name]
    target = get_models_dir() / f"{name}.pth"
    if target.is_file() and target.stat().st_size > 1024 * 1024:
        return target
    logger.info("Downloading %s ... (%s)", name, url)
    headers = {"User-Agent": "watermark-toolkit/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(target, "wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total > 0:
                    pct = downloaded / total * 100
                    logger.info(
                        "  ↓ %s %d%% (%.1f / %.1f MB)",
                        name,
                        int(pct),
                        downloaded / 1024 / 1024,
                        total / 1024 / 1024,
                    )
    logger.info("Saved %s", target)
    return target


def _load_model(name: str, device: str = "auto"):
    """Lazy load model qua spandrel — return ImageModelDescriptor."""
    cache_key = f"{name}@{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _LOAD_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        import torch
        from spandrel import ModelLoader

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = _download_model(name)
        loader = ModelLoader(device=torch.device(device))
        model = loader.load_from_file(str(model_path))
        model.cpu() if device == "cpu" else model.cuda()
        model.eval()

        _MODEL_CACHE[cache_key] = model
        logger.info("Loaded %s on %s, scale=%d", name, device, model.scale)
        return model


def _bgr_to_tensor(img: np.ndarray):
    """uint8 BGR → float32 RGB tensor [1, 3, H, W] in 0..1."""
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def _tensor_to_bgr(t) -> np.ndarray:
    """float tensor [1,3,H,W] in 0..1 → uint8 BGR."""
    arr = t.squeeze(0).clamp(0, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW → HWC
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _process_tile(model, tile_bgr: np.ndarray) -> np.ndarray:
    import torch

    t = _bgr_to_tensor(tile_bgr)
    if next(model.model.parameters()).is_cuda:
        t = t.cuda()
    with torch.no_grad():
        out = model(t)
    return _tensor_to_bgr(out.float())


def upscale_ai(
    img: np.ndarray,
    *,
    scale: int = 2,
    model_name: str | None = None,
    tile: int = 512,
    overlap: int = 16,
    device: str = "auto",
) -> np.ndarray:
    """AI upscale qua Real-ESRGAN.

    Args:
        img: BGR uint8.
        scale: 2 hoặc 4. Auto chọn model phù hợp.
        model_name: override (vd "realesr-general-x4v3" cho fast).
        tile: tile size (px) cho memory-safe inference. 512 OK cho 8GB RAM.
        overlap: overlap giữa các tile để tránh seam.
        device: "cpu" / "cuda" / "auto".

    Returns:
        BGR uint8 với kích thước scale × input.
    """
    if scale not in (2, 4):
        raise ValueError("scale phải là 2 hoặc 4")

    if model_name is None:
        # Default: x4 cho cả x2 (downscale sau) — cho chất lượng tốt nhất
        # Hoặc x2plus cho speed nếu chỉ cần x2
        model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"

    model = _load_model(model_name, device=device)
    h, w = img.shape[:2]

    # Nếu ảnh nhỏ vừa với 1 tile, chạy trực tiếp
    if max(h, w) <= tile:
        return _process_tile(model, img)

    out_h = h * model.scale
    out_w = w * model.scale
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Tile loop với overlap
    step = tile - overlap
    n_tiles_y = (h + step - 1) // step
    n_tiles_x = (w + step - 1) // step
    total = n_tiles_y * n_tiles_x
    done = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * step
            x0 = tx * step
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            # Pad cuối nếu cần
            if y1 - y0 < tile and y1 == h:
                y0 = max(0, h - tile)
            if x1 - x0 < tile and x1 == w:
                x0 = max(0, w - tile)
            tile_in = img[y0:y1, x0:x1]
            tile_out = _process_tile(model, tile_in)

            # Vị trí trong output
            oy0 = y0 * model.scale
            ox0 = x0 * model.scale
            oy1 = oy0 + tile_out.shape[0]
            ox1 = ox0 + tile_out.shape[1]

            # Crop overlap region (chỉ giữ phần inner để tránh seam)
            in_top = (overlap // 2) * model.scale if ty > 0 else 0
            in_left = (overlap // 2) * model.scale if tx > 0 else 0
            in_bot = (overlap // 2) * model.scale if ty < n_tiles_y - 1 else 0
            in_right = (overlap // 2) * model.scale if tx < n_tiles_x - 1 else 0

            cropped = tile_out[
                in_top : tile_out.shape[0] - in_bot, in_left : tile_out.shape[1] - in_right
            ]
            out[oy0 + in_top : oy1 - in_bot, ox0 + in_left : ox1 - in_right] = cropped

            done += 1
            logger.info("  tile %d/%d", done, total)

    # Nếu user yêu cầu scale=2 nhưng model x4 → downscale ½ với Lanczos
    if scale != model.scale:
        target_h = h * scale
        target_w = w * scale
        out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return out


def upscale_ai_safe(img: np.ndarray, scale: int = 2, **kw) -> np.ndarray:
    """Wrapper với fallback Lanczos nếu AI fail (không có deps, OOM, …)."""
    try:
        return upscale_ai(img, scale=scale, **kw)
    except Exception as exc:
        logger.warning("AI upscale fail (%s) — fallback Lanczos", exc)
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
