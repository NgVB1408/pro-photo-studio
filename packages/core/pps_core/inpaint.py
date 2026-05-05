"""Inpainting backends: OpenCV (TELEA/NS) và LaMa (qua iopaint).

Backend chọn qua tham số `backend` hoặc Settings.inpaint_backend.
LaMa lazy-import để OpenCV-only setup không cần torch.

Ảnh lớn (4K/6K/8K):
- OpenCV: chạy trực tiếp, RAM ~ 4× kích thước ảnh raw.
- LaMa: dùng HdStrategy=CROP để chỉ inpaint vùng quanh mask ở độ phân giải gốc,
  giữ chi tiết toàn ảnh và giảm VRAM. Bật tự động khi side > crop_trigger_size.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InpaintBackend(StrEnum):
    OPENCV = "opencv"
    LAMA = "lama"


HdStrategy = Literal["original", "resize", "crop"]
OpencvMethod = Literal["telea", "ns"]
_OPENCV_METHODS: dict[str, int] = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def _validate_inputs(image: np.ndarray, mask: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise ValueError(f"image phải uint8, nhận {image.dtype}")
    if mask.dtype != np.uint8:
        raise ValueError(f"mask phải uint8, nhận {mask.dtype}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"image phải 3 hoặc 4 kênh, nhận shape={image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask phải 2 chiều (H, W), nhận shape={mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image vs mask khác kích thước: {image.shape[:2]} vs {mask.shape}")
    if not np.any(mask):
        raise ValueError("Mask rỗng — không có pixel nào để inpaint")


def inpaint_opencv(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    method: OpencvMethod = "telea",
    radius: int = 3,
    edge_pad: int = 64,
) -> np.ndarray:
    """OpenCV inpaint (TELEA hoặc Navier-Stokes). Nhanh, không cần GPU.

    Khi mask nằm sát rìa ảnh, inpaint thiếu context dẫn tới artifact (vùng xám
    đen). Tự động mirror-pad ảnh `edge_pad` px nếu phát hiện mask gần biên,
    inpaint trên ảnh đã pad, rồi crop về size gốc.
    """
    _validate_inputs(image, mask)
    flag = _OPENCV_METHODS.get(method.lower())
    if flag is None:
        raise ValueError(f"method phải thuộc {list(_OPENCV_METHODS)}, nhận {method!r}")
    if radius < 1:
        raise ValueError("radius >= 1")

    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image

    # Phát hiện mask có chạm rìa ảnh không
    h, w = bgr.shape[:2]
    near_edge = (
        mask[:edge_pad, :].any()
        or mask[-edge_pad:, :].any()
        or mask[:, :edge_pad].any()
        or mask[:, -edge_pad:].any()
    )

    if near_edge and edge_pad > 0:
        bgr_pad = cv2.copyMakeBorder(
            bgr,
            edge_pad,
            edge_pad,
            edge_pad,
            edge_pad,
            borderType=cv2.BORDER_REFLECT_101,
        )
        # CRITICAL: mirror mask cùng cách để inpaint KHÔNG sample từ vùng
        # mirror của watermark (nếu mask sát rìa, padded region chứa
        # mirror của logo → nếu mask không phủ vùng đó, blue sẽ bị
        # propagate ngược vào vùng mask).
        mask_pad = cv2.copyMakeBorder(
            mask,
            edge_pad,
            edge_pad,
            edge_pad,
            edge_pad,
            borderType=cv2.BORDER_REFLECT_101,
        )
        result_pad = cv2.inpaint(bgr_pad, mask_pad, radius, flag)
        result = result_pad[edge_pad : edge_pad + h, edge_pad : edge_pad + w]
        logger.debug(
            "OpenCV inpaint xong: method=%s radius=%d (mirror-pad %dpx, mirror mask)",
            method,
            radius,
            edge_pad,
        )
    else:
        result = cv2.inpaint(bgr, mask, radius, flag)
        logger.debug("OpenCV inpaint xong: method=%s radius=%d", method, radius)

    return result


_LAMA_MODELS: dict[tuple[str, str], object] = {}  # cache theo (model_name, device)

SUPPORTED_LAMA_MODELS = (
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "manga",
    "migan",
    "cv2",  # opencv inpaint via iopaint, không khuyến nghị (đã có backend riêng)
)


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' -> 'cuda' nếu có, sang 'mps' (Apple Silicon), cuối cùng 'cpu'."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_lama(device: str, model_name: str = "lama") -> object:
    """Lazy-load model qua iopaint. Trả về callable model. Cache theo (model, device)."""
    if model_name not in SUPPORTED_LAMA_MODELS:
        raise ValueError(f"model {model_name!r} không hỗ trợ. Chọn: {SUPPORTED_LAMA_MODELS}")

    actual_device = resolve_device(device)
    cache_key = (model_name, actual_device)
    if cache_key in _LAMA_MODELS:
        return _LAMA_MODELS[cache_key]

    try:
        from iopaint.model_manager import ModelManager  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Backend 'lama' cần gói iopaint + torch. Cài bằng:\n"
            "    pip install -r requirements-lama.txt\n"
            "hoặc:\n"
            "    pip install 'watermark-toolkit[lama]'"
        ) from exc

    logger.info(
        "Loading model %r (device=%s) — lần đầu sẽ tải weights",
        model_name,
        actual_device,
    )
    model = ModelManager(name=model_name, device=actual_device)
    _LAMA_MODELS[cache_key] = model
    return model


def inpaint_lama(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    device: str = "cpu",
    model: str = "lama",
    hd_strategy: HdStrategy = "crop",
    crop_margin: int = 196,
    crop_trigger_size: int = 1280,
    resize_limit: int = 2048,
) -> np.ndarray:
    """LaMa inpaint qua iopaint, an toàn cho ảnh 4K/6K/8K.

    hd_strategy:
      - "crop"     : (mặc định, KHUYẾN NGHỊ cho ảnh lớn) cắt vùng quanh mask
                     ở độ phân giải gốc để inpaint, paste lại. Giữ toàn bộ
                     chi tiết ngoài vùng watermark.
      - "resize"   : resize ảnh xuống resize_limit trước khi inpaint, sau đó
                     scale ngược. Chất lượng mượt nhưng mất sharpness.
      - "original" : inpaint full size — chỉ dùng cho ảnh ≤ 2K hoặc GPU lớn,
                     dễ OOM với 8K.

    crop_trigger_size: chỉ áp dụng strategy khi cạnh dài > giá trị này.
    crop_margin: padding (px) quanh mask trước khi crop.
    """
    _validate_inputs(image, mask)

    try:
        from iopaint.schema import HDStrategy, InpaintRequest  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Backend 'lama' cần gói iopaint. Cài: pip install -r requirements-lama.txt"
        ) from exc

    actual_device = resolve_device(device)
    inpaint_model = _load_lama(actual_device, model_name=model)

    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    strategy_map = {
        "original": HDStrategy.ORIGINAL,
        "resize": HDStrategy.RESIZE,
        "crop": HDStrategy.CROP,
    }
    strategy_enum = strategy_map.get(hd_strategy.lower())
    if strategy_enum is None:
        raise ValueError(f"hd_strategy phải thuộc {list(strategy_map)}, nhận {hd_strategy!r}")

    h, w = rgb.shape[:2]
    longest = max(h, w)
    if hd_strategy == "original" and longest > 2048:
        logger.warning(
            "Ảnh %dx%d + hd_strategy=original có thể OOM. Khuyến nghị 'crop'.",
            w,
            h,
        )

    request = InpaintRequest(
        hd_strategy=strategy_enum,
        hd_strategy_crop_margin=crop_margin,
        hd_strategy_crop_trigger_size=crop_trigger_size,
        hd_strategy_resize_limit=resize_limit,
    )
    logger.info(
        "%s inpaint: %dx%d, strategy=%s, device=%s",
        model,
        w,
        h,
        hd_strategy,
        actual_device,
    )
    result_rgb = inpaint_model(rgb, mask, request)  # type: ignore[operator]
    result = cv2.cvtColor(np.asarray(result_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return result


def inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    backend: str | InpaintBackend = InpaintBackend.OPENCV,
    opencv_method: OpencvMethod = "telea",
    opencv_radius: int = 3,
    lama_device: str = "auto",
    lama_model: str = "lama",
    hd_strategy: HdStrategy = "crop",
    crop_margin: int = 196,
    crop_trigger_size: int = 1280,
    resize_limit: int = 2048,
) -> np.ndarray:
    """Dispatcher chính. Chọn backend qua enum/string.

    lama_device='auto' tự chọn cuda/mps/cpu. Tham số `hd_strategy`/`lama_model`/
    `crop_margin`/`crop_trigger_size`/`resize_limit` chỉ ảnh hưởng backend LaMa.
    """
    name = backend.value if isinstance(backend, InpaintBackend) else str(backend).lower()
    if name == InpaintBackend.OPENCV.value:
        return inpaint_opencv(image, mask, method=opencv_method, radius=opencv_radius)
    if name == InpaintBackend.LAMA.value:
        return inpaint_lama(
            image,
            mask,
            device=lama_device,
            model=lama_model,
            hd_strategy=hd_strategy,
            crop_margin=crop_margin,
            crop_trigger_size=crop_trigger_size,
            resize_limit=resize_limit,
        )
    raise ValueError(f"backend phải là 'opencv' hoặc 'lama', nhận {backend!r}")
