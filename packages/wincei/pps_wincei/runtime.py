"""Runtime detection — GPU/CPU, provider negotiation, model dispatch.

Quality-first design:
    1. Detect CUDA via torch.cuda + onnxruntime providers.
    2. Prefer GPU model variants (segformer-b3-ade20k, SAM2-Large).
    3. Fall back to CPU model variants (segformer-b0-ade20k) gracefully.
    4. Cache model sessions across calls.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_lock = threading.Lock()


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved compute environment."""

    has_cuda: bool
    cuda_device_name: str
    cuda_vram_gb: float
    onnx_providers: tuple[str, ...]
    use_gpu: bool

    def banner(self) -> str:
        if self.use_gpu:
            return (
                f"🚀 GPU: {self.cuda_device_name} ({self.cuda_vram_gb:.1f}GB VRAM) "
                f"| ONNX providers: {', '.join(self.onnx_providers)}"
            )
        return "💻 CPU mode (CUDA not available)"


@lru_cache(maxsize=1)
def detect_runtime(force_cpu: bool = False) -> RuntimeProfile:
    """Probe environment. Cached after first call."""
    if force_cpu or os.environ.get("PPS_FORCE_CPU"):
        return RuntimeProfile(
            has_cuda=False,
            cuda_device_name="",
            cuda_vram_gb=0.0,
            onnx_providers=("CPUExecutionProvider",),
            use_gpu=False,
        )

    has_cuda = False
    device_name = ""
    vram_gb = 0.0
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        ordered = []
        if has_cuda and "CUDAExecutionProvider" in available:
            ordered.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:  # DirectML on Windows
            ordered.append("DmlExecutionProvider")
        ordered.append("CPUExecutionProvider")
        providers = tuple(ordered)
    except ImportError:
        pass

    use_gpu = has_cuda and "CUDAExecutionProvider" in providers

    profile = RuntimeProfile(
        has_cuda=has_cuda,
        cuda_device_name=device_name,
        cuda_vram_gb=vram_gb,
        onnx_providers=providers,
        use_gpu=use_gpu,
    )
    logger.info(profile.banner())
    return profile


def pick_segmentation_model(profile: RuntimeProfile) -> str:
    """Quality-tiered HuggingFace segmentation model id selection.

    Returns:
        HF model id (downloaded by `transformers.AutoModel`).
    """
    # ADE20K labels include ceiling (5), window (8), sky (2), wall (0) — exactly
    # what we need for indoor real-estate photo enhancement.
    if profile.use_gpu and profile.cuda_vram_gb >= 6.0:
        # SegFormer-B3 — best quality / accuracy trade-off on real-estate photos
        return "nvidia/segformer-b3-finetuned-ade-512-512"
    if profile.use_gpu and profile.cuda_vram_gb >= 3.0:
        # Mid quality, fits on smaller GPU (2-4GB)
        return "nvidia/segformer-b1-finetuned-ade-512-512"
    # Smallest variant — CPU-friendly, ~14M params
    return "nvidia/segformer-b0-finetuned-ade-512-512"


# ADE20K class indices we care about
# Full list: https://github.com/CSAILVision/sceneparsing/blob/master/sceneparsing-categories.md
ADE20K_INDEX = {
    "wall": 0,
    "sky": 2,
    "floor": 3,
    "ceiling": 5,
    "windowpane": 8,  # "window" in ADE20K is class 8
    "door": 14,
    "lamp": 36,
    "light": 82,
    "screen": 130,
}
