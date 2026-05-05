"""Pro Photo Studio — ML inference wrappers.

Each submodule wraps one external model with a uniform `Predictor` protocol.
All predictors lazy-load weights on first call to avoid eager torch import
in pipelines that don't use ML.

Submodules:
    qwen        Qwen-Image-Lightning (instruction-based editing)
    qwen_edit   Qwen-Edit-2509 (multi-angle synthesis)
    supir       SUPIR (SOTA image restoration / upscale)
    sam2        Segment Anything 2 (click-mask)
    controlnet  ControlNet sky / upright / depth
    lama        LaMa Cleaner (object removal)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
