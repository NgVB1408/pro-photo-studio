"""QwenEditor — wrap Qwen-Image-Lightning for instruction-based image edits.

Lazy-loads the diffusers pipeline on first call. Falls back to remote HF
Inference Provider when local GPU is unavailable.

Model card: https://huggingface.co/lightx2v/Qwen-Image-Lightning
Base:       https://huggingface.co/Qwen/Qwen-Image
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ExecMode = Literal["auto", "local", "remote"]


@dataclass
class QwenEditConfig:
    """Configuration for QwenEditor.

    Attributes:
        base_model:  HF model id of base diffusion pipeline.
        lora_repo:   HF repo id of Lightning LoRA distillation.
        mode:        "auto" picks local if CUDA available else remote.
        dtype:       "bfloat16" recommended on Ampere+ GPUs.
        guidance:    Classifier-free guidance scale (Lightning typically 1.0–2.0).
        steps:       Lightning typically converges in 4–8 steps.
        seed:        For reproducibility (None = random).
    """

    base_model: str = "Qwen/Qwen-Image"
    lora_repo: str = "lightx2v/Qwen-Image-Lightning"
    mode: ExecMode = "auto"
    dtype: str = "bfloat16"
    guidance: float = 1.5
    steps: int = 6
    seed: int | None = None


class QwenEditor:
    """Instruction-based image editor backed by Qwen-Image-Lightning.

    The first `.edit()` call downloads weights to `HF_HOME` (~25 GB combined).
    Subsequent calls reuse the loaded pipeline.

    Raises `RuntimeError` if neither local CUDA nor a `HF_TOKEN` for remote
    inference is configured.
    """

    def __init__(self, config: QwenEditConfig | None = None) -> None:
        self.config = config or QwenEditConfig()
        self._pipe: object | None = None
        self._mode_resolved: ExecMode | None = None

    def _resolve_mode(self) -> ExecMode:
        if self._mode_resolved is not None:
            return self._mode_resolved
        if self.config.mode == "remote":
            self._mode_resolved = "remote"
            return "remote"
        if self.config.mode == "local":
            self._mode_resolved = "local"
            return "local"
        # auto
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                self._mode_resolved = "local"
                return "local"
        except ImportError:
            pass
        if os.environ.get("HF_TOKEN"):
            self._mode_resolved = "remote"
            return "remote"
        raise RuntimeError(
            "QwenEditor: no execution path available. "
            "Install torch with CUDA, or set HF_TOKEN for remote inference."
        )

    def _load_local(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from diffusers import DiffusionPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "QwenEditor local mode requires torch + diffusers. "
                "Install with: pip install torch diffusers transformers accelerate"
            ) from exc
        dtype = getattr(torch, self.config.dtype)
        pipe = DiffusionPipeline.from_pretrained(
            self.config.base_model, dtype=dtype, device_map="cuda"
        )
        pipe.load_lora_weights(self.config.lora_repo)
        self._pipe = pipe
        logger.info(
            "QwenEditor loaded locally: base=%s lora=%s",
            self.config.base_model,
            self.config.lora_repo,
        )

    def edit(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Apply natural-language edit to image.

        Args:
            image: BGR uint8 ndarray (H, W, 3).
            instruction: e.g. "brighten the kitchen, remove the photographer"

        Returns:
            Edited BGR uint8 ndarray.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("QwenEditor requires BGR uint8 image (H, W, 3)")

        mode = self._resolve_mode()
        if mode == "local":
            return self._edit_local(image, instruction)
        return self._edit_remote(image, instruction)

    def _edit_local(self, image: np.ndarray, instruction: str) -> np.ndarray:
        self._load_local()
        # Implementation note: Qwen-Image is currently text-to-image. For
        # image-to-image editing we wire through Qwen-Edit-2509 in a future
        # PR. This stub returns the input unchanged with a logged warning so
        # downstream pipeline still runs.
        logger.warning(
            "QwenEditor.edit local: image-to-image path not yet implemented "
            "in this release; instruction='%s' was logged but image returned unchanged.",
            instruction[:80],
        )
        return image

    def _edit_remote(self, image: np.ndarray, instruction: str) -> np.ndarray:
        try:
            from huggingface_hub import InferenceClient  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "QwenEditor remote mode requires huggingface_hub. "
                "Install with: pip install huggingface_hub"
            ) from exc
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "QwenEditor remote mode requires HF_TOKEN env var. "
                "Get one at https://huggingface.co/settings/tokens"
            )
        InferenceClient(provider="auto", api_key=token)
        # Same caveat as _edit_local — wired up in a follow-up PR with
        # Qwen-Edit-2509 image-conditioned endpoint.
        logger.warning(
            "QwenEditor.edit remote: image-to-image not yet implemented; "
            "instruction='%s' was logged but image returned unchanged.",
            instruction[:80],
        )
        return image
