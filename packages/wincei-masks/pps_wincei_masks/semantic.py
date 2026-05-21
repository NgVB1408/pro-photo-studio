"""SegFormer-B3 ADE20K wrapper — output per-class soft masks 0..1.

ADE20K classes mày cần cho ảnh BĐS:
    wall    = 0    (tường)
    floor   = 3    (sàn)
    ceiling = 5    (trần)
    window  = 8    (cửa sổ — không kính, chỉ frame + kính trắng + sky)
    door    = 14   (cửa đi)
    sky     = 2    (trời nhìn qua kính → union với window)
    lamp    = 36   (đèn rời)
    light   = 82   (đèn dây/đèn chùm)

Soft mask = softmax probabilities (float32 0..1) — KHÔNG argmax để giữ uncertainty
cho refinement downstream (PyMatting cần trimap chứ không phải binary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

ADE20K_CLASSES = {
    "wall": 0,
    "sky": 2,
    "floor": 3,
    "ceiling": 5,
    "window": 8,
    "door": 14,
    "lamp": 36,
    "light": 82,
}

# Pick model by available VRAM.
_MODEL_TIERS = [
    (8, "nvidia/segformer-b5-finetuned-ade-640-640"),
    (5, "nvidia/segformer-b3-finetuned-ade-512-512"),
    (3, "nvidia/segformer-b2-finetuned-ade-512-512"),
    (1.8, "nvidia/segformer-b1-finetuned-ade-512-512"),
    (0, "nvidia/segformer-b0-finetuned-ade-512-512"),
]


def _select_model(min_vram_gb: float = 0.0) -> str:
    """Pick largest SegFormer model that fits free VRAM (or B3 default if CPU)."""
    if not torch.cuda.is_available():
        return "nvidia/segformer-b3-finetuned-ade-512-512"
    try:
        free = torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        free = min_vram_gb or 0.0
    for need, name in _MODEL_TIERS:
        if free >= need:
            return name
    return "nvidia/segformer-b0-finetuned-ade-512-512"


@dataclass
class SemanticResult:
    """Per-class soft mask + argmax label + raw probs."""
    probs: np.ndarray              # (H, W, K) float32, K = len(class_ids)
    argmax_id: np.ndarray          # (H, W) int — direct ADE20K id
    class_id_index: dict[int, int] # ade20k_id → index trong probs last dim
    image_size: tuple[int, int]    # (H, W)
    model_name: str

    def get_soft(self, ade_id: int) -> np.ndarray:
        """Soft mask (float32 0..1) cho 1 class. Returns zeros nếu class không có."""
        idx = self.class_id_index.get(ade_id)
        if idx is None:
            return np.zeros(self.image_size, dtype=np.float32)
        return self.probs[..., idx]

    def get_binary(self, ade_id: int, threshold: float = 0.5) -> np.ndarray:
        return (self.get_soft(ade_id) >= threshold).astype(np.uint8) * 255


class SemanticSegmenter:
    """Lazy-loading SegFormer wrapper. Idempotent: load 1 lần dùng cho cả batch."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.model_name = model_name or _select_model()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("SegFormer load %s on %s", self.model_name, self.device)
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()

    @torch.inference_mode()
    def segment(
        self,
        image_bgr: np.ndarray,
        keep_classes: list[int] | None = None,
    ) -> SemanticResult:
        """Run inference + extract per-class soft masks at native resolution.

        Args:
            image_bgr: (H,W,3) uint8 BGR.
            keep_classes: list of ADE20K class ids to extract (None = all standard).

        Returns:
            SemanticResult.
        """
        if keep_classes is None:
            keep_classes = list(ADE20K_CLASSES.values())

        h, w = image_bgr.shape[:2]
        rgb = image_bgr[..., ::-1]
        pil = Image.fromarray(rgb)

        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, K=150, h', w') @ 128x128 hoặc nhỏ hơn

        # Softmax + argmax ở LOW-RES trước (an toàn RAM)
        probs_lowres = torch.nn.functional.softmax(logits.float(), dim=1)[0]  # (150, h', w')
        argmax_lowres = torch.argmax(probs_lowres, dim=0).cpu().numpy().astype(np.int32)

        # Chỉ slice + upsample các class CẦN (tránh OOM với 150 class full-res)
        idx = torch.tensor(keep_classes, device=probs_lowres.device, dtype=torch.long)
        probs_sliced_lowres = probs_lowres.index_select(0, idx)  # (K_keep, h', w')
        probs_up = torch.nn.functional.interpolate(
            probs_sliced_lowres.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]  # (K_keep, H, W)
        probs_sliced = probs_up.cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # (H, W, K_keep)

        # Upsample argmax (nearest, KHÔNG bilinear) cho reference
        argmax_id = cv2.resize(argmax_lowres, (w, h), interpolation=cv2.INTER_NEAREST)

        return SemanticResult(
            probs=probs_sliced.astype(np.float32),
            argmax_id=argmax_id,
            class_id_index={c: i for i, c in enumerate(keep_classes)},
            image_size=(h, w),
            model_name=self.model_name,
        )

    def free(self) -> None:
        del self.model
        del self.processor
        if self.device == "cuda":
            torch.cuda.empty_cache()
