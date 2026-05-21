"""SAM 2 wrapper — boundary perfect mask từ click point.

Backends (auto-detect):
    [A] sam2 (Meta, segment-anything-2)         — best quality
    [B] segment_anything (SAM 1.0)              — fallback, lighter
    [C] None                                     — graceful skip

Checkpoint tiers:
    - SAM 2 Hiera Tiny     (38MB)   ← khuyến nghị CPU
    - SAM 2 Hiera Small    (185MB)
    - SAM 2 Hiera Base+    (323MB)
    - SAM 2 Hiera Large    (898MB)  ← need 4GB+ VRAM
    - SAM 1.0 ViT-B        (375MB)
    - SAM 1.0 ViT-L        (1.2GB)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _has_sam2() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


def _has_sam1() -> bool:
    try:
        from segment_anything import SamPredictor  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SAMResult:
    mask: np.ndarray  # (H, W) uint8 0/255
    score: float


HIGH_RES_CONFIG = {
    # Cho automatic mask generator (toàn ảnh)
    "points_per_side": 64,           # default 32 → 64 quét dày hơn
    "pred_iou_thresh": 0.95,         # chỉ giữ vùng mặt nạ confidence cao
    "stability_score_thresh": 0.96,  # biên siêu cứng, tránh tràn
    "crop_n_layers": 1,              # 1 crop layer cho ảnh > 2K
    "min_mask_region_area": 500,     # filter blob nhỏ
}


class SAMEngine:
    """Unified SAM 1/2 wrapper với common predict_from_point API.

    Có 2 mode:
        - Promptable (predict_from_point/points): dùng cho VLM-guided pipeline.
        - Automatic (generate_all_masks): dùng cho full scan ảnh khó (slower).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        backend: str = "auto",  # 'auto' | 'sam2' | 'sam1' | 'fallback'
        device: str = "auto",
        high_res: bool = False,
    ):
        self.high_res = high_res
        if backend == "auto":
            # Prefer SAM 2 nếu cài + có checkpoint
            sam2_ckpt = Path.home() / ".cache" / "sam2" / "sam2_hiera_tiny.pt"
            if _has_sam2() and (checkpoint is not None or sam2_ckpt.exists()):
                backend = "sam2"
            elif _has_sam1():
                # SAM 1 fallback if vit_b checkpoint exists
                sam1_ckpt = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"
                if checkpoint is not None or sam1_ckpt.exists():
                    backend = "sam1"
                    log.info("Using SAM 1.0 (vit_b) — SAM 2 checkpoint not found")
                else:
                    backend = "fallback"
            else:
                backend = "fallback"

        self.backend = backend
        self.device = self._pick_device(device)
        self._predictor = None
        self._image_set = False

        if backend == "sam2":
            self._init_sam2(checkpoint)
        elif backend == "sam1":
            self._init_sam1(checkpoint)
        else:
            log.warning(
                "SAM không cài. Fallback dùng GrabCut (kém hơn nhiều). "
                "pip install 'git+https://github.com/facebookresearch/sam2.git' để dùng SAM2."
            )

    def _pick_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _init_sam2(self, ckpt: str | Path | None) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(f"sam2 import fail: {exc}")

        # Default to Hiera Tiny if no checkpoint provided
        if ckpt is None:
            ckpt = Path.home() / ".cache" / "sam2" / "sam2_hiera_tiny.pt"
            if not ckpt.exists():
                raise RuntimeError(
                    f"SAM 2 checkpoint missing: {ckpt}. "
                    "Download: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
                )
        model_cfg = "sam2_hiera_t.yaml"  # default
        if "small" in str(ckpt).lower():
            model_cfg = "sam2_hiera_s.yaml"
        elif "base" in str(ckpt).lower():
            model_cfg = "sam2_hiera_b+.yaml"
        elif "large" in str(ckpt).lower():
            model_cfg = "sam2_hiera_l.yaml"

        log.info("Loading SAM 2 (%s) on %s", ckpt, self.device)
        model = build_sam2(model_cfg, str(ckpt), device=self.device)
        self._predictor = SAM2ImagePredictor(model)

    def _init_sam1(self, ckpt: str | Path | None) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(f"segment_anything import fail: {exc}")

        # Auto-locate SAM 1 checkpoint trong ~/.cache/sam/
        if ckpt is None:
            cache_dir = Path.home() / ".cache" / "sam"
            for fname in ("sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"):
                candidate = cache_dir / fname
                if candidate.exists():
                    ckpt = candidate
                    log.info("Auto-detected SAM 1 checkpoint: %s", ckpt)
                    break
            else:
                raise RuntimeError(
                    f"SAM 1 checkpoint missing. Tải về: bash scripts/download_sam.sh vit_b"
                )
        ckpt = Path(ckpt)
        model_type = "vit_b"
        if "vit_l" in ckpt.name:
            model_type = "vit_l"
        elif "vit_h" in ckpt.name:
            model_type = "vit_h"
        log.info("Loading SAM 1 (%s, %s) on %s", model_type, ckpt, self.device)
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(self.device)
        self._predictor = SamPredictor(sam)

    def set_image(self, image_bgr: np.ndarray) -> None:
        if self._predictor is None:
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        self._image_set = True

    def predict_from_point(
        self,
        image_bgr: np.ndarray,
        point_xy: tuple[int, int] | list[int],
        *,
        labels: np.ndarray | None = None,
        multimask: bool = False,
    ) -> SAMResult:
        """Predict mask cho 1 click point."""
        if not self._image_set:
            self.set_image(image_bgr)

        if self._predictor is None:
            # Fallback: GrabCut từ point (kém nhưng vẫn ra mask)
            return self._grabcut_from_point(image_bgr, point_xy)

        x, y = int(point_xy[0]), int(point_xy[1])
        point_coords = np.array([[x, y]])
        point_labels = labels if labels is not None else np.array([1])  # 1 = FG

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )
        if multimask:
            best = int(np.argmax(scores))
            return SAMResult(mask=(masks[best] * 255).astype(np.uint8), score=float(scores[best]))
        return SAMResult(mask=(masks[0] * 255).astype(np.uint8), score=float(scores[0]))

    def predict_from_points(
        self,
        image_bgr: np.ndarray,
        points_xy: list[tuple[int, int]],
        *,
        multimask: bool = False,
    ) -> SAMResult:
        """Multi-point prompt → 1 mask. Tất cả points đều là FG."""
        if not self._image_set:
            self.set_image(image_bgr)

        if self._predictor is None:
            return self._grabcut_from_point(image_bgr, points_xy[0] if points_xy else (0, 0))

        coords = np.array([[int(x), int(y)] for x, y in points_xy])
        labels = np.ones(len(coords), dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask,
        )
        if multimask:
            best = int(np.argmax(scores))
            return SAMResult(mask=(masks[best] * 255).astype(np.uint8), score=float(scores[best]))
        return SAMResult(mask=(masks[0] * 255).astype(np.uint8), score=float(scores[0]))

    def generate_all_masks_high_res(self, image_bgr: np.ndarray) -> list[dict]:
        """DEPRECATED v0.3.3 — DISABLED.

        SAM2AutomaticMaskGenerator quét tự do toàn ảnh → ảo giác phân loại
        sai thớ đá lò sưởi / nệm sofa thành cấu trúc nhà.

        Pipeline mới CHỈ dùng SAM2ImagePredictor với point/box prompts từ VLM
        hoặc semantic centroids — không quét tự do.

        Returns: empty list (no-op).
        """
        log.warning(
            "SAM2AutomaticMaskGenerator DISABLED v0.3.3 — dùng predict_from_points/box."
        )
        return []

    def _grabcut_from_point(self, image_bgr: np.ndarray, point_xy) -> SAMResult:
        """GrabCut fallback. Định nghĩa rect ±15% quanh click point làm trimap PR_FGD."""
        h, w = image_bgr.shape[:2]
        x, y = int(point_xy[0]), int(point_xy[1])
        rect_w, rect_h = w // 4, h // 4
        rect = (
            max(0, x - rect_w // 2),
            max(0, y - rect_h // 2),
            min(w - 1, rect_w),
            min(h - 1, rect_h),
        )
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return SAMResult(mask=np.zeros((h, w), dtype=np.uint8), score=0.0)
        out = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 255, 0).astype(np.uint8)
        return SAMResult(mask=out, score=0.3)  # low confidence
