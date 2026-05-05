"""Interactive mask painter — vẽ mask bằng chuột trên cửa sổ OpenCV.

Phím tắt:
  - Chuột trái kéo : tô mask
  - Chuột phải kéo : xoá mask
  - +/= hoặc ]     : tăng brush
  - - hoặc [       : giảm brush
  - r              : reset mask
  - s              : save mask + thoát
  - q hoặc ESC     : thoát không save
  - h              : in help

Window có thể dùng cho ảnh tới 8K — sẽ tự fit-to-screen để hiển thị nhưng
mask được lưu ở resolution gốc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WINDOW_NAME = "Watermark Mask Painter — h: help"


class _PainterState:
    def __init__(self, image: np.ndarray, *, max_display_dim: int = 1280) -> None:
        self.image = image
        h, w = image.shape[:2]
        scale = min(1.0, max_display_dim / max(h, w))
        self.scale = scale
        self.disp_size = (int(w * scale), int(h * scale))
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.brush = max(8, int(min(h, w) * 0.01))
        self.drawing = False
        self.erasing = False
        self.dirty = True

    def to_orig(self, x: int, y: int) -> tuple[int, int]:
        return int(x / self.scale), int(y / self.scale)

    def render(self) -> np.ndarray:
        # Overlay mask đỏ trên ảnh để dễ thấy
        overlay = self.image.copy()
        red = np.zeros_like(self.image)
        red[..., 2] = 255
        idx = self.mask > 0
        overlay[idx] = cv2.addWeighted(overlay[idx], 0.4, red[idx], 0.6, 0)

        # Vẽ vòng brush trên góc dưới-trái cho biết kích thước
        h, w = overlay.shape[:2]
        cv2.circle(overlay, (40, h - 40), self.brush, (0, 255, 255), 2)
        cv2.putText(
            overlay, f"brush={self.brush}px  mask={int(idx.sum())}px",
            (80, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        if self.scale < 1.0:
            return cv2.resize(overlay, self.disp_size, interpolation=cv2.INTER_AREA)
        return overlay


def _make_callback(state: _PainterState):
    def on_mouse(event, x, y, flags, _param):  # noqa: ANN001
        ox, oy = state.to_orig(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            cv2.circle(state.mask, (ox, oy), state.brush, 255, -1)
            state.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.erasing = True
            cv2.circle(state.mask, (ox, oy), state.brush, 0, -1)
            state.dirty = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drawing:
                cv2.circle(state.mask, (ox, oy), state.brush, 255, -1)
                state.dirty = True
            elif state.erasing:
                cv2.circle(state.mask, (ox, oy), state.brush, 0, -1)
                state.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            state.erasing = False
    return on_mouse


def paint_mask(
    image: np.ndarray,
    *,
    initial_mask: np.ndarray | None = None,
    save_to: str | Path | None = None,
    max_display_dim: int = 1280,
) -> np.ndarray | None:
    """Mở cửa sổ vẽ mask. Trả mask uint8 hoặc None nếu user huỷ.

    initial_mask: nếu có sẽ load làm starting point (vd: từ auto_mask).
    save_to: nếu set, lưu mask khi user nhấn 's'.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Cần ảnh BGR/BGRA")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    state = _PainterState(image, max_display_dim=max_display_dim)
    if initial_mask is not None:
        if initial_mask.shape != image.shape[:2]:
            initial_mask = cv2.resize(
                initial_mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        state.mask = (initial_mask > 0).astype(np.uint8) * 255

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *state.disp_size)
    cv2.setMouseCallback(WINDOW_NAME, _make_callback(state))

    print("Painter ready. h = help, s = save, q/ESC = exit.")
    saved = False
    try:
        while True:
            if state.dirty:
                cv2.imshow(WINDOW_NAME, state.render())
                state.dirty = False
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):  # ESC
                break
            if key == ord("h"):
                print(
                    "[Painter] Left=draw  Right=erase  +/-: brush  r=reset  "
                    "s=save  q/ESC=quit"
                )
            elif key in (ord("+"), ord("="), ord("]")):
                state.brush = min(500, state.brush + 4)
                state.dirty = True
            elif key in (ord("-"), ord("[")):
                state.brush = max(2, state.brush - 4)
                state.dirty = True
            elif key == ord("r"):
                state.mask[:] = 0
                state.dirty = True
            elif key == ord("s"):
                saved = True
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)

    if save_to and saved:
        from .utils import write_image
        write_image(save_to, state.mask)
        logger.info("Đã lưu mask: %s", save_to)
    return state.mask if saved else None
