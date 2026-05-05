"""Video pipeline — đọc video, inpaint từng frame, ghi lại.

3 chế độ mask:
  - "static": dùng 1 mask cố định (do user vẽ/build từ frame đầu) cho mọi frame.
              NHANH NHẤT, phù hợp 95% trường hợp watermark đứng yên.
  - "auto":   detect mask trên frame đầu, tái dùng cho mọi frame.
  - "redetect": chạy auto_mask trên TỪNG frame. Chậm nhưng bám watermark di động.

⚠ Lưu ý:
  - OpenCV VideoWriter KHÔNG ghi audio. Sau khi tạo video sạch, dùng FFmpeg
    để merge audio gốc:
        ffmpeg -i clean.mp4 -i input.mp4 -c:v copy -c:a copy -map 0:v -map 1:a output.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from .inpaint import InpaintBackend, inpaint
from .mask import (
    build_mask_from_boxes,
    build_mask_from_color,
    build_mask_from_image,
    combine_masks,
    dilate_mask,
)

logger = logging.getLogger(__name__)

MaskMode = Literal["static", "auto", "redetect"]


def _build_static_mask_from_frame(
    frame: np.ndarray,
    *,
    boxes: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    color_lower: tuple[int, int, int] | None = None,
    color_upper: tuple[int, int, int] = (255, 255, 255),
    use_auto: bool = False,
    dilate_iters: int = 2,
) -> np.ndarray:
    masks = []
    if boxes:
        masks.append(build_mask_from_boxes(frame, boxes))
    if mask_path:
        masks.append(build_mask_from_image(frame, mask_path))
    if color_lower:
        masks.append(build_mask_from_color(frame, lower=color_lower, upper=color_upper))
    if use_auto:
        from .detect import auto_mask

        masks.append(auto_mask(frame, dilate_iters=0))
    if not masks:
        raise ValueError("Cần ít nhất 1 nguồn mask: boxes/mask_path/color_lower/use_auto")
    out = combine_masks(masks)
    if dilate_iters > 0:
        out = dilate_mask(out, iterations=dilate_iters)
    return out


def _pick_fourcc(out_ext: str) -> int:
    ext = out_ext.lower().lstrip(".")
    # mp4v hoạt động trên OpenCV stock build cho .mp4
    if ext in {"mp4", "m4v"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext == "avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    if ext == "mov":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext == "webm":
        return cv2.VideoWriter_fourcc(*"VP90")
    raise ValueError(f"Định dạng video không hỗ trợ: {ext}")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mask_mode: MaskMode = "static",
    boxes: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    color_lower: tuple[int, int, int] | None = None,
    color_upper: tuple[int, int, int] = (255, 255, 255),
    backend: str = "opencv",
    opencv_method: str = "telea",
    opencv_radius: int = 3,
    lama_device: str = "cpu",
    hd_strategy: str = "crop",
    dilate_iters: int = 2,
    max_frames: int | None = None,
    progress: bool = True,
) -> dict:
    """Inpaint video. Trả dict thống kê (frames, fps, duration_s)."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames_target = min(n_frames_total, max_frames) if max_frames else n_frames_total

    fourcc = _pick_fourcc(out_path.suffix)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được VideoWriter cho {out_path}")

    backend_enum = InpaintBackend(backend)
    static_mask: np.ndarray | None = None
    bar = tqdm(
        total=n_frames_target if n_frames_target > 0 else None,
        disable=not progress,
        desc=in_path.name,
        unit="frame",
    )

    n_processed = 0
    try:
        while True:
            if max_frames and n_processed >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            if mask_mode == "redetect":
                from .detect import auto_mask

                m = auto_mask(frame, dilate_iters=dilate_iters)
            else:
                if static_mask is None:
                    static_mask = _build_static_mask_from_frame(
                        frame,
                        boxes=boxes,
                        mask_path=mask_path,
                        color_lower=color_lower,
                        color_upper=color_upper,
                        use_auto=(mask_mode == "auto"),
                        dilate_iters=dilate_iters,
                    )
                m = static_mask

            cleaned = inpaint(
                frame,
                m,
                backend=backend_enum,
                opencv_method=opencv_method,
                opencv_radius=opencv_radius,
                lama_device=lama_device,
                hd_strategy=hd_strategy,
            )
            writer.write(cleaned)
            n_processed += 1
            bar.update(1)
    finally:
        bar.close()
        cap.release()
        writer.release()

    duration = n_processed / fps if fps else 0
    logger.info(
        "Video xong: %d frames @ %.2f fps -> %s (%.1fs nội dung)",
        n_processed,
        fps,
        out_path,
        duration,
    )
    return {
        "frames": n_processed,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": round(duration, 2),
        "output": str(out_path),
        "ffmpeg_audio_merge": (
            f'ffmpeg -i "{out_path}" -i "{in_path}" -c:v copy -c:a copy '
            f'-map 0:v -map 1:a "{out_path.with_stem(out_path.stem + "_with_audio")}"'
        ),
    }
