import cv2
import numpy as np
import pytest

from pps_core.video import _pick_fourcc, process_video


def _make_test_video(path, frames=10, w=160, h=120, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("Codec mp4v không khả dụng trong môi trường này")
    try:
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (50 + i * 5, 80, 120)
            cv2.putText(frame, "WM", (w - 50, h - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_pick_fourcc_known():
    assert _pick_fourcc(".mp4") == cv2.VideoWriter_fourcc(*"mp4v")
    assert _pick_fourcc(".avi") == cv2.VideoWriter_fourcc(*"XVID")


def test_pick_fourcc_unknown_raises():
    with pytest.raises(ValueError):
        _pick_fourcc(".mkv")


def test_process_video_static_box(tmp_path):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    _make_test_video(src, frames=8, w=160, h=120)

    info = process_video(
        src, dst,
        mask_mode="static",
        boxes=[(110, 90, 50, 30)],
        backend="opencv",
        progress=False,
    )
    assert info["frames"] > 0
    assert dst.exists()
    cap = cv2.VideoCapture(str(dst))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n > 0


def test_process_video_max_frames_limit(tmp_path):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    _make_test_video(src, frames=20)
    info = process_video(
        src, dst,
        mask_mode="static",
        boxes=[(100, 80, 50, 30)],
        backend="opencv",
        max_frames=3,
        progress=False,
    )
    assert info["frames"] == 3


def test_process_video_missing_mask_raises(tmp_path):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    _make_test_video(src, frames=4)
    with pytest.raises(ValueError):
        process_video(src, dst, mask_mode="static", backend="opencv", progress=False)
