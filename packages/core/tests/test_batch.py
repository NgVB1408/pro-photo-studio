import cv2
import numpy as np
from pps_core.batch import IMAGE_EXTS, find_images, run_batch
from pps_core.utils import write_image


def _make_test_image(path, h=80, w=120):
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    cv2.putText(img, "X", (w - 30, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    write_image(path, img)


def test_find_images_in_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.webp").write_bytes(b"")

    flat = find_images(tmp_path)
    assert {p.name for p in flat} == {"a.jpg", "b.png"}

    deep = find_images(tmp_path, recursive=True)
    assert {p.name for p in deep} == {"a.jpg", "b.png", "c.webp"}


def test_find_images_glob(tmp_path):
    _make_test_image(tmp_path / "p1.jpg")
    _make_test_image(tmp_path / "p2.jpg")
    (tmp_path / "x.png").write_bytes(b"")
    files = find_images(tmp_path / "*.jpg")
    assert {p.name for p in files} == {"p1.jpg", "p2.jpg"}


def test_run_batch_processes_files(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    for i in range(3):
        _make_test_image(src_dir / f"img{i}.png")

    files = find_images(src_dir)
    result = run_batch(
        files,
        out_dir,
        workers=1,  # sequential để dễ debug + tránh spawn overhead
        skip_existing=False,
        progress=False,
        backend="opencv",
        opencv_method="telea",
        opencv_radius=3,
        boxes=[(90, 60, 30, 20)],
        dilate=2,
    )
    assert len(result.success) == 3
    assert len(result.failed) == 0
    for f in files:
        assert (out_dir / f"{f.stem}_clean.png").exists()


def test_run_batch_skip_existing(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()
    _make_test_image(src_dir / "a.png")
    # Tạo sẵn output để skip
    (out_dir / "a_clean.png").write_bytes(b"already")

    result = run_batch(
        find_images(src_dir),
        out_dir,
        workers=1,
        skip_existing=True,
        progress=False,
        boxes=[(0, 0, 5, 5)],
    )
    assert len(result.skipped) == 1
    assert (out_dir / "a_clean.png").read_bytes() == b"already"


def test_image_exts_contains_common():
    assert {".jpg", ".png", ".webp", ".jpeg"} <= IMAGE_EXTS
