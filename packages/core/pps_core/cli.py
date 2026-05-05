"""Command-line interface — `watermark <subcommand>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Settings, load_settings
from .inpaint import InpaintBackend, SUPPORTED_LAMA_MODELS, inpaint
from .mask import (
    build_mask_from_boxes,
    build_mask_from_color,
    build_mask_from_image,
    combine_masks,
    dilate_mask,
)
from .unsplash import UnsplashClient
from .utils import ensure_dir, read_image, timed, write_image

logger = logging.getLogger("watermark.cli")


def _parse_box(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"box phải dạng x,y,w,h — nhận {value!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"box phải là số nguyên: {value!r}") from exc
    return x, y, w, h


def _parse_rgb(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"color phải dạng B,G,R — nhận {value!r}")
    try:
        b, g, r = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"color phải là số nguyên: {value!r}") from exc
    for ch in (b, g, r):
        if not 0 <= ch <= 255:
            raise argparse.ArgumentTypeError(f"color channel ngoài [0,255]: {value!r}")
    return b, g, r


def _add_inpaint_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["opencv", "lama"],
                   help="Engine inpaint (mặc định lấy từ env INPAINT_BACKEND)")
    p.add_argument("--method", choices=["telea", "ns"],
                   help="Phương pháp OpenCV inpaint (telea | ns)")
    p.add_argument("--radius", type=int,
                   help="Bán kính inpaint OpenCV (px)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="Thiết bị chạy LaMa (auto chọn cuda/mps/cpu)")
    p.add_argument("--lama-model", choices=list(SUPPORTED_LAMA_MODELS), default="lama",
                   help="Mô hình LaMa (lama/mat/migan/zits/fcf/...)")
    p.add_argument("--hd-strategy", choices=["original", "resize", "crop"], default="crop",
                   help="Chiến lược ảnh HD (crop khuyến nghị cho ảnh ≥ 4K)")
    p.add_argument("--crop-margin", type=int, default=196,
                   help="Padding (px) quanh mask khi crop (LaMa)")
    p.add_argument("--crop-trigger-size", type=int, default=1280,
                   help="Chỉ áp HD strategy khi cạnh dài > giá trị này")
    p.add_argument("--resize-limit", type=int, default=2048,
                   help="Resize ảnh xuống cạnh tối đa này khi hd-strategy=resize")


def _add_mask_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--auto", action="store_true",
                   help="Tự phát hiện watermark (kết hợp text/bright/edge/logo)")
    p.add_argument("--auto-strategy",
                   choices=["text", "bright", "edge", "logo", "auto"], default="auto",
                   help="Chiến lược auto-detect: text=MSER, bright=logo trắng, "
                        "edge=cạnh bất thường, logo=corner-logo, auto=tổng hợp")
    p.add_argument("--no-border-only", action="store_true",
                   help="Khi --auto, KHÔNG giới hạn ở vùng rìa ảnh")
    p.add_argument("--box", action="append", type=_parse_box, default=[],
                   metavar="x,y,w,h",
                   help="Bounding box (x,y,w,h), có thể dùng nhiều lần")
    p.add_argument("--mask", help="File mask grayscale có sẵn")
    p.add_argument("--color-lower", type=_parse_rgb, metavar="B,G,R",
                   help="Mask theo dải màu — biên dưới (B,G,R)")
    p.add_argument("--color-upper", type=_parse_rgb, metavar="B,G,R",
                   default=(255, 255, 255),
                   help="Mask theo dải màu — biên trên (B,G,R)")
    p.add_argument("--dilate", type=int, default=2,
                   help="Số lần phồng mask (mở rộng vùng quanh logo)")
    p.add_argument("--save-mask", help="Lưu mask đã build ra file (debug)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watermark",
        description=(
            "Bộ công cụ production: xoá watermark (tự phát hiện, hàng loạt, "
            "video, Web UI), cải thiện ảnh + tải ảnh HD từ Unsplash/Dropbox."
        ),
    )
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--env-file", default=".env", help="Đường dẫn file .env (mặc định './.env')")

    sub = p.add_subparsers(dest="command", required=True)

    # remove
    rm = sub.add_parser("remove", help="Xoá watermark khỏi 1 ảnh")
    rm.add_argument("input", help="Đường dẫn ảnh đầu vào")
    rm.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    rm.add_argument("--quality", type=int, default=95, help="Chất lượng JPEG/WebP 1..100")
    rm.add_argument("--keep-exif", action="store_true",
                    help="Sao chép EXIF từ ảnh gốc (chỉ jpg/webp)")
    _add_mask_options(rm)
    _add_inpaint_options(rm)
    rm.set_defaults(func=cmd_remove)

    # auto: shortcut cho `remove --auto`
    au = sub.add_parser(
        "auto", help="Tự phát hiện + xoá watermark (không cần box/mask)",
    )
    au.add_argument("input", help="Đường dẫn ảnh đầu vào")
    au.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    au.add_argument("--auto-strategy",
                    choices=["text", "bright", "edge", "logo", "auto"], default="auto",
                    help="Chiến lược tự phát hiện (mặc định 'auto' kết hợp tất cả)")
    au.add_argument("--no-border-only", action="store_true",
                    help="KHÔNG giới hạn vùng rìa khi tự phát hiện")
    au.add_argument("--dilate", type=int, default=4,
                    help="Số lần phồng mask")
    au.add_argument("--save-mask", help="Lưu mask đã phát hiện ra file (debug)")
    au.add_argument("--quality", type=int, default=95)
    au.add_argument("--keep-exif", action="store_true")
    _add_inpaint_options(au)
    au.set_defaults(func=cmd_auto)

    # batch
    bt = sub.add_parser("batch", help="Xử lý cả thư mục/glob song song")
    bt.add_argument("source", help="Thư mục hoặc glob pattern (vd: 'photos/*.jpg')")
    bt.add_argument("-o", "--out-dir", required=True, help="Thư mục đầu ra")
    bt.add_argument("--recursive", action="store_true",
                    help="Đệ quy xuống các thư mục con")
    bt.add_argument("--workers", type=int, help="Số process song song")
    bt.add_argument("--suffix", default="_clean",
                    help="Hậu tố thêm vào tên file đầu ra (vd: _clean)")
    bt.add_argument("--out-format",
                    help="Định dạng đầu ra (.jpg/.png/.webp); mặc định giữ định dạng gốc")
    bt.add_argument("--no-skip-existing", action="store_true",
                    help="Ghi đè file đã tồn tại (mặc định bỏ qua)")
    _add_mask_options(bt)
    _add_inpaint_options(bt)
    bt.set_defaults(func=cmd_batch)

    # video
    vd = sub.add_parser("video", help="Xoá watermark khỏi video (xử lý từng frame)")
    vd.add_argument("input", help="Đường dẫn video đầu vào")
    vd.add_argument("-o", "--output", required=True, help="Đường dẫn video đầu ra")
    vd.add_argument("--mask-mode", choices=["static", "auto", "redetect"],
                    default="static",
                    help="static: 1 mask cho mọi frame. auto: detect frame đầu, "
                         "tái dùng. redetect: detect mỗi frame (chậm).")
    vd.add_argument("--max-frames", type=int,
                    help="Giới hạn N frame đầu (debug)")
    _add_mask_options(vd)
    _add_inpaint_options(vd)
    vd.set_defaults(func=cmd_video)

    # paint
    pt = sub.add_parser("paint", help="Vẽ mask thủ công bằng chuột (cửa sổ tương tác)")
    pt.add_argument("input", help="Đường dẫn ảnh")
    pt.add_argument("--save-to", required=True, help="Đường dẫn lưu mask PNG")
    pt.add_argument("--init-from-auto", action="store_true",
                    help="Khởi tạo mask từ kết quả tự phát hiện")
    pt.set_defaults(func=cmd_paint)

    # enhance — local enhancement, thay thế Autoenhance.ai
    en = sub.add_parser(
        "enhance",
        help=(
            "Cải thiện ảnh (cân bằng trắng / exposure / vibrance / sharpen) — "
            "không cần ML"
        ),
    )
    en.add_argument("input", help="Đường dẫn ảnh đầu vào")
    en.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    en.add_argument(
        "--preset",
        choices=["studio", "real_estate", "portrait", "product", "outdoor"],
        default="studio",
        help="Preset: studio (chất lượng cao nhất, mặc định), real_estate, "
             "portrait, product, outdoor",
    )
    en.add_argument("--quality", type=int, default=92,
                    help="Chất lượng JPEG/WebP 1..100")
    en.set_defaults(func=cmd_enhance)

    # batch-enhance
    eb = sub.add_parser(
        "batch-enhance", help="Cải thiện cả thư mục ảnh",
    )
    eb.add_argument("source", help="Thư mục hoặc glob pattern")
    eb.add_argument("-o", "--out-dir", required=True, help="Thư mục đầu ra")
    eb.add_argument("--recursive", action="store_true",
                    help="Đệ quy xuống các thư mục con")
    eb.add_argument(
        "--preset",
        choices=["studio", "real_estate", "portrait", "product", "outdoor"],
        default="studio",
        help="Preset cải thiện",
    )
    eb.add_argument("--workers", type=int,
                    help="(Hiện tại xử lý tuần tự — flag dự phòng)")
    eb.add_argument("--suffix", default="_enhanced",
                    help="Hậu tố thêm vào tên file đầu ra")
    eb.add_argument("--quality", type=int, default=92)
    eb.set_defaults(func=cmd_batch_enhance)

    # ============ Real estate (Autoenhance.ai parity) ============

    sky = sub.add_parser(
        "replace-sky", help="Thay trời xám bằng trời xanh / hoàng hôn / nhiều mây",
    )
    sky.add_argument("input", help="Đường dẫn ảnh đầu vào")
    sky.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    sky.add_argument(
        "--preset",
        choices=[
            "blue", "sunset", "overcast", "dramatic",
            "blue_clear", "blue_clouds", "sunset_warm", "golden_hour",
            "dramatic_storm", "overcast_soft",
        ],
        default="blue_clouds",
        help="Kiểu trời. Khuyên dùng các preset v2: blue_clouds, golden_hour, "
             "sunset_warm, dramatic_storm, overcast_soft (có mây + atmospheric haze). "
             "Các preset cũ (blue, sunset, overcast, dramatic) được map sang v2.",
    )
    sky.add_argument("--sky-image",
                     help="Ảnh trời thay thế tuỳ chọn (ưu tiên hơn preset)")
    sky.add_argument("--blend", type=float, default=1.0,
                     help="Mức trộn 0..1 (1 = thay 100%%)")
    sky.add_argument("--feather", type=int, default=21,
                     help="Kernel làm mềm rìa mask (px)")
    sky.add_argument("--save-mask", help="Lưu mask vùng trời để debug")
    sky.add_argument("--quality", type=int, default=95)
    sky.set_defaults(func=cmd_replace_sky)

    wp = sub.add_parser(
        "window-pull", help="Kéo sáng cửa sổ bị cháy (ảnh nội thất)",
    )
    wp.add_argument("input", help="Đường dẫn ảnh nội thất")
    wp.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    wp.add_argument("--strength", type=float, default=0.7,
                    help="Cường độ 0..1 (1 = mạnh nhất)")
    wp.add_argument("--quality", type=int, default=95)
    wp.set_defaults(func=cmd_window_pull)

    lw = sub.add_parser(
        "lawn-enhance", help="Tăng độ tươi cỏ (selective HSV)",
    )
    lw.add_argument("input", help="Đường dẫn ảnh đầu vào")
    lw.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    lw.add_argument("--sat-boost", type=float, default=0.5,
                    help="Tăng độ bão hoà 0..1")
    lw.add_argument("--hue-shift", type=int, default=-3,
                    help="Dịch hue (âm = về xanh tươi)")
    lw.add_argument("--value-lift", type=float, default=0.08,
                    help="Nâng vùng tối trong cỏ 0..1")
    lw.add_argument("--quality", type=int, default=95)
    lw.set_defaults(func=cmd_enhance_lawn)

    vc = sub.add_parser(
        "correct-vertical", help="Kéo thẳng đường dọc (sửa nghiêng phối cảnh)",
    )
    vc.add_argument("input", help="Đường dẫn ảnh")
    vc.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    vc.add_argument("--max-angle", type=float, default=8.0,
                    help="Chỉ sửa khi |độ nghiêng| ≤ giá trị này (độ)")
    vc.add_argument("--no-crop", action="store_true",
                    help="KHÔNG crop biên đen sau khi xoay")
    vc.add_argument("--quality", type=int, default=95)
    vc.set_defaults(func=cmd_correct_vertical)

    cs = sub.add_parser(
        "classify", help="Tự tag cảnh: nội thất / ngoại thất / trên cao (không cần ML)",
    )
    cs.add_argument("input", help="Đường dẫn ảnh")
    cs.add_argument("--json", action="store_true",
                    help="Xuất kết quả ở dạng JSON")
    cs.set_defaults(func=cmd_classify)

    re = sub.add_parser(
        "realestate",
        help=(
            "Pipeline đầy đủ: phân loại cảnh → kéo thẳng → "
            "thay trời / kéo cửa sổ / tăng tươi cỏ theo loại cảnh"
        ),
    )
    re.add_argument("input", help="Đường dẫn ảnh đầu vào")
    re.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    re.add_argument(
        "--sky-preset",
        choices=[
            "blue", "sunset", "overcast", "dramatic",
            "blue_clear", "blue_clouds", "sunset_warm", "golden_hour",
            "dramatic_storm", "overcast_soft",
        ],
        default="blue_clouds",
        help="Kiểu trời (xem replace-sky để biết chi tiết)",
    )
    re.add_argument("--sky-image", help="Ảnh trời thay thế tuỳ chọn")
    re.add_argument("--no-sky", action="store_true", help="Tắt thay trời")
    re.add_argument("--no-window", action="store_true",
                    help="Tắt kéo sáng cửa sổ")
    re.add_argument("--no-lawn", action="store_true",
                    help="Tắt tăng tươi cỏ")
    re.add_argument("--no-vertical", action="store_true",
                    help="Tắt kéo thẳng đường dọc")
    re.add_argument("--quality", type=int, default=95)
    re.set_defaults(func=cmd_realestate)

    # composite — restore từ ảnh gốc
    cp = sub.add_parser(
        "composite",
        help="Xoá logo bằng cách paste vùng từ ảnh gốc (cần cả 2 phiên bản)",
    )
    cp.add_argument("--original", required=True, help="Ảnh GỐC chưa có logo")
    cp.add_argument("--watermarked", required=True, help="Ảnh ĐÃ có logo")
    cp.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra")
    cp.add_argument("--no-align", action="store_true",
                    help="Bỏ tự căn chỉnh ORB (dùng khi 2 ảnh đã khít)")
    cp.add_argument("--threshold", type=int, default=25,
                    help="Ngưỡng pixel diff (mặc định 25)")
    cp.add_argument("--feather", type=int, default=3,
                    help="Làm mềm rìa mask khi blend (px)")
    cp.add_argument("--quality", type=int, default=95)
    cp.set_defaults(func=cmd_composite)

    # eval
    ev = sub.add_parser("eval", help="So sánh chất lượng 2 ảnh (PSNR / SSIM / MAE)")
    ev.add_argument("--reference", required=True, help="Ảnh tham chiếu (gốc)")
    ev.add_argument("--target", required=True, help="Ảnh đã xử lý")
    ev.add_argument("--json", action="store_true",
                    help="Xuất kết quả ở dạng JSON")
    ev.set_defaults(func=cmd_eval)

    # serve
    sv = sub.add_parser("serve", help="Mở Web UI (Gradio) — kéo thả ảnh")
    sv.add_argument("--host", default="127.0.0.1",
                    help="Địa chỉ bind (mặc định 127.0.0.1, dùng 0.0.0.0 cho LAN)")
    sv.add_argument("--port", type=int, default=7860, help="Cổng (mặc định 7860)")
    sv.add_argument("--share", action="store_true",
                    help="Tạo public URL qua gradio (lưu ý về privacy)")
    sv.set_defaults(func=cmd_serve)

    # unsplash-search / unsplash-download / dropbox-list / dropbox-download / pipeline
    us = sub.add_parser("unsplash-search", help="Tìm ảnh trên Unsplash")
    us.add_argument("query", help="Từ khoá tìm kiếm")
    us.add_argument("--per-page", type=int, default=10,
                    help="Số kết quả mỗi trang (1..30)")
    us.add_argument("--page", type=int, default=1, help="Số trang")
    us.add_argument("--orientation", choices=["landscape", "portrait", "squarish"],
                    help="Lọc theo bố cục")
    us.set_defaults(func=cmd_unsplash_search)

    ud = sub.add_parser("unsplash-download", help="Tải ảnh HD từ Unsplash")
    ud.add_argument("--query", help="Từ khoá (dùng với --count)")
    ud.add_argument("--id", help="ID ảnh cụ thể trên Unsplash")
    ud.add_argument("--count", type=int, default=1, help="Số ảnh cần tải")
    ud.add_argument("--size", choices=["raw", "full", "regular", "small", "thumb"],
                    default="raw", help="Kích thước (raw = chất lượng cao nhất)")
    ud.add_argument("--out", default="downloads", help="Thư mục lưu (mặc định 'downloads')")
    ud.add_argument("--orientation", choices=["landscape", "portrait", "squarish"])
    ud.set_defaults(func=cmd_unsplash_download)

    dl = sub.add_parser("dropbox-list", help="Liệt kê file trên Dropbox")
    dl.add_argument("--folder", default="", help="Đường dẫn folder (rỗng = root)")
    dl.add_argument("--recursive", action="store_true",
                    help="Đệ quy xuống các thư mục con")
    dl.set_defaults(func=cmd_dropbox_list)

    dd = sub.add_parser("dropbox-download", help="Tải file/folder từ Dropbox")
    dd.add_argument("--path", help="Đường dẫn file đơn lẻ")
    dd.add_argument("--folder", help="Đường dẫn folder (tải toàn bộ)")
    dd.add_argument("--recursive", action="store_true",
                    help="Đệ quy khi tải folder")
    dd.add_argument("--out", default="downloads", help="Thư mục lưu")
    dd.set_defaults(func=cmd_dropbox_download)

    pl = sub.add_parser("pipeline",
                        help="Tải ảnh từ Unsplash rồi (tuỳ chọn) xoá watermark")
    pl.add_argument("query", help="Từ khoá Unsplash")
    pl.add_argument("--count", type=int, default=1, help="Số ảnh tải")
    pl.add_argument("--out", default="output", help="Thư mục đầu ra")
    pl.add_argument("--size", default="raw",
                    choices=["raw", "full", "regular", "small", "thumb"])
    pl.add_argument("--no-clean", action="store_true",
                    help="Chỉ tải, không xoá watermark")
    pl.set_defaults(func=cmd_pipeline)

    return p


def _resolve_backend(args: argparse.Namespace, settings: Settings) -> InpaintBackend:
    name = getattr(args, "backend", None) or settings.inpaint_backend
    return InpaintBackend(name)


def _build_mask_from_args(args: argparse.Namespace, image):
    masks = []
    if getattr(args, "auto", False):
        from .detect import auto_mask
        masks.append(auto_mask(
            image,
            strategy=getattr(args, "auto_strategy", "auto"),
            border_only=not getattr(args, "no_border_only", False),
            dilate_iters=0,
        ))
    if getattr(args, "box", None):
        masks.append(build_mask_from_boxes(image, args.box))
    if getattr(args, "mask", None):
        masks.append(build_mask_from_image(image, args.mask))
    if getattr(args, "color_lower", None) is not None:
        masks.append(build_mask_from_color(
            image, lower=args.color_lower, upper=args.color_upper,
        ))
    if not masks:
        raise SystemExit(
            "Cần ít nhất một nguồn mask: --auto, --box, --mask, hoặc --color-lower"
        )
    mask = combine_masks(masks)
    if args.dilate > 0:
        mask = dilate_mask(mask, iterations=args.dilate)
    return mask


def _inpaint_from_args(image, mask, args: argparse.Namespace, settings: Settings):
    backend = _resolve_backend(args, settings)
    method = getattr(args, "method", None) or settings.opencv_method
    radius = getattr(args, "radius", None) or settings.opencv_radius
    return inpaint(
        image, mask,
        backend=backend,
        opencv_method=method,
        opencv_radius=radius,
        lama_device=getattr(args, "device", "auto"),
        lama_model=getattr(args, "lama_model", "lama"),
        hd_strategy=getattr(args, "hd_strategy", "crop"),
        crop_margin=getattr(args, "crop_margin", 196),
        crop_trigger_size=getattr(args, "crop_trigger_size", 1280),
        resize_limit=getattr(args, "resize_limit", 2048),
    )


def cmd_remove(args: argparse.Namespace, settings: Settings) -> int:
    image = read_image(args.input)
    mask = _build_mask_from_args(args, image)
    if args.save_mask:
        write_image(args.save_mask, mask)
        logger.info("Đã lưu mask: %s", args.save_mask)

    backend = _resolve_backend(args, settings)
    with timed(f"inpaint[{backend.value}]"):
        result = _inpaint_from_args(image, mask, args, settings)

    out = write_image(
        args.output, result, quality=args.quality,
        exif_source=args.input if args.keep_exif else None,
    )
    logger.info("✓ Đã ghi: %s", out)
    return 0


def cmd_auto(args: argparse.Namespace, settings: Settings) -> int:
    image = read_image(args.input)
    from .detect import auto_mask
    mask = auto_mask(
        image,
        strategy=args.auto_strategy,
        border_only=not args.no_border_only,
        dilate_iters=args.dilate,
    )
    if args.save_mask:
        write_image(args.save_mask, mask)
        logger.info("Đã lưu mask: %s", args.save_mask)

    if mask.sum() == 0:
        logger.error(
            "Tự phát hiện không tìm thấy watermark. Thử --auto-strategy khác "
            "(text/bright/edge/logo/auto) hoặc dùng `watermark paint` để vẽ "
            "mask thủ công."
        )
        return 1

    backend = _resolve_backend(args, settings)
    with timed(f"inpaint[{backend.value}]"):
        result = _inpaint_from_args(image, mask, args, settings)
    out = write_image(
        args.output, result, quality=args.quality,
        exif_source=args.input if args.keep_exif else None,
    )
    logger.info("✓ Đã ghi: %s", out)
    return 0


def cmd_batch(args: argparse.Namespace, settings: Settings) -> int:
    from .batch import find_images, run_batch
    files = find_images(args.source, recursive=args.recursive)
    if not files:
        logger.error("Không tìm thấy ảnh nào tại %s", args.source)
        return 1
    logger.info("Tìm thấy %d ảnh", len(files))

    backend_name = (args.backend or settings.inpaint_backend)
    inpaint_kwargs = {
        "backend": backend_name,
        "opencv_method": args.method or settings.opencv_method,
        "opencv_radius": args.radius or settings.opencv_radius,
        "lama_device": args.device,
        "hd_strategy": args.hd_strategy,
        "auto": args.auto,
        "auto_strategy": args.auto_strategy,
        "border_only": not args.no_border_only,
        "boxes": args.box if args.box else None,
        "mask_path": args.mask,
        "color_lower": args.color_lower,
        "color_upper": args.color_upper,
        "dilate": args.dilate,
        "quality": 95,
    }
    result = run_batch(
        files, args.out_dir,
        workers=args.workers,
        skip_existing=not args.no_skip_existing,
        suffix=args.suffix,
        out_format=args.out_format,
        **inpaint_kwargs,
    )
    print(result.summary())
    if result.failed:
        print("Các ảnh thất bại (10 đầu):")
        for path, err in result.failed[:10]:
            print(f"  ✗ {path.name}: {err}")
    return 0 if not result.failed else 1


def cmd_video(args: argparse.Namespace, settings: Settings) -> int:
    from .video import process_video
    info = process_video(
        args.input, args.output,
        mask_mode=args.mask_mode,
        boxes=args.box if args.box else None,
        mask_path=args.mask,
        color_lower=args.color_lower,
        color_upper=args.color_upper,
        backend=args.backend or settings.inpaint_backend,
        opencv_method=args.method or settings.opencv_method,
        opencv_radius=args.radius or settings.opencv_radius,
        lama_device=args.device,
        hd_strategy=args.hd_strategy,
        dilate_iters=args.dilate,
        max_frames=args.max_frames,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(
        "\nMẹo: Để giữ audio gốc, copy + chạy lệnh ffmpeg ở key "
        "'ffmpeg_audio_merge' phía trên."
    )
    return 0


def cmd_paint(args: argparse.Namespace, settings: Settings) -> int:
    from .paint import paint_mask
    image = read_image(args.input)
    initial = None
    if args.init_from_auto:
        from .detect import auto_mask
        initial = auto_mask(image, dilate_iters=2)
    mask = paint_mask(image, initial_mask=initial, save_to=args.save_to)
    if mask is None:
        logger.warning("Không lưu mask (người dùng huỷ).")
        return 1
    return 0


def cmd_enhance(args: argparse.Namespace, settings: Settings) -> int:
    from .enhance import enhance_preset
    img = read_image(args.input)
    with timed(f"enhance[{args.preset}]"):
        result = enhance_preset(img, args.preset)
    out = write_image(
        args.output, result, quality=args.quality,
        exif_source=args.input,
    )
    logger.info("✓ Đã cải thiện: %s", out)
    return 0


def cmd_batch_enhance(args: argparse.Namespace, settings: Settings) -> int:
    from .batch import find_images
    from .enhance import enhance_preset
    files = find_images(args.source, recursive=args.recursive)
    if not files:
        logger.error("Không tìm thấy ảnh tại %s", args.source)
        return 1
    out_dir = ensure_dir(args.out_dir)
    logger.info("Cải thiện %d ảnh với preset=%s", len(files), args.preset)
    ok, fail = 0, 0
    import time
    t0 = time.perf_counter()
    for f in files:
        try:
            img = read_image(f)
            result = enhance_preset(img, args.preset)
            out_name = f"{f.stem}{args.suffix}{f.suffix}"
            write_image(out_dir / out_name, result, quality=args.quality, exif_source=f)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: %s", f.name, exc)
            fail += 1
    dt = time.perf_counter() - t0
    print(
        f"Hoàn tất: thành công={ok}  thất bại={fail}  "
        f"tổng={dt:.1f}s ({dt / max(len(files), 1):.2f}s/ảnh)"
    )
    return 0 if fail == 0 else 1


def cmd_composite(args: argparse.Namespace, settings: Settings) -> int:
    from .composite import composite_from_original
    report = composite_from_original(
        args.original, args.watermarked, args.output,
        align=not args.no_align,
        diff_threshold=args.threshold,
        feather_px=args.feather,
        quality=args.quality,
    )
    print(f"✓ Đã lưu: {report.output_path}")
    print(f"  Kích thước     : {report.image_size[0]}×{report.image_size[1]}")
    print(f"  Pixel khác biệt: {report.diff_pixels} ({report.mask_coverage_pct}%)")
    print(f"  Đã căn chỉnh   : {'có' if report.used_align else 'không'}")
    return 0


def cmd_replace_sky(args: argparse.Namespace, settings: Settings) -> int:
    import cv2
    from .realestate import replace_sky, load_sky_from_path
    img = read_image(args.input)
    sky_img = load_sky_from_path(args.sky_image) if args.sky_image else None
    with timed("replace_sky"):
        out, mask = replace_sky(
            img, preset=args.preset, sky_image=sky_img,
            blend_strength=args.blend, feather=args.feather,
        )
    write_image(args.output, out, quality=args.quality, exif_source=args.input)
    if args.save_mask:
        cv2.imwrite(str(args.save_mask), mask)
    coverage = float((mask > 128).mean()) * 100
    print(f"✓ Đã lưu: {args.output}  (vùng trời chiếm {coverage:.1f}%)")
    return 0


def cmd_window_pull(args: argparse.Namespace, settings: Settings) -> int:
    from .realestate import window_pull
    img = read_image(args.input)
    with timed("window_pull"):
        out, mask = window_pull(img, strength=args.strength)
    write_image(args.output, out, quality=args.quality, exif_source=args.input)
    coverage = float((mask > 128).mean()) * 100
    print(f"✓ Đã lưu: {args.output}  (cửa sổ chiếm {coverage:.2f}%)")
    return 0


def cmd_enhance_lawn(args: argparse.Namespace, settings: Settings) -> int:
    from .realestate import enhance_lawn
    img = read_image(args.input)
    with timed("enhance_lawn"):
        out, mask = enhance_lawn(
            img, sat_boost=args.sat_boost,
            hue_shift=args.hue_shift, value_lift=args.value_lift,
        )
    write_image(args.output, out, quality=args.quality, exif_source=args.input)
    coverage = float((mask > 128).mean()) * 100
    print(f"✓ Đã lưu: {args.output}  (cỏ chiếm {coverage:.1f}%)")
    return 0


def cmd_correct_vertical(args: argparse.Namespace, settings: Settings) -> int:
    from .realestate import correct_vertical
    img = read_image(args.input)
    with timed("correct_vertical"):
        out, report = correct_vertical(
            img, max_angle=args.max_angle, crop=not args.no_crop,
        )
    write_image(args.output, out, quality=args.quality, exif_source=args.input)
    print(f"✓ Đã lưu: {args.output}")
    print(
        f"  Độ nghiêng phát hiện: {report.angle_deg:.2f}° "
        f"(số đường dọc: {report.line_count}, "
        f"đã xoay: {'có' if report.rotated else 'không'})"
    )
    return 0


def cmd_classify(args: argparse.Namespace, settings: Settings) -> int:
    from .realestate import classify_scene
    img = read_image(args.input)
    report = classify_scene(img)
    data = {
        "tag": report.tag,
        "confidence": round(report.confidence, 3),
        "sky_ratio": round(report.sky_ratio, 4),
        "grass_ratio": round(report.grass_ratio, 4),
        "avg_brightness": round(report.avg_brightness, 4),
        "edge_density": round(report.edge_density, 4),
    }
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        tag_vi = {
            "interior": "Nội thất",
            "exterior": "Ngoại thất",
            "aerial": "Trên cao (drone)",
            "unknown": "Không xác định",
        }.get(data["tag"], data["tag"])
        print(f"Loại cảnh         : {tag_vi} ({data['tag']})")
        print(f"Độ tin cậy        : {data['confidence']}")
        print(f"Tỉ lệ trời        : {data['sky_ratio']}")
        print(f"Tỉ lệ cỏ          : {data['grass_ratio']}")
        print(f"Độ sáng trung bình: {data['avg_brightness']}")
        print(f"Mật độ cạnh       : {data['edge_density']}")
    return 0


def cmd_realestate(args: argparse.Namespace, settings: Settings) -> int:
    from .realestate import enhance_realestate_full, load_sky_from_path
    img = read_image(args.input)
    sky_img = load_sky_from_path(args.sky_image) if args.sky_image else None
    with timed("realestate_full"):
        out, report = enhance_realestate_full(
            img,
            sky_preset=args.sky_preset,
            sky_image=sky_img,
            enable_sky=not args.no_sky,
            enable_window_pull=not args.no_window,
            enable_lawn=not args.no_lawn,
            enable_vertical=not args.no_vertical,
        )
    write_image(args.output, out, quality=args.quality, exif_source=args.input)
    tag_vi = {
        "interior": "Nội thất",
        "exterior": "Ngoại thất",
        "aerial": "Trên cao",
        "unknown": "Không xác định",
    }.get(report.scene.tag, report.scene.tag)
    print(f"✓ Đã lưu: {args.output}")
    print(f"  Loại cảnh       : {tag_vi} (tin cậy {report.scene.confidence:.2f})")
    print(
        f"  Trời/Cỏ         : "
        f"{report.scene.sky_ratio:.2%} / {report.scene.grass_ratio:.2%}"
    )
    print(
        f"  Kéo thẳng       : {report.vertical.angle_deg:+.2f}° "
        f"({'đã sửa' if report.vertical.rotated else 'không sửa'})"
    )
    print(f"  Đã thay trời    : {'có' if report.sky_replaced else 'không'}")
    print(f"  Đã kéo cửa sổ   : {'có' if report.windows_recovered else 'không'}")
    print(f"  Đã tăng tươi cỏ : {'có' if report.lawn_enhanced else 'không'}")
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from .quality import compare_files
    report = compare_files(args.reference, args.target)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        d = report.as_dict()
        print(f"PSNR             : {d['psnr']:.3f} dB")
        print(f"SSIM             : {d['ssim']:.5f}")
        print(f"MAE              : {d['mae']:.4f}")
        print(f"Sai khác lớn nhất: {d['max_diff']}")
        print(f"Tỉ lệ pixel khác : {d['different_pixels_ratio']*100:.3f}%")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    from .webui import serve
    logger.info("Khởi động Web UI tại http://%s:%d", args.host, args.port)
    serve(host=args.host, port=args.port, share=args.share)
    return 0


def cmd_unsplash_search(args: argparse.Namespace, settings: Settings) -> int:
    client = UnsplashClient(settings.require_unsplash())
    photos = client.search(
        args.query, per_page=args.per_page, page=args.page,
        orientation=args.orientation,
    )
    if not photos:
        print("Không có kết quả.")
        return 0
    for ph in photos:
        print(f"{ph.id}\t{ph.width}x{ph.height}\t{ph.user_name}\t{ph.description or ''}")
    return 0


def cmd_unsplash_download(args: argparse.Namespace, settings: Settings) -> int:
    client = UnsplashClient(settings.require_unsplash())
    out_dir = ensure_dir(args.out)
    if args.id:
        photo = client.get_photo(args.id)
        client.download(photo, size=args.size, out_dir=out_dir)
        return 0
    if args.query:
        photos = client.search(args.query, per_page=args.count,
                               orientation=args.orientation)
        if not photos:
            logger.warning("Không có kết quả cho %r", args.query)
            return 1
        client.download_many(photos, size=args.size, out_dir=out_dir)
        return 0
    logger.error("Cần truyền --id hoặc --query")
    return 2


def cmd_dropbox_list(args: argparse.Namespace, settings: Settings) -> int:
    from .dropbox_client import DropboxClient
    files = DropboxClient().list_folder(args.folder, recursive=args.recursive)
    if not files:
        print("Thư mục rỗng.")
        return 0
    for i, f in enumerate(files, 1):
        print(f"[{i}] {f.path} ({f.size_mb:.2f} MB)")
    return 0


def cmd_dropbox_download(args: argparse.Namespace, settings: Settings) -> int:
    from .dropbox_client import DropboxClient
    client = DropboxClient()
    if args.path:
        client.download(args.path, out_dir=args.out)
        return 0
    if args.folder is not None:
        client.download_folder(args.folder, out_dir=args.out, recursive=args.recursive)
        return 0
    logger.error("Cần truyền --path hoặc --folder")
    return 2


def cmd_pipeline(args: argparse.Namespace, settings: Settings) -> int:
    client = UnsplashClient(settings.require_unsplash())
    out_dir = ensure_dir(args.out)
    photos = client.search(args.query, per_page=args.count)
    if not photos:
        logger.warning("Không có kết quả cho %r", args.query)
        return 1
    for ph in photos:
        path = client.download(ph, size=args.size, out_dir=out_dir)
        if args.no_clean:
            continue
        img = read_image(path)
        h, w = img.shape[:2]
        box_w, box_h = w // 5, h // 10
        mask = build_mask_from_boxes(img, [(w - box_w, h - box_h, box_w, box_h)])
        cleaned = inpaint(img, mask, backend=InpaintBackend(settings.inpaint_backend))
        out_path = Path(out_dir) / f"clean_{path.name}"
        write_image(out_path, cleaned)
        logger.info("Pipeline: %s -> %s", path.name, out_path.name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.env_file)
    try:
        return args.func(args, settings)
    except KeyboardInterrupt:
        logger.warning("Đã huỷ theo yêu cầu người dùng")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Lỗi: %s", exc, exc_info=settings.log_level == "DEBUG")
        return 1


if __name__ == "__main__":
    sys.exit(main())
