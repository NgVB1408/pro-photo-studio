"""Gradio web UI — drag-and-drop browser interface.

Lazy import gradio để không bắt buộc cài cho user CLI-only.
Cài: pip install gradio>=4.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_settings
from .detect import auto_mask
from .inpaint import InpaintBackend, inpaint
from .mask import combine_masks, dilate_mask

logger = logging.getLogger(__name__)

# Cho phép ảnh tới 8K (cạnh dài 8192px). Ảnh > 8K mới bị resize.
# 6K real estate (6000x4000) sẽ GIỮ NGUYÊN 6K.
MAX_DIMENSION = 8192

# Chất lượng JPEG đầu ra cho mọi tab — ưu tiên chất lượng tối đa.
OUTPUT_JPEG_QUALITY = 95

# Tuỳ chọn kích thước đầu ra (cạnh dài). Ảnh sẽ chỉ thu nhỏ — KHÔNG upscale.
OUTPUT_SIZE_CHOICES: list[tuple[str, int | None]] = [
    ("Giữ nguyên (chất lượng tối đa)", None),
    ("6K (cạnh dài 6000)", 6000),
    ("4K (cạnh dài 3840)", 3840),
    ("Full HD (cạnh dài 1920)", 1920),
]
_OUTPUT_SIZE_LABELS = [label for label, _ in OUTPUT_SIZE_CHOICES]
_OUTPUT_SIZE_MAP = dict(OUTPUT_SIZE_CHOICES)


def _resize_output(bgr: np.ndarray, label: str) -> np.ndarray:
    """Thu nhỏ ảnh xuống cạnh dài target nếu cần. KHÔNG bao giờ upscale.

    Dùng INTER_LANCZOS4 (chất lượng cao nhất) khi downscale.
    """
    target = _OUTPUT_SIZE_MAP.get(label)
    if target is None:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= target:
        return bgr  # đã nhỏ hơn hoặc bằng target → không upscale
    scale = target / longest
    new_w, new_h = round(w * scale), round(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _bgr_from_rgb_or_rgba(rgb_or_rgba: np.ndarray) -> np.ndarray:
    if rgb_or_rgba.ndim == 2:
        return cv2.cvtColor(rgb_or_rgba, cv2.COLOR_GRAY2BGR)
    if rgb_or_rgba.shape[2] == 4:
        return cv2.cvtColor(rgb_or_rgba, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(rgb_or_rgba, cv2.COLOR_RGB2BGR)


def _rgb_from_bgr(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _maybe_downscale(bgr: np.ndarray, mask_layer: np.ndarray | None):
    """Auto-downscale ảnh quá to. Trả (bgr, mask_layer, scale)."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIMENSION:
        return bgr, mask_layer, 1.0
    scale = MAX_DIMENSION / longest
    new_w, new_h = int(w * scale), int(h * scale)
    logger.warning(
        "Ảnh %dx%d quá to (>%d) — auto-resize xuống %dx%d (scale=%.3f) cho UI nhanh.",
        w,
        h,
        MAX_DIMENSION,
        new_w,
        new_h,
        scale,
    )
    bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = None
    if mask_layer is not None:
        mask_small = cv2.resize(
            mask_layer,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return bgr_small, mask_small, scale


def _process(
    image_dict,
    backend: str,
    dilate_iters: int,
    hd_strategy: str,
    lama_model: str,
    output_size: str = "Giữ nguyên (chất lượng tối đa)",
    progress=None,
):
    """Gradio handler cho tab Xoá watermark.

    image_dict = {"background": rgb, "layers": [rgba mask vẽ tay]}.
    """
    t0 = time.perf_counter()
    if progress is not None:
        progress(0.0, desc="Đang đọc ảnh")

    if image_dict is None:
        return None, "⚠ Hãy tải lên ảnh."

    bg = image_dict.get("background") if isinstance(image_dict, dict) else image_dict
    if bg is None:
        return None, "⚠ Không đọc được ảnh."

    bgr = _bgr_from_rgb_or_rgba(np.asarray(bg))
    h0, w0 = bgr.shape[:2]
    logger.info("[ui] Nhận ảnh %dx%d (%.1f MP)", w0, h0, w0 * h0 / 1e6)

    user_mask_layer = None
    layers = image_dict.get("layers", []) if isinstance(image_dict, dict) else []
    if layers:
        layer0 = np.asarray(layers[0])
        if layer0.ndim == 3 and layer0.shape[2] == 4:
            user_mask_layer = (layer0[..., 3] > 0).astype(np.uint8) * 255

    if progress is not None:
        progress(0.15, desc="Resize nếu ảnh quá to")
    bgr, user_mask_layer, scale = _maybe_downscale(bgr, user_mask_layer)
    h, w = bgr.shape[:2]

    user_mask = user_mask_layer
    used_auto = False
    if user_mask is None or user_mask.sum() == 0:
        if progress is not None:
            progress(0.35, desc="Tự phát hiện watermark (chưa vẽ mask)")
        logger.info(
            "[ui] Chạy auto-detect (strategy=logo) trên %dx%d ...",
            w,
            h,
        )
        user_mask = auto_mask(bgr, strategy="logo", dilate_iters=0)
        used_auto = True
        if user_mask.sum() == 0:
            return (
                None,
                "⚠ Không phát hiện được watermark tự động. Hãy vẽ vùng "
                "watermark trên ảnh (chọn brush, kéo chuột) rồi bấm "
                "'Xoá watermark'.",
            )

    if progress is not None:
        progress(0.6, desc=f"Phồng mask ({dilate_iters} lần)")
    final_mask = dilate_mask(combine_masks([user_mask]), iterations=dilate_iters)

    if progress is not None:
        progress(0.7, desc=f"Đang inpaint bằng {backend}")
    logger.info(
        "[ui] Inpaint backend=%s, mask_pixels=%d (%.2f%%), nguồn=%s",
        backend,
        int((final_mask > 0).sum()),
        100 * float((final_mask > 0).sum()) / final_mask.size,
        "tự động" if used_auto else "vẽ tay",
    )
    settings = load_settings()
    t_inpaint = time.perf_counter()
    try:
        result_bgr = inpaint(
            bgr,
            final_mask,
            backend=InpaintBackend(backend),
            opencv_method=settings.opencv_method,
            opencv_radius=settings.opencv_radius,
            lama_device="auto",
            lama_model=lama_model,
            hd_strategy=hd_strategy,
        )
    except Exception as exc:
        return None, f"❌ Lỗi inpaint: {type(exc).__name__}: {exc}"
    inpaint_dt = time.perf_counter() - t_inpaint

    if progress is not None:
        progress(0.92, desc="Đang resize đầu ra")
    result_bgr = _resize_output(result_bgr, output_size)

    if progress is not None:
        progress(0.95, desc="Đang encode kết quả")
    total = time.perf_counter() - t0
    coverage = float((final_mask > 0).sum()) / final_mask.size * 100
    extra = ""
    if scale < 1.0:
        extra = f"  | Tự resize từ {w0}×{h0} → {w}×{h} cho hiệu năng UI"
    info = (
        f"Engine: {backend} ({inpaint_dt:.2f}s) | "
        f"Mask: {coverage:.2f}% ({'tự động' if used_auto else 'vẽ tay'}) | "
        f"Đầu ra: {result_bgr.shape[1]}×{result_bgr.shape[0]} | "
        f"Tổng: {total:.2f}s{extra}"
    )
    logger.info("[ui] Done: %s", info)
    return _rgb_from_bgr(result_bgr), info


def _is_lama_available() -> bool:
    """Probe nhanh: iopaint có thể import được không."""
    try:
        import iopaint  # noqa: F401

        return True
    except ImportError:
        return False


_CUSTOM_CSS = """
/* ===== Layout ===== */
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.gradio-container { font-size: 15.5px !important; }
footer { display: none !important; }

/* ===== Header banner — TEXT TRẮNG NỔI BẬT ===== */
#wm-banner {
    background: linear-gradient(135deg, #4338ca 0%, #1d4ed8 50%, #0369a1 100%);
    color: #ffffff !important;
    padding: 2rem 2.4rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(67, 56, 202, 0.32);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
}
#wm-banner *, #wm-banner h1, #wm-banner h2, #wm-banner h3,
#wm-banner p, #wm-banner strong, #wm-banner em, #wm-banner code,
#wm-banner span, #wm-banner li {
    color: #ffffff !important;
    font-weight: 600 !important;
}
#wm-banner h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.6rem 0 !important;
    letter-spacing: -0.5px;
}
#wm-banner p { font-size: 1.1rem !important; margin: 0 !important; line-height: 1.55; }
#wm-banner strong { font-weight: 800 !important; }
#wm-banner code {
    background: rgba(255,255,255,0.25);
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    font-size: 0.95em;
    font-weight: 700 !important;
}

/* ===== Tabs ===== */
.tab-nav { border-bottom: 2px solid #e5e7eb !important; }
.tab-nav button {
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    padding: 0.85rem 1.3rem !important;
    color: #475569 !important;
}
.tab-nav button.selected {
    border-bottom: 3px solid #4338ca !important;
    color: #4338ca !important;
    background: #eef2ff !important;
}

/* ===== Primary buttons ===== */
.gr-button-primary, button.primary,
button.lg.primary, .gradio-container button.primary {
    font-weight: 800 !important;
    font-size: 1.08rem !important;
    padding: 0.85rem 1.4rem !important;
    background: linear-gradient(135deg, #4338ca 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(67,56,202,0.32) !important;
    letter-spacing: 0.2px;
}
.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(67,56,202,0.42) !important;
    background: linear-gradient(135deg, #3730a3 0%, #1e40af 100%) !important;
}

/* ===== Markdown body — DỄ ĐỌC ===== */
.gradio-container .prose, .gradio-container .markdown {
    font-size: 15.5px !important;
    line-height: 1.7 !important;
    color: #1e293b !important;
}
.gradio-container .prose strong, .gradio-container .markdown strong {
    font-weight: 800 !important;
    color: #0f172a !important;
}
.gradio-container .prose code, .gradio-container .markdown code {
    background: #eef2ff !important;
    color: #4338ca !important;
    padding: 0.15rem 0.45rem;
    border-radius: 5px;
    font-weight: 700 !important;
    font-size: 0.92em;
}

/* ===== Cards ===== */
.guide-card, .warn-card, .success-card {
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin: 0.6rem 0 1rem 0;
    font-size: 15.5px;
    line-height: 1.65;
}
.guide-card {
    background: #eef2ff;
    border-left: 5px solid #4338ca;
    color: #1e293b;
}
.guide-card strong { color: #312e81 !important; }
.warn-card {
    background: #fef3c7;
    border-left: 5px solid #d97706;
    color: #1e293b;
}
.warn-card strong { color: #78350f !important; }
.success-card {
    background: #d1fae5;
    border-left: 5px solid #059669;
    color: #1e293b;
}
.success-card strong { color: #064e3b !important; }
.guide-card h3, .warn-card h3, .success-card h3 {
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
    font-size: 1.1rem !important;
}

/* ===== Section headings rõ ràng ===== */
.gradio-container h1 { font-size: 1.8rem !important; font-weight: 800 !important; }
.gradio-container h2 {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin-top: 1.4rem !important;
    color: #1e293b !important;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.4rem;
}
.gradio-container h3 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #4338ca !important;
    margin-top: 1.1rem !important;
}

/* ===== Tables (HDSD) — viền rõ, text to ===== */
.gradio-container table {
    border-collapse: collapse;
    margin: 0.8rem 0;
    width: 100%;
    font-size: 15px;
}
.gradio-container table th {
    background: #4338ca !important;
    color: #ffffff !important;
    font-weight: 700;
    padding: 0.7rem 1rem;
    text-align: left;
    border: 1px solid #312e81;
}
.gradio-container table td {
    padding: 0.65rem 1rem;
    border: 1px solid #e5e7eb;
    color: #1e293b !important;
}
.gradio-container table tr:nth-child(even) td { background: #f8fafc; }
.gradio-container table tr:hover td { background: #eef2ff; }
.gradio-container table strong { color: #4338ca !important; }

/* ===== Form labels rõ hơn ===== */
.gradio-container label .label, label > .label-content,
.gradio-container .gr-form > * > label,
span[data-testid="block-label"], .block label > span {
    font-weight: 700 !important;
    color: #1e293b !important;
    font-size: 14.5px !important;
}

/* ===== Accordion summary ===== */
.gradio-container details summary {
    font-weight: 700 !important;
    color: #4338ca !important;
    padding: 0.7rem 1rem !important;
    background: #eef2ff !important;
    border-radius: 8px !important;
    font-size: 15.5px !important;
}
.gradio-container details[open] summary {
    background: #c7d2fe !important;
    margin-bottom: 0.4rem;
}

/* ===== Sliders + dropdowns nhiều space hơn ===== */
.gr-slider, .gr-dropdown { margin-bottom: 0.6rem; }
.gr-info, .info { font-size: 13.5px !important; color: #475569 !important; }
"""


def _build_theme(gr):
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="*neutral_50",
        block_radius="*radius_lg",
        block_shadow="0 1px 3px rgba(0,0,0,0.06)",
        button_primary_text_color="#ffffff",
    )


_HDSD_OVERVIEW = """
## 📖 Hướng dẫn sử dụng nhanh

**Watermark Toolkit** là bộ công cụ xử lý ảnh production, chạy 100% trên máy
chủ này — ảnh KHÔNG gửi đi đâu khác. 7 chức năng chính, chia 2 nhóm:

### 🧹 Nhóm 1 — Xoá watermark / logo
| Tab | Khi nào dùng | Chất lượng |
|---|---|---|
| 🎨 **Xoá watermark (1 ảnh)** | Có 1 ảnh có logo, không có ảnh gốc | Tốt |
| 🪄 **Ghép từ ảnh gốc** | Có CẢ ảnh gốc và ảnh có logo | ⭐ Hoàn hảo |
| 📦 **Ghép hàng loạt** | Có 2 thư mục (gốc + có logo), tên file giống nhau | ⭐ Hoàn hảo |
| ⚡ **Tự xoá hàng loạt** | Có 1 thư mục ảnh có logo, không có ảnh gốc | Tốt |

### ✨ Nhóm 2 — Cải thiện ảnh (thay Autoenhance.ai)
| Tab | Khi nào dùng |
|---|---|
| ✨ **Cải thiện ảnh** | 1 ảnh thô → ảnh polished (5 preset + tinh chỉnh) |
| 🚀 **Cải thiện hàng loạt** | N ảnh thô → file zip ảnh đã cải thiện |
| 🏠 **Bất động sản** | Ảnh BĐS: tự thay trời / kéo cửa sổ / kéo thẳng dọc / tăng tươi cỏ |

### 🎯 Bạn nên bắt đầu từ tab nào?

```
       ┌─────────────────────────────┐
       │ Bạn có ảnh GỐC (chưa logo)? │
       └──────────────┬──────────────┘
                      │
        ┌─────────────┴─────────────┐
       Có                            Không
        │                              │
        ▼                              ▼
  Nhiều ảnh hay 1?              Nhiều ảnh hay 1?
   │        │                    │        │
  1 ảnh    Nhiều ảnh           1 ảnh    Nhiều ảnh
   │        │                    │        │
   ▼        ▼                    ▼        ▼
 🪄 Ghép  📦 Ghép              🎨 Xoá   ⚡ Tự xoá
        hàng loạt              watermark  hàng loạt
```
"""


_HDSD_TABS_DETAIL = """
## 📚 Hướng dẫn chi tiết từng tab

### 🎨 Tab "Xoá watermark (1 ảnh)"
**Mục đích:** Xoá 1 watermark khỏi 1 ảnh khi không có ảnh gốc.

**Cách dùng:**
1. Tải ảnh lên ô bên trái
2. **Cách A — tự động**: Bỏ trống mask → bấm "🪄 Xoá watermark"
3. **Cách B — vẽ tay**: Chọn brush đỏ, kéo chuột lên vùng có logo → bấm "🪄 Xoá watermark"
4. Chờ vài giây → ảnh kết quả hiện bên phải

**Mẹo:**
- Engine `opencv` (mặc định): ~1s ảnh 4K / ~2s ảnh 6K, đủ cho 99% case
- Engine `lama`: chất lượng cao hơn nhưng cần GPU (chỉ hiện khi đã cài torch + iopaint)
- "Số lần phồng mask" 3 = vừa đủ cho watermark có viền soft. Tăng lên 5–8 nếu logo có shadow/glow

---

### 🪄 Tab "Ghép từ ảnh gốc (2 ảnh — chính xác tuyệt đối)"
**Mục đích:** Khi có cả ảnh gốc (chưa logo) và ảnh đã chỉnh có logo — kết quả PIXEL-PERFECT.

**Cách dùng:**
1. Tải ảnh GỐC vào ô 1
2. Tải ảnh CÓ LOGO vào ô 2
3. Để mặc định "Tự căn chỉnh ORB" nếu 2 ảnh có thể lệch nhẹ
4. Bấm "🪄 Ghép từ ảnh gốc"

**Khi nào dùng:**
- Photographer giao 2 phiên bản: bản raw (có watermark Autoenhance) + bản preview
- Có ảnh studio gốc + ảnh đã thêm logo brand

**Mẹo:**
- "Ngưỡng khác biệt" = 15 (mặc định): bắt cả watermark mờ. Tăng = chỉ bắt watermark rõ
- "Làm mềm rìa" = 3: blend Poisson tự nhiên giữa 2 ảnh

---

### 📦 Tab "Ghép hàng loạt (nhiều cặp ảnh)"
**Mục đích:** Xử lý cả ngàn cặp ảnh cùng lúc.

**Cách dùng:**
1. Chọn nhiều file vào ô "Thư mục ảnh GỐC"
2. Chọn nhiều file vào ô "Thư mục ảnh CÓ logo"
3. Bấm "📦 Xử lý hàng loạt" → tải file zip kết quả

**⚠ QUAN TRỌNG:** tên file 2 thư mục phải GIỐNG NHAU (vd `0C1A5014.jpeg` ở cả 2). Tool ghép cặp theo tên (bỏ extension).

---

### ⚡ Tab "Tự xoá hàng loạt"
**Mục đích:** Có 1 thư mục ảnh có watermark, không có ảnh gốc → tự phát hiện + xoá tất cả.

**Cách dùng:**
1. Chọn nhiều file vào ô "Thư mục ảnh CÓ logo"
2. Bấm "⚡ Xử lý hàng loạt" → tải file zip

**Lưu ý:** Phụ thuộc bộ phát hiện tự động — kết quả không đẹp bằng tab "Ghép từ ảnh gốc". Nếu không vừa ý, hãy lấy ảnh gốc và dùng tab Ghép.

---

### ✨ Tab "Cải thiện ảnh"
**Mục đích:** Thay thế Autoenhance.ai. 1 ảnh thô → 1 ảnh polished.

**Cách dùng:**
1. Tải ảnh thô lên
2. Chọn preset:
   - **studio** (mặc định, khuyên dùng): chất lượng cao nhất, WB robust + chi tiết halo-free
   - **real_estate**: BĐS
   - **portrait**: chân dung (giữ tone da, denoise)
   - **product**: sản phẩm
   - **outdoor**: ngoài trời (mạnh tay hơn)
3. (Tuỳ chọn) Mở "Tinh chỉnh thông số" để chỉnh slider
4. Bấm "✨ Cải thiện ảnh"

**Slider giải thích:**
- **CLAHE clip**: tăng tương phản local (1=nhẹ, 4=mạnh)
- **Cứu vùng cháy**: kéo down highlight (cứu trời/cửa sổ blown out)
- **Nâng vùng tối**: lift shadow detail
- **Vibrance**: tăng độ tươi nhưng giữ tone da
- **Sharpen**: làm nét cuối pipeline
- **Gamma**: <1 sáng hơn, >1 tối hơn (0.95 = sáng nhẹ)

---

### 🚀 Tab "Cải thiện hàng loạt"
**Mục đích:** Cải thiện cả ngàn ảnh thô bằng 1 preset.

**Cách dùng:**
1. Chọn nhiều ảnh
2. Chọn preset
3. Bấm "🚀 Cải thiện hàng loạt" → tải file zip

**Hiệu năng (đo thật trên CPU):** ~0.8s/ảnh 1080p, ~3.3s/ảnh 4K, ~9.6s/ảnh 6K. Studio preset chậm hơn các preset khác do dùng halo-free detail enhance.

---

### 🏠 Tab "Bất động sản"
**Mục đích:** Pipeline đặc thù cho ảnh BĐS — tự nhận diện cảnh rồi áp đúng filter.

**Cách dùng:**
1. Tải ảnh
2. Chọn kiểu trời (blue/sunset/overcast/dramatic) HOẶC tải ảnh trời tuỳ chỉnh
3. Bật/tắt từng tính năng (mặc định bật hết)
4. Tinh chỉnh slider cường độ
5. Bấm "🏠 Xử lý ảnh BĐS"

**Tool tự nhận diện:**
- Ngoại thất → thay trời + tăng tươi cỏ + kéo thẳng dọc
- Nội thất → kéo sáng cửa sổ cháy + kéo thẳng dọc
- Trên cao (drone) → cả thay trời + tăng tươi cỏ

Báo cáo bên phải hiển thị: loại cảnh nhận diện, tỉ lệ trời/cỏ, độ nghiêng, các bước đã áp.
"""


_HDSD_FAQ = """
## ❓ Câu hỏi thường gặp

### 🔒 Ảnh của tôi có an toàn không?
**An toàn 100%.** Ảnh được xử lý ngay trên máy chủ này, **không gửi đi đâu khác**, không lưu vào cơ sở dữ liệu. File tạm tự dọn sau khi xử lý xong.

### 📐 Hỗ trợ ảnh tới kích thước nào?
Tới **8K** (cạnh dài tối đa 8192px). Ảnh **6K** (6000×4000) giữ **nguyên** kích thước. Chỉ ảnh lớn hơn 8K mới bị thu nhỏ.

### 🖼️ Định dạng ảnh nào được hỗ trợ?
**JPG, PNG, WebP, BMP, TIFF.** File đầu ra mặc định là JPG chất lượng cao (92/100).

### ⚠ Sao tự phát hiện không bắt được watermark?
Bộ tự phát hiện chỉ tốt cho **watermark thương mại đặt sát góc ảnh** (như Autoenhance.ai, iStock, Getty…). Nếu không bắt được:
- **Cách 1:** Vẽ mask thủ công bằng **brush đỏ** trên tab "🎨 Xoá watermark"
- **Cách 2:** Nếu có **ảnh gốc**, hãy dùng tab "🪄 Ghép từ ảnh gốc" — kết quả hoàn hảo

### ⏱ Thời gian xử lý bao lâu?
*(số đo thật trên CPU 8 luồng, ảnh JPEG)*

| Chức năng | Ảnh 4K | Ảnh 6K |
|---|---|---|
| Xoá watermark (OpenCV) | ~1s | ~2s |
| Cải thiện ảnh — preset **studio** | ~3.3s | ~9.6s |
| Cải thiện ảnh — preset khác | ~1.4s | ~3.5s |
| Ghép từ ảnh gốc | ~0.1s | ~0.2s |
| Bất động sản (full pipeline) | ~1.7s | ~4s |

> **Mẹo tốc độ:** Nếu cần xử lý ngàn ảnh 6K nhanh, chọn preset `real_estate` thay vì `studio` (gần bằng chất lượng, chạy nhanh gấp 3 lần).

### 💡 Tôi nên dùng tab nào trước?
- **Có 1 ảnh có watermark:** dùng tab **"🎨 Xoá watermark"**
- **Có cả ảnh gốc và ảnh có logo:** dùng tab **"🪄 Ghép từ ảnh gốc"** (chính xác tuyệt đối)
- **Có nhiều ảnh:** dùng các tab có chữ **"hàng loạt"**
- **Muốn cải thiện chất lượng ảnh:** dùng tab **"✨ Cải thiện ảnh"**, chọn preset **"studio"**
- **Ảnh bất động sản:** dùng tab **"🏠 Bất động sản"** — tự thay trời / kéo cửa sổ / kéo thẳng dọc
"""


def build_ui():
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Web UI cần gradio. Cài: pip install 'gradio>=4.0'") from exc

    lama_ok = _is_lama_available()
    backend_choices = ["opencv", "lama"] if lama_ok else ["opencv"]
    lama_note = (
        ""
        if lama_ok
        else (
            "\n\n> ⚠ Engine **LaMa** chưa khả dụng (chưa cài `iopaint` + `torch`). "
            "Dùng **opencv** — đủ cho 99% trường hợp watermark."
        )
    )

    # Gradio 6.x cảnh báo theme/css nên pass qua launch(), nhưng vẫn nhận
    # ở Blocks() (deprecation, không phải error). Suppress warning để console
    # sạch. Khi Gradio thực sự xoá API thì TypeError sẽ trigger fallback.
    import warnings

    blocks_kwargs = {
        "title": "Watermark Toolkit — Bộ công cụ ảnh chuyên nghiệp",
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*moved from the Blocks constructor.*",
            )
            ui_ctx = gr.Blocks(
                theme=_build_theme(gr),
                css=_CUSTOM_CSS,
                **blocks_kwargs,
            )
    except TypeError:
        ui_ctx = gr.Blocks(**blocks_kwargs)

    with ui_ctx as ui:
        ui._wm_theme = _build_theme(gr)
        ui._wm_css = _CUSTOM_CSS
        gr.Markdown(
            f"""
<div id="wm-banner">

# 🪄 Watermark Toolkit

**Xoá watermark · Ghép từ ảnh gốc · Cải thiện ảnh BĐS** — tất cả chạy ngay
trên máy bạn, ảnh KHÔNG gửi lên cloud. Hỗ trợ ảnh tới `8K`, giữ nguyên `6K`.

</div>
{lama_note}
"""
        )

        # ====== TAB 0: HDSD ======
        with gr.Tab("📖 Hướng dẫn sử dụng"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown(_HDSD_OVERVIEW)
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
<div class="guide-card">

### 💡 Mẹo nhanh
- Bắt đầu bằng tab **🎨 Xoá watermark** nếu chỉ có 1 ảnh
- Có ảnh gốc + ảnh có logo → tab **🪄 Ghép từ ảnh gốc** (tốt nhất)
- Cải thiện ảnh thô → tab **✨ Cải thiện ảnh**, chọn preset `studio`
- Ảnh BĐS → tab **🏠 Bất động sản**

</div>

<div class="warn-card">

### ⚠ Lưu ý
- Tên file 2 thư mục batch phải GIỐNG NHAU (chỉ khác extension OK)
- Auto-detect không phải lúc nào cũng đúng — kiểm tra mask trước khi xử lý hàng loạt

</div>

<div class="success-card">

### ✅ Riêng tư
- Ảnh KHÔNG gửi lên server bên thứ 3
- Mọi xử lý CPU local
- File tạm tự xoá

</div>
                        """
                    )
            with gr.Accordion("📚 Hướng dẫn chi tiết từng tab (mở để đọc)", open=False):
                gr.Markdown(_HDSD_TABS_DETAIL)
            with gr.Accordion("❓ FAQ — Câu hỏi thường gặp", open=False):
                gr.Markdown(_HDSD_FAQ)

        # TAB 1: Auto/brush (1 ảnh)
        with gr.Tab("🎨 Xoá watermark (1 ảnh)"):
            gr.Markdown(
                "Tải lên **1 ảnh** có watermark. Vẽ chuột đỏ lên vùng watermark "
                "hoặc bỏ trống để **tự phát hiện**. Phù hợp khi không có ảnh gốc."
            )
            with gr.Accordion("📖 Cách dùng", open=False):
                gr.Markdown(
                    "**3 bước:**\n"
                    "1. Tải ảnh có watermark vào ô bên trái\n"
                    "2. Chọn 1 trong 2 cách:\n"
                    "   - **Tự động**: bỏ trống mask, tool tự phát hiện logo ở 4 góc\n"
                    "   - **Vẽ tay**: chọn brush đỏ, kéo chuột lên vùng có logo\n"
                    "3. Bấm **🪄 Xoá watermark**\n\n"
                    '**Mẹo:** Tăng "Số lần phồng mask" (5–8) nếu logo có viền mờ/shadow.'
                )
            with gr.Row():
                with gr.Column():
                    editor = gr.ImageEditor(
                        label="Ảnh + vẽ mask (chuột)",
                        type="numpy",
                        layers=False,
                        brush=gr.Brush(
                            colors=["#ff0000"],
                            color_mode="fixed",
                            default_size=40,
                            default_color="#ff0000",
                        ),
                        eraser=gr.Eraser(default_size=40),
                    )
                    backend_dd = gr.Dropdown(
                        backend_choices,
                        value="opencv",
                        label="Engine xử lý",
                        info="opencv: nhanh CPU, đủ 99% case. lama: chất lượng cao nhất (cần torch).",
                    )
                    lama_model_dd = gr.Dropdown(
                        ["lama", "mat", "migan", "zits", "fcf"],
                        value="lama",
                        label="Mô hình LaMa",
                        visible=False,
                    )
                    hd_strategy_dd = gr.Dropdown(
                        ["crop", "resize", "original"],
                        value="crop",
                        label="Chiến lược ảnh HD",
                        info="crop = chỉ inpaint quanh mask (giữ chi tiết, tiết kiệm RAM).",
                    )
                    dilate_sl = gr.Slider(
                        0,
                        20,
                        value=3,
                        step=1,
                        label="Số lần phồng mask",
                        info="Mở rộng vùng quanh logo. Tăng nếu logo có viền soft.",
                    )
                    output_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra",
                        info="Chỉ thu nhỏ — không upscale. JPEG quality 95.",
                    )
                    run_btn = gr.Button("🪄 Xoá watermark", variant="primary")
                with gr.Column():
                    out_image = gr.Image(label="Ảnh kết quả", type="numpy")
                    info_box = gr.Textbox(label="Thông tin xử lý", interactive=False)

            def _toggle_lama_dropdown(backend_value: str):
                return gr.update(visible=(backend_value == "lama" and lama_ok))

            backend_dd.change(
                _toggle_lama_dropdown,
                inputs=[backend_dd],
                outputs=[lama_model_dd],
            )

            def _process_handler(
                image_dict,
                backend,
                dilate_iters,
                hd_strategy,
                lama_model,
                output_size,
                progress=gr.Progress(),
            ):
                return _process(
                    image_dict,
                    backend,
                    dilate_iters,
                    hd_strategy,
                    lama_model,
                    output_size=output_size,
                    progress=progress,
                )

            _process_handler.__name__ = "_process"

            run_btn.click(
                _process_handler,
                inputs=[
                    editor,
                    backend_dd,
                    dilate_sl,
                    hd_strategy_dd,
                    lama_model_dd,
                    output_size_dd,
                ],
                outputs=[out_image, info_box],
                api_name="_process",
            )

        # TAB 2: Composite (2 ảnh — có ảnh gốc)
        with gr.Tab("🪄 Ghép từ ảnh gốc (2 ảnh — chính xác tuyệt đối)"):
            gr.Markdown(
                "Tải lên **2 ảnh**: ảnh GỐC (chưa có logo) và ảnh ĐÃ CHỈNH có logo.\n\n"
                "Tool sẽ so sánh 2 ảnh để tìm vùng watermark, paste pixel từ ảnh "
                "gốc + Poisson blend → kết quả **hoàn hảo, không cần inpaint đoán**."
            )
            with gr.Accordion("📖 Cách dùng", open=False):
                gr.Markdown(
                    "**Khi nào dùng:** Photographer giao 2 phiên bản (raw + có watermark "
                    "preview), hoặc bạn có ảnh studio gốc + ảnh đã thêm logo brand.\n\n"
                    "**3 bước:**\n"
                    "1. Tải ảnh GỐC (chưa có logo) vào ô 1\n"
                    "2. Tải ảnh ĐÃ CÓ LOGO vào ô 2 (kích thước có thể khác — sẽ tự resize)\n"
                    "3. Bấm **🪄 Ghép từ ảnh gốc**\n\n"
                    "**Tự căn chỉnh ORB**: bật nếu 2 ảnh hơi lệch (re-crop, re-encode). "
                    "Tắt nếu chắc chắn 2 ảnh khít nhau pixel-by-pixel.\n\n"
                    "**Ngưỡng khác biệt 15** (mặc định): bắt cả watermark mờ. "
                    "Tăng lên 25–30 nếu chỉ muốn bắt watermark rõ."
                )
            with gr.Row():
                with gr.Column():
                    img_original = gr.Image(
                        label="1. Ảnh GỐC (không có logo)",
                        type="filepath",
                    )
                    img_watermarked = gr.Image(
                        label="2. Ảnh CÓ logo",
                        type="filepath",
                    )
                    align_cb = gr.Checkbox(
                        value=True,
                        label="Tự căn chỉnh bằng ORB (bật nếu 2 ảnh hơi lệch)",
                    )
                    threshold_sl = gr.Slider(
                        5,
                        50,
                        value=15,
                        step=1,
                        label="Ngưỡng khác biệt (15 = bắt cả watermark mờ)",
                        info="Pixel diff ≥ ngưỡng được coi là watermark.",
                    )
                    feather_sl = gr.Slider(
                        0,
                        15,
                        value=3,
                        step=1,
                        label="Làm mềm rìa mask (px)",
                    )
                    composite_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra",
                        info="Chỉ thu nhỏ — không upscale.",
                    )
                    composite_btn = gr.Button(
                        "🪄 Ghép từ ảnh gốc",
                        variant="primary",
                    )
                with gr.Column():
                    composite_out = gr.Image(label="Ảnh kết quả", type="numpy")
                    composite_info = gr.Textbox(label="Thông tin xử lý", interactive=False)

            def _composite_handler(
                original_path,
                watermarked_path,
                do_align,
                threshold,
                feather,
                output_size,
                progress=gr.Progress(),
            ):
                if not original_path or not watermarked_path:
                    return None, "⚠ Cần tải lên cả 2 ảnh (gốc và có logo)."
                progress(0.1, desc="Đang đọc 2 ảnh")
                import tempfile

                from .composite import composite_from_original
                from .utils import read_image, write_image

                tmp = Path(tempfile.gettempdir()) / "wm_composite_out.jpg"
                progress(0.4, desc="So sánh + căn chỉnh + Poisson blend")
                try:
                    report = composite_from_original(
                        original_path,
                        watermarked_path,
                        tmp,
                        align=do_align,
                        diff_threshold=int(threshold),
                        feather_px=int(feather),
                        quality=OUTPUT_JPEG_QUALITY,
                        keep_exif=False,
                    )
                except Exception as exc:
                    return None, f"❌ Lỗi: {type(exc).__name__}: {exc}"
                progress(0.9, desc="Đang resize đầu ra")
                result = read_image(tmp)
                result = _resize_output(result, output_size)
                # Ghi lại với chất lượng cao + size đã chọn
                write_image(tmp, result, quality=OUTPUT_JPEG_QUALITY)
                progress(0.96, desc="Đang encode kết quả")
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                info = (
                    f"Pixel khác biệt: {report.diff_pixels} ({report.mask_coverage_pct}%)  "
                    f"| Đã căn chỉnh: {'✓' if report.used_align else '✗'}  "
                    f"| Kích thước đầu ra: {result.shape[1]}×{result.shape[0]}"
                )
                return result_rgb, info

            composite_btn.click(
                _composite_handler,
                inputs=[
                    img_original,
                    img_watermarked,
                    align_cb,
                    threshold_sl,
                    feather_sl,
                    composite_size_dd,
                ],
                outputs=[composite_out, composite_info],
                api_name="composite",
            )

        # TAB 3: BATCH (nhiều cặp ảnh)
        with gr.Tab("📦 Ghép hàng loạt (nhiều cặp ảnh)"):
            gr.Markdown(
                "Tải lên **2 thư mục cùng lúc** (chọn nhiều file): thư mục ảnh GỐC và "
                "thư mục ảnh CÓ logo. Tool ghép cặp theo TÊN FILE, composite tất cả, "
                "trả về file zip.\n\n"
                "**Lưu ý**: tên file 2 thư mục phải GIỐNG NHAU "
                "(vd: `0C1A5014.jpeg` có trong cả 2 thư mục)."
            )
            with gr.Accordion("📖 Cách dùng", open=False):
                gr.Markdown(
                    "**Quy trình chuẩn cho photographer:**\n"
                    "1. Folder `originals/` chứa 100 ảnh chưa logo\n"
                    "2. Folder `previews/` chứa 100 ảnh đã thêm watermark cho khách xem\n"
                    "3. Khách trả tiền → bạn cần xoá watermark khỏi 100 ảnh\n\n"
                    "**Cách dùng:**\n"
                    "1. Mở Windows Explorer, kéo TẤT CẢ ảnh trong `originals/` "
                    'vào ô "Thư mục ảnh GỐC"\n'
                    '2. Tương tự cho `previews/` vào ô "Thư mục ảnh CÓ logo"\n'
                    "3. Bấm **📦 Xử lý hàng loạt**\n"
                    "4. Tải file zip về, giải nén → 100 ảnh sạch\n\n"
                    "**Tốc độ:** ~0.2s/cặp ảnh 4K, ~0.3s/cặp ảnh 6K. "
                    "100 cặp ảnh 6K ≈ 30 giây."
                )
            with gr.Row():
                with gr.Column():
                    files_orig = gr.Files(
                        label="1. Thư mục ảnh GỐC (chọn nhiều file)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    files_wm = gr.Files(
                        label="2. Thư mục ảnh CÓ logo (chọn nhiều file, cùng tên)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    batch_threshold = gr.Slider(
                        5,
                        50,
                        value=15,
                        step=1,
                        label="Ngưỡng khác biệt",
                    )
                    batch_align = gr.Checkbox(value=True, label="Tự căn chỉnh ORB")
                    batch_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra mỗi ảnh",
                        info="Áp dụng cho tất cả ảnh trong zip.",
                    )
                    batch_btn = gr.Button("📦 Xử lý hàng loạt", variant="primary")
                with gr.Column():
                    batch_zip = gr.File(label="Kết quả (file zip)", interactive=False)
                    batch_log = gr.Textbox(
                        label="Nhật ký xử lý",
                        lines=12,
                        interactive=False,
                    )

            def _batch_handler(
                originals,
                watermarkeds,
                threshold,
                do_align,
                output_size,
                progress=gr.Progress(),
            ):
                if not originals or not watermarkeds:
                    return None, "⚠ Cần tải lên cả 2 thư mục."
                # Map theo tên file (bỏ extension, lowercase)
                orig_map = {Path(f).stem.lower(): f for f in originals}
                wm_map = {Path(f).stem.lower(): f for f in watermarkeds}
                common = sorted(set(orig_map) & set(wm_map))
                if not common:
                    return None, (
                        f"⚠ Không ghép được cặp ảnh nào (tên file không trùng).\n"
                        f"Thư mục gốc: {len(orig_map)} ảnh, "
                        f"thư mục có logo: {len(wm_map)} ảnh."
                    )

                import tempfile
                import zipfile

                from .composite import composite_from_original
                from .utils import read_image, write_image

                tmp_dir = Path(tempfile.mkdtemp(prefix="wm_batch_"))
                logs = [f"Ghép được {len(common)} cặp ảnh:"]
                ok, fail = 0, 0
                t0 = time.perf_counter()
                for i, key in enumerate(common):
                    progress(i / len(common), desc=f"[{i + 1}/{len(common)}] {key}")
                    out_name = Path(wm_map[key]).name
                    out_path = tmp_dir / out_name
                    try:
                        report = composite_from_original(
                            orig_map[key],
                            wm_map[key],
                            out_path,
                            align=do_align,
                            diff_threshold=int(threshold),
                            feather_px=3,
                            quality=OUTPUT_JPEG_QUALITY,
                            keep_exif=False,
                        )
                        # Resize đầu ra nếu user chọn
                        if _OUTPUT_SIZE_MAP.get(output_size) is not None:
                            img_out = _resize_output(read_image(out_path), output_size)
                            write_image(out_path, img_out, quality=OUTPUT_JPEG_QUALITY)
                        ok += 1
                        logs.append(
                            f"  ✓ {key}: {report.diff_pixels}px ({report.mask_coverage_pct}%)"
                        )
                    except Exception as exc:
                        fail += 1
                        logs.append(f"  ✗ {key}: {type(exc).__name__}: {exc}")

                # Zip output
                progress(0.95, desc="Đang đóng gói file zip")
                zip_path = tmp_dir / "ket_qua.zip"
                with zipfile.ZipFile(
                    zip_path,
                    "w",
                    zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                ) as zf:
                    for f in tmp_dir.iterdir():
                        if f.suffix.lower() != ".zip":
                            zf.write(f, f.name)

                dt = time.perf_counter() - t0
                logs.append(
                    f"\nHoàn tất: thành công={ok}  thất bại={fail}  "
                    f"tổng thời gian={dt:.1f}s ({dt / max(len(common), 1):.2f}s/ảnh)"
                )
                return str(zip_path), "\n".join(logs)

            batch_btn.click(
                _batch_handler,
                inputs=[
                    files_orig,
                    files_wm,
                    batch_threshold,
                    batch_align,
                    batch_size_dd,
                ],
                outputs=[batch_zip, batch_log],
                api_name="batch_composite",
            )

        # TAB 4: BATCH INPAINT — chỉ 1 folder, auto-detect + inpaint
        with gr.Tab("⚡ Tự xoá hàng loạt (1 thư mục, không cần ảnh gốc)"):
            gr.Markdown(
                "Tải lên **1 thư mục ảnh có logo** (không cần ảnh gốc). Tool sẽ "
                "**tự phát hiện** logo ở 4 góc + inpaint, trả về file zip.\n\n"
                "Phù hợp khi chỉ có ảnh đã chỉnh, không có bản gốc. Chất lượng "
                "phụ thuộc bộ phát hiện — nếu chưa ưng ý, hãy dùng tab "
                "**Ghép từ ảnh gốc** với cặp ảnh gốc + đã chỉnh."
            )
            with gr.Accordion("📖 Cách dùng", open=False):
                gr.Markdown(
                    "**Quy trình:**\n"
                    "1. Chọn nhiều ảnh có logo (kéo thả từ Explorer)\n"
                    '2. Chỉnh "Số lần phồng mask" (3 = vừa đủ, 5–8 cho logo có viền soft)\n'
                    "3. Chọn phương pháp inpaint:\n"
                    "   - `telea`: nhanh, mịn (mặc định)\n"
                    "   - `ns`: Navier-Stokes, sắc nét hơn ở edge\n"
                    "4. Bấm **⚡ Xử lý hàng loạt** → tải file zip\n\n"
                    "**Lưu ý:** Tool chỉ phát hiện được watermark commercial sát góc "
                    "(Autoenhance.ai, iStock, Getty…). Nếu logo đặt giữa ảnh hoặc nền "
                    "phức tạp → dùng tab **🎨 Xoá watermark** vẽ tay."
                )
            with gr.Row():
                with gr.Column():
                    batch_inpaint_files = gr.Files(
                        label="Thư mục ảnh CÓ logo (chọn nhiều file)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    bi_dilate = gr.Slider(
                        0,
                        15,
                        value=3,
                        step=1,
                        label="Số lần phồng mask",
                    )
                    bi_method = gr.Dropdown(
                        ["telea", "ns"],
                        value="telea",
                        label="Phương pháp inpaint (OpenCV)",
                        info="telea: nhanh, mịn. ns: Navier-Stokes, sắc nét hơn ở edge.",
                    )
                    bi_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra mỗi ảnh",
                    )
                    bi_btn = gr.Button("⚡ Xử lý hàng loạt", variant="primary")
                with gr.Column():
                    bi_zip = gr.File(label="Kết quả (file zip)", interactive=False)
                    bi_log = gr.Textbox(
                        label="Nhật ký xử lý",
                        lines=12,
                        interactive=False,
                    )

            def _batch_inpaint_handler(
                files,
                dilate_iters,
                method,
                output_size,
                progress=gr.Progress(),
            ):
                if not files:
                    return None, "⚠ Cần tải lên ít nhất 1 ảnh."
                import tempfile
                import zipfile

                from .detect import auto_mask
                from .inpaint import InpaintBackend
                from .inpaint import inpaint as inp
                from .utils import read_image, write_image

                tmp_dir = Path(tempfile.mkdtemp(prefix="wm_batch_inp_"))
                logs = [f"Nhận {len(files)} ảnh:"]
                ok, miss = 0, 0
                t0 = time.perf_counter()

                for i, fp in enumerate(files):
                    progress(
                        i / len(files),
                        desc=f"[{i + 1}/{len(files)}] {Path(fp).name}",
                    )
                    try:
                        img = read_image(fp)
                        h, w = img.shape[:2]
                        if max(h, w) > MAX_DIMENSION:
                            scale = MAX_DIMENSION / max(h, w)
                            img = cv2.resize(
                                img,
                                (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        mask = auto_mask(
                            img,
                            strategy="logo",
                            dilate_iters=int(dilate_iters),
                        )
                        coverage = float((mask > 0).sum()) / mask.size * 100
                        if mask.sum() == 0:
                            logs.append(f"  ⊘ {Path(fp).name}: không phát hiện được logo")
                            # Vẫn copy ảnh gốc (resize nếu user chọn) để khách
                            # có đủ file đầu ra
                            out_img = _resize_output(img, output_size)
                            write_image(
                                tmp_dir / Path(fp).name,
                                out_img,
                                quality=OUTPUT_JPEG_QUALITY,
                            )
                            miss += 1
                            continue
                        result = inp(
                            img,
                            mask,
                            backend=InpaintBackend.OPENCV,
                            opencv_method=method,
                            opencv_radius=5,
                        )
                        result = _resize_output(result, output_size)
                        write_image(
                            tmp_dir / Path(fp).name,
                            result,
                            quality=OUTPUT_JPEG_QUALITY,
                        )
                        logs.append(f"  ✓ {Path(fp).name}: phủ {coverage:.2f}% diện tích")
                        ok += 1
                    except Exception as exc:
                        miss += 1
                        logs.append(f"  ✗ {Path(fp).name}: {type(exc).__name__}: {exc}")

                progress(0.97, desc="Đang đóng gói file zip")
                zip_path = tmp_dir / "ket_qua.zip"
                with zipfile.ZipFile(
                    zip_path,
                    "w",
                    zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                ) as zf:
                    for f in tmp_dir.iterdir():
                        if f.suffix.lower() != ".zip":
                            zf.write(f, f.name)

                dt = time.perf_counter() - t0
                logs.append(
                    f"\nHoàn tất: thành công={ok}  bỏ qua={miss}  "
                    f"tổng thời gian={dt:.1f}s ({dt / max(len(files), 1):.2f}s/ảnh)"
                )
                return str(zip_path), "\n".join(logs)

            bi_btn.click(
                _batch_inpaint_handler,
                inputs=[batch_inpaint_files, bi_dilate, bi_method, bi_size_dd],
                outputs=[bi_zip, bi_log],
                api_name="batch_inpaint",
            )

        # TAB 5: ENHANCE — thay thế Autoenhance.ai bằng pipeline local
        with gr.Tab("✨ Cải thiện ảnh (thay Autoenhance.ai)"):
            gr.Markdown(
                "Tải ảnh thô lên, chọn preset → nhận ảnh đã cải thiện. **Không cần** "
                "service ngoài như Autoenhance.ai. Pipeline local: cân bằng trắng + "
                "khôi phục vùng cháy + nâng vùng tối + tăng độ tươi + làm nét."
            )
            with gr.Accordion("📖 Cách dùng & ý nghĩa từng preset", open=False):
                gr.Markdown(
                    "**Cách dùng:**\n"
                    "1. Tải ảnh thô\n"
                    "2. Chọn preset phù hợp (xem bảng dưới)\n"
                    '3. (Tuỳ chọn) Mở "Tinh chỉnh thông số" để chỉnh slider\n'
                    "4. Bấm **✨ Cải thiện ảnh**\n\n"
                    "**Khi nào dùng preset nào:**\n\n"
                    "| Preset | Phù hợp | Đặc điểm |\n"
                    "|---|---|---|\n"
                    "| `studio` ⭐ | Mặc định, mọi loại ảnh | Chất lượng cao nhất: WB robust + halo-free detail enhance |\n"
                    "| `real_estate` | Ảnh BĐS interior/exterior | Cân bằng exposure, vibrance vừa phải |\n"
                    "| `portrait` | Chân dung | Giữ tone da, denoise nhẹ |\n"
                    "| `product` | Sản phẩm e-commerce | White-patch WB, sharpen mạnh |\n"
                    "| `outdoor` | Phong cảnh ngoài trời | Cứu vùng cháy + tăng tươi mạnh |\n\n"
                    "**Khuyên:** Bắt đầu bằng `studio`, nếu không vừa ý mới chuyển preset."
                )
            with gr.Row():
                with gr.Column():
                    enh_input = gr.Image(
                        label="Ảnh thô (raw)",
                        type="filepath",
                    )
                    enh_preset = gr.Dropdown(
                        ["studio", "real_estate", "portrait", "product", "outdoor"],
                        value="studio",
                        label="Preset",
                        info=(
                            "studio: chất lượng cao nhất (WB robust + halo-free). "
                            "real_estate: BĐS. portrait: chân dung. "
                            "product: sản phẩm. outdoor: ngoài trời."
                        ),
                    )
                    with gr.Accordion("⚙ Tinh chỉnh thông số (nâng cao)", open=False):
                        enh_clahe = gr.Slider(
                            0,
                            4,
                            value=1.8,
                            step=0.1,
                            label="CLAHE clip (tăng tương phản local)",
                        )
                        enh_highlight = gr.Slider(
                            0,
                            1,
                            value=0.4,
                            step=0.05,
                            label="Cứu vùng cháy (highlight recovery)",
                        )
                        enh_shadow = gr.Slider(
                            0,
                            1,
                            value=0.35,
                            step=0.05,
                            label="Nâng vùng tối (shadow lift)",
                        )
                        enh_vibrance = gr.Slider(
                            0,
                            1,
                            value=0.25,
                            step=0.05,
                            label="Tăng độ tươi (vibrance, giữ tone da)",
                        )
                        enh_sharpen = gr.Slider(
                            0,
                            1,
                            value=0.5,
                            step=0.05,
                            label="Làm nét (sharpen)",
                        )
                        enh_gamma = gr.Slider(
                            0.5,
                            1.5,
                            value=0.95,
                            step=0.05,
                            label="Gamma (<1 sáng hơn, >1 tối hơn)",
                        )
                    enh_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra",
                        info="Chỉ thu nhỏ — không upscale.",
                    )
                    enh_btn = gr.Button("✨ Cải thiện ảnh", variant="primary")
                with gr.Column():
                    enh_out = gr.Image(label="Ảnh kết quả", type="numpy")
                    enh_info = gr.Textbox(label="Thông tin xử lý", interactive=False)

            def _enhance_handler(
                input_path,
                preset_name,
                clahe,
                highlight,
                shadow,
                vibrance_v,
                sharpen,
                gamma,
                output_size,
                progress=gr.Progress(),
            ):
                if not input_path:
                    return None, "⚠ Hãy tải lên ảnh."
                progress(0.1, desc="Đang đọc ảnh")
                from .enhance import PRESETS, EnhanceParams, enhance, enhance_studio
                from .utils import read_image

                try:
                    img = read_image(input_path)
                except Exception as exc:
                    return None, f"❌ Không đọc được ảnh: {exc}"

                h, w = img.shape[:2]
                if max(h, w) > MAX_DIMENSION:
                    scale = MAX_DIMENSION / max(h, w)
                    img = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                # Merge slider overrides với preset base
                base = PRESETS[preset_name]
                params = EnhanceParams(
                    white_balance=base.white_balance,
                    clahe_clip=float(clahe),
                    clahe_tile=base.clahe_tile,
                    highlight_recovery=float(highlight),
                    shadow_lift=float(shadow),
                    vibrance=float(vibrance_v),
                    saturation_boost=base.saturation_boost,
                    unsharp_amount=float(sharpen),
                    unsharp_sigma=base.unsharp_sigma,
                    denoise_strength=base.denoise_strength,
                    gamma=float(gamma),
                )

                progress(0.5, desc="Đang chạy pipeline cải thiện")
                t0 = time.perf_counter()
                # Studio preset → đi qua enhance_studio (WB robust + halo-free)
                # để webui và CLI nhất quán.
                if preset_name == "studio":
                    result = enhance_studio(img, params)
                else:
                    result = enhance(img, params)
                dt = time.perf_counter() - t0
                progress(0.92, desc="Đang resize đầu ra")
                result = _resize_output(result, output_size)
                progress(0.96, desc="Đang encode kết quả")
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                info = (
                    f"Preset: {preset_name} | "
                    f"Kích thước đầu ra: {result.shape[1]}×{result.shape[0]} | "
                    f"Thời gian: {dt:.2f}s"
                )
                return result_rgb, info

            enh_btn.click(
                _enhance_handler,
                inputs=[
                    enh_input,
                    enh_preset,
                    enh_clahe,
                    enh_highlight,
                    enh_shadow,
                    enh_vibrance,
                    enh_sharpen,
                    enh_gamma,
                    enh_size_dd,
                ],
                outputs=[enh_out, enh_info],
                api_name="enhance",
            )

        # TAB 6: BATCH ENHANCE
        with gr.Tab("🚀 Cải thiện hàng loạt (cả ngàn ảnh)"):
            gr.Markdown(
                "Tải lên N ảnh thô, chọn preset, nhận file zip ảnh đã cải thiện. "
                "Pipeline local CPU — preset `studio` ~9.6s/ảnh 6K, các preset khác ~3.5s/ảnh 6K, **không phụ thuộc service ngoài**."
            )
            with gr.Accordion("📖 Cách dùng", open=False):
                gr.Markdown(
                    "**3 bước:**\n"
                    "1. Chọn nhiều ảnh thô (kéo thả hoặc chọn từ Explorer)\n"
                    "2. Chọn preset (`studio` khuyên dùng)\n"
                    "3. Bấm **🚀 Cải thiện hàng loạt** → tải file zip\n\n"
                    "**Tốc độ tham khảo (CPU 8 luồng, đo thật):**\n"
                    "- 100 ảnh 6K, preset `studio`: ~16 phút\n"
                    "- 100 ảnh 6K, preset `real_estate`: ~6 phút\n"
                    "- 1000 ảnh 6K, preset `real_estate`: ~1 giờ\n\n"
                    "**Mẹo:** Studio chậm gấp 3 lần preset khác do dùng halo-free "
                    "detail enhance. Nếu cần tốc độ, chọn `real_estate`.\n\n"
                    "**Ghi chú:** Không có slider tinh chỉnh ở tab này — nếu cần "
                    "tinh chỉnh, dùng tab **✨ Cải thiện ảnh** test 1 ảnh trước, "
                    "ưng rồi mới batch."
                )
            with gr.Row():
                with gr.Column():
                    be_files = gr.Files(
                        label="Ảnh thô (chọn nhiều file)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    be_preset = gr.Dropdown(
                        ["studio", "real_estate", "portrait", "product", "outdoor"],
                        value="studio",
                        label="Preset",
                        info="studio = chất lượng cao nhất. real_estate = nhanh gấp 3 lần.",
                    )
                    be_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra mỗi ảnh",
                    )
                    be_btn = gr.Button("🚀 Cải thiện hàng loạt", variant="primary")
                with gr.Column():
                    be_zip = gr.File(label="Kết quả (file zip)", interactive=False)
                    be_log = gr.Textbox(
                        label="Nhật ký xử lý",
                        lines=12,
                        interactive=False,
                    )

            def _batch_enhance_handler(
                files,
                preset_name,
                output_size,
                progress=gr.Progress(),
            ):
                if not files:
                    return None, "⚠ Cần tải lên ảnh."
                import tempfile
                import zipfile

                from .enhance import enhance_preset
                from .utils import read_image, write_image

                tmp_dir = Path(tempfile.mkdtemp(prefix="wm_batch_enh_"))
                logs = [f"Nhận {len(files)} ảnh, preset={preset_name}:"]
                ok, fail = 0, 0
                t0 = time.perf_counter()

                for i, fp in enumerate(files):
                    progress(
                        i / len(files),
                        desc=f"[{i + 1}/{len(files)}] {Path(fp).name}",
                    )
                    try:
                        img = read_image(fp)
                        h, w = img.shape[:2]
                        if max(h, w) > MAX_DIMENSION:
                            scale = MAX_DIMENSION / max(h, w)
                            img = cv2.resize(
                                img,
                                (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        result = enhance_preset(img, preset_name)
                        result = _resize_output(result, output_size)
                        write_image(
                            tmp_dir / Path(fp).name,
                            result,
                            quality=OUTPUT_JPEG_QUALITY,
                        )
                        ok += 1
                        logs.append(f"  ✓ {Path(fp).name}")
                    except Exception as exc:
                        fail += 1
                        logs.append(f"  ✗ {Path(fp).name}: {type(exc).__name__}: {exc}")

                progress(0.97, desc="Đang đóng gói file zip")
                zip_path = tmp_dir / "anh_da_cai_thien.zip"
                with zipfile.ZipFile(
                    zip_path,
                    "w",
                    zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                ) as zf:
                    for f in tmp_dir.iterdir():
                        if f.suffix.lower() != ".zip":
                            zf.write(f, f.name)
                dt = time.perf_counter() - t0
                logs.append(
                    f"\nHoàn tất: thành công={ok}  thất bại={fail}  "
                    f"tổng thời gian={dt:.1f}s ({dt / max(len(files), 1):.2f}s/ảnh)"
                )
                return str(zip_path), "\n".join(logs)

            be_btn.click(
                _batch_enhance_handler,
                inputs=[be_files, be_preset, be_size_dd],
                outputs=[be_zip, be_log],
                api_name="batch_enhance",
            )

        # ============= TAB 7: REAL ESTATE (Autoenhance.ai parity) ===========
        with gr.Tab("🏠 Bất động sản (trời / cửa sổ / cỏ / kéo thẳng)"):
            gr.Markdown(
                "**Pipeline thông minh tự nhận cảnh** — auto phát hiện "
                "nội thất / ngoại thất / aerial rồi áp lần lượt: "
                "**kéo thẳng dọc 4-point (Adobe Upright)** → thay trời "
                "(ngoại thất) → kéo sáng cửa sổ (nội thất) → tăng tươi cỏ "
                "(ngoại thất). Giữ nguyên 6K.\n\n"
                "**Cải tiến PRO-grade:**\n"
                "- Trời có **mây Perlin noise** thật, không gradient phẳng\n"
                "- Sky **tự match nhiệt độ màu** với scene (warm/cool)\n"
                "- **Cast màu trời** lên cửa kính (subtle reflection)\n"
                "- Kéo thẳng dọc bằng **4-point perspective warp** thật, không phải rotate 2D"
            )
            with gr.Accordion("📖 Cách dùng & cách tool nhận diện cảnh", open=False):
                gr.Markdown(
                    "**Cách dùng:**\n"
                    "1. Tải ảnh BĐS lên\n"
                    "2. Chọn kiểu trời mong muốn (xanh / hoàng hôn / mây / kịch tính)\n"
                    "   - HOẶC tải ảnh trời tuỳ chỉnh (sẽ ưu tiên hơn preset)\n"
                    "3. Bật/tắt từng tính năng (mặc định bật hết, tool sẽ tự bỏ qua tính "
                    "năng không phù hợp với loại cảnh)\n"
                    "4. Tinh chỉnh slider cường độ\n"
                    "5. Bấm **🏠 Xử lý ảnh BĐS**\n\n"
                    "**Cách tool nhận diện loại cảnh** (heuristic, không cần ML):\n\n"
                    "| Loại cảnh | Đặc trưng | Tool áp |\n"
                    "|---|---|---|\n"
                    "| Ngoại thất | Có trời ≥ 10%, ít edge ở phần trên | Thay trời + tăng tươi cỏ + kéo thẳng |\n"
                    "| Nội thất | Trời < 5%, nhiều edge (ceiling/đèn) | Kéo sáng cửa sổ + kéo thẳng |\n"
                    "| Trên cao (drone) | Trời ≥ 35% hoặc cỏ ≥ 30% | Thay trời + tăng tươi cỏ |\n\n"
                    "**Báo cáo bên phải** hiển thị:\n"
                    "- Loại cảnh + độ tin cậy\n"
                    "- Tỉ lệ trời / cỏ thực tế trong ảnh\n"
                    "- Độ nghiêng dọc detect được\n"
                    "- Các bước đã áp / không áp"
                )

            with gr.Row():
                with gr.Column(scale=1):
                    re_input = gr.Image(
                        type="numpy",
                        label="Ảnh thô (raw)",
                        image_mode="RGB",
                        height=320,
                    )
                    with gr.Row():
                        re_sky_preset = gr.Dropdown(
                            choices=[
                                "blue_clouds",
                                "blue_clear",
                                "golden_hour",
                                "sunset_warm",
                                "dramatic_storm",
                                "overcast_soft",
                                "twilight_blue",
                            ],
                            value="blue_clouds",
                            label="Kiểu trời",
                            info="blue_clouds: xanh có mây (phổ biến). "
                            "golden_hour: vàng dramatic. "
                            "sunset_warm: hoàng hôn cam. "
                            "dramatic_storm: mây đen luxury. "
                            "twilight_blue: chạng vạng xanh tím.",
                        )
                        re_sky_source = gr.Dropdown(
                            choices=["auto", "real_photo", "procedural"],
                            value="auto",
                            label="Nguồn trời",
                            info="auto: dùng ảnh thật nếu có, fallback procedural. "
                            "real_photo: bắt buộc dùng ảnh thật từ library. "
                            "procedural: tạo trời tổng hợp Perlin clouds.",
                        )
                        re_sky_image = gr.Image(
                            type="numpy",
                            label="Ảnh trời tự upload (override hết)",
                            image_mode="RGB",
                            height=120,
                        )
                    with gr.Row():
                        re_enable_sky = gr.Checkbox(value=True, label="Thay trời")
                        re_enable_window = gr.Checkbox(value=True, label="Kéo sáng cửa sổ")
                        re_enable_lawn = gr.Checkbox(value=True, label="Tăng tươi cỏ")
                        re_enable_vertical = gr.Checkbox(value=True, label="Kéo thẳng dọc")
                    with gr.Row():
                        re_sky_blend = gr.Slider(
                            0.3,
                            1.0,
                            0.85,
                            step=0.05,
                            label="Mức trộn trời (blend)",
                        )
                        re_window_strength = gr.Slider(
                            0.2,
                            1.0,
                            0.7,
                            step=0.05,
                            label="Cường độ kéo cửa sổ",
                        )
                        re_lawn_boost = gr.Slider(
                            0.0,
                            1.0,
                            0.5,
                            step=0.05,
                            label="Độ tươi cỏ",
                        )
                    re_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra",
                        info="Chỉ thu nhỏ — không upscale.",
                    )
                    re_brackets = gr.Files(
                        label="🌅 (Tuỳ chọn) Bracket exposures cho HDR cửa sổ",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    gr.Markdown(
                        "*Bracket exposures: nếu bạn chụp 2-5 ảnh ở các EV "
                        "khác nhau (vd −2/0/+2), upload lên đây để **HDR thật** "
                        "recover detail ngoài cửa sổ. Không có cũng OK — "
                        "tool dùng pseudo-HDR từ 1 ảnh.*"
                    )
                    re_btn = gr.Button("🏠 Xử lý ảnh BĐS", variant="primary")
                    with gr.Accordion("📥 Quản lý sky library", open=False):
                        sky_lib_info = gr.Markdown("Loading...")
                        with gr.Row():
                            sky_lib_dl_btn = gr.Button(
                                "📥 Tải bộ trời mẫu CC0 (~12 ảnh)",
                                size="sm",
                            )
                            sky_lib_refresh_btn = gr.Button(
                                "🔄 Làm mới index",
                                size="sm",
                            )

                with gr.Column(scale=1):
                    re_output = gr.Image(label="Ảnh kết quả", height=320)
                    re_report = gr.Textbox(label="Báo cáo phân tích cảnh", lines=8)

            def _realestate_handler(
                img_rgb,
                sky_preset,
                sky_source,
                sky_rgb,
                en_sky,
                en_win,
                en_lawn,
                en_vert,
                sky_blend,
                win_strength,
                lawn_boost,
                output_size,
                brackets_files,
            ):
                from .realestate import (
                    classify_scene,
                    correct_vertical,
                    enhance_realestate_full,
                    replace_sky,
                    window_pull,
                )
                from .utils import read_image

                if img_rgb is None:
                    return None, "⚠ Chưa có ảnh đầu vào."
                t0 = time.perf_counter()
                bgr = _bgr_from_rgb_or_rgba(img_rgb)
                bgr, _, scale = _maybe_downscale(bgr, None)
                sky_bgr = _bgr_from_rgb_or_rgba(sky_rgb) if sky_rgb is not None else None
                # Load brackets nếu user upload
                bracket_imgs: list = []
                if brackets_files:
                    for f in brackets_files:
                        try:
                            b = read_image(f)
                            if b.shape[:2] != bgr.shape[:2]:
                                # Resize bracket về size ảnh chính
                                b = cv2.resize(
                                    b,
                                    (bgr.shape[1], bgr.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4,
                                )
                            bracket_imgs.append(b)
                        except Exception as exc:
                            logger.warning("Bỏ qua bracket %s: %s", f, exc)

                try:
                    # Custom pipeline để có thể truyền sky_source + brackets
                    scene = classify_scene(bgr)
                    out = bgr.copy()
                    vert_report = type(
                        "V",
                        (),
                        {
                            "angle_deg": 0.0,
                            "line_count": 0,
                            "rotated": False,
                            "upright_skew": 0.0,
                            "upright_direction": "",
                        },
                    )()
                    if en_vert:
                        out, vert_report = correct_vertical(out, upright=True)

                    sky_done = False
                    if en_sky and scene.tag in ("exterior", "aerial") and scene.sky_ratio > 0.05:
                        out, _ = replace_sky(
                            out,
                            preset=sky_preset,
                            sky_image=sky_bgr,
                            sky_source=sky_source,
                            blend_strength=float(sky_blend),
                            feather=21,
                        )
                        sky_done = True

                    win_done = False
                    if en_win and scene.tag == "interior":
                        out, mask_w = window_pull(
                            out,
                            strength=float(win_strength),
                            brackets=bracket_imgs if bracket_imgs else None,
                            use_hdr=True,
                        )
                        win_done = mask_w.sum() > 0

                    lawn_done = False
                    if en_lawn and scene.tag in ("exterior", "aerial"):
                        out, mask_l = enhance_realestate_full.__globals__["enhance_lawn"](
                            out, sat_boost=float(lawn_boost)
                        )
                        lawn_done = mask_l.sum() > 0

                    # Wrap report
                    class _R:
                        pass

                    report = _R()
                    report.scene = scene
                    report.vertical = vert_report
                    report.sky_replaced = sky_done
                    report.windows_recovered = win_done
                    report.lawn_enhanced = lawn_done
                except Exception as exc:
                    logger.exception("realestate handler fail")
                    return None, f"❌ Lỗi: {type(exc).__name__}: {exc}"
                # Resize đầu ra
                out = _resize_output(out, output_size)
                dt = time.perf_counter() - t0
                tag_vi = {
                    "interior": "Nội thất",
                    "exterior": "Ngoại thất",
                    "aerial": "Trên cao (drone)",
                    "unknown": "Không xác định",
                }.get(report.scene.tag, report.scene.tag)

                # Sky source info
                sky_src_info = ""
                if sky_done:
                    sky_src_info = f" (nguồn: {sky_source})"
                # HDR info
                hdr_info = ""
                if report.windows_recovered and bracket_imgs:
                    hdr_info = f" — HDR Mertens với {len(bracket_imgs)} bracket"
                elif report.windows_recovered:
                    hdr_info = " — pseudo-HDR (single image)"
                # Vertical info (upright vs rotate)
                if hasattr(report.vertical, "upright_skew") and report.vertical.upright_skew > 0:
                    vert_info = (
                        f"perspective warp skew={report.vertical.upright_skew:.3f} "
                        f"direction={report.vertical.upright_direction}"
                    )
                else:
                    vert_info = f"rotate 2D {report.vertical.angle_deg:+.2f}°"

                lines = [
                    f"⏱  Thời gian: {dt:.2f}s  (tỉ lệ resize: {scale:.2f})",
                    f"🏷  Loại cảnh: {tag_vi} (độ tin cậy {report.scene.confidence:.2f})",
                    f"🌤  Tỉ lệ trời: {report.scene.sky_ratio:.1%}",
                    f"🌱 Tỉ lệ cỏ: {report.scene.grass_ratio:.1%}",
                    f"💡 Độ sáng trung bình: {report.scene.avg_brightness:.2f}",
                    f"📐 Sửa nghiêng: {vert_info} "
                    f"(số đường: {report.vertical.line_count}, "
                    f"đã áp: {'✓' if report.vertical.rotated else '✗'})",
                    f"☁️  Đã thay trời: {'✓' if report.sky_replaced else '✗'}{sky_src_info}",
                    f"🪟 Đã kéo sáng cửa sổ: {'✓' if report.windows_recovered else '✗'}{hdr_info}",
                    f"🌿 Đã tăng tươi cỏ: {'✓' if report.lawn_enhanced else '✗'}",
                ]
                return _rgb_from_bgr(out), "\n".join(lines)

            re_btn.click(
                _realestate_handler,
                inputs=[
                    re_input,
                    re_sky_preset,
                    re_sky_source,
                    re_sky_image,
                    re_enable_sky,
                    re_enable_window,
                    re_enable_lawn,
                    re_enable_vertical,
                    re_sky_blend,
                    re_window_strength,
                    re_lawn_boost,
                    re_size_dd,
                    re_brackets,
                ],
                outputs=[re_output, re_report],
                api_name="realestate",
            )

            # Sky library management
            def _sky_lib_status() -> str:
                from .sky_assets import stats

                s = stats()
                lines = [
                    f"📂 **Library**: `{s['library_dir']}`",
                    f"📊 **Tổng**: {s['total_count']} ảnh",
                ]
                if s["by_category"]:
                    lines.append("**Theo category**:")
                    for cat, count in sorted(s["by_category"].items()):
                        lines.append(f"- {cat}: {count}")
                else:
                    lines.append("*(Trống — bấm 'Tải bộ trời mẫu' hoặc copy ảnh vào folder trên)*")
                return "\n".join(lines)

            def _download_samples_handler():
                from .sky_assets import download_samples

                results = download_samples(timeout=20)
                total = sum(results.values())
                return _sky_lib_status() + (f"\n\n✅ Đã tải {total} ảnh trời CC0 từ Pexels.")

            def _refresh_index_handler():
                from .sky_assets import refresh_index

                refresh_index()
                return _sky_lib_status() + "\n\n🔄 Đã làm mới index."

            ui.load(_sky_lib_status, outputs=sky_lib_info)
            sky_lib_dl_btn.click(_download_samples_handler, outputs=sky_lib_info)
            sky_lib_refresh_btn.click(_refresh_index_handler, outputs=sky_lib_info)

        # ============= TAB 8: PRO DEVELOP (Lightroom-style) ===========
        with gr.Tab("🎚 Pro Develop (Lightroom-style)"):
            gr.Markdown(
                "**Pipeline pro-grade** cho photographer 20 năm. "
                "Tone curve 4-region, WB Kelvin/Tint, Texture/Clarity/Dehaze, "
                "Local adjustments (radial + graduated), Lens correction từ "
                "10+ profile có sẵn. **Tất cả slider Lightroom Develop** trong "
                "1 tab."
            )
            with gr.Accordion("📖 Cách dùng & thứ tự pro chuẩn", open=False):
                gr.Markdown(
                    "**Workflow pro chuẩn:**\n"
                    "1. **Lens correction** trước (sửa distortion/vignetting/CA)\n"
                    "2. **WB** (temp/tint hoặc eyedropper)\n"
                    "3. **Tone curve** (exposure → highlights/shadows → "
                    "contrast → blacks/whites)\n"
                    "4. **Dehaze** (atmospheric clarity nếu cần)\n"
                    "5. **Texture/Clarity** (mid-freq detail)\n"
                    "6. **Local adjustments** (vignette, darken sky, "
                    "graduated filter)\n\n"
                    "**Lưu ý quan trọng:**\n"
                    "- Lens correction tốt nhất có EXIF → tự nhận lens\n"
                    "- Nếu không có EXIF, chọn profile thủ công khớp lens"
                )

            with gr.Row():
                with gr.Column(scale=1):
                    pd_input = gr.Image(
                        type="filepath",
                        label="Ảnh đầu vào",
                    )

                    with gr.Tab("📸 Lens"):
                        pd_lens_profile = gr.Dropdown(
                            choices=["(không áp lens correction)"]
                            + [
                                f"{p['id']} — {p['name']}"
                                for p in __import__(
                                    "pps_core.lens",
                                    fromlist=["list_lens_profiles"],
                                ).list_lens_profiles()
                            ],
                            value="(không áp lens correction)",
                            label="Lens profile (tự áp distortion + vignette + CA)",
                            info="Auto-detect từ EXIF nếu chọn 'auto-exif'.",
                        )
                        pd_lens_intensity = gr.Slider(
                            0,
                            1,
                            1.0,
                            step=0.05,
                            label="Cường độ correction",
                        )
                        pd_lens_ca = gr.Checkbox(
                            value=True,
                            label="Sửa chromatic aberration (subtle)",
                        )

                    with gr.Tab("🌡 White Balance"):
                        pd_temp = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Temperature (-1 cool / +1 warm)",
                            info="Tương đương Kelvin slider Lightroom",
                        )
                        pd_tint = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Tint (-1 green / +1 magenta)",
                        )

                    with gr.Tab("🎚 Tone Curve"):
                        pd_exposure = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Exposure (±1 EV)",
                        )
                        pd_contrast = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Contrast (S-curve strength)",
                        )
                        pd_highlights = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Highlights",
                        )
                        pd_shadows = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Shadows",
                        )
                        pd_whites = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Whites",
                        )
                        pd_blacks = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Blacks",
                        )

                    with gr.Tab("✨ Detail"):
                        pd_texture = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Texture (mid-freq detail)",
                        )
                        pd_clarity = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Clarity (local midtone contrast)",
                        )
                        pd_dehaze = gr.Slider(
                            0,
                            1,
                            0.0,
                            step=0.05,
                            label="Dehaze (atmospheric)",
                        )

                    with gr.Tab("🎯 Local"):
                        pd_vignette_amt = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Vignette (-1 dark rìa / +1 bright)",
                        )
                        pd_vignette_mid = gr.Slider(
                            0,
                            1,
                            0.5,
                            step=0.05,
                            label="Vignette midpoint",
                        )
                        gr.Markdown("**Graduated filter — Sky**")
                        pd_grad_y = gr.Slider(
                            0,
                            1,
                            0.45,
                            step=0.05,
                            label="Horizon position (0=top, 1=bottom)",
                        )
                        pd_grad_amt = gr.Slider(
                            -1,
                            1,
                            0.0,
                            step=0.05,
                            label="Graduated darken amount (sky)",
                        )

                    pd_size_dd = gr.Dropdown(
                        _OUTPUT_SIZE_LABELS,
                        value=_OUTPUT_SIZE_LABELS[0],
                        label="Kích thước đầu ra",
                    )
                    pd_run_btn = gr.Button(
                        "🎚 Process Pro Develop",
                        variant="primary",
                    )

                with gr.Column(scale=1):
                    pd_output = gr.Image(label="Kết quả", type="numpy")
                    pd_info = gr.Textbox(
                        label="Thông tin xử lý",
                        lines=8,
                        interactive=False,
                    )

            def _pro_develop_handler(
                input_path,
                lens_profile_str,
                lens_intensity,
                lens_ca,
                temp,
                tint,
                exposure,
                contrast,
                highlights,
                shadows,
                whites,
                blacks,
                texture_amt,
                clarity_amt,
                dehaze_amt,
                vign_amt,
                vign_mid,
                grad_y,
                grad_amt,
                output_size,
                progress=gr.Progress(),
            ):
                if not input_path:
                    return None, "⚠ Hãy upload ảnh."
                from .lens import auto_correct_lens
                from .local_adjust import darken_sky_grad
                from .local_adjust import vignette as vignette_fn
                from .tone import (
                    ToneParams,
                    apply_tone_full,
                    temperature_tint,
                )
                from .utils import read_image

                progress(0.05, desc="Đọc ảnh")
                try:
                    img = read_image(input_path)
                except Exception as exc:
                    return None, f"❌ Không đọc được ảnh: {exc}"

                # Resize quá to
                h, w = img.shape[:2]
                if max(h, w) > MAX_DIMENSION:
                    sc = MAX_DIMENSION / max(h, w)
                    img = cv2.resize(
                        img,
                        (int(w * sc), int(h * sc)),
                        interpolation=cv2.INTER_AREA,
                    )

                t0 = time.perf_counter()
                steps = []

                # 1. Lens correction
                progress(0.15, desc="Lens correction")
                if lens_profile_str and not lens_profile_str.startswith("("):
                    profile_id = lens_profile_str.split(" — ")[0]
                    img, lens_info = auto_correct_lens(
                        img,
                        profile_id=profile_id,
                        intensity=float(lens_intensity),
                        correct_chromatic=bool(lens_ca),
                    )
                    if lens_info.get("applied"):
                        steps.append(f"📸 Lens: {lens_info['profile_name']}")

                # 2. WB
                progress(0.30, desc="White balance")
                if abs(temp) > 1e-3 or abs(tint) > 1e-3:
                    img = temperature_tint(img, temp_shift=float(temp), tint_shift=float(tint))
                    steps.append(f"🌡 WB: temp={temp:+.2f} tint={tint:+.2f}")

                # 3. Tone + Detail
                progress(0.50, desc="Tone curve + Detail")
                tone_params = ToneParams(
                    temp_shift=0.0,
                    tint_shift=0.0,  # đã áp ở step 2
                    exposure=float(exposure),
                    contrast=float(contrast),
                    highlights=float(highlights),
                    shadows=float(shadows),
                    lights=float(highlights) * 0.5,
                    darks=float(shadows) * 0.5,
                    whites=float(whites),
                    blacks=float(blacks),
                    texture=float(texture_amt),
                    clarity=float(clarity_amt),
                    dehaze=float(dehaze_amt),
                )
                img = apply_tone_full(img, tone_params)
                if any(
                    abs(v) > 1e-3 for v in (exposure, contrast, highlights, shadows, whites, blacks)
                ):
                    steps.append("🎚 Tone curve áp dụng")
                if any(abs(v) > 1e-3 for v in (texture_amt, clarity_amt, dehaze_amt)):
                    steps.append(
                        f"✨ Detail: tex={texture_amt:+.2f} "
                        f"clar={clarity_amt:+.2f} dehaze={dehaze_amt:.2f}"
                    )

                # 4. Local — Graduated darken sky
                progress(0.75, desc="Local adjustments")
                if abs(grad_amt) > 1e-3:
                    img = darken_sky_grad(
                        img,
                        horizon_y_frac=float(grad_y),
                        amount=float(grad_amt),
                        feather=0.4,
                    )
                    steps.append(f"🎯 Graduated sky: y={grad_y:.2f} amt={grad_amt:+.2f}")

                # 5. Vignette (cuối cùng theo thứ tự pro)
                if abs(vign_amt) > 1e-3:
                    img = vignette_fn(
                        img,
                        amount=float(vign_amt),
                        midpoint=float(vign_mid),
                        feather=0.6,
                    )
                    steps.append(f"🎯 Vignette: amt={vign_amt:+.2f} mid={vign_mid:.2f}")

                # 6. Output resize
                progress(0.92, desc="Resize đầu ra")
                img = _resize_output(img, output_size)

                progress(0.97, desc="Encode")
                dt = time.perf_counter() - t0
                info = (
                    (
                        f"⏱ Thời gian: {dt:.2f}s\n"
                        f"📐 Kích thước đầu ra: {img.shape[1]}×{img.shape[0]}\n"
                        f"📋 Bước đã áp:\n"
                    )
                    + "\n".join(f"  {s}" for s in steps)
                    if steps
                    else (
                        f"⏱ {dt:.2f}s | Kích thước: "
                        f"{img.shape[1]}×{img.shape[0]} | "
                        "(không có chỉnh sửa nào áp dụng)"
                    )
                )
                return _rgb_from_bgr(img), info

            pd_run_btn.click(
                _pro_develop_handler,
                inputs=[
                    pd_input,
                    pd_lens_profile,
                    pd_lens_intensity,
                    pd_lens_ca,
                    pd_temp,
                    pd_tint,
                    pd_exposure,
                    pd_contrast,
                    pd_highlights,
                    pd_shadows,
                    pd_whites,
                    pd_blacks,
                    pd_texture,
                    pd_clarity,
                    pd_dehaze,
                    pd_vignette_amt,
                    pd_vignette_mid,
                    pd_grad_y,
                    pd_grad_amt,
                    pd_size_dd,
                ],
                outputs=[pd_output, pd_info],
                api_name="pro_develop",
            )

    return ui


def serve(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    ui = build_ui()
    # concurrency=3 đủ cho UI + 1-2 request song song mà không OOM.
    ui.queue(default_concurrency_limit=3, max_size=20)
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "max_file_size": "200mb",  # đủ cho ảnh 6K-8K PNG (raw ~100MB)
    }
    # Theme + CSS pass qua launch() (Gradio 6 API)
    if hasattr(ui, "_wm_theme"):
        launch_kwargs["theme"] = ui._wm_theme
    if hasattr(ui, "_wm_css"):
        launch_kwargs["css"] = ui._wm_css
    try:
        ui.launch(**launch_kwargs)
    except TypeError:
        # Gradio < 6: theme/css không nhận trên launch — fallback về Blocks
        launch_kwargs.pop("theme", None)
        launch_kwargs.pop("css", None)
        ui.launch(**launch_kwargs)


if __name__ == "__main__":
    serve()
