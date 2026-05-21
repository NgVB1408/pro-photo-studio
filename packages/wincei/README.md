# pps-wincei v0.2 — AI Window + Ceiling Fixer (Pro)

> Tool tách từ **Pro Photo Studio**, chuyên fix **2 vấn đề cứng đầu nhất** của ảnh BĐS:
> 🪟 **Cửa sổ cháy trắng** → Reinhard local tonemap + guided filter + chroma recovery
> 🏠 **Trần ám màu** → Bradford CAT chromatic adaptation về D65 + local luminance equalize

## ✨ Khác v0.1

| Phần | v0.1 | **v0.2** |
|------|------|----------|
| Detector | rembg saliency suy luận | **SegFormer ADE20K class-id** (window=8, ceiling=5) |
| Window fix | HSV V compression | **Reinhard local tonemap + guided filter** (linear-space) |
| Ceiling fix | LAB shift cứng | **Bradford CAT (XYZ space) + local lum equalize** |
| GPU | Không | **CUDA / DirectML auto-dispatch, fp16** |
| Self-eval | Không | **7-metric scorer + verdict pass/review/fail + recommendations** |
| Pipeline | 1 ảnh / model load | **Share 1 segmentation inference cho cả 2 mask** |

---

## 🚀 Cài đặt

### CPU (mọi máy)
```bash
pip install pps-wincei
```

### GPU NVIDIA (CUDA 11.8/12.1, nhanh hơn 5-15x)
```bash
# Trước: cài torch với CUDA matching driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Sau: cài pps-wincei
pip install "pps-wincei[gpu]"
```

### Kiểm tra GPU detect
```bash
pps-wincei --version  # in ra banner GPU/CPU lúc đầu mỗi run
```

---

## 📸 CLI

### Cơ bản
```bash
pps-wincei nha-cua.jpg
# → nha-cua.wincei.jpg
```

### Đầy đủ
```bash
pps-wincei nha-cua.jpg \
  --out fixed.jpg \
  --debug \
  --debug-dir ./out \
  --window 1.2 \
  --ceiling 0.9 \
  --include-lamps
```

### Batch
```bash
pps-wincei *.jpg --out-dir ./fixed --debug
```

### 🌐 Folder mode + HTML comparison viewer (realtyedit.app style)
Workflow chuyên nghiệp cho khách BĐS:
```bash
# Mày drop 100 ảnh BĐS vào ./inputs/
pps-wincei-folder --inputs ./inputs --outputs ./outputs

# Output:
#   ./outputs/<tất cả ảnh đã fix>
#   ./outputs/comparison.html   ← Mở browser, slider before/after như realtyedit.app
```

**Tính năng viewer:**
- 🖱️ Slider before/after (kéo / hover)
- ⬛⬜ Side-by-side mode
- 🔍 Filter theo verdict (pass / review / fail / scope_violation)
- 📊 Mỗi card hiện: context (room, lighting, mood), decisions của AI, scope ΔE, score
- 💾 1 file HTML duy nhất, share zip cho khách qua Zalo

**Quality preservation:**
- JPG → JPG **quality 98, 4:4:4 chroma**, EXIF + ICC preserved
- PNG → PNG optimize, ICC preserved
- TIFF → 16-bit LZW (nếu input 16-bit), EXIF preserved
- Resolution input 4608×3072 → output **giữ nguyên 4608×3072**
- Camera metadata (Sony A7M4, lens, ISO, shutter) → preserved

### Output báo cáo (mỗi ảnh)
```
📷 Input  : nha-cua.jpg
💾 Output : nha-cua.wincei.jpg
📐 Size   : 4096×2731
⏱️  Time   : 2.34s  (infer 320ms)
🚀 GPU: NVIDIA RTX 3060 (12.0GB VRAM) | ONNX providers: CUDAExecutionProvider, CPUExecutionProvider

🪟 WINDOW (mask 8.2%)  applied=True  clipped 18.4%→1.1%
🏠 CEILING (mask 12.4%)  applied=True  cast 14.3→1.8

🤖 SELF-EVAL: PASS  score=0.892/1.000
   ✓ window_highlight_recovery: clip 18.4% → 1.1% (giảm 94%)
   ✓ ceiling_neutrality_lab: |ΔA|+|ΔB|=1.8
   ✓ global_consistency_psnr: PSNR untouched=46.2dB
   ✓ edge_preservation_ssim: SSIM biên=0.943
   ✓ natural_look_chroma_balance: saturation OK (mean=0.31)
```

---

## 🤖 AI tự đánh giá output (Self-Eval)

7 metrics + verdict:

| Metric | Trọng số | Đo gì |
|--------|---------|-------|
| `window_highlight_recovery` | 20% | Vùng blown >0.95 luma giảm bao nhiêu % |
| `window_no_halo` | 10% | Edge overshoot p95 biên cửa sổ |
| `ceiling_neutrality_lab` | 20% | `|ΔA|+|ΔB|` LAB trong vùng ceiling |
| `ceiling_luminance_uniform` | 10% | std(L*) trong ceiling region |
| `global_consistency_psnr` | 20% | PSNR vùng KHÔNG mask (không bị bleed) |
| `edge_preservation_ssim` | 10% | SSIM biên mask (không smudge sang wall) |
| `natural_look_chroma_balance` | 10% | mean saturation toàn ảnh |

**Verdict:**
- `pass` ≥ 0.85 → giao luôn cho khách
- `review` 0.65-0.84 → đọc recommendations + chạy lại với param điều chỉnh
- `fail` < 0.65 → mask sai hoặc ảnh quá khó, dùng manual hoặc tăng strength

---

## 🐍 Python API

```python
from pps_wincei import process_image

result = process_image(
    "input.jpg",
    "output.jpg",
    debug_dir="./debug",        # mask + overlay + scorecard.json
    window_strength=1.0,        # 0..1.5+
    ceiling_strength=0.85,      # 0..1
    include_lamps=False,        # gộp đèn vào ceiling mask
    expand_window_with_sky=True,
    self_evaluate=True,
)

print(result.report())
print(result.evaluation["verdict"])      # "pass" / "review" / "fail"
print(result.evaluation["overall_score"])  # 0..1
for rec in result.evaluation["recommendations"]:
    print("👉", rec)
```

Hoặc chạy individual stages:

```python
import cv2
from pps_wincei import segment, detect_window_mask, detect_ceiling_mask
from pps_wincei import fix_window_highlights, fix_ceiling_neutrality, evaluate

img = cv2.imread("input.jpg")
seg = segment(img)
win = detect_window_mask(img, seg=seg)
ceil = detect_ceiling_mask(img, seg=seg)

step1, _ = fix_window_highlights(img, win, strength=1.0)
step2, _ = fix_ceiling_neutrality(step1, ceil, strength=0.85)

eval = evaluate(before=img, after=step2, window_mask=win, ceiling_mask=ceil, seg=seg)
print(eval.verdict, eval.overall_score)
```

---

## ⚙️ Pipeline nội bộ

```
Input ảnh BĐS (JPG / PNG / TIFF)
    │
    ├─► SegFormer ADE20K (1 inference call, GPU nếu có)
    │       151 classes — chọn:
    │       ├─ window (id=8) → window mask
    │       ├─ sky (id=2) → union với window (optional)
    │       ├─ ceiling (id=5) → ceiling mask
    │       └─ lamp/light (36/82) → union ceiling (optional)
    │
    ├─► WINDOW FIX (Reinhard local tonemap)
    │       1. sRGB → linear
    │       2. Luminance = Rec.709 weights
    │       3. Reinhard global+local operator → tm_lum
    │       4. Channel-ratio reapply → preserve chroma
    │       5. HSV saturation boost (compensate desaturation)
    │       6. Guided filter mask blend (edge-aware, no halo)
    │       7. Linear → sRGB → uint8
    │
    ├─► CEILING FIX (Bradford CAT)
    │       1. sRGB → linear → XYZ
    │       2. Estimate src illuminant từ bright ceiling pixels (top 25%)
    │       3. Bradford CAT matrix toward D65, blend by `strength`
    │       4. Apply CAT in XYZ
    │       5. Local luminance equalize (Gaussian σ=40, cap 0.85-1.18)
    │       6. Guided filter mask blend
    │       7. XYZ → linear → sRGB → uint8
    │
    └─► SELF-EVAL (7 metric scorers)
            → verdict + recommendations
```

---

## ❓ FAQ

### Cần GPU không?
**Không** — chạy được CPU. Nhưng GPU nhanh hơn 5-15x. Với GTX 750 Ti 2GB+ là đủ cho SegFormer-B0.

### AI có yêu cầu mạng không?
Lần đầu: có (tải SegFormer ~85-240MB tuỳ model size). Sau đó: không.

### Output có khác format input không?
Không. JPG→JPG 95, PNG→PNG, TIFF→TIFF (LZW). Resolution giữ nguyên 100%.

### AI detect sai window/ceiling?
1. Chạy `--debug` xem mask + overlay
2. Đổi parameter:
   - Mask thiếu cửa sổ → giữ `--no-sky-as-window` mặc định OFF (sky thường visible qua window)
   - Mask thiếu đèn trần → thêm `--include-lamps`
   - Mask tràn ra wall → giảm strength hoặc dùng GPU + model lớn hơn
3. Trên GPU cao (>6GB VRAM) sẽ tự upgrade SegFormer-B3 → mask chính xác hơn

### Self-eval cho ra `review` — có nên giao khách không?
Đọc `recommendations` — thường là chỉnh `--window` hoặc `--ceiling` strength rồi chạy lại. Nếu vẫn `review` sau 2-3 lần thử, ảnh đó khó, manual touch-up.

---

## 🐛 Báo lỗi & hỗ trợ

- GitHub: https://github.com/NgVB1408/pro-photo-studio/issues
- Zalo: 0876 254 585
- Email: luvdraco88@gmail.com

## 📜 License

Apache-2.0
