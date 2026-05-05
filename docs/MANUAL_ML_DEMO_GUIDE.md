# Manual ML Demo Guide — Cho Pitch Deck

**Mục tiêu:** trong 60 phút, có 3 cặp before/after thật cho 3 feature
"net-new" (virtual staging, multi-angle, instruction edit) trên 2 hero photo
để show NĐT — không phải mockup, không phải PPT.

**Tại sao manual:** Phase 3 (1.5 tuần) sẽ tự động hoá thành pipeline. Nhưng
để pitch HÔM NAY, chạy thủ công trên Colab dùng các notebook đã có sẵn
trong Drive folder của mày.

**Hardware cần:** Colab Free (T4 GPU 15GB) là đủ cho Qwen-Image-Lightning.
SUPIR và SD3.5 cần Colab Pro (A100 hoặc V100, $10/tháng) hoặc HuggingFace
Inference Provider ($0.001/image).

---

## Hero photo recommendation

Top 3 ảnh đáng làm hero (đã chấm điểm):

| ID | Lý do |
|---|---|
| **0C1A5044** | Kitchen có cửa sổ lớn nhìn ra deck/garden — virtual staging + multi-angle dễ thấy |
| **0C1A5029** | Living room — instruction edit "brighten room" sẽ rõ |
| **0C1A5023** | Bedroom với window pull cháy — SUPIR upscale highlight detail |

→ Pick 2 trong 3 này.

---

## Demo 1 · Virtual Staging (phòng trống → có nội thất)

**Tool:** Stable Diffusion 3.5 + IPAdapter (hoặc Replicate API).

**Bước:**
1. Mở Colab notebook mới · runtime: T4 GPU
2. Chạy:
   ```python
   !pip install -q diffusers transformers accelerate
   import torch
   from diffusers import StableDiffusion3Img2ImgPipeline
   pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
       "stabilityai/stable-diffusion-3.5-medium",
       torch_dtype=torch.bfloat16
   ).to("cuda")
   ```
3. Upload `0C1A5044.jpeg` (kitchen trống) lên Colab
4. Run inpaint với prompt:
   ```
   "modern Scandinavian kitchen with light wood island, three bar stools,
   pendant lights, marble countertop, fully furnished, real estate
   photography, professional"
   ```
   strength=0.55, guidance_scale=7.5, 30 steps
5. Download output → save `feature_demos/virtual_staging/5044_after_furnished.jpg`
6. Save original → `feature_demos/virtual_staging/5044_before_empty.jpg`

**Kết quả expected:** kitchen trống → kitchen có bàn đảo, ghế bar, đèn pendant.
Đối thủ KHÔNG có feature này.

**Thời gian:** 5–8 phút trên Colab T4.

---

## Demo 2 · Multi-angle Synthesis (1 ảnh → N góc)

**Tool:** Notebook `Multiple-angles.ipynb` đã có sẵn trong Drive folder của mày.

Notebook dùng model `dx8152/Qwen-Edit-2509-Multiple-angles` (HuggingFace public).

**Bước:**
1. Mở `Multiple-angles.ipynb` trong Colab
2. Chạy 2 cell đầu (install + load pipe)
3. Upload `0C1A5044.jpeg`
4. Run với 3 prompt khác nhau:
   - `"same room from left side angle"` → save `5044_angle_left.jpg`
   - `"same room from right side angle"` → save `5044_angle_right.jpg`
   - `"bird's eye view of same room"` → save `5044_angle_top.jpg`
5. Save vào `feature_demos/multi_angle/`

**Kết quả expected:** 3 góc nhìn khác nhau của cùng kitchen, consistent
furniture/lighting. Cực kỳ có giá trị cho RE listing — đối thủ chỉ cho 1 góc duy nhất.

**Thời gian:** 10–15 phút (3 inference × ~3 phút).

---

## Demo 3 · Instruction Editing (NLP edit)

**Tool:** Notebook `Bản sao của Qwen-Image-Lightning.ipynb` (Drive).

Model: `lightx2v/Qwen-Image-Lightning` (HF public, LoRA distilled cho nhanh).

**Bước:**
1. Mở notebook trong Colab
2. Chạy install + load
3. Upload `0C1A5029.jpeg`
4. Run 3 instruction edit:
   - `"brighten the kitchen and add warm afternoon sunlight through windows"`
   - `"remove the photographer's reflection in the mirror"`
   - `"replace the gray sky outside with a clear blue sky"`
5. Save vào `feature_demos/instruction_edit/5029_<task>.jpg`

**Kết quả expected:** 3 edit ngôn ngữ tự nhiên — NĐT thấy ngay UX khác
biệt vs vendor (toggle 20 cái → 1 câu lệnh).

**Thời gian:** 6–10 phút (3 inference × ~2-3 phút trên T4).

---

## Demo 4 · SUPIR Upscale 4K (optional, cần Colab Pro)

**Tool:** SUPIR repository (https://github.com/Fanghua-Yu/SUPIR).

SUPIR là SOTA upscale 2025 — phục hồi detail tốt hơn Real-ESRGAN nhiều.

**Bước:**
1. Clone SUPIR trong Colab Pro (A100):
   ```bash
   !git clone https://github.com/Fanghua-Yu/SUPIR.git
   !cd SUPIR && pip install -r requirements.txt
   ```
2. Download checkpoint (SUPIR-v0Q ~6GB):
   ```bash
   !wget https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt
   ```
3. Run trên `0C1A5023.jpeg`:
   ```bash
   !python test.py --img 0C1A5023.jpeg --SUPIR_sign Q --upscale 2
   ```
4. So sánh detail crop (window frame, carpet texture) với original
5. Save → `feature_demos/upscale_supir/5023_2x.jpg`

**Thời gian:** 30–60 giây/ảnh trên A100.

---

## Tổng hợp output cho pitch

Sau khi xong 4 demo:
```
feature_demos/
  virtual_staging/
    5044_before_empty.jpg
    5044_after_furnished.jpg
  multi_angle/
    5044_original.jpg
    5044_angle_left.jpg
    5044_angle_right.jpg
    5044_angle_top.jpg
  instruction_edit/
    5029_brighten_kitchen.jpg
    5029_no_reflection.jpg
    5029_blue_sky.jpg
  upscale_supir/
    5023_original.jpg
    5023_2x_supir.jpg
```

Chạy `examples/build_feature_demo_composites.py` (sẽ viết ở Phase 3) để
tạo composite slide-ready với label.

Hoặc thủ công ghép trong Figma/Keynote — dán pair before/after.

---

## Slide structure đề xuất

| Slide | Nội dung |
|---|---|
| 1 | Vấn đề: photographer làm thủ công, tốn 30 phút/listing |
| 2 | Vendor hiện tại: AutoEnhance + Manuka làm color basic, $1-3/ảnh |
| 3 | **Demo 1 — Virtual Staging** before/after (5 phút staging time bằng software) |
| 4 | **Demo 2 — Multi-angle** (1 ảnh → 3 góc, listing 3x conversion) |
| 5 | **Demo 3 — Instruction edit** (UX 1 câu vs 20 toggle) |
| 6 | Feature matrix table (FEATURE_MATRIX_FOR_INVESTORS.md) |
| 7 | Roadmap: Phase 3 = automate trong 1.5 tuần (cần GPU credit) |
| 8 | Ask: $1.5K/tháng GPU + 10 beta customer + 10 tuần runway |
| 9 | Team + Q&A |

→ Slides 3–5 là chỗ NĐT "wow". Tất cả còn lại là support cho 3 slide đó.

---

## Sau khi user chạy xong

Commit output vào repo:
```bash
cd pro-photo-studio
mkdir -p investor_showcase/feature_demos
# Copy output từ Colab download → investor_showcase/feature_demos/...
git add investor_showcase/feature_demos/
git commit -m "demo: manual ML output for pitch (5044, 5029, 5023)"
git push
```

Báo tao biết khi xong → tao build composite slide-ready + update INVESTOR_BRIEF.
