# Investor Showcase — TLDR

**Đọc trước:**

Color-enhance head-to-head với AutoEnhance.ai và Manuka **không phải chỗ
chúng tôi thắng**. 3 vendor này (gồm cả PPS v2 OpenCV pipeline) đều produce
output comparable cho ảnh interior bình thường — đó là **table stakes**, ai
cũng làm được. Cố tình tuned cho "punchier" sẽ ra artifacts.

**Wedge thật của PPS v2 là 3 chức năng KHÔNG đối thủ nào có:**

1. **Virtual Staging** — phòng trống → có nội thất (SD3.5 + IPAdapter)
2. **Multi-angle Synthesis** — 1 ảnh → 3 góc nhìn (Qwen-Edit-2509)
3. **Instruction Editing** — "làm sáng nhà bếp", "xóa phản chiếu trong gương" (Qwen-Image-Lightning)

→ Xem `../docs/FEATURE_MATRIX_FOR_INVESTORS.md` để đối sánh feature.
→ Xem `../docs/MANUAL_ML_DEMO_GUIDE.md` để chạy demo ML thật cho 2 hero photo trên Colab.

---

## Section 1 · Color enhance baseline (parity với vendor)

11 ảnh customer chạy qua 3 pipeline:
- Cột A: Original
- Cột B1: AutoEnhance.ai (free trial, có watermark)
- Cột B2: Manuka enhanced
- Cột C: PPS v2 baseline (perspective + real_estate + studio finish)

**Đánh giá honest:** Output PPS v2 ngang vendor, không vượt trội rõ rệt. Đây
là baseline chứng minh chúng tôi làm được level 1 (color enhance) — *không
phải* differentiator của pitch.

| Photo | Folder |
|---|---|
| 0C1A5014 | [view](./all/0C1A5014/) |
| 0C1A5017 | [view](./all/0C1A5017/) |
| 0C1A5020 | [view](./all/0C1A5020/) |
| 0C1A5023 | [view](./all/0C1A5023/) |
| 0C1A5026 | [view](./all/0C1A5026/) |
| 0C1A5029 | [view](./all/0C1A5029/) |
| 0C1A5032 | [view](./all/0C1A5032/) |
| 0C1A5035 | [view](./all/0C1A5035/) |
| 0C1A5038 | [view](./all/0C1A5038/) |
| 0C1A5041 | [view](./all/0C1A5041/) |
| 0C1A5044 | [view](./all/0C1A5044/) |

Mỗi folder có 4 ảnh riêng + 4-cột composite + 2-cột before/after.

---

## Section 2 · 3 net-new features (THE pitch)

**Trạng thái:** ML stages chưa wire vào pipeline tự động (Phase 3, 1.5 tuần).
Để có demo cho pitch deck **NGAY hôm nay**, làm theo
`../docs/MANUAL_ML_DEMO_GUIDE.md`:

1. Pick 2 hero photo (đề xuất: 5029, 5044)
2. Chạy notebook **Qwen-Image-Lightning.ipynb** (Drive) trên Colab — instruction edit
3. Chạy notebook **Multiple-angles.ipynb** (Drive) trên Colab — multi-angle synthesis
4. Tạo 2-3 prompt cho virtual staging trên SD3.5 web UI hoặc Replicate
5. Drop output vào `feature_demos/`

Output dạng:
```
feature_demos/
  virtual_staging/
    5029_before_empty.jpg
    5029_after_furnished.jpg     ← wow demo
  multi_angle/
    5044_original.jpg
    5044_angle_left.jpg          ← wow demo
    5044_angle_right.jpg
  instruction_edit/
    5029_original.jpg
    5029_brighten_kitchen.jpg    ← wow demo
```

**Đây mới là phần làm NĐT "wow"** — không vendor nào hiện làm được.

---

## Cách dùng folder này trong pitch

- Slide 1 (build trust): "Chúng tôi cũng làm được phần basic" → Section 1
- Slide 2 (the ask): "Đây là chỗ chúng tôi DUY NHẤT có" → Section 2 demos
- Slide 3 (timeline): Phase 3 = automate Section 2 trong 1.5 tuần
- Slide 4 (validation): Phase 7 = launch beta cho 10 RE photographer/agency
