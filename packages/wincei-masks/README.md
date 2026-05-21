# pps-wincei-masks v0.1 — Smart Room Segmentation cho Photoshop

> Phân vùng tự động **sàn / tường / trần / cửa sổ / cửa đi + phào chỉ** (crown / baseboard / casing).
> Output: **PNG mask mỗi class + multi-page TIFF + colored overlay + PSD (optional)** — nhân viên Photoshop ctrl-click thumbnail = selection ngay.

## Mục đích

Thay vì nhân viên ngồi vẽ lasso 5-15 phút/ảnh, AI tự tách trước:

| Class | ADE20K id | Use case |
|---|---|---|
| `wall` | 0 | Đổi màu tường, sơn lại |
| `floor` | 3 | Đổi vân sàn, lau sạch |
| `ceiling` | 5 | Hạ sáng, chỉnh ám màu |
| `window` | 8 + sky | Fix outdoor view, glare |
| `door` | 14 | Mask cửa ra vào |
| `crown` | heuristic | Phào trần (seam wall\|ceiling) |
| `baseboard` | heuristic | Phào chân tường (seam wall\|floor) |
| `casing` | heuristic | Nẹp viền cửa/cửa sổ |

## Pipeline

```
Input ảnh BĐS (6K JPG/PNG/TIFF)
    │
    ├─► SegFormer-B3 ADE20K (auto-pick by VRAM up to B5)
    │     softmax → 7 soft prob maps full-res
    │
    ├─► PyMatting closed-form refinement (default ON)
    │     Auto-trimap từ soft prob → alpha biên đẹp sub-pixel
    │     (fallback guided filter nếu pymatting không cài)
    │
    ├─► Phào chỉ heuristic (default ON)
    │     a. Seam(wall,ceiling) → dilate ±18px → band
    │     b. Canny + LSD horizontal line filter
    │     c. Morphology close + remove small components
    │     d. → crown / baseboard / casing
    │
    └─► Export
          masks/<stem>/<stem>_wall.png       (8-bit alpha)
          masks/<stem>/<stem>_floor.png
          masks/<stem>/<stem>_ceiling.png
          masks/<stem>/<stem>_window.png
          masks/<stem>/<stem>_door.png
          masks/<stem>/<stem>_crown.png
          masks/<stem>/<stem>_baseboard.png
          masks/<stem>/<stem>_casing.png
          masks/<stem>/<stem>_overlay.jpg    (color preview)
          masks/<stem>/<stem>_channels.tif   (multi-page TIFF)
          masks/<stem>/<stem>.psd            (--psd)
```

## Install

```bash
pip install pps-wincei-masks
# Biên đẹp nhất:
pip install "pps-wincei-masks[matting]"   # PyMatting closed-form
# PSD writer:
pip install "pps-wincei-masks[psd]"       # pytoshop
```

## CLI

```bash
# Single
pps-wincei-masks DSC01527.jpg

# Batch folder
pps-wincei-masks --inputs ./fused --outputs ./psmasks

# Fast (no PyMatting, guided filter only)
pps-wincei-masks foto.jpg --no-refine

# Chỉ semantic, không phào chỉ
pps-wincei-masks foto.jpg --no-molding

# Bao gồm cả đèn
pps-wincei-masks foto.jpg --include-lights

# Force CPU
pps-wincei-masks foto.jpg --device cpu

# Resume mode
pps-wincei-masks --inputs ./fused --outputs ./psmasks --skip-existing
```

## Photoshop workflow nhân viên (30s/ảnh)

1. Open `<stem>_overlay.jpg` trong Bridge để duyệt
2. Open ảnh gốc trong Photoshop
3. File → Place Linked `<stem>_<class>.png` → load as layer
4. **Ctrl+Click thumbnail mask** = active selection
5. Tạo Adjustment Layer → fix chỉ vùng đó
   - Vd: Curves trên window mask → kéo outdoor xuống
   - Vd: Hue/Sat trên wall mask → repaint tường

Hoặc dùng TIFF channels:
1. File → Open `<stem>_channels.tif`
2. Channels panel → 8 channels named (wall/floor/ceiling/...)
3. Ctrl+Click channel name = selection

## Tốc độ

| Stage | GPU (RTX 3060) | CPU only |
|---|---|---|
| SegFormer-B3 inference | 0.4s | 6-12s |
| PyMatting refine 5 masks | 4-8s | 4-8s |
| Phào chỉ heuristic | 0.5s | 0.5s |
| Export PNG+TIFF+overlay | 1-2s | 1-2s |
| **Total / ảnh 6K** | **6-12s** | **15-25s** |

## Smart defaults — tin tao đi

- `SegFormer-B3` mặc định (không downgrade B0)
- PyMatting refinement ON (biên đẹp hơn nhiều so với raw argmax)
- Phào chỉ heuristic ON (3 loại detect đầy đủ)
- Multi-page TIFF ON (Photoshop friendly)
- Overlay JPG ON (QC nhanh trên Bridge)

## Limit + caveats

- **Phào chỉ heuristic**: dựa edge + seam → bỏ sót nếu phào quá mảnh (<5px) hoặc sơn cùng màu tường
- **SegFormer-B3 trên CPU**: chậm (~6-12s/ảnh) — chấp nhận được cho batch overnight
- **PSD writer (`pytoshop`)**: không stable, fallback dùng multi-page TIFF (cách an toàn hơn)
- **6K ảnh**: PyMatting đã auto-downscale → 1600px rồi upsample. Set `--matting-max-side 2400` nếu RAM > 16GB
