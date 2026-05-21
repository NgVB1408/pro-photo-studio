# Workflow Tổng — Dropbox khách → Masks Photoshop

> 2 dự án song song, chain với nhau khi xử lý folder khách gửi.

```
┌───────────────────────────────────────────────────────────────────┐
│  ./inputs/dropbox/                                                │
│      DSC01526.jpg  (EV-3)  ┐                                      │
│      DSC01527.jpg  (EV 0)  ├─ bracket 1                           │
│      DSC01528.jpg  (EV+3)  ┘                                      │
│      DSC01529.jpg  (EV-3)  ┐                                      │
│      ...                    │                                     │
│      (393 ảnh = 131 bracket × 3)                                  │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼  pps-wincei-hdr (Mertens, ~7s/group CPU)
┌───────────────────────────────────────────────────────────────────┐
│  ./outputs_hdr/                                                   │
│      DSC01527.jpg  (fused, EXIF từ EV=0)                          │
│      DSC01530.jpg                                                 │
│      ...  (131 ảnh)                                               │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼  pps-wincei-masks (SegFormer + matting, ~6min/ảnh CPU)
┌───────────────────────────────────────────────────────────────────┐
│  ./outputs_masks/                                                 │
│      DSC01527/                                                    │
│          DSC01527_wall.png        (51.7%)                         │
│          DSC01527_floor.png       ( 8.6%)                         │
│          DSC01527_ceiling.png     ( 1.4%)                         │
│          DSC01527_window.png      ( + sky)                        │
│          DSC01527_door.png        (11.4%)                         │
│          DSC01527_opening.png     (window ∪ door ∪ sky)           │
│          DSC01527_crown.png       (phào trần)                     │
│          DSC01527_baseboard.png   (phào chân tường)               │
│          DSC01527_casing.png      (nẹp viền)                      │
│          DSC01527_overlay.jpg     (color QC preview)              │
│          DSC01527_channels.tif    (Photoshop multi-page)          │
│      DSC01530/                                                    │
│      ...                                                          │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼  Nhân viên Photoshop (~30s/ảnh)
                       Adjust per-mask
```

## Lệnh 1-dòng

```bash
# 1. Fuse bracket
pps-wincei-hdr --inputs ./inputs/dropbox --outputs ./outputs_hdr

# 2. Extract masks
pps-wincei-masks --inputs ./outputs_hdr --outputs ./outputs_masks

# Hoặc 1 lệnh chain:
pps-wincei-hdr --inputs ./inputs/dropbox --outputs ./outputs_hdr --chain-wincei
```

## Tốc độ

| Stage         | CPU (i5-7500)  | GPU (RTX 3060)  |
|---------------|----------------|------------------|
| HDR fuse 131 group | ~15 phút   | ~10 phút         |
| Masks 131 ảnh      | ~13 giờ    | ~25 phút         |
| Photoshop adjust   | ~65 phút (30s × 131) | 65 phút   |

Khuyến nghị mode batch CPU: chạy overnight, sáng hôm sau giao nhân viên đã có sẵn folder masks/.

## Resume

```bash
pps-wincei-masks --inputs ./outputs_hdr --outputs ./outputs_masks --skip-existing
```

→ Bỏ qua ảnh đã có folder masks, tiếp tục từ ảnh chưa làm.
