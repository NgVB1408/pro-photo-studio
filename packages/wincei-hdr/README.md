# pps-wincei-hdr v0.1

> HDR bracket fusion (Mertens 2007) cho ảnh BĐS — Sony A7M4 AEB → 1 LDR JPG, EXIF preserved.

## Khi nào dùng

Khách gửi folder ảnh với **Sony AEB** (Auto Exposure Bracketing) — mỗi cảnh có 3-5 shot liên tiếp với EV khác nhau (vd: −3, 0, +3). Tool tự gom triplet → fuse thành 1 ảnh có cả outdoor view + indoor detail.

## Khác wincei (window+ceiling fix)

| | `pps-wincei` | `pps-wincei-hdr` |
|---|---|---|
| Input | 1 JPG | N JPG bracket cùng cảnh |
| Outdoor blown | Compress tone (vẫn xám) | **Recover từ shot −EV** (cây + sky thật) |
| Cần GPU | Khuyến nghị | Không cần |
| Tốc độ | 30-60s/ảnh | 5-15s/group |
| Output | JPG q98 | JPG q98 |

## Pipeline

```
Sony A7M4 AEB ±3EV (3 shot)
    ├─ EXIF detect: DateTimeOriginal ±2s + EV variation
    ├─ AlignMTB (handheld jitter compensation)
    ├─ createMergeMertens (contrast + saturation + exposure measure)
    ├─ Output: BGR uint8 → JPG q98 + EXIF từ shot EV≈0
    └─ (optional) → pps-wincei window+ceiling polish
```

## Install

```bash
pip install pps-wincei-hdr
# Chain wincei downstream:
pip install "pps-wincei-hdr[chain]"
```

## CLI

```bash
# Drop khách's bracket folder vào ./inputs
pps-wincei-hdr --inputs ./inputs --outputs ./outputs

# Tripod input → tắt align để nhanh hơn
pps-wincei-hdr --no-align

# Pull outdoor mạnh hơn (giảm exposure-weight)
pps-wincei-hdr --exposure-weight 0.4

# Auto chain wincei sau khi fuse
pps-wincei-hdr --chain-wincei

# Dry-run xem bracket detect
pps-wincei-hdr --dry-run
```

## Output filename

Tên file output = tên của shot **EV≈0** trong group (reference shot). Ví dụ:
- Input: `DSC01628.jpg` (-3 EV), `DSC01629.jpg` (0 EV), `DSC01630.jpg` (+3 EV)
- Output: `DSC01629.jpg` (fused)

Khách reference theo tên giữa = match với JPG họ đã preview trên máy ảnh.

## EXIF

- Camera body / lens / ISO / aperture: giữ từ reference shot
- Shutter speed: giữ từ reference (EV=0 shutter)
- ICC profile: preserved
- `UserComment`: `pps-wincei-hdr v0.1 | Mertens fusion | n=3 | EV[-3.0,+3.0] | ref=DSC01629.jpg` (audit trail)

## Bracket detection rules

1. Sort theo `DateTimeOriginal` rồi `filename`
2. Group nếu: ΔDateTimeOriginal ≤ `--time-tolerance` (default 3s) **VÀ** `ExposureBiasValue` khác trong group
3. Group size 2-7
4. EV variation bắt buộc (1 ảnh không phải bracket)

Không detect được → ảnh đi vào `singletons`. Dùng `--keep-singletons` để copy nguyên.

## Limit cứng

Mertens KHÔNG cần CRF, nhưng:
- Nếu input chỉ có 1 EV (không bracket thật) → output ≈ shot gốc
- Handheld lệch quá (>1% frame width) → AlignMTB không cứu được, có thể bị ghost
- Object di chuyển trong frame (cây lay, người đi) → ghost trong fusion → cần thêm de-ghost module (chưa có)
