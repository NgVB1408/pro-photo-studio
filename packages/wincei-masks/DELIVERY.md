# Bàn giao khách — Wincei Stack (pps-wincei + pps-wincei-hdr + pps-wincei-masks + pps-wincei-api)

> **Real-estate photo automation suite — bàn giao đầy đủ tài liệu, code, API, .exe.**

## 📦 Bộ giao

| File | Mô tả |
|---|---|
| `wincei-stack.exe` | All-in-one Windows binary (PyInstaller) |
| `wincei-api/` | FastAPI service REST |
| `docs/DELIVERY.md` | Tài liệu này (VN) |
| `docs/API_REFERENCE.md` | Endpoint reference + curl examples |
| `docs/POSTMAN_COLLECTION.json` | Postman import |
| `samples/input/` | 6 ảnh test |
| `samples/output/` | Output mẫu (HDR + masks + regions JSON) |

---

## 1️⃣ 4 PHẦN CHÍNH

### Phần A — `pps-wincei` (window+ceiling fix)
Fix cửa sổ blown + trần ám màu trên ảnh single shot.
- SegFormer ADE20K detect mask
- Reinhard tonemap + Bradford CAT chromatic adaptation
- AI self-eval 7 metrics

### Phần B — `pps-wincei-hdr` (HDR bracket fusion)
Gộp 3-5 shot Sony AEB (-3, 0, +3 EV) → 1 ảnh có outdoor view thật.
- Auto-detect bracket qua EXIF DateTimeOriginal + ExposureBiasValue
- AlignMTB compensate handheld
- Mertens exposure fusion

### Phần C — `pps-wincei-masks` (smart segmentation)
Tách wall / floor / ceiling / window / door / phào chỉ → Photoshop masks.
- **2 engines**:
  - `semantic` (default): SegFormer ADE20K + PyMatting refine
  - `vlm-sam2` (pro): Ollama VLM + SAM 2 (sub-pixel boundary)
- **Precision mode**: SegFormer-B5 + multi-scale TTA + DenseCRF + tile matting full-res
- **AI Supervisor**: 7-metric eval per mask, verdict pass/review/fail/no_target
- **Real Estate Vision Analyzer**: JSON bbox normalized [0..1000]
- Output: PNG mỗi class + multi-page TIFF + overlay JPG + PSD optional

### Phần D — `pps-wincei-api` (REST service)
FastAPI wrap cả 3 module → endpoint cho integrate web/mobile/desktop.

---

## 2️⃣ CÀI ĐẶT

### Cách 1 — Windows .exe (đơn giản nhất)

```cmd
:: Double-click wincei-stack.exe
:: hoặc terminal:
wincei-stack.exe --help
```

### Cách 2 — Python pip

```bash
pip install pps-wincei pps-wincei-hdr pps-wincei-masks pps-wincei-api
pip install pymatting   # khuyến nghị cho biên mượt
```

### Cách 3 — Source

```bash
git clone https://github.com/NgVB1408/pro-photo-studio
cd pro-photo-studio
uv sync --all-packages
```

### Engine 'vlm-sam2' (optional — chất lượng cao nhất)

```bash
# 1. Cài Ollama (https://ollama.com/download)
ollama serve

# 2. Pull VLM
ollama pull qwen2.5vl:7b      # 4.7GB, balance
# hoặc
ollama pull llama3.2-vision:11b  # 7.1GB, smartest

# 3. Cài SAM 2
pip install "git+https://github.com/facebookresearch/sam2.git"

# 4. Download SAM 2 checkpoint
mkdir -p ~/.cache/sam2
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt \
     -o ~/.cache/sam2/sam2_hiera_tiny.pt
```

---

## 3️⃣ CÁCH DÙNG

### CLI

```bash
# HDR fuse
pps-wincei-hdr --inputs ./dropbox --outputs ./fused

# Window+ceiling fix
pps-wincei foto.jpg --out fixed.jpg

# Smart masks (semantic engine — default)
pps-wincei-masks foto.jpg --outputs ./masks

# Precision mode (chất lượng cao)
pps-wincei-masks foto.jpg --outputs ./masks --precision

# VLM+SAM2 engine (pro mode, cần Ollama + SAM2)
pps-wincei-masks foto.jpg --outputs ./masks --engine vlm-sam2

# Regions JSON output
pps-wincei-regions foto.jpg --pretty
pps-wincei-regions foto.jpg --out result.json

# Folder batch
pps-wincei-masks --inputs ./fused --outputs ./psmasks --skip-existing

# Full chain HDR → masks
pps-wincei-hdr --inputs ./dropbox --outputs ./fused
pps-wincei-masks --inputs ./fused --outputs ./psmasks --precision
```

### REST API

```bash
# Khởi động server
pps-wincei-api
# → http://localhost:8088
# → Swagger: http://localhost:8088/docs
# → ReDoc:   http://localhost:8088/redoc

# Mock mode (test integration nhanh, không tốn CPU)
curl -X POST http://localhost:8088/api/v1/segment-masks \
  -F "files=@foto.jpg" -F "mock=true"

# Real async job
JOB=$(curl -s -X POST http://localhost:8088/api/v1/segment-masks \
  -F "files=@foto.jpg" | jq -r .job_id)

# Poll
curl http://localhost:8088/api/v1/jobs/$JOB

# Download zip kết quả
curl -O http://localhost:8088/api/v1/jobs/$JOB/download

# Regions JSON (sync)
curl -X POST http://localhost:8088/api/v1/detect-regions \
  -F "file=@foto.jpg"
```

---

## 4️⃣ OUTPUT STRUCTURE

### Masks folder (`pps-wincei-masks`)

```
outputs/<stem>/
  ├── <stem>_wall.png            # Tường (alpha 8-bit)
  ├── <stem>_floor.png           # Sàn
  ├── <stem>_ceiling.png         # Trần
  ├── <stem>_window.png          # Cửa sổ
  ├── <stem>_door.png            # Cửa đi
  ├── <stem>_opening.png         # Cửa kính / cửa sổ kính (union)
  ├── <stem>_crown.png           # Phào trần
  ├── <stem>_baseboard.png       # Phào chân tường
  ├── <stem>_casing.png          # Nẹp cửa
  ├── <stem>_overlay.jpg         # Color preview QC
  ├── <stem>_channels.tif        # Multi-page TIFF (Photoshop Channels)
  ├── <stem>.psd                 # Optional Photoshop file (cần pytoshop)
  └── <stem>_qc_report.json      # AI Supervisor verdict + metrics
```

### Regions JSON (`pps-wincei-regions`)

```json
{
  "project_type": "Real Estate Segmentation",
  "image_size": {"width": 4608, "height": 3072},
  "camera_angle": "wide_angle",
  "detected_elements": {
    "ceiling": {
      "coordinates": [0, 0, 280, 1000],
      "confidence": 0.94,
      "has_crown_molding": false,
      "area_pct": 12.5
    },
    "windows": [
      {"id": 1, "coordinates": [250, 120, 680, 450], "type": "Casement Window", "confidence": 0.91},
      {"id": 2, "coordinates": [250, 530, 680, 880], "type": "Casement Window", "confidence": 0.88}
    ],
    "walls": {"coordinates": [0, 0, 820, 1000], "confidence": 0.96, "area_pct": 51.7},
    "floor": {"coordinates": [780, 0, 1000, 1000], "confidence": 0.95, "area_pct": 8.6}
  },
  "reasoning_steps": [
    "Step 1: SegFormer ADE20K inference → 9 class soft prob",
    "Step 2: camera_angle = wide_angle",
    "Step 3: subtract curtain (3.2% cov) khỏi window mask",
    "Step 4: crown_molding = false",
    "Step 5: window panes detected = 2"
  ]
}
```

Dùng Python để crop ceiling:
```python
from PIL import Image
img = Image.open("foto.jpg")
w, h = img.size
ceiling = json_result["detected_elements"]["ceiling"]
ymin, xmin, ymax, xmax = ceiling["coordinates"]
box = (xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000)
img.crop(box).save("ceiling.jpg")
```

---

## 5️⃣ AI SUPERVISOR — Verdict & Quality Gate

Mỗi ảnh process xong → AI self-eval 7 metrics:

| Metric | Trọng số | Đo gì |
|---|---|---|
| `coverage_sanity` | 20% | Coverage class có nằm trong expected range? |
| `boundary_smoothness` | 15% | Biên có zig-zag không? |
| `no_orphan_blobs` | 15% | Mask có 1 component lớn dominant? |
| `edge_alignment` | 25% | Biên mask có align với edge ảnh? |
| `hole_rate` | 15% | Mask có lỗ rỗng bên trong? |
| `soft_alpha_quality` | 10% | Biên có gradient mượt sub-pixel? |

**Verdict:**
- `pass` ≥ 0.85 → giao luôn
- `review` 0.65-0.84 → đọc recommendations, có thể tự fix Photoshop
- `fail` < 0.65 → re-run với precision mode hoặc manual

Quality gate auto retry trong CLI: `--precision` hoặc `--retry-on-fail`.

---

## 6️⃣ WORKFLOW TỔNG (recommended)

```
Khách Dropbox bracket folder (393 ảnh = 131×3)
    │
    ▼  pps-wincei-hdr (~7s/group CPU)
./fused/ (131 ảnh fused, EXIF preserved)
    │
    ▼  pps-wincei-masks --precision (~10-15min/ảnh CPU)
./psmasks/<stem>/
    ├── 9 PNG masks
    ├── overlay.jpg
    ├── channels.tif
    └── qc_report.json
    │
    ▼  Photoshop nhân viên (30s/ảnh)
Final retouched listings
```

---

## 7️⃣ TROUBLESHOOTING

| Triệu chứng | Khắc phục |
|---|---|
| `MemoryError` semantic inference | Giảm `--matting-max-side`, hoặc dùng `--engine semantic` không precision |
| `Window mask 0%` | Cửa thực = glass door → dùng `_opening.png` (union window+door+sky) |
| Phào chỉ 0% | Ảnh modern không có phào nổi → đúng, không phải bug |
| `verdict=fail` | Chạy lại với `--precision --retry-on-fail` |
| Ollama "model not found" | `ollama pull qwen2.5vl:7b` |
| SAM2 checkpoint missing | Tải `sam2_hiera_tiny.pt` về `~/.cache/sam2/` |
| API server không khởi động | Check port 8088 free: `netstat -ano | findstr 8088` |

---

## 8️⃣ HỖ TRỢ

- **Email**: luvdraco88@gmail.com
- **Zalo**: 0876 254 585
- **GitHub**: https://github.com/NgVB1408/pro-photo-studio
- **Issues**: https://github.com/NgVB1408/pro-photo-studio/issues

License: Apache-2.0
