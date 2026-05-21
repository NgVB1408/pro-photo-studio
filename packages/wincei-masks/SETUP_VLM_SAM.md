# Setup VLM (Ollama) + SAM 2 — Full Pro Stack

> Mục tiêu: kéo điểm ceiling từ ⚠️ review (~0.72) lên ✅ pass (≥0.92) bằng VLM directive + SAM 2 segmentation.

## 1. Cài Ollama

### Mac / Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download installer: https://ollama.com/download/OllamaSetup.exe

Verify:
```bash
ollama --version
```

## 2. Pull VLM base + build bds-brain

```bash
# Khởi động Ollama (background)
ollama serve &

# Pull base VLM (~4.7GB, qwen2.5-vl 7B)
ollama pull qwen2.5vl:7b

# Hoặc với VRAM thấp:
# ollama pull llama3.2-vision:11b

# Build bds-brain custom model từ Modelfile
ollama create bds-brain -f packages/wincei-masks/Modelfile.bds-brain

# Test
ollama list   # phải thấy "bds-brain"
ollama run bds-brain "test"   # quit by Ctrl-D
```

## 3. Cài SAM 2

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"

# Download checkpoint (chọn 1):
mkdir -p ~/.cache/sam2

# Tiny (38MB, CPU OK)
curl -L -o ~/.cache/sam2/sam2_hiera_tiny.pt \
     https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

# Small (185MB)
# curl -L -o ~/.cache/sam2/sam2_hiera_small.pt \
#      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# Base+ (323MB, recommended cho biên tốt nhất CPU)
# curl -L -o ~/.cache/sam2/sam2_hiera_base_plus.pt \
#      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Large (898MB, cần 4GB+ VRAM)
# curl -L -o ~/.cache/sam2/sam2_hiera_large.pt \
#      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## 4. Test full stack

### Via CLI
```bash
pps-wincei-masks foto.jpg --engine vlm-sam2 \
  --vlm-model bds-brain \
  --sam-checkpoint ~/.cache/sam2/sam2_hiera_tiny.pt \
  --outputs ./outputs
```

### Via API endpoint /api/v1/full-recovery-ceiling
```bash
curl -X POST http://localhost:8088/api/v1/full-recovery-ceiling \
  -F "file=@foto.jpg" \
  -F "vlm_model=bds-brain" \
  -F "output_format=json" \
  | jq
```

Response:
```json
{
  "source_file": "foto.jpg",
  "vlm_model": "bds-brain",
  "vlm_response_ms": 2340,
  "points_detected": [[2310, 180], [1500, 220], [3200, 200]],
  "sam_score": 0.94,
  "ceiling_cov_pct": 12.3,
  "output_png": "./wincei_storage/outputs/ceiling_recovery/foto_ceiling_recovery.png",
  "method": "vlm_sam2_full_recovery"
}
```

Or download PNG directly:
```bash
curl -X POST http://localhost:8088/api/v1/full-recovery-ceiling \
  -F "file=@foto.jpg" \
  --output ceiling_recovery.png
```

## 5. Cấu trúc hoạt động

```
ảnh input
   │
   ▼  POST /api/v1/full-recovery-ceiling
Ollama bds-brain (Qwen2.5-VL custom Modelfile)
   │   System prompt: "bỏ qua nhãn ADE20K, dùng perspective + đèn + drywall"
   │   Return: {"points": [[x1,y1], [x2,y2], ...]}
   │
   ▼  Multi-point prompt
SAM 2 (Hiera Tiny/Small/Base+)
   │   predictor.predict(point_coords, point_labels=[1,1,1,...], multimask_output=True)
   │   "Loang mực" theo edge thật của ceiling, bám phào chỉ + drywall corners
   │
   ▼
Output: PNG RGBA (alpha = ceiling mask) + score + JSON report
```

## 6. Tốc độ thực tế

| Stage | CPU | GPU RTX 3060 | GPU RTX 4090 |
|---|---|---|---|
| Ollama VLM query | 2-8s | 1-3s | 0.5-1s |
| SAM 2 set_image (1 lần/ảnh) | 5-15s | 1-2s | 0.3s |
| SAM 2 predict_from_points | 0.5-2s | 0.1s | 0.05s |
| **Total / ảnh 6K** | **8-25s** | **2-5s** | **<2s** |

## 7. Troubleshooting

| Vấn đề | Khắc phục |
|---|---|
| `Ollama không chạy` | `ollama serve` (background process) |
| `Model 'bds-brain' chưa pull` | Re-run `ollama create bds-brain -f Modelfile.bds-brain` |
| `SAM 2 checkpoint missing` | curl tải `sam2_hiera_tiny.pt` về `~/.cache/sam2/` |
| `OOM khi SAM 2 set_image` | Downscale ảnh max_side 2048, hoặc dùng GPU |
| `VLM trả về JSON sai schema` | Check `ollama run bds-brain` test prompt, edit Modelfile SYSTEM |
| `Ceiling vẫn thấp` | VLM có thể chỉ điểm sai; debug raw response qua `?output_format=json` |

## 8. Tune VLM cho ảnh đặc biệt

Khách có ảnh khó? Edit `Modelfile.bds-brain` thêm rules:

```dockerfile
SYSTEM """
... existing rules ...

THÊM CHO ẢNH ĐẶC BIỆT:
- Trần dốc (skylight): vẫn coi là ceiling, KHÔNG là wall.
- Trần kép (drop ceiling): điểm rải cả 2 mức.
- Trần kính (glass): vẫn là ceiling, không phải window.
- Ảnh chụp từ dưới lên (low angle): mở rộng vùng ceiling ra cả góc nghiêng.
"""
```

Rebuild:
```bash
ollama rm bds-brain
ollama create bds-brain -f Modelfile.bds-brain
```
