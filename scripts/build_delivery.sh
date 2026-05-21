#!/usr/bin/env bash
# Build delivery package cho khách — wincei-stack.exe + docs + API + samples
#
# Usage:
#   bash scripts/build_delivery.sh
#
# Output: ./delivery/wincei_delivery_v{VERSION}_{DATE}.zip

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="0.3.0"
DATE=$(date +%Y%m%d)
DELIVERY_DIR="$ROOT/delivery/wincei_delivery_v${VERSION}_${DATE}"
ZIP_PATH="$ROOT/delivery/wincei_delivery_v${VERSION}_${DATE}.zip"

echo "🔨 Building delivery package..."
echo "   Version: $VERSION"
echo "   Date   : $DATE"
echo "   Target : $ZIP_PATH"
echo ""

rm -rf "$DELIVERY_DIR"
mkdir -p "$DELIVERY_DIR"/{bin,docs,samples/input,samples/output,api,scripts}

echo "📝 Copy documentation..."
cp packages/wincei-masks/DELIVERY.md "$DELIVERY_DIR/docs/"
cp packages/wincei-masks/README.md   "$DELIVERY_DIR/docs/MASKS_README.md"
cp packages/wincei-masks/SETUP_VLM_SAM.md "$DELIVERY_DIR/docs/SETUP_VLM_SAM.md"
cp packages/wincei-masks/Modelfile.bds-brain "$DELIVERY_DIR/docs/" 2>/dev/null || true
cp packages/wincei-hdr/README.md     "$DELIVERY_DIR/docs/HDR_README.md"
cp packages/wincei-hdr/WORKFLOW.md   "$DELIVERY_DIR/docs/WORKFLOW.md"
cp packages/wincei-api/README.md     "$DELIVERY_DIR/docs/API_README.md"
cp packages/wincei/README.md         "$DELIVERY_DIR/docs/WINCEI_README.md" 2>/dev/null || true
cp scripts/setup_vlm_sam.sh          "$DELIVERY_DIR/scripts/setup_vlm_sam.sh" 2>/dev/null || true
chmod +x "$DELIVERY_DIR/scripts/setup_vlm_sam.sh" 2>/dev/null || true

echo "🌐 Copy API source..."
mkdir -p "$DELIVERY_DIR/api/pps_wincei_api"
cp -r packages/wincei-api/pps_wincei_api/* "$DELIVERY_DIR/api/pps_wincei_api/"
cp packages/wincei-api/pyproject.toml "$DELIVERY_DIR/api/"

echo "📚 Copy FULL source code (4 packages)..."
mkdir -p "$DELIVERY_DIR/src/packages"
for pkg in wincei wincei-hdr wincei-masks wincei-api; do
  if [ -d "packages/$pkg" ]; then
    mkdir -p "$DELIVERY_DIR/src/packages/$pkg"
    # Use cp + manual exclude (rsync không có sẵn trên Git Bash)
    cp -r "packages/$pkg/." "$DELIVERY_DIR/src/packages/$pkg/"
    find "$DELIVERY_DIR/src/packages/$pkg/" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find "$DELIVERY_DIR/src/packages/$pkg/" -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
    find "$DELIVERY_DIR/src/packages/$pkg/" -type f -name '*.pyc' -delete 2>/dev/null || true
    echo "   ✓ $pkg"
  fi
done

echo "📐 Copy workspace config..."
cp pyproject.toml "$DELIVERY_DIR/src/pyproject.toml" 2>/dev/null || true
cp uv.lock "$DELIVERY_DIR/src/uv.lock" 2>/dev/null || true
cp LICENSE "$DELIVERY_DIR/src/LICENSE" 2>/dev/null || true

# Setup-from-source guide
cat > "$DELIVERY_DIR/src/SETUP.md" <<'EOFSETUP'
# Build từ Source

## Prerequisites
- Python 3.11 hoặc 3.12 (KHÔNG 3.13)
- pip hoặc uv
- Optional: GPU NVIDIA (CUDA 12.x) cho tốc độ

## Cài đặt (uv — khuyến nghị)

```bash
# Cài uv
curl -LsSf https://astral.sh/uv/install.sh | sh    # Linux/Mac
# Hoặc Windows PowerShell:
irm https://astral.sh/uv/install.ps1 | iex

cd <thư-mục-này>
uv sync --all-packages
```

Sau khi sync, 5 CLI có sẵn:
- `pps-wincei` — window+ceiling fix
- `pps-wincei-folder` — batch + HTML viewer
- `pps-wincei-hdr` — HDR bracket fusion
- `pps-wincei-masks` — smart segmentation + AI eval
- `pps-wincei-regions` — regions JSON output
- `pps-wincei-api` — REST API server

## Cài đặt (pip thuần)

```bash
pip install ./packages/wincei
pip install ./packages/wincei-hdr
pip install ./packages/wincei-masks
pip install ./packages/wincei-api

# Optional dependencies
pip install pymatting       # cho biên mượt
pip install pytoshop        # cho PSD output
```

## Test cài đặt

```bash
pps-wincei-api --version
pps-wincei-masks --version
pps-wincei-hdr --version
```

## Khởi động API server

```bash
pps-wincei-api
# → http://localhost:8088
# Swagger UI: http://localhost:8088/docs
```

## GPU acceleration (optional)

```bash
# Cài torch với CUDA matching driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

GPU > 6GB VRAM → tự động dùng SegFormer-B3/B5 (chất lượng cao nhất).
CPU only → tự fall back B0/B1 (vẫn hoạt động, chậm hơn 5-15x).
EOFSETUP

echo "🐳 Add Dockerfile..."
cat > "$DELIVERY_DIR/src/Dockerfile" <<'EOFDOCKER'
# Multi-stage build cho production deployment
FROM python:3.11-slim AS builder
WORKDIR /build
COPY src/ /build/src/
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        ./src/packages/wincei \
        ./src/packages/wincei-hdr \
        ./src/packages/wincei-masks \
        ./src/packages/wincei-api \
        pymatting

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/pps-wincei-api /usr/local/bin/

EXPOSE 8088
ENV WINCEI_HOST=0.0.0.0 WINCEI_PORT=8088
CMD ["pps-wincei-api"]
EOFDOCKER

echo "🔄 Add docker-compose..."
cat > "$DELIVERY_DIR/src/docker-compose.yml" <<'EOFCOMPOSE'
version: "3.9"
services:
  wincei-api:
    build:
      context: ..
      dockerfile: src/Dockerfile
    image: wincei-api:0.2.2
    ports:
      - "8088:8088"
    volumes:
      - ./wincei_storage:/app/wincei_storage
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      WINCEI_STORAGE: /app/wincei_storage
      WINCEI_PORT: "8088"
      WINCEI_CORS: "*"
    restart: unless-stopped
EOFCOMPOSE

echo "📦 Copy PyInstaller spec..."
cp packages/wincei-api/wincei_stack.spec "$DELIVERY_DIR/bin/"
cat > "$DELIVERY_DIR/bin/BUILD_EXE.md" <<'EOF'
# Build wincei-stack.exe

```bash
# 1. Cài PyInstaller
pip install pyinstaller

# 2. Build
cd <pro-photo-studio root>
pyinstaller packages/wincei-api/wincei_stack.spec

# 3. Output
ls dist/wincei-stack.exe  # ~2-3 GB
```

## Run

```bash
wincei-stack.exe                    # API server :8088
wincei-stack.exe wincei foto.jpg    # window+ceiling fix
wincei-stack.exe hdr --inputs ...   # HDR fuse
wincei-stack.exe masks foto.jpg     # segmentation
wincei-stack.exe regions foto.jpg   # JSON output
```
EOF

echo "🖼️ Copy sample outputs (nếu có)..."
if [ -d "C:/Users/kulam/wincei_test/outputs_hdr" ]; then
  cp C:/Users/kulam/wincei_test/outputs_hdr/DSC0152*.jpg "$DELIVERY_DIR/samples/output/" 2>/dev/null || true
fi
if [ -d "C:/Users/kulam/wincei_test/outputs_masks" ]; then
  cp -r C:/Users/kulam/wincei_test/outputs_masks/DSC01527 "$DELIVERY_DIR/samples/output/masks_DSC01527" 2>/dev/null || true
fi
if [ -f "C:/Users/kulam/wincei_test/DSC01527_regions.json" ]; then
  cp C:/Users/kulam/wincei_test/DSC01527_regions.json "$DELIVERY_DIR/samples/output/" 2>/dev/null || true
fi

echo "📋 Generate Postman collection..."
cat > "$DELIVERY_DIR/docs/POSTMAN_COLLECTION.json" <<'EOF'
{
  "info": {
    "name": "Wincei API v0.2",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health",
      "request": {
        "method": "GET",
        "url": {"raw": "{{baseUrl}}/api/v1/health", "host": ["{{baseUrl}}"], "path": ["api", "v1", "health"]}
      }
    },
    {
      "name": "Window+Ceiling (sync mock)",
      "request": {
        "method": "POST",
        "url": {"raw": "{{baseUrl}}/api/v1/window-ceiling", "host": ["{{baseUrl}}"], "path": ["api", "v1", "window-ceiling"]},
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file"},
            {"key": "mock", "value": "true", "type": "text"}
          ]
        }
      }
    },
    {
      "name": "HDR Fuse (mock)",
      "request": {
        "method": "POST",
        "url": {"raw": "{{baseUrl}}/api/v1/hdr-fuse", "host": ["{{baseUrl}}"], "path": ["api", "v1", "hdr-fuse"]},
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "files", "type": "file"},
            {"key": "mock", "value": "true", "type": "text"}
          ]
        }
      }
    },
    {
      "name": "Segment Masks (mock)",
      "request": {
        "method": "POST",
        "url": {"raw": "{{baseUrl}}/api/v1/segment-masks", "host": ["{{baseUrl}}"], "path": ["api", "v1", "segment-masks"]},
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "files", "type": "file"},
            {"key": "refine_edges", "value": "true", "type": "text"},
            {"key": "detect_molding", "value": "true", "type": "text"},
            {"key": "mock", "value": "true", "type": "text"}
          ]
        }
      }
    },
    {
      "name": "Detect Regions JSON (mock)",
      "request": {
        "method": "POST",
        "url": {"raw": "{{baseUrl}}/api/v1/detect-regions", "host": ["{{baseUrl}}"], "path": ["api", "v1", "detect-regions"]},
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file"},
            {"key": "mock", "value": "true", "type": "text"}
          ]
        }
      }
    },
    {
      "name": "List Jobs",
      "request": {
        "method": "GET",
        "url": {"raw": "{{baseUrl}}/api/v1/jobs?limit=50", "host": ["{{baseUrl}}"], "path": ["api", "v1", "jobs"], "query": [{"key": "limit", "value": "50"}]}
      }
    },
    {
      "name": "Get Job by ID",
      "request": {
        "method": "GET",
        "url": {"raw": "{{baseUrl}}/api/v1/jobs/:id", "host": ["{{baseUrl}}"], "path": ["api", "v1", "jobs", ":id"]}
      }
    },
    {
      "name": "Download Job Output",
      "request": {
        "method": "GET",
        "url": {"raw": "{{baseUrl}}/api/v1/jobs/:id/download", "host": ["{{baseUrl}}"], "path": ["api", "v1", "jobs", ":id", "download"]}
      }
    }
  ],
  "variable": [
    {"key": "baseUrl", "value": "http://localhost:8088"}
  ]
}
EOF

echo "🛠️ Quick test scripts..."
cat > "$DELIVERY_DIR/scripts/test_api.sh" <<'EOF'
#!/usr/bin/env bash
# Test các endpoint chính qua curl
set -e
BASE="${1:-http://localhost:8088}"
echo "=== Health ==="
curl -s "$BASE/api/v1/health" | python -m json.tool
echo ""
echo "=== Window+Ceiling mock ==="
curl -s -X POST -F "file=@samples/input/foto.jpg" -F "mock=true" \
     "$BASE/api/v1/window-ceiling" | python -m json.tool
echo ""
echo "=== Segment Masks mock ==="
curl -s -X POST -F "files=@samples/input/foto.jpg" -F "mock=true" \
     "$BASE/api/v1/segment-masks" | python -m json.tool
echo ""
echo "=== Detect Regions mock ==="
curl -s -X POST -F "file=@samples/input/foto.jpg" -F "mock=true" \
     "$BASE/api/v1/detect-regions" | python -m json.tool
EOF
chmod +x "$DELIVERY_DIR/scripts/test_api.sh"

cat > "$DELIVERY_DIR/scripts/test_api.ps1" <<'EOF'
# Windows PowerShell test
param([string]$BaseUrl = "http://localhost:8088")
Write-Host "=== Health ===" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/api/v1/health" | ConvertTo-Json -Depth 5
Write-Host "`n=== Segment Masks mock ===" -ForegroundColor Cyan
$f = "samples/input/foto.jpg"
$form = @{ files = Get-Item $f; mock = "true" }
Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/segment-masks" -Form $form | ConvertTo-Json -Depth 5
EOF

cat > "$DELIVERY_DIR/README.md" <<EOF
# Wincei Delivery Package v${VERSION}

> Bàn giao ngày ${DATE}. **Đầy đủ source code 4 package + API + docs + samples.**

## Cấu trúc

\`\`\`
.
├── README.md                  ← Bạn đang đọc
├── docs/                      ← Documentation
│   ├── DELIVERY.md            ← Tài liệu chính (VN)
│   ├── WORKFLOW.md            ← Pipeline tổng
│   ├── HDR_README.md
│   ├── MASKS_README.md
│   ├── WINCEI_README.md
│   ├── API_README.md
│   └── POSTMAN_COLLECTION.json
├── src/                       ← FULL source code 4 package
│   ├── packages/
│   │   ├── wincei/            ← window+ceiling fix
│   │   ├── wincei-hdr/        ← HDR bracket fusion
│   │   ├── wincei-masks/      ← smart segmentation + AI eval + regions JSON
│   │   └── wincei-api/        ← FastAPI REST service
│   ├── pyproject.toml         ← uv workspace config
│   ├── uv.lock                ← reproducible lock
│   ├── Dockerfile             ← Production deploy
│   ├── docker-compose.yml
│   ├── SETUP.md               ← Build instructions
│   └── LICENSE
├── api/                       ← API source standalone (subset của src/packages/wincei-api)
├── bin/
│   ├── wincei_stack.spec      ← PyInstaller spec → wincei-stack.exe
│   └── BUILD_EXE.md
├── samples/
│   ├── input/                 ← Test ảnh
│   └── output/                ← Output mẫu (HDR + 9 masks + regions JSON)
└── scripts/
    ├── test_api.sh
    └── test_api.ps1
\`\`\`

## Quick start (3 đường)

### A. Build từ source (đầy đủ control)
\`\`\`bash
cd src/
# Xem SETUP.md để biết chi tiết
uv sync --all-packages
pps-wincei-api    # → http://localhost:8088
\`\`\`

### B. Docker (production)
\`\`\`bash
cd src/
docker compose up -d
\`\`\`

### C. Build .exe (cho non-tech end-user)
\`\`\`bash
cd src/
pip install pyinstaller
pyinstaller ../bin/wincei_stack.spec
# → dist/wincei-stack.exe
\`\`\`

## Tài liệu

1. \`docs/DELIVERY.md\` → tổng quan + workflow
2. \`docs/API_README.md\` → REST API reference
3. \`docs/MASKS_README.md\` → Smart segmentation
4. \`docs/HDR_README.md\` → HDR bracket fusion
5. \`docs/WINCEI_README.md\` → window+ceiling fix
6. \`docs/WORKFLOW.md\` → Pipeline end-to-end
7. \`docs/POSTMAN_COLLECTION.json\` → import Postman test API

## License

Apache-2.0 (xem \`src/LICENSE\`)

## Hỗ trợ

luvdraco88@gmail.com · Zalo 0876 254 585 · GitHub: NgVB1408/pro-photo-studio
EOF

echo ""
echo "🗜️ Zip delivery..."
cd "$ROOT/delivery"
rm -f "$(basename "$ZIP_PATH")"
python -c "
import shutil, sys
src = sys.argv[1]
dst = sys.argv[2]
shutil.make_archive(dst.replace('.zip', ''), 'zip', src)
print(f'Wrote: {dst}')
" "$DELIVERY_DIR" "$ZIP_PATH"

echo ""
echo "✅ Delivery package built:"
echo "   $ZIP_PATH"
echo "   Size: $(du -h "$ZIP_PATH" | cut -f1)"
