#!/usr/bin/env bash
# Build FULL SOURCE archive — toàn bộ mã nguồn dự án cho khách.
#
# Khác build_portable_delivery.sh:
#   - Full source (tất cả file trong repo, exclude .venv/.git/build artifacts)
#   - Không chứa weights/cache
#   - Khách tự sync + setup theo SETUP.md
#
# Output: delivery/wincei_SOURCE_v0.3.0_<date>.tar.gz + .zip

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="0.3.0"
DATE=$(date +%Y%m%d)
PKG_NAME="wincei_SOURCE_v${VERSION}_${DATE}"
PKG_DIR="$ROOT/delivery/${PKG_NAME}"
ZIP_PATH="$ROOT/delivery/${PKG_NAME}.zip"
TGZ_PATH="$ROOT/delivery/${PKG_NAME}.tar.gz"

echo "🔨 Building FULL SOURCE archive..."
echo "   Target zip: $ZIP_PATH"
echo "   Target tgz: $TGZ_PATH"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

echo "📂 Copying all source files (exclude build artifacts)..."

# Use git ls-files để có exact list trong tracking, plus untracked không bị ignore
if [ -d ".git" ]; then
    # Tracked files
    git ls-files | while read -r f; do
        target_dir=$(dirname "$PKG_DIR/$f")
        mkdir -p "$target_dir"
        cp "$f" "$PKG_DIR/$f"
    done
else
    # Fallback: copy manually with exclusions
    rsync_or_cp() {
        local src="$1"
        local dst="$2"
        if command -v rsync &>/dev/null; then
            rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
                  --exclude='delivery' --exclude='.git' --exclude='*.pyc' \
                  --exclude='wincei_storage' --exclude='dist' --exclude='build' \
                  "$src/" "$dst/"
        else
            cp -r "$src/." "$dst/"
            find "$dst" -type d \( -name '__pycache__' -o -name '.venv' -o -name 'delivery' \
                                 -o -name '.git' -o -name 'wincei_storage' -o -name 'dist' \
                                 -o -name 'build' -o -name '*.egg-info' \) \
                -exec rm -rf {} + 2>/dev/null || true
            find "$dst" -type f -name '*.pyc' -delete 2>/dev/null || true
        fi
    }
    rsync_or_cp "." "$PKG_DIR"
fi

# Cleanup any remaining junk
find "$PKG_DIR" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$PKG_DIR" -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
find "$PKG_DIR" -type f -name '*.pyc' -delete 2>/dev/null || true
rm -rf "$PKG_DIR/.venv" "$PKG_DIR/.git" "$PKG_DIR/wincei_storage" "$PKG_DIR/delivery" 2>/dev/null || true

# Ensure checklists present
echo "📋 Verify checklists..."
test -f "$PKG_DIR/packages/wincei-masks/USAGE_CHECKLIST.md" && echo "   ✓ USAGE_CHECKLIST.md"
test -f "$PKG_DIR/packages/wincei-masks/UPGRADE_CHECKLIST.md" && echo "   ✓ UPGRADE_CHECKLIST.md"
test -f "$PKG_DIR/packages/wincei-masks/DELIVERY.md" && echo "   ✓ DELIVERY.md"
test -f "$PKG_DIR/packages/wincei-masks/SETUP_VLM_SAM.md" && echo "   ✓ SETUP_VLM_SAM.md"
test -f "$PKG_DIR/packages/wincei-masks/Modelfile.bds-brain" && echo "   ✓ Modelfile.bds-brain"

# Generate root-level handoff README for source archive
cat > "$PKG_DIR/HANDOFF_README.md" <<'EOFHANDOFF'
# 📦 WINCEI v0.3.0 — FULL SOURCE BÀN GIAO

> Toàn bộ mã nguồn dự án + tài liệu vận hành chuyên nghiệp.
> Khách hàng tự setup theo `packages/wincei-masks/SETUP_VLM_SAM.md`.

## 📁 Cấu trúc

```
wincei_SOURCE_v0.3.0_<date>/
├── HANDOFF_README.md              ← Bạn đang đọc
├── README.md                       ← Main project doc (VN)
├── ARCHITECTURE.md
├── RUNBOOK.md
├── SECURITY.md
├── CONTRIBUTING.md
├── LICENSE                         ← Apache-2.0
├── pyproject.toml                  ← uv workspace config
├── uv.lock                         ← Lock file reproducible
├── Dockerfile + docker-compose.yml ← Production deploy
│
├── packages/                       ← 4 Python packages
│   ├── wincei/                     ← Window+ceiling fix
│   │   └── pps_wincei/             (12 modules)
│   ├── wincei-hdr/                 ← HDR Mertens fusion
│   │   ├── pps_wincei_hdr/         (6 modules)
│   │   └── WORKFLOW.md
│   ├── wincei-masks/               ← Smart segmentation v0.3.0
│   │   ├── pps_wincei_masks/       (16 modules)
│   │   ├── DELIVERY.md             ← Tài liệu khách
│   │   ├── USAGE_CHECKLIST.md      ← Checklist sử dụng hàng ngày
│   │   ├── UPGRADE_CHECKLIST.md    ← Checklist nâng cấp
│   │   ├── SETUP_VLM_SAM.md        ← Cài Ollama + SAM 2
│   │   └── Modelfile.bds-brain     ← Custom Ollama model
│   └── wincei-api/                 ← FastAPI REST 9 endpoints
│       └── pps_wincei_api/         (server + 7 routers + workers + UI)
│
├── scripts/                        ← Automation
│   ├── setup_vlm_sam.sh            ← One-shot installer
│   ├── full_recovery_ceiling.py    ← Standalone CLI
│   ├── build_portable_delivery.sh  ← Build portable zip
│   ├── build_full_source.sh        ← Build full source archive
│   └── release_v0.3.0.sh           ← Git + GitHub release
│
└── docs/                           ← Showcase + meta documentation
```

## 🚀 BÀN GIAO 3 CÁCH

### Cách 1 — Source thuần (cho team dev)
1. Giải nén archive
2. Đọc `packages/wincei-masks/SETUP_VLM_SAM.md`
3. Chạy `bash scripts/setup_vlm_sam.sh` (auto installer)
4. `pps-wincei-api` → `http://localhost:8000/`

### Cách 2 — Docker production
```bash
cd wincei_SOURCE_v0.3.0_*/
docker compose up -d
# → http://localhost:8000/
```

### Cách 3 — Pip install local
```bash
pip install ./packages/wincei
pip install ./packages/wincei-hdr
pip install ./packages/wincei-masks
pip install ./packages/wincei-api
pps-wincei-api
```

## 📚 TÀI LIỆU CHÍNH (đọc theo thứ tự)

1. `packages/wincei-masks/DELIVERY.md` — Tổng quan + workflow
2. `packages/wincei-masks/USAGE_CHECKLIST.md` — Vận hành hàng ngày
3. `packages/wincei-masks/UPGRADE_CHECKLIST.md` — Nâng cấp
4. `packages/wincei-masks/SETUP_VLM_SAM.md` — Cài VLM + SAM 2
5. `packages/wincei-masks/README.md` — API reference smart segmentation
6. `packages/wincei-hdr/WORKFLOW.md` — Pipeline HDR
7. `packages/wincei-api/README.md` — REST API reference

## 🎯 ENDPOINTS API (port 8000)

| Endpoint | Mô tả |
|---|---|
| `GET /` | ⭐ Web UI drag-drop (Việt hoá) |
| `GET /docs` | Swagger UI |
| `GET /redoc` | ReDoc |
| `POST /api/v1/full-recovery-ceiling` | ⭐ VLM + SAM2 + Sobel hybrid |
| `POST /api/v1/segment-masks` | Smart segmentation 9 masks + AI eval |
| `POST /api/v1/detect-regions` | JSON bbox normalized [0..1000] |
| `POST /api/v1/hdr-fuse` | Mertens HDR bracket fusion |
| `POST /api/v1/window-ceiling` | Fix cửa sổ + trần ám màu |
| `GET /api/v1/jobs` | List jobs |
| `GET /api/v1/jobs/{id}` | Job status |
| `GET /api/v1/jobs/{id}/download` | Download zip output |
| `GET /api/v1/health` | Health + GPU + versions |

## 🧪 ĐÃ KIỂM ĐỊNH

| Metric | Value | Status |
|---|---|---|
| Overall verdict (DSC01527 modern interior) | 0.852 | ✅ PASS |
| Wall coverage | 51.07% | ✅ |
| Floor edge alignment | 0.99 | ✅ Near perfect |
| Opening (door+window+sky) score | 0.89 | ✅ |
| Bracket detection accuracy | 131/131 | ✅ 100% |
| HDR blown highlight recovery | 29-81% | ✅ |
| Casing leakage cap | < 6% | ✅ |

## 📞 HỖ TRỢ

- **Email**: luvdraco88@gmail.com
- **Zalo**: 0876 254 585
- **GitHub**: https://github.com/NgVB1408/pro-photo-studio
- **GitHub Issues**: https://github.com/NgVB1408/pro-photo-studio/issues
- **License**: Apache-2.0
EOFHANDOFF

# ─────────────────── Build archives ───────────────────
echo ""
echo "🗜️ Building archives..."

cd "$ROOT/delivery"
rm -f "${PKG_NAME}.zip" "${PKG_NAME}.tar.gz"

# ZIP
python - <<PYEOF
import shutil
shutil.make_archive("${PKG_NAME}", "zip", "${PKG_NAME}")
PYEOF
ZIP_SIZE=$(du -h "${PKG_NAME}.zip" | cut -f1)
echo "   ✓ ZIP : ${PKG_NAME}.zip ($ZIP_SIZE)"

# TAR.GZ (Linux/Mac friendly)
if command -v tar &> /dev/null; then
    tar -czf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
    TGZ_SIZE=$(du -h "${PKG_NAME}.tar.gz" | cut -f1)
    echo "   ✓ TGZ : ${PKG_NAME}.tar.gz ($TGZ_SIZE)"
fi

# Count files
N_FILES=$(find "$PKG_DIR" -type f | wc -l)

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  ✅ FULL SOURCE ARCHIVE BUILT"
echo "  Files in archive: $N_FILES"
echo "  ZIP : $ZIP_PATH"
echo "  TGZ : $TGZ_PATH"
echo "═════════════════════════════════════════════════════════════"
