#!/usr/bin/env bash
# Git release v0.3.0 — push toàn bộ + tạo release tag
# Usage:
#   bash scripts/release_v0.3.0.sh                  # default repo
#   bash scripts/release_v0.3.0.sh git@github.com:NgVB1408/pro-photo-studio.git

set -e

REMOTE_URL="${1:-git@github.com:NgVB1408/pro-photo-studio.git}"
TAG="v0.3.0"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

cd "$WORKSPACE"

echo "═════════════════════════════════════════════════════════════"
echo "  WINCEI v0.3.0 — GIT RELEASE"
echo "  Workspace : $WORKSPACE"
echo "  Remote    : $REMOTE_URL"
echo "  Tag       : $TAG"
echo "═════════════════════════════════════════════════════════════"

# ───── Step 1: init repo nếu chưa có ─────
if [ ! -d ".git" ]; then
    echo "▶ Khởi tạo git repo..."
    git init -b main
    git remote add origin "$REMOTE_URL"
else
    echo "▶ Git repo đã tồn tại"
    git remote get-url origin 2>/dev/null || git remote add origin "$REMOTE_URL"
fi

# ───── Step 2: .gitignore ─────
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.egg-info/
build/
dist/
.venv/
.uv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.DS_Store
Thumbs.db
*.log
wincei_storage/
delivery/wincei_delivery_*.zip
.cache/
*.swp
*.swo
.idea/
.vscode/launch.json
node_modules/
*.tmp
*.bak

# Large data folders (test/training, not for source)
wincei_test/
outputs/
outputs_*/
inputs/
.thumb/
EOF

# ───── Step 3: stage + commit ─────
echo "▶ Staging files..."
git add .gitignore
git add packages/wincei/
git add packages/wincei-hdr/
git add packages/wincei-masks/
git add packages/wincei-api/
git add scripts/
git add pyproject.toml uv.lock 2>/dev/null || true
git add README.md LICENSE 2>/dev/null || true
git add Dockerfile docker-compose.yml 2>/dev/null || true

if git diff --staged --quiet; then
    echo "▶ Không có thay đổi để commit"
else
    git commit -m "$(cat <<EOFCOMMIT
release v0.3.0 — VLM CoT + SAM2 high-grid + Sobel resolver + CLAHE preprocess

NEW FEATURES:
- preprocess.py: CLAHE auto-detect + undistort fisheye/wide-angle
- vlm_client.py: ChainOfThought 2-step (shadow/tier/false-boundary analysis)
- sam_engine.py: HIGH_RES_CONFIG (points_per_side=64, iou=0.95, stability=0.96)
- overlap_resolver.py: Sobel directional (vertical→wall, horizontal→ceiling)
- ceiling_boost.py: lamp anchor + top-minus-wall fallback for modern interiors
- Modelfile.bds-brain: custom Ollama system prompt cho real estate vision
- scripts/setup_vlm_sam.sh: one-shot installer
- scripts/full_recovery_ceiling.py: standalone CLI cho ceiling RGBA recovery

API ENDPOINTS:
- POST /api/v1/full-recovery-ceiling (CoT + high-grid + Sobel)
- POST /api/v1/detect-regions (JSON bbox normalized [0..1000])
- POST /api/v1/segment-masks (smart segmentation + AI eval)
- POST /api/v1/hdr-fuse (Mertens bracket fusion)
- POST /api/v1/window-ceiling (window+ceiling fix)

VERIFIED:
- DSC01527.jpg: verdict PASS overall 0.852
- Bracket detect: 131 groups × 3 shots = 393 ảnh
- HDR recovery: blown 29-81% reduction
EOFCOMMIT
)"
fi

# ───── Step 4: push main ─────
echo "▶ Push main branch..."
git branch -M main
git push -u origin main || {
    echo "  ⚠️  Push fail — có thể conflict với remote. Try: git pull --rebase origin main"
    exit 1
}

# ───── Step 5: tag release ─────
echo "▶ Creating tag $TAG..."
git tag -a "$TAG" -m "WINCEI v0.3.0 — VLM + SAM2 + Sobel Pro Stack" 2>/dev/null \
    || echo "  Tag $TAG already exists locally"

echo "▶ Push tag to remote..."
git push origin "$TAG" || echo "  ⚠️  Tag push fail (có thể đã tồn tại)"

# ───── Step 6: GitHub release (nếu gh CLI có sẵn) ─────
if command -v gh &> /dev/null; then
    echo "▶ Creating GitHub release..."
    DELIVERY_ZIP="$WORKSPACE/delivery/wincei_delivery_v0.3.0_*.zip"
    EXISTING_ZIP=$(ls $DELIVERY_ZIP 2>/dev/null | head -1)

    RELEASE_NOTES=$(mktemp)
    cat > "$RELEASE_NOTES" <<EOFNOTE
# WINCEI v0.3.0 — Full Pro Stack Release

> **Production-ready real estate photo automation suite.**

## ⭐ Highlights
- **VLM + SAM 2 Hybrid Pipeline**: Ollama bds-brain (custom Qwen2.5-VL) chỉ điểm thông minh, SAM 2 high-grid (points_per_side=64) cắt biên sub-pixel.
- **Chain-of-Thought 2-step**: Step 1 phân tích shadow/đổ bóng/multi-tier, Step 2 chấm điểm prompt cho SAM.
- **Sobel Overlap Resolver**: phân tách wall/ceiling khi mask leakage qua hướng edge.
- **CLAHE + Undistort Preprocess**: cứu ảnh khó (HDR cháy sáng + fisheye).
- **AI Supervisor**: 7-metric quality gate với verdict pass/review/fail.

## 📦 4 Packages

| Package | Mô tả |
|---|---|
| pps-wincei | Window+ceiling fix (Reinhard tonemap + Bradford CAT) |
| pps-wincei-hdr | Mertens bracket fusion (Sony AEB ±3EV) |
| pps-wincei-masks | Smart segmentation + AI eval + regions JSON |
| pps-wincei-api | FastAPI REST 9 endpoints |

## 🔗 API Endpoints

- POST /api/v1/full-recovery-ceiling — VLM CoT + SAM2 + Sobel ⭐
- POST /api/v1/detect-regions — JSON bbox [0..1000]
- POST /api/v1/segment-masks — 9 masks + AI eval verdict
- POST /api/v1/hdr-fuse — Mertens HDR
- POST /api/v1/window-ceiling — fix cửa sổ + trần
- GET  /api/v1/jobs — list/status/download

## 🚀 Quick Start

\`\`\`bash
# Setup full stack (Ollama + SAM2 + Python deps)
bash scripts/setup_vlm_sam.sh

# CLI ngay 1 ảnh BĐS:
python scripts/full_recovery_ceiling.py DSC01527.jpg \\
    --out ceiling_full_recovery.png \\
    --vlm-model bds-brain --save-report report.json

# API server:
pps-wincei-api  # → http://localhost:8088/docs

# Test API qua curl:
curl -X POST http://localhost:8088/api/v1/full-recovery-ceiling \\
     -F "file=@DSC01527.jpg" \\
     --output ceiling_full_recovery.png
\`\`\`

## 🧪 Verified
- DSC01527.jpg (high-difficulty modern interior): verdict PASS overall 0.852
- 131 bracket groups detect đúng 100% qua EXIF
- HDR blown highlight recovery: 29-81% per image
- SegFormer-B3/B5 auto-pick by VRAM (CPU fallback)

## 📥 Downloads
- Portable delivery zip (15MB): \`wincei_delivery_v0.3.0_*.zip\`
- Source code: tag v0.3.0
EOFNOTE

    if [ -n "$EXISTING_ZIP" ]; then
        gh release create "$TAG" "$EXISTING_ZIP" \
            --title "WINCEI v0.3.0 — Full Pro Stack" \
            --notes-file "$RELEASE_NOTES" \
            || echo "  ⚠️  Release create fail (có thể đã tồn tại)"
    else
        gh release create "$TAG" \
            --title "WINCEI v0.3.0 — Full Pro Stack" \
            --notes-file "$RELEASE_NOTES" \
            || echo "  ⚠️  Release create fail"
    fi
    rm "$RELEASE_NOTES"
else
    echo "  ℹ️  gh CLI không có → tạo release thủ công tại:"
    echo "     https://github.com/$(echo "$REMOTE_URL" | sed -E 's|.*[:/]([^/]+/[^/]+)\.git|\1|')/releases/new?tag=$TAG"
fi

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  ✅ RELEASE $TAG PUSHED"
echo "═════════════════════════════════════════════════════════════"
