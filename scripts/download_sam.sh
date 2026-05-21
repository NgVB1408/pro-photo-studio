#!/usr/bin/env bash
# Download SAM 1.0 (facebookresearch/segment-anything) checkpoint.
# Usage: bash scripts/download_sam.sh [vit_b|vit_l|vit_h]
#
# Models:
#   vit_b: 375 MB — CPU OK, recommended cho zoom-crop window mode
#   vit_l: 1.2 GB — GPU 4GB+
#   vit_h: 2.5 GB — GPU 8GB+

set -e
VARIANT="${1:-vit_b}"
CACHE_DIR="$HOME/.cache/sam"
mkdir -p "$CACHE_DIR"

case "$VARIANT" in
    vit_b)
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        FILE="sam_vit_b_01ec64.pth"
        SIZE="375 MB"
        ;;
    vit_l)
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        FILE="sam_vit_l_0b3195.pth"
        SIZE="1.2 GB"
        ;;
    vit_h)
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        FILE="sam_vit_h_4b8939.pth"
        SIZE="2.5 GB"
        ;;
    *)
        echo "❌ Unknown variant: $VARIANT. Use vit_b | vit_l | vit_h"
        exit 1
        ;;
esac

OUT="$CACHE_DIR/$FILE"
if [ -f "$OUT" ]; then
    echo "✓ Already exists: $OUT"
    du -h "$OUT"
    exit 0
fi

echo "📥 Downloading SAM 1.0 $VARIANT ($SIZE) → $OUT"
curl -L -o "$OUT" "$URL"

# Pip install segment_anything
echo "📦 Installing segment-anything from GitHub..."
pip install --quiet "git+https://github.com/facebookresearch/segment-anything.git" \
    || echo "⚠️  pip install fail — install manually nếu cần"

echo ""
echo "✅ Done. Test:"
echo "  pps-wincei-window foto.jpg --zoom --sam-checkpoint $OUT"
