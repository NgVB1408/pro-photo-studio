#!/usr/bin/env bash
# WINCEI v0.3.0 — One-shot installer for VLM (Ollama) + SAM 2 + Python env.
#
# Usage:
#   bash scripts/setup_vlm_sam.sh                    # default qwen2.5vl:7b
#   bash scripts/setup_vlm_sam.sh llama3.2-vision    # alt VLM
#   bash scripts/setup_vlm_sam.sh qwen2.5vl:7b cpu   # force CPU torch
#
# Sau khi chạy xong:
#   ✓ Ollama installed + service running
#   ✓ qwen2.5vl (or chosen) pulled
#   ✓ bds-brain custom model built from Modelfile
#   ✓ SAM 2 pip-installed + sam2_hiera_tiny.pt downloaded
#   ✓ Python venv at .venv/ with all deps
#   ✓ All 4 packages installed in editable mode
#   ✓ Smoke test passed

set -e
set -o pipefail

VLM_BASE="${1:-qwen2.5vl:7b}"
TORCH_VARIANT="${2:-auto}"    # 'auto' | 'cpu' | 'cu121'
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
SAM_CACHE="$HOME/.cache/sam2"

echo "═════════════════════════════════════════════════════════════"
echo "  WINCEI v0.3.0 — FULL STACK INSTALLER"
echo "  Workspace : $WORKSPACE"
echo "  VLM base  : $VLM_BASE"
echo "  Torch     : $TORCH_VARIANT"
echo "═════════════════════════════════════════════════════════════"

# ───────────────────────────────────────────────────────────────────
# STEP 1: Ollama install + service
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 1/6: Ollama install + start service"

if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found, installing..."
    case "$(uname -s)" in
        Linux*|Darwin*)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "  ⚠️  Windows detected — download installer manually:"
            echo "     https://ollama.com/download/OllamaSetup.exe"
            echo "  Then re-run this script."
            exit 1
            ;;
        *)
            echo "  ❌ OS không nhận diện được. Cài Ollama thủ công: https://ollama.com/download"
            exit 1
            ;;
    esac
fi
echo "  ✓ Ollama installed: $(ollama --version 2>&1 | head -1)"

# Start service (background)
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  Starting ollama serve in background..."
    (nohup ollama serve > /tmp/ollama.log 2>&1 &) || true
    for i in {1..15}; do
        sleep 1
        curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && break
    done
fi
curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 \
    && echo "  ✓ Ollama service: http://localhost:11434" \
    || { echo "  ❌ Ollama service không lên. Check /tmp/ollama.log"; exit 1; }

# ───────────────────────────────────────────────────────────────────
# STEP 2: Pull VLM + build bds-brain
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 2/6: Pull $VLM_BASE (~4-7GB, có thể mất 5-15 phút)"
ollama pull "$VLM_BASE"

echo ""
echo "  Building bds-brain custom Modelfile..."
MODELFILE="$WORKSPACE/packages/wincei-masks/Modelfile.bds-brain"
if [ ! -f "$MODELFILE" ]; then
    echo "  ❌ Modelfile not found: $MODELFILE"
    exit 1
fi

TMP_MODELFILE=$(mktemp)
sed "s|^FROM .*|FROM $VLM_BASE|" "$MODELFILE" > "$TMP_MODELFILE"
ollama create bds-brain -f "$TMP_MODELFILE"
rm "$TMP_MODELFILE"
echo "  ✓ bds-brain registered:"
ollama list | grep -E "bds-brain|$VLM_BASE" | sed 's/^/    /'

# ───────────────────────────────────────────────────────────────────
# STEP 3: Python venv + uv
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 3/6: Python venv + uv workspace sync"

cd "$WORKSPACE"

if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    case "$(uname -s)" in
        Linux*|Darwin*)
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
            ;;
    esac
fi
echo "  ✓ uv: $(uv --version 2>&1 | head -1)"

echo "  Syncing workspace (all 4 packages)..."
uv sync --all-packages
echo "  ✓ Workspace synced"

# ───────────────────────────────────────────────────────────────────
# STEP 4: PyTorch + SAM 2
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 4/6: PyTorch + SAM 2"

PYTHON_BIN="$WORKSPACE/.venv/Scripts/python.exe"
[ -f "$PYTHON_BIN" ] || PYTHON_BIN="$WORKSPACE/.venv/bin/python"
[ -f "$PYTHON_BIN" ] || { echo "  ❌ venv python not found"; exit 1; }

# PyTorch variant
case "$TORCH_VARIANT" in
    cu121)
        echo "  Installing torch CUDA 12.1..."
        "$PYTHON_BIN" -m pip install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        echo "  Installing torch CPU..."
        "$PYTHON_BIN" -m pip install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu
        ;;
    auto)
        echo "  torch already installed via uv sync (auto-detected)"
        ;;
esac

# SAM 2
echo "  Installing SAM 2 from GitHub..."
"$PYTHON_BIN" -m pip install --quiet "git+https://github.com/facebookresearch/sam2.git" || {
    echo "  ⚠️  SAM 2 install fail (graceful — pipeline có fallback GrabCut)"
}

# Checkpoint
mkdir -p "$SAM_CACHE"
CKPT="$SAM_CACHE/sam2_hiera_tiny.pt"
if [ ! -f "$CKPT" ]; then
    echo "  Downloading sam2_hiera_tiny.pt (38MB)..."
    curl -L -o "$CKPT" \
         https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
fi
echo "  ✓ SAM 2 checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"

# Pre-load HuggingFace SegFormer cache (eager download)
echo "  Pre-loading SegFormer-B3 weights..."
"$PYTHON_BIN" -c "
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
m = 'nvidia/segformer-b3-finetuned-ade-512-512'
SegformerImageProcessor.from_pretrained(m)
SegformerForSemanticSegmentation.from_pretrained(m)
print('SegFormer-B3 cached')
" 2>&1 | tail -3

# ───────────────────────────────────────────────────────────────────
# STEP 5: Optional dependencies
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 5/6: Optional deps (pymatting + pillow + etc.)"
"$PYTHON_BIN" -m pip install --quiet pymatting tifffile piexif

# ───────────────────────────────────────────────────────────────────
# STEP 6: Smoke test
# ───────────────────────────────────────────────────────────────────
echo ""
echo "▶ STEP 6/6: Smoke test"

"$PYTHON_BIN" -c "
import requests
# Test Ollama bds-brain available
r = requests.get('http://localhost:11434/api/tags')
models = [m['name'] for m in r.json()['models']]
assert any('bds-brain' in m for m in models), f'bds-brain not in {models}'
print('  ✓ bds-brain registered')

# Test all packages
from pps_wincei import process_image
from pps_wincei_hdr import detect_brackets
from pps_wincei_masks import extract_masks, __version__
print(f'  ✓ pps-wincei-masks v{__version__}')
from pps_wincei_masks.preprocess import apply_clahe, undistort_image
from pps_wincei_masks.overlap_resolver import resolve_all_overlaps
from pps_wincei_masks.vlm_client import OllamaVLM, ChainOfThoughtResponse
from pps_wincei_masks.sam_engine import SAMEngine, HIGH_RES_CONFIG
print(f'  ✓ CoT + SAM2 HIGH_RES_CONFIG: points_per_side={HIGH_RES_CONFIG[\"points_per_side\"]}')

# Test API
from pps_wincei_api.server import create_app
app = create_app()
endpoints = [r.path for r in app.routes if hasattr(r, 'methods')]
assert '/api/v1/full-recovery-ceiling' in endpoints
print('  ✓ /api/v1/full-recovery-ceiling endpoint registered')
"

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  ✅ FULL STACK INSTALLED"
echo ""
echo "  Quick test ngay với 1 ảnh BĐS:"
echo ""
echo "  # Engine semantic (no GPU needed):"
echo "  $PYTHON_BIN -m pps_wincei_masks.cli foto.jpg --outputs ./outputs"
echo ""
echo "  # Engine VLM+SAM2 (full recovery ceiling):"
echo "  $PYTHON_BIN -m pps_wincei_masks.cli foto.jpg --outputs ./outputs \\"
echo "      --engine vlm-sam2 --vlm-model bds-brain \\"
echo "      --sam-checkpoint $CKPT"
echo ""
echo "  # API server:"
echo "  $WORKSPACE/.venv/Scripts/pps-wincei-api.exe"
echo "  → http://localhost:8088/docs"
echo ""
echo "  curl -X POST http://localhost:8088/api/v1/full-recovery-ceiling \\"
echo "       -F 'file=@DSC01527.jpg' \\"
echo "       --output ceiling_full_recovery.png"
echo "═════════════════════════════════════════════════════════════"
