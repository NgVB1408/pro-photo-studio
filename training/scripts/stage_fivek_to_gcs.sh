#!/usr/bin/env bash
# Stream FiveK from HF to GCS bucket as a GCS mirror.
#
# Run from a machine with disk space + bandwidth — Cloud Shell home is only
# 5 GB ephemeral, FiveK is ~50 GB. Use one of:
#   - your laptop (will take hours on home connection)
#   - a Vertex AI Workbench notebook (us-central1, GCS direct, ~10-15 min)
#   - a Cloud Build job (limited disk; not recommended)
#   - a one-shot e2-standard-16 GCE VM in us-central1 (cheapest: ~$1 for the
#     whole staging job; remember to delete the VM after)
#
# This script assumes you're already on such a machine. It does NOT install
# pps-data — install it first: `uv pip install -e packages/data`.

set -euo pipefail

BUCKET="${GCS_BUCKET:-pps-training-exalted-splicer-497201-s8}"
EXPERT="${EXPERT:-c}"
SPLIT="${SPLIT:-train}"
MAX_ROWS="${MAX_ROWS:-}"

DEST="gs://${BUCKET}/datasets/fivek/${EXPERT}/${SPLIT}"

echo "Staging FiveK expert=${EXPERT} split=${SPLIT} → ${DEST}"
echo ""

# Require HF token — must be the FRESH token, not any of the 3 revoked ones.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN env var not set. Export a fresh read-scope token first." >&2
  echo "Do NOT use any of the 3 revoked tokens (see SECURITY.md)." >&2
  exit 1
fi

# Sanity check: pps-data installed?
if ! python -c "import pps_data" 2>/dev/null; then
  echo "ERROR: pps_data not importable. Run from repo root with the workspace installed:" >&2
  echo "  uv sync && source .venv/bin/activate" >&2
  exit 1
fi

# Stream rows, write each as <id>_input.jpg / <id>_target.jpg locally, batch upload.
python - <<PY
import os
import io
import sys
from pathlib import Path
from pps_data import stream_fivek

expert = os.environ["EXPERT"]
split = os.environ["SPLIT"]
max_rows = int(os.environ["MAX_ROWS"]) if os.environ.get("MAX_ROWS") else None

tmp = Path("/tmp/fivek-stage")
tmp.mkdir(exist_ok=True)

ds = stream_fivek(expert=expert, split=split)
for i, row in enumerate(ds):
    if max_rows is not None and i >= max_rows:
        break
    inp = row.get("input_image")
    tgt = row.get(f"expert_{expert}")
    if inp is None or tgt is None:
        continue
    # PIL image → JPEG bytes
    (tmp / f"{i:05d}_input.jpg").write_bytes(_jpeg(inp))
    (tmp / f"{i:05d}_target.jpg").write_bytes(_jpeg(tgt))
    if (i + 1) % 100 == 0:
        print(f"  staged {i+1} pairs locally", file=sys.stderr)

print(f"Total staged: {i+1} pairs", file=sys.stderr)


def _jpeg(img) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()
PY

echo ""
echo "Uploading to ${DEST}"
gcloud storage cp -r /tmp/fivek-stage/* "${DEST}/" --no-clobber

echo ""
echo "Done. Listing:"
gcloud storage ls "${DEST}/" | head -5
