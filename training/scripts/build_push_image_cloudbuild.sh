#!/usr/bin/env bash
# Alternative: build via Cloud Build (no local Docker, faster push).
# Cloud Build is included in free tier (120 build-minutes/day).
# Run from repo root.

set -euo pipefail

PROJECT="${GCP_PROJECT:-exalted-splicer-497201-s8}"
REGION="${GCP_REGION:-us-central1}"
REPO="${AR_REPO:-pps-training}"
IMAGE_NAME="${IMAGE_NAME:-qwen-edit}"

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}"

gcloud builds submit \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --tag="${URI}:${SHA}" \
  --dockerfile=training/Dockerfile.vertex \
  .

# Cloud Build tags only the SHA tag; add :latest after the build succeeds.
gcloud artifacts docker tags add \
  "${URI}:${SHA}" \
  "${URI}:latest" \
  --project="${PROJECT}"

echo ""
echo "Done. Pushed:"
echo "  ${URI}:${SHA}"
echo "  ${URI}:latest"
