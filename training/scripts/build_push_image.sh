#!/usr/bin/env bash
# Build + push the Vertex training container.
# Run from repo root.
#
# Tagged by git SHA + 'latest'. Image is large (~6 GB) — first push from
# Cloud Shell will be slow; for repeated iterations use Cloud Build instead
# (faster network, no local docker required).

set -euo pipefail

PROJECT="${GCP_PROJECT:-exalted-splicer-497201-s8}"
REGION="${GCP_REGION:-us-central1}"
REPO="${AR_REPO:-pps-training}"
IMAGE_NAME="${IMAGE_NAME:-qwen-edit}"

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"
URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}"

echo "Building ${URI}:${SHA}"
docker build \
  -f training/Dockerfile.vertex \
  -t "${URI}:${SHA}" \
  -t "${URI}:latest" \
  .

echo ""
echo "Authenticating Docker to Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo ""
echo "Pushing ${URI}:${SHA} + ${URI}:latest"
docker push "${URI}:${SHA}"
docker push "${URI}:latest"

echo ""
echo "Done."
