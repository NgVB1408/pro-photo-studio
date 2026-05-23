#!/usr/bin/env bash
# Cloud Shell bootstrap — run from Cloud Shell after `git pull origin worktree-gcp-training-setup`.
#
# Checks every gate the Vertex job needs before the build+submit:
#   - GCP project active
#   - Required APIs enabled
#   - Service account exists with right IAM
#   - HF_TOKEN_READ + HF_TOKEN_WRITE secrets have versions
#   - HF private repo accessible
#   - Artifact Registry repo exists
#   - GCS bucket reachable
#
# Exits non-zero on first failed gate so we never burn V100 time on a misconfigured run.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-exalted-splicer-497201-s8}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://pps-training-${PROJECT_ID}}"
SA="${SA:-pps-training@${PROJECT_ID}.iam.gserviceaccount.com}"
AR_REPO="${AR_REPO:-us-central1-docker.pkg.dev/${PROJECT_ID}/pps-training}"
HF_USER="${HF_USER:-Nvb1408}"
HF_REPO_TEST="${HF_REPO_TEST:-${HF_USER}/pps-qwen-edit-v1-test}"
HF_REPO_PROD="${HF_REPO_PROD:-${HF_USER}/pps-qwen-edit-v1}"

ok()   { printf "\033[32m[ OK ]\033[0m %s\n" "$*"; }
fail() { printf "\033[31m[FAIL]\033[0m %s\n" "$*"; exit 1; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }

echo "=== Cloud Shell bootstrap: ${PROJECT_ID} ==="

# 1. Active project
[ "$(gcloud config get-value project 2>/dev/null)" = "${PROJECT_ID}" ] \
  || gcloud config set project "${PROJECT_ID}" >/dev/null
ok "project ${PROJECT_ID} active"

# 2. APIs
for api in aiplatform.googleapis.com artifactregistry.googleapis.com \
           secretmanager.googleapis.com cloudbuild.googleapis.com; do
  if gcloud services list --enabled --format="value(NAME)" | grep -qx "${api}"; then
    ok "API ${api}"
  else
    warn "enabling ${api}"
    gcloud services enable "${api}"
  fi
done

# 3. Service account
gcloud iam service-accounts describe "${SA}" >/dev/null 2>&1 \
  && ok "SA ${SA}" \
  || fail "SA missing: ${SA}"

# 4. Secrets — must have at least version 1
for secret in HF_TOKEN_READ HF_TOKEN_WRITE; do
  v=$(gcloud secrets versions list "${secret}" --project="${PROJECT_ID}" \
        --filter="state=ENABLED" --format="value(name)" 2>/dev/null | head -1)
  if [ -n "${v}" ]; then
    ok "secret ${secret} (version ${v})"
  else
    fail "secret ${secret} has no enabled version — run: echo -n hf_xxx | gcloud secrets versions add ${secret} --data-file=- --project=${PROJECT_ID}"
  fi
done

# 5. HF private repo accessible (uses staged READ token)
HF_TOKEN=$(gcloud secrets versions access latest --secret=HF_TOKEN_READ --project="${PROJECT_ID}")
for repo in "${HF_REPO_TEST}" "${HF_REPO_PROD}"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/api/models/${repo}")
  if [ "${code}" = "200" ]; then
    ok "HF repo ${repo} accessible"
  else
    fail "HF repo ${repo} returned ${code} — create via huggingface_hub.create_repo"
  fi
done
unset HF_TOKEN

# 6. Artifact Registry
gcloud artifacts repositories describe pps-training \
  --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1 \
  && ok "AR repo pps-training" \
  || fail "AR repo missing: ${AR_REPO}"

# 7. Bucket
gsutil ls -b "${BUCKET}" >/dev/null 2>&1 \
  && ok "bucket ${BUCKET}" \
  || fail "bucket missing: ${BUCKET}"

# 8. GPU quota — V100 in us-central1 must be >= 1
v100=$(gcloud compute regions describe "${REGION}" \
        --format="value(quotas.filter('metric:NVIDIA_V100_GPUS').limit)" 2>/dev/null || echo "0")
if [ "${v100%.*}" -ge 1 ] 2>/dev/null; then
  ok "V100 quota = ${v100}"
else
  warn "V100 quota appears 0 — submit will fail with QUOTA_EXCEEDED"
fi

echo ""
echo "=== All gates pass. Next steps: ==="
cat <<EOF

# Build + push container (Cloud Build, no local docker required):
bash training/scripts/build_push_image_cloudbuild.sh

# Submit SMOKE job (max_steps=100, ~5-10 min on V100, ~\$0.40):
gcloud ai custom-jobs create \\
  --region=${REGION} \\
  --display-name=pps-qwen-edit-smoke-\$(date +%Y%m%d-%H%M) \\
  --service-account=${SA} \\
  --config=training/configs/vertex_train_smoke.yaml \\
  --args=--output-repo,${HF_REPO_TEST}

# Watch logs:
gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION}

# Once smoke pushes to HF successfully → submit PROD (max_steps=4000, ~3-4h, ~\$10):
gcloud ai custom-jobs create \\
  --region=${REGION} \\
  --display-name=pps-qwen-edit-prod-\$(date +%Y%m%d-%H%M) \\
  --service-account=${SA} \\
  --config=training/configs/vertex_train.yaml \\
  --args=--output-repo,${HF_REPO_PROD}

EOF
