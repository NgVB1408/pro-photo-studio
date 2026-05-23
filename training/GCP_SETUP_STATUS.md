# GCP Setup Status — Training

Generated 2026-05-23 from Cloud Shell (luvdraco88@gmail.com). Setup-only — no GPU spawned, no Vertex job submitted, no container built. Cost incurred so far: $0 (bucket empty, AR empty, secrets empty).

## Resources provisioned

| Resource | Value |
|---|---|
| `project_id` | `exalted-splicer-497201-s8` |
| `project_number` | `746312392183` |
| `region` | `us-central1` |
| `billing_account` | `011540-3F44A4-3311AE` (active) |
| `bucket_uri` | `gs://pps-training-exalted-splicer-497201-s8` (uniform access, public-access-prevented) |
| `service_account` | `pps-training@exalted-splicer-497201-s8.iam.gserviceaccount.com` |
| `artifact_registry_repo` | `us-central1-docker.pkg.dev/exalted-splicer-497201-s8/pps-training` (docker format) |
| `secret_hf_read` | `projects/746312392183/secrets/HF_TOKEN_READ` (stub, no version yet) |
| `secret_hf_write` | `projects/746312392183/secrets/HF_TOKEN_WRITE` (stub, no version yet) |

## APIs enabled

- `compute.googleapis.com` (was already on)
- `storage.googleapis.com` (was already on)
- `aiplatform.googleapis.com` (enabled by this setup)
- `artifactregistry.googleapis.com` (enabled by this setup)
- `secretmanager.googleapis.com` (enabled by this setup)

## IAM bindings on `pps-training` SA

Project-level:
- `roles/aiplatform.user`
- `roles/artifactregistry.reader`
- `roles/secretmanager.secretAccessor`

Bucket-only (`gs://pps-training-...` only — NOT project-wide):
- `roles/storage.objectAdmin`

On the SA resource itself (so user can impersonate for testing):
- `roles/iam.serviceAccountTokenCreator` → `user:luvdraco88@gmail.com`

Rationale: principle of least privilege. SA can run Vertex jobs, pull container images, read secrets, and read/write its own bucket — nothing else. No `storage.admin`, no `secretmanager.admin`, no `compute.admin`.

Tested via impersonation from Cloud Shell: SA can list, write, read, and delete objects in its bucket. ✅

## GPU quota — BLOCKER for real fine-tune

| GPU | us-central1 limit | Notes |
|---|---|---|
| NVIDIA K80 | 1 | EOL (sm_3.7), modern PyTorch incompatible. Ignore. |
| NVIDIA P100 16GB | 1 | Workable for LoRA rank ≤ 8, batch 1. ~$1.46/h on-demand. |
| NVIDIA V100 16GB | 1 | Best available without quota increase. ~$2.48/h on-demand. **Target for first real run.** |
| NVIDIA P4 8GB | 1 | Too small for Qwen-Image-Edit. |
| NVIDIA T4 | 0 | Quota request required. |
| NVIDIA L4 / A100 / H100 | 0 (no entry) | Quota request required; Google typically denies new personal accounts. |

Quota increase request URL:
https://console.cloud.google.com/iam-admin/quotas?project=exalted-splicer-497201-s8

Filter by "L4" or "A100" → us-central1 → Edit Quotas → request 1 unit. Approval 1-7 days; some denied outright for personal accounts.

## Artifacts written to this branch

All under `training/`:

| File | Purpose |
|---|---|
| `Dockerfile.vertex` | Custom training container based on Vertex AI prebuilt pytorch-gpu.2-3.py310. Installs workspace + training extras. |
| `configs/vertex_train.yaml` | Vertex AI CustomJob spec. Targets V100 in us-central1. Wires SA + bucket + secret env vars. Do NOT submit until gates clear (see file header). |
| `scripts/build_push_image.sh` | Local docker build + push to Artifact Registry. |
| `scripts/build_push_image_cloudbuild.sh` | Alternative: build via Cloud Build (no local docker, faster from Cloud Shell). |
| `scripts/stage_fivek_to_gcs.sh` | Stream FiveK from HF to GCS mirror. Must run from a machine with disk + bandwidth — Cloud Shell ephemeral 5 GB is not enough for ~50 GB dataset. |

## Gates carried forward

- ✅ Tokens (HF×2, Dropbox) confirmed revoked by user on 2026-05-23 → Secret Manager can be used freely.
- ⏳ FiveK weights remain research-only — no commercial release of any fine-tuned LoRA.
- ⏳ Fresh `HF_TOKEN` with read scope (for dataset stream) — user has not yet created. Stage into `HF_TOKEN_READ` when ready.
- ⏳ Fresh `HF_TOKEN` with write scope (for checkpoint push) — same.
- ⛔ `training/finetune_qwen_edit.py` real training loop is **not implemented** — file is currently a scaffold (line 181 explicitly says "intentionally not wired yet"). Submitting the Vertex job in its current state will exit 0 without training, wasting V100 time.

## Strict order of next steps (do not skip)

1. **Implement** the diffusers + peft + accelerate LoRA loop in `training/finetune_qwen_edit.py`. Behind feature flags per the file's TODO. Test with `--dry-run` first.
2. **Create** fresh HF tokens (separate read + write) on HuggingFace.
3. **Stage** them into Secret Manager:
   ```
   echo -n "hf_read_xxx"  | gcloud secrets versions add HF_TOKEN_READ  --data-file=- --project=exalted-splicer-497201-s8
   echo -n "hf_write_xxx" | gcloud secrets versions add HF_TOKEN_WRITE --data-file=- --project=exalted-splicer-497201-s8
   ```
4. **Stage** FiveK dataset to `gs://pps-training-.../datasets/fivek/` via `training/scripts/stage_fivek_to_gcs.sh` (from a non-Cloud-Shell machine).
5. **Build + push** container:
   ```
   bash training/scripts/build_push_image_cloudbuild.sh
   ```
6. **Submit** first real Vertex job as a smoke test (override `train.max_steps: 100` in config) before paying for 4000 steps:
   ```
   HF_OUTPUT_REPO=<your-private-org>/pps-qwen-edit-v1-test \
   gcloud ai custom-jobs create \
     --region=us-central1 \
     --display-name=pps-qwen-edit-smoke-$(date +%Y%m%d-%H%M) \
     --service-account=pps-training@exalted-splicer-497201-s8.iam.gserviceaccount.com \
     --config=training/configs/vertex_train.yaml
   ```
7. **Submit** real run after smoke passes.

## Open questions

- Submit L4 / A100 quota request now (long lead time) or stick with V100?
- W&B logging — provision `WANDB_API_KEY` secret + flip `WANDB_DISABLED=false` in `vertex_train.yaml`?
- Custom training container vs Vertex prebuilt — current Dockerfile uses prebuilt as base, which is the right balance.
