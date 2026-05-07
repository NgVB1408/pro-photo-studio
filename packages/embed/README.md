# pps-embed

Vector store cho **photo gene** + **algorithm gene** trên Qdrant; metadata
trên Postgres. Mục đích: khi có ảnh BĐS mới, retrieve top-K ảnh "đã được
chỉnh đẹp" tương tự + parameter set tương ứng → reuse params thay vì tinh
chỉnh thủ công lại.

## Cài đặt

```powershell
.\.venv-agents\Scripts\python.exe -m pip install --ignore-requires-python -e packages/embed
# Optional: CLIP cho 512d feature thêm
.\.venv-agents\Scripts\python.exe -m pip install "open-clip-torch>=2.24" "torch>=2.1"
```

Env cần:

```
QDRANT_URL=https://<oracle-vm-ip>:6333
QDRANT_API_KEY=<random-key>
DATABASE_URL=postgresql+asyncpg://user:pass@<oracle-atp-host>/pps
```

## Chạy Qdrant trên Oracle Cloud VM

```powershell
ssh ubuntu@<oracle-vm-ip>
docker compose -f deploy/docker/qdrant/docker-compose.yml up -d
docker compose logs -f qdrant
```

(Mở port 6333 trong Security List của VCN. Bật TLS bằng Caddy hoặc Traefik
trước khi expose ra internet.)

## CLI

```powershell
# Index 1 ảnh
.\.venv-agents\Scripts\python.exe -m pps_embed index-photo .\image.jpg

# Tìm top-5 ảnh tương tự
.\.venv-agents\Scripts\python.exe -m pps_embed query .\new_photo.jpg -k 5

# Index 1 parameter set
.\.venv-agents\Scripts\python.exe -m pps_embed index-algo .\villa_params.json --name villa_luxury

# Áp Postgres migration
.\.venv-agents\Scripts\python.exe -m pps_embed migrate
```

## API

```python
from pps_embed import EmbedStore, photo_embedding

store = EmbedStore(url="https://qdrant.example:6333", api_key="...")
await store.ensure_collections()
pid = await store.upsert_photo(img_bgr, payload={"source": "kitchen.jpg"})
hits = await store.query_similar_photos(query_img, k=5)
for h in hits:
    print(h.score, h.payload)
```

## Embedding chi tiết

| Component | Dim | Implementation |
|---|---|---|
| pHash | 64 | 8×8 DCT hash của ảnh resize 32×32 |
| LAB histogram | 96 | 32 bins / channel L,a,b — normalised |
| Saliency stats | 16 | 4 quadrants × (mean, std, min, max) — dùng `pps_core.saliency_sharpen` |
| OpenCLIP (optional) | 512 | ViT-B/32 LAION-2B, normalised |

Total: 176d (default) hoặc 688d (with CLIP).

Algorithm embedding (256d): canonicalise params JSON → SHA-256 seed →
key-frequency histogram (1024 buckets) → Gaussian Random Projection.
Deterministic — same params luôn cùng vector.

## Test

```powershell
.\.venv-agents\Scripts\python.exe -m pytest packages/embed/tests -ra
```

Test dùng Qdrant `:memory:` mode + SQLAlchemy schema in-memory — không cần
Postgres / Qdrant thật.

## Schema

```
photos               (id PK, width, height, source, owner, ...)
algorithms           (id PK, name, params_json, ...)
embeddings           (qdrant_point_id PK, photo_id FK, algorithm_id FK, dim, model)
audit_log            (job_id, photo_id, algorithm_id, dataset_provenance JSON, scores JSON)
dataset_entries      (dataset, repo_id, split, row_idx, photo_id, license_tag)
```

`dataset_entries` ghi provenance từ pps-data: mỗi row FiveK/LSD/SUN dùng để
train được track lại ở đây — phục vụ FiveK research-only license.

## Mở rộng

- Thêm pgvector vào Postgres để khỏi cần Qdrant nếu vol vector < 100k.
- Thêm `query_similar_with_rerank(image, thumb_loader=...)` để re-rank
  bằng PSNR/SSIM thay vì chỉ vector distance.
- Hot-swap CLIP backbone: hiện ViT-B/32; cân nhắc EVA-CLIP hoặc DINOv2 cho
  recall cao hơn trên BĐS.
