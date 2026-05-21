# pps-wincei-api v0.1

> FastAPI REST service gói 3 dự án: `pps-wincei` (window+ceiling) + `pps-wincei-hdr` (HDR bracket fusion) + `pps-wincei-masks` (smart segmentation + AI QC).

## Khởi động

```bash
pps-wincei-api
# → http://0.0.0.0:8088
# Swagger UI: http://localhost:8088/docs
# ReDoc    : http://localhost:8088/redoc
```

## Env vars

```bash
WINCEI_STORAGE=./wincei_storage    # uploads + outputs + jobs JSON
WINCEI_MAX_UPLOAD_MB=200
WINCEI_HOST=0.0.0.0
WINCEI_PORT=8088
WINCEI_API_KEY=                     # rỗng = no auth
WINCEI_CORS=*
WINCEI_MOCK_DEFAULT=false
```

## Endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/api/v1/health` | health + GPU + version |
| POST | `/api/v1/window-ceiling` | fix cửa sổ blown + trần ám màu |
| POST | `/api/v1/hdr-fuse` | Mertens HDR fusion từ N ảnh bracket |
| POST | `/api/v1/segment-masks` | phân vùng + phào chỉ + AI QC |
| GET | `/api/v1/jobs` | list recent jobs |
| GET | `/api/v1/jobs/{id}` | job status |
| GET | `/api/v1/jobs/{id}/download` | download zip output |

## Mock mode

Mọi endpoint POST có flag `mock=true` → trả stub JSON ngay lập tức cho khách integration test, không tốn CPU:

```bash
curl -X POST http://localhost:8088/api/v1/segment-masks \
  -F "files=@foto.jpg" \
  -F "mock=true"
```

Response:

```json
{
  "mode": "sync",
  "mock": true,
  "eval": {
    "verdict": "pass",
    "overall_score": 0.88,
    "per_mask": {
      "wall":    {"coverage": 0.517, "verdict": "pass"},
      "floor":   {"coverage": 0.086, "verdict": "pass"},
      "ceiling": {"coverage": 0.014, "verdict": "pass"},
      "opening": {"coverage": 0.124, "verdict": "pass"},
      ...
    }
  }
}
```

## Workflow async (khuyến nghị production)

```bash
# 1. Submit job
JOB=$(curl -s -X POST http://localhost:8088/api/v1/segment-masks \
  -F "files=@DSC01527.jpg" \
  -F "refine_edges=true" \
  -F "detect_molding=true" | jq -r .job_id)

# 2. Poll status
while true; do
  S=$(curl -s http://localhost:8088/api/v1/jobs/$JOB | jq -r .status)
  echo "status: $S"
  [ "$S" = "completed" ] && break
  [ "$S" = "failed" ] && break
  sleep 5
done

# 3. Download
curl -O http://localhost:8088/api/v1/jobs/$JOB/download
```

## Workflow sync (chỉ cho window-ceiling đơn ảnh)

```bash
curl -X POST http://localhost:8088/api/v1/window-ceiling \
  -F "file=@foto.jpg" \
  -F "mode=sync"
```

Block đến khi xong, trả response với output_url + eval.

## Docker

```dockerfile
FROM python:3.11-slim
RUN pip install pps-wincei-api
EXPOSE 8088
CMD ["pps-wincei-api"]
```

## Tích hợp client code

```python
import requests

r = requests.post(
    "http://localhost:8088/api/v1/segment-masks",
    files=[("files", open("foto.jpg", "rb"))],
    data={"refine_edges": "true", "detect_molding": "true"},
)
job_id = r.json()["job_id"]

# Poll
import time
while True:
    s = requests.get(f"http://localhost:8088/api/v1/jobs/{job_id}").json()
    if s["status"] in ("completed", "failed"):
        break
    time.sleep(3)

# Download
zip_bytes = requests.get(f"http://localhost:8088/api/v1/jobs/{job_id}/download").content
with open("masks.zip", "wb") as f:
    f.write(zip_bytes)
```

## Tích hợp JS/TS

```typescript
const fd = new FormData();
fd.append("files", fileBlob, "foto.jpg");
fd.append("detect_molding", "true");

const { job_id } = await fetch("/api/v1/segment-masks", {
  method: "POST", body: fd
}).then(r => r.json());

// Poll
let status;
do {
  await new Promise(r => setTimeout(r, 3000));
  status = await fetch(`/api/v1/jobs/${job_id}`).then(r => r.json());
} while (!["completed", "failed"].includes(status.status));

// Download
const blob = await fetch(`/api/v1/jobs/${job_id}/download`).then(r => r.blob());
```
