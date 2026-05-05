# Pro Photo Studio — Tóm tắt chức năng

**Sản phẩm:** Nền tảng nâng cấp ảnh BĐS production-grade · cạnh tranh AutoEnhance.ai + Imagen-AI
**Repo:** github.com/NgVB1408/pro-photo-studio (private) · **License:** Apache 2.0
**Ngày báo cáo:** 5/5/2026

---

## Đã làm xong (✅)

| Chức năng | Trạng thái |
|---|---|
| **Sửa phối cảnh / Lens correction** (Adobe Upright + Brown-Conrady) | ✅ |
| **HDR fusion 3–7 ảnh bracket** (Mertens + deghost + LAB normalize) | ✅ |
| **Auto-detect bracket sets** trong batch (EXIF + brightness + filename) | ✅ |
| **Replace sky** procedural (6 preset cloud Perlin noise) | ✅ |
| **Window pull** (kéo chi tiết cửa sổ cháy) | ✅ |
| **Lawn enhancement** (cỏ tươi tự nhiên) | ✅ |
| **Scene classify** (interior / exterior / aerial) | ✅ |
| **Virtual Twilight** (Day → Sunset + window glow) | ✅ |
| **Selective sharpening** (saliency-aware, không plastic) | ✅ |
| **Tone coherency** (batch-wide LAB anchor — cả series cùng "vibe") | ✅ |
| **TV blackout / Fire fireplace / Photog removal** | ✅ |
| **Auto privacy** (face + license plate blur) | ✅ |
| **Pre-flight QC** (blur / exposure / dimension warn) | ✅ |
| **REST API public** (POST job → poll → download) | ✅ |
| **Deterministic seed** (cùng seed → output byte-identical) | ✅ |
| **CI/CD** matrix Linux+Windows × Python 3.11+3.12, 8 job xanh | ✅ |
| **Secret-leak protection** 3 lớp (gitleaks + scanner custom + .gitignore) | ✅ |

**Test:** 242 pass, 0 fail · **Code:** ~5,200 LOC production · **Stack:** Python 3.11/3.12, FastAPI, OpenCV, Pydantic v2

---

## Đang làm (🔄 — 5 ngày tới)

- 🔄 **API key auth** + rate limit per key
- 🔄 **Persistence Postgres** (job history + audit log)
- 🔄 **S3 storage** cho ảnh output (R2/MinIO/AWS)
- 🔄 **Webhook signed** (HMAC-SHA256) khi job xong
- 🔄 **Docker-compose dev** (1 lệnh chạy full stack local)

---

## Sắp làm (📋 — 8 tuần tới)

| Tuần | Chức năng | Cần |
|---|---|---|
| 3 | **AI Upscale x2/x4** (SUPIR — SOTA 2025) | GPU credit |
| 3 | **Instruction edit** ("brighten kitchen") qua Qwen-Image-Lightning | GPU credit |
| 3 | **Click-mask object removal** (SAM 2 + LaMa) | GPU credit |
| 3 | **AI sky replace** (ControlNet LoRA) thay procedural | GPU credit |
| 4 | **Virtual staging** (phòng trống → có nội thất) — feature mới | GPU credit |
| 4 | **Multi-angle synthesis** (1 ảnh → nhiều góc) — feature mới | GPU credit |
| 5 | **Web portal Next.js 15** (upload + preview + billing) | — |
| 6 | **Stripe billing** (per-image credits) | Stripe account |
| 6 | **Deploy production** (Docker + K8s + RunPod GPU autoscale) | — |
| 7 | **Beta launch** (10 paid pilot RE photographer/agency) | Warm intro |

---

## 3 chức năng đối thủ KHÔNG có

1. **Virtual staging** — phòng trống → có nội thất (SD3.5 + IPAdapter)
2. **Multi-angle synthesis** — 1 ảnh → nhiều góc nhìn (Qwen-Edit-2509)
3. **Instruction edit ngôn ngữ tự nhiên** — "làm sáng nhà bếp", "xóa người trong gương"

---

## Cần từ NĐT

1. **GPU credit** RunPod/Lambda Labs — ~$1,500/tháng (Phase ML + beta)
2. **Beta pipeline** — 10 nhiếp ảnh BĐS / agency cho paid pilot
3. **Runway** 8–10 tuần kỹ sư tập trung

---

## Verify trong 60 giây

```bash
git clone git@github.com:NgVB1408/pro-photo-studio.git
cd pro-photo-studio && uv sync
uv run pytest packages/core/tests packages/api/tests   # 242 pass
uv run pps-api && open http://localhost:8000/docs       # Swagger UI
```

CI dashboard: github.com/NgVB1408/pro-photo-studio/actions — 8/8 xanh.
