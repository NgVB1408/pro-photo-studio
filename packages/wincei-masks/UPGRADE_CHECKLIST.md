# 🔄 WINCEI v0.3.0 — CHECKLIST NÂNG CẤP

> Quy trình nâng cấp hệ thống an toàn, không gián đoạn dịch vụ.

---

## A. CHECKLIST TRƯỚC KHI NÂNG CẤP

### A.1. Backup dữ liệu

- [ ] Backup folder `wincei_storage/` (uploads + outputs + job records)
- [ ] Backup folder `~/.cache/huggingface/` (SegFormer weights, ~1.2 GB)
- [ ] Backup folder `~/.cache/sam2/` (SAM 2 checkpoints, ~40 MB)
- [ ] Backup file cấu hình môi trường: `.env`, `.env.production`
- [ ] Backup custom Ollama Modelfile nếu đã edit `bds-brain`
- [ ] Ghi nhận version đang dùng:
  ```bash
  curl -s http://localhost:8000/api/v1/health
  ```

### A.2. Verify state trước khi upgrade

- [ ] Check Git working tree clean:
  ```bash
  cd <wincei root>
  git status
  ```
- [ ] Đảm bảo không có job đang chạy:
  ```bash
  curl -s http://localhost:8000/api/v1/jobs?limit=10
  ```
- [ ] Stop API server:
  ```bash
  # Windows
  taskkill /IM pps-wincei-api.exe /F
  # Linux/Mac
  pkill -f pps-wincei-api
  ```

### A.3. Test rollback path

- [ ] Verify tag hiện tại có thể restore:
  ```bash
  git tag --list
  # Phải thấy v0.3.0, v0.2.x...
  ```
- [ ] Lưu commit hash hiện tại để rollback:
  ```bash
  git rev-parse HEAD > backup_commit_hash.txt
  ```

---

## B. CHECKLIST UPGRADE GIT (CÁC PHIÊN BẢN MINOR/PATCH)

### B.1. Fetch + verify

- [ ] Fetch updates:
  ```bash
  git fetch --all --tags
  ```
- [ ] Xem release notes:
  ```bash
  git log --oneline v0.3.0..origin/main
  ```
- [ ] Đọc CHANGELOG.md trên GitHub release page

### B.2. Apply upgrade

- [ ] Pull main:
  ```bash
  git pull origin main
  ```
- [ ] Re-sync dependencies:
  ```bash
  uv sync --all-packages
  ```
- [ ] Verify packages installed đúng version mới:
  ```bash
  .venv/Scripts/python.exe -c "
  import pps_wincei_masks, pps_wincei_api, pps_wincei_hdr, pps_wincei
  print('masks:', pps_wincei_masks.__version__)
  print('api:', pps_wincei_api.__version__)
  print('hdr:', pps_wincei_hdr.__version__)
  print('wincei:', pps_wincei.__version__)
  "
  ```

### B.3. Smoke test post-upgrade

- [ ] Start API server lại
- [ ] Health check: `curl http://localhost:8000/api/v1/health`
- [ ] Test 1 ảnh mock mode:
  ```bash
  curl -X POST http://localhost:8000/api/v1/segment-masks \
       -F "files=@samples/input/DSC01527.jpg" -F "mock=true"
  ```
- [ ] Test 1 ảnh thật → verify verdict không tệ hơn trước
- [ ] Compare output mask coverage với version cũ → đảm bảo regression < 5%

### B.4. Rollback nếu có vấn đề

```bash
git reset --hard $(cat backup_commit_hash.txt)
uv sync --all-packages
```

---

## C. CHECKLIST UPGRADE MAJOR (v0.x → v1.0)

Khi có breaking changes (API endpoints rename, schema changes):

### C.1. Đọc migration guide

- [ ] Đọc `MIGRATION.md` trong release notes
- [ ] List các endpoint deprecated → update client code
- [ ] List schema changes → migrate database/storage nếu có

### C.2. Test trên môi trường dev trước

- [ ] Clone repo sang folder mới:
  ```bash
  git clone -b main https://github.com/NgVB1408/pro-photo-studio.git wincei_v1_test
  ```
- [ ] Cài full stack riêng (Ollama + SAM 2 + Python env)
- [ ] Chạy regression test trên 5-10 ảnh sample
- [ ] So sánh output với prod để verify không có regression nghiêm trọng

### C.3. Coordinate với khách hàng

- [ ] Thông báo trước 7 ngày downtime dự kiến
- [ ] Schedule maintenance window
- [ ] Có rollback plan trong 30 phút
- [ ] Communicate breaking changes qua email + Zalo

---

## D. CHECKLIST UPGRADE WEIGHTS / MODELS

### D.1. SegFormer model size upgrade

| Model | Size | VRAM cần | Tốc độ | Chất lượng |
|---|---|---|---|---|
| B0 | 13 MB | 1 GB | 0.3s/ảnh | ⭐⭐ |
| B1 | 53 MB | 2 GB | 0.5s/ảnh | ⭐⭐⭐ |
| B3 (current default) | 200 MB | 5 GB | 0.8s/ảnh | ⭐⭐⭐⭐ |
| B5 | 380 MB | 8 GB | 1.5s/ảnh | ⭐⭐⭐⭐⭐ |

- [ ] Kiểm tra VRAM máy hiện tại
- [ ] Set env: `WINCEI_DEFAULT_MODEL=nvidia/segformer-b5-finetuned-ade-640-640`
- [ ] Test 1 ảnh để verify performance acceptable
- [ ] Adjust `--matting-max-side` nếu OOM

### D.2. SAM 2 checkpoint upgrade

| Variant | Size | VRAM | Chất lượng biên |
|---|---|---|---|
| Tiny (current) | 38 MB | 1 GB | ⭐⭐⭐ |
| Small | 185 MB | 2 GB | ⭐⭐⭐⭐ |
| Base+ | 323 MB | 3 GB | ⭐⭐⭐⭐⭐ |
| Large | 898 MB | 5 GB | ⭐⭐⭐⭐⭐ (best) |

- [ ] Download checkpoint:
  ```bash
  curl -L -o ~/.cache/sam2/sam2_hiera_base_plus.pt \
       https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
  ```
- [ ] Update CLI flag: `--sam-checkpoint ~/.cache/sam2/sam2_hiera_base_plus.pt`
- [ ] Smoke test với 1 ảnh khó (multi-tier ceiling)

### D.3. VLM upgrade

```bash
# Pull newer VLM version
ollama pull qwen2.5vl:14b           # to lớn hơn 7b, chất lượng cao hơn
ollama pull llama3.2-vision:90b     # nếu RAM > 64 GB

# Rebuild bds-brain với base mới
sed -i 's|^FROM .*|FROM qwen2.5vl:14b|' packages/wincei-masks/Modelfile.bds-brain
ollama rm bds-brain
ollama create bds-brain -f packages/wincei-masks/Modelfile.bds-brain
```

- [ ] Test bds-brain với prompt sample
- [ ] Run full pipeline 1 ảnh → so sánh verdict với version cũ

---

## E. CHECKLIST UPGRADE INFRASTRUCTURE

### E.1. Migration sang GPU server

- [ ] Verify CUDA driver matching torch version:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  ```
- [ ] Reinstall torch CUDA variant:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- [ ] Restart API → verify `"gpu_available": true` trong `/health`
- [ ] Benchmark: 1 ảnh CPU vs GPU → verify speedup 5-15x

### E.2. Migration sang Docker production

- [ ] Build image:
  ```bash
  cd <wincei root>/src
  docker build -t wincei-api:0.3.0 -f Dockerfile .
  ```
- [ ] Run với docker-compose:
  ```bash
  docker compose up -d
  ```
- [ ] Mount persistent volumes cho:
  - `/app/wincei_storage` (uploads + outputs)
  - `/root/.cache/huggingface` (model weights)
  - `/root/.cache/sam2` (SAM checkpoints)
- [ ] Setup health check + auto-restart:
  ```yaml
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  ```

### E.3. Setup reverse proxy + SSL

- [ ] Cài Nginx hoặc Caddy
- [ ] Cấu hình proxy_pass đến `http://wincei-api:8000`
- [ ] Setup SSL Let's Encrypt
- [ ] Bật rate limit: 10 req/min per IP
- [ ] Bật API key auth: env `WINCEI_API_KEY=<token>`
- [ ] Restrict CORS: env `WINCEI_CORS=https://your-domain.com`

---

## F. CHECKLIST DEFINITION OF DONE — UPGRADE HOÀN TẤT

Sau mỗi upgrade, verify checklist sau trước khi closeout ticket:

- [ ] API server start không lỗi
- [ ] Health endpoint return 200 + version chính xác
- [ ] Web UI mở được + drag-drop hoạt động
- [ ] Mock mode trả response < 1s
- [ ] Real test 1 ảnh sample → verdict ≥ 0.75
- [ ] Web UI/Swagger UI/ReDoc accessible
- [ ] Log không có ERROR/CRITICAL trong 1 giờ đầu
- [ ] Backup folder vẫn nguyên (chỉ thêm, không bị xoá)
- [ ] Git commit hash mới được ghi vào release log
- [ ] Notify khách hàng đã upgrade xong + version mới

---

## G. LỊCH UPGRADE ĐỀ XUẤT

| Tần suất | Hạng mục |
|---|---|
| Hàng tuần | Pull patches (git pull) + sync deps |
| Hàng tháng | Update SegFormer/SAM checkpoint nếu có |
| Hàng quý | Major version upgrade (v0.x → v0.y) |
| 6 tháng | Audit security + dependency vulnerability scan |
| Hàng năm | Migration major (v0.x → v1.0) hoặc upgrade infrastructure |

---

## H. LIÊN HỆ HỖ TRỢ UPGRADE

- **Email**: luvdraco88@gmail.com
- **Zalo**: 0876 254 585
- **GitHub Releases**: https://github.com/NgVB1408/pro-photo-studio/releases
- **License**: Apache-2.0
