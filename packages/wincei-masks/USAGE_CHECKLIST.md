# 📋 WINCEI v0.3.0 — CHECKLIST SỬ DỤNG

> Tài liệu vận hành chính thức cho khách hàng. Áp dụng từ ngày bàn giao.

---

## A. CHECKLIST KHỞI ĐỘNG LẦN ĐẦU

### A.1. Yêu cầu hệ thống tối thiểu

| Hạng mục | Tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | 4 nhân Intel/AMD 2.5GHz | 8 nhân 3.0GHz+ |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA 6+ GB VRAM |
| Đĩa | 10 GB trống | 30 GB (bundle weights + cache) |
| OS | Windows 10/11, Ubuntu 22.04, macOS 13+ | Windows 11 |
| Python | 3.11 hoặc 3.12 | 3.11 (LTS) |
| Mạng | 50 Mbps (lần đầu setup) | 100 Mbps |

### A.2. Thứ tự khởi động

- [ ] **Bước 1**: Giải nén `wincei_PORTABLE_v0.3.0_*.zip` vào ổ đĩa có ≥30 GB trống
- [ ] **Bước 2**: Đọc `README.md` ở thư mục gốc (có bảng BEFORE/AFTER)
- [ ] **Bước 3** (Windows): Double-click `launchers/run_api.bat`
- [ ] **Bước 3** (Linux/Mac): `bash launchers/run_api.sh`
- [ ] **Bước 4**: Lần đầu — chờ tự động cài đặt 10-30 phút (Ollama + Python + dependencies)
- [ ] **Bước 5**: Mở trình duyệt `http://localhost:8000/`
- [ ] **Bước 6**: Verify "✅ status: ok" trên health page

### A.3. Smoke test sau khi setup

- [ ] Web UI mở được tại `http://localhost:8000/`
- [ ] Drag-drop ảnh `samples/input/DSC01527.jpg` vào dropzone
- [ ] Mode = "Smart Segmentation" + Mock = "ON" → click Run → response < 1 giây
- [ ] Đổi Mock = "OFF" → đợi job complete (~6-10 phút CPU lần đầu)
- [ ] Verdict trả `pass` hoặc `review` → hệ thống hoạt động tốt

---

## B. CHECKLIST DÙNG HÀNG NGÀY

### B.1. Workflow tiêu chuẩn — 1 ảnh đơn

- [ ] Khởi động API server (`run_api.bat` hoặc đã chạy nền)
- [ ] Mở `http://localhost:8000/`
- [ ] Drag-drop ảnh BĐS vào dropzone
- [ ] Chọn pipeline phù hợp:
  - **Smart Segmentation** — tách 9 mask + AI eval (default cho retoucher)
  - **Window+Ceiling Fix** — chỉnh tone cửa sổ blown + trần ám
  - **Full Recovery Ceiling** — VLM+SAM 2 cho ảnh khó (cần Ollama đã cài)
  - **Detect Regions** — chỉ trả JSON bbox (cho dev tích hợp)
- [ ] Click "Run" → quan sát progress bar
- [ ] Xem kết quả tại chỗ (BEFORE/AFTER + scorecard verdict)
- [ ] Đọc verdict:
  - `pass` (≥0.85) → giao thẳng khách hoặc retoucher
  - `review` (0.65-0.84) → mở Photoshop tinh chỉnh trong 30 giây
  - `fail` (<0.65) → chạy lại với option khác hoặc manual
- [ ] Click "Download Zip" → lưu output về máy

### B.2. Workflow batch — 5-100 ảnh

- [ ] Chuẩn bị folder ảnh BĐS (tránh > 200 MB tổng cho 1 batch)
- [ ] Mở PowerShell/Terminal tại thư mục dự án
- [ ] Chạy lệnh CLI batch:
  ```powershell
  .\.venv\Scripts\pps-wincei-masks.exe `
      --inputs "D:\photos\batch1" `
      --outputs "D:\masks_output" `
      --skip-existing
  ```
- [ ] Theo dõi tiến độ trong terminal
- [ ] Sau khi xong: mở `D:\masks_output\comparison.html` để duyệt
- [ ] Filter theo verdict (pass / review / fail) trên HTML viewer
- [ ] Bàn giao folder `masks_output/` cho retoucher

### B.3. Workflow HDR bracket — Sony AEB

- [ ] Khách gửi folder ảnh bracket (3 shot/cảnh, EV -2/0/+2 hoặc -3/0/+3)
- [ ] Kiểm tra EXIF: ảnh phải có `DateTimeOriginal` và `ExposureBiasValue`
- [ ] Chạy CLI:
  ```powershell
  .\.venv\Scripts\pps-wincei-hdr.exe `
      --inputs "D:\raw_bracket" `
      --outputs "D:\hdr_fused"
  ```
- [ ] Verify: số groups = số ảnh ÷ 3
- [ ] Nối tiếp pipeline masks nếu cần:
  ```powershell
  .\.venv\Scripts\pps-wincei-masks.exe `
      --inputs "D:\hdr_fused" `
      --outputs "D:\final_masks"
  ```

### B.4. Workflow API integration — cho team dev

- [ ] Khởi động API server
- [ ] Đọc Swagger UI: `http://localhost:8000/docs`
- [ ] Test mock mode trước:
  ```bash
  curl -X POST http://localhost:8000/api/v1/segment-masks \
       -F "files=@foto.jpg" -F "mock=true"
  ```
- [ ] Submit job thật → poll status → download zip
- [ ] Tích hợp vào pipeline production của khách (Python/Node/Go đều dùng được)
- [ ] Set up auth nếu mở public: env `WINCEI_API_KEY=<token>`

---

## C. CHECKLIST KIỂM TRA CHẤT LƯỢNG

Trước khi giao output cho khách hàng cuối:

- [ ] Verdict overall ≥ 0.80
- [ ] Wall coverage 30-70%
- [ ] Floor edge alignment ≥ 0.85
- [ ] Window/Door biên tách rõ (không tràn sang tường)
- [ ] Casing leakage < 6% (đã được cap tự động)
- [ ] Overlap ceiling∩wall = 0 (Sobel resolver đã xử lý)
- [ ] Output PNG dung lượng > 50 KB (không trống)
- [ ] Overlay JPG nhìn đúng cấu trúc kiến trúc
- [ ] Lưu vào folder khách + ghi nhận verdict vào báo cáo

---

## D. CHECKLIST XỬ LÝ SỰ CỐ

| Triệu chứng | Khắc phục |
|---|---|
| API server không start | Check port 8000 free: `netstat -ano \| findstr 8000` |
| `Ollama not running` | `ollama serve` trong PowerShell, để open |
| `Model bds-brain chưa pull` | `bash scripts/setup_vlm_sam.sh` |
| `SAM 2 checkpoint missing` | Download manual: `sam2_hiera_tiny.pt` về `~/.cache/sam2/` |
| Verdict luôn `fail` | Bật `--precision` mode hoặc dùng VLM+SAM2 engine |
| Ceiling 0% cov | Là ảnh modern không có trần visible (bình thường) |
| Window 0% cov | Cửa thực = glass door → dùng mask `opening` thay vì `window` |
| Job timeout 30 phút | Chia nhỏ batch < 50 ảnh hoặc dùng GPU |
| RAM OOM | Giảm `--matting-max-side 1200` (default 1600) |
| HuggingFace 429 rate limit | Đợi 5 phút hoặc set `HF_TOKEN` env var |

---

## E. CHECKLIST AN TOÀN DỮ LIỆU

- [ ] Không upload ảnh khách lên public cloud nếu chưa được phép
- [ ] Backup folder `wincei_storage/` định kỳ (chứa uploads + outputs + jobs)
- [ ] Cleanup `wincei_storage/jobs/` cũ hơn 30 ngày để tránh đầy đĩa
- [ ] Bảo vệ API server bằng API key nếu mở public (env `WINCEI_API_KEY`)
- [ ] Không commit file `.env`, `*.token`, `secrets.json` lên git
- [ ] Restrict CORS production: env `WINCEI_CORS=https://your-domain.com`

---

## F. LIÊN HỆ HỖ TRỢ

- **Email**: luvdraco88@gmail.com
- **Zalo**: 0876 254 585
- **GitHub Issues**: https://github.com/NgVB1408/pro-photo-studio/issues
- **License**: Apache-2.0
