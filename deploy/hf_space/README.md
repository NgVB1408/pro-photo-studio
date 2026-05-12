---
title: Pro Photo Studio
emoji: 🏡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Demo nâng cao ảnh bất động sản (real-estate photo enhancement)
suggested_hardware: cpu-basic
suggested_storage: small
hf_oauth: false
sleep_time: 600
---

# Pro Photo Studio — demo trên Hugging Face Spaces

Demo CPU của pipeline nâng cao ảnh bất động sản tại
<https://github.com/NgVB1408/pro-photo-studio>. Tải ảnh interior hoặc
exterior lên → áp WB / CLAHE / highlight recovery / shadow lift / sky+lawn
nếu phát hiện. Production pipeline có thêm HDR + AI inpaint + 8K upscale qua
API.

## Cài đặt local

```bash
pip install -r requirements.txt
python app.py
```

## Deploy lại Space này

Repo gốc có workflow `.github/workflows/hf-space-deploy.yml` (`workflow_dispatch`)
push toàn bộ thư mục `deploy/hf_space/` lên Space. Cần secret `HF_SPACES_TOKEN`
và repo variable `ALLOW_HF_DEPLOY=true`.
