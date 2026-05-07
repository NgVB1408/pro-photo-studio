# Training

Scripts để fine-tune Qwen-Edit / SDXL trên FiveK + đánh giá PSNR/SSIM/LPIPS.

> **GATE**: Phase 5 của `docs/ROADMAP.md` đang chờ revoke 3 token đã leak
> trong notebooks (HF×2, Dropbox — xem `SECURITY.md`). Code ở đây chạy được
> với synthetic data + chế độ dry-run. **Không** push checkpoint thật, **không**
> set `HF_TOKEN` write-scope vào CI cho tới khi 3 token kia revoke xong.

## Layout

```
training/
  finetune_qwen_edit.py   # LoRA fine-tune trên FiveK expert C
  evaluate.py             # PSNR / SSIM / LPIPS held-out split
  configs/
    fivek_lora.yaml       # learning_rate, lora_rank, batch, etc.
  notebooks/              # FiftyOne walk-through (chưa commit, chờ token revoke)
  reports/                # output JSON từ evaluate.py
  tests/                  # smoke tests (dùng synthetic 32×32 pair)
```

## Quy trình

```powershell
# 1) Đảm bảo dataset stream được
.\.venv-agents\Scripts\python.exe -m pps_data sample fivek -n 5

# 2) Smoke evaluate trên synthetic
.\.venv-agents\Scripts\python.exe -m pytest training/tests -ra

# 3) Real evaluate (cần GPU + HF_TOKEN read scope)
.\.venv-agents\Scripts\python.exe training/evaluate.py --checkpoint base --split val --n 50

# 4) Fine-tune LoRA — CHỈ chạy sau khi revoke 3 token + có HF_TOKEN write scope mới
.\.venv-agents\Scripts\python.exe training/finetune_qwen_edit.py \
    --config training/configs/fivek_lora.yaml \
    --output-repo "<your-private-org>/pps-qwen-edit-v1"
```

## License — RẤT QUAN TRỌNG

FiveK là **research-only**. Weights fine-tune từ FiveK chỉ dùng nội bộ, KHÔNG
ship commercial. Mọi run đều log dataset provenance vào Postgres `audit_log`
qua `pps-embed` để truy vết về sau.
