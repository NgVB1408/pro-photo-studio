# pps-agents

Multi-agent orchestration cho Pro Photo Studio. 5 chuyên gia (Geometry,
LightBlend, MicroContrast, Cleanup, Output) chạy `analyze()` song song trên
ảnh gốc, sau đó `apply()` tuần tự theo thứ tự cố định để giữ pixel grid nhất
quán; cuối cùng Director chấm điểm theo SOP và 3 câu tự vấn của user, trả
verdict + recommendations cho user duyệt.

## Cài đặt

```powershell
# Trong root pro-photo-studio (cần Python 3.11/3.12; 3.13 cũng chạy nếu install với --ignore-requires-python)
.venv-agents\Scripts\python.exe -m pip install --ignore-requires-python -e packages/core
.venv-agents\Scripts\python.exe -m pip install --no-deps -e packages/agents
```

## Dùng CLI

```powershell
.venv-agents\Scripts\python.exe -m pps_agents process .\in.jpg .\out.jpg `
    --property villa_luxury --long-edge 7680 --report .\out_qc.json -y
```

Argument quan trọng:

- `--property` — quyết định preset texture/dehaze/màu cho từng phân khúc:
  `villa_luxury` / `apartment_modern` / `studio_minimal` / `commercial_showroom`
  / `twilight_cabin` / `generic`.
- `--long-edge` — cạnh dài đầu ra (default 7680 = 8K).
- `--workers` — số thread cho phase analyze (default 5).
- `-y` — bỏ qua xác nhận user, ghi luôn.
- `--report` — ghi JSON QC report kèm output.

## Pipeline

```
                 ┌──────────────────────────────────────────┐
                 │  JobContext (image, property, target)    │
                 └────────────────┬─────────────────────────┘
                                  │
       ╔══════════════════════════╪══════════════════════════╗
       ║           ANALYZE phase (parallel — threadpool)     ║
       ╚══════╤═══════╤═══════════╪═══════════╤═══════╤══════╝
              ▼       ▼           ▼           ▼       ▼
         Geometry Light    MicroContrast  Cleanup  Output
              │       │           │           │       │
              └───────┴─────┬─────┴───────────┴───────┘
                            │ StagePlans
                            ▼
       ┌─────────────────────────────────────────────────────┐
       │     APPLY phase (deterministic, serial)             │
       │  geometry → lightblend → microcontrast → cleanup →  │
       │  output                                              │
       └────────────────┬────────────────────────────────────┘
                        │ image
                        ▼
       ┌─────────────────────────────────────────────────────┐
       │     DIRECTOR QC (read-only)                         │
       │  Q1 halo @ 200%   Q2 ceiling neutral   Q3 move-in   │
       │  + 5 SOP scorers (verticals, lens, sharpness,       │
       │    shadow noise, consistency vs input)              │
       └────────────────┬────────────────────────────────────┘
                        │ verdict + findings + recs
                        ▼
              user reviews → approve / re-run
```

Tại sao analyze parallel + apply serial:
- `analyze()` đọc ảnh gốc, không touch state shared → an toàn cho thread,
  CPU-bound qua OpenCV/NumPy đều release GIL.
- `apply()` thay đổi pixel grid (geometry warp) hoặc tone (lightblend).
  Chạy serial theo thứ tự cố định để stage sau làm việc trên pixel của
  stage trước → pipeline deterministic, easy-to-debug.

## Director QC — 3 câu hỏi của user

| Câu | Cách đo | Score range |
|---|---|---|
| **Q1** "Khách phóng 200% ở góc cửa sổ có thấy lem không?" | Đo luminance trong vành 2–7 px ngoài vùng V>240. Halo = excess > 30 grey levels so với mean ảnh. | 1.0 (no halo) ↘ 0.0 (severe) |
| **Q2** "Trần thật neutral hay vẫn ám xanh?" | Top 25% bright-low-sat region → đo `\|a-128\|+\|b-128\|` trong LAB. | 1.0 nếu Δ ≤ 4, giảm tuyến tính tới 0 ở Δ=20 |
| **Q3** "Có cảm giác muốn dời vào ở ngay không?" | Composite proxy: dynamic range (p1..p99 ≈ 200), clip ratios, vibrance median, midtone std. | Weighted 0..1 |

Plus 5 SOP scorers (verticals 90°, lens distortion residual, sharpness
uniformity across 4 quadrants, shadow noise, consistency vs original via
PSNR). Verdict tổng hợp theo trọng số, ngưỡng `pass ≥ 0.78`, `review ≥ 0.55`.

## Test

```powershell
.venv-agents\Scripts\python.exe -m pytest packages/agents/tests -ra
```

Tests dùng synthesised interior với đầy đủ feature (cửa sổ blown, sàn gỗ,
sofa tối, đường dọc nghiêng, TV đen) để mọi agent đều có việc làm — không
cần ảnh thật.

## Mở rộng

Thêm agent mới = subclass `BaseAgent`, implement `_analyze` + `_apply`,
register trong `Orchestrator(agents=[...])`. Director nhận `original` +
`final` nên hoàn toàn agnostic về số agent.

## Trạng thái

| Agent | Trạng thái | Delegate vào |
|---|---|---|
| Geometry | hoạt động — Hough vanishing point + sub-pixel CA correction | `pps_core.realestate.correct_vertical` |
| LightBlend | hoạt động — highlight recovery + window pull + halo feather | `pps_core.realestate.window_pull`, `pps_core.enhance.{highlight_recovery, shadow_lift}` |
| MicroContrast | hoạt động — multi-band texture + hue-aware clarity + skin-safe sharpen | `pps_core.tone.dehaze`, `pps_core.enhance.guided_filter`, `pps_core.saliency_sharpen.compute_saliency` |
| Cleanup | hoạt động — sky/lawn/TV; photog-reflection deferred to user | `pps_core.realestate.{detect_sky_mask, detect_lawn_mask, replace_sky, enhance_lawn}` |
| Output | hoạt động — shadow denoise + Lanczos hoặc Real-ESRGAN nếu có + output sharpen | `pps_core.enhance.upscale_realesrgan` (optional) |
| Director | hoạt động — 3 user questions + 5 SOP scorers | `pps_core.quality.psnr` |

Stages có ML chưa wire (Qwen-Edit, SAM 2, ControlNet sky LoRA — xem
`docs/ROADMAP.md` Phase 3) sẽ tự degrade về fallback CPU mà không crash.
