# pps-data

Stream MIT-Adobe FiveK / LSD / SUN từ Hugging Face Hub mà **không tốn ổ cứng**
— `datasets.IterableDataset` với `streaming=True`. Có CLI tiếng Việt để lấy
mẫu nhanh + mở subset trên FiftyOne để soi visual.

## Cài đặt

```powershell
.\.venv-agents\Scripts\python.exe -m pip install --ignore-requires-python -e packages/data
# Tùy chọn: thêm FiftyOne để inspect visual
.\.venv-agents\Scripts\python.exe -m pip install "fiftyone>=1.0"
```

Set token đọc HF (cần cho dataset gated):

```powershell
$env:HF_TOKEN = "<read-token>"
```

## CLI

```powershell
.\.venv-agents\Scripts\python.exe -m pps_data list
# Datasets hỗ trợ:
#    fivek   MichelangeloC/MIT-Adobe-FiveK   (default ...)
#    lsd     fffiloni/LSD-Dataset            (default ...)
#    sun     VicharVision/sun397             (default ...)

.\.venv-agents\Scripts\python.exe -m pps_data sample fivek -n 5 --out fixtures --expert c
# → fixtures/fivek/00000_input_image.jpg, 00000_expert_c.jpg, ...

.\.venv-agents\Scripts\python.exe -m pps_data inspect fivek -n 50
# Mở FiftyOne tại http://localhost:5151
```

Override mirror bằng env hoặc `--mirror`:

```powershell
$env:PPS_FIVEK_REPO = "yourname/your-private-fivek-mirror"
.\.venv-agents\Scripts\python.exe -m pps_data sample fivek -n 5
```

## API Python

```python
from pps_data import stream_fivek, stream_lsd, stream_sun

ds = stream_fivek(expert="c")          # streaming, không tải
for row in ds:                          # IterableDataset
    raw_img    = row["input_image"]    # PIL.Image hoặc dict
    expert_img = row["expert_c"]       # PIL.Image
    break
```

## License gate

FiveK + SUN là **research-only**. Đọc `LICENSES.md` trước khi train model
thương mại trên đó. Quy tắc trong repo:

- Không re-host dataset (ta chỉ stream).
- Weights fine-tune từ FiveK → HF Private repo, không push public.
- Mỗi training run ghi dataset provenance vào `audit_log` (Phase B).

## Test

```powershell
.\.venv-agents\Scripts\python.exe -m pytest packages/data/tests -ra
```

Test mock HF endpoint nên không cần kết nối mạng / token thật.

## Cấu trúc

```
packages/data/
  pps_data/
    __init__.py            # public exports
    loaders/
      _common.py           # load_streaming + take helpers
      fivek.py             # stream_fivek(expert)
      lsd.py               # stream_lsd
      sun.py               # stream_sun
    fiftyone_views.py      # register_sampled_view (optional dep)
    cli.py                 # Typer app, help text VN
    messages_vi.py         # i18n strings
    cards/fivek.yaml       # dataset metadata
  tests/
  LICENSES.md
  README.md
  pyproject.toml
```

## Tương lai

- Phase B (`pps-embed`) đọc loaders ở đây để build vector index.
- Phase C (`training/`) gọi `stream_fivek("c")` cho LoRA fine-tune.
- Có thể thêm loader khác (Reds, DPED) mà không đụng API hiện tại.
