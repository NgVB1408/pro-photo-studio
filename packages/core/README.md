# pps-core

Deterministic OpenCV pipeline for real-estate photo enhancement.

This package contains zero ML dependencies — every stage runs on CPU with
NumPy + OpenCV. ML-backed alternatives live in `pps-ai`.

## Modules

| Module | Purpose |
|---|---|
| `enhance` | Studio-grade 8-step color pipeline (WB, CLAHE, highlight/shadow, vibrance, detail) |
| `realestate` | Scene-aware RE pipeline (sky/lawn/window/vertical/classify) |
| `hdr` | Mertens bracket fusion + deghost + LAB color normalize + pseudo-HDR |
| `twilight` | Day → Sunset transform (sky composite + window glow + warm tone) |
| `tone_coherency` | Batch tone anchor (LAB) + static presets (warm/cool/real_estate) |
| `perspective` | 4-point upright via Hough vanishing point |
| `lens` | Brown-Conrady distortion correction |
| `auto_privacy` | Face + license-plate detection and blur |
| `tv_blackout` | Detect TV → blacken |
| `fire_fireplace` | Composite fire into fireplace |
| `photog_removal` | Mirror-reflection inpaint |
| `saliency_sharpen` | Saliency-aware selective sharpening |
| `sky_seg_ai` | rembg-based sky segmentation (optional) |
| `inpaint` | OpenCV TELEA/NS dispatch + LaMa hook (optional) |
| `composite` | Pixel-perfect 2-image diff + Poisson blend |
| `detect`, `mask` | Watermark/logo detection strategies |
| `preflight` | Pre-flight QC (blur, exposure, dimension, focus uniformity) |
| `quality` | PSNR/SSIM/sharpness metrics |
| `batch` | ProcessPoolExecutor batch runner |
| `video` | Video frame ops |
| `cli` | argparse CLI for all subcommands |

## Install

```bash
uv add pps-core                  # core (CPU-only)
uv add 'pps-core[raw]'           # + RAW/DNG support via rawpy
uv add 'pps-core[sky-ai]'        # + ONNX-based sky segmentation
uv add 'pps-core[dropbox]'       # + Dropbox download
```

## Quick API

```python
from pps_core.realestate import enhance_realestate_full
from pps_core.utils import read_image, write_image

img = read_image("photo.jpg")
out, report = enhance_realestate_full(img, sky_preset="blue_clouds")
write_image("photo_enhanced.jpg", out)
```

## Tests

```bash
uv run --package=pps-core pytest packages/core/tests
```
