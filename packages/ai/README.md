# pps-ai

ML inference wrappers for Pro Photo Studio.

Each submodule wraps one external model (Qwen-Image-Lightning, SUPIR,
SAM 2, ControlNet, LaMa) behind a uniform `Predictor` protocol. All
weights download lazily on first call so users who don't need a given
model never pay the disk-space cost.

> Status: Phase 0 scaffold. Real model wiring lands in Phase 3.

## Submodules

- `pps_ai.qwen` — instruction-based image editing via
  `lightx2v/Qwen-Image-Lightning` (LoRA distillation of `Qwen/Qwen-Image`).
- `pps_ai.supir` — SOTA image restoration / upscale (planned).
- `pps_ai.sam2` — Segment Anything 2 click-mask (planned).
- `pps_ai.controlnet` — sky / depth / upright ControlNets (planned).
- `pps_ai.lama` — LaMa Cleaner object removal via ONNX (planned).

## Install

ML backends are NOT pip extras because their distribution paths vary.
See the per-module installation notes (or `docs/MODELS.md` once Phase 3
ships).

## License

Apache 2.0 for the wrapper code. Model weights pull under their own
licenses (Qwen: Apache 2.0; SD3.5: Stability AI Community License;
SAM 2: Apache 2.0; SUPIR: Apache 2.0).
