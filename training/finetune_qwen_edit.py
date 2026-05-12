"""LoRA fine-tune Qwen-Image-Edit on FiveK pairs.

This is a SCAFFOLD. The actual training loop requires `peft`, `accelerate`,
`diffusers`, and a GPU — none of which are exercised in CI. Per
SECURITY.md, real training runs are gated until the 3 leaked tokens are
revoked. The script:

* validates the config + flags,
* opens a streaming dataset via ``pps_data.stream_fivek``,
* if `--dry-run`: prints what would happen and exits 0,
* otherwise: imports `peft`/`accelerate` lazily and runs the loop.

Usage:

    python training/finetune_qwen_edit.py \
        --config training/configs/fivek_lora.yaml \
        --output-repo myorg/pps-qwen-edit-v1 \
        --dry-run

Roadmap — losses + control techniques for v2 of the LoRA recipe (after the
SECURITY.md token revoke gate is cleared):

* **Cross-Attention Control** (Hertz et al., "Prompt-to-Prompt") — at
  inference, intervene in the U-Net's cross-attention maps so structure
  comes from one image (raw) while colour / texture come from the
  reference expert C edit. We expose this as a ``--cross-attention-mode
  {p2p,blend,off}`` flag on the inference script; training uses it to
  generate spatially-faithful pairs for the loss below.
* **Semantic Consistency Loss** — auxiliary term that runs the predicted
  edit through a frozen vision encoder (CLIP or DINOv2) and penalises
  divergence from the encoder embedding of the input raw. Forces the LoRA
  to keep "this is still the same room" identity stable while learning
  the expert C tone. Weighted with ``loss = mse + lambda * (1 - cos(z_x,
  z_y_hat))``; default ``lambda=0.1`` per the FiveK paper experiments.
* **Latent-space Poisson blend** for compositing edited regions back into
  the raw — same idea as the OpenCV ``seamlessClone`` we plan for the
  ``CleanupAgent``, but in the VAE latent grid so colour adaptation is
  diffeomorphic and seam-free at decode.

These are not active in the current scaffold — they'll be wired into
``train()`` once the gate clears, with each technique behind a feature flag
in ``configs/fivek_lora.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("pps.finetune")


@dataclass
class FineTuneConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "FineTuneConfig":
        try:
            import yaml  # PyYAML
        except ImportError:
            import tomllib  # never used; placeholder

            raise RuntimeError("Install PyYAML: pip install pyyaml")
        return cls(raw=yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw.get("lora", {})

    @property
    def train(self) -> dict[str, Any]:
        return self.raw.get("train", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen-Edit on FiveK")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-repo",
        help="Override output_repo (HF Private). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + dataset stream, do not run training.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = FineTuneConfig.from_yaml(args.config)

    output_repo = args.output_repo or cfg.output.get("hf_private_repo")
    if not args.dry_run and not output_repo:
        print(
            "error: --output-repo is required for real training runs",
            file=sys.stderr,
        )
        return 2

    # Stream dataset to verify access (works in dry-run too).
    try:
        from pps_data import stream_fivek
    except ImportError as e:
        print(f"error: pps-data is not installed ({e}).", file=sys.stderr)
        return 2

    try:
        ds = stream_fivek(
            expert=cfg.dataset.get("expert", "c"),
            split=cfg.dataset.get("split", "train"),
            mirror=cfg.dataset.get("mirror"),
        )
        first = next(iter(ds))
        log.info("dataset OK — first row keys: %s", list(first.keys()))
    except StopIteration:
        log.error("dataset stream returned 0 rows — check HF_TOKEN + mirror")
        return 2
    except Exception as e:
        log.error("dataset stream failed: %s", e)
        return 2

    if args.dry_run:
        print(json.dumps(
            {
                "status": "dry_run_ok",
                "config_path": str(args.config),
                "dataset": cfg.dataset,
                "model_base": cfg.model.get("base"),
                "lora_rank": cfg.lora.get("rank"),
                "max_steps": cfg.train.get("max_steps"),
                "output_repo": output_repo,
            },
            indent=2,
        ))
        return 0

    # ------------------------------------------------------------------
    # Real training path — fully scaffolded, never imported in tests/CI.
    # ------------------------------------------------------------------
    log.info("real training run — importing accelerate / peft / diffusers")
    try:
        import accelerate  # noqa: F401
        import peft  # noqa: F401
        from diffusers import AutoPipelineForImage2Image  # noqa: F401
    except ImportError as e:
        print(
            f"error: real training requires `pip install accelerate peft diffusers`. {e}",
            file=sys.stderr,
        )
        return 2

    # NOTE: actual fine-tune loop is intentionally left as TODO comments below
    # until the SECURITY.md gate is unblocked. The structure is well-known
    # diffusers LoRA — we don't ship a placeholder loop that gives misleading
    # results.
    log.error(
        "Training loop intentionally not wired yet — gated by SECURITY.md "
        "(3 leaked tokens must be revoked first). "
        "When ready: implement the standard diffusers + peft LoRA loop here, "
        "log scores via `pps_embed.AuditLog`, push to %s.",
        output_repo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
