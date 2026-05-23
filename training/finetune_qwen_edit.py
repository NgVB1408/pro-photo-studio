"""LoRA fine-tune Qwen-Image-Edit on FiveK pairs.

Implementation status:

* ``--dry-run``  — validate config + dataset stream; no model load. Cheap.
* ``--smoke``    — full pipeline wired (Secret Manager, accelerate, peft, optimizer
                   step) with synthetic 64×64 pairs, ``max_steps=2``, no checkpoint
                   push. Verifies imports + tensor shapes on CPU/GPU. ~30s on CPU.
* default        — real training. Requires ``PPS_HF_TOKEN_READ_SECRET`` and
                   ``PPS_HF_TOKEN_WRITE_SECRET`` env vars (Secret Manager paths) or
                   fallback ``HF_TOKEN`` env. Streams FiveK → encodes through the
                   base VAE → trains LoRA on UNet attention modules → pushes to
                   ``--output-repo`` only after success.

Secret Manager integration:

    PPS_HF_TOKEN_READ_SECRET="projects/.../secrets/HF_TOKEN_READ/versions/latest"
    PPS_HF_TOKEN_WRITE_SECRET="projects/.../secrets/HF_TOKEN_WRITE/versions/latest"

Audit log:

    PPS_AUDIT_DB_URL="postgresql://..."  — if set, every run inserts a row into
    ``audit_log`` with dataset provenance + final loss + duration. Otherwise the
    same payload is written to ``<output>/audit.json``.

Usage:

    python training/finetune_qwen_edit.py \\
        --config training/configs/fivek_lora.yaml \\
        --output-repo myorg/pps-qwen-edit-v1 \\
        --dry-run

    python training/finetune_qwen_edit.py \\
        --config training/configs/fivek_lora.yaml \\
        --smoke

Roadmap — losses + control techniques for v2 of the LoRA recipe:

* **Cross-Attention Control** (Hertz et al., "Prompt-to-Prompt") — at
  inference, intervene in the U-Net's cross-attention maps so structure
  comes from one image (raw) while colour / texture come from the
  reference expert C edit. Flag: ``--cross-attention-mode {p2p,blend,off}``.
* **Semantic Consistency Loss** — auxiliary term that runs the predicted
  edit through a frozen vision encoder (CLIP or DINOv2) and penalises
  divergence from the encoder embedding of the input raw.
  ``loss = mse + lambda * (1 - cos(z_x, z_y_hat))``; default ``lambda=0.1``.
* **Latent-space Poisson blend** for compositing — same idea as OpenCV
  ``seamlessClone`` but in the VAE latent grid.

Each technique behind a feature flag in ``configs/fivek_lora.yaml`` (additional
keys to be added when wired).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("pps.finetune")


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


@dataclass
class FineTuneConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "FineTuneConfig":
        try:
            import yaml  # PyYAML
        except ImportError:
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

    @property
    def logging_cfg(self) -> dict[str, Any]:
        return self.raw.get("logging", {})


# ----------------------------------------------------------------------
# Secret Manager — fetch HF tokens at runtime (not baked into image)
# ----------------------------------------------------------------------


def _fetch_secret(resource: str) -> str | None:
    """Resolve a Secret Manager resource path like
    ``projects/X/secrets/NAME/versions/latest`` to its payload string.

    Returns None if the SDK isn't installed or fetch fails — caller decides
    whether to fall back to env vars or error out.
    """
    try:
        from google.cloud import secretmanager  # type: ignore
    except ImportError:
        log.warning("google-cloud-secret-manager not installed; cannot fetch %s",
                    resource)
        return None
    try:
        client = secretmanager.SecretManagerServiceClient()
        resp = client.access_secret_version(name=resource)
        return resp.payload.data.decode("utf-8")
    except Exception:
        log.exception("Secret Manager fetch failed for %s", resource)
        return None


def resolve_hf_tokens() -> tuple[str | None, str | None]:
    """Return (read_token, write_token), preferring Secret Manager paths in
    ``PPS_HF_TOKEN_READ_SECRET`` / ``PPS_HF_TOKEN_WRITE_SECRET`` env vars,
    falling back to ``HF_TOKEN`` env (used for both)."""
    read_path = os.environ.get("PPS_HF_TOKEN_READ_SECRET")
    write_path = os.environ.get("PPS_HF_TOKEN_WRITE_SECRET")
    read_tok = _fetch_secret(read_path) if read_path else None
    write_tok = _fetch_secret(write_path) if write_path else None
    fallback = os.environ.get("HF_TOKEN")
    return (read_tok or fallback, write_tok or fallback)


# ----------------------------------------------------------------------
# Audit log — DB row or JSON file
# ----------------------------------------------------------------------


def write_audit(payload: dict[str, Any], output_dir: Path) -> None:
    """Insert one row into ``audit_log`` (Postgres) when ``PPS_AUDIT_DB_URL``
    is set; otherwise write ``<output_dir>/audit.json``.
    """
    db_url = os.environ.get("PPS_AUDIT_DB_URL")
    if db_url:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session

            from pps_embed import AuditLog
        except ImportError as e:
            log.warning("Cannot write audit DB (import failed: %s); fall back to JSON", e)
        else:
            try:
                engine = create_engine(db_url)
                with Session(engine) as ses:
                    row = AuditLog(
                        job_id=payload.get("job_id", "pps-qwen-edit"),
                        dataset_provenance=payload.get("dataset_provenance"),
                        scores=payload.get("scores"),
                        duration_seconds=payload.get("duration_seconds"),
                        note=payload.get("note", ""),
                    )
                    ses.add(row)
                    ses.commit()
                log.info("audit row inserted: job_id=%s", payload.get("job_id"))
                return
            except Exception:
                log.exception("audit DB insert failed; fall back to JSON")
    # JSON fallback
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = output_dir / "audit.json"
    fp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("audit JSON: %s", fp)


# ----------------------------------------------------------------------
# Synthetic dataset for --smoke
# ----------------------------------------------------------------------


def _synthetic_pairs(n: int = 8, size: int = 64) -> Iterator[dict]:
    """Yield ``n`` synthetic (raw, expert) pairs. PIL images, deterministic.

    Used by --smoke to verify the loop runs without touching HF.
    """
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(42)
    for i in range(n):
        raw_arr = (rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        # "edit" = slight brightness shift
        edit_arr = np.clip(raw_arr.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        yield {
            "raw": Image.fromarray(raw_arr),
            "c": Image.fromarray(edit_arr),
            "index": i,
        }


# ----------------------------------------------------------------------
# Real training loop (lazy imports)
# ----------------------------------------------------------------------


def train(
    cfg: FineTuneConfig,
    *,
    output_repo: str | None,
    smoke: bool,
    output_dir: Path,
    max_steps_override: int | None = None,
) -> int:
    """Full LoRA fine-tune. Returns exit code (0 OK)."""
    t0 = time.perf_counter()
    log.info("=== train start (smoke=%s, output_repo=%s) ===", smoke, output_repo)

    # ---- Lazy imports — only when actually training ----
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader, IterableDataset
    except ImportError as e:
        log.error("Training stack missing — install accelerate peft diffusers torch: %s", e)
        return 2

    set_seed(cfg.train.get("seed", 42))

    # ---- Tokens ----
    read_tok, write_tok = resolve_hf_tokens()
    if not smoke:
        if not read_tok:
            log.error("No HF read token (set PPS_HF_TOKEN_READ_SECRET or HF_TOKEN)")
            return 2
        if not write_tok and output_repo:
            log.warning("No HF write token; cannot push to %s", output_repo)
    else:
        log.info("smoke mode — skipping HF token requirement")

    # ---- Diffusers pipeline ----
    base = cfg.model.get("base", "lightx2v/Qwen-Image-Lightning")
    log.info("loading base pipeline %s ...", base)
    try:
        from diffusers import AutoPipelineForImage2Image
    except ImportError as e:
        log.error("diffusers missing: %s", e)
        return 2

    if smoke:
        # Use a tiny SD model to keep CPU smoke fast (~30s vs 30min for Qwen).
        # Verifies wiring without 20GB download.
        base_for_smoke = "hf-internal-testing/tiny-stable-diffusion-pipe"
        log.info("smoke override: base = %s", base_for_smoke)
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                base_for_smoke,
                torch_dtype=torch.float32,
            )
        except Exception:
            log.exception("smoke pipeline load failed — trying offline mock UNet")
            return _smoke_offline_mock(cfg)
    else:
        dtype = torch.bfloat16 if cfg.train.get("mixed_precision") == "bf16" else torch.float32
        pipe = AutoPipelineForImage2Image.from_pretrained(
            base, torch_dtype=dtype, token=read_tok,
        )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = getattr(pipe, "text_encoder", None)
    scheduler = pipe.scheduler

    # Freeze VAE + text encoder
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)

    # ---- LoRA wrap UNet ----
    lora_cfg = LoraConfig(
        r=cfg.lora.get("rank", 16),
        lora_alpha=cfg.lora.get("alpha", 32),
        lora_dropout=cfg.lora.get("dropout", 0.05),
        target_modules=cfg.lora.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    n_train = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in unet.parameters())
    log.info("LoRA params: %d trainable / %d total (%.2f%%)",
             n_train, n_total, 100 * n_train / max(1, n_total))

    if cfg.train.get("gradient_checkpointing"):
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        elif hasattr(unet.base_model, "enable_gradient_checkpointing"):
            unet.base_model.enable_gradient_checkpointing()

    # ---- Dataset ----
    expert_col = cfg.dataset.get("expert", "c")

    if smoke:
        ds_iter = list(_synthetic_pairs(n=8, size=64))
        log.info("smoke dataset: %d synthetic pairs", len(ds_iter))
    else:
        try:
            from pps_data import stream_fivek
        except ImportError as e:
            log.error("pps_data missing: %s", e)
            return 2
        ds_iter = stream_fivek(
            expert=expert_col,
            split=cfg.dataset.get("split", "train"),
            mirror=cfg.dataset.get("mirror"),
            token=read_tok,
        )

    class PairDataset(IterableDataset):
        def __init__(self, src, expert):
            self.src = src
            self.expert = expert

        def __iter__(self):
            for row in self.src:
                # Schema varies by mirror:
                #   logasja/mit-adobe-fivek: original / augmented
                #   yuukicammy/MIT-Adobe-FiveK: raw / expert_<x>
                #   synthetic smoke pairs: raw / target
                raw = (
                    row.get("raw")
                    or row.get("input")
                    or row.get("original")
                )
                target = (
                    row.get(self.expert)
                    or row.get("target")
                    or row.get(f"expert_{self.expert}")
                    or row.get("augmented")
                )
                if raw is None or target is None:
                    continue
                yield {"raw": raw, "target": target}

    ds = PairDataset(ds_iter, expert_col)

    def collate(batch):
        from PIL import Image
        import torchvision.transforms as T
        H = 256 if smoke else 512
        tf = T.Compose([
            T.Resize((H, H)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        raws = torch.stack([tf(b["raw"].convert("RGB")) for b in batch])
        targets = torch.stack([tf(b["target"].convert("RGB")) for b in batch])
        return {"raw": raws, "target": targets}

    bs = cfg.train.get("batch_size", 4)
    if smoke:
        bs = 2
    dl = DataLoader(ds, batch_size=bs, collate_fn=collate)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=cfg.train.get("learning_rate", 1e-4),
    )

    # ---- Accelerator ----
    mp = cfg.train.get("mixed_precision", "no")
    if smoke and not torch.cuda.is_available():
        mp = "no"   # CPU bf16 buggy in some torch builds
    acc = Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=cfg.train.get("grad_accumulation", 1),
    )
    unet, optimizer, dl = acc.prepare(unet, optimizer, dl)
    vae = vae.to(acc.device)
    if text_encoder is not None:
        text_encoder = text_encoder.to(acc.device)

    max_steps = cfg.train.get("max_steps", 4000)
    if smoke:
        max_steps = 2
    if max_steps_override is not None:
        max_steps = max_steps_override
        log.info("max_steps override active: %d", max_steps)
    save_every = cfg.train.get("save_every_steps", 500)
    log_every = cfg.logging_cfg.get("log_every_steps", 25)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train ----
    step = 0
    final_loss = float("nan")
    unet.train()
    try:
        while step < max_steps:
            for batch in dl:
                with acc.accumulate(unet):
                    raw = batch["raw"]
                    target = batch["target"]

                    # Encode through VAE
                    with torch.no_grad():
                        target_latents = vae.encode(target).latent_dist.sample() * 0.18215

                    # Sample noise + timestep
                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps,
                        (target_latents.shape[0],), device=acc.device,
                    ).long()
                    noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

                    # Text encoder dummy: use empty prompt embeddings if text encoder
                    if text_encoder is not None and hasattr(pipe, "tokenizer"):
                        enc_input = pipe.tokenizer(
                            [""] * raw.shape[0],
                            padding="max_length",
                            max_length=pipe.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).to(acc.device)
                        with torch.no_grad():
                            text_emb = text_encoder(**enc_input)[0]
                    else:
                        # No text encoder → zero embeddings of expected size
                        text_emb = torch.zeros(
                            raw.shape[0],
                            77,
                            getattr(unet.config, "cross_attention_dim", 768),
                            device=acc.device,
                        )

                    # Predict noise
                    pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
                    loss = F.mse_loss(pred, noise)

                    acc.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                final_loss = float(loss.detach().cpu().item())
                step += 1
                if step % log_every == 0 or step == max_steps or smoke:
                    log.info("step %d/%d  loss=%.4f", step, max_steps, final_loss)
                if step >= max_steps:
                    break
                if step % save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    unwrapped = acc.unwrap_model(unet)
                    unwrapped.save_pretrained(ckpt_dir)
                    log.info("saved %s", ckpt_dir)
    except Exception:
        log.exception("training loop crashed")
        return 3

    # ---- Final save ----
    unwrapped = acc.unwrap_model(unet)
    final_dir = output_dir / "final"
    unwrapped.save_pretrained(final_dir)
    log.info("final adapter saved to %s", final_dir)

    duration = time.perf_counter() - t0

    # ---- Audit log ----
    write_audit({
        "job_id": os.environ.get("CLOUD_ML_JOB_ID", f"pps-qwen-edit-{int(t0)}"),
        "dataset_provenance": {
            "dataset": "fivek",
            "expert": cfg.dataset.get("expert"),
            "mirror": cfg.dataset.get("mirror"),
            "smoke": smoke,
        },
        "scores": {"final_loss": final_loss, "max_steps": max_steps, "actual_steps": step},
        "duration_seconds": duration,
        "note": f"output_repo={output_repo}",
    }, output_dir)

    # ---- Push ----
    if smoke:
        log.info("smoke complete — not pushing")
        return 0
    if not output_repo:
        log.warning("no --output-repo; skip push")
        return 0
    if not write_tok:
        log.error("no HF write token; cannot push to %s", output_repo)
        return 4

    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=write_tok)
        create_repo(output_repo, private=True, exist_ok=True, token=write_tok)
        api.upload_folder(
            folder_path=str(final_dir),
            repo_id=output_repo,
            commit_message=f"LoRA r{cfg.lora.get('rank')} steps={step} loss={final_loss:.4f}",
            token=write_tok,
        )
        log.info("pushed adapter to https://huggingface.co/%s", output_repo)
    except Exception:
        log.exception("push to HF Hub failed")
        return 5

    return 0


def _smoke_offline_mock(cfg: FineTuneConfig) -> int:
    """Fully offline smoke: skip HF download, instantiate minimal UNet + VAE
    stubs to verify our forward/backward wiring. Returns 0 on success.
    """
    log.warning("offline smoke: faking pipeline (no HF call)")
    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        log.error("missing torch/peft: %s", e)
        return 2

    # Toy UNet-like model: 3 conv + cross-attn-shape linear, with 'to_q'/'to_k'/'to_v' modules
    class ToyAttention(torch.nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.to_q = torch.nn.Linear(dim, dim)
            self.to_k = torch.nn.Linear(dim, dim)
            self.to_v = torch.nn.Linear(dim, dim)
            self.to_out = torch.nn.Sequential(torch.nn.Linear(dim, dim))

        def forward(self, x):
            q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
            return self.to_out(q + k + v)

    class ToyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyAttention()
            self.proj_in = torch.nn.Conv2d(4, 32, 1)
            self.proj_out = torch.nn.Conv2d(32, 4, 1)

        def forward(self, x):
            h = self.proj_in(x)
            B, C, H, W = h.shape
            h_flat = h.flatten(2).transpose(1, 2)
            h_flat = self.attn(h_flat)
            h = h_flat.transpose(1, 2).reshape(B, C, H, W)
            return self.proj_out(h)

    unet = ToyUNet()
    lora_cfg = LoraConfig(
        r=cfg.lora.get("rank", 16),
        lora_alpha=cfg.lora.get("alpha", 32),
        lora_dropout=cfg.lora.get("dropout", 0.05),
        target_modules=cfg.lora.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)

    for s in range(2):
        x = torch.randn(2, 4, 8, 8)
        y = unet(x)
        loss = F.mse_loss(y, torch.zeros_like(y))
        opt.zero_grad(); loss.backward(); opt.step()
        log.info("offline smoke step %d loss=%.4f", s, loss.item())
    log.info("offline smoke OK")
    return 0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen-Edit on FiveK")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-repo",
        help="HF Private repo for final LoRA push. Required for real runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/qwen-edit-latest"),
        help="Local directory for checkpoints + audit.json.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config + dataset stream; no model load.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Full pipeline with synthetic 64x64 pairs, max_steps=2, no push.",
    )
    parser.add_argument(
        "--offline-smoke", action="store_true",
        help="Pure-stub smoke (no HF call at all). Fastest, lowest fidelity.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override train.max_steps from config (useful for Vertex smoke runs).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Env fallback: PPS_MAX_STEPS lets Vertex submitters override without
    # editing the args block in vertex_train.yaml.
    if args.max_steps is None and os.environ.get("PPS_MAX_STEPS"):
        try:
            args.max_steps = int(os.environ["PPS_MAX_STEPS"])
        except ValueError:
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = FineTuneConfig.from_yaml(args.config)

    output_repo = args.output_repo or cfg.output.get("hf_private_repo")
    if not args.dry_run and not args.smoke and not args.offline_smoke and not output_repo:
        print("error: --output-repo is required for real training runs", file=sys.stderr)
        return 2

    # ---- Dry-run path (no torch/diffusers needed) ----
    if args.dry_run:
        try:
            from pps_data import stream_fivek
        except ImportError as e:
            print(f"error: pps-data missing ({e})", file=sys.stderr)
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
        print(json.dumps({
            "status": "dry_run_ok",
            "config_path": str(args.config),
            "dataset": cfg.dataset,
            "model_base": cfg.model.get("base"),
            "lora_rank": cfg.lora.get("rank"),
            "max_steps": cfg.train.get("max_steps"),
            "output_repo": output_repo,
        }, indent=2))
        return 0

    # ---- Offline-smoke (no HF, pure stubs) ----
    if args.offline_smoke:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return _smoke_offline_mock(cfg)

    # ---- Real train OR smoke (with HF download) ----
    return train(
        cfg,
        output_repo=output_repo,
        smoke=args.smoke,
        output_dir=args.output_dir,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
