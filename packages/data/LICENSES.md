# Dataset licences — read before any redistribution

The `pps-data` package wraps Hugging Face mirrors of public datasets. We
**stream** content; we never re-host it. Anything you derive from these
datasets carries the upstream licence. Read each upstream before shipping a
product on top of it.

## MIT-Adobe FiveK

* Upstream: <https://data.csail.mit.edu/graphics/fivek/>
* Default HF mirror: `MichelangeloC/MIT-Adobe-FiveK`
* Licence: **research-only**
* Permitted: in-house experiments, academic publications.
* Not permitted: shipping fine-tuned model weights commercially, training
  closed-source SaaS that monetises FiveK-derived behaviour.
* Action: keep weights derived from FiveK in HF *Private* repos. Do not push
  them to a customer-facing endpoint without legal review.

## LSD (Large-scale Scene De-scattering)

* Upstream: paper-listed, sources vary by mirror.
* Default HF mirror: `fffiloni/LSD-Dataset` (override via `PPS_LSD_REPO`).
* Licence: check the active mirror — some are CC-BY-NC, some Apache.
* Action: log the mirror id + licence in `audit_log` for every fine-tune run.

## SUN Database

* Upstream: <https://vision.princeton.edu/projects/2010/SUN/>
* Default HF mirror: `VicharVision/sun397`
* Licence: research-only.
* Action: same as FiveK.

## Your private dataset (HF Private)

* Bring your own contractor-supplied retouched pairs into a HF Private
  dataset. Keep `private: true` in the dataset metadata. Set `HF_TOKEN` to a
  token with **read-only** scope when streaming from a worker, **write**
  scope only on the upload box.

## Auditing

Every training and fine-tune run should record:

* Dataset repo id + git revision (`datasets.config.HF_DATASETS_OFFLINE = false`
  + `dataset.cache_files()` in non-streaming mode).
* Active mirror at run time (we resolve env vars at load — log them).
* Licence string.

Stored in Postgres `audit_log.dataset_provenance` (Phase B schema).
