"""FiftyOne integration — visual QC of sampled subsets.

Loading a 5,000-photo dataset into FiftyOne is wasteful when the use-case is
*human inspection*. We materialise a sampled subset (default 200 rows) into a
named FiftyOne dataset, leaving the rest streamed.

Optional dependency: ``pip install pps-data[fiftyone]``.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


def register_sampled_view(
    rows: Iterable[dict[str, Any]],
    *,
    name: str,
    image_field: str = "input_image",
    label_field: str | None = "expert_c",
    output_dir: str | os.PathLike[str] = "fixtures/fiftyone",
    persistent: bool = True,
) -> Any:
    """Materialise a sampled stream into a FiftyOne dataset.

    Each row's ``image_field`` is decoded with PIL and saved to ``output_dir``;
    if ``label_field`` is set the corresponding image is saved alongside. The
    FiftyOne sample carries a ``pair_path`` field linking the two on disk.
    """
    try:
        import fiftyone as fo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FiftyOne not installed — run: pip install 'pps-data[fiftyone]'"
        ) from exc
    from PIL import Image

    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    if name in fo.list_datasets():
        ds = fo.load_dataset(name)
        ds.delete()
    ds = fo.Dataset(name=name, persistent=persistent)

    samples = []
    for i, row in enumerate(rows):
        img = _to_pil(row.get(image_field))
        if img is None:
            continue
        in_path = out / f"{i:05d}_input.jpg"
        img.save(in_path, quality=92)
        sample = fo.Sample(filepath=str(in_path))
        if label_field and (label_img := _to_pil(row.get(label_field))):
            label_path = out / f"{i:05d}_target.jpg"
            label_img.save(label_path, quality=92)
            sample["target_path"] = str(label_path)
        samples.append(sample)
    ds.add_samples(samples)
    log.info("registered %s with %d samples in FiftyOne", name, len(samples))
    return ds


def _to_pil(obj: Any) -> Any:
    """Best-effort cast of a HF dataset cell into a PIL image."""
    if obj is None:
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj))
    if isinstance(obj, dict):
        if "bytes" in obj and obj["bytes"]:
            return Image.open(io.BytesIO(obj["bytes"]))
        if "path" in obj and obj["path"]:
            return Image.open(obj["path"])
    if isinstance(obj, str):
        return Image.open(obj)
    return None
