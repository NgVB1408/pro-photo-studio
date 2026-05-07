"""Dataset ingestion + curation for Pro Photo Studio.

Streaming-first: every loader returns a Hugging Face ``IterableDataset`` so
local disk is only touched when the caller explicitly samples. FiftyOne is
optional (extras=["fiftyone"]) — used for visual inspection of small subsets.
"""

from .loaders.fivek import FIVEK_EXPERTS, stream_fivek
from .loaders.lsd import stream_lsd
from .loaders.sun import stream_sun

__all__ = [
    "FIVEK_EXPERTS",
    "stream_fivek",
    "stream_lsd",
    "stream_sun",
]

__version__ = "0.1.0"
