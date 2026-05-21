"""pps-wincei-hdr — HDR bracket fusion cho ảnh BĐS.

Pipeline:
    Sony A7M4 AEB 3-shot (-3, 0, +3 EV)
        ├─► Group by EXIF DateTimeOriginal ±2s + EV signature
        ├─► AlignMTB (handheld jitter compensation)
        ├─► createMergeMertens (exposure fusion, không cần CRF)
        ├─► Output: 1 LDR JPG quality 98 + EXIF từ EV=0 shot
        └─► (Optional) → pipe vào pps-wincei window+ceiling fix
"""

from .__version__ import __version__
from .bracket_detect import BracketGroup, detect_brackets
from .fusion import fuse_mertens, align_brackets

__all__ = [
    "__version__",
    "BracketGroup",
    "detect_brackets",
    "fuse_mertens",
    "align_brackets",
]
