"""GPU auto-detection cho Real-ESRGAN + spandrel models.

Priority: CUDA (NVIDIA) > MPS (Apple) > DirectML (AMD/Intel Windows) > CPU.

Mỗi tier có expected_speedup tương đối vs CPU baseline:
    cuda      → 10-100x (RTX 30/40-series)
    mps       → 10-20x (Apple M1/M2/M3)
    directml  → 3-10x (AMD/Intel modern GPU)
    cpu       → 1x baseline

Caller dùng:
    >>> from pps_core.device import detect_best_device, get_torch_device
    >>> info = detect_best_device()
    >>> info.kind      # "cuda" | "mps" | "directml" | "cpu"
    >>> info.label     # human-friendly
    >>> dev = get_torch_device()   # torch.device(...) compatible với mọi backend
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceInfo",
    "detect_best_device",
    "get_torch_device",
    "describe_device",
]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Kết quả detection."""

    kind: str  # "cuda" | "mps" | "directml" | "cpu"
    label: str  # display name (e.g. "NVIDIA GeForce RTX 3060")
    vram_gb: float | None  # None nếu unknown
    expected_speedup: str  # "10-100x" | "1x" ...

    @property
    def is_gpu(self) -> bool:
        return self.kind != "cpu"

    def short_summary(self) -> str:
        if self.vram_gb:
            return f"{self.label} ({self.vram_gb:.1f}GB, ~{self.expected_speedup})"
        return f"{self.label} (~{self.expected_speedup})"


@lru_cache(maxsize=1)
def detect_best_device() -> DeviceInfo:
    """Auto-detect tốt nhất → trả DeviceInfo.

    Cache 1 lần — re-import torch backends nhiều lần waste.
    Override qua env `PPS_FORCE_DEVICE=cpu` nếu cần ép.
    """
    force = os.environ.get("PPS_FORCE_DEVICE", "").lower().strip()
    if force in {"cpu", "cuda", "mps", "directml"}:
        logger.info("PPS_FORCE_DEVICE=%s — ép backend", force)
        if force == "cpu":
            return _cpu_info()
        if force == "cuda":
            return _try_cuda() or _cpu_info()
        if force == "mps":
            return _try_mps() or _cpu_info()
        if force == "directml":
            return _try_directml() or _cpu_info()

    # Normal priority chain
    for probe in (_try_cuda, _try_mps, _try_directml):
        info = probe()
        if info is not None:
            return info

    return _cpu_info()


def _try_cuda() -> DeviceInfo | None:
    """NVIDIA CUDA — fastest path."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)
    except Exception as exc:
        logger.warning("CUDA detect fail: %s", exc)
        return None

    # Estimate speedup theo class card (rough heuristic)
    speedup = "10-100x"
    name_lower = name.lower()
    if any(k in name_lower for k in ("rtx 40", "rtx 30", "a100", "h100")):
        speedup = "50-100x"
    elif any(k in name_lower for k in ("rtx 20", "rtx 1", "gtx 16", "gtx 1080", "gtx 1070")):
        speedup = "20-50x"
    elif any(k in name_lower for k in ("gtx 10", "gtx 9", "mx ")):
        speedup = "10-30x"

    return DeviceInfo(kind="cuda", label=name, vram_gb=vram_gb, expected_speedup=speedup)


def _try_mps() -> DeviceInfo | None:
    """Apple Silicon Metal Performance Shaders."""
    try:
        import torch
    except ImportError:
        return None
    if not hasattr(torch.backends, "mps"):
        return None
    if not torch.backends.mps.is_available():
        return None

    # Get chip name từ system_profiler nếu macOS
    label = "Apple Silicon GPU"
    if sys.platform == "darwin":
        try:
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            chip = r.stdout.strip()
            if chip:
                label = f"Apple {chip}"
        except Exception:
            pass

    return DeviceInfo(kind="mps", label=label, vram_gb=None, expected_speedup="10-20x")


def _try_directml() -> DeviceInfo | None:
    """DirectML cho AMD/Intel GPU trên Windows. Cần `pip install torch-directml`."""
    if sys.platform != "win32":
        return None
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if torch_directml.device_count() == 0:
            return None
        # Pick device 0 (primary discrete usually)
        label = torch_directml.device_name(0)
        # VRAM not exposed by torch-directml — fallback to wmic via Win32_VideoController
        vram_gb = _query_amd_intel_vram(label)
        # Speedup heuristic theo GPU vendor
        label_lower = label.lower()
        speedup = "3-10x"
        if any(k in label_lower for k in ("rx 7", "rx 6", "rx 5", "intel arc")):
            speedup = "10-30x"
        elif any(k in label_lower for k in ("rx 4", "vega", "rx 580", "intel iris xe")):
            speedup = "5-15x"
        elif "r5" in label_lower or "r7 2" in label_lower:
            speedup = "2-5x"  # older entry-level
        return DeviceInfo(
            kind="directml", label=label, vram_gb=vram_gb, expected_speedup=speedup,
        )
    except Exception as exc:
        logger.warning("DirectML probe fail: %s", exc)
        return None


def _query_amd_intel_vram(label: str) -> float | None:
    """Best-effort VRAM lookup qua wmic (Windows only). Trả None nếu fail."""
    try:
        import subprocess
        r = subprocess.run(
            ["wmic", "path", "Win32_VideoController", "get", "Name,AdapterRAM", "/format:list"],
            capture_output=True, text=True, timeout=3,
        )
        cur_name = ""
        cur_ram = 0
        label_lc = label.lower()
        for line in r.stdout.splitlines():
            if line.startswith("Name="):
                cur_name = line[5:].strip()
            elif line.startswith("AdapterRAM="):
                try:
                    cur_ram = int(line[11:].strip() or 0)
                except ValueError:
                    cur_ram = 0
                if cur_name and cur_ram and label_lc in cur_name.lower():
                    return cur_ram / (1024**3)
        return None
    except Exception:
        return None


def _cpu_info() -> DeviceInfo:
    """CPU fallback."""
    label = "CPU"
    try:
        import platform
        label = f"CPU ({platform.processor() or platform.machine()})"
    except Exception:
        pass
    return DeviceInfo(kind="cpu", label=label, vram_gb=None, expected_speedup="1x")


def get_torch_device():
    """Return torch.device tương ứng best detected.

    DirectML special case: trả torch_directml.device() (object riêng, không phải torch.device).
    Caller phải đối xử polymorphic — cả 2 đều có .type và truyền vào tensor.to() được.
    """
    info = detect_best_device()
    if info.kind == "cuda":
        import torch
        return torch.device("cuda:0")
    if info.kind == "mps":
        import torch
        return torch.device("mps")
    if info.kind == "directml":
        import torch_directml
        return torch_directml.device(0)
    import torch
    return torch.device("cpu")


def describe_device() -> str:
    """1-line human summary (cho UI status bar)."""
    info = detect_best_device()
    icon = {
        "cuda": "🟢",
        "mps": "🍎",
        "directml": "🟦",
        "cpu": "⚪",
    }.get(info.kind, "•")
    return f"{icon} {info.short_summary()}"
