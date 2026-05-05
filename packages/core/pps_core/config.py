from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

InpaintBackendName = Literal["opencv", "lama"]
OpencvMethodName = Literal["telea", "ns"]
LamaDevice = Literal["auto", "cpu", "cuda", "mps"]

_VALID_BACKENDS = {"opencv", "lama"}
_VALID_OPENCV_METHODS = {"telea", "ns"}
_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True)
class Settings:
    unsplash_access_key: str | None
    inpaint_backend: InpaintBackendName
    opencv_method: OpencvMethodName
    opencv_radius: int
    lama_device: LamaDevice
    output_dir: Path
    log_level: str

    def require_unsplash(self) -> str:
        if not self.unsplash_access_key:
            raise RuntimeError(
                "UNSPLASH_ACCESS_KEY chưa được cấu hình. "
                "Tạo file .env từ .env.example và điền access key."
            )
        return self.unsplash_access_key


def _coerce_choice(value: str, valid: set[str], field: str, default: str) -> str:
    v = (value or default).strip().lower()
    if v not in valid:
        raise ValueError(f"{field} phải là một trong {sorted(valid)} (got {value!r})")
    return v


def _coerce_int(value: str, field: str, default: int, *, minimum: int = 1) -> int:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field} phải là số nguyên (got {value!r})") from exc
    if v < minimum:
        raise ValueError(f"{field} phải >= {minimum} (got {v})")
    return v


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    if env_file:
        load_dotenv(env_file, override=False)

    backend = _coerce_choice(
        os.getenv("INPAINT_BACKEND", "opencv"), _VALID_BACKENDS, "INPAINT_BACKEND", "opencv"
    )
    method = _coerce_choice(
        os.getenv("OPENCV_METHOD", "telea"), _VALID_OPENCV_METHODS, "OPENCV_METHOD", "telea"
    )
    device = _coerce_choice(
        os.getenv("LAMA_DEVICE", "auto"), _VALID_DEVICES, "LAMA_DEVICE", "auto"
    )
    radius = _coerce_int(os.getenv("OPENCV_RADIUS", "3"), "OPENCV_RADIUS", 3)
    output_dir = Path(os.getenv("OUTPUT_DIR", "output")).expanduser()
    log_level = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()

    settings = Settings(
        unsplash_access_key=os.getenv("UNSPLASH_ACCESS_KEY") or None,
        inpaint_backend=backend,  # type: ignore[arg-type]
        opencv_method=method,  # type: ignore[arg-type]
        opencv_radius=radius,
        lama_device=device,  # type: ignore[arg-type]
        output_dir=output_dir,
        log_level=log_level,
    )
    _configure_logging(settings.log_level)
    return settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
