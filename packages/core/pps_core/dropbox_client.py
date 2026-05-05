"""Dropbox client — list folder, download single/many files.

Token đọc từ env var DROPBOX_ACCESS_TOKEN. KHÔNG bao giờ hardcode token vào code.
SDK: pip install dropbox
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .utils import ensure_dir, safe_filename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DropboxFile:
    path: str
    name: str
    size: int

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)


class DropboxError(RuntimeError):
    pass


class DropboxClient:
    """Wrapper an toàn quanh dropbox SDK với retry + progress."""

    def __init__(
        self,
        access_token: str | None = None,
        *,
        max_retries: int = 3,
    ) -> None:
        token = access_token or os.getenv("DROPBOX_ACCESS_TOKEN")
        if not token:
            raise DropboxError(
                "Thiếu DROPBOX_ACCESS_TOKEN. Đặt vào .env hoặc env var, không hardcode trong code."
            )
        try:
            import dropbox  # type: ignore
        except ImportError as exc:
            raise DropboxError("Cần SDK 'dropbox'. Cài: pip install dropbox") from exc

        self._dbx = dropbox.Dropbox(token, timeout=30)
        self._dropbox = dropbox
        self._max_retries = max(1, max_retries)

    def list_folder(self, folder: str = "", *, recursive: bool = False) -> list[DropboxFile]:
        """Liệt kê file trong folder (root = chuỗi rỗng theo Dropbox API)."""
        files: list[DropboxFile] = []
        try:
            result = self._call(self._dbx.files_list_folder, folder, recursive=recursive)
            files.extend(self._extract_files(result))
            while result.has_more:
                result = self._call(self._dbx.files_list_folder_continue, result.cursor)
                files.extend(self._extract_files(result))
        except self._dropbox.exceptions.ApiError as exc:
            raise DropboxError(f"Dropbox list_folder lỗi: {exc}") from exc
        return files

    def _extract_files(self, result) -> list[DropboxFile]:  # type: ignore[no-untyped-def]
        FileMeta = self._dropbox.files.FileMetadata
        out: list[DropboxFile] = []
        for entry in result.entries:
            if isinstance(entry, FileMeta):
                out.append(
                    DropboxFile(
                        path=entry.path_display,
                        name=entry.name,
                        size=int(entry.size),
                    )
                )
        return out

    def download(
        self,
        dropbox_path: str,
        out_dir: str | Path = "downloads",
        *,
        local_name: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        out = ensure_dir(out_dir)
        name = safe_filename(local_name or Path(dropbox_path).name)
        target = out / name

        try:
            metadata, response = self._call(self._dbx.files_download, dropbox_path)
        except self._dropbox.exceptions.ApiError as exc:
            raise DropboxError(f"Dropbox files_download lỗi: {exc}") from exc

        total = int(getattr(metadata, "size", 0))
        bar = tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=target.name,
            disable=not show_progress,
        )
        try:
            with target.open("wb") as fh, bar:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
        finally:
            response.close()

        logger.info("Đã tải %s (%.2f MB)", target, total / (1024 * 1024))
        return target

    def download_many(
        self,
        dropbox_paths: Iterable[str],
        out_dir: str | Path = "downloads",
    ) -> list[Path]:
        results: list[Path] = []
        for path in dropbox_paths:
            try:
                results.append(self.download(path, out_dir=out_dir))
            except DropboxError as exc:
                logger.error("Bỏ qua %s do lỗi: %s", path, exc)
        return results

    def download_folder(
        self,
        folder: str = "",
        out_dir: str | Path = "downloads",
        *,
        recursive: bool = False,
    ) -> list[Path]:
        files = self.list_folder(folder, recursive=recursive)
        if not files:
            logger.warning("Folder %r rỗng", folder or "/")
            return []
        return self.download_many((f.path for f in files), out_dir=out_dir)

    def _call(self, func, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Retry với exponential backoff cho lỗi mạng/transient."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self._dropbox.exceptions.AuthError:
                raise  # auth fail, retry vô ích
            except self._dropbox.exceptions.ApiError:
                raise  # lỗi nghiệp vụ, không retry
            except Exception as exc:  # network/timeouts/rate
                last_exc = exc
                wait = min(2**attempt, 15)
                logger.warning(
                    "Dropbox call lần %d lỗi: %s — chờ %ds",
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise DropboxError(f"Hết retry: {last_exc}") from last_exc
