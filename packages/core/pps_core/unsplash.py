"""Unsplash API client — search ảnh + tải full HD.

Đăng ký Access Key miễn phí: https://unsplash.com/developers
Đặt vào biến môi trường UNSPLASH_ACCESS_KEY (file .env).

API docs: https://unsplash.com/documentation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import requests
from tqdm import tqdm

from .utils import ensure_dir, safe_filename

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.unsplash.com"
DEFAULT_TIMEOUT = (5, 30)  # (connect, read)
DEFAULT_USER_AGENT = "watermark-toolkit/0.1 (+https://example.invalid)"

Orientation = Literal["landscape", "portrait", "squarish"]
ImageSize = Literal["raw", "full", "regular", "small", "thumb"]


@dataclass(frozen=True)
class UnsplashPhoto:
    id: str
    description: str | None
    width: int
    height: int
    urls: dict[str, str]
    download_location: str  # endpoint cần GET để báo download (theo TOS Unsplash)
    user_name: str

    def url(self, size: ImageSize = "raw") -> str:
        if size not in self.urls:
            raise KeyError(f"Không có size {size!r}. Có: {list(self.urls)}")
        return self.urls[size]


class UnsplashError(RuntimeError):
    pass


class UnsplashClient:
    """HTTP client với retry + tôn trọng rate-limit Unsplash (50 req/h demo, 5000 production)."""

    def __init__(
        self,
        access_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        if not access_key:
            raise ValueError("access_key không được rỗng")
        self._key = access_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Accept-Version": "v1",
                "Authorization": f"Client-ID {access_key}",
                "User-Agent": user_agent,
            }
        )

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> dict:
        url = f"{self._base}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Unsplash GET %s lần %d lỗi: %s", path, attempt, exc)
            else:
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    logger.warning("Rate-limited 429, chờ %ds", wait)
                    time.sleep(wait)
                    continue
                if 500 <= resp.status_code < 600:
                    wait = min(2 ** attempt, 15)
                    logger.warning("Server %d, chờ %ds", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    raise UnsplashError(
                        f"Unsplash {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()
            time.sleep(min(2 ** attempt, 10))
        raise UnsplashError(f"Hết retry cho {url}") from last_exc

    def search(
        self,
        query: str,
        *,
        per_page: int = 10,
        page: int = 1,
        orientation: Orientation | None = None,
    ) -> list[UnsplashPhoto]:
        if not query.strip():
            raise ValueError("query không được rỗng")
        params: dict[str, str | int] = {
            "query": query,
            "per_page": max(1, min(30, per_page)),
            "page": max(1, page),
        }
        if orientation:
            params["orientation"] = orientation
        data = self._get("/search/photos", params)
        results = data.get("results", [])
        return [_to_photo(item) for item in results]

    def get_photo(self, photo_id: str) -> UnsplashPhoto:
        data = self._get(f"/photos/{photo_id}")
        return _to_photo(data)

    def random(
        self,
        *,
        query: str | None = None,
        count: int = 1,
        orientation: Orientation | None = None,
    ) -> list[UnsplashPhoto]:
        params: dict[str, str | int] = {"count": max(1, min(30, count))}
        if query:
            params["query"] = query
        if orientation:
            params["orientation"] = orientation
        data = self._get("/photos/random", params)
        # Khi count >= 1, API trả list; nhưng count=1 cũng trả list nếu có param count.
        if isinstance(data, dict):
            data = [data]
        return [_to_photo(item) for item in data]

    def trigger_download(self, photo: UnsplashPhoto) -> None:
        """Bắt buộc theo Unsplash TOS: ping download_location trước khi tải file."""
        try:
            self._session.get(photo.download_location, timeout=self._timeout)
        except requests.RequestException as exc:
            logger.warning("Không trigger được download tracking: %s", exc)

    def download(
        self,
        photo: UnsplashPhoto,
        *,
        size: ImageSize = "raw",
        out_dir: str | Path = "downloads",
        filename: str | None = None,
        chunk_size: int = 64 * 1024,
        show_progress: bool = True,
    ) -> Path:
        self.trigger_download(photo)
        url = photo.url(size)
        out = ensure_dir(out_dir)
        name = filename or f"unsplash_{photo.id}_{size}.jpg"
        target = out / safe_filename(name)

        with self._session.get(url, stream=True, timeout=self._timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            iterator = resp.iter_content(chunk_size=chunk_size)
            bar = tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=target.name,
                disable=not show_progress,
            )
            with target.open("wb") as fh, bar:
                for chunk in iterator:
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
        logger.info("Đã tải %s (%d bytes)", target, target.stat().st_size)
        return target

    def download_many(
        self,
        photos: Iterable[UnsplashPhoto],
        *,
        size: ImageSize = "raw",
        out_dir: str | Path = "downloads",
    ) -> list[Path]:
        return [self.download(p, size=size, out_dir=out_dir) for p in photos]


def _to_photo(item: dict) -> UnsplashPhoto:
    user = item.get("user") or {}
    links = item.get("links") or {}
    return UnsplashPhoto(
        id=str(item.get("id", "")),
        description=item.get("description") or item.get("alt_description"),
        width=int(item.get("width", 0)),
        height=int(item.get("height", 0)),
        urls=dict(item.get("urls") or {}),
        download_location=str(links.get("download_location", "")),
        user_name=str(user.get("name", "")),
    )
