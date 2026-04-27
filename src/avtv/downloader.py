import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

Kind = Literal["video", "image"]


class Downloader:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path_for(self, url: str, kind: Kind) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        ext = Path(urlparse(url).path).suffix or (".mp4" if kind == "video" else ".jpg")
        sub = "clips" if kind == "video" else "images"
        target = self.cache_dir / sub / f"{digest}{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _do_get(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    async def fetch(self, url: str, kind: Kind) -> Path:
        target = self._path_for(url, kind)
        if target.exists() and target.stat().st_size > 0:
            return target
        data = await self._do_get(url)
        target.write_bytes(data)
        return target

    def fetch_sync(self, url: str, kind: Kind) -> Path:
        import asyncio

        return asyncio.run(self.fetch(url, kind))
