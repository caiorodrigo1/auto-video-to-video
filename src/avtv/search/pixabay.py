from typing import ClassVar

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class PixabayAdapter:
    name = "pixabay"
    media_kinds: ClassVar[set[MediaKind]] = {"video", "image"}
    VIDEO_URL = "https://pixabay.com/api/videos/"
    IMAGE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        if kind == "video":
            return await self._video(query, limit)
        return await self._image(query, limit)

    async def _video(self, query: str, limit: int) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.VIDEO_URL,
                params={"key": self.api_key, "q": query, "per_page": max(limit, 3)},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("hits", [])[:limit]:
            videos = h.get("videos", {})
            # Prefer "large", fall back to "medium"
            best = videos.get("large") or videos.get("medium")
            if not best:
                continue
            out.append(
                Candidate(
                    source=self.name,
                    url=best["url"],
                    preview_url=None,
                    kind="video",
                    duration=float(h.get("duration", 0)),
                    width=int(best.get("width", 0)),
                    height=int(best.get("height", 0)),
                    license="Pixabay License",
                    attribution=h.get("user"),
                )
            )
        return out

    async def _image(self, query: str, limit: int) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.IMAGE_URL,
                params={
                    "key": self.api_key,
                    "q": query,
                    "per_page": max(limit, 3),
                    "image_type": "photo",
                },
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("hits", [])[:limit]:
            out.append(
                Candidate(
                    source=self.name,
                    url=h["largeImageURL"],
                    preview_url=None,
                    kind="image",
                    duration=None,
                    width=int(h.get("imageWidth", 0)),
                    height=int(h.get("imageHeight", 0)),
                    license="Pixabay License",
                    attribution=h.get("user"),
                )
            )
        return out
