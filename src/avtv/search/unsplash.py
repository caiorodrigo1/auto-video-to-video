from typing import ClassVar

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class UnsplashAdapter:
    name = "unsplash"
    media_kinds: ClassVar[set[MediaKind]] = {"image"}
    URL = "https://api.unsplash.com/search/photos"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "image"
    ) -> list[Candidate]:
        if kind != "image":
            return []
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.URL,
                headers={"Authorization": f"Client-ID {self.api_key}"},
                params={"query": query, "per_page": limit, "orientation": "landscape"},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("results", []):
            out.append(
                Candidate(
                    source=self.name,
                    url=h["urls"]["full"],
                    preview_url=None,
                    kind="image",
                    duration=None,
                    width=int(h.get("width", 0)),
                    height=int(h.get("height", 0)),
                    license="Unsplash License",
                    attribution=(h.get("user") or {}).get("name"),
                )
            )
        return out
