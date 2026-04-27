from typing import ClassVar

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class PexelsAdapter:
    name = "pexels"
    media_kinds: ClassVar[set[MediaKind]] = {"video"}
    BASE_URL = "https://api.pexels.com"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        if kind != "video":
            return []
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                f"{self.BASE_URL}/videos/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": limit},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for v in data.get("videos", []):
            files = v.get("video_files", [])
            if not files:
                continue
            # Pick highest resolution file
            best = max(files, key=lambda f: f.get("width", 0) * f.get("height", 0))
            out.append(
                Candidate(
                    source=self.name,
                    url=best["link"],
                    preview_url=v.get("image"),
                    kind="video",
                    duration=float(v.get("duration", 0)),
                    width=int(best.get("width", v.get("width", 0))),
                    height=int(best.get("height", v.get("height", 0))),
                    license="Pexels License",
                    attribution=(v.get("user") or {}).get("name"),
                )
            )
        return out
