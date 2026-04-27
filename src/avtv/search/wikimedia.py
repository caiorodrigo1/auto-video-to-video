import re
from typing import ClassVar

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class WikimediaAdapter:
    name = "wikimedia"
    media_kinds: ClassVar[set[MediaKind]] = {"video", "image"}
    API = "https://commons.wikimedia.org/w/api.php"

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "image"
    ) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.get(
                self.API,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srnamespace": "6",
                    "srsearch": f"{query} filetype:{'video' if kind == 'video' else 'bitmap'}",
                    "srlimit": limit,
                },
            )
            search.raise_for_status()
            titles = [s["title"] for s in search.json().get("query", {}).get("search", [])]
            if not titles:
                return []

            info = await client.get(
                self.API,
                params={
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(titles),
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata",
                },
            )
            info.raise_for_status()
            pages = info.json().get("query", {}).get("pages", {})

        out: list[Candidate] = []
        for page in pages.values():
            iis = page.get("imageinfo", [])
            if not iis:
                continue
            ii = iis[0]
            ext = ii.get("extmetadata", {}) or {}
            license_str = (ext.get("LicenseShortName") or {}).get("value", "Unknown")
            artist_html = (ext.get("Artist") or {}).get("value", "") or ""
            artist = re.sub(r"<[^>]+>", "", artist_html).strip() or None
            out.append(
                Candidate(
                    source=self.name,
                    url=ii["url"],
                    preview_url=None,
                    kind=kind,
                    duration=None,
                    width=int(ii.get("width", 0)),
                    height=int(ii.get("height", 0)),
                    license=license_str,
                    attribution=artist,
                )
            )
        return out
