from typing import Any, ClassVar

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind

VIDEO_FORMATS = {"h.264", "mp4", "mpeg4", "h.264 hd"}
IMAGE_FORMATS = {"jpeg", "jpg", "png"}


class ArchiveOrgAdapter:
    name = "archive_org"
    media_kinds: ClassVar[set[MediaKind]] = {"video", "image"}
    SEARCH_URL = "https://archive.org/advancedsearch.php"
    METADATA_URL = "https://archive.org/metadata"

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        mediatype = "movies" if kind == "video" else "image"
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("q", f"{query} AND mediatype:{mediatype}"),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "creator"),
            ("rows", limit),
            ("output", "json"),
        ]
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(self.SEARCH_URL, params=params)
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])

            out: list[Candidate] = []
            for d in docs:
                ident = d["identifier"]
                meta_r = await client.get(f"{self.METADATA_URL}/{ident}")
                if meta_r.status_code != 200:
                    continue
                files = meta_r.json().get("files", [])
                pick = self._pick_file(files, kind)
                if not pick:
                    continue
                file_url = f"https://archive.org/download/{ident}/{pick['name']}"
                out.append(
                    Candidate(
                        source=self.name,
                        url=file_url,
                        preview_url=None,
                        kind=kind,
                        duration=float(pick.get("length", 0)) if kind == "video" else None,
                        width=int(pick.get("width", 0) or 0),
                        height=int(pick.get("height", 0) or 0),
                        license="Public Domain / various",
                        attribution=d.get("creator"),
                    )
                )
            return out

    @staticmethod
    def _pick_file(files: list[dict[str, Any]], kind: MediaKind) -> dict[str, Any] | None:
        target = VIDEO_FORMATS if kind == "video" else IMAGE_FORMATS
        for f in files:
            fmt = (f.get("format") or "").lower()
            if any(t in fmt for t in target):
                return f
        return None
