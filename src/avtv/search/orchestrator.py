import asyncio

from avtv.models import Candidate, VisualBrief
from avtv.search.base import MediaKind, SearchAdapter, dedupe_candidates


class SearchOrchestrator:
    def __init__(self, adapters: list[SearchAdapter], top_k: int = 8) -> None:
        self.adapters = adapters
        self.top_k = top_k

    def _adapters_for(self, kind: MediaKind) -> list[SearchAdapter]:
        return [a for a in self.adapters if kind in a.media_kinds]

    async def _search_all(self, query: str, kind: MediaKind) -> list[Candidate]:
        relevant = self._adapters_for(kind)
        coros = [a.search(query, limit=self.top_k, kind=kind) for a in relevant]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: list[Candidate] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            out.extend(r)  # type: ignore[arg-type]
        return out

    async def search_for_brief(self, brief: VisualBrief) -> list[Candidate]:
        if brief.kind == "continuation":
            return []

        media_kind: MediaKind = "video" if brief.kind == "video" else "image"

        primary = await self._search_all(brief.query_en, media_kind)
        if primary:
            return dedupe_candidates(primary)

        fallback = await self._search_all(brief.fallback_query, media_kind)
        return dedupe_candidates(fallback)

    async def search_images_for_brief(self, brief: VisualBrief) -> list[Candidate]:
        """Search images using the brief's queries. Used by the selector for
        image_montage fallback (continuation briefs OR video pool exhausted).

        Always queries 'image' adapters regardless of brief.kind. Tries
        query_en first, then fallback_query. Empty if no image adapters or
        both queries return nothing.
        """
        primary = await self._search_all(brief.query_en, "image")
        if primary:
            return dedupe_candidates(primary)
        fallback = await self._search_all(brief.fallback_query, "image")
        return dedupe_candidates(fallback)

    async def search_for_briefs(
        self, briefs: list[VisualBrief], concurrency: int = 5
    ) -> dict[int, list[Candidate]]:
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(brief: VisualBrief) -> tuple[int, list[Candidate]]:
            async with sem:
                return brief.idx, await self.search_for_brief(brief)

        results = await asyncio.gather(*[_bounded(b) for b in briefs])
        return dict(results)
