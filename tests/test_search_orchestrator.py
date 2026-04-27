import asyncio  # noqa: F401

import pytest

from avtv.models import Candidate, VisualBrief
from avtv.search.base import SearchAdapter  # noqa: F401
from avtv.search.orchestrator import SearchOrchestrator


class FakeAdapter:
    def __init__(self, name: str, media_kinds: set, results: dict):
        self.name = name
        self.media_kinds = media_kinds
        self._results = results  # {(query, kind): [Candidate, ...]}
        self.calls: list = []

    async def search(self, query, limit=10, kind="video"):
        self.calls.append((query, kind))
        return self._results.get((query, kind), [])


def _c(source, url):
    return Candidate(
        source=source, url=url, kind="video", duration=8.0,
        width=1920, height=1080, license="x",
    )


@pytest.mark.asyncio
async def test_orchestrator_combines_video_adapters():
    a1 = FakeAdapter("a", {"video"}, {("forest", "video"): [_c("a", "u1")]})
    a2 = FakeAdapter("b", {"video"}, {("forest", "video"): [_c("b", "u2")]})
    orch = SearchOrchestrator(adapters=[a1, a2])

    brief = VisualBrief(
        idx=1, query_en="forest", fallback_query="nature", kind="video",
    )
    results = await orch.search_for_brief(brief)
    urls = {c.url for c in results}
    assert urls == {"u1", "u2"}


@pytest.mark.asyncio
async def test_orchestrator_uses_fallback_when_primary_empty():
    a = FakeAdapter("a", {"video"}, {("nature", "video"): [_c("a", "u-nature")]})
    orch = SearchOrchestrator(adapters=[a])

    brief = VisualBrief(
        idx=1, query_en="forest", fallback_query="nature", kind="video",
    )
    results = await orch.search_for_brief(brief)
    assert len(results) == 1
    assert results[0].url == "u-nature"
    # Both queries were tried
    assert a.calls == [("forest", "video"), ("nature", "video")]


@pytest.mark.asyncio
async def test_orchestrator_skips_continuation():
    a = FakeAdapter("a", {"video"}, {})
    orch = SearchOrchestrator(adapters=[a])
    brief = VisualBrief(
        idx=1, query_en="x", fallback_query="y", kind="continuation",
    )
    results = await orch.search_for_brief(brief)
    assert results == []
    assert a.calls == []
