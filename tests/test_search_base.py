from avtv.models import Candidate
from avtv.search.base import SearchAdapter, dedupe_candidates  # noqa: F401


def _make(url: str, w: int = 1920, h: int = 1080) -> Candidate:
    return Candidate(
        source="test", url=url, kind="video", duration=8.0,
        width=w, height=h, license="x",
    )


def test_dedupe_keeps_first_occurrence_per_url():
    cs = [_make("a"), _make("b"), _make("a"), _make("c")]
    out = dedupe_candidates(cs)
    assert [c.url for c in out] == ["a", "b", "c"]
