from typing import Literal, Protocol

from avtv.models import Candidate

MediaKind = Literal["video", "image"]


class SearchAdapter(Protocol):
    name: str
    media_kinds: set[MediaKind]

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]: ...


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Dedupe by URL preserving first occurrence."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        if c.url in seen:
            continue
        seen.add(c.url)
        out.append(c)
    return out
