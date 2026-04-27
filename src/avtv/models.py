from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    idx: int
    start: float
    end: float
    text: str
    is_empty: bool

    @classmethod
    def from_text(cls, idx: int, start: float, end: float, text: str) -> "Block":
        stripped = text.strip()
        return cls(idx=idx, start=start, end=end, text=text, is_empty=not stripped)


VisualKind = Literal["video", "image", "continuation"]
MediaKind = Literal["video", "image"]


class VisualBrief(BaseModel):
    idx: int
    query_en: str
    fallback_query: str
    kind: VisualKind
    continuity_hint: str | None = None
    notes: str | None = None


class Candidate(BaseModel):
    source: str
    url: str
    preview_url: str | None = None
    kind: MediaKind
    duration: float | None = None
    width: int
    height: int
    license: str
    attribution: str | None = None


class Selection(BaseModel):
    idx: int
    source: str
    url: str
    kind: VisualKind
    trim_start: float | None = None
    trim_end: float | None = None
    speed_factor: float | None = None
    loop: bool = False
    attribution: str | None = None
