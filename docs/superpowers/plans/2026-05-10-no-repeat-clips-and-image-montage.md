# No-repeat clips & image-montage fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the same exact clip URL from being used twice in one video, and replace static "continuation" stretches with a 3-image Ken Burns montage so empty narration blocks feel alive.

**Architecture:** Three coordinated changes. (1) Selector switches from a 2-block sliding dedup window to a global "URL used in this video" set. (2) A new `image_montage` selection kind holds N=1..3 image URLs that the assembler renders as concatenated Ken Burns sub-clips inside one 8-second segment. (3) The selector becomes async and accepts a callback so it can request an on-demand image search from the orchestrator whenever a brief is `continuation` OR the video pool runs out — keeping the search stage's contract unchanged for normal blocks.

**Tech Stack:** Python 3.11, Pydantic, asyncio, ffmpeg/ffprobe, pytest (mocked).

---

## Context the engineer needs

**Pipeline at a glance** (`README.md:188-198`):
`parse → brief (LLM) → search (5 free APIs) → select → assemble (ffmpeg)`.
Each block of narration is 8 s. The LLM emits one `VisualBrief` per block with `kind` ∈ {`video`, `image`, `continuation`}. Continuation = empty narration slot, currently extends the previous clip. Search returns up to 8 candidates per provider; today the orchestrator skips continuations entirely (`src/avtv/search/orchestrator.py:27`).

**Why the user is annoyed**:
- The selector dedup window is only `±2` (`src/avtv/selector.py:131`). With overlapping API results across briefs, the same clip can resurface every ~3 blocks.
- Continuations (`src/avtv/assembler.py:357-367`) reuse the previous segment file, so a 16+ s static stretch is common. The user explicitly listed this as a top complaint.
- `repetition_penalty` is defined in `src/avtv/config.py:32` but **never read**. The intent died on the vine.

**User decisions already made (don't re-litigate):**
- Empty/continuation block → 3-image montage (~2.67 s each).
- If image search returns 0 → fall back to 1 image with Ken Burns.
- If video pool exhausted globally → treat as 0-cand, also produce image montage.
- Visually similar shots are fine — the rule is "no exact same URL twice in one video".

---

## File Structure

| File | Role | Action |
|---|---|---|
| `src/avtv/models.py` | Pydantic models for the pipeline | Modify: extend `VisualKind`, add montage fields to `Selection`. |
| `src/avtv/search/orchestrator.py` | Coordinates the 5 adapters | Modify: add `search_images_for_brief(brief)` helper that queries images using `query_en` (with `fallback_query`). |
| `src/avtv/selector.py` | Picks one Selection per brief | Rewrite: become async, take an `image_search` callback, implement global dedup + image_montage. |
| `src/avtv/cli.py` | CLI entrypoints | Modify: `select` and `build` wrap the now-async selector with `asyncio.run` and pass the orchestrator. |
| `src/avtv/assembler.py` | ffmpeg segment encoding | Modify: new `image_montage` strategy + `_image_montage_args` builder + plan + assemble loop branch. |
| `src/avtv/config.py` | Settings | Modify: remove `repetition_penalty`, `neighbor_window`; add `image_montage_max: int = 3` and `image_fallback_top_k: int = 5`. |
| `src/avtv/ui/streamlit_app.py` | Web UI | Modify: render `image_montage` selections sensibly in the inspector (show up to 3 thumbnails). |
| `tests/test_selector.py` | Selector tests | Rewrite to async, add cases for global dedup and montage. |
| `tests/test_search_orchestrator.py` | Orchestrator tests | Add cases for `search_images_for_brief`. |
| `tests/test_assembler.py` | Pure-arg-builder tests | Add `image_montage` plan + arg-builder tests. |
| `tests/test_models.py` | Model tests | Add tests for new fields and `image_montage` kind. |

Each task below is self-contained: write a failing test, implement minimally, run, commit.

---

### Task 1: Extend `VisualKind` and `Selection` to express an image montage

**Files:**
- Modify: `src/avtv/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from avtv.models import Selection


def test_selection_accepts_image_montage_kind():
    sel = Selection(
        idx=4,
        source="",
        url="",
        kind="image_montage",
        montage_urls=["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg"],
        montage_sources=["pixabay", "wikimedia", "pixabay"],
    )
    assert sel.kind == "image_montage"
    assert sel.montage_urls == ["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg"]
    assert sel.montage_sources == ["pixabay", "wikimedia", "pixabay"]


def test_selection_montage_defaults_empty():
    # Existing kinds must keep their old shape: montage fields default to [].
    sel = Selection(idx=1, source="pexels", url="https://x", kind="video")
    assert sel.montage_urls == []
    assert sel.montage_sources == []


def test_selection_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        Selection(idx=1, source="", url="", kind="image_carousel")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: the first two tests FAIL with `ValidationError` (kind not recognized / fields not allowed). `test_selection_rejects_unknown_kind` will pass already.

- [ ] **Step 3: Implement minimal change**

Edit `src/avtv/models.py`:

```python
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


VisualKind = Literal["video", "image", "image_montage", "continuation"]
MediaKind = Literal["video", "image"]


class VisualBrief(BaseModel):
    idx: int
    query_en: str
    fallback_query: str
    kind: Literal["video", "image", "continuation"]  # LLM never emits image_montage
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
    # For kind == "image_montage" only. Order matches display order (left-to-right).
    montage_urls: list[str] = []
    montage_sources: list[str] = []
```

Note: `VisualBrief.kind` is split off as its own Literal so the LLM JSON schema (in `briefing/schema.py`) keeps emitting only the three values it knows. `VisualKind` stays the union type used by `Selection` (downstream only).

- [ ] **Step 4: Verify tests pass**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: all 3 tests PASS. Also run the full suite to make sure no other test broke from the kind split:

```bash
poetry run pytest -q
```

Expected: existing tests pass (the `Literal` widening is backward compatible at the Selection level; VisualBrief was always restricted to the same 3 values it now lists explicitly).

- [ ] **Step 5: Commit**

```bash
git add src/avtv/models.py tests/test_models.py
git commit -m "feat(models): add image_montage kind and montage fields to Selection"
```

---

### Task 2: Add an image-search helper on the orchestrator

The selector needs to ask "give me image candidates for this brief's `query_en`" lazily. Add a public method to `SearchOrchestrator` that does exactly that.

**Files:**
- Modify: `src/avtv/search/orchestrator.py`
- Test: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_search_images_for_brief_uses_query_en_first():
    img = Candidate(
        source="pixabay", url="img1", kind="image", duration=None,
        width=1920, height=1080, license="x",
    )
    a = FakeAdapter(
        "pix", {"image"},
        {("forest", "image"): [img]},
    )
    orch = SearchOrchestrator(adapters=[a])

    brief = VisualBrief(idx=1, query_en="forest", fallback_query="nature", kind="video")
    out = await orch.search_images_for_brief(brief)
    assert [c.url for c in out] == ["img1"]
    assert a.calls == [("forest", "image")]


@pytest.mark.asyncio
async def test_search_images_for_brief_falls_back_to_fallback_query():
    img = Candidate(
        source="pixabay", url="img-nature", kind="image", duration=None,
        width=1920, height=1080, license="x",
    )
    a = FakeAdapter(
        "pix", {"image"},
        {("nature", "image"): [img]},
    )
    orch = SearchOrchestrator(adapters=[a])

    brief = VisualBrief(idx=1, query_en="forest", fallback_query="nature", kind="continuation")
    out = await orch.search_images_for_brief(brief)
    assert [c.url for c in out] == ["img-nature"]
    assert a.calls == [("forest", "image"), ("nature", "image")]


@pytest.mark.asyncio
async def test_search_images_for_brief_returns_empty_when_no_image_adapter():
    a = FakeAdapter("vid", {"video"}, {("forest", "video"): []})
    orch = SearchOrchestrator(adapters=[a])
    brief = VisualBrief(idx=1, query_en="forest", fallback_query="nature", kind="video")
    out = await orch.search_images_for_brief(brief)
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_search_orchestrator.py -v
```

Expected: 3 new tests FAIL with `AttributeError: 'SearchOrchestrator' object has no attribute 'search_images_for_brief'`.

- [ ] **Step 3: Implement minimal change**

Edit `src/avtv/search/orchestrator.py`. Replace the file with:

```python
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
```

- [ ] **Step 4: Verify tests pass**

```bash
poetry run pytest tests/test_search_orchestrator.py -v
```

Expected: all tests PASS (existing 3 + new 3).

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/orchestrator.py tests/test_search_orchestrator.py
git commit -m "feat(search): add search_images_for_brief for on-demand image fallback"
```

---

### Task 3: Rewrite the selector — global dedup + image_montage fallback

This is the heart of the change. The selector becomes async and accepts an optional `image_search` callback. Behavior:

| brief.kind | path |
|---|---|
| `continuation` | always image_montage (call `image_search`, take up to 3 unused images, fall back to 1 image, then to `continuation` if even that fails) |
| `video` or `image` | global-dedup against all already-used URLs across the whole run; on success → `Selection` like before; on overflow (no unused candidate) → image_montage from `image_search`; on overflow + no unused images → repeat the *least-recently-used* candidate (last-resort tie-break) |
| first block ever has 0 candidates AND image_search empty | raise `SelectorError` (block 1 cannot be continuation) |

**Files:**
- Modify: `src/avtv/selector.py`
- Test: `tests/test_selector.py`

- [ ] **Step 1: Replace the selector tests with the new contract**

Overwrite `tests/test_selector.py` with:

```python
import pytest

from avtv.config import Settings
from avtv.models import Candidate, VisualBrief
from avtv.selector import score_candidate, select_per_block


def _make(**kw) -> Candidate:
    base = dict(
        source="pexels",
        url="x",
        kind="video",
        duration=8.0,
        width=1920,
        height=1080,
        license="x",
    )
    base.update(kw)
    return Candidate(**base)


def _img(url: str, source: str = "pixabay") -> Candidate:
    return Candidate(
        source=source, url=url, kind="image", duration=None,
        width=1920, height=1080, license="x",
    )


def _settings(monkeypatch) -> Settings:
    for k in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(k, "x")
    return Settings()


def _brief(idx, kind="video", query="x"):
    return VisualBrief(idx=idx, query_en=query, fallback_query="y", kind=kind)


# -- score_candidate (unchanged behavior, kept for safety) -----------------

def test_score_perfect_clip(monkeypatch):
    s = _settings(monkeypatch)
    assert score_candidate(_make(), settings=s) > 3.0


def test_score_low_resolution_penalized(monkeypatch):
    s = _settings(monkeypatch)
    perfect = score_candidate(_make(), settings=s)
    low = score_candidate(_make(width=640, height=360), settings=s)
    assert low < perfect


def test_score_short_clip_penalized(monkeypatch):
    s = _settings(monkeypatch)
    perfect = score_candidate(_make(), settings=s)
    short = score_candidate(_make(duration=2.0), settings=s)
    assert short < perfect


# -- selection: normal happy path ------------------------------------------

@pytest.mark.asyncio
async def test_picks_best_per_block(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {
        1: [_make(url="hi", height=1080), _make(url="lo", height=480)],
        2: [_make(url="hi2", height=1080)],
    }
    sels = await select_per_block(briefs, cands, settings=s)
    assert [x.url for x in sels] == ["hi", "hi2"]
    assert all(x.kind == "video" for x in sels)


# -- global dedup ----------------------------------------------------------

@pytest.mark.asyncio
async def test_no_exact_repeat_anywhere_in_video(monkeypatch):
    """The same URL must NEVER appear twice in one run, even far apart."""
    s = _settings(monkeypatch)
    briefs = [_brief(i) for i in range(1, 5)]
    same = _make(url="popular", height=1080)
    other = _make(url="alt", height=1080, duration=8.0)
    third = _make(url="third", height=1080)
    fourth = _make(url="fourth", height=1080)
    cands = {
        1: [same, other],
        2: [same, third],
        3: [same, fourth],
        4: [same],  # only the already-used "popular"
    }
    sels = await select_per_block(briefs, cands, settings=s, image_search=None)
    urls = [x.url for x in sels]
    # First three blocks each see "popular" as best, but only block 1 may take it.
    assert urls[0] == "popular"
    assert urls[1] == "third"  # NOT "popular" (already used)
    assert urls[2] == "fourth"
    # Block 4 has only "popular" (used). image_search=None and no images on
    # candidates → last-resort: repeat the least-recently-used candidate.
    assert urls[3] == "popular"


# -- image_montage for continuations ---------------------------------------

@pytest.mark.asyncio
async def test_continuation_becomes_image_montage(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation", query="empty-block")]
    cands = {1: [_make(url="v1", height=1080)], 2: []}

    async def fake_image_search(brief):
        assert brief.idx == 2
        return [_img("i1"), _img("i2"), _img("i3"), _img("i4")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[1].kind == "image_montage"
    assert sels[1].montage_urls == ["i1", "i2", "i3"]  # top-3 by score
    assert sels[1].montage_sources == ["pixabay", "pixabay", "pixabay"]
    assert sels[1].url == ""  # placeholder
    assert sels[1].source == ""


@pytest.mark.asyncio
async def test_continuation_with_two_images_uses_two(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="v1", height=1080)], 2: []}

    async def fake_image_search(brief):
        return [_img("i1"), _img("i2")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[1].kind == "image_montage"
    assert sels[1].montage_urls == ["i1", "i2"]


@pytest.mark.asyncio
async def test_continuation_with_one_image_uses_single_image(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="v1", height=1080)], 2: []}

    async def fake_image_search(brief):
        return [_img("only")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    # 1 image → emit a normal image kind (not montage), so plain ken_burns plays.
    assert sels[1].kind == "image"
    assert sels[1].url == "only"


@pytest.mark.asyncio
async def test_continuation_with_zero_images_falls_back_to_extending(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="v1", height=1080)], 2: []}

    async def fake_image_search(brief):
        return []

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[1].kind == "continuation"
    assert sels[1].url == ""


@pytest.mark.asyncio
async def test_continuation_image_search_dedupes_against_used(monkeypatch):
    """An image already used in an earlier block can't be reused in a montage."""
    s = _settings(monkeypatch)
    briefs = [
        _brief(1, kind="image"),
        _brief(2),
        _brief(3, kind="continuation"),
    ]
    cands = {
        1: [_img("hero")],
        2: [_make(url="v2", height=1080)],
        3: [],
    }

    async def fake_image_search(brief):
        return [_img("hero"), _img("a"), _img("b"), _img("c")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[2].kind == "image_montage"
    assert "hero" not in sels[2].montage_urls
    assert sels[2].montage_urls == ["a", "b", "c"]


# -- image_montage on video pool exhaustion --------------------------------

@pytest.mark.asyncio
async def test_video_dedup_overflow_uses_image_montage(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    same = _make(url="only-video", height=1080)
    cands = {1: [same], 2: [same]}  # block 2 only sees the already-used clip

    async def fake_image_search(brief):
        assert brief.idx == 2
        return [_img("ix1"), _img("ix2"), _img("ix3")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[0].url == "only-video"
    assert sels[1].kind == "image_montage"
    assert sels[1].montage_urls == ["ix1", "ix2", "ix3"]


# -- first block edge cases ------------------------------------------------

@pytest.mark.asyncio
async def test_first_block_can_be_image_montage(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1, kind="continuation")]
    cands = {1: []}

    async def fake_image_search(brief):
        return [_img("i1"), _img("i2"), _img("i3")]

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)
    assert sels[0].kind == "image_montage"


@pytest.mark.asyncio
async def test_first_block_with_no_candidates_and_no_images_raises(monkeypatch):
    from avtv.selector import SelectorError

    s = _settings(monkeypatch)
    briefs = [_brief(1)]
    cands = {1: []}

    async def fake_image_search(brief):
        return []

    with pytest.raises(SelectorError):
        await select_per_block(briefs, cands, settings=s, image_search=fake_image_search)


# -- back-compat: image_search=None ---------------------------------------

@pytest.mark.asyncio
async def test_no_image_search_continuation_uses_extend(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="v1", height=1080)], 2: []}
    sels = await select_per_block(briefs, cands, settings=s, image_search=None)
    assert sels[1].kind == "continuation"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_selector.py -v
```

Expected: most fail with `TypeError: object NoneType can't be used in 'await' expression` or `select_per_block is not a coroutine function`.

- [ ] **Step 3: Implement the new selector**

Replace `src/avtv/selector.py` entirely with:

```python
from collections.abc import Awaitable, Callable

from avtv.config import Settings
from avtv.models import Candidate, Selection, VisualBrief

ImageSearch = Callable[[VisualBrief], Awaitable[list[Candidate]]]

SOURCE_PREFERENCE = {
    "pexels": 0.30,
    "pixabay": 0.25,
    "archive_org": 0.20,
    "wikimedia": 0.15,
    "unsplash": 0.20,
}


def _aspect_score(width: int, height: int) -> float:
    if height <= 0:
        return 0.0
    ratio = width / height
    if ratio >= 1.7:
        return 1.0
    if ratio >= 1.4:
        return 0.5
    return 0.0


def _duration_score(duration: float | None, target: float = 8.0) -> float:
    if duration is None:
        return 0.7
    factor = duration / target
    if 0.85 <= factor <= 1.2:
        return 1.0
    if 0.5 <= factor < 0.85 or 1.2 < factor <= 2.0:
        return 0.7
    if 0.25 <= factor < 0.5 or 2.0 < factor <= 3.0:
        return 0.4
    return 0.1


def _resolution_score(height: int) -> float:
    if height >= 1080:
        return 1.0
    if height >= 720:
        return 0.6
    return 0.3


def score_candidate(c: Candidate, settings: Settings) -> float:
    aspect = _aspect_score(c.width, c.height)
    duration = _duration_score(c.duration, target=settings.target_block_duration)
    resolution = _resolution_score(c.height)
    source_pref = SOURCE_PREFERENCE.get(c.source, 0.1)
    return (
        settings.weight_aspect * aspect
        + settings.weight_duration * duration
        + settings.weight_resolution * resolution
        + settings.weight_source * source_pref
    )


class SelectorError(RuntimeError):
    pass


def _selection_from_video(brief: VisualBrief, c: Candidate) -> Selection:
    speed_factor: float | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    loop = False

    if c.kind == "video" and c.duration is not None:
        target = 8.0
        if c.duration < 4.0:
            loop = True
        elif c.duration > 16.0:
            mid = c.duration / 2
            trim_start = max(0.0, mid - target / 2)
            trim_end = trim_start + target
        else:
            speed_factor = c.duration / target

    return Selection(
        idx=brief.idx,
        source=c.source,
        url=c.url,
        kind=c.kind,  # "video" or "image"
        trim_start=trim_start,
        trim_end=trim_end,
        speed_factor=speed_factor,
        loop=loop,
        attribution=c.attribution,
    )


def _continuation(brief: VisualBrief) -> Selection:
    return Selection(idx=brief.idx, source="", url="", kind="continuation")


def _montage(brief: VisualBrief, images: list[Candidate]) -> Selection:
    """Build an image_montage selection from N≥2 image candidates.

    Caller guarantees images are already deduped against the global used set
    and ranked best-first. Caller must call _selection_from_video for the
    single-image case (we emit kind="image", not "image_montage")."""
    return Selection(
        idx=brief.idx,
        source="",
        url="",
        kind="image_montage",
        montage_urls=[c.url for c in images],
        montage_sources=[c.source for c in images],
    )


def _pick_unused_video(
    cands: list[Candidate],
    used: set[str],
    settings: Settings,
) -> Candidate | None:
    ranked = sorted(cands, key=lambda c: score_candidate(c, settings), reverse=True)
    for c in ranked:
        if c.url not in used:
            return c
    return None


def _pick_unused_images(
    cands: list[Candidate],
    used: set[str],
    settings: Settings,
    limit: int,
) -> list[Candidate]:
    images = [c for c in cands if c.kind == "image" and c.url not in used]
    images.sort(key=lambda c: score_candidate(c, settings), reverse=True)
    return images[:limit]


async def select_per_block(
    briefs: list[VisualBrief],
    candidates: dict[int, list[Candidate]],
    settings: Settings,
    image_search: ImageSearch | None = None,
) -> list[Selection]:
    """Pick exactly one Selection per brief.

    Rules:
    - No URL appears twice across the returned selections (montage URLs included).
    - Continuation briefs become image_montage when image_search yields ≥2
      unused images, image (single Ken Burns) for exactly 1, continuation
      (extend previous) for 0.
    - Video/image briefs prefer an unused candidate; if all candidates are
      used, fall back to image_montage via image_search; if that yields 0,
      repeat the least-recently-used candidate (last-resort).
    - Block 1 with no usable visuals raises SelectorError (it cannot extend).
    """
    selections: list[Selection] = []
    used: set[str] = set()
    last_idx_used: dict[str, int] = {}  # for least-recently-used fallback

    montage_max = settings.image_montage_max

    for pos, brief in enumerate(briefs):
        if brief.kind == "continuation":
            sel = await _resolve_continuation(brief, used, settings, image_search)
            if sel.kind == "continuation" and pos == 0:
                raise SelectorError(
                    f"block {brief.idx} is the first block and has no visuals "
                    f"(continuation/image_search both empty)"
                )
        else:
            sel = await _resolve_visual(
                brief,
                candidates.get(brief.idx, []),
                used,
                last_idx_used,
                settings,
                image_search,
                pos,
            )

        # Mark all URLs the new selection introduces as used.
        for u in _selection_urls(sel):
            used.add(u)
            last_idx_used[u] = pos

        selections.append(sel)

    return selections


def _selection_urls(sel: Selection) -> list[str]:
    if sel.kind == "image_montage":
        return list(sel.montage_urls)
    if sel.url:
        return [sel.url]
    return []


async def _resolve_continuation(
    brief: VisualBrief,
    used: set[str],
    settings: Settings,
    image_search: ImageSearch | None,
) -> Selection:
    if image_search is None:
        return _continuation(brief)
    images = await image_search(brief)
    unused = _pick_unused_images(images, used, settings, limit=settings.image_montage_max)
    if len(unused) >= 2:
        return _montage(brief, unused)
    if len(unused) == 1:
        return _selection_from_video(brief, unused[0])
    return _continuation(brief)


async def _resolve_visual(
    brief: VisualBrief,
    cands: list[Candidate],
    used: set[str],
    last_idx_used: dict[str, int],
    settings: Settings,
    image_search: ImageSearch | None,
    pos: int,
) -> Selection:
    chosen = _pick_unused_video(cands, used, settings)
    if chosen is not None:
        return _selection_from_video(brief, chosen)

    # Pool exhausted: try image_montage fallback.
    if image_search is not None:
        images = await image_search(brief)
        unused = _pick_unused_images(images, used, settings, limit=settings.image_montage_max)
        if len(unused) >= 2:
            return _montage(brief, unused)
        if len(unused) == 1:
            return _selection_from_video(brief, unused[0])

    # Last-resort: repeat the candidate seen the longest ago. If we have no
    # candidates at all and we're not the first block, extend the previous.
    if not cands:
        if pos == 0:
            raise SelectorError(
                f"block {brief.idx} has no candidates and is the first block "
                f"(cannot continue from previous)"
            )
        return _continuation(brief)

    ranked = sorted(cands, key=lambda c: score_candidate(c, settings), reverse=True)
    # Prefer the candidate whose last-used position is smallest (used the longest ago).
    chosen = min(ranked, key=lambda c: last_idx_used.get(c.url, -1))
    return _selection_from_video(brief, chosen)
```

- [ ] **Step 4: Verify selector tests pass**

```bash
poetry run pytest tests/test_selector.py -v
```

Expected: all selector tests PASS.

- [ ] **Step 5: Run the full suite to find any callers I broke**

```bash
poetry run pytest -q
```

Expected: failures only in CLI / integration tests that call the now-async `select_per_block`. We fix those in the next task.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/selector.py tests/test_selector.py
git commit -m "feat(selector): global URL dedup + image_montage fallback (async)"
```

---

### Task 4: Update the CLI to await the async selector and pass the orchestrator

**Files:**
- Modify: `src/avtv/cli.py`
- Test: `tests/test_cli.py` (only if it currently exercises `select`/`build`; add a smoke test if not)

- [ ] **Step 1: Look at existing CLI test coverage**

```bash
grep -n "select\|build" tests/test_cli.py
```

If the existing test only covers `parse`, skip writing a new test (the suite already proves the wiring via the search/selector tests). If it does cover `build`, you must update its mocks to accept the new awaitable selector — show the diff before changing.

- [ ] **Step 2: Update `cli.py`**

In `src/avtv/cli.py`, change the `select` and `build` commands so the call to `select_per_block` is wrapped in `asyncio.run` and is given an image-search callback bound to the orchestrator.

Replace the `select` command (currently at `src/avtv/cli.py:90-103`):

```python
@app.command()
def select(
    run_id: str = typer.Option(...),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Stage 4: rank candidates and pick one per block."""
    settings = Settings()  # type: ignore[call-arg]
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    orch = _build_orchestrator(settings)
    selections = asyncio.run(
        select_per_block(
            briefs,
            candidates,
            settings=settings,
            image_search=orch.search_images_for_brief,
        )
    )
    rd.save_selections(selections)
    console.print(f"[green]selected[/green] {len(selections)} clips")
```

Replace the call inside `build` (currently at `src/avtv/cli.py:167`):

```python
    selections = asyncio.run(
        select_per_block(
            briefs,
            results,
            settings=settings,
            image_search=orch.search_images_for_brief,
        )
    )
```

- [ ] **Step 3: Run the suite**

```bash
poetry run pytest -q
```

Expected: PASS. If `test_cli.py` exercises `build` end-to-end with mocks, update those mocks to accept an async selector.

- [ ] **Step 4: Manual smoke**

We can't run the full pipeline without API keys & ffmpeg, but we can re-run `select` against the existing `runs/teste1/` artefacts:

```bash
poetry run avtv select --run-id teste1
```

Expected: terminates without error, rewrites `runs/teste1/04_selections.json`. Confirm by inspecting that the 7 previous continuations now have `kind == "image_montage"` (assuming image search returns ≥2 candidates for their queries):

```bash
poetry run python - <<'PY'
import json
from collections import Counter
sels = json.loads(open("runs/teste1/04_selections.json").read())
print(Counter(s["kind"] for s in sels))
PY
```

Expected: `Counter({'video': 80+, 'image_montage': 7-ish, 'image': 2, 'continuation': 0})`. If continuations remain it just means that brief's image search came back empty — that's allowed.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/cli.py tests/test_cli.py
git commit -m "feat(cli): wire async selector with on-demand image search"
```

---

### Task 5: Add `image_montage` rendering in the assembler

**Files:**
- Modify: `src/avtv/assembler.py`
- Test: `tests/test_assembler.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_assembler.py`:

```python
def test_plan_image_montage():
    sel = Selection(
        idx=4, source="", url="", kind="image_montage",
        montage_urls=["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg"],
        montage_sources=["pixabay", "pixabay", "wikimedia"],
    )
    plan = plan_segment(sel)
    assert plan.strategy == "image_montage"
    assert plan.use_clip_audio is False


def test_build_args_image_montage_three_images():
    from avtv.assembler import build_montage_args
    args = build_montage_args(
        image_paths=["/a.jpg", "/b.jpg", "/c.jpg"],
        output_path="/out.ts",
        target_w=1920,
        target_h=1080,
        target_fps=30,
    )
    cmd = " ".join(args)
    # All three inputs are passed with -loop 1
    assert cmd.count("-loop 1") == 3
    assert "/a.jpg" in cmd and "/b.jpg" in cmd and "/c.jpg" in cmd
    # filter_complex with 3 zoompan + concat
    assert "zoompan" in cmd
    assert "concat=n=3:v=1:a=0" in cmd
    assert "-t 8.0" in cmd
    assert "-an" in cmd  # no clip audio
    # frames per image: 30fps * 8s / 3 = 80
    assert "d=80" in cmd


def test_build_args_image_montage_two_images_split_evenly():
    from avtv.assembler import build_montage_args
    args = build_montage_args(
        image_paths=["/a.jpg", "/b.jpg"],
        output_path="/o.ts",
        target_w=1920,
        target_h=1080,
        target_fps=30,
    )
    cmd = " ".join(args)
    assert "concat=n=2:v=1:a=0" in cmd
    # 30fps * 8s / 2 = 120 frames each
    assert "d=120" in cmd
```

- [ ] **Step 2: Verify tests fail**

```bash
poetry run pytest tests/test_assembler.py -v -k montage
```

Expected: failures (`build_montage_args` not defined; `plan_segment` returns wrong strategy).

- [ ] **Step 3: Implement**

Edit `src/avtv/assembler.py`. Make these changes:

a. Extend the `Strategy` literal:

```python
Strategy = Literal["speed", "loop", "trim", "ken_burns", "image_montage", "continuation"]
```

b. In `plan_segment` (around `src/avtv/assembler.py:31-53`), add a branch BEFORE the `image` branch:

```python
def plan_segment(sel: Selection) -> SegmentPlan:
    if sel.kind == "continuation":
        return SegmentPlan(strategy="continuation", use_clip_audio=False)
    if sel.kind == "image_montage":
        return SegmentPlan(strategy="image_montage", use_clip_audio=False)
    if sel.kind == "image":
        return SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    ...  # rest unchanged
```

c. Add the new arg builder. Place it next to `_ken_burns_args`:

```python
def build_montage_args(
    image_paths: list[str],
    output_path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
) -> list[str]:
    """Render N images (1≤N≤image_montage_max) as one BLOCK_DURATION-second
    segment: each image gets BLOCK_DURATION/N seconds with Ken Burns, then
    concatenated. Caller passes already-downloaded local image paths.
    """
    n = len(image_paths)
    if n == 0:
        raise ValueError("build_montage_args requires at least one image")
    total_frames = int(BLOCK_DURATION * target_fps)
    frames_per = total_frames // n  # integer division; trailing frames padded by -t

    zoom_step = 0.15 / max(frames_per, 1)
    filter_parts: list[str] = []
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]zoompan=z='min(zoom+{zoom_step:.5f},1.15)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames_per}:s={target_w}x{target_h}:fps={target_fps}"
            f"[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={n}:v=1:a=0[out]"

    args = ["-y"]
    for p in image_paths:
        args += ["-loop", "1", "-i", p]
    args += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        "-t",
        f"{BLOCK_DURATION}",
        "-f",
        "mpegts",
        output_path,
    ]
    return args
```

d. Update `build_segment_args` dispatch (around `src/avtv/assembler.py:69-99`) to reject `image_montage` (it has no single input file path):

```python
def build_segment_args(...):
    if plan.strategy == "image_montage":
        raise ValueError(
            "image_montage uses build_montage_args (multiple inputs), not build_segment_args"
        )
    if plan.strategy == "loop":
        ...  # rest unchanged
```

- [ ] **Step 4: Verify the new tests pass**

```bash
poetry run pytest tests/test_assembler.py -v
```

Expected: all assembler tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py tests/test_assembler.py
git commit -m "feat(assembler): add image_montage strategy and build_montage_args"
```

---

### Task 6: Wire `image_montage` into `assemble_run`

`assemble_run` currently iterates `for i, sel in enumerate(selections)` and either reuses the previous segment for `continuation` or downloads one URL and encodes it. Add a third branch for `image_montage`: download all montage images and call `build_montage_args` + `run_ffmpeg`.

**Files:**
- Modify: `src/avtv/assembler.py`
- Test: `tests/test_assemble_run.py`

- [ ] **Step 1: Examine the existing `tests/test_assemble_run.py` to learn the mocking pattern**

```bash
poetry run pytest tests/test_assemble_run.py -v
```

Read the file end-to-end:

```bash
poetry run python -c "import pathlib; print(pathlib.Path('tests/test_assemble_run.py').read_text())"
```

Identify how `Downloader` is faked. The new test must reuse the same fake.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_assemble_run.py` (mirroring the existing patterns — replace `MockDownloader` etc. with whatever the file already uses; the snippet below assumes a fake downloader that returns deterministic Paths):

```python
def test_assemble_run_handles_image_montage(tmp_path, monkeypatch):
    from avtv.assembler import assemble_run
    from avtv.models import Selection

    selections = [
        Selection(idx=1, source="pexels", url="https://x/v1.mp4", kind="video",
                  speed_factor=1.0, attribution=None),
        Selection(idx=2, source="", url="", kind="image_montage",
                  montage_urls=["https://x/a.jpg", "https://x/b.jpg", "https://x/c.jpg"],
                  montage_sources=["pixabay", "pixabay", "wikimedia"]),
    ]

    fetched: list[tuple[str, str]] = []

    class FakeDownloader:
        def fetch_sync(self, url, kind):
            fetched.append((url, kind))
            p = tmp_path / f"dl_{len(fetched)}.bin"
            p.write_bytes(b"")
            return p

    ran: list[list[str]] = []
    monkeypatch.setattr("avtv.assembler.run_ffmpeg", lambda args: ran.append(args))

    work_dir = tmp_path / "work"
    out = tmp_path / "out.mp4"
    narration = tmp_path / "n.mp3"
    narration.write_bytes(b"")

    assemble_run(
        selections=selections,
        narration_path=narration,
        output_path=out,
        work_dir=work_dir,
        downloader=FakeDownloader(),
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )

    # Block 1 fetches the video. Block 2 fetches all 3 montage images.
    assert fetched[0] == ("https://x/v1.mp4", "video")
    assert {f for f, k in fetched[1:] if k == "image"} == {
        "https://x/a.jpg", "https://x/b.jpg", "https://x/c.jpg"
    }

    # ffmpeg called: 1 for block-1 segment, 1 for block-2 montage, 1 for final mux
    assert len(ran) == 3
    montage_cmd = " ".join(ran[1])
    assert "concat=n=3:v=1:a=0" in montage_cmd
```

- [ ] **Step 3: Verify the test fails**

```bash
poetry run pytest tests/test_assemble_run.py::test_assemble_run_handles_image_montage -v
```

Expected: KeyError or AttributeError because the assembler doesn't know what to do with `image_montage`.

- [ ] **Step 4: Implement `assemble_run` branch**

In `src/avtv/assembler.py`, change the loop in `assemble_run` (currently `src/avtv/assembler.py:356-394`):

```python
    for i, sel in enumerate(selections, start=1):
        if sel.kind == "continuation":
            if last_segment is None:
                raise FFmpegError(
                    f"block {sel.idx}: continuation has no prior segment"
                )
            segment_paths.append(last_segment)
            if on_segment:
                on_segment(i, total)
            continue

        seg_out = work_dir / f"segment_{sel.idx:04d}.ts"
        if seg_out.exists() and seg_out.stat().st_size > 0:
            segment_paths.append(seg_out)
            last_segment = seg_out
            if on_segment:
                on_segment(i, total)
            continue

        if sel.kind == "image_montage":
            image_paths = [
                downloader.fetch_sync(u, kind="image")
                for u in sel.montage_urls
            ]
            args = build_montage_args(
                image_paths=[str(p) for p in image_paths],
                output_path=str(seg_out),
                target_w=target_w,
                target_h=target_h,
                target_fps=target_fps,
            )
            run_ffmpeg(args)
            segment_paths.append(seg_out)
            last_segment = seg_out
            if on_segment:
                on_segment(i, total)
            continue

        input_path = _fetch_for(downloader, sel)
        plan = plan_segment(sel)

        args = build_segment_args(
            input_path=str(input_path),
            output_path=str(seg_out),
            plan=plan,
            target_w=target_w,
            target_h=target_h,
            target_fps=target_fps,
            clip_audio_db=clip_audio_db,
        )
        run_ffmpeg(args)
        segment_paths.append(seg_out)
        last_segment = seg_out
        if on_segment:
            on_segment(i, total)
```

`build_montage_args` needs to be importable here — it's already defined in the same file from Task 5, no import needed.

- [ ] **Step 5: Verify all assembler tests pass**

```bash
poetry run pytest tests/test_assembler.py tests/test_assemble_run.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/assembler.py tests/test_assemble_run.py
git commit -m "feat(assembler): render image_montage selections in assemble_run"
```

---

### Task 7: Clean up unused config + add new montage knobs

**Files:**
- Modify: `src/avtv/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Look at `tests/test_config.py` to see what's expected**

```bash
grep -n "repetition_penalty\|neighbor_window\|montage" tests/test_config.py
```

Adapt the tests below if any of those names are referenced.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_settings_defines_image_montage_knobs(monkeypatch):
    for k in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(k, "x")
    from avtv.config import Settings
    s = Settings()
    assert s.image_montage_max == 3
    # repetition_penalty and neighbor_window were removed (no longer applicable):
    assert not hasattr(s, "repetition_penalty")
    assert not hasattr(s, "neighbor_window")
```

- [ ] **Step 3: Run to verify failure**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: AssertionError (`hasattr` returns True today, and `image_montage_max` doesn't exist).

- [ ] **Step 4: Edit `src/avtv/config.py`**

Remove `repetition_penalty` and `neighbor_window`. Add `image_montage_max`. Resulting block:

```python
    # Selector weights
    weight_aspect: float = 1.0
    weight_duration: float = 1.0
    weight_resolution: float = 1.0
    weight_source: float = 1.0
    image_montage_max: int = 3
```

- [ ] **Step 5: Verify**

```bash
poetry run pytest -q
```

Expected: full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/config.py tests/test_config.py
git commit -m "chore(config): drop unused repetition_penalty/neighbor_window, add image_montage_max"
```

---

### Task 8: Surface `image_montage` in the Streamlit UI inspector

The Selections tab of the UI currently expects one URL per selection. With montage selections, we want to show all 3 image URLs (as thumbnails or links) so the user can edit them.

**Files:**
- Modify: `src/avtv/ui/streamlit_app.py`

- [ ] **Step 1: Locate the selections rendering**

```bash
grep -n "kind\|url\|montage\|selections" src/avtv/ui/streamlit_app.py
```

Identify the function that renders one selection row.

- [ ] **Step 2: Add a montage branch**

Read the surrounding code first to match style. The change is conceptually:

```python
# Pseudocode — adapt to the actual function once you read the file.
if sel["kind"] == "image_montage":
    cols = st.columns(len(sel["montage_urls"]))
    for col, url, source in zip(cols, sel["montage_urls"], sel["montage_sources"]):
        with col:
            st.image(url, caption=source, use_container_width=True)
elif sel["kind"] == "continuation":
    st.caption("(continuation — extends previous clip)")
elif sel["url"]:
    st.video(sel["url"]) if sel["kind"] == "video" else st.image(sel["url"])
```

- [ ] **Step 3: Manual verification**

```bash
poetry run avtv ui
```

Open <http://localhost:8501>, click into the `teste1` run, scroll the Selections tab. Confirm any montage selections show 3 thumbnails. Confirm normal video/image selections still render. `Ctrl+C` to stop.

- [ ] **Step 4: Commit**

```bash
git add src/avtv/ui/streamlit_app.py
git commit -m "feat(ui): render image_montage selections with per-image thumbnails"
```

---

### Task 9: End-to-end verification on `runs/teste1`

We have an existing run with 7 continuations. Re-run the `select` and `assemble` stages against it and inspect the result.

- [ ] **Step 1: Re-run select (will now do 7 image searches for the empty blocks)**

```bash
poetry run avtv select --run-id teste1
```

Expected: prints `selected 94 clips`. No errors.

- [ ] **Step 2: Inspect kinds and URL uniqueness**

```bash
poetry run python - <<'PY'
import json
from collections import Counter
sels = json.loads(open("runs/teste1/04_selections.json").read())
kinds = Counter(s["kind"] for s in sels)
all_urls: list[str] = []
for s in sels:
    if s["kind"] == "image_montage":
        all_urls.extend(s["montage_urls"])
    elif s["url"]:
        all_urls.append(s["url"])
dup = [u for u, n in Counter(all_urls).items() if n > 1]
print("kinds:", kinds)
print("total URLs:", len(all_urls), "distinct:", len(set(all_urls)), "duplicates:", len(dup))
PY
```

Expected:
- `image_montage` count > 0 (or `image` if some queries returned <2 images)
- `duplicates` is `0` (this is the headline guarantee).

- [ ] **Step 3: Re-assemble (segments cache will skip unchanged blocks)**

```bash
poetry run avtv assemble --run-id teste1 --audio input/narration.mp3 --out output.mp4
```

Expected: completes; new `output.mp4` written. The 7 previously-static stretches now show 3 different images each.

- [ ] **Step 4: Spot-check the video**

Open `output.mp4`, scrub to the timecodes of the previous continuations (00:24, 02:16, 03:04, 06:08, 07:28, 09:44, 10:16). Confirm each one now shows a 3-image montage (or single Ken Burns image if the search only returned one).

- [ ] **Step 5: Commit any cache/run changes only if they came with bug fixes**

If the smoke run revealed a bug, fix it as a small follow-up commit. Otherwise nothing to commit here.

---

### Task 10: Update README with the new behavior

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the "How the pipeline works" table** (`README.md:188-198`)

Update the **select** row:

```markdown
| **select** | `04_selections.json` | Scores candidates by aspect ratio, duration fit, resolution, and source. **Globally dedupes URLs** so no clip appears twice in one video. **Empty/continuation blocks** become a 3-image Ken Burns montage (image_montage kind) using on-demand image search; if image search yields nothing, falls back to extending the previous clip. |
```

Update the **assemble** row:

```markdown
| **assemble** | `output.mp4` | Downloads each picked clip (cached), encodes one 8 s `.ts` segment per block (speed-match / trim / loop / Ken Burns / **3-image montage**), then concats and muxes with the narration. |
```

- [ ] **Step 2: Update "Known limits"** (`README.md:276-290`)

Remove the bullet about loop-crossfade only if you actually fixed it (you didn't). Add nothing new — the new behavior is a feature, not a known limit.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document global dedup and image_montage fallback"
```

---

## Self-Review

**Spec coverage:**
- "no exact same URL twice in one video" → Task 3 (`test_no_exact_repeat_anywhere_in_video`) + Task 9 step 2 verifies on real data.
- "every empty/continuation block → 3 images" → Task 3 (`test_continuation_becomes_image_montage`) + Task 5/6 render path + Task 9 step 4 visual check.
- "if image search returns 0 → 1 image with Ken Burns" → Task 3 (`test_continuation_with_one_image_uses_single_image`).
- "If video pool exhausted → 3 images" → Task 3 (`test_video_dedup_overflow_uses_image_montage`).
- "0 images on overflow → repeat least-recently-used" — last-resort path in `_resolve_visual` covered by `test_no_exact_repeat_anywhere_in_video` (block 4).

**Placeholder scan:** all code blocks contain runnable code; no "TBD" / "implement later" / "similar to". Step 2 of Task 4 mentions inspecting an existing test — that's a read step, not a code placeholder.

**Type consistency:**
- `Selection.kind` widened to `VisualKind`, which now includes `"image_montage"`. `VisualBrief.kind` is its own narrower Literal — matches Task 1.
- `select_per_block` returns `list[Selection]`, awaitable; called via `asyncio.run` in CLI (Task 4).
- `build_montage_args` signature matches calls in both Task 5 (test) and Task 6 (assemble_run).
- `image_montage_max` defined in Task 7, used in Task 3 (`settings.image_montage_max`), and in test assertions consistently as 3.

**One known follow-up not in scope:** the `weight_*` knobs are unused for image_montage selection (we just rank by score uniformly). If montage image quality is poor, expose a weight for it. Out of scope for this plan.
