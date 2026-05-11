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


def test_pexels_strongly_preferred_over_other_sources(monkeypatch):
    """Pexels has the most relevant keyword search; gap must be wide enough
    that a Pexels candidate wins against same-spec candidates from other
    sources. Regression guard: if anyone tightens SOURCE_PREFERENCE again,
    the 'bacon for Hoover Dam' problem returns."""
    s = _settings(monkeypatch)
    pexels = _make(source="pexels")
    pixabay = _make(source="pixabay")
    wikimedia = _make(source="wikimedia")
    pexels_score = score_candidate(pexels, settings=s)
    other_max = max(
        score_candidate(pixabay, settings=s),
        score_candidate(wikimedia, settings=s),
    )
    # Gap must exceed any single non-source axis swing (worst-case 0.4 from
    # duration band drop). 0.5 is the minimum margin that survives that.
    assert pexels_score - other_max >= 0.5


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


@pytest.mark.asyncio
async def test_video_brief_with_empty_pool_and_empty_image_search_extends(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {1: [_make(url="v1", height=1080)], 2: []}

    async def fake(_b):
        return []

    sels = await select_per_block(briefs, cands, settings=s, image_search=fake)
    assert sels[0].url == "v1"
    assert sels[1].kind == "continuation"


@pytest.mark.asyncio
async def test_first_block_continuation_with_no_image_search_raises(monkeypatch):
    from avtv.selector import SelectorError

    s = _settings(monkeypatch)
    briefs = [_brief(1, kind="continuation")]
    cands: dict[int, list] = {}
    with pytest.raises(SelectorError):
        await select_per_block(briefs, cands, settings=s, image_search=None)
