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


def test_score_perfect_clip(monkeypatch):
    s = _settings(monkeypatch)
    c = _make()
    score = score_candidate(c, settings=s)
    # 1.0 (aspect 16:9) + 1.0 (8s ideal) + 1.0 (1080p) + 0.3 (pexels) = 3.3
    assert score > 3.0


def test_score_low_resolution_penalized(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(width=640, height=360)
    score = score_candidate(c, settings=s)
    perfect = score_candidate(_make(), settings=s)
    assert score < perfect


def test_score_short_clip_penalized(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(duration=2.0)
    perfect = score_candidate(_make(), settings=s)
    assert score_candidate(c, settings=s) < perfect


def test_score_image_candidate(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(kind="image", duration=None, width=4000, height=3000)
    # aspect 4:3 = 1.33, falls in lower band
    score = score_candidate(c, settings=s)
    assert score > 0  # still scorable


def _brief(idx, kind="video"):
    return VisualBrief(
        idx=idx,
        query_en="x",
        fallback_query="y",
        kind=kind,
    )


def test_select_picks_best_per_block(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {
        1: [_make(url="hi-res", height=1080), _make(url="lo-res", height=480)],
        2: [_make(url="hi-res2", height=1080)],
    }
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[0].url == "hi-res"
    assert sels[1].url == "hi-res2"


def test_select_dedupes_within_neighbor_window(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2), _brief(3)]
    # Best candidate is the same URL for all three blocks; dedup should
    # avoid using it back-to-back in the ±2 window.
    same = _make(url="popular", height=1080)
    other = _make(url="alt", height=1080)
    cands = {
        1: [same, other],
        2: [same, other],
        3: [same, other],
    }
    sels = select_per_block(briefs, cands, settings=s)
    urls = [s.url for s in sels]
    assert urls[0] == "popular"
    assert urls[1] == "alt"  # blocked by neighbor dedup
    assert urls[2] == "alt"  # also blocked


def test_select_continuation_passthrough(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="primary", height=1080)], 2: []}
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[1].kind == "continuation"
    assert sels[1].url == ""  # placeholder, assembler resolves


def test_select_empty_candidates_falls_back_to_continuation(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {1: [_make(url="ok", height=1080)], 2: []}  # block 2 has nothing
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[1].kind == "continuation"


def test_select_first_block_no_candidates_raises(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1)]
    cands = {1: []}
    import pytest

    from avtv.selector import SelectorError

    with pytest.raises(SelectorError):
        select_per_block(briefs, cands, settings=s)
