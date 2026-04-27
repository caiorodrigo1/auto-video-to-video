from avtv.config import Settings
from avtv.models import Candidate
from avtv.selector import score_candidate


def _make(**kw) -> Candidate:
    base = dict(
        source="pexels", url="x", kind="video",
        duration=8.0, width=1920, height=1080, license="x",
    )
    base.update(kw)
    return Candidate(**base)


def _settings(monkeypatch) -> Settings:
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PEXELS_API_KEY",
              "PIXABAY_API_KEY", "UNSPLASH_API_KEY"]:
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
