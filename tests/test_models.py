import pytest
from pydantic import ValidationError

from avtv.models import Block, Candidate, Selection, VisualBrief


def test_block_basic():
    b = Block(idx=1, start=0.0, end=8.0, text="hello", is_empty=False)
    assert b.idx == 1
    assert b.end - b.start == 8.0


def test_block_empty_text_is_empty_true():
    # is_empty is computed from text, not passed in
    b = Block.from_text(idx=2, start=8.0, end=16.0, text="   ")
    assert b.is_empty is True


def test_block_with_text_is_empty_false():
    b = Block.from_text(idx=2, start=8.0, end=16.0, text="hello world")
    assert b.is_empty is False


def test_visual_brief_validates_kind():
    vb = VisualBrief(
        idx=1,
        query_en="forest sunlight",
        fallback_query="nature",
        kind="video",
        continuity_hint=None,
        notes=None,
    )
    assert vb.kind == "video"


def test_visual_brief_rejects_bad_kind():
    with pytest.raises(ValidationError):
        VisualBrief(
            idx=1,
            query_en="x",
            fallback_query="y",
            kind="movie",  # not allowed
            continuity_hint=None,
            notes=None,
        )


def test_candidate_video_has_duration():
    c = Candidate(
        source="pexels",
        url="https://example.com/v.mp4",
        preview_url=None,
        kind="video",
        duration=10.0,
        width=1920,
        height=1080,
        license="Pexels License",
        attribution="John Doe",
    )
    assert c.duration == 10.0


def test_selection_video_with_speed():
    s = Selection(
        idx=5,
        source="pexels",
        url="https://x/v.mp4",
        kind="video",
        trim_start=None,
        trim_end=None,
        speed_factor=0.9,
        loop=False,
        attribution=None,
    )
    assert s.speed_factor == 0.9
