from pathlib import Path

import pytest

from avtv.parser import ParseError, parse_script

FIXTURE = Path(__file__).parent / "fixtures" / "sample_script.txt"


def test_parse_sample_returns_four_blocks():
    blocks = parse_script(FIXTURE.read_text())
    assert len(blocks) == 4


def test_parse_first_block_fields():
    blocks = parse_script(FIXTURE.read_text())
    b = blocks[0]
    assert b.idx == 1
    assert b.start == 0.0
    assert b.end == 8.0
    assert b.text == "First block of narration."
    assert b.is_empty is False


def test_parse_empty_block_detected():
    blocks = parse_script(FIXTURE.read_text())
    assert blocks[2].is_empty is True
    assert blocks[2].idx == 3


def test_parse_rejects_non_sequential_idx():
    bad = """\
PROMPT 001 | 00:00 - 00:08
hi
------------------------------------------------------------

PROMPT 003 | 00:08 - 00:16
gap
------------------------------------------------------------
"""
    with pytest.raises(ParseError, match="non-sequential"):
        parse_script(bad)


def test_parse_rejects_wrong_duration():
    bad = """\
PROMPT 001 | 00:00 - 00:09
hi
------------------------------------------------------------
"""
    with pytest.raises(ParseError, match="duration"):
        parse_script(bad)


def test_parse_handles_multiline_body():
    multi = """\
PROMPT 001 | 00:00 - 00:08
Line one.
Line two.
------------------------------------------------------------
"""
    blocks = parse_script(multi)
    assert blocks[0].text == "Line one.\nLine two."
