from unittest.mock import MagicMock, patch

import pytest

from avtv.briefing.claude import ClaudeProvider
from avtv.models import Block


@pytest.fixture
def mock_anthropic_response():
    """Build a fake anthropic Message with a tool_use block."""
    msg = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "briefs": [
            {
                "idx": 1,
                "query_en": "forest sunlight",
                "fallback_query": "nature",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            },
            {
                "idx": 2,
                "query_en": "wild bison",
                "fallback_query": "large mammal",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            },
        ]
    }
    msg.content = [tool_block]
    msg.stop_reason = "tool_use"
    return msg


def test_claude_provider_returns_briefs(mock_anthropic_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [
        Block.from_text(1, 0.0, 8.0, "forest words"),
        Block.from_text(2, 8.0, 16.0, "bison words"),
    ]

    with patch("avtv.briefing.claude.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_anthropic_response

        provider = ClaudeProvider()
        briefs = provider.generate_briefs(blocks)

    assert len(briefs) == 2
    assert briefs[0].query_en == "forest sunlight"
    assert briefs[1].query_en == "wild bison"


def test_claude_provider_raises_on_count_mismatch(mock_anthropic_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [Block.from_text(i, (i - 1) * 8.0, i * 8.0, "x") for i in range(1, 5)]

    with patch("avtv.briefing.claude.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_anthropic_response  # only 2 briefs
        provider = ClaudeProvider()
        with pytest.raises(ValueError, match="count mismatch"):
            provider.generate_briefs(blocks)
