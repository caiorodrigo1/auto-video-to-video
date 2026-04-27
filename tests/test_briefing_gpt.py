import json
from unittest.mock import MagicMock, patch

import pytest

from avtv.briefing.gpt import OpenAIProvider
from avtv.models import Block


def _set_env(monkeypatch):
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(key, "x")


@pytest.fixture
def mock_openai_response():
    payload = {
        "briefs": [
            {
                "idx": 1,
                "query_en": "forest sunlight",
                "fallback_query": "nature",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            }
        ]
    }
    completion = MagicMock()
    choice = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice.message = msg
    completion.choices = [choice]
    return completion


def test_openai_provider_returns_briefs(mock_openai_response, monkeypatch):
    _set_env(monkeypatch)
    blocks = [Block.from_text(1, 0.0, 8.0, "x")]

    with patch("avtv.briefing.gpt.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create.return_value = mock_openai_response

        provider = OpenAIProvider()
        briefs = provider.generate_briefs(blocks)

    assert len(briefs) == 1
    assert briefs[0].query_en == "forest sunlight"
