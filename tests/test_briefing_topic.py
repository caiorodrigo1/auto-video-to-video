from unittest.mock import MagicMock, patch

import pytest

from avtv.models import Block


def _blocks() -> list[Block]:
    return [
        Block.from_text(1, 0.0, 8.0, "The 1970 Ford Maverick was unveiled in April 1969."),
        Block.from_text(2, 8.0, 16.0, "It was a compact muscle car for the working class."),
    ]


def _set_keys(monkeypatch) -> None:
    for k in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(k, "x")


def test_extract_topic_via_claude(monkeypatch):
    _set_keys(monkeypatch)
    from avtv.briefing.topic import extract_topic_via_llm
    from avtv.config import Settings

    fake_msg = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Classic American muscle cars of the 1970s"
    fake_msg.content = [text_block]

    with patch("avtv.briefing.topic.Anthropic") as MockAnth:
        MockAnth.return_value.messages.create.return_value = fake_msg
        topic = extract_topic_via_llm(_blocks(), provider_name="claude", settings=Settings())

    assert topic == "Classic American muscle cars of the 1970s"


def test_extract_topic_via_gpt(monkeypatch):
    _set_keys(monkeypatch)
    from avtv.briefing.topic import extract_topic_via_llm
    from avtv.config import Settings

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "Classic American muscle cars of the 1970s"

    with patch("avtv.briefing.topic.OpenAI") as MockOAI:
        MockOAI.return_value.chat.completions.create.return_value = fake_resp
        topic = extract_topic_via_llm(_blocks(), provider_name="gpt", settings=Settings())

    assert topic == "Classic American muscle cars of the 1970s"
    # GPT-5 rejects 'max_tokens'; must send 'max_completion_tokens' instead.
    call_kwargs = MockOAI.return_value.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in call_kwargs
    assert "max_tokens" not in call_kwargs


def test_extract_topic_strips_whitespace(monkeypatch):
    _set_keys(monkeypatch)
    from avtv.briefing.topic import extract_topic_via_llm
    from avtv.config import Settings

    fake_msg = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "  \n  Vintage muscle cars  \n  "
    fake_msg.content = [text_block]

    with patch("avtv.briefing.topic.Anthropic") as MockAnth:
        MockAnth.return_value.messages.create.return_value = fake_msg
        topic = extract_topic_via_llm(_blocks(), provider_name="claude", settings=Settings())

    assert topic == "Vintage muscle cars"


def test_extract_topic_unknown_provider_raises(monkeypatch):
    _set_keys(monkeypatch)
    from avtv.briefing.topic import extract_topic_via_llm
    from avtv.config import Settings

    with pytest.raises(ValueError, match="unknown provider"):
        extract_topic_via_llm(_blocks(), provider_name="grok", settings=Settings())
