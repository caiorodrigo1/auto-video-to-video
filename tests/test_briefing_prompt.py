from avtv.briefing.prompt import SYSTEM_PROMPT, build_user_message
from avtv.models import Block


def test_system_prompt_mentions_english_queries():
    assert "english" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_continuation():
    assert "continuation" in SYSTEM_PROMPT.lower()


def test_user_message_includes_all_blocks():
    blocks = [
        Block.from_text(1, 0.0, 8.0, "first"),
        Block.from_text(2, 8.0, 16.0, ""),
        Block.from_text(3, 16.0, 24.0, "third"),
    ]
    msg = build_user_message(blocks)
    assert "[BLOCK 1]" in msg
    assert "first" in msg
    assert "[BLOCK 2]" in msg
    assert "[empty]" in msg.lower()
    assert "[BLOCK 3]" in msg
    assert "third" in msg
