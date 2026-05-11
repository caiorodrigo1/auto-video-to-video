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


def test_build_system_prompt_without_topic_matches_legacy():
    from avtv.briefing.prompt import SYSTEM_PROMPT, build_system_prompt

    # Without a topic, the function returns the exact legacy SYSTEM_PROMPT.
    assert build_system_prompt(None) == SYSTEM_PROMPT


def test_build_system_prompt_injects_topic():
    from avtv.briefing.prompt import build_system_prompt

    out = build_system_prompt("Classic muscle cars of the 1970s")
    # Includes the topic verbatim
    assert "Classic muscle cars of the 1970s" in out
    # And appears BEFORE the rules so the LLM sees context first
    rule_anchor = "RULES:"
    assert out.index("Classic muscle cars") < out.index(rule_anchor)
    # The original rules are still present (we appended, didn't replace)
    assert "query_en" in out


def test_build_system_prompt_with_empty_string_treated_as_none():
    from avtv.briefing.prompt import SYSTEM_PROMPT, build_system_prompt

    assert build_system_prompt("") == SYSTEM_PROMPT
    assert build_system_prompt("   ") == SYSTEM_PROMPT
