from anthropic import Anthropic
from openai import OpenAI

from avtv.config import Settings
from avtv.models import Block

TOPIC_SYSTEM_PROMPT = """\
You are analyzing a documentary narration to identify its overall theme.
Read all the blocks below. Reply with ONE concise sentence (max 12 words)
describing the video's subject. Be specific — name the category, era, or
domain. Examples of good answers:
  - "Classic American muscle cars of the 1970s"
  - "H-shaped barndominium home plans"
  - "Roman aqueducts and ancient water engineering"
Reply with the sentence only. No quotes, no prefix, no extra text.
"""


def _blocks_to_user_msg(blocks: list[Block]) -> str:
    lines = ["Here is the full narration script:\n"]
    for b in blocks:
        body = b.text.strip() or "[empty]"
        lines.append(f"[BLOCK {b.idx}] {body}")
    return "\n".join(lines)


def _extract_via_claude(blocks: list[Block], settings: Settings) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=128,
        system=TOPIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _blocks_to_user_msg(blocks)}],
    )
    text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ValueError("Claude topic call returned no text")
    return text_blocks[0].text.strip()


def _extract_via_gpt(blocks: list[Block], settings: Settings) -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.gpt_model,
        messages=[
            {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": _blocks_to_user_msg(blocks)},
        ],
        max_completion_tokens=128,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI topic call returned empty content")
    return content.strip()


def extract_topic_via_llm(
    blocks: list[Block],
    provider_name: str,
    settings: Settings,
) -> str:
    """Ask the configured LLM provider for a one-sentence video theme.

    Returns the stripped sentence. Raises ValueError on empty response or
    unknown provider.
    """
    if provider_name == "claude":
        return _extract_via_claude(blocks, settings)
    if provider_name == "gpt":
        return _extract_via_gpt(blocks, settings)
    raise ValueError(f"unknown provider: {provider_name}")
