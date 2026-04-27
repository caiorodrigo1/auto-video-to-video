from typing import Any

from anthropic import Anthropic

from avtv.briefing.prompt import SYSTEM_PROMPT, build_user_message
from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.config import Settings
from avtv.models import Block, VisualBrief

TOOL_NAME = "emit_visual_briefs"


class ClaudeProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()  # type: ignore[call-arg]
        self.client = Anthropic(api_key=self.settings.anthropic_api_key)

    def generate_briefs(self, blocks: list[Block]) -> list[VisualBrief]:
        user_msg = build_user_message(blocks)

        response = self.client.messages.create(
            model=self.settings.claude_model,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Emit one VisualBrief per script block.",
                    "input_schema": BRIEF_TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user_msg}],
        )

        tool_blocks: list[Any] = [
            b for b in response.content if getattr(b, "type", None) == "tool_use"
        ]
        if not tool_blocks:
            raise ValueError("Claude did not return a tool_use block")

        briefs = parse_briefs_response(tool_blocks[0].input)

        if len(briefs) != len(blocks):
            raise ValueError(f"count mismatch: got {len(briefs)} briefs for {len(blocks)} blocks")

        return briefs
