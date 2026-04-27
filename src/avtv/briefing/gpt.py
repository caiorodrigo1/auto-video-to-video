import json

from openai import OpenAI

from avtv.briefing.prompt import SYSTEM_PROMPT, build_user_message
from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.config import Settings
from avtv.models import Block, VisualBrief


class OpenAIProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()  # type: ignore[call-arg]
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def generate_briefs(self, blocks: list[Block]) -> list[VisualBrief]:
        user_msg = build_user_message(blocks)

        response = self.client.chat.completions.create(
            model=self.settings.gpt_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "visual_briefs",
                    "schema": BRIEF_TOOL_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content")

        data = json.loads(content)
        briefs = parse_briefs_response(data)

        if len(briefs) != len(blocks):
            raise ValueError(f"count mismatch: got {len(briefs)} briefs for {len(blocks)} blocks")

        return briefs
