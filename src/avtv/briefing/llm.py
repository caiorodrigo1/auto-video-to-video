from typing import Protocol

from avtv.models import Block, VisualBrief


class LLMProvider(Protocol):
    def generate_briefs(
        self,
        blocks: list[Block],
        topic: str | None = None,
    ) -> list[VisualBrief]: ...
