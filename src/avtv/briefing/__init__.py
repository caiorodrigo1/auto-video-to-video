from avtv.briefing.llm import LLMProvider  # noqa: F401

__all__ = ["LLMProvider", "get_provider"]


def get_provider(name: str) -> LLMProvider:
    if name == "claude":
        from avtv.briefing.claude import ClaudeProvider

        return ClaudeProvider()
    if name == "gpt":
        from avtv.briefing.gpt import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name}")
