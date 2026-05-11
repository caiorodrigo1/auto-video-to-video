# Topic-aware queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the brief stage from emitting ambiguous queries (e.g. `"Maverick"` returning Top Gun airplane scenes when the video is about classic cars) by adding a `topic` stage where the LLM proposes the video's theme, the user confirms or edits it, and the confirmed theme is injected into the brief's system prompt so subsequent queries are disambiguated and topic-aware.

**Architecture:** Insert a new pipeline stage `topic` between `parse` and `brief`. A small dedicated LLM call reads the parsed blocks and returns a one-sentence theme. CLI prints it and waits for `typer.prompt` confirmation (Enter accepts the suggestion, typing replaces it). The Streamlit UI auto-runs topic detection after file upload, then shows an editable text field; the user confirms by clicking "▶ Start build". The confirmed string is persisted to `runs/<id>/00_topic.json` and threaded through `LLMProvider.generate_briefs(blocks, topic=...)`, which prepends a topic clause to the existing `SYSTEM_PROMPT`. Brief stage stays backwards-compatible: if `00_topic.json` is absent, briefs are generated without any topic context (old run behavior).

**Tech Stack:** Python 3.11, Pydantic, Anthropic SDK, OpenAI SDK, Typer, Streamlit. No new dependencies.

---

## Context the engineer needs

**Current pipeline** (`README.md` and `src/avtv/cli.py`):
```
parse → brief (LLM, per-block queries) → search (5 APIs) → select → assemble (ffmpeg)
```
Artifacts: `01_blocks.json`, `02_visual_briefs.json`, `03_search_results.json`, `04_selections.json`.

**The brief stage today** (`src/avtv/briefing/prompt.py`): tells the LLM to emit `query_en` (3–7 words) "specific" per block. It does NOT instruct the LLM to disambiguate proper nouns or use a single global theme as context. So when the script says "Maverick" in a video about classic cars, the LLM emits `query_en="Maverick"` and Pexels returns Top Gun.

**Why the user wants user-in-the-loop confirmation:** LLM topic detection from a long script is imperfect. The user knows the actual theme (or wants to override the LLM). The plan's design (locked in via memory):
- New stage between parse and brief
- Always prompts the user (Enter accepts LLM suggestion; typing replaces)
- Topic injected into system prompt (LLM decides per-query when to use it; we do NOT mechanically prefix every query)

**Provider abstraction** (`src/avtv/briefing/{__init__,llm,claude,gpt}.py`):
- `LLMProvider` is a Protocol with `generate_briefs(blocks)`.
- Two impls: `ClaudeProvider` (Anthropic tool_use), `OpenAIProvider` (json_schema response).
- Both load `Settings` and use it to pick model.

We will EXTEND the protocol with an OPTIONAL kwarg, keeping back-compat for callers that don't pass `topic`.

**Out-of-scope follow-ups** (don't do here):
- Per-block "topic confidence" gating
- Caching topic across runs with the same script hash
- CLIP-based visual reranking

---

## File Structure

| File | Role | Action |
|---|---|---|
| `src/avtv/models.py` | Pydantic models | Modify: add `Topic` model |
| `src/avtv/runs.py` | RunDir artifacts | Modify: add `TOPIC = "00_topic.json"` + `save_topic`/`load_topic`/`has_topic` |
| `src/avtv/briefing/topic.py` | Topic extraction | **Create**: `extract_topic_via_llm(blocks, provider_name, settings) -> str` — one small LLM call (Claude or GPT), returns a single sentence |
| `src/avtv/briefing/prompt.py` | Brief system prompt | Modify: `build_system_prompt(topic: str \| None) -> str` returns the existing system prompt plus an optional topic clause |
| `src/avtv/briefing/llm.py` | Provider Protocol | Modify: `generate_briefs(blocks, topic=None)` |
| `src/avtv/briefing/claude.py` | Claude provider | Modify: accept `topic`, use `build_system_prompt(topic)` |
| `src/avtv/briefing/gpt.py` | GPT provider | Modify: same |
| `src/avtv/cli.py` | CLI commands | Modify: add `topic` command, update `build` to run topic stage (interactive), update `brief` to pass topic if present |
| `src/avtv/ui/streamlit_app.py` | Streamlit UI | Modify: (1) fix existing bug — wrap `_run_stage_select` in `asyncio.run` (regression from prior PR); (2) add `_run_stage_topic`; (3) New-build form runs parse + topic on upload, shows editable theme field, build button uses it |
| `tests/test_models.py` | Model tests | Modify: append Topic tests |
| `tests/test_runs.py` | RunDir tests | Modify: append topic save/load test |
| `tests/test_briefing_prompt.py` | Prompt builder tests | Modify or create: tests for `build_system_prompt` with and without topic |
| `tests/test_briefing_topic.py` | Topic extraction tests | **Create**: mocked Claude + GPT topic extraction |
| `tests/test_briefing_claude.py` | Claude provider tests | Modify: assert topic flows into system prompt when passed |
| `tests/test_briefing_gpt.py` | GPT provider tests | Modify: same |
| `tests/test_cli.py` | CLI tests | Modify: smoke `topic` command non-interactive path (via `--topic` flag) |
| `README.md` | Documentation | Modify: pipeline table includes the new `topic` stage |

---

### Task 1: Add `Topic` model + RunDir support

`Topic` is a tiny Pydantic record persisted to `runs/<id>/00_topic.json`. It records the final confirmed topic plus an audit field for "did the human override the LLM suggestion".

**Files:**
- Modify: `src/avtv/models.py`
- Modify: `src/avtv/runs.py`
- Test: `tests/test_models.py`, `tests/test_runs.py`

- [ ] **Step 1: Write the failing model test**

Append to `tests/test_models.py`:

```python
def test_topic_model_round_trip():
    from avtv.models import Topic

    t = Topic(
        topic="Classic American muscle cars of the 1970s",
        llm_suggestion="Classic American muscle cars of the 1970s",
        source="user",
    )
    assert t.topic == "Classic American muscle cars of the 1970s"
    assert t.source == "user"
    dumped = t.model_dump()
    reloaded = Topic.model_validate(dumped)
    assert reloaded == t


def test_topic_rejects_invalid_source():
    from avtv.models import Topic
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Topic(topic="x", llm_suggestion="x", source="cosmic")
```

- [ ] **Step 2: Verify the tests fail**

```bash
poetry run pytest tests/test_models.py::test_topic_model_round_trip tests/test_models.py::test_topic_rejects_invalid_source -v
```

Expected: `ImportError: cannot import name 'Topic' from 'avtv.models'`.

- [ ] **Step 3: Add the model**

In `src/avtv/models.py`, append at the bottom (after the existing `Selection` class):

```python
class Topic(BaseModel):
    topic: str
    llm_suggestion: str
    source: Literal["user", "llm"]
```

(`Literal` and `BaseModel` are already imported at the top of the file.)

- [ ] **Step 4: Tests pass**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: existing tests + the two new ones pass.

- [ ] **Step 5: Add the RunDir test**

Append to `tests/test_runs.py`:

```python
def test_rundir_save_and_load_topic(tmp_path):
    from avtv.models import Topic
    from avtv.runs import RunDir

    rd = RunDir(base_dir=tmp_path, run_id="r1")
    rd.ensure()
    assert rd.has_topic() is False

    t = Topic(topic="vintage cars", llm_suggestion="vintage cars", source="llm")
    rd.save_topic(t)

    assert rd.has_topic() is True
    loaded = rd.load_topic()
    assert loaded == t
```

- [ ] **Step 6: Verify the new test fails**

```bash
poetry run pytest tests/test_runs.py::test_rundir_save_and_load_topic -v
```

Expected: `AttributeError: 'RunDir' object has no attribute 'has_topic'`.

- [ ] **Step 7: Update RunDir**

In `src/avtv/runs.py`:

a. Add the constant near the other artifact names (currently at line 13-16):

```python
class RunDir:
    BLOCKS = "01_blocks.json"
    BRIEFS = "02_visual_briefs.json"
    SEARCH = "03_search_results.json"
    SELECTIONS = "04_selections.json"
    TOPIC = "00_topic.json"
```

b. Add import for `Topic` next to the existing model imports at the top of the file:

```python
from avtv.models import Block, Candidate, Selection, Topic, VisualBrief
```

c. Add save/load/has methods. Append after `load_selections` (around line 75):

```python
    def save_topic(self, topic: Topic) -> None:
        (self.path / self.TOPIC).write_text(
            json.dumps(topic.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_topic(self) -> Topic:
        raw = json.loads((self.path / self.TOPIC).read_text(encoding="utf-8"))
        return Topic.model_validate(raw)

    def has_topic(self) -> bool:
        return (self.path / self.TOPIC).exists()
```

- [ ] **Step 8: Tests pass**

```bash
poetry run pytest tests/test_runs.py tests/test_models.py -v
poetry run pytest -q
```

Expected: all pass; full suite is 90 + 3 new tests = 93 passed.

- [ ] **Step 9: Commit**

```bash
git add src/avtv/models.py src/avtv/runs.py tests/test_models.py tests/test_runs.py
git commit -m "feat(models): add Topic model and RunDir save/load for 00_topic.json"
```

---

### Task 2: Topic extraction module + tests

A small dedicated function that calls the LLM (Claude or OpenAI based on provider name) with a tiny prompt asking for a one-sentence theme. Returns a `str`.

**Files:**
- Create: `src/avtv/briefing/topic.py`
- Test: `tests/test_briefing_topic.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_briefing_topic.py`:

```python
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
```

- [ ] **Step 2: Verify the tests fail**

```bash
poetry run pytest tests/test_briefing_topic.py -v
```

Expected: `ModuleNotFoundError: No module named 'avtv.briefing.topic'`.

- [ ] **Step 3: Create the module**

Create `src/avtv/briefing/topic.py`:

```python
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
        max_tokens=128,
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
```

- [ ] **Step 4: Tests pass**

```bash
poetry run pytest tests/test_briefing_topic.py -v
poetry run pytest -q
```

Expected: 4 new tests pass; full suite 97 passed (was 93, +4).

- [ ] **Step 5: Commit**

```bash
git add src/avtv/briefing/topic.py tests/test_briefing_topic.py
git commit -m "feat(briefing): add LLM-driven topic extraction (Claude + GPT)"
```

---

### Task 3: Make the brief system prompt topic-aware

Today `briefing/prompt.py` exports `SYSTEM_PROMPT` as a constant string. We replace that with `build_system_prompt(topic: str | None) -> str` so the brief stage can optionally prepend a topic clause. Existing callers that imported `SYSTEM_PROMPT` are updated to call `build_system_prompt(None)` for the same behavior.

**Files:**
- Modify: `src/avtv/briefing/prompt.py`
- Test: `tests/test_briefing_prompt.py`

- [ ] **Step 1: Look at the existing prompt test**

```bash
poetry run python -c "import pathlib; print(pathlib.Path('tests/test_briefing_prompt.py').read_text())"
```

Read the existing assertions so the new tests match the file's style.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_briefing_prompt.py`:

```python
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
```

- [ ] **Step 3: Verify failure**

```bash
poetry run pytest tests/test_briefing_prompt.py -v
```

Expected: `ImportError: cannot import name 'build_system_prompt'`.

- [ ] **Step 4: Update `prompt.py`**

Replace `src/avtv/briefing/prompt.py` entirely with:

```python
from avtv.models import Block

SYSTEM_PROMPT = """\
You are a video researcher producing B-roll search queries for a long-form
documentary video. The narration audio is split into fixed 8-second blocks.
For each block you must produce one search-query "visual brief".

RULES:
1. Always produce queries in ENGLISH, regardless of the input language.
2. Choose `kind`:
   - "video" for generic action, scenery, behavior — anything stock footage covers.
   - "image" for named historical entities, specific people, dated events
     (an archival photo + Ken Burns will work better than generic stock).
   - "continuation" ONLY for blocks marked [empty] — the previous clip will
     be extended to cover the gap.
3. `query_en` is the primary, specific query (3-7 words).
4. `fallback_query` is a strictly more generic alternative (2-5 words).
5. Do not repeat the same `query_en` in three or more consecutive blocks.
6. `continuity_hint` (optional) is a one-line stylistic note like
   "wide shot, daytime, woodland" to keep visual tone coherent.
7. `notes` (optional) is a brief rationale, useful for debugging.

Read the entire script first; choose visuals so the video has a natural
arc and variation. Output strictly conforms to the provided JSON schema.
"""


_TOPIC_CLAUSE = """\
VIDEO TOPIC: {topic}

The above is the overall theme of this documentary. Use it to disambiguate
proper nouns and enrich vague queries. For example, in a video about
classic cars, the query for a block mentioning "Maverick" should be
something like "Ford Maverick 1970 muscle car" — NOT just "Maverick"
(which would match the Top Gun aircraft). Do not mechanically prefix every
query with the topic; only add context where the query would otherwise be
ambiguous or too generic.

"""


def build_system_prompt(topic: str | None) -> str:
    """Build the brief-stage system prompt, optionally prefixed with a topic clause.

    Empty/whitespace-only `topic` is treated as None and returns the legacy
    prompt unchanged.
    """
    if topic is None or not topic.strip():
        return SYSTEM_PROMPT
    return _TOPIC_CLAUSE.format(topic=topic.strip()) + SYSTEM_PROMPT


def build_user_message(blocks: list[Block]) -> str:
    lines: list[str] = ["Here is the full script:\n"]
    for b in blocks:
        body = b.text if not b.is_empty else "[empty]"
        lines.append(f"[BLOCK {b.idx}] ({b.start:.0f}-{b.end:.0f}s)\n{body}\n")
    return "\n".join(lines)
```

- [ ] **Step 5: Tests pass**

```bash
poetry run pytest tests/test_briefing_prompt.py -v
poetry run pytest -q
```

Expected: 3 new tests pass; full suite 100 passed.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/briefing/prompt.py tests/test_briefing_prompt.py
git commit -m "feat(briefing): add build_system_prompt(topic) for topic-aware queries"
```

---

### Task 4: Thread `topic` through the LLM providers

Both `ClaudeProvider.generate_briefs` and `OpenAIProvider.generate_briefs` get an optional `topic: str | None = None` kwarg. They build the system prompt via `build_system_prompt(topic)` instead of using the static `SYSTEM_PROMPT` import. The `LLMProvider` Protocol signature is widened to match.

**Files:**
- Modify: `src/avtv/briefing/llm.py`
- Modify: `src/avtv/briefing/claude.py`
- Modify: `src/avtv/briefing/gpt.py`
- Test: `tests/test_briefing_claude.py`, `tests/test_briefing_gpt.py`

- [ ] **Step 1: Write the failing test for Claude**

Append to `tests/test_briefing_claude.py`:

```python
def test_claude_provider_threads_topic_into_system_prompt(mock_anthropic_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [
        Block.from_text(1, 0.0, 8.0, "x"),
        Block.from_text(2, 8.0, 16.0, "y"),
    ]

    with patch("avtv.briefing.claude.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_anthropic_response

        provider = ClaudeProvider()
        provider.generate_briefs(blocks, topic="Classic muscle cars of the 1970s")

    call = instance.messages.create.call_args
    system_arg = call.kwargs["system"]
    # `system` is a list with one cache_control block containing text.
    assert isinstance(system_arg, list)
    system_text = system_arg[0]["text"]
    assert "Classic muscle cars of the 1970s" in system_text
    assert "RULES:" in system_text  # legacy rules still present
```

- [ ] **Step 2: Write the failing test for GPT**

Append to `tests/test_briefing_gpt.py`:

```python
def test_gpt_provider_threads_topic_into_system_prompt(mock_openai_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [
        Block.from_text(1, 0.0, 8.0, "x"),
        Block.from_text(2, 8.0, 16.0, "y"),
    ]

    with patch("avtv.briefing.gpt.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create.return_value = mock_openai_response

        provider = OpenAIProvider()
        provider.generate_briefs(blocks, topic="Roman aqueducts")

    call = instance.chat.completions.create.call_args
    messages = call.kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "Roman aqueducts" in system_msg["content"]
    assert "RULES:" in system_msg["content"]
```

If `mock_openai_response` is not already a fixture in `tests/test_briefing_gpt.py`, look at how the existing tests in that file set up the response and adapt. If you need to create a fixture, define it at the top of the file as a `@pytest.fixture` returning a MagicMock with `.choices[0].message.content` set to a valid JSON briefs string.

- [ ] **Step 3: Verify both tests fail**

```bash
poetry run pytest tests/test_briefing_claude.py::test_claude_provider_threads_topic_into_system_prompt tests/test_briefing_gpt.py::test_gpt_provider_threads_topic_into_system_prompt -v
```

Expected: failures because `generate_briefs` doesn't accept `topic`.

- [ ] **Step 4: Update the Protocol**

Replace `src/avtv/briefing/llm.py`:

```python
from typing import Protocol

from avtv.models import Block, VisualBrief


class LLMProvider(Protocol):
    def generate_briefs(
        self,
        blocks: list[Block],
        topic: str | None = None,
    ) -> list[VisualBrief]: ...
```

- [ ] **Step 5: Update Claude provider**

In `src/avtv/briefing/claude.py`:

a. Change the import (line 5):
```python
from avtv.briefing.prompt import build_system_prompt, build_user_message
```
(drop `SYSTEM_PROMPT`)

b. Update the method signature and the system prompt construction:

```python
    def generate_briefs(
        self,
        blocks: list[Block],
        topic: str | None = None,
    ) -> list[VisualBrief]:
        user_msg = build_user_message(blocks)

        response = self.client.messages.create(
            model=self.settings.claude_model,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": build_system_prompt(topic),
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
```

- [ ] **Step 6: Update GPT provider**

In `src/avtv/briefing/gpt.py`:

a. Change the import:
```python
from avtv.briefing.prompt import build_system_prompt, build_user_message
```

b. Update method:

```python
    def generate_briefs(
        self,
        blocks: list[Block],
        topic: str | None = None,
    ) -> list[VisualBrief]:
        user_msg = build_user_message(blocks)

        response = self.client.chat.completions.create(
            model=self.settings.gpt_model,
            messages=[
                {"role": "system", "content": build_system_prompt(topic)},
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
```

- [ ] **Step 7: Verify**

```bash
poetry run pytest tests/test_briefing_claude.py tests/test_briefing_gpt.py -v
poetry run pytest -q
```

Expected: existing claude/gpt tests pass (they call `generate_briefs(blocks)` without topic and that still works because of the default `None`), plus the two new "threads topic" tests pass. Full suite 102 passed.

- [ ] **Step 8: Commit**

```bash
git add src/avtv/briefing/llm.py src/avtv/briefing/claude.py src/avtv/briefing/gpt.py tests/test_briefing_claude.py tests/test_briefing_gpt.py
git commit -m "feat(briefing): thread topic through provider.generate_briefs"
```

---

### Task 5: CLI `topic` command + wire `build` and `brief`

`avtv topic --run-id X` extracts the topic via LLM, prints the suggestion, prompts the user (Enter accepts, typing replaces), saves `00_topic.json`. A non-interactive mode `--topic "..."` skips the LLM call and prompt.

`avtv brief --run-id X` reads `00_topic.json` if present and passes its `topic` field to `provider.generate_briefs`. Backwards compatible: no topic file → no topic passed.

`avtv build` runs `parse → topic (interactive) → brief → search → select → assemble`.

**Files:**
- Modify: `src/avtv/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Look at `tests/test_cli.py` first (it currently exercises only `parse`):
```bash
poetry run python -c "import pathlib; print(pathlib.Path('tests/test_cli.py').read_text())"
```

Append a test for the non-interactive `--topic` path. The interactive prompt is harder to test cleanly; we cover non-interactive only:

```python
def test_cli_topic_non_interactive_saves_run_artifact(tmp_path, monkeypatch):
    """`avtv topic --run-id X --topic "..."` writes 00_topic.json without LLM call."""
    from typer.testing import CliRunner

    for k in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(k, "x")

    from avtv.cli import app
    from avtv.models import Block
    from avtv.runs import RunDir

    rd = RunDir(base_dir=tmp_path, run_id="r1")
    rd.ensure()
    rd.save_blocks([Block.from_text(1, 0.0, 8.0, "hello")])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "topic",
            "--run-id", "r1",
            "--runs-dir", str(tmp_path),
            "--topic", "Test theme",
        ],
    )
    assert result.exit_code == 0, result.output
    assert rd.has_topic()
    t = rd.load_topic()
    assert t.topic == "Test theme"
    assert t.source == "user"
```

- [ ] **Step 2: Verify failure**

```bash
poetry run pytest tests/test_cli.py -v
```

Expected: command `topic` not found (Typer returns non-zero exit).

- [ ] **Step 3: Add the `topic` command + wire `build` and `brief`**

In `src/avtv/cli.py`:

a. Add imports at the top (after existing imports):

```python
from avtv.briefing.topic import extract_topic_via_llm
from avtv.models import Topic
```

b. Add the new command. Place it AFTER `brief` and BEFORE `search` (around line 73, just after the existing `brief` command):

```python
@app.command()
def topic(
    run_id: str = typer.Option(..., help="Existing run id"),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
    provider: str | None = typer.Option(None, help="claude | gpt"),  # noqa: B008
    topic: str | None = typer.Option(  # noqa: B008
        None,
        "--topic",
        help="Skip LLM detection and save this topic verbatim (non-interactive).",
    ),
) -> None:
    """Stage 1.5: detect (or set) the video's overall topic.

    Interactive: runs the LLM, prints the suggestion, waits for user input.
    Non-interactive (--topic "..."): saves the given topic and exits.
    """
    settings = Settings()  # type: ignore[call-arg]
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    blocks = rd.load_blocks()
    name = provider or settings.default_llm_provider

    if topic is not None:
        # Non-interactive path: save the provided topic and exit.
        record = Topic(topic=topic, llm_suggestion=topic, source="user")
        rd.save_topic(record)
        console.print(f"[green]topic saved (non-interactive)[/green]: {topic}")
        return

    suggested = extract_topic_via_llm(blocks, provider_name=name, settings=settings)
    console.print(f"[bold]Suggested topic:[/bold] {suggested}")
    typed = typer.prompt(
        "Press Enter to accept, or type a replacement",
        default=suggested,
        show_default=False,
    ).strip()

    final = typed if typed else suggested
    source = "user" if typed and typed != suggested else "llm"
    record = Topic(topic=final, llm_suggestion=suggested, source=source)
    rd.save_topic(record)
    console.print(f"[green]topic saved[/green]: {final}  ({source})")
```

c. Update the `brief` command to pass topic. Currently at `src/avtv/cli.py:57-71`:

```python
@app.command()
def brief(
    run_id: str = typer.Option(..., help="Existing run id"),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
    provider: str | None = typer.Option(None, help="claude | gpt"),  # noqa: B008
) -> None:
    """Stage 2: generate visual briefs via LLM."""
    settings = Settings()  # type: ignore[call-arg]
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    blocks = rd.load_blocks()
    name = provider or settings.default_llm_provider
    llm = get_provider(name)
    topic_str = rd.load_topic().topic if rd.has_topic() else None
    briefs = llm.generate_briefs(blocks, topic=topic_str)
    rd.save_briefs(briefs)
    console.print(f"[green]briefed[/green] via {name}: {len(briefs)} briefs")
```

d. Update `build` to run the topic stage. Currently at `src/avtv/cli.py:135-185`. Insert the topic stage between `parsed N blocks` and `briefed via N`. The `build` command becomes:

```python
@app.command()
def build(
    audio: Path = typer.Option(..., exists=True),  # noqa: B008
    script: Path = typer.Option(..., exists=True),  # noqa: B008
    out: Path = typer.Option(...),  # noqa: B008
    provider: str | None = typer.Option(None),  # noqa: B008
    run_id: str | None = typer.Option(None),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
    cache_dir: str | None = typer.Option(None),  # noqa: B008
    topic: str | None = typer.Option(  # noqa: B008
        None,
        "--topic",
        help="Skip LLM topic detection; use this string verbatim.",
    ),
) -> None:
    """Run the full pipeline: parse → topic → brief → search → select → assemble."""
    settings = Settings()  # type: ignore[call-arg]
    base = _runs_dir(runs_dir)
    rd = RunDir(base_dir=base, run_id=run_id) if run_id else RunDir.new(base)
    rd.ensure()

    console.print(f"[bold]run-id:[/bold] {rd.run_id}")

    blocks = parse_script(script.read_text(encoding="utf-8"))
    rd.save_blocks(blocks)
    console.print(f"  parsed {len(blocks)} blocks")

    name = provider or settings.default_llm_provider

    if topic is not None:
        rec = Topic(topic=topic, llm_suggestion=topic, source="user")
        rd.save_topic(rec)
        console.print(f"  topic (non-interactive): {topic}")
        topic_str: str | None = topic
    else:
        suggested = extract_topic_via_llm(blocks, provider_name=name, settings=settings)
        console.print(f"  suggested topic: {suggested}")
        typed = typer.prompt(
            "Press Enter to accept, or type a replacement",
            default=suggested,
            show_default=False,
        ).strip()
        final = typed if typed else suggested
        source = "user" if typed and typed != suggested else "llm"
        rec = Topic(topic=final, llm_suggestion=suggested, source=source)
        rd.save_topic(rec)
        console.print(f"  topic confirmed: {final}  ({source})")
        topic_str = final

    llm = get_provider(name)
    briefs = llm.generate_briefs(blocks, topic=topic_str)
    rd.save_briefs(briefs)
    console.print(f"  briefed via {name}")

    orch = _build_orchestrator(settings)
    results = asyncio.run(orch.search_for_briefs(briefs, concurrency=settings.search_concurrency))
    rd.save_search_results(results)
    console.print(f"  searched: {sum(len(v) for v in results.values())} candidates")

    selections = asyncio.run(
        select_per_block(
            briefs,
            results,
            settings=settings,
            image_search=orch.search_images_for_brief,
        )
    )
    rd.save_selections(selections)
    console.print(f"  selected {len(selections)} clips")

    cache = Path(cache_dir) if cache_dir else Path(settings.cache_dir)
    downloader = Downloader(cache_dir=cache)
    work = rd.path / "segments"
    assemble_run(
        selections=selections,
        narration_path=audio,
        output_path=out,
        work_dir=work,
        downloader=downloader,
        target_w=settings.target_resolution[0],
        target_h=settings.target_resolution[1],
        target_fps=settings.target_fps,
        clip_audio_db=settings.clip_audio_db_offset,
    )
    console.print(f"[green]build complete[/green] → {out}")
```

- [ ] **Step 4: Update the `runs` command listing to show topic stage**

Currently `src/avtv/cli.py` `runs` command (around line 198-207) shows parse/brief/search/select. Add topic. Find:

```python
        for name, label in [
            (RunDir.BLOCKS, "parse"),
            (RunDir.BRIEFS, "brief"),
            (RunDir.SEARCH, "search"),
            (RunDir.SELECTIONS, "select"),
        ]:
```

Change to:

```python
        for name, label in [
            (RunDir.BLOCKS, "parse"),
            (RunDir.TOPIC, "topic"),
            (RunDir.BRIEFS, "brief"),
            (RunDir.SEARCH, "search"),
            (RunDir.SELECTIONS, "select"),
        ]:
```

And in the `inspect` command (around line 217-225), find:

```python
    for name, label in [
        (RunDir.BLOCKS, "blocks"),
        (RunDir.BRIEFS, "briefs"),
        (RunDir.SEARCH, "search results"),
        (RunDir.SELECTIONS, "selections"),
    ]:
```

Change to:

```python
    for name, label in [
        (RunDir.BLOCKS, "blocks"),
        (RunDir.TOPIC, "topic"),
        (RunDir.BRIEFS, "briefs"),
        (RunDir.SEARCH, "search results"),
        (RunDir.SELECTIONS, "selections"),
    ]:
```

- [ ] **Step 5: Tests pass**

```bash
poetry run pytest tests/test_cli.py -v
poetry run pytest -q
```

Expected: new CLI test passes; full suite 103 passed.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/cli.py tests/test_cli.py
git commit -m "feat(cli): add topic command and topic stage to build pipeline"
```

---

### Task 6: Streamlit UI — fix the async-select regression + add topic step

Two things in one task because the New-build form must be reworked anyway:

1. **Regression fix:** `_run_stage_select` in `src/avtv/ui/streamlit_app.py:108` still calls `select_per_block` synchronously. This crashes today (the merged PR #1 did not catch it; CLI tests don't exercise the UI path). Wrap in `asyncio.run` with `image_search=orch.search_images_for_brief`.

2. **Topic step:** After parse, before brief, run topic extraction, show editable text field, build button uses the confirmed value.

**Files:**
- Modify: `src/avtv/ui/streamlit_app.py`
- Test: smoke-only (Streamlit UIs are hard to unit-test cleanly)

- [ ] **Step 1: Fix the async-select regression**

In `src/avtv/ui/streamlit_app.py`, find `_run_stage_select` (currently around line 105-110):

```python
def _run_stage_select(rd: RunDir, settings: Settings) -> int:
    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    sels = select_per_block(briefs, candidates, settings=settings)
    rd.save_selections(sels)
    return len(sels)
```

Replace with:

```python
def _run_stage_select(rd: RunDir, settings: Settings) -> int:
    import asyncio

    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    orch = SearchOrchestrator(adapters=_build_adapters(settings), top_k=settings.search_top_k)
    sels = asyncio.run(
        select_per_block(
            briefs,
            candidates,
            settings=settings,
            image_search=orch.search_images_for_brief,
        )
    )
    rd.save_selections(sels)
    return len(sels)
```

If `SearchOrchestrator` and `_build_adapters` (or whatever the existing stage uses to build the orchestrator in this file) aren't already imported / defined, look at how `_run_stage_search` (around line 95) builds its orchestrator — reuse the same helper. If `_run_stage_search` uses a local helper, factor it out so `_run_stage_select` can call it too. Concretely: if `_run_stage_search` does `orch = SearchOrchestrator(...)` inline, extract a module-level `_build_orchestrator(settings)` function and have both stages call it.

- [ ] **Step 2: Add the topic stage helper**

Just below `_run_stage_parse` (around line 87 in the modified file), add:

```python
def _run_stage_topic_extract(rd: RunDir, provider_name: str, settings: Settings) -> str:
    """Run the LLM topic detector. Returns the suggested topic string."""
    from avtv.briefing.topic import extract_topic_via_llm

    blocks = rd.load_blocks()
    return extract_topic_via_llm(blocks, provider_name=provider_name, settings=settings)


def _run_stage_topic_save(rd: RunDir, topic: str, suggested: str) -> None:
    from avtv.models import Topic

    source = "user" if topic.strip() != suggested.strip() else "llm"
    rd.save_topic(Topic(topic=topic.strip(), llm_suggestion=suggested, source=source))
```

- [ ] **Step 3: Reshape the New-build flow**

The current New-build flow runs all stages in one button click. We need a two-step UX: file upload → topic confirmation → start build.

Replace the New-build section (currently around lines 203-269) with this structure:

```python
if mode == "new":
    st.title("New build")

    col1, col2 = st.columns([2, 1])

    with col1:
        audio_file = st.file_uploader("Audio (MP3)", type=["mp3"])
        script_file = st.file_uploader("Script (DOTTI SYNC .txt)", type=["txt"])

    with col2:
        provider = st.radio(
            "LLM provider",
            ["claude", "gpt"],
            index=0 if settings.default_llm_provider == "claude" else 1,
        )
        custom_id = st.text_input("Run ID (optional)", placeholder="auto-generated")

    files_ready = audio_file is not None and script_file is not None

    # Phase 1: files uploaded but topic not yet detected.
    if files_ready and "pending_topic_run" not in st.session_state:
        if st.button("📑 Detect topic", type="primary"):
            rd = (
                RunDir(base_dir=runs_base, run_id=custom_id)
                if custom_id
                else RunDir.new(runs_base)
            )
            rd.ensure()
            audio_path, script_path = _save_uploads(rd, audio_file, script_file)
            with st.spinner("Parsing & detecting topic…"):
                _run_stage_parse(rd, script_path)
                suggested = _run_stage_topic_extract(rd, provider, settings)
            st.session_state["pending_topic_run"] = {
                "run_id": rd.run_id,
                "audio_path": str(audio_path),
                "suggested_topic": suggested,
                "provider": provider,
            }
            st.rerun()

    # Phase 2: topic suggested, awaiting confirmation.
    if "pending_topic_run" in st.session_state:
        ctx = st.session_state["pending_topic_run"]
        st.info(f"Run ID: `{ctx['run_id']}`")
        st.markdown(f"**Suggested topic:** {ctx['suggested_topic']}")
        confirmed_topic = st.text_input(
            "Confirm or edit the topic",
            value=ctx["suggested_topic"],
            key="confirmed_topic_field",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            cancel = st.button("✖ Cancel")
        with col_b:
            go = st.button("▶ Start build", type="primary")

        if cancel:
            st.session_state.pop("pending_topic_run", None)
            st.rerun()

        if go:
            rd = RunDir(base_dir=runs_base, run_id=ctx["run_id"])
            _run_stage_topic_save(rd, confirmed_topic, ctx["suggested_topic"])
            audio_path = Path(ctx["audio_path"])
            provider = ctx["provider"]

            with st.status("Running pipeline…", expanded=True) as status:
                st.write(f"**Run ID:** `{rd.run_id}`")
                st.write(f"📑 Topic: `{confirmed_topic}`")

                st.write(f"🧠 **Brief** (via {provider})")
                n_briefs = _run_stage_brief(rd, provider)
                st.write(f"  → {n_briefs} briefs")

                st.write("🔍 **Search**")
                n_cands = _run_stage_search(rd, settings)
                st.write(f"  → {n_cands} candidates")

                st.write("🎯 **Select**")
                n_sels = _run_stage_select(rd, settings)
                st.write(f"  → {n_sels} clips")

                st.write("🎬 **Assemble** (download + ffmpeg)")
                seg_bar = st.progress(0.0, text="Encoding segments…")

                def on_segment(i: int, total: int) -> None:
                    seg_bar.progress(i / total, text=f"Segment {i}/{total}")

                def on_mux() -> None:
                    seg_bar.progress(1.0, text="Muxing final MP4…")

                out_path = _run_stage_assemble(
                    rd, audio_path, settings, on_segment=on_segment, on_mux=on_mux
                )
                seg_bar.empty()
                st.write(f"  → {out_path}")

                status.update(label="✅ Build complete", state="complete")

            st.session_state["run_id"] = rd.run_id
            st.session_state["mode"] = "run"
            st.session_state.pop("pending_topic_run", None)
            st.rerun()
```

- [ ] **Step 4: Update `_run_stage_brief` to read topic**

The existing `_run_stage_brief` (around line 87):

```python
def _run_stage_brief(rd: RunDir, provider_name: str) -> int:
    blocks = rd.load_blocks()
    llm = get_provider(provider_name)
    briefs = llm.generate_briefs(blocks)
    rd.save_briefs(briefs)
    return len(briefs)
```

Update to:

```python
def _run_stage_brief(rd: RunDir, provider_name: str) -> int:
    blocks = rd.load_blocks()
    llm = get_provider(provider_name)
    topic_str = rd.load_topic().topic if rd.has_topic() else None
    briefs = llm.generate_briefs(blocks, topic=topic_str)
    rd.save_briefs(briefs)
    return len(briefs)
```

- [ ] **Step 5: Update the STAGES list and stage flags**

In the file (around the top), find the stage indicators / per-run status (whatever variable corresponds to `STAGES` or the stage-marker dict). Add a row for topic:

```python
STAGES = [
    ("parse", RunDir.BLOCKS),
    ("topic", RunDir.TOPIC),  # NEW
    ("brief", RunDir.BRIEFS),
    ("search", RunDir.SEARCH),
    ("select", RunDir.SELECTIONS),
]
```

If the file uses a different shape for stage indicators, adapt — the goal is for the inspector's stage column to show ✓/· for topic too. Read the surrounding code first to match style.

- [ ] **Step 6: Run the suite**

```bash
poetry run pytest -q
```

Expected: 103 passed (no UI-specific tests, but module import correctness verified).

- [ ] **Step 7: Smoke-launch the UI**

```bash
poetry run python -c "from avtv.ui import streamlit_app; print('OK')"
```

Expected: `OK`. If a `NameError` or `ImportError` fires, check that `SearchOrchestrator` / `_build_adapters` / `_build_orchestrator` exist where you referenced them.

For an end-to-end visual smoke, optionally start the UI:
```bash
poetry run avtv ui --port 8501 &
```
…visit `http://localhost:8501`, upload an MP3+TXT, click "Detect topic", confirm the editable field appears with the suggestion, edit it, click "Start build", verify the pipeline runs to assemble. Then `pkill -f "streamlit run"`.

- [ ] **Step 8: Commit**

```bash
git add src/avtv/ui/streamlit_app.py
git commit -m "feat(ui): topic confirmation step + fix async-select regression"
```

---

### Task 7: README + smoke verification

- [ ] **Step 1: Update the pipeline table** (`README.md` around line 188)

Replace the row block:

```markdown
| **parse** | `01_blocks.json` | Splits DOTTI SYNC into typed `Block` records. |
| **brief** | `02_visual_briefs.json` | LLM (Claude / GPT) reads the full script and emits one English search query + fallback per block, plus `kind` (`video` / `image` / `continuation`). |
```

With:

```markdown
| **parse** | `01_blocks.json` | Splits DOTTI SYNC into typed `Block` records. |
| **topic** | `00_topic.json` | LLM proposes the video's overall theme in one sentence. You confirm or edit before the brief stage runs. Theme is injected into the brief system prompt so queries become topic-aware (e.g. `"Maverick"` in a classic-cars video becomes `"Ford Maverick muscle car 1970"` and no longer matches the Top Gun aircraft). |
| **brief** | `02_visual_briefs.json` | LLM (Claude / GPT) reads the full script and emits one English search query + fallback per block, plus `kind` (`video` / `image` / `continuation`). Uses the confirmed topic for disambiguation. |
```

Also update the ASCII pipeline diagram at the top of the README (`README.md:8-12`):

```
input/audio.mp3 ─┐
                 ├─► parse ─► topic ─► brief ─► search ─► select ─► assemble ─► output.mp4
input/script.txt ┘        (LLM+you)  (LLM)    (5 APIs)            (ffmpeg)
```

- [ ] **Step 2: Run the full suite one more time**

```bash
poetry run pytest -q
```

Expected: 103 passed.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document the new topic stage and topic-aware briefs"
```

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/topic-aware-queries
gh pr create --title "feat: topic-aware queries (user-confirmed theme) + fix UI async-select regression" --body "$(cat <<'PRBODY'
## Summary
- New \`topic\` stage between \`parse\` and \`brief\`: LLM proposes a one-sentence theme, user confirms via CLI prompt or Streamlit text field.
- Confirmed theme is injected into the brief system prompt — LLM uses it to disambiguate proper nouns (so \"Maverick\" in a classic-cars video becomes \"Ford Maverick muscle car\" instead of Top Gun aircraft).
- Persisted as \`runs/<id>/00_topic.json\` with audit field (\`source: user | llm\`).
- Drive-by: fix UI \`_run_stage_select\` async regression introduced during PR #1 (the merged code called the now-async selector synchronously).

## Test plan
- [x] Unit: 103 tests pass (was 90 baseline; +13 new).
- [x] CLI smoke: \`avtv topic --run-id X --topic "Test"\` writes \`00_topic.json\` non-interactively.
- [ ] CLI interactive: \`avtv topic --run-id X\` shows suggestion and accepts override.
- [ ] UI smoke: New-build form shows "Detect topic" → suggestion → editable field → Start build.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PRBODY
)"
```

---

## Self-Review

**Spec coverage:**
- "LLM identifies theme from sync file" → Task 2 (`extract_topic_via_llm`).
- "Ask user for confirmation" → Task 5 (CLI `typer.prompt`) and Task 6 (UI two-phase form).
- "User can type a correction" → Same.
- "Theme goes into brief queries" → Tasks 3+4 (`build_system_prompt(topic)` + provider plumbing).
- "Persistent artifact" → Task 1 (`00_topic.json`).
- "Backward compat: brief still works without topic" → Task 3 returns legacy `SYSTEM_PROMPT` when `topic is None`, Task 4 defaults `topic=None`, Task 5 reads file only if present.
- "Pipeline visualization" → Task 5 updates `runs`/`inspect`, Task 6 updates UI stage indicators, Task 7 updates README.

**Placeholder scan:** No `TBD`/`TODO`/`similar to Task N` patterns. Each step includes runnable code or a precise command.

**Type consistency:**
- `Topic.source: Literal["user", "llm"]` is consistent across tasks (Tasks 1, 5, 6).
- `generate_briefs(blocks, topic: str | None = None)` is consistent across Tasks 3, 4, 5, 6.
- `extract_topic_via_llm(blocks, provider_name: str, settings: Settings)` is consistent across Tasks 2, 5, 6.
- `build_system_prompt(topic: str | None) -> str` is consistent across Tasks 3 and 4.
- `RunDir.TOPIC = "00_topic.json"` named consistently in Tasks 1 and 5.

**One soft spot worth surfacing during execution:** the UI two-phase form relies on `st.session_state["pending_topic_run"]` to survive a `st.rerun()` between phases. If Streamlit's rerun behavior diverges from expectations, the implementer should fall back to a single-phase form where "Detect topic" expands an inline editor without rerunning. Either pattern satisfies the spec; the two-phase one is cleaner UX.
