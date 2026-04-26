# auto-video-to-video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that ingests an MP3 narration plus a DOTTI SYNC script (8-second blocks) and produces a 1080p YouTube documentary MP4, finding B-roll automatically from free video/image APIs.

**Architecture:** Pipeline with disk-persisted artifacts between stages (parser → briefing → search → selector → downloader → assembler). Each stage is a separate module re-runnable in isolation. LLM-driven visual briefing supports both Claude and GPT.

**Tech Stack:** Python 3.11+, Poetry, Typer, pydantic, httpx, anthropic, openai, ffmpeg-python, internetarchive, rich, tenacity, pytest, respx (HTTP mocking).

**Spec:** `docs/superpowers/specs/2026-04-26-auto-video-to-video-design.md`

> **Naming note:** spec lists the OpenAI provider file as `briefing/openai.py`. To avoid shadowing the `openai` SDK package, this plan uses `briefing/gpt.py` instead. Functionally identical.

---

## File Structure

```
src/avtv/
├── __init__.py                # exports __version__
├── cli.py                     # Typer app (one file — orchestrates stages)
├── config.py                  # Settings via pydantic-settings
├── models.py                  # Block, VisualBrief, Candidate, Selection
├── parser.py                  # DOTTI SYNC text → list[Block]
├── briefing/
│   ├── __init__.py            # exports get_provider()
│   ├── llm.py                 # LLMProvider Protocol
│   ├── prompt.py              # system + user prompt construction
│   ├── claude.py              # ClaudeProvider
│   └── gpt.py                 # OpenAIProvider
├── search/
│   ├── __init__.py
│   ├── base.py                # SearchAdapter Protocol
│   ├── pexels.py
│   ├── pixabay.py
│   ├── archive_org.py
│   ├── wikimedia.py
│   ├── unsplash.py
│   └── orchestrator.py        # parallel search across adapters
├── selector.py                # ranking + neighbor dedup
├── downloader.py              # cache by URL hash
├── assembler.py               # ffmpeg pipeline
└── runs.py                    # run-dir helpers (paths, JSON load/save)

tests/
├── conftest.py                # shared fixtures (sample blocks, mock LLM responses)
├── test_parser.py
├── test_models.py
├── test_briefing_prompt.py
├── test_briefing_claude.py    # respx-mocked
├── test_briefing_gpt.py       # respx-mocked
├── test_search_pexels.py      # respx-mocked
├── test_search_pixabay.py
├── test_search_archive.py
├── test_search_wikimedia.py
├── test_search_unsplash.py
├── test_search_orchestrator.py
├── test_selector.py
├── test_downloader.py
├── test_assembler.py          # mocks ffmpeg invocation
└── fixtures/
    ├── sample_script.txt      # 5-block DOTTI SYNC
    ├── pexels_response.json
    ├── pixabay_response.json
    └── ...
```

---

## Phase 1 — Bootstrap

### Task 1: Initialize Poetry project

**Files:**
- Create: `pyproject.toml`
- Create: `src/avtv/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`
- Create: `.python-version`

- [ ] **Step 1: Initialize Poetry**

```bash
cd /Users/caiosantos/Projects/auto-video-to-video
poetry init --name avtv --python "^3.11" --no-interaction
```

- [ ] **Step 2: Add runtime deps**

```bash
poetry add typer pydantic pydantic-settings httpx "anthropic>=0.39" "openai>=1.50" ffmpeg-python internetarchive rich tenacity
```

- [ ] **Step 3: Add dev deps**

```bash
poetry add --group dev pytest pytest-asyncio respx ruff mypy
```

- [ ] **Step 4: Configure src layout in pyproject.toml**

Add to `pyproject.toml` under `[tool.poetry]`:

```toml
packages = [{include = "avtv", from = "src"}]
```

Add at the bottom of `pyproject.toml`:

```toml
[tool.poetry.scripts]
avtv = "avtv.cli:app"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 5: Create package skeletons**

`src/avtv/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`.python-version`:
```
3.11
```

`.env.example`:
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
PEXELS_API_KEY=
PIXABAY_API_KEY=
UNSPLASH_API_KEY=
DEFAULT_LLM_PROVIDER=claude
```

- [ ] **Step 6: Verify install**

```bash
poetry install
poetry run python -c "import avtv; print(avtv.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml poetry.lock src/ tests/ .env.example .python-version
git commit -m "chore: bootstrap poetry project with deps and src layout"
```

---

### Task 2: Settings module

**Files:**
- Create: `src/avtv/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

`tests/test_config.py`:
```python
import os
from avtv.config import Settings


def test_settings_loads_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PEXELS_API_KEY", "test-pexels")
    monkeypatch.setenv("PIXABAY_API_KEY", "test-pixabay")
    monkeypatch.setenv("UNSPLASH_API_KEY", "test-unsplash")

    s = Settings()
    assert s.anthropic_api_key == "test-anthropic"
    assert s.default_llm_provider == "claude"
    assert s.target_block_duration == 8.0
    assert s.target_resolution == (1920, 1080)
    assert s.target_fps == 30
    assert s.clip_audio_db_offset == -20.0


def test_settings_default_provider_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gpt")

    s = Settings()
    assert s.default_llm_provider == "gpt"
```

- [ ] **Step 2: Run test, expect fail**

```bash
poetry run pytest tests/test_config.py -v
```
Expected: FAIL with import error.

- [ ] **Step 3: Implement Settings**

`src/avtv/config.py`:
```python
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    anthropic_api_key: str
    openai_api_key: str
    pexels_api_key: str
    pixabay_api_key: str
    unsplash_api_key: str

    # LLM
    default_llm_provider: Literal["claude", "gpt"] = "claude"
    claude_model: str = "claude-sonnet-4-6"
    gpt_model: str = "gpt-5"

    # Pipeline
    target_block_duration: float = 8.0
    target_resolution: tuple[int, int] = (1920, 1080)
    target_fps: int = 30
    clip_audio_db_offset: float = -20.0

    # Selector weights
    weight_aspect: float = 1.0
    weight_duration: float = 1.0
    weight_resolution: float = 1.0
    weight_source: float = 1.0
    repetition_penalty: float = 0.5
    neighbor_window: int = 2

    # Search
    search_top_k: int = 8
    search_concurrency: int = 5

    # Ken Burns
    ken_burns_zoom_start: float = 1.0
    ken_burns_zoom_end: float = 1.15
    ken_burns_pan_pct: float = 0.05

    # Loop crossfade
    loop_crossfade_seconds: float = 0.3

    # Paths
    runs_dir: str = "runs"
    cache_dir: str = "cache"
```

- [ ] **Step 4: Run test, expect pass**

```bash
poetry run pytest tests/test_config.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/config.py tests/test_config.py
git commit -m "feat(config): add Settings with API keys and pipeline defaults"
```

---

### Task 3: Data models

**Files:**
- Create: `src/avtv/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

`tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from avtv.models import Block, Candidate, Selection, VisualBrief


def test_block_basic():
    b = Block(idx=1, start=0.0, end=8.0, text="hello", is_empty=False)
    assert b.idx == 1
    assert b.end - b.start == 8.0


def test_block_empty_text_is_empty_true():
    # is_empty is computed from text, not passed in
    b = Block.from_text(idx=2, start=8.0, end=16.0, text="   ")
    assert b.is_empty is True


def test_block_with_text_is_empty_false():
    b = Block.from_text(idx=2, start=8.0, end=16.0, text="hello world")
    assert b.is_empty is False


def test_visual_brief_validates_kind():
    vb = VisualBrief(
        idx=1,
        query_en="forest sunlight",
        fallback_query="nature",
        kind="video",
        continuity_hint=None,
        notes=None,
    )
    assert vb.kind == "video"


def test_visual_brief_rejects_bad_kind():
    with pytest.raises(ValidationError):
        VisualBrief(
            idx=1,
            query_en="x",
            fallback_query="y",
            kind="movie",  # not allowed
            continuity_hint=None,
            notes=None,
        )


def test_candidate_video_has_duration():
    c = Candidate(
        source="pexels",
        url="https://example.com/v.mp4",
        preview_url=None,
        kind="video",
        duration=10.0,
        width=1920,
        height=1080,
        license="Pexels License",
        attribution="John Doe",
    )
    assert c.duration == 10.0


def test_selection_video_with_speed():
    s = Selection(
        idx=5,
        source="pexels",
        url="https://x/v.mp4",
        kind="video",
        trim_start=None,
        trim_end=None,
        speed_factor=0.9,
        loop=False,
        attribution=None,
    )
    assert s.speed_factor == 0.9
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_models.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement models**

`src/avtv/models.py`:
```python
from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    idx: int
    start: float
    end: float
    text: str
    is_empty: bool

    @classmethod
    def from_text(cls, idx: int, start: float, end: float, text: str) -> "Block":
        stripped = text.strip()
        return cls(idx=idx, start=start, end=end, text=text, is_empty=not stripped)


VisualKind = Literal["video", "image", "continuation"]
MediaKind = Literal["video", "image"]


class VisualBrief(BaseModel):
    idx: int
    query_en: str
    fallback_query: str
    kind: VisualKind
    continuity_hint: str | None = None
    notes: str | None = None


class Candidate(BaseModel):
    source: str
    url: str
    preview_url: str | None = None
    kind: MediaKind
    duration: float | None = None
    width: int
    height: int
    license: str
    attribution: str | None = None


class Selection(BaseModel):
    idx: int
    source: str
    url: str
    kind: VisualKind
    trim_start: float | None = None
    trim_end: float | None = None
    speed_factor: float | None = None
    loop: bool = False
    attribution: str | None = None
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_models.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/models.py tests/test_models.py
git commit -m "feat(models): add Block, VisualBrief, Candidate, Selection pydantic models"
```

---

### Task 4: Run-dir helpers

**Files:**
- Create: `src/avtv/runs.py`
- Create: `tests/test_runs.py`

- [ ] **Step 1: Write failing test**

`tests/test_runs.py`:
```python
import json
from pathlib import Path

from avtv.models import Block
from avtv.runs import RunDir


def test_run_dir_creates_paths(tmp_path):
    rd = RunDir(base_dir=tmp_path, run_id="abc")
    rd.ensure()
    assert (tmp_path / "abc").exists()
    assert (tmp_path / "abc" / "logs").exists()


def test_run_dir_save_load_blocks(tmp_path):
    rd = RunDir(base_dir=tmp_path, run_id="abc")
    rd.ensure()
    blocks = [Block.from_text(1, 0.0, 8.0, "hi"), Block.from_text(2, 8.0, 16.0, "")]
    rd.save_blocks(blocks)
    loaded = rd.load_blocks()
    assert loaded == blocks
    # Confirm pretty JSON on disk
    data = json.loads((tmp_path / "abc" / "01_blocks.json").read_text())
    assert isinstance(data, list)
    assert data[0]["idx"] == 1


def test_run_dir_new_id_is_timestamp(tmp_path):
    rd = RunDir.new(base_dir=tmp_path)
    # default ID is YYYYMMDD-HHMMSS or similar — at least 8 chars
    assert len(rd.run_id) >= 8
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_runs.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement RunDir**

`src/avtv/runs.py`:
```python
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from avtv.models import Block, Candidate, Selection, VisualBrief


class RunDir:
    BLOCKS = "01_blocks.json"
    BRIEFS = "02_visual_briefs.json"
    SEARCH = "03_search_results.json"
    SELECTIONS = "04_selections.json"

    def __init__(self, base_dir: Path, run_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id

    @classmethod
    def new(cls, base_dir: Path) -> "RunDir":
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(base_dir=base_dir, run_id=run_id)

    @property
    def path(self) -> Path:
        return self.base_dir / self.run_id

    @property
    def logs_path(self) -> Path:
        return self.path / "logs"

    def ensure(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _save_models(self, name: str, items: list[BaseModel]) -> None:
        out = [m.model_dump() for m in items]
        (self.path / name).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def _load_models(self, name: str, model_cls: type[BaseModel]) -> list[Any]:
        raw = json.loads((self.path / name).read_text())
        return [model_cls.model_validate(item) for item in raw]

    def save_blocks(self, blocks: list[Block]) -> None:
        self._save_models(self.BLOCKS, blocks)

    def load_blocks(self) -> list[Block]:
        return self._load_models(self.BLOCKS, Block)

    def save_briefs(self, briefs: list[VisualBrief]) -> None:
        self._save_models(self.BRIEFS, briefs)

    def load_briefs(self) -> list[VisualBrief]:
        return self._load_models(self.BRIEFS, VisualBrief)

    def save_search_results(self, results: dict[int, list[Candidate]]) -> None:
        out = {str(k): [c.model_dump() for c in v] for k, v in results.items()}
        (self.path / self.SEARCH).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def load_search_results(self) -> dict[int, list[Candidate]]:
        raw = json.loads((self.path / self.SEARCH).read_text())
        return {int(k): [Candidate.model_validate(c) for c in v] for k, v in raw.items()}

    def save_selections(self, selections: list[Selection]) -> None:
        self._save_models(self.SELECTIONS, selections)

    def load_selections(self) -> list[Selection]:
        return self._load_models(self.SELECTIONS, Selection)

    def has(self, name: str) -> bool:
        return (self.path / name).exists()
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_runs.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/runs.py tests/test_runs.py
git commit -m "feat(runs): add RunDir helper for stage artifacts"
```

---

## Phase 2 — Parser

### Task 5: DOTTI SYNC parser

**Files:**
- Create: `src/avtv/parser.py`
- Create: `tests/fixtures/sample_script.txt`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Create fixture**

`tests/fixtures/sample_script.txt`:
```
============================================================
SINCRONIZACAO DOTTI SYNC - BLOCOS DE 8 SEGUNDOS
============================================================
Arquivo: sample.mp3
Duracao: 00:32
Total de prompts: 4
============================================================

PROMPT 001 | 00:00 - 00:08
First block of narration.
------------------------------------------------------------

PROMPT 002 | 00:08 - 00:16
Second block, also has some words.
------------------------------------------------------------

PROMPT 003 | 00:16 - 00:24

------------------------------------------------------------

PROMPT 004 | 00:24 - 00:32
Final block here.
------------------------------------------------------------
```

- [ ] **Step 2: Write failing test**

`tests/test_parser.py`:
```python
from pathlib import Path

import pytest

from avtv.parser import ParseError, parse_script

FIXTURE = Path(__file__).parent / "fixtures" / "sample_script.txt"


def test_parse_sample_returns_four_blocks():
    blocks = parse_script(FIXTURE.read_text())
    assert len(blocks) == 4


def test_parse_first_block_fields():
    blocks = parse_script(FIXTURE.read_text())
    b = blocks[0]
    assert b.idx == 1
    assert b.start == 0.0
    assert b.end == 8.0
    assert b.text == "First block of narration."
    assert b.is_empty is False


def test_parse_empty_block_detected():
    blocks = parse_script(FIXTURE.read_text())
    assert blocks[2].is_empty is True
    assert blocks[2].idx == 3


def test_parse_rejects_non_sequential_idx():
    bad = """\
PROMPT 001 | 00:00 - 00:08
hi
------------------------------------------------------------

PROMPT 003 | 00:08 - 00:16
gap
------------------------------------------------------------
"""
    with pytest.raises(ParseError, match="non-sequential"):
        parse_script(bad)


def test_parse_rejects_wrong_duration():
    bad = """\
PROMPT 001 | 00:00 - 00:09
hi
------------------------------------------------------------
"""
    with pytest.raises(ParseError, match="duration"):
        parse_script(bad)


def test_parse_handles_multiline_body():
    multi = """\
PROMPT 001 | 00:00 - 00:08
Line one.
Line two.
------------------------------------------------------------
"""
    blocks = parse_script(multi)
    assert blocks[0].text == "Line one.\nLine two."
```

- [ ] **Step 3: Run, expect fail**

```bash
poetry run pytest tests/test_parser.py -v
```
Expected: FAIL.

- [ ] **Step 4: Implement parser**

`src/avtv/parser.py`:
```python
import re

from avtv.models import Block

BLOCK_DURATION = 8.0
SEPARATOR = "-" * 60

_HEADER_RE = re.compile(
    r"^PROMPT\s+(\d+)\s*\|\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*$"
)


class ParseError(ValueError):
    pass


def _to_seconds(mm: str, ss: str) -> float:
    return int(mm) * 60 + int(ss)


def parse_script(text: str) -> list[Block]:
    """Parse DOTTI SYNC text into a list of Block."""
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _HEADER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        idx = int(m.group(1))
        start = _to_seconds(m.group(2), m.group(3))
        end = _to_seconds(m.group(4), m.group(5))
        if abs((end - start) - BLOCK_DURATION) > 0.01:
            raise ParseError(
                f"block {idx}: duration {end - start}s != {BLOCK_DURATION}s"
            )
        # body until separator
        body_lines: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != SEPARATOR:
            body_lines.append(lines[i])
            i += 1
        # consume separator
        if i < len(lines):
            i += 1
        text_body = "\n".join(body_lines).strip("\n")
        blocks.append(Block.from_text(idx=idx, start=start, end=end, text=text_body))

    if not blocks:
        raise ParseError("no blocks found")

    for expected, b in enumerate(blocks, start=1):
        if b.idx != expected:
            raise ParseError(
                f"non-sequential idx: expected {expected}, got {b.idx}"
            )

    return blocks
```

- [ ] **Step 5: Run, expect pass**

```bash
poetry run pytest tests/test_parser.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/avtv/parser.py tests/test_parser.py tests/fixtures/sample_script.txt
git commit -m "feat(parser): parse DOTTI SYNC script into Block list"
```

---

## Phase 3 — Briefing

### Task 6: LLMProvider Protocol + prompt builder

**Files:**
- Create: `src/avtv/briefing/__init__.py`
- Create: `src/avtv/briefing/llm.py`
- Create: `src/avtv/briefing/prompt.py`
- Create: `tests/test_briefing_prompt.py`

- [ ] **Step 1: Write failing test**

`tests/test_briefing_prompt.py`:
```python
from avtv.briefing.prompt import build_user_message, SYSTEM_PROMPT
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
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_briefing_prompt.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement Protocol and prompt**

`src/avtv/briefing/llm.py`:
```python
from typing import Protocol

from avtv.models import Block, VisualBrief


class LLMProvider(Protocol):
    def generate_briefs(self, blocks: list[Block]) -> list[VisualBrief]: ...
```

`src/avtv/briefing/prompt.py`:
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


def build_user_message(blocks: list[Block]) -> str:
    lines: list[str] = ["Here is the full script:\n"]
    for b in blocks:
        body = b.text if not b.is_empty else "[empty]"
        lines.append(f"[BLOCK {b.idx}] ({b.start:.0f}-{b.end:.0f}s)\n{body}\n")
    return "\n".join(lines)
```

`src/avtv/briefing/__init__.py`:
```python
from avtv.briefing.llm import LLMProvider

__all__ = ["LLMProvider", "get_provider"]


def get_provider(name: str) -> LLMProvider:
    if name == "claude":
        from avtv.briefing.claude import ClaudeProvider
        return ClaudeProvider()
    if name == "gpt":
        from avtv.briefing.gpt import OpenAIProvider
        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name}")
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_briefing_prompt.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/avtv/briefing/ tests/test_briefing_prompt.py
git commit -m "feat(briefing): add LLMProvider Protocol and prompt builders"
```

---

### Task 7: Brief schema for tool/structured output

**Files:**
- Create: `src/avtv/briefing/schema.py`
- Create: `tests/test_briefing_schema.py`

- [ ] **Step 1: Write failing test**

`tests/test_briefing_schema.py`:
```python
from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.models import VisualBrief


def test_schema_has_required_top_level():
    schema = BRIEF_TOOL_SCHEMA
    assert schema["type"] == "object"
    assert "briefs" in schema["properties"]
    assert schema["required"] == ["briefs"]


def test_parse_briefs_response_valid():
    data = {
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
    briefs = parse_briefs_response(data)
    assert len(briefs) == 1
    assert isinstance(briefs[0], VisualBrief)
    assert briefs[0].query_en == "forest sunlight"
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_briefing_schema.py -v
```

- [ ] **Step 3: Implement schema**

`src/avtv/briefing/schema.py`:
```python
from typing import Any

from avtv.models import VisualBrief

BRIEF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "briefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "query_en": {"type": "string"},
                    "fallback_query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["video", "image", "continuation"],
                    },
                    "continuity_hint": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": [
                    "idx", "query_en", "fallback_query", "kind",
                    "continuity_hint", "notes",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["briefs"],
    "additionalProperties": False,
}


def parse_briefs_response(data: dict[str, Any]) -> list[VisualBrief]:
    return [VisualBrief.model_validate(item) for item in data["briefs"]]
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_briefing_schema.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/briefing/schema.py tests/test_briefing_schema.py
git commit -m "feat(briefing): add JSON schema for VisualBrief tool/structured output"
```

---

### Task 8: ClaudeProvider

**Files:**
- Create: `src/avtv/briefing/claude.py`
- Create: `tests/test_briefing_claude.py`

- [ ] **Step 1: Write failing test**

`tests/test_briefing_claude.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from avtv.briefing.claude import ClaudeProvider
from avtv.models import Block


@pytest.fixture
def mock_anthropic_response():
    """Build a fake anthropic Message with a tool_use block."""
    msg = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "briefs": [
            {
                "idx": 1,
                "query_en": "forest sunlight",
                "fallback_query": "nature",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            },
            {
                "idx": 2,
                "query_en": "wild bison",
                "fallback_query": "large mammal",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            },
        ]
    }
    msg.content = [tool_block]
    msg.stop_reason = "tool_use"
    return msg


def test_claude_provider_returns_briefs(mock_anthropic_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [
        Block.from_text(1, 0.0, 8.0, "forest words"),
        Block.from_text(2, 8.0, 16.0, "bison words"),
    ]

    with patch("avtv.briefing.claude.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_anthropic_response

        provider = ClaudeProvider()
        briefs = provider.generate_briefs(blocks)

    assert len(briefs) == 2
    assert briefs[0].query_en == "forest sunlight"
    assert briefs[1].query_en == "wild bison"


def test_claude_provider_raises_on_count_mismatch(mock_anthropic_response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")

    blocks = [Block.from_text(i, (i - 1) * 8.0, i * 8.0, "x") for i in range(1, 5)]

    with patch("avtv.briefing.claude.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = mock_anthropic_response  # only 2 briefs
        provider = ClaudeProvider()
        with pytest.raises(ValueError, match="count mismatch"):
            provider.generate_briefs(blocks)
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_briefing_claude.py -v
```

- [ ] **Step 3: Implement ClaudeProvider**

`src/avtv/briefing/claude.py`:
```python
from anthropic import Anthropic

from avtv.briefing.prompt import SYSTEM_PROMPT, build_user_message
from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.config import Settings
from avtv.models import Block, VisualBrief

TOOL_NAME = "emit_visual_briefs"


class ClaudeProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
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

        tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_blocks:
            raise ValueError("Claude did not return a tool_use block")

        briefs = parse_briefs_response(tool_blocks[0].input)

        if len(briefs) != len(blocks):
            raise ValueError(
                f"count mismatch: got {len(briefs)} briefs for {len(blocks)} blocks"
            )

        return briefs
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_briefing_claude.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/briefing/claude.py tests/test_briefing_claude.py
git commit -m "feat(briefing): add ClaudeProvider with tool_use + prompt caching"
```

---

### Task 9: OpenAIProvider

**Files:**
- Create: `src/avtv/briefing/gpt.py`
- Create: `tests/test_briefing_gpt.py`

- [ ] **Step 1: Write failing test**

`tests/test_briefing_gpt.py`:
```python
import json
from unittest.mock import MagicMock, patch

import pytest

from avtv.briefing.gpt import OpenAIProvider
from avtv.models import Block


def _set_env(monkeypatch):
    for key in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PEXELS_API_KEY",
        "PIXABAY_API_KEY", "UNSPLASH_API_KEY",
    ]:
        monkeypatch.setenv(key, "x")


@pytest.fixture
def mock_openai_response():
    payload = {
        "briefs": [
            {
                "idx": 1, "query_en": "forest sunlight",
                "fallback_query": "nature", "kind": "video",
                "continuity_hint": None, "notes": None,
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
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_briefing_gpt.py -v
```

- [ ] **Step 3: Implement OpenAIProvider**

`src/avtv/briefing/gpt.py`:
```python
import json

from openai import OpenAI

from avtv.briefing.prompt import SYSTEM_PROMPT, build_user_message
from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.config import Settings
from avtv.models import Block, VisualBrief


class OpenAIProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
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
            raise ValueError(
                f"count mismatch: got {len(briefs)} briefs for {len(blocks)} blocks"
            )

        return briefs
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_briefing_gpt.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/briefing/gpt.py tests/test_briefing_gpt.py
git commit -m "feat(briefing): add OpenAIProvider with structured outputs"
```

---

## Phase 4 — Search

### Task 10: SearchAdapter Protocol + base

**Files:**
- Create: `src/avtv/search/__init__.py`
- Create: `src/avtv/search/base.py`
- Create: `tests/test_search_base.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_base.py`:
```python
import pytest

from avtv.search.base import SearchAdapter, dedupe_candidates
from avtv.models import Candidate


def _make(url: str, w: int = 1920, h: int = 1080) -> Candidate:
    return Candidate(
        source="test", url=url, kind="video", duration=8.0,
        width=w, height=h, license="x",
    )


def test_dedupe_keeps_first_occurrence_per_url():
    cs = [_make("a"), _make("b"), _make("a"), _make("c")]
    out = dedupe_candidates(cs)
    assert [c.url for c in out] == ["a", "b", "c"]
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_search_base.py -v
```

- [ ] **Step 3: Implement Protocol**

`src/avtv/search/base.py`:
```python
from typing import Literal, Protocol

from avtv.models import Candidate

MediaKind = Literal["video", "image"]


class SearchAdapter(Protocol):
    name: str
    media_kinds: set[MediaKind]

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]: ...


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Dedupe by URL preserving first occurrence."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        if c.url in seen:
            continue
        seen.add(c.url)
        out.append(c)
    return out
```

`src/avtv/search/__init__.py`: (empty for now)

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_search_base.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/ tests/test_search_base.py
git commit -m "feat(search): add SearchAdapter Protocol and dedupe helper"
```

---

### Task 11: Pexels adapter

**Files:**
- Create: `src/avtv/search/pexels.py`
- Create: `tests/fixtures/pexels_response.json`
- Create: `tests/test_search_pexels.py`

- [ ] **Step 1: Create fixture**

`tests/fixtures/pexels_response.json`:
```json
{
  "page": 1, "per_page": 2, "total_results": 100,
  "videos": [
    {
      "id": 1, "width": 1920, "height": 1080, "duration": 12,
      "url": "https://pexels.com/video/1",
      "image": "https://pexels.com/poster.jpg",
      "user": {"name": "Alice"},
      "video_files": [
        {"id": 100, "quality": "hd", "width": 1920, "height": 1080,
         "link": "https://pexels.com/v1.mp4"}
      ]
    },
    {
      "id": 2, "width": 1280, "height": 720, "duration": 5,
      "url": "https://pexels.com/video/2",
      "image": "https://pexels.com/p2.jpg",
      "user": {"name": "Bob"},
      "video_files": [
        {"id": 200, "quality": "sd", "width": 1280, "height": 720,
         "link": "https://pexels.com/v2.mp4"}
      ]
    }
  ]
}
```

- [ ] **Step 2: Write failing test**

`tests/test_search_pexels.py`:
```python
import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from avtv.search.pexels import PexelsAdapter

FIX = json.loads((Path(__file__).parent / "fixtures" / "pexels_response.json").read_text())


@pytest.mark.asyncio
@respx.mock
async def test_pexels_search_videos():
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=Response(200, json=FIX)
    )
    adapter = PexelsAdapter(api_key="testkey")
    results = await adapter.search("forest", limit=2, kind="video")
    assert len(results) == 2
    assert results[0].source == "pexels"
    assert results[0].url == "https://pexels.com/v1.mp4"
    assert results[0].duration == 12
    assert results[0].width == 1920
    assert results[0].attribution == "Alice"
```

- [ ] **Step 3: Run, expect fail**

```bash
poetry run pytest tests/test_search_pexels.py -v
```

- [ ] **Step 4: Implement PexelsAdapter**

`src/avtv/search/pexels.py`:
```python
import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class PexelsAdapter:
    name = "pexels"
    media_kinds = {"video"}
    BASE_URL = "https://api.pexels.com"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        if kind != "video":
            return []
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                f"{self.BASE_URL}/videos/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": limit},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for v in data.get("videos", []):
            files = v.get("video_files", [])
            if not files:
                continue
            # Pick highest resolution file
            best = max(files, key=lambda f: f.get("width", 0) * f.get("height", 0))
            out.append(
                Candidate(
                    source=self.name,
                    url=best["link"],
                    preview_url=v.get("image"),
                    kind="video",
                    duration=float(v.get("duration", 0)),
                    width=int(best.get("width", v.get("width", 0))),
                    height=int(best.get("height", v.get("height", 0))),
                    license="Pexels License",
                    attribution=(v.get("user") or {}).get("name"),
                )
            )
        return out
```

- [ ] **Step 5: Run, expect pass**

```bash
poetry run pytest tests/test_search_pexels.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/avtv/search/pexels.py tests/test_search_pexels.py tests/fixtures/pexels_response.json
git commit -m "feat(search): add Pexels video adapter"
```

---

### Task 12: Pixabay adapter

**Files:**
- Create: `src/avtv/search/pixabay.py`
- Create: `tests/fixtures/pixabay_video_response.json`
- Create: `tests/fixtures/pixabay_image_response.json`
- Create: `tests/test_search_pixabay.py`

- [ ] **Step 1: Create fixtures**

`tests/fixtures/pixabay_video_response.json`:
```json
{
  "total": 100, "totalHits": 100,
  "hits": [
    {
      "id": 1, "duration": 10, "user": "Alice",
      "videos": {
        "large": {"url": "https://pixabay.com/v1-large.mp4", "width": 1920, "height": 1080},
        "medium": {"url": "https://pixabay.com/v1-med.mp4", "width": 1280, "height": 720}
      }
    }
  ]
}
```

`tests/fixtures/pixabay_image_response.json`:
```json
{
  "total": 50, "totalHits": 50,
  "hits": [
    {
      "id": 10,
      "largeImageURL": "https://pixabay.com/img-large.jpg",
      "imageWidth": 4000, "imageHeight": 3000,
      "user": "Carol"
    }
  ]
}
```

- [ ] **Step 2: Write failing test**

`tests/test_search_pixabay.py`:
```python
import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from avtv.search.pixabay import PixabayAdapter

VIDEO_FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "pixabay_video_response.json").read_text()
)
IMAGE_FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "pixabay_image_response.json").read_text()
)


@pytest.mark.asyncio
@respx.mock
async def test_pixabay_video_search():
    respx.get("https://pixabay.com/api/videos/").mock(
        return_value=Response(200, json=VIDEO_FIX)
    )
    a = PixabayAdapter(api_key="k")
    results = await a.search("nature", kind="video")
    assert len(results) == 1
    assert results[0].source == "pixabay"
    assert results[0].url == "https://pixabay.com/v1-large.mp4"
    assert results[0].duration == 10
    assert results[0].kind == "video"


@pytest.mark.asyncio
@respx.mock
async def test_pixabay_image_search():
    respx.get("https://pixabay.com/api/").mock(
        return_value=Response(200, json=IMAGE_FIX)
    )
    a = PixabayAdapter(api_key="k")
    results = await a.search("forest", kind="image")
    assert len(results) == 1
    assert results[0].kind == "image"
    assert results[0].url == "https://pixabay.com/img-large.jpg"
    assert results[0].width == 4000
    assert results[0].duration is None
```

- [ ] **Step 3: Run, expect fail**

```bash
poetry run pytest tests/test_search_pixabay.py -v
```

- [ ] **Step 4: Implement PixabayAdapter**

`src/avtv/search/pixabay.py`:
```python
import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class PixabayAdapter:
    name = "pixabay"
    media_kinds = {"video", "image"}
    VIDEO_URL = "https://pixabay.com/api/videos/"
    IMAGE_URL = "https://pixabay.com/api/"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        if kind == "video":
            return await self._video(query, limit)
        return await self._image(query, limit)

    async def _video(self, query: str, limit: int) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.VIDEO_URL,
                params={"key": self.api_key, "q": query, "per_page": max(limit, 3)},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("hits", [])[:limit]:
            videos = h.get("videos", {})
            # Prefer "large", fall back to "medium"
            best = videos.get("large") or videos.get("medium")
            if not best:
                continue
            out.append(
                Candidate(
                    source=self.name,
                    url=best["url"],
                    preview_url=None,
                    kind="video",
                    duration=float(h.get("duration", 0)),
                    width=int(best.get("width", 0)),
                    height=int(best.get("height", 0)),
                    license="Pixabay License",
                    attribution=h.get("user"),
                )
            )
        return out

    async def _image(self, query: str, limit: int) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.IMAGE_URL,
                params={
                    "key": self.api_key, "q": query,
                    "per_page": max(limit, 3),
                    "image_type": "photo",
                },
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("hits", [])[:limit]:
            out.append(
                Candidate(
                    source=self.name,
                    url=h["largeImageURL"],
                    preview_url=None,
                    kind="image",
                    duration=None,
                    width=int(h.get("imageWidth", 0)),
                    height=int(h.get("imageHeight", 0)),
                    license="Pixabay License",
                    attribution=h.get("user"),
                )
            )
        return out
```

- [ ] **Step 5: Run, expect pass**

```bash
poetry run pytest tests/test_search_pixabay.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/avtv/search/pixabay.py tests/test_search_pixabay.py tests/fixtures/pixabay_*
git commit -m "feat(search): add Pixabay video/image adapter"
```

---

### Task 13: Internet Archive adapter

**Files:**
- Create: `src/avtv/search/archive_org.py`
- Create: `tests/test_search_archive.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_archive.py`:
```python
import json
from unittest.mock import MagicMock, patch

import pytest
import respx
from httpx import Response

from avtv.search.archive_org import ArchiveOrgAdapter

SEARCH_RESPONSE = {
    "response": {
        "docs": [
            {"identifier": "test_item_1", "title": "Test Documentary", "creator": "Public"}
        ]
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_archive_search_returns_video():
    respx.get("https://archive.org/advancedsearch.php").mock(
        return_value=Response(200, json=SEARCH_RESPONSE)
    )

    # Mock metadata call
    metadata = {
        "files": [
            {"name": "test.mp4", "format": "h.264", "length": "30.5",
             "width": "1920", "height": "1080"},
            {"name": "test.gif", "format": "Animated GIF"},
        ]
    }
    respx.get("https://archive.org/metadata/test_item_1").mock(
        return_value=Response(200, json=metadata)
    )

    a = ArchiveOrgAdapter()
    results = await a.search("history", kind="video", limit=1)
    assert len(results) == 1
    assert results[0].source == "archive_org"
    assert "test_item_1" in results[0].url
    assert results[0].duration == 30.5
    assert results[0].width == 1920
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_search_archive.py -v
```

- [ ] **Step 3: Implement ArchiveOrgAdapter**

`src/avtv/search/archive_org.py`:
```python
import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind

VIDEO_FORMATS = {"h.264", "mp4", "mpeg4", "h.264 hd"}
IMAGE_FORMATS = {"jpeg", "jpg", "png"}


class ArchiveOrgAdapter:
    name = "archive_org"
    media_kinds = {"video", "image"}
    SEARCH_URL = "https://archive.org/advancedsearch.php"
    METADATA_URL = "https://archive.org/metadata"

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "video"
    ) -> list[Candidate]:
        mediatype = "movies" if kind == "video" else "image"
        params = {
            "q": f"{query} AND mediatype:{mediatype}",
            "fl[]": ["identifier", "title", "creator"],
            "rows": limit,
            "output": "json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(self.SEARCH_URL, params=params)
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])

            out: list[Candidate] = []
            for d in docs:
                ident = d["identifier"]
                meta_r = await client.get(f"{self.METADATA_URL}/{ident}")
                if meta_r.status_code != 200:
                    continue
                files = meta_r.json().get("files", [])
                pick = self._pick_file(files, kind)
                if not pick:
                    continue
                file_url = f"https://archive.org/download/{ident}/{pick['name']}"
                out.append(
                    Candidate(
                        source=self.name,
                        url=file_url,
                        preview_url=None,
                        kind=kind,
                        duration=float(pick.get("length", 0)) if kind == "video" else None,
                        width=int(pick.get("width", 0) or 0),
                        height=int(pick.get("height", 0) or 0),
                        license="Public Domain / various",
                        attribution=d.get("creator"),
                    )
                )
            return out

    @staticmethod
    def _pick_file(files: list[dict], kind: MediaKind) -> dict | None:
        target = VIDEO_FORMATS if kind == "video" else IMAGE_FORMATS
        for f in files:
            fmt = (f.get("format") or "").lower()
            if any(t in fmt for t in target):
                return f
        return None
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_search_archive.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/archive_org.py tests/test_search_archive.py
git commit -m "feat(search): add Internet Archive adapter for video/image"
```

---

### Task 14: Wikimedia Commons adapter

**Files:**
- Create: `src/avtv/search/wikimedia.py`
- Create: `tests/test_search_wikimedia.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_wikimedia.py`:
```python
import pytest
import respx
from httpx import Response

from avtv.search.wikimedia import WikimediaAdapter

SEARCH_RESPONSE = {
    "query": {
        "search": [
            {"title": "File:Bison.jpg"},
        ]
    }
}

IMAGEINFO_RESPONSE = {
    "query": {
        "pages": {
            "1": {
                "title": "File:Bison.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Bison.jpg",
                        "width": 4000,
                        "height": 3000,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "Wiki User"},
                        },
                    }
                ],
            }
        }
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_wikimedia_image_search():
    respx.get("https://commons.wikimedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=SEARCH_RESPONSE),
            Response(200, json=IMAGEINFO_RESPONSE),
        ]
    )
    a = WikimediaAdapter()
    results = await a.search("bison", kind="image", limit=1)
    assert len(results) == 1
    assert results[0].source == "wikimedia"
    assert results[0].url == "https://upload.wikimedia.org/Bison.jpg"
    assert results[0].width == 4000
    assert "CC BY-SA" in results[0].license
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_search_wikimedia.py -v
```

- [ ] **Step 3: Implement WikimediaAdapter**

`src/avtv/search/wikimedia.py`:
```python
import re

import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class WikimediaAdapter:
    name = "wikimedia"
    media_kinds = {"video", "image"}
    API = "https://commons.wikimedia.org/w/api.php"

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "image"
    ) -> list[Candidate]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            search = await client.get(
                self.API,
                params={
                    "action": "query", "format": "json",
                    "list": "search", "srnamespace": "6",
                    "srsearch": f"{query} filetype:{'video' if kind == 'video' else 'bitmap'}",
                    "srlimit": limit,
                },
            )
            search.raise_for_status()
            titles = [s["title"] for s in search.json().get("query", {}).get("search", [])]
            if not titles:
                return []

            info = await client.get(
                self.API,
                params={
                    "action": "query", "format": "json",
                    "titles": "|".join(titles),
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata",
                },
            )
            info.raise_for_status()
            pages = info.json().get("query", {}).get("pages", {})

        out: list[Candidate] = []
        for page in pages.values():
            iis = page.get("imageinfo", [])
            if not iis:
                continue
            ii = iis[0]
            ext = ii.get("extmetadata", {}) or {}
            license_str = (ext.get("LicenseShortName") or {}).get("value", "Unknown")
            artist_html = (ext.get("Artist") or {}).get("value", "") or ""
            artist = re.sub(r"<[^>]+>", "", artist_html).strip() or None
            out.append(
                Candidate(
                    source=self.name,
                    url=ii["url"],
                    preview_url=None,
                    kind=kind,
                    duration=None,
                    width=int(ii.get("width", 0)),
                    height=int(ii.get("height", 0)),
                    license=license_str,
                    attribution=artist,
                )
            )
        return out
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_search_wikimedia.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/wikimedia.py tests/test_search_wikimedia.py
git commit -m "feat(search): add Wikimedia Commons adapter"
```

---

### Task 15: Unsplash adapter

**Files:**
- Create: `src/avtv/search/unsplash.py`
- Create: `tests/test_search_unsplash.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_unsplash.py`:
```python
import pytest
import respx
from httpx import Response

from avtv.search.unsplash import UnsplashAdapter

RESP = {
    "results": [
        {
            "id": "abc",
            "width": 4000, "height": 3000,
            "urls": {"full": "https://img.unsplash.com/abc"},
            "user": {"name": "Eve"},
        }
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_unsplash_image_only():
    respx.get("https://api.unsplash.com/search/photos").mock(
        return_value=Response(200, json=RESP)
    )
    a = UnsplashAdapter(api_key="k")
    results = await a.search("forest", kind="image")
    assert len(results) == 1
    assert results[0].source == "unsplash"
    assert results[0].url == "https://img.unsplash.com/abc"
    assert results[0].kind == "image"


@pytest.mark.asyncio
async def test_unsplash_rejects_video():
    a = UnsplashAdapter(api_key="k")
    results = await a.search("forest", kind="video")
    assert results == []
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_search_unsplash.py -v
```

- [ ] **Step 3: Implement UnsplashAdapter**

`src/avtv/search/unsplash.py`:
```python
import httpx

from avtv.models import Candidate
from avtv.search.base import MediaKind


class UnsplashAdapter:
    name = "unsplash"
    media_kinds = {"image"}
    URL = "https://api.unsplash.com/search/photos"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self, query: str, limit: int = 10, kind: MediaKind = "image"
    ) -> list[Candidate]:
        if kind != "image":
            return []
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                self.URL,
                headers={"Authorization": f"Client-ID {self.api_key}"},
                params={"query": query, "per_page": limit, "orientation": "landscape"},
            )
            r.raise_for_status()
            data = r.json()

        out: list[Candidate] = []
        for h in data.get("results", []):
            out.append(
                Candidate(
                    source=self.name,
                    url=h["urls"]["full"],
                    preview_url=None,
                    kind="image",
                    duration=None,
                    width=int(h.get("width", 0)),
                    height=int(h.get("height", 0)),
                    license="Unsplash License",
                    attribution=(h.get("user") or {}).get("name"),
                )
            )
        return out
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_search_unsplash.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/unsplash.py tests/test_search_unsplash.py
git commit -m "feat(search): add Unsplash image adapter"
```

---

### Task 16: Search orchestrator

**Files:**
- Create: `src/avtv/search/orchestrator.py`
- Create: `tests/test_search_orchestrator.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_orchestrator.py`:
```python
import asyncio

import pytest

from avtv.models import Candidate, VisualBrief
from avtv.search.base import SearchAdapter
from avtv.search.orchestrator import SearchOrchestrator


class FakeAdapter:
    def __init__(self, name: str, media_kinds: set, results: dict):
        self.name = name
        self.media_kinds = media_kinds
        self._results = results  # {(query, kind): [Candidate, ...]}
        self.calls: list = []

    async def search(self, query, limit=10, kind="video"):
        self.calls.append((query, kind))
        return self._results.get((query, kind), [])


def _c(source, url):
    return Candidate(
        source=source, url=url, kind="video", duration=8.0,
        width=1920, height=1080, license="x",
    )


@pytest.mark.asyncio
async def test_orchestrator_combines_video_adapters():
    a1 = FakeAdapter("a", {"video"}, {("forest", "video"): [_c("a", "u1")]})
    a2 = FakeAdapter("b", {"video"}, {("forest", "video"): [_c("b", "u2")]})
    orch = SearchOrchestrator(adapters=[a1, a2])

    brief = VisualBrief(
        idx=1, query_en="forest", fallback_query="nature", kind="video",
    )
    results = await orch.search_for_brief(brief)
    urls = {c.url for c in results}
    assert urls == {"u1", "u2"}


@pytest.mark.asyncio
async def test_orchestrator_uses_fallback_when_primary_empty():
    a = FakeAdapter("a", {"video"}, {("nature", "video"): [_c("a", "u-nature")]})
    orch = SearchOrchestrator(adapters=[a])

    brief = VisualBrief(
        idx=1, query_en="forest", fallback_query="nature", kind="video",
    )
    results = await orch.search_for_brief(brief)
    assert len(results) == 1
    assert results[0].url == "u-nature"
    # Both queries were tried
    assert a.calls == [("forest", "video"), ("nature", "video")]


@pytest.mark.asyncio
async def test_orchestrator_skips_continuation():
    a = FakeAdapter("a", {"video"}, {})
    orch = SearchOrchestrator(adapters=[a])
    brief = VisualBrief(
        idx=1, query_en="x", fallback_query="y", kind="continuation",
    )
    results = await orch.search_for_brief(brief)
    assert results == []
    assert a.calls == []
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_search_orchestrator.py -v
```

- [ ] **Step 3: Implement orchestrator**

`src/avtv/search/orchestrator.py`:
```python
import asyncio

from avtv.models import Candidate, VisualBrief
from avtv.search.base import MediaKind, SearchAdapter, dedupe_candidates


class SearchOrchestrator:
    def __init__(self, adapters: list[SearchAdapter], top_k: int = 8) -> None:
        self.adapters = adapters
        self.top_k = top_k

    def _adapters_for(self, kind: MediaKind) -> list[SearchAdapter]:
        return [a for a in self.adapters if kind in a.media_kinds]

    async def _search_all(
        self, query: str, kind: MediaKind
    ) -> list[Candidate]:
        relevant = self._adapters_for(kind)
        coros = [a.search(query, limit=self.top_k, kind=kind) for a in relevant]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: list[Candidate] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            out.extend(r)
        return out

    async def search_for_brief(self, brief: VisualBrief) -> list[Candidate]:
        if brief.kind == "continuation":
            return []

        media_kind: MediaKind = "video" if brief.kind == "video" else "image"

        primary = await self._search_all(brief.query_en, media_kind)
        if primary:
            return dedupe_candidates(primary)

        fallback = await self._search_all(brief.fallback_query, media_kind)
        return dedupe_candidates(fallback)

    async def search_for_briefs(
        self, briefs: list[VisualBrief], concurrency: int = 5
    ) -> dict[int, list[Candidate]]:
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(brief: VisualBrief) -> tuple[int, list[Candidate]]:
            async with sem:
                return brief.idx, await self.search_for_brief(brief)

        results = await asyncio.gather(*[_bounded(b) for b in briefs])
        return dict(results)
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_search_orchestrator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/search/orchestrator.py tests/test_search_orchestrator.py
git commit -m "feat(search): add orchestrator with parallel search and fallback query"
```

---

## Phase 5 — Selector

### Task 17: Score function

**Files:**
- Create: `src/avtv/selector.py`
- Create: `tests/test_selector.py`

- [ ] **Step 1: Write failing test (score only)**

`tests/test_selector.py`:
```python
from avtv.config import Settings
from avtv.models import Candidate
from avtv.selector import score_candidate


def _make(**kw) -> Candidate:
    base = dict(
        source="pexels", url="x", kind="video",
        duration=8.0, width=1920, height=1080, license="x",
    )
    base.update(kw)
    return Candidate(**base)


def _settings(monkeypatch) -> Settings:
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PEXELS_API_KEY",
              "PIXABAY_API_KEY", "UNSPLASH_API_KEY"]:
        monkeypatch.setenv(k, "x")
    return Settings()


def test_score_perfect_clip(monkeypatch):
    s = _settings(monkeypatch)
    c = _make()
    score = score_candidate(c, settings=s)
    # 1.0 (aspect 16:9) + 1.0 (8s ideal) + 1.0 (1080p) + 0.3 (pexels) = 3.3
    assert score > 3.0


def test_score_low_resolution_penalized(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(width=640, height=360)
    score = score_candidate(c, settings=s)
    perfect = score_candidate(_make(), settings=s)
    assert score < perfect


def test_score_short_clip_penalized(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(duration=2.0)
    perfect = score_candidate(_make(), settings=s)
    assert score_candidate(c, settings=s) < perfect


def test_score_image_candidate(monkeypatch):
    s = _settings(monkeypatch)
    c = _make(kind="image", duration=None, width=4000, height=3000)
    # aspect 4:3 = 1.33, falls in lower band
    score = score_candidate(c, settings=s)
    assert score > 0  # still scorable
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_selector.py -v
```

- [ ] **Step 3: Implement score**

`src/avtv/selector.py`:
```python
from avtv.config import Settings
from avtv.models import Candidate

SOURCE_PREFERENCE = {
    "pexels": 0.30,
    "pixabay": 0.25,
    "archive_org": 0.20,
    "wikimedia": 0.15,
    "unsplash": 0.20,
}


def _aspect_score(width: int, height: int) -> float:
    if height <= 0:
        return 0.0
    ratio = width / height
    if ratio >= 1.7:
        return 1.0
    if ratio >= 1.4:
        return 0.5
    return 0.0


def _duration_score(duration: float | None, target: float = 8.0) -> float:
    if duration is None:
        # images get neutral mid score on this axis
        return 0.7
    factor = duration / target
    if 0.85 <= factor <= 1.2:
        return 1.0
    if 0.5 <= factor < 0.85 or 1.2 < factor <= 2.0:
        return 0.7
    if 0.25 <= factor < 0.5 or 2.0 < factor <= 3.0:
        return 0.4
    return 0.1


def _resolution_score(height: int) -> float:
    if height >= 1080:
        return 1.0
    if height >= 720:
        return 0.6
    return 0.3


def score_candidate(c: Candidate, settings: Settings) -> float:
    aspect = _aspect_score(c.width, c.height)
    duration = _duration_score(c.duration, target=settings.target_block_duration)
    resolution = _resolution_score(c.height)
    source_pref = SOURCE_PREFERENCE.get(c.source, 0.1)
    return (
        settings.weight_aspect * aspect
        + settings.weight_duration * duration
        + settings.weight_resolution * resolution
        + settings.weight_source * source_pref
    )
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_selector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/selector.py tests/test_selector.py
git commit -m "feat(selector): add candidate scoring function"
```

---

### Task 18: Selector with neighbor dedup

**Files:**
- Modify: `src/avtv/selector.py`
- Modify: `tests/test_selector.py`

- [ ] **Step 1: Add failing tests for select_per_block**

Append to `tests/test_selector.py`:
```python
from avtv.models import VisualBrief
from avtv.selector import select_per_block, _selection_from


def _brief(idx, kind="video"):
    return VisualBrief(
        idx=idx, query_en="x", fallback_query="y", kind=kind,
    )


def test_select_picks_best_per_block(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {
        1: [_make(url="hi-res", height=1080), _make(url="lo-res", height=480)],
        2: [_make(url="hi-res2", height=1080)],
    }
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[0].url == "hi-res"
    assert sels[1].url == "hi-res2"


def test_select_dedupes_within_neighbor_window(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2), _brief(3)]
    # Best candidate is the same URL for all three blocks; dedup should
    # avoid using it back-to-back in the ±2 window.
    same = _make(url="popular", height=1080)
    other = _make(url="alt", height=1080)
    cands = {
        1: [same, other],
        2: [same, other],
        3: [same, other],
    }
    sels = select_per_block(briefs, cands, settings=s)
    urls = [s.url for s in sels]
    assert urls[0] == "popular"
    assert urls[1] == "alt"   # blocked by neighbor dedup
    assert urls[2] == "alt"   # also blocked


def test_select_continuation_passthrough(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2, kind="continuation")]
    cands = {1: [_make(url="primary", height=1080)], 2: []}
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[1].kind == "continuation"
    assert sels[1].url == ""  # placeholder, assembler resolves


def test_select_empty_candidates_falls_back_to_continuation(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1), _brief(2)]
    cands = {1: [_make(url="ok", height=1080)], 2: []}  # block 2 has nothing
    sels = select_per_block(briefs, cands, settings=s)
    assert sels[1].kind == "continuation"


def test_select_first_block_no_candidates_raises(monkeypatch):
    s = _settings(monkeypatch)
    briefs = [_brief(1)]
    cands = {1: []}
    import pytest
    from avtv.selector import SelectorError
    with pytest.raises(SelectorError):
        select_per_block(briefs, cands, settings=s)
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement select_per_block**

Append to `src/avtv/selector.py`:
```python
from avtv.models import Selection, VisualBrief


class SelectorError(RuntimeError):
    pass


def _selection_from(brief: VisualBrief, c: Candidate) -> Selection:
    speed_factor = None
    trim_start = None
    trim_end = None
    loop = False

    if c.kind == "video" and c.duration is not None:
        target = 8.0
        if c.duration < 4.0:
            loop = True
        elif c.duration > 16.0:
            mid = c.duration / 2
            trim_start = max(0.0, mid - target / 2)
            trim_end = trim_start + target
        else:
            speed_factor = c.duration / target

    return Selection(
        idx=brief.idx,
        source=c.source,
        url=c.url,
        kind=c.kind,
        trim_start=trim_start,
        trim_end=trim_end,
        speed_factor=speed_factor,
        loop=loop,
        attribution=c.attribution,
    )


def _continuation(brief: VisualBrief) -> Selection:
    return Selection(
        idx=brief.idx,
        source="",
        url="",
        kind="continuation",
        attribution=None,
    )


def select_per_block(
    briefs: list[VisualBrief],
    candidates: dict[int, list[Candidate]],
    settings: Settings,
) -> list[Selection]:
    selections: list[Selection] = []
    used_recent: list[str] = []  # URLs in ±neighbor_window range

    for brief in briefs:
        if brief.kind == "continuation":
            selections.append(_continuation(brief))
            used_recent.append("")
            continue

        cands = candidates.get(brief.idx, [])

        if not cands:
            if brief.idx == 1:
                raise SelectorError(
                    f"block {brief.idx} has no candidates and is the first block "
                    f"(cannot continue from previous)"
                )
            selections.append(_continuation(brief))
            used_recent.append("")
            continue

        ranked = sorted(cands, key=lambda c: score_candidate(c, settings), reverse=True)

        window = used_recent[-settings.neighbor_window :]
        chosen = next((c for c in ranked if c.url not in window), ranked[0])

        sel = _selection_from(brief, chosen)
        selections.append(sel)
        used_recent.append(chosen.url)

    return selections
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/selector.py tests/test_selector.py
git commit -m "feat(selector): add per-block selection with neighbor dedup and trim/speed/loop policy"
```

---

## Phase 6 — Downloader

### Task 19: URL-hash cache downloader

**Files:**
- Create: `src/avtv/downloader.py`
- Create: `tests/test_downloader.py`

- [ ] **Step 1: Write failing test**

`tests/test_downloader.py`:
```python
import pytest
import respx
from httpx import Response

from avtv.downloader import Downloader


@pytest.mark.asyncio
@respx.mock
async def test_download_caches_by_url_hash(tmp_path):
    respx.get("https://example.com/clip.mp4").mock(
        return_value=Response(200, content=b"fake-video-bytes")
    )
    d = Downloader(cache_dir=tmp_path)
    path1 = await d.fetch("https://example.com/clip.mp4", kind="video")
    assert path1.exists()
    assert path1.read_bytes() == b"fake-video-bytes"
    assert path1.parent.name == "clips"
    assert path1.suffix == ".mp4"


@pytest.mark.asyncio
@respx.mock
async def test_download_skips_existing(tmp_path):
    route = respx.get("https://example.com/x.mp4").mock(
        return_value=Response(200, content=b"abc")
    )
    d = Downloader(cache_dir=tmp_path)
    await d.fetch("https://example.com/x.mp4", kind="video")
    await d.fetch("https://example.com/x.mp4", kind="video")
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_download_image_kind_uses_images_dir(tmp_path):
    respx.get("https://example.com/photo.jpg").mock(
        return_value=Response(200, content=b"img")
    )
    d = Downloader(cache_dir=tmp_path)
    p = await d.fetch("https://example.com/photo.jpg", kind="image")
    assert p.parent.name == "images"
    assert p.suffix == ".jpg"
```

- [ ] **Step 2: Run, expect fail**

```bash
poetry run pytest tests/test_downloader.py -v
```

- [ ] **Step 3: Implement Downloader**

`src/avtv/downloader.py`:
```python
import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

Kind = Literal["video", "image"]


class Downloader:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path_for(self, url: str, kind: Kind) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        ext = Path(urlparse(url).path).suffix or (".mp4" if kind == "video" else ".jpg")
        sub = "clips" if kind == "video" else "images"
        target = self.cache_dir / sub / f"{digest}{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _do_get(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    async def fetch(self, url: str, kind: Kind) -> Path:
        target = self._path_for(url, kind)
        if target.exists() and target.stat().st_size > 0:
            return target
        data = await self._do_get(url)
        target.write_bytes(data)
        return target
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/downloader.py tests/test_downloader.py
git commit -m "feat(downloader): add URL-hash cached downloader with retries"
```

---

## Phase 7 — Assembler

### Task 20: Speed-match decision (pure logic)

**Files:**
- Create: `src/avtv/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write failing test**

`tests/test_assembler.py`:
```python
import pytest

from avtv.assembler import SegmentPlan, plan_segment
from avtv.models import Selection


def _sel(**kw):
    base = dict(
        idx=1, source="pexels", url="u", kind="video",
        trim_start=None, trim_end=None, speed_factor=None,
        loop=False, attribution=None,
    )
    base.update(kw)
    return Selection(**base)


def test_plan_speed_match_normal():
    sel = _sel(speed_factor=0.9)
    plan = plan_segment(sel, source_duration=7.2)
    assert plan.strategy == "speed"
    assert plan.speed_factor == 0.9
    assert plan.use_clip_audio is True


def test_plan_speed_extreme_drops_audio():
    sel = _sel(speed_factor=2.5)
    plan = plan_segment(sel, source_duration=20.0)
    assert plan.strategy == "speed"
    assert plan.use_clip_audio is False  # outside atempo range


def test_plan_loop_for_short_clip():
    sel = _sel(loop=True)
    plan = plan_segment(sel, source_duration=3.0)
    assert plan.strategy == "loop"
    assert plan.use_clip_audio is False


def test_plan_trim_for_long_clip():
    sel = _sel(trim_start=10.0, trim_end=18.0)
    plan = plan_segment(sel, source_duration=30.0)
    assert plan.strategy == "trim"
    assert plan.trim_start == 10.0
    assert plan.trim_end == 18.0
    assert plan.use_clip_audio is True


def test_plan_image_ken_burns():
    sel = _sel(kind="image", speed_factor=None)
    plan = plan_segment(sel, source_duration=None)
    assert plan.strategy == "ken_burns"
    assert plan.use_clip_audio is False


def test_plan_continuation():
    sel = _sel(kind="continuation", url="")
    plan = plan_segment(sel, source_duration=None)
    assert plan.strategy == "continuation"
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement plan_segment**

`src/avtv/assembler.py`:
```python
from dataclasses import dataclass
from typing import Literal

from avtv.models import Selection

Strategy = Literal["speed", "loop", "trim", "ken_burns", "continuation"]
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0


@dataclass
class SegmentPlan:
    strategy: Strategy
    speed_factor: float | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    use_clip_audio: bool = True


def plan_segment(sel: Selection, source_duration: float | None) -> SegmentPlan:
    if sel.kind == "continuation":
        return SegmentPlan(strategy="continuation", use_clip_audio=False)
    if sel.kind == "image":
        return SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    if sel.loop:
        return SegmentPlan(strategy="loop", use_clip_audio=False)
    if sel.trim_start is not None and sel.trim_end is not None:
        return SegmentPlan(
            strategy="trim",
            trim_start=sel.trim_start,
            trim_end=sel.trim_end,
            use_clip_audio=True,
        )
    if sel.speed_factor is not None:
        in_atempo_range = ATEMPO_MIN <= sel.speed_factor <= ATEMPO_MAX
        return SegmentPlan(
            strategy="speed",
            speed_factor=sel.speed_factor,
            use_clip_audio=in_atempo_range,
        )
    # No transform: clip is exactly target duration
    return SegmentPlan(strategy="speed", speed_factor=1.0, use_clip_audio=True)
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py tests/test_assembler.py
git commit -m "feat(assembler): add SegmentPlan decision logic"
```

---

### Task 21: ffmpeg command builders for each strategy

**Files:**
- Modify: `src/avtv/assembler.py`
- Modify: `tests/test_assembler.py`

- [ ] **Step 1: Add failing tests for command builders**

Append to `tests/test_assembler.py`:
```python
from avtv.assembler import build_segment_args, BLOCK_DURATION


def test_build_args_speed_with_audio():
    plan = SegmentPlan(strategy="speed", speed_factor=0.9, use_clip_audio=True)
    args = build_segment_args(
        input_path="/in/clip.mp4", output_path="/out/seg.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-i /in/clip.mp4" in cmd
    assert "/out/seg.ts" in cmd
    # setpts factor for slowing down to fit 8s: 1/0.9
    assert "setpts=" in cmd
    assert "atempo=0.9" in cmd
    assert "volume=-20.0dB" in cmd
    assert "scale=" in cmd
    assert "1920" in cmd and "1080" in cmd


def test_build_args_speed_no_audio():
    plan = SegmentPlan(strategy="speed", speed_factor=3.0, use_clip_audio=False)
    args = build_segment_args(
        input_path="/in.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-an" in cmd  # audio dropped


def test_build_args_trim():
    plan = SegmentPlan(
        strategy="trim", trim_start=10.0, trim_end=18.0, use_clip_audio=True,
    )
    args = build_segment_args(
        input_path="/i.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-ss 10.0" in cmd
    assert "-t 8.0" in cmd


def test_build_args_loop_short_clip():
    plan = SegmentPlan(strategy="loop", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-stream_loop" in cmd
    assert "-t 8.0" in cmd


def test_build_args_ken_burns_image():
    plan = SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.jpg", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "zoompan" in cmd
    assert "-t 8.0" in cmd
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement command builders**

Append to `src/avtv/assembler.py`:
```python
BLOCK_DURATION = 8.0


def build_segment_args(
    input_path: str,
    output_path: str,
    plan: SegmentPlan,
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> list[str]:
    """Construct ffmpeg argv (without leading 'ffmpeg') for one segment."""
    if plan.strategy == "loop":
        return _loop_args(input_path, output_path, target_w, target_h, target_fps)
    if plan.strategy == "trim":
        return _trim_args(
            input_path, output_path, plan, target_w, target_h, target_fps,
            clip_audio_db,
        )
    if plan.strategy == "speed":
        return _speed_args(
            input_path, output_path, plan, target_w, target_h, target_fps,
            clip_audio_db,
        )
    if plan.strategy == "ken_burns":
        return _ken_burns_args(
            input_path, output_path, target_w, target_h, target_fps,
        )
    raise ValueError(f"build_segment_args: unsupported strategy {plan.strategy}")


def _scale_pad_filter(target_w: int, target_h: int) -> str:
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _speed_args(
    input_path: str, output_path: str, plan: SegmentPlan,
    target_w: int, target_h: int, target_fps: int, clip_audio_db: float,
) -> list[str]:
    factor = plan.speed_factor or 1.0
    setpts_factor = 1.0 / factor  # ffmpeg uses inverse for setpts
    vfilter = (
        f"setpts={setpts_factor:.4f}*PTS,{_scale_pad_filter(target_w, target_h)},"
        f"fps={target_fps}"
    )
    args = [
        "-y", "-i", input_path,
        "-vf", vfilter,
        "-t", f"{BLOCK_DURATION}",
        "-c:v", "libx264", "-preset", "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af", f"atempo={factor:.4f},volume={clip_audio_db}dB",
            "-c:a", "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _trim_args(
    input_path: str, output_path: str, plan: SegmentPlan,
    target_w: int, target_h: int, target_fps: int, clip_audio_db: float,
) -> list[str]:
    args = [
        "-y",
        "-ss", f"{plan.trim_start}",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af", f"volume={clip_audio_db}dB",
            "-c:a", "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _loop_args(
    input_path: str, output_path: str,
    target_w: int, target_h: int, target_fps: int,
) -> list[str]:
    return [
        "-y",
        "-stream_loop", "-1",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        "-f", "mpegts", output_path,
    ]


def _ken_burns_args(
    input_path: str, output_path: str,
    target_w: int, target_h: int, target_fps: int,
) -> list[str]:
    total_frames = int(BLOCK_DURATION * target_fps)
    # Linear zoom from 1.0 to 1.15 across total_frames
    zoompan = (
        f"zoompan=z='min(zoom+0.0007,1.15)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={target_w}x{target_h}:fps={target_fps}"
    )
    return [
        "-y",
        "-loop", "1",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", zoompan,
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        "-f", "mpegts", output_path,
    ]
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py tests/test_assembler.py
git commit -m "feat(assembler): add ffmpeg arg builders for speed/trim/loop/ken_burns strategies"
```

---

### Task 22: Concat list + final mux command

**Files:**
- Modify: `src/avtv/assembler.py`
- Modify: `tests/test_assembler.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_assembler.py`:
```python
from avtv.assembler import (
    build_concat_demuxer_file,
    build_final_mux_args,
    validate_continuations,
)


def test_concat_demuxer_file_format():
    out = build_concat_demuxer_file(["/a.ts", "/b.ts"])
    assert "file '/a.ts'" in out
    assert "file '/b.ts'" in out
    assert out.endswith("\n")


def test_final_mux_args_includes_amix_and_codecs():
    args = build_final_mux_args(
        concat_list_path="/c.txt",
        narration_path="/n.mp3",
        output_path="/o.mp4",
    )
    cmd = " ".join(args)
    assert "-f concat" in cmd
    assert "/c.txt" in cmd
    assert "/n.mp3" in cmd
    assert "amix=inputs=2" in cmd
    assert "-c:v copy" in cmd
    assert "/o.mp4" in cmd


def test_validate_continuations_rejects_first_block():
    bad = [_sel(idx=1, kind="continuation", url="")]
    import pytest
    with pytest.raises(ValueError, match="first block"):
        validate_continuations(bad)


def test_validate_continuations_accepts_normal_run():
    ok = [
        _sel(idx=1, url="A"),
        _sel(idx=2, kind="continuation", url=""),
        _sel(idx=3, url="B"),
    ]
    validate_continuations(ok)  # no raise
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement helpers**

Append to `src/avtv/assembler.py`:
```python
def validate_continuations(selections: list[Selection]) -> None:
    """Reject runs where block 1 is a continuation (no prior source to extend)."""
    if not selections:
        return
    if selections[0].kind == "continuation":
        raise ValueError(
            f"block {selections[0].idx}: first block cannot be continuation"
        )


def build_concat_demuxer_file(segment_paths: list[str]) -> str:
    return "\n".join(f"file '{p}'" for p in segment_paths) + "\n"


def build_final_mux_args(
    concat_list_path: str,
    narration_path: str,
    output_path: str,
) -> list[str]:
    return [
        "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-i", narration_path,
        "-filter_complex",
        "[0:a]anull[clip];[clip][1:a]amix=inputs=2:duration=longest[out]",
        "-map", "0:v", "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py tests/test_assembler.py
git commit -m "feat(assembler): add continuation validator, concat list, and final mux command"
```

---

### Task 23: Assembler runner (subprocess invocation)

**Files:**
- Modify: `src/avtv/assembler.py`
- Create: `tests/test_assembler_runner.py`

- [ ] **Step 1: Write failing test**

`tests/test_assembler_runner.py`:
```python
from unittest.mock import MagicMock, patch

from avtv.assembler import run_ffmpeg


def test_run_ffmpeg_invokes_subprocess():
    with patch("avtv.assembler.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        run_ffmpeg(["-i", "x.mp4", "out.ts"])
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "ffmpeg"
        assert "-i" in called_args


def test_run_ffmpeg_raises_on_nonzero():
    import pytest
    from avtv.assembler import FFmpegError

    with patch("avtv.assembler.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="boom")
        with pytest.raises(FFmpegError, match="boom"):
            run_ffmpeg(["-i", "x"])
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement runner**

Append to `src/avtv/assembler.py`:
```python
import subprocess


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args: list[str]) -> None:
    full = ["ffmpeg", *args]
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}\n"
            f"command: {' '.join(full)}"
        )
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py tests/test_assembler_runner.py
git commit -m "feat(assembler): add subprocess runner for ffmpeg invocations"
```

---

### Task 24: Stage orchestrator: assemble_run

**Files:**
- Modify: `src/avtv/assembler.py`
- Create: `tests/test_assemble_run.py`

- [ ] **Step 1: Write failing test (mocking everything heavy)**

`tests/test_assemble_run.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from avtv.assembler import assemble_run
from avtv.models import Selection


def _video_sel(idx, url, **kw):
    base = dict(
        idx=idx, source="pexels", url=url, kind="video",
        speed_factor=1.0, loop=False, attribution=None,
    )
    base.update(kw)
    return Selection(**base)


def test_assemble_run_invokes_ffmpeg_for_each_segment(tmp_path):
    sels = [_video_sel(1, "u1"), _video_sel(2, "u2")]

    # Fake downloader returns a path
    fake_dl = MagicMock()
    fake_dl.fetch_sync = MagicMock(side_effect=lambda url, kind: tmp_path / f"{url}.mp4")

    # Probe always returns 8s
    with patch("avtv.assembler.run_ffmpeg") as mock_run, \
         patch("avtv.assembler.probe_duration", return_value=8.0):
        assemble_run(
            selections=sels,
            narration_path=tmp_path / "narration.mp3",
            output_path=tmp_path / "out.mp4",
            work_dir=tmp_path / "work",
            downloader=fake_dl,
            target_w=1920, target_h=1080, target_fps=30,
            clip_audio_db=-20.0,
        )
        # 2 segment encodes + 1 final mux = 3 ffmpeg calls
        assert mock_run.call_count == 3
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement assemble_run + probe**

In `src/avtv/downloader.py`, append a sync wrapper:

```python
def fetch_sync(self, url: str, kind: Kind) -> Path:
    import asyncio
    return asyncio.run(self.fetch(url, kind))
```

Append to `src/avtv/assembler.py`:
```python
import json
from pathlib import Path

from avtv.downloader import Downloader


def probe_duration(path: Path) -> float:
    """Probe media duration via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr}")
    fmt = json.loads(result.stdout).get("format", {})
    return float(fmt.get("duration", 0.0))


def _fetch_for(downloader: Downloader, sel: Selection) -> Path:
    media_kind = "video" if sel.kind == "video" else "image"
    return downloader.fetch_sync(sel.url, kind=media_kind)


def assemble_run(
    selections: list[Selection],
    narration_path: Path,
    output_path: Path,
    work_dir: Path,
    downloader: Downloader,
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    validate_continuations(selections)

    segment_paths: list[Path] = []
    last_segment: Path | None = None

    for sel in selections:
        if sel.kind == "continuation":
            # Continuation reuses the prior segment file as-is. No re-fetch,
            # no re-encode — just reference the same .ts in the concat list.
            assert last_segment is not None  # validate_continuations guarantees
            segment_paths.append(last_segment)
            continue

        input_path = _fetch_for(downloader, sel)
        duration = probe_duration(input_path) if sel.kind == "video" else None
        plan = plan_segment(sel, duration)

        seg_out = work_dir / f"segment_{sel.idx:04d}.ts"
        args = build_segment_args(
            input_path=str(input_path),
            output_path=str(seg_out),
            plan=plan,
            target_w=target_w,
            target_h=target_h,
            target_fps=target_fps,
            clip_audio_db=clip_audio_db,
        )
        run_ffmpeg(args)
        segment_paths.append(seg_out)
        last_segment = seg_out

    list_path = work_dir / "concat.txt"
    list_path.write_text(build_concat_demuxer_file([str(p) for p in segment_paths]))

    mux_args = build_final_mux_args(
        concat_list_path=str(list_path),
        narration_path=str(narration_path),
        output_path=str(output_path),
    )
    run_ffmpeg(mux_args)
    return output_path
```

- [ ] **Step 4: Run, expect pass**

- [ ] **Step 5: Commit**

```bash
git add src/avtv/assembler.py src/avtv/downloader.py tests/test_assemble_run.py
git commit -m "feat(assembler): add assemble_run orchestrator with continuation resolution"
```

---

## Phase 8 — CLI

### Task 25: CLI skeleton + parse / brief / search / select stages

**Files:**
- Create: `src/avtv/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test (smoke for parse stage)**

`tests/test_cli.py`:
```python
from pathlib import Path
from typer.testing import CliRunner

from avtv.cli import app

runner = CliRunner()


def test_cli_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "build" in r.stdout
    assert "parse" in r.stdout


def test_parse_creates_blocks_json(tmp_path, monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PEXELS_API_KEY",
              "PIXABAY_API_KEY", "UNSPLASH_API_KEY"]:
        monkeypatch.setenv(k, "x")

    script = tmp_path / "s.txt"
    script.write_text(
        (Path(__file__).parent / "fixtures" / "sample_script.txt").read_text()
    )
    runs = tmp_path / "runs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNS_DIR", str(runs))

    r = runner.invoke(
        app, ["parse", "--script", str(script), "--run-id", "test-run",
              "--runs-dir", str(runs)],
    )
    assert r.exit_code == 0, r.stdout
    assert (runs / "test-run" / "01_blocks.json").exists()
```

- [ ] **Step 2: Run, expect fail**

- [ ] **Step 3: Implement CLI parse + skeleton**

`src/avtv/cli.py`:
```python
from pathlib import Path

import typer
from rich.console import Console

from avtv.parser import parse_script
from avtv.runs import RunDir

app = typer.Typer(help="auto-video-to-video CLI")
console = Console()


def _runs_dir(runs_dir: str | None) -> Path:
    return Path(runs_dir) if runs_dir else Path("runs")


@app.command()
def parse(
    script: Path = typer.Option(..., exists=True, readable=True),
    run_id: str | None = typer.Option(None),
    runs_dir: str | None = typer.Option(None),
):
    """Stage 1: parse DOTTI SYNC script into 01_blocks.json."""
    base = _runs_dir(runs_dir)
    rd = RunDir(base_dir=base, run_id=run_id) if run_id else RunDir.new(base)
    rd.ensure()
    blocks = parse_script(script.read_text())
    rd.save_blocks(blocks)
    console.print(f"[green]parsed[/green] {len(blocks)} blocks → {rd.path}")
```

- [ ] **Step 4: Run, expect pass**

```bash
poetry run pytest tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/avtv/cli.py tests/test_cli.py
git commit -m "feat(cli): add Typer skeleton with parse stage"
```

---

### Task 26: CLI brief / search / select stages

**Files:**
- Modify: `src/avtv/cli.py`

- [ ] **Step 1: Add stage commands**

Append to `src/avtv/cli.py`:
```python
import asyncio

from avtv.briefing import get_provider
from avtv.config import Settings
from avtv.search.orchestrator import SearchOrchestrator
from avtv.search.pexels import PexelsAdapter
from avtv.search.pixabay import PixabayAdapter
from avtv.search.archive_org import ArchiveOrgAdapter
from avtv.search.wikimedia import WikimediaAdapter
from avtv.search.unsplash import UnsplashAdapter
from avtv.selector import select_per_block


def _build_orchestrator(settings: Settings) -> SearchOrchestrator:
    adapters = [
        PexelsAdapter(api_key=settings.pexels_api_key),
        PixabayAdapter(api_key=settings.pixabay_api_key),
        ArchiveOrgAdapter(),
        WikimediaAdapter(),
        UnsplashAdapter(api_key=settings.unsplash_api_key),
    ]
    return SearchOrchestrator(adapters=adapters, top_k=settings.search_top_k)


@app.command()
def brief(
    run_id: str = typer.Option(..., help="Existing run id"),
    runs_dir: str | None = typer.Option(None),
    provider: str | None = typer.Option(None, help="claude | gpt"),
):
    """Stage 2: generate visual briefs via LLM."""
    settings = Settings()
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    blocks = rd.load_blocks()
    name = provider or settings.default_llm_provider
    llm = get_provider(name)
    briefs = llm.generate_briefs(blocks)
    rd.save_briefs(briefs)
    console.print(f"[green]briefed[/green] via {name}: {len(briefs)} briefs")


@app.command()
def search(
    run_id: str = typer.Option(...),
    runs_dir: str | None = typer.Option(None),
):
    """Stage 3: query free APIs for candidates per brief."""
    settings = Settings()
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    briefs = rd.load_briefs()
    orch = _build_orchestrator(settings)
    results = asyncio.run(
        orch.search_for_briefs(briefs, concurrency=settings.search_concurrency)
    )
    rd.save_search_results(results)
    total = sum(len(v) for v in results.values())
    console.print(f"[green]searched[/green] {len(briefs)} briefs, {total} candidates")


@app.command()
def select(
    run_id: str = typer.Option(...),
    runs_dir: str | None = typer.Option(None),
):
    """Stage 4: rank candidates and pick one per block."""
    settings = Settings()
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    selections = select_per_block(briefs, candidates, settings=settings)
    rd.save_selections(selections)
    console.print(f"[green]selected[/green] {len(selections)} clips")
```

- [ ] **Step 2: Smoke check**

```bash
poetry run avtv --help
```
Expected: lists all commands.

- [ ] **Step 3: Commit**

```bash
git add src/avtv/cli.py
git commit -m "feat(cli): add brief, search, and select stage commands"
```

---

### Task 27: CLI assemble + build

**Files:**
- Modify: `src/avtv/cli.py`

- [ ] **Step 1: Implement assemble + build**

Append to `src/avtv/cli.py`:
```python
from avtv.assembler import assemble_run
from avtv.downloader import Downloader


@app.command()
def assemble(
    run_id: str = typer.Option(...),
    audio: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    runs_dir: str | None = typer.Option(None),
    cache_dir: str | None = typer.Option(None),
):
    """Stage 5: download clips and assemble final MP4."""
    settings = Settings()
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    selections = rd.load_selections()
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
    console.print(f"[green]done[/green] → {out}")


@app.command()
def build(
    audio: Path = typer.Option(..., exists=True),
    script: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    provider: str | None = typer.Option(None),
    run_id: str | None = typer.Option(None),
    runs_dir: str | None = typer.Option(None),
    cache_dir: str | None = typer.Option(None),
):
    """Run the full pipeline: parse → brief → search → select → assemble."""
    settings = Settings()
    base = _runs_dir(runs_dir)
    rd = RunDir(base_dir=base, run_id=run_id) if run_id else RunDir.new(base)
    rd.ensure()

    console.print(f"[bold]run-id:[/bold] {rd.run_id}")

    blocks = parse_script(script.read_text())
    rd.save_blocks(blocks)
    console.print(f"  parsed {len(blocks)} blocks")

    name = provider or settings.default_llm_provider
    llm = get_provider(name)
    briefs = llm.generate_briefs(blocks)
    rd.save_briefs(briefs)
    console.print(f"  briefed via {name}")

    orch = _build_orchestrator(settings)
    results = asyncio.run(
        orch.search_for_briefs(briefs, concurrency=settings.search_concurrency)
    )
    rd.save_search_results(results)
    console.print(f"  searched: {sum(len(v) for v in results.values())} candidates")

    selections = select_per_block(briefs, results, settings=settings)
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

- [ ] **Step 2: Smoke check**

```bash
poetry run avtv build --help
```
Expected: shows all flags.

- [ ] **Step 3: Commit**

```bash
git add src/avtv/cli.py
git commit -m "feat(cli): add assemble and build commands for full pipeline"
```

---

### Task 28: CLI utility commands

**Files:**
- Modify: `src/avtv/cli.py`

- [ ] **Step 1: Add runs / inspect / clean**

Append to `src/avtv/cli.py`:
```python
import shutil


@app.command()
def runs(runs_dir: str | None = typer.Option(None)):
    """List all runs and their state."""
    base = _runs_dir(runs_dir)
    if not base.exists():
        console.print("[dim](no runs)[/dim]")
        return
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        stages = []
        for name, label in [
            (RunDir.BLOCKS, "parse"),
            (RunDir.BRIEFS, "brief"),
            (RunDir.SEARCH, "search"),
            (RunDir.SELECTIONS, "select"),
        ]:
            mark = "✓" if (child / name).exists() else "·"
            stages.append(f"{mark}{label}")
        console.print(f"{child.name}  {' '.join(stages)}")


@app.command()
def inspect(run_id: str, runs_dir: str | None = typer.Option(None)):
    """Show stage status for a single run."""
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    if not rd.path.exists():
        console.print(f"[red]no such run:[/red] {run_id}")
        raise typer.Exit(1)
    for name, label in [
        (RunDir.BLOCKS, "blocks"),
        (RunDir.BRIEFS, "briefs"),
        (RunDir.SEARCH, "search results"),
        (RunDir.SELECTIONS, "selections"),
    ]:
        p = rd.path / name
        status = f"[green]✓[/green] {p.stat().st_size}B" if p.exists() else "[dim]·[/dim]"
        console.print(f"  {label:18s} {status}")


@app.command()
def clean(run_id: str, runs_dir: str | None = typer.Option(None)):
    """Remove a run directory (keeps download cache)."""
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    if not rd.path.exists():
        console.print(f"[red]no such run:[/red] {run_id}")
        raise typer.Exit(1)
    shutil.rmtree(rd.path)
    console.print(f"[yellow]removed[/yellow] {rd.path}")
```

- [ ] **Step 2: Smoke check**

```bash
poetry run avtv runs
poetry run avtv inspect --help
poetry run avtv clean --help
```
Expected: all run cleanly.

- [ ] **Step 3: Commit**

```bash
git add src/avtv/cli.py
git commit -m "feat(cli): add runs, inspect, and clean utility commands"
```

---

## Phase 9 — Docs & polish

### Task 29: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README with usage instructions**

`README.md`:
```markdown
# auto-video-to-video

Generate long-form YouTube documentary MP4s from a narration MP3 and a
DOTTI SYNC script (8-second blocks) by automatically finding B-roll from
free video/image APIs.

## Status: MVP

Free sources only (Pexels, Pixabay, Internet Archive, Wikimedia, Unsplash).

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- `ffmpeg` (e.g. `brew install ffmpeg`)
- API keys: Anthropic OR OpenAI (LLM); Pexels, Pixabay, Unsplash (search)

## Setup

```bash
git clone <repo>
cd auto-video-to-video
poetry install
cp .env.example .env
# edit .env with your keys
```

## Usage

Full pipeline:

```bash
poetry run avtv build \
  --audio input/narration.mp3 \
  --script input/script.txt \
  --out output.mp4
```

Stages individually (resume from any point):

```bash
poetry run avtv parse    --script input/script.txt --run-id myrun
poetry run avtv brief    --run-id myrun --provider claude
poetry run avtv search   --run-id myrun
poetry run avtv select   --run-id myrun
poetry run avtv assemble --run-id myrun --audio input/narration.mp3 --out output.mp4
```

Inspect:

```bash
poetry run avtv runs
poetry run avtv inspect myrun
```

## Manual override

To swap a clip for a specific block, edit `runs/<run-id>/04_selections.json`
and re-run only assembly:

```bash
poetry run avtv assemble --run-id <id> --audio input/narration.mp3 --out output.mp4
```

## Provider selection

```bash
# .env
DEFAULT_LLM_PROVIDER=claude  # or gpt

# or per command
poetry run avtv brief --run-id <id> --provider gpt
```

## Architecture

See [`docs/superpowers/specs/2026-04-26-auto-video-to-video-design.md`](docs/superpowers/specs/2026-04-26-auto-video-to-video-design.md).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and override instructions"
```

---

### Task 30: gitignore tidy + final lint pass

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append project-specific ignores**

Append to `.gitignore`:
```
# auto-video-to-video runtime
runs/
cache/
input/
*.mp4
*.mp3
output*

# Python tooling
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 2: Run lint + types**

```bash
poetry run ruff check src/ tests/
poetry run ruff format --check src/ tests/
poetry run mypy src/
```
Expected: no errors. If any, fix and re-run.

- [ ] **Step 3: Run full test suite**

```bash
poetry run pytest -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: tighten gitignore and run final lint/type pass"
```

---

### Task 31: End-to-end smoke (manual)

**Files:** (none — manual sanity check)

- [ ] **Step 1: Place real `sample.mp3` and `sample_script.txt`** in `input/`. Use a 30-second sample (4 blocks).

- [ ] **Step 2: Run full build**

```bash
poetry run avtv build \
  --audio input/sample.mp3 \
  --script input/sample_script.txt \
  --out output.mp4
```

Expected: completes without errors. `output.mp4` plays at 1920x1080, 30 fps, with narration audible and B-roll visible.

- [ ] **Step 3: Verify intermediate artifacts**

```bash
poetry run avtv inspect <run-id>
```
Expected: all four stages green.

- [ ] **Step 4: Test manual override**

Edit `runs/<run-id>/04_selections.json`, change one clip URL to a different one from `03_search_results.json`, then:

```bash
poetry run avtv assemble --run-id <id> --audio input/sample.mp3 --out output.mp4
```
Expected: re-runs only assembly stage, swapped clip appears in output.

---

## Out of scope for this plan (future tasks)

- Storyblocks / Artgrid paid adapters (task list will mirror Tasks 11-15)
- CLIP-based visual reranking
- Subtitle burn-in
- Background music mixing
- Web preview UI
- Cloud deployment
- Concurrency for the assembly stage (segments are encoded sequentially in the MVP)
- **Crossfade at loop seam** for clips < 4s. Spec specified 0.3s crossfade,
  but MVP loop uses simple `-stream_loop` without crossfade. Real impact is
  small (short-clip selection should be rare) and crossfaded looping in
  ffmpeg requires manual concat-with-xfade — defer until after first builds
  show whether it's actually visible.
