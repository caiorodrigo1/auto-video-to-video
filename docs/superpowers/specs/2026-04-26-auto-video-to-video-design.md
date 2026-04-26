# auto-video-to-video — Design Spec

**Date:** 2026-04-26
**Status:** Approved (pending implementation plan)

## 1. Context & goals

Automate the production of long-form horizontal YouTube documentary videos from a narration MP3 plus a pre-segmented script. The system finds illustrative B-roll for each segment of the script (querying free online video banks), then assembles a final MP4 synced to the original narration.

**Primary use case:** "faceless" documentary YouTube channels (5-15 min videos) covering mixed niches — generic content (lifestyle, finance, science) and specific/historical content (named events, places, people). Narration is mostly English but may be in any language.

**MVP goal:** validate the idea using only free video sources. Future iteration may add paid sources (Storyblocks, Artgrid).

## 2. Scope

**In scope (MVP):**

- Input: MP3 + script in **DOTTI SYNC format** (fixed 8-second blocks, indexed)
- LLM-driven visual briefing per block (Claude **and** GPT, both supported)
- Multi-source search across free video/image APIs
- Automatic clip selection per block with manual override via JSON
- Final MP4 assembly with speed-matched clips, audio mixing, and Ken Burns for images

**Out of scope (MVP):**

- Paid stock APIs (Storyblocks, Artgrid, Envato)
- Burn-in subtitles (audio remains as-is)
- Background music
- Custom transitions other than hard cuts
- Web UI (CLI only)
- Cloud deployment (runs locally)

## 3. Input

### MP3
Narration audio, single file, any duration. Sample rate / codec irrelevant — ffmpeg will normalize during mux.

### Script — DOTTI SYNC format

Fixed-width 8-second blocks, plain text:

```
============================================================
SINCRONIZACAO DOTTI SYNC - BLOCOS DE 8 SEGUNDOS
============================================================
Arquivo: <name>.mp3
Duracao: HH:MM
Total de prompts: <N>
============================================================

PROMPT 001 | 00:00 - 00:08
<narration text for this 8-second block>
------------------------------------------------------------

PROMPT 002 | 00:08 - 00:16
<narration text>
------------------------------------------------------------
...
```

**Edge cases:**

- **Empty blocks:** some `PROMPT NNN` blocks have no text (continuation of previous narration or silence). Detected by empty body between header and `---` separator.
- **Mid-sentence cuts:** block text frequently ends mid-sentence; resolved at the briefing layer using neighboring blocks for context.
- **Language:** any (German in current sample, mostly English expected). LLM produces English search queries regardless.

## 4. Output

**Single file:** `output.mp4`

| Spec | Value |
|---|---|
| Resolution | 1920x1080 |
| Aspect ratio | 16:9 |
| Framerate | 30 fps |
| Video codec | H.264 |
| Audio codec | AAC |
| Audio mix | Narration at 0 dB + clip audio at -20 dB |
| Subtitles | None |
| Background music | None |
| Transitions | Hard cuts |
| Fallback for no clip | Image with Ken Burns effect |
| Empty-block handling | Extend previous clip |

## 5. Architecture overview

### Approach: pipeline with disk-persisted intermediate artifacts

Each stage reads from and writes to disk. Re-running any stage individually is supported (idempotent). Manual editing of intermediate JSON between stages is supported and intended.

### Run lifecycle

```
input/audio.mp3        ┐
input/script.txt       ┘
        │
        ▼  parser
   01_blocks.json                      [Block(idx, start, end, text, is_empty)]
        │
        ▼  briefing  (Claude or GPT, full-script context)
   02_visual_briefs.json               [VisualBrief(idx, query_en, fallback_query, kind, continuity_hint)]
        │
        ▼  search    (orchestrator → N adapters in parallel)
   03_search_results.json              {idx: [Candidate(...)]}
        │
        ▼  selector  (ranking + neighbor dedup)
   04_selections.json                  {idx: Selection(url, kind, trim/speed)}
        │
        ▼  downloader (cache by URL hash, cross-run)
   cache/clips/<hash>.mp4 / cache/images/<hash>.jpg
        │
        ▼  assembler (ffmpeg)
   output.mp4
```

### Run directory layout

```
runs/<run-id>/
  01_blocks.json
  02_visual_briefs.json
  03_search_results.json
  04_selections.json
  logs/
    parser.log
    briefing.log
    search.log
    selector.log
    assembler.log
  output.mp4
```

Downloaded media is cached **outside the run dir** (`cache/clips/`, `cache/images/`) so multiple runs share downloads.

## 6. Data models (pydantic)

```python
class Block(BaseModel):
    idx: int                       # 1-based, matches DOTTI SYNC numbering
    start: float                   # seconds
    end: float                     # seconds (always start + 8.0)
    text: str                      # narration text, may be empty
    is_empty: bool                 # True iff text is whitespace-only

class VisualBrief(BaseModel):
    idx: int
    query_en: str                  # primary search query, English
    fallback_query: str            # more generic alternative
    kind: Literal["video", "image", "continuation"]
    continuity_hint: str | None    # e.g., "wide shot, daytime, woodland"
    notes: str | None              # LLM rationale (optional, for debugging)

class Candidate(BaseModel):
    source: str                    # "pexels" | "pixabay" | "archive_org" | "wikimedia" | "unsplash"
    url: str                       # download URL
    preview_url: str | None
    kind: Literal["video", "image"]
    duration: float | None         # None for images
    width: int
    height: int
    license: str
    attribution: str | None

class Selection(BaseModel):
    idx: int
    source: str
    url: str
    kind: Literal["video", "image", "continuation"]
    # for video:
    trim_start: float | None       # if trim strategy
    trim_end: float | None
    speed_factor: float | None     # if speed-match strategy (clip_duration / 8.0)
    loop: bool                     # True if short clip needs looping
    # for image: no extra fields (Ken Burns applied uniformly)
    # for continuation: refers back to selection of (idx - 1)
    attribution: str | None
```

## 7. Pipeline stages

### 7.1 Parser

**Module:** `src/avtv/parser.py`

Reads DOTTI SYNC text, emits `list[Block]`. Pure function, no I/O outside the file read.

- Skips header (everything before first `PROMPT NNN`).
- Parses `PROMPT NNN | MM:SS - MM:SS` lines into idx + start/end.
- Body is everything between header line and next `------` separator.
- `is_empty` if body is whitespace-only.
- Validates: idx is sequential starting at 1, no gaps; durations are 8.0s; start/end consistent.

### 7.2 Briefing

**Module:** `src/avtv/briefing/`

#### LLM provider abstraction

```python
class LLMProvider(Protocol):
    def generate_briefs(
        self, blocks: list[Block], topic_hint: str | None = None
    ) -> list[VisualBrief]: ...

class ClaudeProvider(LLMProvider):  # uses anthropic SDK
class OpenAIProvider(LLMProvider):  # uses openai SDK

PROVIDERS = {"claude": ClaudeProvider, "gpt": OpenAIProvider}
```

#### Defaults

| Provider | Default model | Override via |
|---|---|---|
| Claude | `claude-sonnet-4-6` | `--model` |
| OpenAI | `gpt-5` | `--model` |

#### Prompt structure

- **System prompt** (cacheable):
  - Role: "video researcher producing B-roll search queries for documentary"
  - Output schema (JSON of `VisualBrief[]`)
  - Heuristics:
    - Always English queries, regardless of input language
    - `kind: "image"` for named historical entities/events; `kind: "video"` for generic action/scenes
    - `kind: "continuation"` for empty blocks (no narration text)
    - `fallback_query` strictly more generic than `query_en`
    - Avoid same query in ≥3 consecutive blocks (variation)
- **User message** (per run):
  - Full script numbered by block

#### Structured output

- Claude: forced `tool_use` with the `VisualBrief[]` schema
- OpenAI: `response_format={"type": "json_schema", ...}` (Structured Outputs)

#### Validation

- Pydantic-validate every `VisualBrief`. Single retry on failure.
- Assert `len(briefs) == len(blocks)`. Mismatch → fatal error.

#### Cost target

~$0.05-0.15 per video for both providers. Not a decisive factor at MVP scale.

### 7.3 Search

**Module:** `src/avtv/search/`

#### Sources (free tier)

| Source | API | Strength | Free limit |
|---|---|---|---|
| Pexels | REST + key | Modern stock, EN-indexed, 1080p | 200 req/h |
| Pixabay | REST + key | Generic stock, complements Pexels | 100 req/min |
| Internet Archive | `internetarchive` lib / REST | Historical, public domain | none |
| Wikimedia Commons | MediaWiki API (no key) | Educational, historical, CC media | none |
| Unsplash | REST + key | Premium images (Ken Burns fallback) | 50 req/h |

#### Adapter Protocol

```python
class SearchAdapter(Protocol):
    name: str
    media_kinds: set[Literal["video", "image"]]

    async def search(
        self,
        query: str,
        limit: int = 10,
        kind: Literal["video", "image"] = "video",
    ) -> list[Candidate]: ...
```

#### Orchestrator

For each `VisualBrief`:

1. **Routing by kind:**
   - `video` → Pexels + Pixabay + Archive + Wikimedia (parallel)
   - `image` → Unsplash + Pixabay-images + Wikimedia (parallel)
   - `continuation` → skip search entirely
2. **Two-tier query attempt:** try `query_en`; if no usable candidates, retry with `fallback_query`.
3. **Concurrency cap:** semaphore of 5 simultaneous searches across the run.
4. **Output:** top-K (default K=8) candidates per block, normalized to `Candidate` schema.

### 7.4 Selector

**Module:** `src/avtv/selector.py`

Ranks candidates per block, picks one, dedupes against neighbors.

#### Score formula

```
score = aspect_ratio_match
      + duration_match
      + resolution_score
      + source_preference
      - repetition_penalty
```

| Component | Values |
|---|---|
| `aspect_ratio_match` | 1.0 if ≥1.7:1; 0.5 if 1.4-1.7; 0 below |
| `duration_match` | 1.0 if 6.8 ≤ D ≤ 9.6 (factor 0.85-1.2); 0.7 if 4-6.8 or 9.6-16; 0.4 if 2-4 or 16-24; 0.1 otherwise |
| `resolution_score` | 1.0 if ≥1080p; 0.6 if 720p; 0.3 if smaller |
| `source_preference` | Pexels 0.3, Pixabay 0.25, Archive 0.2, Wikimedia 0.15 (tiebreaker) |
| `repetition_penalty` | -0.5 if same URL is selected within ±2 neighbors |

Weights configurable in `config.py`.

#### Neighbor dedup

Sliding window of ±2 blocks. If the top-ranked candidate's URL is already in that range, fall through to #2, etc.

#### Fallback when nothing usable

1. Force-retry with `kind: "image"` against Unsplash using `fallback_query`.
2. If still nothing, downgrade to `kind: "continuation"` (extend previous block's clip).
3. If block 1 fails entirely, fatal error with a report of which blocks need manual intervention.

### 7.5 Downloader

**Module:** `src/avtv/downloader.py`

- Cache key: SHA-256 of normalized URL.
- Storage: `cache/clips/<hash>.mp4`, `cache/images/<hash>.jpg` — shared across runs.
- HTTP via `httpx`, with `tenacity` retries (3x, exponential backoff) for transient failures.
- Skips download if cache file exists and size > 0.

### 7.6 Assembler

**Module:** `src/avtv/assembler.py`

ffmpeg pipeline using `ffmpeg-python`. **Requires** `ffmpeg` installed system-wide.

#### Per-block segment generation

Each block becomes an exactly **8.0-second segment** at 1920x1080, 30 fps.

##### Speed-match strategy (videos)

Let `D` = clip duration, `factor = D / 8.0`.

| Range | Strategy | Audio |
|---|---|---|
| 6.8 ≤ D ≤ 9.6 (factor 0.85-1.2) | `setpts` speed-match | `atempo` to match |
| 4.0 ≤ D < 6.8 (factor 0.5-0.85) | `setpts` slow-mo + speed-match | `atempo` |
| D < 4.0 | **Loop** with 0.3s crossfade at seam, no speed | Drop clip audio (only narration) |
| 9.6 < D ≤ 16 (factor 1.2-2.0) | `setpts` speedup | `atempo` |
| D > 16 | **Trim** an 8s window from the **middle** of the clip | Original clip audio (no speed change) |

Crossfade at loop seam is enabled by default.

##### Ken Burns (images)

`zoompan` filter:

- Duration: 8.0s @ 30 fps
- Zoom: linear 1.0 → 1.15 (in)
- Direction alternates per consecutive image block (in / out / in / out) to avoid monotony
- Pan: ±5% horizontal drift, alternating
- Output frame: 1920x1080

Parameters are configurable in `config.py`.

##### Continuation blocks

`Selection.kind == "continuation"`:

- Reuse the source from `(idx - 1)`.
- Extend trim window of the original clip by 8s if length permits.
- If not, repeat the previous segment as-is.

##### Per-segment normalization

1. Scale + pad to 1920x1080 (letterbox if aspect doesn't match).
2. Force 30 fps.
3. Normalize clip audio to -20 dB (skipped if no clip audio per rules above).
4. Encode as MPEG-TS (`segment_<idx>.ts`) for fast concat.

#### Final assembly

1. **Concat** all segments via ffmpeg `concat` demuxer (no re-encode).
2. **Mix audio:** narration full + clip audio at -20 dB via `amix` filter.
3. Output: H.264 + AAC `output.mp4`.

#### Performance targets

| Operation | Cold cache | Warm cache (assembly only) |
|---|---|---|
| 14-min input | ~6-10 min | ~1-2 min |

#### Logging

- Each ffmpeg invocation logged to `logs/assembler.log`.
- Failure logs include the exact ffmpeg command for manual reproduction.

## 8. CLI

Built with **Typer**.

```bash
# Full pipeline
avtv build \
  --audio input/audio.mp3 \
  --script input/script.txt \
  --out output.mp4 \
  [--provider claude|gpt] \
  [--model <override>] \
  [--run-id <custom>]

# Individual stages (resume from any point)
avtv parse    --script input/script.txt --run-id <id>
avtv brief    --run-id <id> [--provider claude|gpt]
avtv search   --run-id <id>
avtv select   --run-id <id>
avtv assemble --run-id <id> --audio input/audio.mp3 --out output.mp4

# Utilities
avtv runs                          # list runs and their state
avtv inspect <run-id>              # show stage status
avtv clean <run-id>                # remove run artifacts (keep download cache)
```

Each stage command auto-runs missing prerequisite stages.

## 9. Tech stack

```toml
# pyproject.toml — primary deps
python = "^3.11"
typer = "^0.12"
pydantic = "^2.7"
pydantic-settings = "^2.3"
httpx = "^0.27"
anthropic = "^0.39"
openai = "^1.50"
ffmpeg-python = "^0.2"
internetarchive = "^4.0"
rich = "^13.7"
tenacity = "^9.0"

# dev
pytest = "^8"
pytest-asyncio = "^0.24"
ruff = "^0.6"
mypy = "^1.11"
```

**System deps:** `ffmpeg` (Homebrew or equivalent).

**API keys (`.env`):**

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
PEXELS_API_KEY=
PIXABAY_API_KEY=
UNSPLASH_API_KEY=
DEFAULT_LLM_PROVIDER=claude
```

(Archive.org and Wikimedia don't require keys.)

## 10. Project structure

```
auto-video-to-video/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-26-auto-video-to-video-design.md
├── src/avtv/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── parser.py
│   ├── briefing/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── claude.py
│   │   ├── openai.py
│   │   └── prompt.py
│   ├── search/
│   │   ├── base.py
│   │   ├── pexels.py
│   │   ├── pixabay.py
│   │   ├── archive_org.py
│   │   ├── wikimedia.py
│   │   ├── unsplash.py
│   │   └── orchestrator.py
│   ├── selector.py
│   ├── downloader.py
│   └── assembler.py
├── tests/
│   ├── test_parser.py
│   ├── test_selector.py
│   ├── test_assembler.py
│   └── fixtures/
│       └── sample_script.txt
├── input/                         # gitignored
├── runs/                          # gitignored
└── cache/                         # gitignored
    ├── clips/
    └── images/
```

## 11. Testing strategy (MVP)

Focus on **pure logic**, not integration with external APIs or ffmpeg.

| Module | Tests |
|---|---|
| `parser` | DOTTI SYNC parsing, empty blocks, malformed inputs, sequential idx validation |
| `selector` | Score under varied candidate sets, neighbor dedup, fallback path |
| `assembler` | Speed-factor decision logic (no actual ffmpeg invocation — mocks) |
| `briefing` | Prompt construction, schema validation, retry on invalid output (mock LLM clients) |
| `search` adapters | Response parsing into `Candidate` (mock HTTP) |

**End-to-end smoke test:** manual, with a 5-block sample script and real APIs. Not part of CI.

## 12. Open questions / future work

- **Paid sources:** add Storyblocks, Artgrid adapters once free-tier MVP is validated.
- **Visual reranking with CLIP:** for higher quality, download top-K candidates and rerank by CLIP similarity to the visual brief. Adds latency and storage; defer until MVP feedback shows it's needed.
- **Web UI / preview:** today the only way to preview is opening `output.mp4`. A future version could render a low-res preview after `select` so the user can approve before download/assembly.
- **Voice-driven topic detection:** auto-detect narration tone (contemplative, energetic) from the MP3 to bias visual selection. Not in MVP.
- **Subtitle burn-in:** could be added later as an optional flag without changing the rest of the pipeline.

## 13. Approval

Design approved by user on 2026-04-26 across 5 sections (architecture overview, briefing, search & selection, assembly, CLI/tech-stack/structure). Ready for implementation planning.
