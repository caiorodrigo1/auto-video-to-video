# auto-video-to-video

Turn an MP3 narration plus a DOTTI SYNC script into a finished YouTube
documentary MP4. The pipeline asks an LLM to invent search queries for each
8‑second block, hits five free stock APIs in parallel, ranks the candidates,
and stitches everything together with ffmpeg, synced to your narration.

```
input/audio.mp3 ─┐
                 ├─► parse ─► brief ─► search ─► select ─► assemble ─► output.mp4
input/script.txt ┘            (LLM)   (5 APIs)            (ffmpeg)
```

**Status — MVP.** Single user, runs locally, free sources only. No paid
stock, no subtitle burn-in, no background music.

---

## Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| Package manager | [Poetry](https://python-poetry.org/docs/#installation) |
| Media tooling | `ffmpeg` and `ffprobe` on `PATH` |
| API keys | Anthropic **or** OpenAI (you pick the provider per run); Pexels, Pixabay, Unsplash |

### Installing ffmpeg

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` (Debian/Ubuntu) or distro equivalent |
| Windows | `winget install ffmpeg` — or `choco install ffmpeg`, or download from <https://www.gyan.dev/ffmpeg/builds/> and add the `bin/` folder to `PATH` |

After install, sanity-check both binaries:

```bash
ffmpeg -version
ffprobe -version
```

### Getting API keys (all free tiers)

| Service | Where | Notes |
|---|---|---|
| Anthropic | <https://console.anthropic.com/> | Used for Claude briefing |
| OpenAI | <https://platform.openai.com/api-keys> | Used for GPT briefing |
| Pexels | <https://www.pexels.com/api/new/> | Free, 200 req/h |
| Pixabay | <https://pixabay.com/api/docs/> | Free, 100 req/min |
| Unsplash | <https://unsplash.com/developers> | Free, 50 req/h |

You only need *one* of Anthropic / OpenAI (whichever provider you'll use for
briefing). The other three are for the search stage. Internet Archive and
Wikimedia Commons need no key.

---

## Setup

```bash
git clone <repo-url>
cd auto-video-to-video
poetry install
cp .env.example .env
```

Open `.env` and fill in your keys:

```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
UNSPLASH_API_KEY=...
DEFAULT_LLM_PROVIDER=claude   # or gpt
```

All five keys must exist (even with placeholder values) — the loader
requires them. Set unused ones to a dummy value like `unused`.

---

## Input format — DOTTI SYNC

The script must be plain text with fixed 8‑second blocks. Each block has a
header line `PROMPT NNN | MM:SS - MM:SS` and a body. Bodies may be empty
(the LLM marks them as `continuation` and the previous clip is extended).
The last block may be shorter than 8s (audio tail). Example:

```
============================================================
SINCRONIZACAO DOTTI SYNC - BLOCOS DE 8 SEGUNDOS
============================================================
Arquivo: narration.mp3
Duracao: 12:28
Total de prompts: 94
============================================================

PROMPT 001 | 00:00 - 00:08
First block of narration text here.
------------------------------------------------------------

PROMPT 002 | 00:08 - 00:16
Second block, may continue mid-sentence...
------------------------------------------------------------
```

Indices must be sequential starting at 1; intermediate blocks must be
exactly 8 seconds; the final block may be 0 < dur ≤ 8s.

---

## Usage

### Web UI (recommended)

```bash
poetry run avtv ui
```

Opens <http://localhost:8501>.

1. Click **➕ New build** in the sidebar.
2. Upload your MP3 and your DOTTI SYNC `.txt`.
3. Pick a provider (Claude or GPT) and (optionally) a custom run ID.
4. Click **▶ Start build**. Progress streams live by stage; the slow
   assemble step shows a `Segment N/total` progress bar.
5. When done, the output MP4 plays inline and there's a download button.

The sidebar lists previous runs with stage flags. Click any run to open
its inspector — you can browse the LLM briefs, edit `04_selections.json`
in place to swap a clip, and hit **🎬 Assemble** in the Actions tab to
re-render only the assembly stage (cheap — segments are cached on disk).

To stop the UI: `Ctrl+C` in the terminal, or `pkill -f "streamlit run"`.

### CLI — full pipeline

```bash
poetry run avtv build \
  --audio input/narration.mp3 \
  --script input/script.txt \
  --out output.mp4
```

Add `--provider gpt` to override the LLM for that run.

### CLI — stage by stage

Each stage writes JSON artifacts under `runs/<run-id>/`. You can run them
one at a time, inspect intermediate output, and resume later.

```bash
poetry run avtv parse    --script input/script.txt --run-id myrun
poetry run avtv brief    --run-id myrun --provider claude
poetry run avtv search   --run-id myrun
poetry run avtv select   --run-id myrun
poetry run avtv assemble --run-id myrun \
  --audio input/narration.mp3 --out output.mp4
```

### Inspecting runs

```bash
poetry run avtv runs                  # list all runs with stage flags
poetry run avtv inspect myrun         # detailed status for one run
poetry run avtv clean myrun           # delete a run dir (keeps download cache)
```

### Manual override

The selection layer is editable. Open `runs/<run-id>/04_selections.json`,
swap any block's `url` to a different one (you can copy candidates from
`03_search_results.json`), then re-run only assembly:

```bash
poetry run avtv assemble --run-id <id> --audio input/narration.mp3 --out output.mp4
```

The downloader caches by URL hash, so a swapped URL only fetches the new
clip; everything else is reused.

---

## How the pipeline works

| Stage | Output | Notes |
|---|---|---|
| **parse** | `01_blocks.json` | Splits DOTTI SYNC into typed `Block` records. |
| **brief** | `02_visual_briefs.json` | LLM (Claude / GPT) reads the full script and emits one English search query + fallback per block, plus `kind` (`video` / `image` / `continuation`). |
| **search** | `03_search_results.json` | All five adapters (Pexels, Pixabay, Internet Archive, Wikimedia, Unsplash) run in parallel. Falls back to the LLM's `fallback_query` if the primary returns nothing. |
| **select** | `04_selections.json` | Scores candidates by aspect ratio, duration fit, resolution, and source. Dedupes against ±2 neighbors so the same clip doesn't repeat back-to-back. |
| **assemble** | `output.mp4` | Downloads each picked clip (cached), encodes one 8s `.ts` segment per block (speed-match / trim / loop / Ken Burns), then concats and muxes with the narration. |

Output spec: 1920×1080, 30 fps, H.264 + AAC, hard cuts, narration audio
only. (Clip audio mix at -20 dB was specified but is currently disabled —
see "Known limits" below.)

---

## Cost & timing (typical 12-minute video)

| Item | Cost / Time |
|---|---|
| LLM brief (Claude Sonnet 4.6 or GPT-5) | ~$0.05 – $0.15 |
| API search across 5 providers | free |
| Cold assembly (download + encode 90 segments + mux) | ~5 – 10 min |
| Warm assembly (after editing one selection) | ~1 – 2 min (segments are cached) |

---

## File layout

```
auto-video-to-video/
├── input/                    # your MP3 + script live here (gitignored)
├── runs/                     # one folder per run (gitignored)
│   └── <run-id>/
│       ├── uploads/                  # MP3 + script for UI runs
│       ├── 01_blocks.json
│       ├── 02_visual_briefs.json
│       ├── 03_search_results.json
│       ├── 04_selections.json        # ← editable
│       ├── segments/                 # encoded .ts files
│       └── output.mp4
├── cache/                    # shared download cache (gitignored)
│   ├── clips/<sha-prefix>.mp4
│   └── images/<sha-prefix>.jpg
└── src/avtv/                 # source
```

The `cache/` directory persists across runs — if two videos use the same
clip, it's downloaded only once.

---

## Troubleshooting

**`ffmpeg failed (exit ...)` during assembly.** Check that `ffmpeg` and
`ffprobe` are both on `PATH`. The error message includes the exact
command — copy it and run by hand to see the real ffmpeg output.

**"validation error: ANTHROPIC_API_KEY field required"** on startup. All
five env keys must be present in `.env` (even unused ones — set to
`unused`).

**Last block has wrong duration.** The parser accepts a final block
shorter than 8 s (typical audio tail). If an intermediate block is wrong,
fix the DOTTI SYNC export.

**Streamlit asked for an email and hung the first time.** Already worked
around — `avtv ui` pre-creates `~/.streamlit/credentials.toml`. If you
launched Streamlit some other way, run `avtv ui` once to set it.

**One block has a clip you hate.** Open `04_selections.json`, change the
`url`, save, re-run `avtv assemble`. Or swap it in the UI's Selections
tab.

---

## Development

```bash
poetry run pytest -q            # 70 tests, all mocked, ~0.5s
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
poetry run mypy src/avtv/
```

Tests don't hit any real API or invoke ffmpeg. End-to-end smoke is manual
(use `avtv build` against a real input).

---

## Known limits / not in MVP

- **Clip audio mix at -20 dB:** spec called for ducked clip audio under
  the narration. Disabled because per-segment streams are heterogeneous
  (image / loop / extreme-speed segments have no audio track) and the
  concat demuxer can't unify them. Restoration plan: pad every segment
  with silent audio at encode time, then re-enable the `amix` filter.
- **Loop crossfade:** clips shorter than 4 s are looped to fill 8 s
  without crossfade — the seam is hard. Tolerable for MVP.
- **Paid stock APIs (Storyblocks, Artgrid):** not implemented. Adapter
  pattern is in place; adding one is one new file in `src/avtv/search/`.
- **CLIP visual reranking:** out of scope.
- **Subtitle burn-in / background music:** out of scope.
- **Cloud deployment:** runs locally only. The pipeline is heavy on
  ffmpeg and disk I/O, so serverless edges (Vercel, Cloudflare Workers)
  don't fit; Modal or Fly.io would be the natural targets.

---

## Architecture

Detailed design: [`docs/superpowers/specs/2026-04-26-auto-video-to-video-design.md`](docs/superpowers/specs/2026-04-26-auto-video-to-video-design.md).

Implementation plan (31 tasks executed via subagent-driven development):
[`docs/superpowers/plans/2026-04-26-auto-video-to-video.md`](docs/superpowers/plans/2026-04-26-auto-video-to-video.md).
