# auto-video-to-video

Generate long-form YouTube documentary MP4s from a narration MP3 and a
DOTTI SYNC script (8-second blocks) by automatically finding B-roll from
free video/image APIs.

## Status: MVP

Free sources only (Pexels, Pixabay, Internet Archive, Wikimedia, Unsplash).

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- `ffmpeg` and `ffprobe` on `PATH`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg` (or `choco install ffmpeg`, or download
    from <https://www.gyan.dev/ffmpeg/builds/> and add `bin/` to `PATH`)
  - Linux: `apt install ffmpeg` (or distro equivalent)
- API keys: Anthropic OR OpenAI (LLM); Pexels, Pixabay, Unsplash (search)

## Platform notes

- **macOS / Linux:** primary tested platforms.
- **Windows:** supported. Use PowerShell or cmd; `poetry run avtv ...` works
  the same. The pipeline writes UTF-8 JSON and uses forward-slash absolute
  paths in the ffmpeg concat list, so no encoding or path-separator quirks.

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
