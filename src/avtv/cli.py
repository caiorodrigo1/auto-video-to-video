import asyncio
from pathlib import Path

import typer
from rich.console import Console

from avtv.assembler import assemble_run
from avtv.briefing import get_provider
from avtv.config import Settings
from avtv.downloader import Downloader
from avtv.parser import parse_script
from avtv.runs import RunDir
from avtv.search.archive_org import ArchiveOrgAdapter
from avtv.search.base import SearchAdapter
from avtv.search.orchestrator import SearchOrchestrator
from avtv.search.pexels import PexelsAdapter
from avtv.search.pixabay import PixabayAdapter
from avtv.search.unsplash import UnsplashAdapter
from avtv.search.wikimedia import WikimediaAdapter
from avtv.selector import select_per_block

app = typer.Typer(help="auto-video-to-video CLI")
console = Console()


def _runs_dir(runs_dir: str | None) -> Path:
    return Path(runs_dir) if runs_dir else Path("runs")


@app.command()
def parse(
    script: Path = typer.Option(..., exists=True, readable=True),  # noqa: B008
    run_id: str | None = typer.Option(None),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Stage 1: parse DOTTI SYNC script into 01_blocks.json."""
    base = _runs_dir(runs_dir)
    rd = RunDir(base_dir=base, run_id=run_id) if run_id else RunDir.new(base)
    rd.ensure()
    blocks = parse_script(script.read_text())
    rd.save_blocks(blocks)
    console.print(f"[green]parsed[/green] {len(blocks)} blocks → {rd.path}")


def _build_orchestrator(settings: Settings) -> SearchOrchestrator:
    adapters: list[SearchAdapter] = [
        PexelsAdapter(api_key=settings.pexels_api_key),  # type: ignore[list-item]
        PixabayAdapter(api_key=settings.pixabay_api_key),  # type: ignore[list-item]
        ArchiveOrgAdapter(),  # type: ignore[list-item]
        WikimediaAdapter(),  # type: ignore[list-item]
        UnsplashAdapter(api_key=settings.unsplash_api_key),  # type: ignore[list-item]
    ]
    return SearchOrchestrator(adapters=adapters, top_k=settings.search_top_k)


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
    briefs = llm.generate_briefs(blocks)
    rd.save_briefs(briefs)
    console.print(f"[green]briefed[/green] via {name}: {len(briefs)} briefs")


@app.command()
def search(
    run_id: str = typer.Option(...),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Stage 3: query free APIs for candidates per brief."""
    settings = Settings()  # type: ignore[call-arg]
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
    run_id: str = typer.Option(...),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Stage 4: rank candidates and pick one per block."""
    settings = Settings()  # type: ignore[call-arg]
    rd = RunDir(base_dir=_runs_dir(runs_dir), run_id=run_id)
    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    selections = select_per_block(briefs, candidates, settings=settings)
    rd.save_selections(selections)
    console.print(f"[green]selected[/green] {len(selections)} clips")


@app.command()
def assemble(
    run_id: str = typer.Option(...),  # noqa: B008
    audio: Path = typer.Option(..., exists=True),  # noqa: B008
    out: Path = typer.Option(...),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
    cache_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Stage 5: download clips and assemble final MP4."""
    settings = Settings()  # type: ignore[call-arg]
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
    audio: Path = typer.Option(..., exists=True),  # noqa: B008
    script: Path = typer.Option(..., exists=True),  # noqa: B008
    out: Path = typer.Option(...),  # noqa: B008
    provider: str | None = typer.Option(None),  # noqa: B008
    run_id: str | None = typer.Option(None),  # noqa: B008
    runs_dir: str | None = typer.Option(None),  # noqa: B008
    cache_dir: str | None = typer.Option(None),  # noqa: B008
) -> None:
    """Run the full pipeline: parse → brief → search → select → assemble."""
    settings = Settings()  # type: ignore[call-arg]
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
