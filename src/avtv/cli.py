from pathlib import Path

import typer
from rich.console import Console

from avtv.parser import parse_script
from avtv.runs import RunDir

app = typer.Typer(help="auto-video-to-video CLI")
console = Console()


def _runs_dir(runs_dir: str | None) -> Path:
    return Path(runs_dir) if runs_dir else Path("runs")


@app.command(name="build", hidden=True)
def _build_placeholder() -> None:
    """Placeholder — implemented in Task 27."""
    console.print("TODO: build command not yet implemented")
    raise typer.Exit(code=0)


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
