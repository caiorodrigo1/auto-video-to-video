from pathlib import Path

from typer.testing import CliRunner

from avtv.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "parse" in r.stdout


def test_parse_creates_blocks_json(tmp_path: Path, monkeypatch: object) -> None:
    mp = monkeypatch  # type: ignore[attr-defined]
    for k in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "UNSPLASH_API_KEY",
    ]:
        mp.setenv(k, "x")

    script = tmp_path / "s.txt"
    script.write_text((Path(__file__).parent / "fixtures" / "sample_script.txt").read_text())
    runs = tmp_path / "runs"
    mp.chdir(tmp_path)
    mp.setenv("RUNS_DIR", str(runs))

    r = runner.invoke(
        app,
        ["parse", "--script", str(script), "--run-id", "test-run", "--runs-dir", str(runs)],
    )
    assert r.exit_code == 0, r.stdout
    assert (runs / "test-run" / "01_blocks.json").exists()


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
