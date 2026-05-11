import json

from avtv.models import Block, Candidate
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


def test_run_dir_save_load_search_results_key_round_trip(tmp_path):
    rd = RunDir(base_dir=tmp_path, run_id="abc")
    rd.ensure()
    cand = Candidate(
        source="pexels",
        url="https://example.com/v.mp4",
        kind="video",
        duration=8.0,
        width=1920,
        height=1080,
        license="Pexels License",
    )
    rd.save_search_results({0: [cand], 3: []})

    raw = json.loads((tmp_path / "abc" / "03_search_results.json").read_text())
    assert all(isinstance(k, str) for k in raw)

    loaded = rd.load_search_results()
    assert sorted(loaded.keys()) == [0, 3]
    assert loaded[0][0].url == "https://example.com/v.mp4"


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
