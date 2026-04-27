import json

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
