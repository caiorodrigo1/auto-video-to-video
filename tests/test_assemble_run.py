from unittest.mock import MagicMock, patch

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
