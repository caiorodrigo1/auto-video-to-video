from unittest.mock import MagicMock, patch

from avtv.assembler import assemble_run
from avtv.models import Selection


def _video_sel(idx, url, **kw):
    base = dict(
        idx=idx,
        source="pexels",
        url=url,
        kind="video",
        speed_factor=1.0,
        loop=False,
        attribution=None,
    )
    base.update(kw)
    return Selection(**base)


def test_assemble_run_invokes_ffmpeg_for_each_segment(tmp_path):
    sels = [_video_sel(1, "u1"), _video_sel(2, "u2")]

    # Fake downloader returns a path
    fake_dl = MagicMock()
    fake_dl.fetch_sync = MagicMock(side_effect=lambda url, kind: tmp_path / f"{url}.mp4")

    # Probe always returns 8s
    with (
        patch("avtv.assembler.run_ffmpeg") as mock_run,
        patch("avtv.assembler.probe_duration", return_value=8.0),
    ):
        assemble_run(
            selections=sels,
            narration_path=tmp_path / "narration.mp3",
            output_path=tmp_path / "out.mp4",
            work_dir=tmp_path / "work",
            downloader=fake_dl,
            target_w=1920,
            target_h=1080,
            target_fps=30,
            clip_audio_db=-20.0,
        )
        # 2 segment encodes + 1 final mux = 3 ffmpeg calls
        assert mock_run.call_count == 3


def test_assemble_run_handles_image_montage(tmp_path):
    selections = [
        Selection(
            idx=1,
            source="pexels",
            url="https://x/v1.mp4",
            kind="video",
            speed_factor=1.0,
        ),
        Selection(
            idx=2,
            source="",
            url="",
            kind="image_montage",
            montage_urls=[
                "https://x/a.jpg",
                "https://x/b.jpg",
                "https://x/c.jpg",
            ],
            montage_sources=["pixabay", "pixabay", "wikimedia"],
        ),
    ]

    fetched: list[tuple[str, str]] = []
    fake_dl = MagicMock()

    def _fetch(url, kind):
        fetched.append((url, kind))
        p = tmp_path / f"dl_{len(fetched)}.bin"
        p.write_bytes(b"")
        return p

    fake_dl.fetch_sync = MagicMock(side_effect=_fetch)

    narration = tmp_path / "n.mp3"
    narration.write_bytes(b"")
    out = tmp_path / "out.mp4"
    work_dir = tmp_path / "work"

    with patch("avtv.assembler.run_ffmpeg") as mock_run:
        assemble_run(
            selections=selections,
            narration_path=narration,
            output_path=out,
            work_dir=work_dir,
            downloader=fake_dl,
            target_w=1920,
            target_h=1080,
            target_fps=30,
            clip_audio_db=-20.0,
        )

    # Block 1 fetches the video; block 2 fetches all 3 montage images.
    assert fetched[0] == ("https://x/v1.mp4", "video")
    montage_fetches = {(u, k) for u, k in fetched[1:]}
    assert montage_fetches == {
        ("https://x/a.jpg", "image"),
        ("https://x/b.jpg", "image"),
        ("https://x/c.jpg", "image"),
    }

    # ffmpeg called once per segment (2) + once for the final mux = 3
    assert mock_run.call_count == 3
    montage_argv = mock_run.call_args_list[1].args[0]
    cmd = " ".join(montage_argv)
    assert "concat=n=3:v=1:a=0" in cmd
    assert "-t 8.0" in cmd
