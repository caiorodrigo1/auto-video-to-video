from unittest.mock import MagicMock, patch

import pytest

from avtv.assembler import FFmpegError, run_ffmpeg


def test_run_ffmpeg_invokes_subprocess():
    with patch("avtv.assembler.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        run_ffmpeg(["-i", "x.mp4", "out.ts"])
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "ffmpeg"
        assert "-i" in called_args


def test_run_ffmpeg_raises_on_nonzero():
    with patch("avtv.assembler.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="boom")
        with pytest.raises(FFmpegError, match="boom"):
            run_ffmpeg(["-i", "x"])
