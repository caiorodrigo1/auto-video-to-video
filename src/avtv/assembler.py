import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from avtv.models import Selection

if TYPE_CHECKING:
    from avtv.downloader import Downloader, Kind

Strategy = Literal["speed", "loop", "trim", "ken_burns", "continuation"]
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0


@dataclass
class SegmentPlan:
    strategy: Strategy
    speed_factor: float | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    use_clip_audio: bool = True


def plan_segment(sel: Selection, source_duration: float | None) -> SegmentPlan:
    if sel.kind == "continuation":
        return SegmentPlan(strategy="continuation", use_clip_audio=False)
    if sel.kind == "image":
        return SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    if sel.loop:
        return SegmentPlan(strategy="loop", use_clip_audio=False)
    if sel.trim_start is not None and sel.trim_end is not None:
        return SegmentPlan(
            strategy="trim",
            trim_start=sel.trim_start,
            trim_end=sel.trim_end,
            use_clip_audio=True,
        )
    if sel.speed_factor is not None:
        in_atempo_range = ATEMPO_MIN <= sel.speed_factor <= ATEMPO_MAX
        return SegmentPlan(
            strategy="speed",
            speed_factor=sel.speed_factor,
            use_clip_audio=in_atempo_range,
        )
    # No transform: clip is exactly target duration
    return SegmentPlan(strategy="speed", speed_factor=1.0, use_clip_audio=True)


BLOCK_DURATION = 8.0


def build_segment_args(
    input_path: str,
    output_path: str,
    plan: SegmentPlan,
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> list[str]:
    """Construct ffmpeg argv (without leading 'ffmpeg') for one segment."""
    if plan.strategy == "loop":
        return _loop_args(input_path, output_path, target_w, target_h, target_fps)
    if plan.strategy == "trim":
        return _trim_args(
            input_path,
            output_path,
            plan,
            target_w,
            target_h,
            target_fps,
            clip_audio_db,
        )
    if plan.strategy == "speed":
        return _speed_args(
            input_path,
            output_path,
            plan,
            target_w,
            target_h,
            target_fps,
            clip_audio_db,
        )
    if plan.strategy == "ken_burns":
        return _ken_burns_args(
            input_path,
            output_path,
            target_w,
            target_h,
            target_fps,
        )
    raise ValueError(f"build_segment_args: unsupported strategy {plan.strategy}")


def _scale_pad_filter(target_w: int, target_h: int) -> str:
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _speed_args(
    input_path: str,
    output_path: str,
    plan: SegmentPlan,
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> list[str]:
    factor = plan.speed_factor or 1.0
    setpts_factor = 1.0 / factor  # ffmpeg uses inverse for setpts
    vfilter = (
        f"setpts={setpts_factor:.4f}*PTS,{_scale_pad_filter(target_w, target_h)},fps={target_fps}"
    )
    args = [
        "-y",
        "-i",
        input_path,
        "-vf",
        vfilter,
        "-t",
        f"{BLOCK_DURATION}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af",
            f"atempo={factor:.4f},volume={clip_audio_db}dB",
            "-c:a",
            "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _trim_args(
    input_path: str,
    output_path: str,
    plan: SegmentPlan,
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> list[str]:
    args = [
        "-y",
        "-ss",
        f"{plan.trim_start}",
        "-i",
        input_path,
        "-t",
        f"{BLOCK_DURATION}",
        "-vf",
        f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af",
            f"volume={clip_audio_db}dB",
            "-c:a",
            "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _loop_args(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
) -> list[str]:
    return [
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        input_path,
        "-t",
        f"{BLOCK_DURATION}",
        "-vf",
        f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        "-f",
        "mpegts",
        output_path,
    ]


def _ken_burns_args(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
) -> list[str]:
    total_frames = int(BLOCK_DURATION * target_fps)
    # Linear zoom from 1.0 to 1.15 across total_frames
    zoompan = (
        f"zoompan=z='min(zoom+0.0007,1.15)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={target_w}x{target_h}:fps={target_fps}"
    )
    return [
        "-y",
        "-loop",
        "1",
        "-i",
        input_path,
        "-t",
        f"{BLOCK_DURATION}",
        "-vf",
        zoompan,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        "-f",
        "mpegts",
        output_path,
    ]


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args: list[str]) -> None:
    full = ["ffmpeg", *args]
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}\ncommand: {' '.join(full)}"
        )


def validate_continuations(selections: list[Selection]) -> None:
    """Reject runs where block 1 is a continuation (no prior source to extend)."""
    if not selections:
        return
    if selections[0].kind == "continuation":
        raise ValueError(f"block {selections[0].idx}: first block cannot be continuation")


def build_concat_demuxer_file(segment_paths: list[str]) -> str:
    return "\n".join(f"file '{p}'" for p in segment_paths) + "\n"


def build_final_mux_args(
    concat_list_path: str,
    narration_path: str,
    output_path: str,
) -> list[str]:
    return [
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path,
        "-i",
        narration_path,
        "-filter_complex",
        "[0:a]anull[clip];[clip][1:a]amix=inputs=2:duration=longest[out]",
        "-map",
        "0:v",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]


def probe_duration(path: Path) -> float:
    """Probe media duration via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr}")
    fmt = json.loads(result.stdout).get("format", {})
    return float(fmt.get("duration", 0.0))


def _fetch_for(downloader: "Downloader", sel: Selection) -> Path:
    media_kind: Kind = "video" if sel.kind == "video" else "image"
    return downloader.fetch_sync(sel.url, kind=media_kind)


def assemble_run(
    selections: list[Selection],
    narration_path: Path,
    output_path: Path,
    work_dir: Path,
    downloader: "Downloader",
    target_w: int,
    target_h: int,
    target_fps: int,
    clip_audio_db: float,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    validate_continuations(selections)

    segment_paths: list[Path] = []
    last_segment: Path | None = None

    for sel in selections:
        if sel.kind == "continuation":
            # Continuation reuses the prior segment file as-is. No re-fetch,
            # no re-encode — just reference the same .ts in the concat list.
            assert last_segment is not None  # validate_continuations guarantees
            segment_paths.append(last_segment)
            continue

        input_path = _fetch_for(downloader, sel)
        duration = probe_duration(input_path) if sel.kind == "video" else None
        plan = plan_segment(sel, duration)

        seg_out = work_dir / f"segment_{sel.idx:04d}.ts"
        args = build_segment_args(
            input_path=str(input_path),
            output_path=str(seg_out),
            plan=plan,
            target_w=target_w,
            target_h=target_h,
            target_fps=target_fps,
            clip_audio_db=clip_audio_db,
        )
        run_ffmpeg(args)
        segment_paths.append(seg_out)
        last_segment = seg_out

    list_path = work_dir / "concat.txt"
    list_path.write_text(build_concat_demuxer_file([str(p) for p in segment_paths]))

    mux_args = build_final_mux_args(
        concat_list_path=str(list_path),
        narration_path=str(narration_path),
        output_path=str(output_path),
    )
    run_ffmpeg(mux_args)
    return output_path
