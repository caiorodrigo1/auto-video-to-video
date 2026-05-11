import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from avtv.models import Selection

if TYPE_CHECKING:
    from avtv.downloader import Downloader, Kind

# Callback invoked after each segment finishes (or is reused from cache).
# Signature: (current_index, total) — both 1-based for display.
ProgressCallback = Callable[[int, int], None]

Strategy = Literal["speed", "loop", "trim", "ken_burns", "image_montage", "continuation"]
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0


@dataclass
class SegmentPlan:
    strategy: Strategy
    speed_factor: float | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    use_clip_audio: bool = True


def plan_segment(sel: Selection) -> SegmentPlan:
    if sel.kind == "continuation":
        return SegmentPlan(strategy="continuation", use_clip_audio=False)
    if sel.kind == "image_montage":
        return SegmentPlan(strategy="image_montage", use_clip_audio=False)
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
    if plan.strategy == "image_montage":
        raise ValueError(
            "image_montage uses build_montage_args (multiple inputs), not build_segment_args"
        )
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
    # Linear zoom from 1.0 to 1.15 across total_frames.
    # Render zoompan at 2x target then scale down to kill sub-pixel jitter
    # (zoompan rounds per-frame x/y to integer pixels; the downscale averages
    # the rounding error and produces smooth motion).
    zoompan = (
        f"setsar=1,"
        f"zoompan=z='min(zoom+0.0007,1.15)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={target_w * 2}x{target_h * 2}:fps={target_fps},"
        f"scale={target_w}:{target_h}"
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


def build_montage_args(
    image_paths: list[str],
    output_path: str,
    target_w: int,
    target_h: int,
    target_fps: int,
) -> list[str]:
    """Render N images (1 <= N <= image_montage_max) as one BLOCK_DURATION-second
    segment: each image gets BLOCK_DURATION/N seconds with Ken Burns, then
    concatenated. Caller passes already-downloaded local image paths.
    """
    n = len(image_paths)
    if n == 0:
        raise ValueError("build_montage_args requires at least one image")
    total_frames = int(BLOCK_DURATION * target_fps)
    frames_per = total_frames // n
    zoom_step = 0.15 / max(frames_per, 1)

    filter_parts: list[str] = []
    for i in range(n):
        # Render zoompan at 2x target then scale down to kill sub-pixel
        # jitter (zoompan's per-frame integer-pixel rounding produces visible
        # stutter near zoom=1.0; the downscale averages it out).
        filter_parts.append(
            f"[{i}:v]setsar=1,"
            f"zoompan=z='min(zoom+{zoom_step:.5f},1.15)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames_per}:s={target_w * 2}x{target_h * 2}:fps={target_fps},"
            f"scale={target_w}:{target_h}"
            f"[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={n}:v=1:a=0[out]"

    args = ["-y"]
    for p in image_paths:
        args += ["-loop", "1", "-i", p]
    args += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-an",
        "-t",
        f"{BLOCK_DURATION}",
        "-f",
        "mpegts",
        output_path,
    ]
    return args


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
    """Concat segments and mux with narration audio.

    NOTE: clip-audio mixing was dropped because per-segment streams are
    heterogeneous (image/loop/extreme-speed segments have no audio track),
    which breaks ffmpeg's concat-demuxer audio output. Only narration is
    used as the final audio. Clip audio at -20dB can be added back once
    every segment has a uniform (real or silent) audio stream.
    """
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
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
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
    on_segment: ProgressCallback | None = None,
    on_mux: Callable[[], None] | None = None,
) -> Path:
    """Assemble segments into the final MP4.

    `on_segment(i, total)` fires once per selection (after that segment is
    encoded or reused from cache); `i` and `total` are 1-based.
    `on_mux()` fires once, right before the final concat+mux ffmpeg call.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    validate_continuations(selections)

    segment_paths: list[Path] = []
    last_segment: Path | None = None
    total = len(selections)

    for i, sel in enumerate(selections, start=1):
        if sel.kind == "continuation":
            # Continuation reuses the prior segment file as-is. No re-fetch,
            # no re-encode — just reference the same .ts in the concat list.
            if last_segment is None:
                raise FFmpegError(
                    f"block {sel.idx}: continuation has no prior segment"
                )
            segment_paths.append(last_segment)
            if on_segment:
                on_segment(i, total)
            continue

        seg_out = work_dir / f"segment_{sel.idx:04d}.ts"
        if seg_out.exists() and seg_out.stat().st_size > 0:
            # Segment already encoded — reuse (re-runs after crash are cheap).
            segment_paths.append(seg_out)
            last_segment = seg_out
            if on_segment:
                on_segment(i, total)
            continue

        if sel.kind == "image_montage":
            image_paths = [
                downloader.fetch_sync(u, kind="image")
                for u in sel.montage_urls
            ]
            args = build_montage_args(
                image_paths=[str(p) for p in image_paths],
                output_path=str(seg_out),
                target_w=target_w,
                target_h=target_h,
                target_fps=target_fps,
            )
            run_ffmpeg(args)
            segment_paths.append(seg_out)
            last_segment = seg_out
            if on_segment:
                on_segment(i, total)
            continue

        input_path = _fetch_for(downloader, sel)
        plan = plan_segment(sel)

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
        if on_segment:
            on_segment(i, total)

    list_path = work_dir / "concat.txt"
    # Use absolute, forward-slash paths so the concat demuxer resolves them
    # correctly regardless of the cwd, the concat list's own directory, or
    # the host OS (Windows builds of ffmpeg dislike backslashes inside the
    # 'file ...' lines of the concat manifest).
    list_path.write_text(
        build_concat_demuxer_file([p.resolve().as_posix() for p in segment_paths]),
        encoding="utf-8",
    )

    if on_mux:
        on_mux()
    mux_args = build_final_mux_args(
        concat_list_path=str(list_path),
        narration_path=str(narration_path),
        output_path=str(output_path),
    )
    run_ffmpeg(mux_args)
    return output_path
