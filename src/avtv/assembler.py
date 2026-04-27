from dataclasses import dataclass
from typing import Literal

from avtv.models import Selection

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
            input_path, output_path, plan, target_w, target_h, target_fps,
            clip_audio_db,
        )
    if plan.strategy == "speed":
        return _speed_args(
            input_path, output_path, plan, target_w, target_h, target_fps,
            clip_audio_db,
        )
    if plan.strategy == "ken_burns":
        return _ken_burns_args(
            input_path, output_path, target_w, target_h, target_fps,
        )
    raise ValueError(f"build_segment_args: unsupported strategy {plan.strategy}")


def _scale_pad_filter(target_w: int, target_h: int) -> str:
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _speed_args(
    input_path: str, output_path: str, plan: SegmentPlan,
    target_w: int, target_h: int, target_fps: int, clip_audio_db: float,
) -> list[str]:
    factor = plan.speed_factor or 1.0
    setpts_factor = 1.0 / factor  # ffmpeg uses inverse for setpts
    vfilter = (
        f"setpts={setpts_factor:.4f}*PTS,{_scale_pad_filter(target_w, target_h)},"
        f"fps={target_fps}"
    )
    args = [
        "-y", "-i", input_path,
        "-vf", vfilter,
        "-t", f"{BLOCK_DURATION}",
        "-c:v", "libx264", "-preset", "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af", f"atempo={factor:.4f},volume={clip_audio_db}dB",
            "-c:a", "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _trim_args(
    input_path: str, output_path: str, plan: SegmentPlan,
    target_w: int, target_h: int, target_fps: int, clip_audio_db: float,
) -> list[str]:
    args = [
        "-y",
        "-ss", f"{plan.trim_start}",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast",
    ]
    if plan.use_clip_audio:
        args += [
            "-af", f"volume={clip_audio_db}dB",
            "-c:a", "aac",
        ]
    else:
        args += ["-an"]
    args += ["-f", "mpegts", output_path]
    return args


def _loop_args(
    input_path: str, output_path: str,
    target_w: int, target_h: int, target_fps: int,
) -> list[str]:
    return [
        "-y",
        "-stream_loop", "-1",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", f"{_scale_pad_filter(target_w, target_h)},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        "-f", "mpegts", output_path,
    ]


def _ken_burns_args(
    input_path: str, output_path: str,
    target_w: int, target_h: int, target_fps: int,
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
        "-loop", "1",
        "-i", input_path,
        "-t", f"{BLOCK_DURATION}",
        "-vf", zoompan,
        "-c:v", "libx264", "-preset", "fast",
        "-an",
        "-f", "mpegts", output_path,
    ]
