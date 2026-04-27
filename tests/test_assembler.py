from avtv.assembler import SegmentPlan, build_segment_args, plan_segment
from avtv.models import Selection


def _sel(**kw):
    base = dict(
        idx=1, source="pexels", url="u", kind="video",
        trim_start=None, trim_end=None, speed_factor=None,
        loop=False, attribution=None,
    )
    base.update(kw)
    return Selection(**base)


def test_plan_speed_match_normal():
    sel = _sel(speed_factor=0.9)
    plan = plan_segment(sel, source_duration=7.2)
    assert plan.strategy == "speed"
    assert plan.speed_factor == 0.9
    assert plan.use_clip_audio is True


def test_plan_speed_extreme_drops_audio():
    sel = _sel(speed_factor=2.5)
    plan = plan_segment(sel, source_duration=20.0)
    assert plan.strategy == "speed"
    assert plan.use_clip_audio is False  # outside atempo range


def test_plan_loop_for_short_clip():
    sel = _sel(loop=True)
    plan = plan_segment(sel, source_duration=3.0)
    assert plan.strategy == "loop"
    assert plan.use_clip_audio is False


def test_plan_trim_for_long_clip():
    sel = _sel(trim_start=10.0, trim_end=18.0)
    plan = plan_segment(sel, source_duration=30.0)
    assert plan.strategy == "trim"
    assert plan.trim_start == 10.0
    assert plan.trim_end == 18.0
    assert plan.use_clip_audio is True


def test_plan_image_ken_burns():
    sel = _sel(kind="image", speed_factor=None)
    plan = plan_segment(sel, source_duration=None)
    assert plan.strategy == "ken_burns"
    assert plan.use_clip_audio is False


def test_plan_continuation():
    sel = _sel(kind="continuation", url="")
    plan = plan_segment(sel, source_duration=None)
    assert plan.strategy == "continuation"


def test_build_args_speed_with_audio():
    plan = SegmentPlan(strategy="speed", speed_factor=0.9, use_clip_audio=True)
    args = build_segment_args(
        input_path="/in/clip.mp4", output_path="/out/seg.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-i /in/clip.mp4" in cmd
    assert "/out/seg.ts" in cmd
    # setpts factor for slowing down to fit 8s: 1/0.9
    assert "setpts=" in cmd
    assert "atempo=0.9" in cmd
    assert "volume=-20.0dB" in cmd
    assert "scale=" in cmd
    assert "1920" in cmd and "1080" in cmd


def test_build_args_speed_no_audio():
    plan = SegmentPlan(strategy="speed", speed_factor=3.0, use_clip_audio=False)
    args = build_segment_args(
        input_path="/in.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-an" in cmd  # audio dropped


def test_build_args_trim():
    plan = SegmentPlan(
        strategy="trim", trim_start=10.0, trim_end=18.0, use_clip_audio=True,
    )
    args = build_segment_args(
        input_path="/i.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-ss 10.0" in cmd
    assert "-t 8.0" in cmd


def test_build_args_loop_short_clip():
    plan = SegmentPlan(strategy="loop", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.mp4", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-stream_loop" in cmd
    assert "-t 8.0" in cmd


def test_build_args_ken_burns_image():
    plan = SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.jpg", output_path="/o.ts", plan=plan,
        target_w=1920, target_h=1080, target_fps=30, clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "zoompan" in cmd
    assert "-t 8.0" in cmd
