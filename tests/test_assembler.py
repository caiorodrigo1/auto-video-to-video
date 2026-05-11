from avtv.assembler import (
    SegmentPlan,
    build_concat_demuxer_file,
    build_final_mux_args,
    build_segment_args,
    plan_segment,
    validate_continuations,
)
from avtv.models import Selection


def _sel(**kw):
    base = dict(
        idx=1,
        source="pexels",
        url="u",
        kind="video",
        trim_start=None,
        trim_end=None,
        speed_factor=None,
        loop=False,
        attribution=None,
    )
    base.update(kw)
    return Selection(**base)


def test_plan_speed_match_normal():
    sel = _sel(speed_factor=0.9)
    plan = plan_segment(sel)
    assert plan.strategy == "speed"
    assert plan.speed_factor == 0.9
    assert plan.use_clip_audio is True


def test_plan_speed_extreme_drops_audio():
    sel = _sel(speed_factor=2.5)
    plan = plan_segment(sel)
    assert plan.strategy == "speed"
    assert plan.use_clip_audio is False  # outside atempo range


def test_plan_loop_for_short_clip():
    sel = _sel(loop=True)
    plan = plan_segment(sel)
    assert plan.strategy == "loop"
    assert plan.use_clip_audio is False


def test_plan_trim_for_long_clip():
    sel = _sel(trim_start=10.0, trim_end=18.0)
    plan = plan_segment(sel)
    assert plan.strategy == "trim"
    assert plan.trim_start == 10.0
    assert plan.trim_end == 18.0
    assert plan.use_clip_audio is True


def test_plan_image_ken_burns():
    sel = _sel(kind="image", speed_factor=None)
    plan = plan_segment(sel)
    assert plan.strategy == "ken_burns"
    assert plan.use_clip_audio is False


def test_plan_continuation():
    sel = _sel(kind="continuation", url="")
    plan = plan_segment(sel)
    assert plan.strategy == "continuation"


def test_build_args_speed_with_audio():
    plan = SegmentPlan(strategy="speed", speed_factor=0.9, use_clip_audio=True)
    args = build_segment_args(
        input_path="/in/clip.mp4",
        output_path="/out/seg.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
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
        input_path="/in.mp4",
        output_path="/o.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-an" in cmd  # audio dropped


def test_build_args_trim():
    plan = SegmentPlan(
        strategy="trim",
        trim_start=10.0,
        trim_end=18.0,
        use_clip_audio=True,
    )
    args = build_segment_args(
        input_path="/i.mp4",
        output_path="/o.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-ss 10.0" in cmd
    assert "-t 8.0" in cmd


def test_build_args_loop_short_clip():
    plan = SegmentPlan(strategy="loop", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.mp4",
        output_path="/o.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "-stream_loop" in cmd
    assert "-t 8.0" in cmd


def test_build_args_ken_burns_image():
    plan = SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.jpg",
        output_path="/o.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "zoompan" in cmd
    assert "-t 8.0" in cmd


def test_concat_demuxer_file_format():
    out = build_concat_demuxer_file(["/a.ts", "/b.ts"])
    assert "file '/a.ts'" in out
    assert "file '/b.ts'" in out
    assert out.endswith("\n")


def test_final_mux_args_includes_concat_and_codecs():
    args = build_final_mux_args(
        concat_list_path="/c.txt",
        narration_path="/n.mp3",
        output_path="/o.mp4",
    )
    cmd = " ".join(args)
    assert "-f concat" in cmd
    assert "/c.txt" in cmd
    assert "/n.mp3" in cmd
    # Narration audio is mapped directly (no mix); clip audio dropped for MVP.
    assert "-map 0:v" in cmd
    assert "-map 1:a" in cmd
    assert "-c:v copy" in cmd
    assert "-shortest" in cmd
    assert "/o.mp4" in cmd


def test_validate_continuations_rejects_first_block():
    bad = [_sel(idx=1, kind="continuation", url="")]
    import pytest

    with pytest.raises(ValueError, match="first block"):
        validate_continuations(bad)


def test_validate_continuations_accepts_normal_run():
    ok = [
        _sel(idx=1, url="A"),
        _sel(idx=2, kind="continuation", url=""),
        _sel(idx=3, url="B"),
    ]
    validate_continuations(ok)  # no raise


def test_plan_image_montage():
    sel = Selection(
        idx=4, source="", url="", kind="image_montage",
        montage_urls=["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg"],
        montage_sources=["pixabay", "pixabay", "wikimedia"],
    )
    plan = plan_segment(sel)
    assert plan.strategy == "image_montage"
    assert plan.use_clip_audio is False


def test_build_args_image_montage_three_images():
    from avtv.assembler import build_montage_args
    args = build_montage_args(
        image_paths=["/a.jpg", "/b.jpg", "/c.jpg"],
        output_path="/out.ts",
        target_w=1920,
        target_h=1080,
        target_fps=30,
    )
    cmd = " ".join(args)
    # All three inputs are passed with -loop 1
    assert cmd.count("-loop 1") == 3
    assert "/a.jpg" in cmd and "/b.jpg" in cmd and "/c.jpg" in cmd
    # filter_complex with 3 zoompan + concat
    assert "zoompan" in cmd
    assert "concat=n=3:v=1:a=0" in cmd
    assert "-t 8.0" in cmd
    assert "-an" in cmd  # no clip audio
    # frames per image: 30fps * 8s / 3 = 80
    assert "d=80" in cmd


def test_build_args_image_montage_two_images_split_evenly():
    from avtv.assembler import build_montage_args
    args = build_montage_args(
        image_paths=["/a.jpg", "/b.jpg"],
        output_path="/o.ts",
        target_w=1920,
        target_h=1080,
        target_fps=30,
    )
    cmd = " ".join(args)
    assert "concat=n=2:v=1:a=0" in cmd
    # 30fps * 8s / 2 = 120 frames each
    assert "d=120" in cmd


def test_build_args_image_montage_uses_internal_upscale_for_smooth_zoom():
    """Avoids ffmpeg zoompan sub-pixel jitter by rendering at 2x and downscaling."""
    from avtv.assembler import build_montage_args
    args = build_montage_args(
        image_paths=["/a.jpg", "/b.jpg", "/c.jpg"],
        output_path="/out.ts",
        target_w=1920,
        target_h=1080,
        target_fps=30,
    )
    cmd = " ".join(args)
    # zoompan output is 2x target (3840x2160), then scaled down to 1920x1080
    assert "s=3840x2160" in cmd
    assert "scale=1920:1080" in cmd
    # setsar=1 applied before zoompan
    assert "setsar=1" in cmd


def test_build_args_ken_burns_uses_internal_upscale_for_smooth_zoom():
    plan = SegmentPlan(strategy="ken_burns", use_clip_audio=False)
    args = build_segment_args(
        input_path="/i.jpg",
        output_path="/o.ts",
        plan=plan,
        target_w=1920,
        target_h=1080,
        target_fps=30,
        clip_audio_db=-20.0,
    )
    cmd = " ".join(args)
    assert "s=3840x2160" in cmd
    assert "scale=1920:1080" in cmd
    assert "setsar=1" in cmd
