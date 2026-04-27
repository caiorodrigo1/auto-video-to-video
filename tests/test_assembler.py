from avtv.assembler import plan_segment
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
