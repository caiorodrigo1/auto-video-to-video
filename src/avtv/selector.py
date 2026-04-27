from avtv.config import Settings
from avtv.models import Candidate

SOURCE_PREFERENCE = {
    "pexels": 0.30,
    "pixabay": 0.25,
    "archive_org": 0.20,
    "wikimedia": 0.15,
    "unsplash": 0.20,
}


def _aspect_score(width: int, height: int) -> float:
    if height <= 0:
        return 0.0
    ratio = width / height
    if ratio >= 1.7:
        return 1.0
    if ratio >= 1.4:
        return 0.5
    return 0.0


def _duration_score(duration: float | None, target: float = 8.0) -> float:
    if duration is None:
        # images get neutral mid score on this axis
        return 0.7
    factor = duration / target
    if 0.85 <= factor <= 1.2:
        return 1.0
    if 0.5 <= factor < 0.85 or 1.2 < factor <= 2.0:
        return 0.7
    if 0.25 <= factor < 0.5 or 2.0 < factor <= 3.0:
        return 0.4
    return 0.1


def _resolution_score(height: int) -> float:
    if height >= 1080:
        return 1.0
    if height >= 720:
        return 0.6
    return 0.3


def score_candidate(c: Candidate, settings: Settings) -> float:
    aspect = _aspect_score(c.width, c.height)
    duration = _duration_score(c.duration, target=settings.target_block_duration)
    resolution = _resolution_score(c.height)
    source_pref = SOURCE_PREFERENCE.get(c.source, 0.1)
    return (
        settings.weight_aspect * aspect
        + settings.weight_duration * duration
        + settings.weight_resolution * resolution
        + settings.weight_source * source_pref
    )
