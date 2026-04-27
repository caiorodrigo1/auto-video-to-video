from avtv.config import Settings
from avtv.models import Candidate, Selection, VisualBrief

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


class SelectorError(RuntimeError):
    pass


def _selection_from(brief: VisualBrief, c: Candidate) -> Selection:
    speed_factor: float | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    loop = False

    if c.kind == "video" and c.duration is not None:
        target = 8.0
        if c.duration < 4.0:
            loop = True
        elif c.duration > 16.0:
            mid = c.duration / 2
            trim_start = max(0.0, mid - target / 2)
            trim_end = trim_start + target
        else:
            speed_factor = c.duration / target

    return Selection(
        idx=brief.idx,
        source=c.source,
        url=c.url,
        kind=c.kind,
        trim_start=trim_start,
        trim_end=trim_end,
        speed_factor=speed_factor,
        loop=loop,
        attribution=c.attribution,
    )


def _continuation(brief: VisualBrief) -> Selection:
    return Selection(
        idx=brief.idx,
        source="",
        url="",
        kind="continuation",
        attribution=None,
    )


def select_per_block(
    briefs: list[VisualBrief],
    candidates: dict[int, list[Candidate]],
    settings: Settings,
) -> list[Selection]:
    selections: list[Selection] = []
    used_recent: list[str] = []  # URLs in ±neighbor_window range

    for brief in briefs:
        if brief.kind == "continuation":
            selections.append(_continuation(brief))
            used_recent.append("")
            continue

        cands = candidates.get(brief.idx, [])

        if not cands:
            if brief.idx == 1:
                raise SelectorError(
                    f"block {brief.idx} has no candidates and is the first block "
                    f"(cannot continue from previous)"
                )
            selections.append(_continuation(brief))
            used_recent.append("")
            continue

        ranked = sorted(cands, key=lambda c: score_candidate(c, settings), reverse=True)

        window = used_recent[-settings.neighbor_window :]
        chosen = next(
            (c for c in ranked if c.url not in window),
            max(
                ranked,
                key=lambda c: next((i for i, u in enumerate(used_recent) if u == c.url), -1),
            ),
        )

        sel = _selection_from(brief, chosen)
        selections.append(sel)
        used_recent.append(chosen.url)

    return selections
