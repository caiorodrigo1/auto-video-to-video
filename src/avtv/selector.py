from collections import Counter
from collections.abc import Awaitable, Callable

from avtv.config import Settings
from avtv.models import Candidate, Selection, VisualBrief

ImageSearch = Callable[[VisualBrief], Awaitable[list[Candidate]]]
ProgressCallback = Callable[[int, int], None]

SOURCE_PREFERENCE = {
    "pexels": 1.00,
    "pixabay": 0.30,
    "archive_org": 0.30,
    "wikimedia": 0.30,
    "unsplash": 0.30,
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


def _selection_from_video(brief: VisualBrief, c: Candidate) -> Selection:
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
        kind=c.kind,  # "video" or "image"
        trim_start=trim_start,
        trim_end=trim_end,
        speed_factor=speed_factor,
        loop=loop,
        attribution=c.attribution,
    )


def _continuation(brief: VisualBrief) -> Selection:
    return Selection(idx=brief.idx, source="", url="", kind="continuation")


def _montage(brief: VisualBrief, images: list[Candidate]) -> Selection:
    """Build an image_montage selection from N>=2 image candidates.

    Caller guarantees images are already deduped against the global used set
    and ranked best-first. Caller must call _selection_from_video for the
    single-image case (we emit kind="image", not "image_montage").
    """
    return Selection(
        idx=brief.idx,
        source="",
        url="",
        kind="image_montage",
        montage_urls=[c.url for c in images],
        montage_sources=[c.source for c in images],
    )


def _adjusted_score(
    c: Candidate,
    settings: Settings,
    used_attribs: Counter[str],
) -> float:
    """`score_candidate` minus attribution-repeat penalty.

    Each prior selection that shared the same `attribution` (photographer)
    drops the score by `settings.attribution_penalty`. Keeps the ranking
    soft: if no fresh-author candidate exists, repeats are still picked,
    just deprioritized.
    """
    base = score_candidate(c, settings)
    if not c.attribution:
        return base
    return base - used_attribs[c.attribution] * settings.attribution_penalty


def _pick_unused_video(
    cands: list[Candidate],
    used: set[str],
    used_attribs: Counter[str],
    settings: Settings,
) -> Candidate | None:
    ranked = sorted(
        cands,
        key=lambda c: _adjusted_score(c, settings, used_attribs),
        reverse=True,
    )
    for c in ranked:
        if c.url not in used:
            return c
    return None


def _pick_unused_images(
    cands: list[Candidate],
    used: set[str],
    used_attribs: Counter[str],
    settings: Settings,
    limit: int,
) -> list[Candidate]:
    """Pick top-N unused image candidates, applying attribution penalty
    INCREMENTALLY so a montage doesn't end up with three photos from the
    same photographer (which would otherwise tie on attribute count)."""
    images = [c for c in cands if c.kind == "image" and c.url not in used]
    if not images:
        return []
    # Greedy selection: pick best by adjusted score, count its attribution,
    # then re-rank the rest with the updated counts. This actively spreads
    # the montage across different authors.
    picked: list[Candidate] = []
    running = Counter(used_attribs)
    remaining = list(images)
    while remaining and len(picked) < limit:
        remaining.sort(
            key=lambda c: _adjusted_score(c, settings, running),
            reverse=True,
        )
        chosen = remaining.pop(0)
        picked.append(chosen)
        if chosen.attribution:
            running[chosen.attribution] += 1
    return picked


async def select_per_block(
    briefs: list[VisualBrief],
    candidates: dict[int, list[Candidate]],
    settings: Settings,
    image_search: ImageSearch | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[Selection]:
    """Pick exactly one Selection per brief.

    Rules:
    - No URL appears twice across the returned selections (montage URLs included).
    - Continuation briefs become image_montage when image_search yields >=2
      unused images, image (single Ken Burns) for exactly 1, continuation
      (extend previous) for 0.
    - Video/image briefs prefer an unused candidate; if all candidates are
      used, fall back to image_montage via image_search; if that yields 0,
      repeat the least-recently-used candidate (last-resort).
    - Block 1 with no usable visuals raises SelectorError (it cannot extend).

    `on_progress(completed, total)` fires after each brief is resolved
    (1-based), useful for driving a UI progress bar.
    """
    selections: list[Selection] = []
    used: set[str] = set()
    used_attribs: Counter[str] = Counter()
    last_idx_used: dict[str, int] = {}  # for least-recently-used fallback
    total = len(briefs)

    for pos, brief in enumerate(briefs):
        if brief.kind == "continuation":
            sel = await _resolve_continuation(
                brief, used, used_attribs, settings, image_search
            )
            if sel.kind == "continuation" and pos == 0:
                raise SelectorError(
                    f"block {brief.idx} is the first block and has no visuals "
                    f"(continuation/image_search both empty)"
                )
        else:
            sel = await _resolve_visual(
                brief,
                candidates.get(brief.idx, []),
                used,
                used_attribs,
                last_idx_used,
                settings,
                image_search,
                pos,
            )

        # Mark all URLs the new selection introduces as used.
        for u in _selection_urls(sel):
            used.add(u)
            last_idx_used[u] = pos
        # Mark the attribution(s) so subsequent picks spread across authors.
        for a in _selection_attributions(sel):
            used_attribs[a] += 1

        selections.append(sel)

        if on_progress is not None:
            on_progress(pos + 1, total)

    return selections


def _selection_urls(sel: Selection) -> list[str]:
    if sel.kind == "image_montage":
        return list(sel.montage_urls)
    if sel.url:
        return [sel.url]
    return []


def _selection_attributions(sel: Selection) -> list[str]:
    if sel.kind == "image_montage":
        # montage_sources holds the SOURCE (pixabay etc), not the photographer.
        # We don't track per-image attribution in montage selections (the
        # downstream consumer doesn't need it), so skip — the greedy
        # _pick_unused_images already spread the montage across authors.
        return []
    if sel.attribution:
        return [sel.attribution]
    return []


async def _try_image_fallback(
    brief: VisualBrief,
    used: set[str],
    used_attribs: Counter[str],
    settings: Settings,
    image_search: ImageSearch,
) -> Selection | None:
    """Run image_search and convert results into a montage / single image / None.
    Returns None when fewer than 1 unused image came back."""
    images = await image_search(brief)
    unused = _pick_unused_images(
        images, used, used_attribs, settings, limit=settings.image_montage_max
    )
    if len(unused) >= 2:
        return _montage(brief, unused)
    if len(unused) == 1:
        return _selection_from_video(brief, unused[0])
    return None


async def _resolve_continuation(
    brief: VisualBrief,
    used: set[str],
    used_attribs: Counter[str],
    settings: Settings,
    image_search: ImageSearch | None,
) -> Selection:
    if image_search is None:
        return _continuation(brief)
    sel = await _try_image_fallback(brief, used, used_attribs, settings, image_search)
    return sel if sel is not None else _continuation(brief)


async def _resolve_visual(
    brief: VisualBrief,
    cands: list[Candidate],
    used: set[str],
    used_attribs: Counter[str],
    last_idx_used: dict[str, int],
    settings: Settings,
    image_search: ImageSearch | None,
    pos: int,
) -> Selection:
    chosen = _pick_unused_video(cands, used, used_attribs, settings)
    if chosen is not None:
        return _selection_from_video(brief, chosen)

    # Pool exhausted: try image_montage fallback.
    if image_search is not None:
        sel = await _try_image_fallback(
            brief, used, used_attribs, settings, image_search
        )
        if sel is not None:
            return sel

    # Last-resort: repeat the candidate seen the longest ago. If we have no
    # candidates at all and we're not the first block, extend the previous.
    if not cands:
        if pos == 0:
            raise SelectorError(
                f"block {brief.idx} has no candidates and is the first block "
                f"(cannot continue from previous)"
            )
        return _continuation(brief)

    ranked = sorted(cands, key=lambda c: score_candidate(c, settings), reverse=True)
    # Prefer the candidate whose last-used position is smallest (used the longest ago).
    # Invariant: every cand.url is guaranteed to be in last_idx_used here.
    # _pick_unused_video returned None above means every candidate's URL is in
    # `used`, and `used` and `last_idx_used` are updated together in the
    # select_per_block loop, so direct indexing is safe (no .get default needed).
    chosen = min(ranked, key=lambda c: last_idx_used[c.url])
    return _selection_from_video(brief, chosen)
