import re

from avtv.models import Block

BLOCK_DURATION = 8.0
SEPARATOR = "-" * 60

_HEADER_RE = re.compile(
    r"^PROMPT\s+(\d+)\s*\|\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*$"
)


class ParseError(ValueError):
    pass


def _to_seconds(mm: str, ss: str) -> float:
    return int(mm) * 60 + int(ss)


def parse_script(text: str) -> list[Block]:
    """Parse DOTTI SYNC text into a list of Block."""
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _HEADER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        idx = int(m.group(1))
        start = _to_seconds(m.group(2), m.group(3))
        end = _to_seconds(m.group(4), m.group(5))
        if abs((end - start) - BLOCK_DURATION) > 0.01:
            raise ParseError(
                f"block {idx}: duration {end - start}s != {BLOCK_DURATION}s"
            )
        # body until separator
        body_lines: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != SEPARATOR:
            body_lines.append(lines[i])
            i += 1
        # consume separator
        if i < len(lines):
            i += 1
        text_body = "\n".join(body_lines).strip("\n")
        blocks.append(Block.from_text(idx=idx, start=start, end=end, text=text_body))

    if not blocks:
        raise ParseError("no blocks found")

    for expected, b in enumerate(blocks, start=1):
        if b.idx != expected:
            raise ParseError(
                f"non-sequential idx: expected {expected}, got {b.idx}"
            )

    return blocks
