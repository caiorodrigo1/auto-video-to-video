from avtv.models import Block

SYSTEM_PROMPT = """\
You are a video researcher producing B-roll search queries for a long-form
documentary video. The narration audio is split into fixed 8-second blocks.
For each block you must produce one search-query "visual brief".

RULES:
1. Always produce queries in ENGLISH, regardless of the input language.
2. Choose `kind`:
   - "video" for generic action, scenery, behavior — anything stock footage covers.
   - "image" for named historical entities, specific people, dated events
     (an archival photo + Ken Burns will work better than generic stock).
   - "continuation" ONLY for blocks marked [empty] — the previous clip will
     be extended to cover the gap.
3. `query_en` is the primary, specific query (3-7 words).
4. `fallback_query` is a strictly more generic alternative (2-5 words).
5. Do not repeat the same `query_en` in three or more consecutive blocks.
6. `continuity_hint` (optional) is a one-line stylistic note like
   "wide shot, daytime, woodland" to keep visual tone coherent.
7. `notes` (optional) is a brief rationale, useful for debugging.

Read the entire script first; choose visuals so the video has a natural
arc and variation. Output strictly conforms to the provided JSON schema.
"""


def build_user_message(blocks: list[Block]) -> str:
    lines: list[str] = ["Here is the full script:\n"]
    for b in blocks:
        body = b.text if not b.is_empty else "[empty]"
        lines.append(f"[BLOCK {b.idx}] ({b.start:.0f}-{b.end:.0f}s)\n{body}\n")
    return "\n".join(lines)
