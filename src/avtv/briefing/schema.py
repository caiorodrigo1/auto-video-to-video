from typing import Any

from avtv.models import VisualBrief

BRIEF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "briefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "query_en": {"type": "string"},
                    "fallback_query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["video", "image", "continuation"],
                    },
                    "continuity_hint": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": [
                    "idx", "query_en", "fallback_query", "kind",
                    "continuity_hint", "notes",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["briefs"],
    "additionalProperties": False,
}


def parse_briefs_response(data: dict[str, Any]) -> list[VisualBrief]:
    return [VisualBrief.model_validate(item) for item in data["briefs"]]
