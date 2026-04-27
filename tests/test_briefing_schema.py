from avtv.briefing.schema import BRIEF_TOOL_SCHEMA, parse_briefs_response
from avtv.models import VisualBrief


def test_schema_has_required_top_level():
    schema = BRIEF_TOOL_SCHEMA
    assert schema["type"] == "object"
    assert "briefs" in schema["properties"]
    assert schema["required"] == ["briefs"]


def test_parse_briefs_response_valid():
    data = {
        "briefs": [
            {
                "idx": 1,
                "query_en": "forest sunlight",
                "fallback_query": "nature",
                "kind": "video",
                "continuity_hint": None,
                "notes": None,
            }
        ]
    }
    briefs = parse_briefs_response(data)
    assert len(briefs) == 1
    assert isinstance(briefs[0], VisualBrief)
    assert briefs[0].query_en == "forest sunlight"
