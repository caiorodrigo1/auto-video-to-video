import pytest
import respx
from httpx import Response

from avtv.search.archive_org import ArchiveOrgAdapter

SEARCH_RESPONSE = {
    "response": {
        "docs": [{"identifier": "test_item_1", "title": "Test Documentary", "creator": "Public"}]
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_archive_search_returns_video():
    respx.get("https://archive.org/advancedsearch.php").mock(
        return_value=Response(200, json=SEARCH_RESPONSE)
    )

    # Mock metadata call
    metadata = {
        "files": [
            {
                "name": "test.mp4",
                "format": "h.264",
                "length": "30.5",
                "width": "1920",
                "height": "1080",
            },
            {"name": "test.gif", "format": "Animated GIF"},
        ]
    }
    respx.get("https://archive.org/metadata/test_item_1").mock(
        return_value=Response(200, json=metadata)
    )

    a = ArchiveOrgAdapter()
    results = await a.search("history", kind="video", limit=1)
    assert len(results) == 1
    assert results[0].source == "archive_org"
    assert "test_item_1" in results[0].url
    assert results[0].duration == 30.5
    assert results[0].width == 1920
