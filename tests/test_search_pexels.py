import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from avtv.search.pexels import PexelsAdapter

FIX = json.loads((Path(__file__).parent / "fixtures" / "pexels_response.json").read_text())


@pytest.mark.asyncio
@respx.mock
async def test_pexels_search_videos():
    respx.get("https://api.pexels.com/videos/search").mock(
        return_value=Response(200, json=FIX)
    )
    adapter = PexelsAdapter(api_key="testkey")
    results = await adapter.search("forest", limit=2, kind="video")
    assert len(results) == 2
    assert results[0].source == "pexels"
    assert results[0].url == "https://pexels.com/v1.mp4"
    assert results[0].duration == 12
    assert results[0].width == 1920
    assert results[0].attribution == "Alice"
