import pytest
import respx
from httpx import Response

from avtv.search.unsplash import UnsplashAdapter

RESP = {
    "results": [
        {
            "id": "abc",
            "width": 4000, "height": 3000,
            "urls": {"full": "https://img.unsplash.com/abc"},
            "user": {"name": "Eve"},
        }
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_unsplash_image_only():
    respx.get("https://api.unsplash.com/search/photos").mock(
        return_value=Response(200, json=RESP)
    )
    a = UnsplashAdapter(api_key="k")
    results = await a.search("forest", kind="image")
    assert len(results) == 1
    assert results[0].source == "unsplash"
    assert results[0].url == "https://img.unsplash.com/abc"
    assert results[0].kind == "image"


@pytest.mark.asyncio
async def test_unsplash_rejects_video():
    a = UnsplashAdapter(api_key="k")
    results = await a.search("forest", kind="video")
    assert results == []
