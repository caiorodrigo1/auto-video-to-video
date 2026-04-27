import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from avtv.search.pixabay import PixabayAdapter

VIDEO_FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "pixabay_video_response.json").read_text()
)
IMAGE_FIX = json.loads(
    (Path(__file__).parent / "fixtures" / "pixabay_image_response.json").read_text()
)


@pytest.mark.asyncio
@respx.mock
async def test_pixabay_video_search():
    respx.get("https://pixabay.com/api/videos/").mock(
        return_value=Response(200, json=VIDEO_FIX)
    )
    a = PixabayAdapter(api_key="k")
    results = await a.search("nature", kind="video")
    assert len(results) == 1
    assert results[0].source == "pixabay"
    assert results[0].url == "https://pixabay.com/v1-large.mp4"
    assert results[0].duration == 10
    assert results[0].kind == "video"


@pytest.mark.asyncio
@respx.mock
async def test_pixabay_image_search():
    respx.get("https://pixabay.com/api/").mock(
        return_value=Response(200, json=IMAGE_FIX)
    )
    a = PixabayAdapter(api_key="k")
    results = await a.search("forest", kind="image")
    assert len(results) == 1
    assert results[0].kind == "image"
    assert results[0].url == "https://pixabay.com/img-large.jpg"
    assert results[0].width == 4000
    assert results[0].duration is None
