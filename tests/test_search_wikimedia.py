import pytest
import respx
from httpx import Response

from avtv.search.wikimedia import WikimediaAdapter

SEARCH_RESPONSE = {
    "query": {
        "search": [
            {"title": "File:Bison.jpg"},
        ]
    }
}

IMAGEINFO_RESPONSE = {
    "query": {
        "pages": {
            "1": {
                "title": "File:Bison.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Bison.jpg",
                        "width": 4000,
                        "height": 3000,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "Wiki User"},
                        },
                    }
                ],
            }
        }
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_wikimedia_image_search():
    respx.get("https://commons.wikimedia.org/w/api.php").mock(
        side_effect=[
            Response(200, json=SEARCH_RESPONSE),
            Response(200, json=IMAGEINFO_RESPONSE),
        ]
    )
    a = WikimediaAdapter()
    results = await a.search("bison", kind="image", limit=1)
    assert len(results) == 1
    assert results[0].source == "wikimedia"
    assert results[0].url == "https://upload.wikimedia.org/Bison.jpg"
    assert results[0].width == 4000
    assert "CC BY-SA" in results[0].license
