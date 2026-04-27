import pytest
import respx
from httpx import Response

from avtv.downloader import Downloader


@pytest.mark.asyncio
@respx.mock
async def test_download_caches_by_url_hash(tmp_path):
    respx.get("https://example.com/clip.mp4").mock(
        return_value=Response(200, content=b"fake-video-bytes")
    )
    d = Downloader(cache_dir=tmp_path)
    path1 = await d.fetch("https://example.com/clip.mp4", kind="video")
    assert path1.exists()
    assert path1.read_bytes() == b"fake-video-bytes"
    assert path1.parent.name == "clips"
    assert path1.suffix == ".mp4"


@pytest.mark.asyncio
@respx.mock
async def test_download_skips_existing(tmp_path):
    route = respx.get("https://example.com/x.mp4").mock(
        return_value=Response(200, content=b"abc")
    )
    d = Downloader(cache_dir=tmp_path)
    await d.fetch("https://example.com/x.mp4", kind="video")
    await d.fetch("https://example.com/x.mp4", kind="video")
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_download_image_kind_uses_images_dir(tmp_path):
    respx.get("https://example.com/photo.jpg").mock(
        return_value=Response(200, content=b"img")
    )
    d = Downloader(cache_dir=tmp_path)
    p = await d.fetch("https://example.com/photo.jpg", kind="image")
    assert p.parent.name == "images"
    assert p.suffix == ".jpg"
