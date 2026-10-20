"""Search Screen beyond the Mac (neo2.core.lens.search): "Search with Google" writes a page of NEO's
own (no outside resource, no referrer, posting only to the provider, the user's alone to read) that
uploads a smaller copy of the area in the browser that opens https links, and deletes it after 90 s;
Bing's recipe; text-only sites and "off"; web.open_search is search_site's own way to a site; search
rows are one per host. Nothing is opened, fetched or uploaded: apps.open_url and the web are fakes."""

from __future__ import annotations

import asyncio
import base64
import io
import re
import stat

import pytest
from PIL import Image

from neo2.core import config
from neo2.core.agent.registry import ToolContext
from neo2.core.lens import search
from neo2.core.tools import web


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "folder", lambda: tmp_path / "neo-lens")
    monkeypatch.setattr(search, "_hl", lambda: "hi")
    return tmp_path / "neo-lens"


@pytest.fixture
def opened(monkeypatch):
    from neo2.core.decide import workspace
    from neo2.core.tools.computer import apps

    got: list[tuple[str, str]] = []

    async def open_url(url, browser_id="", browser_name=""):
        got.append((url, browser_id))
        return f"Opened {url}"

    monkeypatch.setattr(apps, "open_url", open_url)
    monkeypatch.setattr(workspace, "default_handler", lambda scheme: "com.example.browser" if scheme == "https" else None)
    return got


def _setting(monkeypatch, value):
    monkeypatch.setenv("NEO_LENS_SEARCH", value)
    monkeypatch.setattr(config, "_settings", None)


AREA = Image.new("RGBA", (3000, 1500), (10, 200, 30, 255))


def _page_image(text: str) -> Image.Image:
    b64 = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', text).group(1)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_google_gets_a_smaller_copy_through_the_https_browser(monkeypatch, opened, _own_folder):
    timers = []
    monkeypatch.setattr(search, "_delete_later", lambda p: timers.append(p))
    assert asyncio.run(search.image_search(AREA)) == "opened"
    [(url, browser)] = opened
    assert browser == "com.example.browser" and url.startswith("file://") and url.endswith(".html")
    [p] = list(_own_folder.glob("*.html"))
    assert timers == [p]
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and stat.S_IMODE(_own_folder.stat().st_mode) == 0o700
    text = p.read_text()
    assert 'action="https://lens.google.com/v3/upload?hl=hi&amp;st=' in text
    assert 'name="encoded_image"' in text and 'enctype="multipart/form-data"' in text
    assert '<meta name="referrer" content="no-referrer">' in text and "form-action https://lens.google.com" in text
    # Nothing on the page loads from anywhere: the only address in it is the form's.
    assert set(re.findall(r"https?://[\w.-]+", text)) == {"https://lens.google.com"}
    assert "<img" not in text and "src=" not in text and "<link" not in text
    img = _page_image(text)
    assert img.format == "JPEG" and img.size == (1000, 500)  # (1000 px on the long side)
    assert img.convert("RGB").getpixel((500, 250))[1] > 150


def test_the_page_is_deleted_after_90_seconds_and_swept_at_start(monkeypatch, opened, _own_folder):
    later = []

    class Loop:
        def call_later(self, s, fn):
            later.append((s, fn))

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: Loop())
    p = search._write("x")
    search._delete_later(p)
    [(secs, fn)] = later
    assert secs == 90.0 and p.exists()
    fn()
    assert not p.exists()
    search._write("left from before")
    search._write("and another")
    assert search.sweep() == 2 and list(_own_folder.glob("*.html")) == []


def test_bings_recipe_sends_the_picture_as_text(monkeypatch, opened, _own_folder):
    _setting(monkeypatch, "bing")
    monkeypatch.setattr(search, "_delete_later", lambda p: None)
    assert search.provider() == {"word": "bing", "label": "Bing", "image": True, "text": True}
    assert asyncio.run(search.image_search(AREA)) == "opened"
    text = next(_own_folder.glob("*.html")).read_text()
    assert 'action="https://www.bing.com/images/search?view=detailv2&amp;iss=sbiupload"' in text
    assert re.search(r'name="imageBin" value="[A-Za-z0-9+/=]{100,}"', text) and "DataTransfer" not in text


def test_a_text_only_site_and_off(monkeypatch, opened):
    _setting(monkeypatch, "duckduckgo")
    assert search.provider() == {"word": "duckduckgo", "label": "DuckDuckGo", "image": False, "text": True}
    assert asyncio.run(search.image_search(AREA)) == "no_image" and not opened
    _setting(monkeypatch, "kagi.com")
    assert search.provider()["label"] == "Kagi"
    _setting(monkeypatch, "off")
    assert search.provider() is None
    assert asyncio.run(search.image_search(AREA)) == "off" and asyncio.run(search.text_search("x")) == "off"


def test_no_browser_for_https_links(monkeypatch, opened):
    from neo2.core.decide import workspace

    monkeypatch.setattr(workspace, "default_handler", lambda scheme: None)
    assert asyncio.run(search.image_search(AREA)) == "no_browser" and not opened


def test_open_search_is_search_sites_own_way(monkeypatch):
    from neo2.core.tools.computer import apps

    seen = []

    async def open_app(target):
        seen.append(target)
        return f"Opened {target}"

    monkeypatch.setattr(apps, "open_app", open_app)
    by_tool = asyncio.run(web.search_site({"site": "google", "query": "blue shoes"}, ToolContext(user_text="")))
    by_lens = asyncio.run(web.open_search("google", "blue shoes"))
    assert by_tool == by_lens == "Searching google for “blue shoes”."
    assert seen[0] == seen[1] and "google" in seen[0] and "blue" in seen[0]
    assert asyncio.run(search.text_search("blue shoes", word="google")) == "opened" and seen[2] == seen[0]


def test_search_rows_one_per_host_and_only_web_links(monkeypatch):
    rows = [
        {"title": "Runner 4 — Brand X", "href": "https://brandx.example/runner-4", "body": "…"},
        {"title": "Runner 4 again", "href": "https://brandx.example/sale", "body": ""},
        {"title": "Shop", "href": "javascript:alert(1)"},
        {"title": "Big Store: Runner 4", "href": "https://www.bigstore.example/p/1"},
        {"title": "x" * 200, "href": "http://third.example/"},
    ]
    asked = []
    monkeypatch.setattr(web, "_has_ddgs", lambda: True)
    monkeypatch.setattr(web, "_race", lambda q, n: asked.append((q, n)) or rows)
    got = search.links("brand x runner 4", 5)
    assert [r["host"] for r in got] == ["brandx.example", "www.bigstore.example", "third.example"]
    assert len(got[2]["title"]) == 90 and asked == [("brand x runner 4", 8)]
    assert search.links("brand x runner 4", 2)[-1]["host"] == "www.bigstore.example"
    monkeypatch.setattr(web, "_has_ddgs", lambda: False)
    assert search.links("anything", 5) == []
    assert (search.wanted("product", False), search.wanted("plant", False), search.wanted("math", False)) == (5, 3, 0)
    assert search.wanted("other", True) == 5
