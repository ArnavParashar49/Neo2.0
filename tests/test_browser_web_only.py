"""An embedded agent's browser (a headless one on a server, often) never reaches the machine it
runs on or its network — the cloud metadata address, localhost, the LAN — whatever led there."""

from __future__ import annotations

import sys
import types

import pytest
from neo_lib import make_runtime

from neo import runtime
from neo.agent import confirm
from neo.agent.registry import ToolContext, registry

METADATA = "http://169.254.169.254/latest/meta-data/"
PUBLIC = "https://93.184.215.14/"  # an address literal: no DNS in tests


class _Page:
    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        self.url = "about:blank"
        self.went: list[str] = []
        self.redirects = redirects or {}

    async def goto(self, url, **kw):
        self.went.append(url)
        self.url = self.redirects.get(url, url)

    async def screenshot(self, **kw):
        return b"png"


class _Browser:
    def __init__(self, page: _Page) -> None:
        self._page = page

    async def page(self):
        return self._page


@pytest.fixture
def agent(tmp_path, monkeypatch):
    import neo.tools.browser as br

    monkeypatch.setattr(br, "_snapshot", lambda page, n: _read(page))
    rt = make_runtime(tmp_path / "a", "a", ["browser_open", "browser_read", "browser_screenshot"])
    return rt


async def _read(page) -> str:
    return f"URL: {page.url}\n[TEXT]\nsecret-access-key=AKIA-EXAMPLE"


@pytest.mark.parametrize("url", [METADATA, "http://localhost:8080/admin", "http://10.0.0.5/", "http://[::1]:9000/"])
async def test_it_never_opens_a_local_address(agent, url):
    page = _Page()
    agent.browser = _Browser(page)
    with runtime.bind(agent.scope()):
        out = await registry().invoke("browser_open", {"url": url}, ToolContext())  # untainted
        assert confirm.peek() is None
    assert out.text.startswith("Error") and page.went == [] and "AKIA" not in out.text


async def test_where_a_redirect_lands_is_checked_too(agent):
    page = _Page({PUBLIC: METADATA})
    agent.browser = _Browser(page)
    with runtime.bind(agent.scope()):
        out = await registry().invoke("browser_open", {"url": PUBLIC}, ToolContext())
        assert out.text.startswith("Error") and "AKIA" not in out.text
        assert page.url == "about:blank"  # left the page
        for tool in ("browser_read", "browser_screenshot"):
            page.url = METADATA  # got there some other way (a click, a script)
            out = await registry().invoke(tool, {}, ToolContext())
            assert out.text.startswith("Error") and "AKIA" not in out.text and not out.images, tool
    page.url = PUBLIC
    with runtime.bind(agent.scope()):
        assert "AKIA" in (await registry().invoke("browser_read", {}, ToolContext())).text  # the web: as before


class _Route:
    def __init__(self, url: str) -> None:
        self.request = types.SimpleNamespace(url=url)
        self.did = ""

    async def abort(self, code="failed"):
        self.did = f"abort:{code}"

    async def continue_(self, **kw):
        self.did = "continue"


async def test_nothing_it_loads_goes_to_a_local_address(tmp_path, monkeypatch):
    """Every request the agent's browser makes — the page, a frame, a script's fetch — is routed
    through a check; NEO.app's own browser has none (the user's own Mac, as always)."""
    routes: list = []

    class _Ctx:
        pages: list = []

        def set_default_timeout(self, ms):
            pass

        async def route(self, pattern, handler):
            routes.append((pattern, handler))

        async def new_page(self):
            return "page"

        async def close(self):
            pass

    class _Chromium:
        async def launch_persistent_context(self, path, **kw):
            return _Ctx()

    class _PW:
        chromium = _Chromium()

        async def start(self):
            return self

        async def stop(self):
            pass

    fake = types.ModuleType("playwright.async_api")
    fake.async_playwright = lambda: _PW()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake)
    import neo.tools.browser as br

    rt = make_runtime(tmp_path / "a", "a")
    with runtime.bind(rt.scope()):
        await br._browser().page()
    assert [p for p, _ in routes] == ["**/*"]
    handler = routes[0][1]
    for url, want in (
        (METADATA, "abort:blockedbyclient"),
        ("http://127.0.0.1:8765/", "abort:blockedbyclient"),
        ("http://192.168.1.1/login", "abort:blockedbyclient"),
        (PUBLIC + "app.js", "continue"),
        ("data:text/plain,hi", "continue"),
    ):
        r = _Route(url)
        await handler(r)
        assert r.did == want, url
    runtime.set_library_mode(False)
    monkeypatch.setattr(br, "_b", br._Browser())
    await br._browser().page()
    assert len(routes) == 1  # NEO.app's browser: unchanged
