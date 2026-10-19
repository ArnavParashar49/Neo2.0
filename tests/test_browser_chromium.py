"""NEO's Chromium on a Mac that hasn't got it (neo2/core/tools/browser.py): the first browser step asks
the user — one card, staged for that very step — and their yes downloads it (`python -m neo2 fetch
--browser`, run by the core's downloads) and carries the step on; a download that fails says why.
Headless pages start on the full Chromium (the "chromium" channel: `--no-shell` installs nothing
else), the headless shell only where nothing else is. Playwright and the download are fakes."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from neo2.core import effects, fetch
from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext, registry
from neo2.core.events import bus
from neo2.core.tools import browser
from tests.test_browser import Ctx, Page, run, world  # noqa: F401  (the fake browser, both ways)


@pytest.fixture
def cold(world, monkeypatch):  # noqa: F811
    """The world's browser not started yet, and Chromium not on this Mac."""
    b, ctx, page = world
    b._ctx = None
    launched: list[int] = []

    async def launch():
        launched.append(1)
        b._ctx, b._gone = ctx, False

    monkeypatch.setattr(b, "_launch", launch)
    here = {"chromium": False}
    monkeypatch.setattr(fetch, "chromium_here", lambda **kw: here["chromium"])
    return b, page, launched, here


def _download(here, *events):
    async def spawn(group):
        assert group == "browser"
        for ev in events:
            yield ev
        if events and events[-1].get("ok"):
            here["chromium"] = True

    fetch.downloads._spawn = spawn


def _approve(p, name, args):
    async def go():
        with confirm.approved(p.action_id):
            return await registry().invoke(name, args, ToolContext(origin="approval"))

    return asyncio.run(go())


def test_the_first_step_asks_to_download_chromium_and_the_yes_carries_it_on(cold):
    b, page, launched, here = cold
    out = run("browser_open", {"url": "shop.example/mug"})
    assert out.halt and out.text.startswith("NEEDS_CONFIRM") and launched == []
    p = confirm.peek()
    assert p.tool == "browser_open" and p.params == {"url": "shop.example/mug"}
    assert p.summary == browser._INSTALL_SUMMARY and "--no-shell" in p.detail
    notes: list[str] = []
    bus().subscribe(lambda ev: ev.type == "note" and notes.append(ev.data["text"]))
    _download(here, {"event": "progress", "all_done": 80, "all_total": 160}, {"event": "done", "ok": True})
    done = _approve(p, "browser_open", {"url": "shop.example/mug"})
    assert done.ok and launched == [1] and confirm.peek() is None
    assert notes[0] == "Downloading NEO's browser (Chromium)…" and notes[-1] == "NEO's browser is installed."
    # Once it's here, a later first step (the browser closed meanwhile) just starts it.
    b._ctx = None
    assert run("browser_open", {"url": "shop.example/mug"}).ok and launched == [1, 1]


def test_a_download_that_fails_says_why_and_offers_it_again(cold):
    _, _, launched, here = cold
    run("browser_read", {})
    p = confirm.peek()
    _download(here, {"event": "item", "id": "chromium", "state": "failed", "code": "offline", "message": "x"},
              {"event": "done", "ok": False, "failed": ["chromium"]})  # fmt: skip
    out = _approve(p, "browser_read", {})
    assert not out.ok and launched == []
    assert out.text.startswith("Error: NEO's browser (Chromium) couldn't be downloaded: NEO couldn't reach Playwright's servers")
    assert "python -m neo2 fetch --browser" in out.text
    assert run("browser_read", {}).halt  # the next step asks again


def test_a_no_cancels_and_nothing_downloads(cold):
    _, _, launched, here = cold
    run("browser_open", {"url": "shop.example"})
    confirm.cancel("user said no")
    assert not fetch.downloads.running("browser") and launched == [] and not here["chromium"]


def test_chromium_here_means_no_card(world):  # noqa: F811
    b, ctx, page = world
    b._ctx = None
    browser_launch: list = []

    async def launch():
        browser_launch.append(1)
        b._ctx = ctx

    b._launch = launch
    assert run("browser_read", {}).ok and browser_launch == [1] and confirm.peek() is None  # (conftest: installed)


# ---- launching: the full Chromium, headless too ---------------------------------------------------------


class _PW:
    def __init__(self, fail_channel=False, missing=False):
        self.calls: list[dict] = []
        self.fail_channel, self.missing = fail_channel, missing
        self.chromium = self

    async def start(self):
        return self

    async def stop(self):
        pass

    async def launch_persistent_context(self, path, **kw):
        self.calls.append(kw)
        if self.missing:
            raise RuntimeError("BrowserType.launch_persistent_context: Executable doesn't exist at /x/chrome\n"
                               "Looks like Playwright was just installed: playwright install")  # fmt: skip
        if self.fail_channel and kw.get("channel"):
            raise RuntimeError("Chromium distribution 'chromium' is not found")
        return _Ctx()


class _Ctx:
    pages: list = []

    def set_default_timeout(self, ms):
        pass

    def on(self, *a):
        pass


def _playwright(monkeypatch, pw):
    fake = types.ModuleType("playwright.async_api")
    fake.async_playwright = lambda: pw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake)
    monkeypatch.setattr(effects, "_allowed", {"browser"})  # the Playwright above is the test's own fake


@pytest.mark.parametrize("fail_channel", [False, True])
def test_a_headless_window_starts_the_full_chromium_first(monkeypatch, fail_channel):
    from neo2.core import config

    monkeypatch.setenv("NEO_BROWSER_VIEW", "window")
    monkeypatch.setenv("NEO_BROWSER_HEADLESS", "1")
    monkeypatch.setattr(config, "_settings", None)
    pw = _PW(fail_channel=fail_channel)
    _playwright(monkeypatch, pw)
    asyncio.run(browser._Browser()._launch())
    assert pw.calls[0]["channel"] == "chromium" and pw.calls[0]["headless"] is True
    assert len(pw.calls) == (2 if fail_channel else 1)  # only the headless shell: the fallback
    if fail_channel:
        assert "channel" not in pw.calls[1]


def test_a_visible_window_needs_no_channel(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_BROWSER_VIEW", "window")
    monkeypatch.delenv("NEO_BROWSER_HEADLESS", raising=False)
    monkeypatch.setattr(config, "_settings", None)
    pw = _PW()
    _playwright(monkeypatch, pw)
    asyncio.run(browser._Browser()._launch())
    assert len(pw.calls) == 1 and "channel" not in pw.calls[0] and pw.calls[0]["headless"] is False


def test_chromium_never_installed_is_said_as_such(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_BROWSER_VIEW", "window")
    monkeypatch.setattr(config, "_settings", None)
    _playwright(monkeypatch, _PW(missing=True))
    with pytest.raises(browser.ChromiumMissing, match="python -m neo2 fetch --browser"):
        asyncio.run(browser._Browser()._launch())


def test_the_page_reader_starts_the_full_chromium_first():
    from neo2.core.tools import render

    class PW:
        def __init__(self, fail):
            self.fail, self.calls = fail, []
            self.chromium = self

        async def launch(self, **kw):
            self.calls.append(kw)
            if self.fail and kw.get("channel"):
                raise RuntimeError("Executable doesn't exist at /x/chromium-1243/chrome")
            return types.SimpleNamespace(is_connected=lambda: True, close=None)

        async def stop(self):
            pass

    for fail in (False, True):
        pw = PW(fail)

        async def start(pw=pw):
            return pw

        r = render._Renderer(start=start)
        try:
            asyncio.run(r._ready())
        finally:
            r._browser = None
        assert pw.calls[0]["channel"] == "chromium" and pw.calls[0]["headless"] is True
        assert [c.get("channel") for c in pw.calls] == (["chromium", None] if fail else ["chromium"])
