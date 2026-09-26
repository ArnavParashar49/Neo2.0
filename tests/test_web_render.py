"""web_fetch reads pages their scripts build: a headless Chromium renders them (neo/tools/render.py).

The browser is faked throughout; the proxy that keeps a page off this Mac and the local network
runs for real on loopback sockets. One optional check drives a real headless Chromium against a
public page (NEO_TEST_RENDER=1).
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import os
import threading
import time
from contextlib import suppress

import httpx
import pytest

from neo.agent.registry import ToolContext
from neo.tools import render, web
from neo.tools.computer import policy

WORDS = "The quiet river bends past the old mill, where herons wait for the morning light. "


def _prose(n: int, tag: str = "") -> str:
    """n distinct paragraphs (an extractor drops repeated ones)."""
    return "".join(
        f"<p>{tag}Paragraph {i} tells how the herons at mill number {i} waited {i * 7} minutes for the light, "
        f"while the river carried {i * 3} leaves past stone {i + 11} and the miller counted {i * 13} sacks.</p>"
        for i in range(n)
    )


SHELL = (
    "<html><head><title>App</title><script src='/static/bundle.js'></script></head><body>"
    "<noscript>You need to enable JavaScript to run this app.</noscript><div id='root'></div></body></html>"
)
ARTICLE = (
    "<html><head><title>Mill</title><script>track()</script></head><body><article><h1>The mill</h1>"
    + _prose(20)
    + "</article><noscript>Please enable JavaScript for the best experience.</noscript></body></html>"
)
BUILT = (
    "<html><head><title>Quotes</title></head><body><main><article><h1>Quotes</h1>"
    + _prose(12, "“")
    + "</article></main></body></html>"
)


# ---- when a page needs rendering --------------------------------------------------------------------


def _texts(html):
    return web._texts(html)


def test_an_app_shell_needs_rendering():
    assert render.needs_render(SHELL, *_texts(SHELL))


def test_an_article_does_not():
    main, full = _texts(ARTICLE)
    assert len(main) > 2000 and not render.needs_render(ARTICLE, main, full)


def test_without_scripts_there_is_nothing_to_render():
    page = "<html><body><p>Short and static.</p></body></html>"
    assert not render.needs_render(page, *_texts(page))


def test_a_page_asking_for_javascript_is_a_shell_even_with_some_words():
    # a few hundred words of "this is an interactive app" around no content at all
    page = (
        "<html><head><script src='/app.js'></script></head><body><noscript><h1>JavaScript Required</h1>"
        f"<p>This is a heavily interactive web application, and JavaScript is required. {WORDS * 4}</p>"
        "</noscript></body></html>"
    )
    main, full = _texts(page)
    assert 300 < len(full) < 2000 and render.needs_render(page, main, full)


def test_an_empty_mount_point_is_a_shell():
    page = (
        "<html><head><script type='module' src='/main.js'></script></head><body>"
        f'<header><p>{WORDS * 5}</p></header><div id="app"></div></body></html>'
    )
    main, full = _texts(page)
    assert len(full) > 300 and render.needs_render(page, main, full)
    unquoted = page.replace('<div id="app">', "<div id=app>")
    assert render.needs_render(unquoted, main, full)


@pytest.mark.parametrize(
    "hostile",
    [
        "<div " * 12_000,  # a tag that never closes
        "<noscript>" * 10_000,  # a <noscript> that never closes
        "<noscript>" + "<a " * 40_000 + "</noscript>",  # tags inside one that never close
    ],
    ids=["unclosed-tag", "unclosed-noscript", "unclosed-tags-in-noscript"],
)
def test_hostile_markup_is_judged_in_linear_time(hostile):
    # a page can't stall the judgement (and every thread with it: the regex holds the GIL)
    page = f"<html><head><script>x()</script></head><body><p>{WORDS * 7}</p>{hostile}</body></html>"
    t = time.perf_counter()
    render.needs_render(page, WORDS * 7, WORDS * 7)
    assert time.perf_counter() - t < 1.0


# ---- web_fetch ------------------------------------------------------------------------------------


@pytest.fixture
def site(monkeypatch):
    """A fake web (httpx's MockTransport) and a fake headless browser. Pages are set per test."""
    pages: dict[str, tuple[int, str]] = {}
    rendered: dict[str, render.Rendered | Exception] = {}
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/old":
            return httpx.Response(301, headers={"location": "https://site.test/app"})
        status, body = pages.get(path, (404, "<html><body>not here</body></html>"))
        if 300 <= status < 400:
            return httpx.Response(status, headers={"location": body})
        return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})

    async def fake_render(url, guard=None, **kw):
        asked.append(url)
        out = rendered.get(url)
        if out is None:
            raise AssertionError(f"rendered a page that didn't need it: {url}")
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(policy, "private_host", lambda url: "")  # no DNS in tests
    monkeypatch.setattr(
        web.httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
    )
    monkeypatch.setattr(render, "render", fake_render)
    return pages, rendered, asked


def _fetch(url):
    return asyncio.run(web.web_fetch({"url": url}, ToolContext()))


def _page(html="", text="", status=200, url="https://site.test/app"):
    return render.Rendered(url=url, title="", html=html, text=text, seconds=0.5, status=status)


def test_a_script_built_page_is_read_the_way_a_browser_shows_it(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = _page(html=BUILT)
    out = _fetch("https://site.test/old")
    assert asked == ["https://site.test/app"]  # where the redirects ended, not the old address
    assert "herons" in out and "enable JavaScript" not in out


def test_an_article_is_never_rendered(site):
    pages, rendered, asked = site
    pages["/story"] = (200, ARTICLE)
    out = _fetch("https://site.test/story")
    assert asked == [] and "herons" in out


def test_an_app_the_extractor_cannot_read_is_read_as_shown(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    feed = "\n".join(f"Post {i}: {WORDS}" for i in range(30))
    rendered["https://site.test/app"] = _page(
        html="<html><body><div id='root'><div><span>x</span></div></div></body></html>", text=feed
    )
    out = _fetch("https://site.test/app")
    assert out.startswith("Post 0:") and "Post 29" in out


def test_when_rendering_fails_the_page_as_sent_stands(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = render.RenderError("the page-reading browser didn't start")
    out = _fetch("https://site.test/app")
    assert asked and "enable JavaScript" in out


def test_a_rendering_with_less_to_read_is_not_used(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = _page(text="Loading…")
    out = _fetch("https://site.test/app")
    assert "enable JavaScript" in out


def test_a_page_that_turns_away_automated_clients_is_not_tried_in_a_browser(site):
    # "Enable JavaScript and cookies to continue" with a 403 is a bot check: the site's answer
    # stands — NEO doesn't try to get past it
    pages, rendered, asked = site
    gate = "<html><head><script src='/challenge.js'></script></head><body><noscript>Enable JavaScript and cookies to continue</noscript></body></html>"
    pages["/pkg"] = (403, gate)
    pages["/busy"] = (503, gate)
    rendered["https://site.test/pkg"] = rendered["https://site.test/busy"] = _page(html=BUILT)
    for path, code in (("/pkg", "403"), ("/busy", "503")):
        out = _fetch(f"https://site.test{path}")
        assert out.startswith("Error fetching") and code in out
    assert asked == []


def test_a_rendering_that_ends_on_an_error_page_is_not_used(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = _page(text="Verifying you are human. " * 40, status=403)
    out = _fetch("https://site.test/app")
    assert asked and "Verifying" not in out and "enable JavaScript" in out


def test_a_rendering_must_show_more_than_the_server_sent(site):
    # the server's page had real words (and a menu); a wall that shows less is not an upgrade
    pages, rendered, asked = site
    nav = "".join(f'<li><a href="/c{i}">Category number {i} of the shop</a></li>' for i in range(30))
    shop = (
        "<html><head><title>Shop</title><script src='/app.js'></script></head><body><nav><ul>"
        + nav
        + "</ul></nav><h1>Stone mill grinder</h1><p>A hand-cranked stone mill grinder that turns grain"
        " into flour in minutes, with a cast iron body, two granite stones and a clamp that fits any"
        " table up to five centimetres thick.</p><div id='root'></div></body></html>"
    )
    main, full = _texts(shop)
    wall = "Verifying you are human. This may take a few seconds. " * 12
    assert len(main) < len(wall) < len(full) and render.needs_render(shop, main, full)
    pages["/grinder"] = (200, shop)
    rendered["https://site.test/grinder"] = _page(text=wall, url="https://site.test/grinder")
    out = _fetch("https://site.test/grinder")
    assert asked and "Verifying" not in out and "granite stones" in out


def test_a_rendering_that_led_to_another_site_says_so(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = _page(html=BUILT, url="https://accounts.other.test/signin?next=%2F")
    out = _fetch("https://site.test/app")
    assert out.startswith("Redirected to accounts.other.test") and "herons" in out
    rendered["https://site.test/app"] = _page(html=BUILT, url="https://www.site.test/app")
    assert not _fetch("https://site.test/app").startswith("Redirected")  # the same site


def test_a_link_that_carries_data_is_never_rendered(site):
    # a short link that lands on a token-bearing page ("Confirming…" and a script): loading it in a
    # browser could let its script act on the token
    pages, rendered, asked = site
    token = "a1b2c3d4e5" * 7
    pages["/s/xyz"] = (302, f"https://site.test/confirm?t={token}")
    pages["/confirm"] = (200, SHELL.replace("<title>App</title>", "<title>Confirming…</title>"))
    rendered[f"https://site.test/confirm?t={token}"] = _page(html=BUILT)
    out = _fetch("https://site.test/s/xyz")
    assert asked == [] and "enable JavaScript" in out


def test_any_failure_to_render_leaves_the_page_as_sent(site):
    pages, rendered, asked = site
    pages["/app"] = (200, SHELL)
    rendered["https://site.test/app"] = RuntimeError("Target page, context or browser has been closed")
    out = _fetch("https://site.test/app")
    assert asked and "enable JavaScript" in out


def test_the_page_is_judged_off_the_voice_loop(site, monkeypatch):
    pages, rendered, asked = site
    pages["/story"] = (200, ARTICLE)
    on_loop = []
    judge = render.needs_render

    def spy(*a):
        on_loop.append(threading.current_thread() is threading.main_thread())
        return judge(*a)

    monkeypatch.setattr(render, "needs_render", spy)
    _fetch("https://site.test/story")
    assert on_loop == [False]


def test_other_errors_are_not_rendered(site):
    pages, rendered, asked = site
    pages["/gone"] = (404, SHELL)
    out = _fetch("https://site.test/gone")
    assert asked == [] and out.startswith("Error fetching") and "404" in out


def test_a_private_address_is_refused_before_anything_loads(monkeypatch):
    called = []
    monkeypatch.setattr(render, "render", lambda *a, **k: called.append(a))
    out = _fetch("http://192.168.1.1/")
    assert out.startswith("Error: not fetching") and called == []


# ---- the proxy: public addresses only --------------------------------------------------------------


@pytest.mark.parametrize(
    "ip, public",
    [
        ("8.8.8.8", True),
        ("2606:4700::6810:84e5", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # a cloud's metadata service
        ("100.100.1.1", False),  # a VPN's shared range (Tailscale)
        ("0.0.0.0", False),
        ("::1", False),
        ("::ffff:127.0.0.1", False),  # this Mac, written as IPv6
        ("fd00::1", False),
        ("fe80::1", False),
        ("fec0::1", False),  # site-local
        ("224.0.0.251", False),
    ],
)
def test_only_public_addresses_are_reached(ip, public):
    assert render.public_ip(ipaddress.ip_address(ip)) is public


async def _server(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _ask(gate, request: str):
    r, w = await asyncio.open_connection("127.0.0.1", gate.port)
    w.write(request.encode())
    await w.drain()
    return r, w


def test_the_proxy_never_connects_to_this_mac():
    async def go():
        hits = []

        async def target(r, w):
            hits.append(1)
            w.close()

        server, port = await _server(target)
        gate = await render._Gate(policy.private_host, 100_000).start()
        try:
            for req in (
                f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n",
                f"GET http://127.0.0.1:{port}/admin HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n",
                f"CONNECT localhost:{port} HTTP/1.1\r\n\r\n",
                f"CONNECT [::1]:{port} HTTP/1.1\r\n\r\n",
                f"GET ftp://127.0.0.1:{port}/ HTTP/1.1\r\n\r\n",
            ):
                r, w = await _ask(gate, req)
                line = await asyncio.wait_for(r.readline(), 5)
                assert line.split()[1] in (b"403", b"400"), (req, line)
                w.close()
        finally:
            await gate.close()
            server.close()
        return hits, gate.refused

    hits, refused = asyncio.run(go())
    assert hits == [] and "127.0.0.1" in refused and "localhost" in refused


def test_a_name_with_any_private_answer_is_refused(monkeypatch):
    answers = {
        "mixed.test": ["93.184.215.14", "10.0.0.7"],
        "public.test": ["93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"],
        "nas.local": ["93.184.215.14"],  # a local name is refused by name, whatever it resolves to
    }

    async def fake_getaddrinfo(self, host, port, **kw):
        return [(0, 0, 0, "", (ip, port)) for ip in answers[host]]

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(
        policy.socket,
        "getaddrinfo",
        lambda host, port, **kw: [(0, 0, 0, "", (ip, port)) for ip in answers[host]],
    )

    async def go():
        gate = render._Gate(policy.private_host, 1000)
        got = [await gate.addresses(h, 443) for h in ("mixed.test", "public.test", "nas.local")]
        return got, gate.refused

    (mixed, public, local), refused = asyncio.run(go())
    assert mixed == [] and local == [] and {"mixed.test", "nas.local"} <= set(refused)
    assert public == answers["public.test"]


def test_a_tunnel_to_an_allowed_address_carries_the_page():
    async def go():
        async def echo(r, w):
            while data := await r.read(1024):
                w.write(data)
                await w.drain()
            w.close()

        server, port = await _server(echo)
        gate = await render._Gate(lambda url: "", 100_000, allow=lambda ip: True).start()
        try:
            r, w = await _ask(gate, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n")
            head = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 5)
            w.write(b"hello over tls")
            await w.drain()
            back = await asyncio.wait_for(r.readexactly(14), 5)
            w.close()
        finally:
            await gate.close()
            server.close()
        return head, back, gate.bytes

    head, back, n = asyncio.run(go())
    assert head.startswith(b"HTTP/1.1 200") and back == b"hello over tls" and n == 14


def test_a_page_is_cut_off_at_its_byte_limit():
    async def go():
        async def flood(r, w):
            w.write(b"x" * 200_000)
            with suppress(OSError):
                await w.drain()
            await asyncio.sleep(0.5)
            w.close()

        server, port = await _server(flood)
        gate = await render._Gate(lambda url: "", 20_000, allow=lambda ip: True).start()
        try:
            r, w = await _ask(gate, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n")
            await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 5)
            got = 0
            while data := await asyncio.wait_for(r.read(65536), 5):
                got += len(data)
            w.close()
            r2, w2 = await _ask(gate, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n")
            again = await asyncio.wait_for(r2.readline(), 5)
            w2.close()
        finally:
            await gate.close()
            server.close()
        return got, gate.capped, again

    got, capped, again = asyncio.run(go())
    assert got <= 20_000 and capped and b" 403 " in again


def test_plain_http_goes_one_request_per_connection():
    async def go():
        seen = []

        async def origin(r, w):
            seen.append(await r.readuntil(b"\r\n\r\n"))
            w.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\nKeep-Alive: timeout=5\r\n\r\nhi"
            )
            await w.drain()
            w.close()

        server, port = await _server(origin)
        gate = await render._Gate(lambda url: "", 100_000, allow=lambda ip: True).start()
        try:
            r, w = await _ask(
                gate,
                f"GET http://127.0.0.1:{port}/p?q=1 HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                "Proxy-Connection: keep-alive\r\nAccept: */*\r\n\r\n",
            )
            reply = await asyncio.wait_for(r.read(), 5)
            w.close()
        finally:
            await gate.close()
            server.close()
        return seen[0].decode(), reply.decode()

    request, reply = asyncio.run(go())
    assert request.startswith("GET /p?q=1 HTTP/1.1\r\n") and "Connection: close" in request
    assert "Proxy-Connection" not in request and "Accept: */*" in request
    head, body = reply.split("\r\n\r\n", 1)
    assert "Connection: close" in head and "keep-alive" not in head.lower() and body == "hi"


# ---- the headless browser (faked) -------------------------------------------------------------------


class FakeRequest:
    def __init__(self, url, rtype, nav, method="GET"):
        self.url, self.resource_type, self._nav, self.method = url, rtype, nav, method

    def is_navigation_request(self):
        return self._nav


class FakeRoute:
    def __init__(self, url, rtype="document", nav=False, method="GET"):
        self.request, self.outcome = FakeRequest(url, rtype, nav, method), ""

    async def abort(self, code=None):
        self.outcome = "abort"

    async def continue_(self, **kw):
        self.outcome = "continue"


class FakeResponse:
    def __init__(self, status, frame):
        self.status, self.frame, self.request = status, frame, FakeRequest("", "document", True)


class FakePage:
    def __init__(self, ctx):
        self.ctx, self.url, self.main_frame, self.handlers = ctx, "about:blank", object(), {}

    def on(self, event, cb):
        self.handlers.setdefault(event, []).append(cb)

    async def _through_route(self, url, rtype, nav=False, method="GET"):
        route = FakeRoute(url, rtype, nav, method)
        await self.ctx.handler(route)
        self.ctx.routes.append(route)
        return route

    async def goto(self, url, wait_until=None, timeout=None):
        s = self.ctx.site
        if (await self._through_route(url, "document", nav=True)).outcome != "continue":
            raise Exception(f"net::ERR_BLOCKED_BY_CLIENT at {url}")
        for sub, rtype, *how in s.get("requests", []):
            await self._through_route(sub, rtype, *how)
        if s.get("hang"):
            await asyncio.sleep(3600)
        status = s.get("status", 200)
        if to := s.get("redirect_to"):
            # the browser follows a redirect on its own (no route call): through the proxy
            host = to.split("/")[2]
            port = int(self.ctx.kwargs["proxy"]["server"].rsplit(":", 1)[1])
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.write(f"GET {to} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
            status = int((await r.readline()).split()[1])
            w.close()
            self.url = to
        else:
            self.url = url
        for cb in self.handlers.get("response", []):
            cb(FakeResponse(status, self.main_frame))

    async def wait_for_load_state(self, state, timeout=None):
        if state == "networkidle" and self.ctx.site.get("busy"):
            await asyncio.sleep(3600)

    async def evaluate(self, js, arg=None):
        s = self.ctx.site
        if arg is None:
            return len(s.get("text", ""))
        return {"title": "T", "html": s.get("html", ""), "text": s.get("text", "")}

    async def close(self):
        pass


class FakeContext:
    def __init__(self, browser, kwargs):
        self.browser, self.kwargs, self.site = browser, kwargs, browser.site
        self.handler, self.routes, self.closed = None, [], False

    def set_default_timeout(self, ms):
        pass

    async def route(self, pattern, handler):
        self.handler = handler

    async def new_page(self):
        return FakePage(self)

    async def close(self):
        self.closed = True


class FakeBrowser:
    version = "141.0.7390.37"

    def __init__(self, site, launch_kwargs):
        self.site, self.launch_kwargs, self.contexts, self.closed = site, launch_kwargs, [], False

    def is_connected(self):
        return not self.closed

    async def new_context(self, **kwargs):
        if why := self.site.get("context_error"):
            raise Exception(why)  # Playwright's own errors (TargetClosedError…)
        ctx = FakeContext(self, kwargs)
        self.contexts.append(ctx)
        return ctx

    async def close(self):
        self.closed = True


class FakePlaywright:
    def __init__(self, site):
        self.site, self.browsers, self.stopped = site, [], False
        self.chromium = self

    async def launch(self, **kwargs):
        b = FakeBrowser(self.site, kwargs)
        self.browsers.append(b)
        return b

    async def stop(self):
        self.stopped = True


def _guard(url):
    host = url.split("/")[2].rsplit(":", 1)[0] if url.count(":") > 1 else url.split("/")[2]
    return (
        "it's on this Mac or the local network" if host.startswith(("192.168.", "10.", "localhost")) else ""
    )


@pytest.fixture
def browser():
    made = []

    def make(site, *, fail=False, idle_s=60.0):
        started = []

        async def start():
            if fail:
                raise RuntimeError(
                    "Executable doesn't exist at /ms-playwright/chromium_headless_shell/chrome"
                )
            pw = FakePlaywright(site)
            started.append(pw)
            return pw

        r = render._Renderer(start=start, idle_s=idle_s)
        made.append(r)
        return r, started

    yield make
    for r in made:
        r.close_now()


def _render(r, url, **kw):
    kw.setdefault("timeout_s", 3.0)
    kw.setdefault("max_bytes", 100_000)
    return asyncio.run(r.render(url, _guard, **kw))


def test_one_browser_serves_every_page_each_in_a_fresh_private_context(browser):
    site = {"html": BUILT, "text": WORDS * 10}
    r, started = browser(site)
    first = _render(r, "https://site.test/a")
    second = _render(r, "https://site.test/b")
    assert first.url == "https://site.test/a" and "herons" in second.html and first.status == 200
    assert len(started) == 1 and r.launches == 1
    (b,) = started[0].browsers
    assert b.launch_kwargs["headless"] is True
    assert len(b.contexts) == 2 and all(c.closed for c in b.contexts)
    kw = b.contexts[0].kwargs
    assert kw["accept_downloads"] is False and kw["service_workers"] == "block"
    assert kw["proxy"]["server"].startswith("http://127.0.0.1:")
    assert "user_agent" not in kw  # Chromium's own, which says it's headless
    assert not any("AutomationControlled" in a for a in b.launch_kwargs["args"])  # nor hides that it's automated
    assert b.contexts[0].kwargs["proxy"] != b.contexts[1].kwargs["proxy"]  # a proxy per page


def test_what_the_page_asks_for_is_checked(browser):
    site = {
        "html": BUILT,
        "requests": [
            ("https://cdn.site.test/hero.png", "image"),
            ("https://cdn.site.test/font.woff2", "font"),
            ("http://192.168.1.1/admin", "fetch"),
            ("http://localhost:8080/api", "xhr"),
            ("file:///etc/passwd", "fetch"),
            ("https://api.site.test/posts", "fetch"),
            ("blob:https://site.test/1234", "script"),
            ("http://site.test:99999/x", "fetch"),  # a port that can't be
            # a form the page submits by itself: reading a page never sends one
            ("https://site.test/confirm", "document", True, "POST"),
            ("https://site.test/search?q=mill", "document", True, "GET"),
            # a link that carries data (a token…), in the page or a frame, isn't opened
            ("https://site.test/approve?t=" + "a1b2c3d4e5" * 6, "document", True, "GET"),
            # an app loading its data by POST (GraphQL, search APIs) still works
            ("https://api.site.test/graphql", "fetch", False, "POST"),
        ],
    }
    r, started = browser(site)
    _render(r, "https://site.test/app")
    (ctx,) = started[0].browsers[0].contexts
    outcome = {x.request.url: x.outcome for x in ctx.routes}
    assert outcome == {
        "https://site.test/app": "continue",
        "https://cdn.site.test/hero.png": "abort",
        "https://cdn.site.test/font.woff2": "abort",
        "http://192.168.1.1/admin": "abort",
        "http://localhost:8080/api": "abort",
        "file:///etc/passwd": "abort",
        "https://api.site.test/posts": "continue",
        "blob:https://site.test/1234": "continue",
        "http://site.test:99999/x": "abort",
        "https://site.test/confirm": "abort",
        "https://site.test/search?q=mill": "continue",
        "https://site.test/approve?t=" + "a1b2c3d4e5" * 6: "abort",
        "https://api.site.test/graphql": "continue",
    }


def test_a_page_on_the_local_network_is_not_loaded(browser):
    r, started = browser({"html": BUILT})
    with pytest.raises(render.RenderError, match="local network"):
        _render(r, "http://192.168.1.1/")
    assert started[0].browsers[0].contexts[0].closed


def test_a_link_that_carries_data_is_not_loaded(browser):
    r, started = browser({"html": BUILT})
    with pytest.raises(render.RenderError, match="carries data"):
        _render(r, "https://site.test/confirm?t=" + "a1b2c3d4e5" * 6)


def test_any_browser_failure_is_a_render_error(browser):
    r, started = browser({"html": BUILT, "context_error": "BrowserContext.new_context: Target closed"})
    with pytest.raises(render.RenderError, match="Target closed"):
        _render(r, "https://site.test/a")


def test_the_browser_never_skips_the_proxy_for_this_mac(monkeypatch):
    # Playwright stops sending loopback through a context's proxy when this is set: never for NEO
    monkeypatch.setenv("PLAYWRIGHT_DISABLE_FORCED_CHROMIUM_PROXIED_LOOPBACK", "1")
    seen = []

    async def start():
        seen.append(os.environ.get("PLAYWRIGHT_DISABLE_FORCED_CHROMIUM_PROXIED_LOOPBACK"))
        return FakePlaywright({"html": BUILT})

    r = render._Renderer(start=start)
    try:
        _render(r, "https://site.test/a")
    finally:
        r.close_now()
    assert seen == [None]


def test_a_redirect_the_browser_follows_itself_is_stopped_by_the_proxy(browser):
    r, started = browser({"html": BUILT, "redirect_to": "http://10.0.0.5/secret"})
    with pytest.raises(render.RenderError, match="where the page led"):
        _render(r, "https://site.test/go")


def test_a_page_that_never_loads_is_given_up_on(browser, monkeypatch):
    monkeypatch.setattr(render, "_GRACE_S", 0.2)
    r, started = browser({"hang": True})
    with pytest.raises(render.RenderError, match="too long"):
        _render(r, "https://site.test/slow", timeout_s=0.2)
    assert started[0].browsers[0].contexts[0].closed


def test_a_page_whose_network_never_goes_quiet_is_read_once_its_text_settles(browser, monkeypatch):
    monkeypatch.setattr(render, "_STABLE_S", 0.3)
    r, _ = browser({"busy": True, "html": BUILT, "text": WORDS * 10})
    out = _render(r, "https://site.test/live", timeout_s=5.0)
    assert out.seconds < 3.0 and "herons" in out.text


def test_stopping_the_request_closes_the_page(browser):
    r, started = browser({"hang": True})

    async def go():
        t = asyncio.ensure_future(r.render("https://site.test/slow", _guard, timeout_s=30, max_bytes=1000))
        await asyncio.sleep(0.3)
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t
        for _ in range(50):
            if started and started[0].browsers[0].contexts[0].closed:
                break
            await asyncio.sleep(0.05)

    asyncio.run(go())
    assert started[0].browsers[0].contexts[0].closed


def test_the_browser_closes_when_idle_and_comes_back_when_needed(browser):
    r, started = browser({"html": BUILT}, idle_s=0.05)
    _render(r, "https://site.test/a")

    async def wait_closed():
        for _ in range(60):
            if started[0].stopped:
                return
            await asyncio.sleep(0.05)

    asyncio.run(wait_closed())
    assert started[0].browsers[0].closed and started[0].stopped
    _render(r, "https://site.test/b")
    assert len(started) == 2 and r.launches == 2


def test_a_browser_that_will_not_start_is_not_retried_at_once(browser):
    r, started = browser({}, fail=True)
    with pytest.raises(render.RenderError, match="didn't start"):
        _render(r, "https://site.test/a")
    calls = []
    r._start = lambda: calls.append(1)
    with pytest.raises(render.RenderError, match="didn't start"):
        _render(r, "https://site.test/a")
    assert calls == []


def test_the_suite_never_starts_a_real_browser():
    if os.environ.get("NEO_TEST_RENDER") == "1":
        pytest.skip("the real browser is on for this run")
    with pytest.raises(render.RenderError, match="no real browser"):
        asyncio.run(render._start_playwright())


# ---- the real thing (optional) --------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NEO_TEST_RENDER") != "1", reason="real headless Chromium: NEO_TEST_RENDER=1"
)
def test_a_real_script_built_page_is_read():
    async def go():
        try:
            fetched = await web.web_fetch({"url": "https://quotes.toscrape.com/js/"}, ToolContext())
            page = await render.render("https://quotes.toscrape.com/js/")
        finally:
            await render.shutdown()
        return fetched, page

    fetched, page = asyncio.run(go())
    assert "Einstein" in page.text and page.status == 200 and not page.capped
    assert "Einstein" in fetched
