"""Read a page that its own scripts build, in a headless Chromium nobody sees (for web_fetch).

Some pages arrive as an empty shell ("enable JavaScript", a root <div> and a bundle of scripts):
the words only exist once the scripts have run. web_fetch hands those here. Each page gets a
fresh, private browser context — no profile, no cookies, never NEO's visible browser — which is
thrown away once the text is out. One headless Chromium serves every call and shuts itself down
after a couple of idle minutes, so a second page in a row is quick.

Reading a page only reads: the page may not submit a form (a navigation that isn't a GET), nor
open a link that carries data (a token in it — a script there could act on it). The browser says
what it is (Chromium's own headless user agent, automation not hidden): a site that turns away
automated clients is taken at its word.

Nothing the page asks for may reach this Mac or the local network. Two guards:

* every request the page makes is checked (route interception, the same rule as web_fetch:
  ``policy.private_host``), and pictures, video and fonts are never downloaded;
* all of the context's traffic goes through a small proxy of our own that connects only to
  public addresses — the address it checked, so a name that re-resolves somewhere else later
  can't slip past. It sees what route interception can't: redirects, websockets, requests a
  worker makes. It also counts the bytes, and cuts the page off at a limit.

The browser runs on a thread of its own, so its traffic never competes with NEO's voice loop.
"""

from __future__ import annotations

import asyncio
import atexit
import ipaddress
import os
import re
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlsplit

from neo2.core import effects

IDLE_S = 120.0  # the headless browser closes after this long without a page to read
TIMEOUT_S = 12.0  # loading and settling one page, at most
MAX_BYTES = 25_000_000  # everything one page may download
MAX_HTML = 3_000_000  # characters of the rendered page handed back
MAX_TEXT = 200_000
CONCURRENCY = 3  # pages rendered at once (web_fetch is parallel-safe)
_GRACE_S = 5.0  # past its own deadline, a render is abandoned after this much longer
_BROKEN_S = 600.0  # after a failed launch, don't try again for this long

# What a page's words never depend on: skipped, which is also most of a page's weight.
_SKIP_TYPES = frozenset({"image", "media", "font", "manifest", "texttrack", "ping"})  # ping: beacons
# How long the page's text must stay unchanged to count as built, when the network never goes
# quiet (analytics beacons, long polls) — and how much text that takes.
_STABLE_S = 2.0
_SETTLED_CHARS = 400
_POLL_S = 0.25

_ARGS = [
    "--disable-quic",  # all traffic over TCP, through the proxy
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--dns-prefetch-disable",
    "--mute-audio",
]


class RenderError(Exception):
    """The page couldn't be rendered; the message says why, for the user."""


@dataclass
class Rendered:
    url: str  # where the page ended up
    title: str
    html: str  # the page as its scripts left it
    text: str  # what the page shows (document.body.innerText)
    seconds: float
    status: int = 0  # the HTTP status of the page it ended on (0 = unknown)
    capped: bool = False  # cut off at MAX_BYTES


# ---- when a page needs rendering ------------------------------------------------------------------

# Every pattern stops at the next tag ([^<>], no .* across the page), so a hostile page is judged
# in linear time: this runs on every fetched page, and a regex holds the GIL while it runs.
_SCRIPT = re.compile(r"<script\b", re.I)
_NOSCRIPT = re.compile(r"<noscript\b[^<>]*>((?:(?!</?noscript\b)[\s\S])*)</noscript>", re.I)
_TAG = re.compile(r"<[^<>]*>")
# An empty element a framework mounts the page into: <div id="root"></div>, <app-root></app-root>
_MOUNT = re.compile(
    r"<(?P<tag>div|main|section|body)\b[^<>]*?\bid\s*=\s*[\"']?(?:root|app|__next|__nuxt|___gatsby|svelte|application)"
    r"(?:[\"'\s/][^<>]*)?>\s*</(?P=tag)>|<app-root\b[^<>]*>\s*</app-root>",
    re.I,
)
_SCAN = 2_000_000  # characters of a page looked at (a shell's markers come early)
THIN_CHARS = 300  # the server sent (next to) nothing to read
_SHELL = 2000  # little enough that a page asking for JavaScript is a shell
_NO_MAIN = 150  # an article extractor found no main text…
_LITTLE = 1000  # …in a page with only a little text around it


def needs_render(html: str, main: str, full: str) -> bool:
    """Is this page built by its scripts? `main` is its extracted main text, `full` all the text
    the server sent. Without a script there's nothing a browser would add."""
    html = html[:_SCAN]
    if not _SCRIPT.search(html):
        return False
    m, f = len(main.strip()), len(full.strip())
    if max(m, f) < THIN_CHARS:
        return True
    if max(m, f) < _SHELL and (_asks_for_js(html) or _MOUNT.search(html)):
        return True
    return m < _NO_MAIN and f < _LITTLE


def _asks_for_js(html: str) -> bool:
    return any("javascript" in _TAG.sub(" ", inner).lower() for inner in _NOSCRIPT.findall(html))


# ---- the proxy: public addresses only -------------------------------------------------------------


def public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """An address on the public internet (not this Mac, the LAN, a VPN's shared range…)."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip.is_site_local:  # fec0::/10 — deprecated, but still a LAN's
            return False
    return ip.is_global and not ip.is_multicast


def _hostport(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


class _Gate:
    """A forward proxy for one page: CONNECT tunnels and plain-HTTP requests, only to public
    addresses, cut off after `max_bytes` from the network."""

    def __init__(self, guard: Callable[[str], str], max_bytes: int, *, allow: Callable = public_ip):
        self._guard, self._allow, self.max_bytes = guard, allow, max_bytes
        self.bytes, self.capped = 0, False
        self.refused: dict[str, str] = {}  # host → why
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task] = set()
        self.port = 0

    @property
    def server(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> _Gate:
        self._server = await asyncio.start_server(self._client, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def close(self) -> None:
        if self._server:
            self._server.close()
        for w in list(self._writers):
            w.close()
        for t in list(self._tasks):
            t.cancel()
        if self._server:
            with suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)

    async def addresses(self, host: str, port: int) -> list[str]:
        """The checked public addresses to connect to ([] = refused)."""
        why = await asyncio.to_thread(self._guard, f"http://{_hostport(host, port)}/")
        if not why:
            try:
                infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except (OSError, UnicodeError):
                return []  # doesn't resolve: the browser reports the failure
            ips = list(dict.fromkeys(info[4][0] for info in infos))
            # like policy.private_host: one private answer is enough to refuse the name
            if ips and all(self._allow(ipaddress.ip_address(ip.split("%")[0])) for ip in ips):
                return ips
            why = "it's on this Mac or the local network"
        self.refused[host.lower()] = why
        return []

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        self._tasks.add(task)
        self._writers.add(writer)
        upstream: asyncio.StreamWriter | None = None
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            line, *headers = head[:-4].decode("latin-1").split("\r\n")
            method, target, version = line.split(" ", 2)
            if method.upper() == "CONNECT":
                u = urlsplit("//" + target)
            else:
                u = urlsplit(target)
                if u.scheme.lower() != "http":
                    return await _reply(writer, b"400 Bad Request")
            host, port = (u.hostname or ""), (u.port or 80)
            if self.capped or not host:
                return await _reply(writer, b"403 Forbidden")
            ips = await self.addresses(host, port)
            if not ips:
                return await _reply(writer, b"403 Forbidden")
            for ip in ips[:3]:  # in the system's order of preference (IPv6 first where it works)
                try:
                    up_r, upstream = await asyncio.wait_for(asyncio.open_connection(ip, port), 4)
                    break
                except (OSError, TimeoutError):
                    continue
            else:
                return await _reply(writer, b"502 Bad Gateway")
            self._writers.add(upstream)
            if method.upper() == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await asyncio.gather(
                    self._pipe(reader, upstream, count=False), self._pipe(up_r, writer, count=True)
                )
                return
            # Plain HTTP: one request per connection — the next one may be for another host, and
            # must be checked on its own.
            keep = [h for h in headers if h.split(":", 1)[0].strip().lower() not in _HOP_HEADERS]
            path = (u.path or "/") + ("?" + u.query if u.query else "")
            upstream.write(
                "\r\n".join([f"{method} {path} {version}", *keep, "Connection: close", "", ""]).encode(
                    "latin-1"
                )
            )
            send = asyncio.ensure_future(self._pipe(reader, upstream, count=False))  # a request body
            try:
                first = await _response_head(up_r)
                await self._pipe(up_r, writer, count=True, first=first)
            finally:
                send.cancel()
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError, ValueError, OSError):
            pass
        finally:
            for w in (writer, upstream):
                if w is not None:
                    w.close()
                    self._writers.discard(w)
            self._tasks.discard(task)

    async def _pipe(
        self, r: asyncio.StreamReader, w: asyncio.StreamWriter, *, count: bool, first: bytes = b""
    ) -> None:
        data = first
        try:
            while True:
                if data:
                    if count:
                        self.bytes += len(data)
                        if self.bytes > self.max_bytes:
                            self.capped = True
                            for other in list(self._writers):
                                other.close()
                            return
                    w.write(data)
                    await w.drain()
                data = await r.read(65536)
                if not data:
                    return
        except (OSError, ConnectionError):
            return
        finally:
            w.close()


_HOP_HEADERS = frozenset({"proxy-connection", "proxy-authorization", "connection", "keep-alive"})


async def _reply(writer: asyncio.StreamWriter, status: bytes) -> None:
    writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    with suppress(OSError, ConnectionError):
        await writer.drain()


async def _response_head(r: asyncio.StreamReader) -> bytes:
    """A plain-HTTP response's head, saying the connection closes after it (so the browser never
    sends a second request down a tunnel that was checked for the first)."""
    head = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 20)
    line, *headers = head[:-4].decode("latin-1").split("\r\n")
    if line.split(" ", 2)[1:2] and line.split(" ", 2)[1].startswith("1"):
        return head  # 100 Continue: relayed as it is
    keep = [h for h in headers if h.split(":", 1)[0].strip().lower() not in _HOP_HEADERS]
    return (
        "\r\n".join([line, *keep, "Connection: close", "Proxy-Connection: close"]).encode("latin-1")
        + b"\r\n\r\n"
    )


# ---- the headless browser -------------------------------------------------------------------------

# Set, Playwright lets a context reach loopback and link-local addresses directly instead of through
# its proxy (the driver inherits NEO's environment): never for this browser.
_LOOPBACK_ENV = "PLAYWRIGHT_DISABLE_FORCED_CHROMIUM_PROXIED_LOOPBACK"


def _not_to_open(method: str, url: str) -> str:
    """Why reading a page must not open this page or frame ("" = fine): reading only reads."""
    if (method or "GET").upper() not in ("GET", "HEAD"):
        return "the page tried to send a form"
    if urlsplit(url).scheme.lower() in ("http", "https"):
        from neo2.core.security.policy import url_carries_data

        if url_carries_data(url):
            return "that link carries data (a token, say), which a script there could act on"
    return ""


async def _start_playwright():
    from playwright.async_api import async_playwright

    effects.guard("browser", "start Playwright's headless Chromium")
    return await async_playwright().start()


_SNAPSHOT_JS = (
    "([h, t]) => ({ title: document.title || '',"
    " html: document.documentElement ? document.documentElement.outerHTML.slice(0, h) : '',"
    " text: document.body ? document.body.innerText.slice(0, t) : '' })"
)
_LEN_JS = "() => document.body ? document.body.innerText.length : 0"


class _Renderer:
    """One headless Chromium on a thread of its own, shared by every render, closed when idle."""

    def __init__(self, *, start=None, idle_s: float = IDLE_S, concurrency: int = CONCURRENCY) -> None:
        self._start = start  # how Playwright starts (the real one when None)
        self._idle_s, self._concurrency = idle_s, concurrency
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # on the renderer's own loop only
        self._pw = self._browser = None
        self._launch_lock: asyncio.Lock | None = None
        self._sem: asyncio.Semaphore | None = None
        self._active = 0
        self._idle: asyncio.TimerHandle | None = None
        self._broken_until, self._broken_why = 0.0, ""
        self.launches = 0

    # -- the thread --------------------------------------------------------------------------------
    def _thread_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop = asyncio.new_event_loop()
                t = threading.Thread(target=_run_forever, args=(loop,), name="neo-render", daemon=True)
                t.start()
                self._loop, self._thread = loop, t
                atexit.register(self.close_now)
            return self._loop

    async def render(
        self, url: str, guard: Callable[[str], str], *, timeout_s: float, max_bytes: int
    ) -> Rendered:
        loop = self._thread_loop()
        fut = asyncio.run_coroutine_threadsafe(self._render(url, guard, timeout_s, max_bytes), loop)
        try:
            # launching Chromium the first time comes on top of the page's own time
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout_s + _GRACE_S + 20)
        except TimeoutError:
            fut.cancel()
            raise RenderError("the page took too long to load") from None

    async def close(self) -> None:
        """Close the browser (from any loop)."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(self._stop_browser(), loop)
        with suppress(Exception):
            await asyncio.wait_for(asyncio.wrap_future(fut), 4)

    def close_now(self, timeout: float = 4.0) -> None:
        """Close the browser and stop the thread, from outside any loop (exit)."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._stop_browser(), loop).result(timeout)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    # -- on the renderer's loop --------------------------------------------------------------------
    async def _ready(self):
        if self._launch_lock is None:
            self._launch_lock = asyncio.Lock()
        async with self._launch_lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            await self._stop_browser()
            if time.monotonic() < self._broken_until:
                raise RenderError(self._broken_why)
            try:
                os.environ.pop(_LOOPBACK_ENV, None)
                self._pw = await (self._start or _start_playwright)()
                # Contexts each get their own proxy; anything outside one goes nowhere.
                self._browser = await self._pw.chromium.launch(
                    headless=True, args=_ARGS, proxy={"server": "http://per-context"}, timeout=20000
                )
                self.launches += 1
            except Exception as e:  # noqa: BLE001 — not installed, or it wouldn't start
                await self._stop_browser()
                self._broken_until = time.monotonic() + _BROKEN_S
                first = (str(e).strip().splitlines() or [type(e).__name__])[0][:160]
                self._broken_why = f"the page-reading browser didn't start ({first})"
                raise RenderError(self._broken_why) from None
            return self._browser

    async def _stop_browser(self) -> None:
        browser, pw, self._browser, self._pw = self._browser, self._pw, None, None
        if browser is not None:
            with suppress(Exception):
                await asyncio.wait_for(browser.close(), 3)
        if pw is not None:
            with suppress(Exception):
                await asyncio.wait_for(pw.stop(), 3)

    def _arm_idle(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
        self._idle = asyncio.get_running_loop().call_later(
            self._idle_s, lambda: asyncio.ensure_future(self._close_if_idle())
        )

    async def _close_if_idle(self) -> None:
        if self._active == 0 and self._launch_lock is not None and not self._launch_lock.locked():
            async with self._launch_lock:
                if self._active == 0:
                    await self._stop_browser()

    async def _render(self, url: str, guard, timeout_s: float, max_bytes: int) -> Rendered:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._concurrency)
        async with self._sem:
            self._active += 1
            if self._idle is not None:
                self._idle.cancel()
            try:
                browser = await self._ready()
                gate = await _Gate(guard, max_bytes).start()
                try:
                    return await asyncio.wait_for(
                        self._read(browser, gate, url, guard, timeout_s), timeout_s + _GRACE_S
                    )
                except TimeoutError:
                    raise RenderError("the page took too long to load") from None
                finally:
                    await gate.close()
            except RenderError:
                raise
            except Exception as e:  # noqa: BLE001 — the browser died, a socket wouldn't open…
                raise RenderError(_first_line(e)) from None
            finally:
                self._active -= 1
                if self._active == 0:
                    self._arm_idle()

    async def _read(self, browser, gate: _Gate, url: str, guard, timeout_s: float) -> Rendered:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        deadline = t0 + timeout_s
        ctx = await browser.new_context(
            proxy={"server": gate.server},
            accept_downloads=False,
            service_workers="block",
            java_script_enabled=True,
            viewport={"width": 1280, "height": 900},
        )
        try:
            ctx.set_default_timeout(5000)
            blocked: list[str] = []
            checked: dict[tuple, str] = {}

            async def why_not(u: str) -> str:
                sp = urlsplit(u)
                scheme = sp.scheme.lower()
                if scheme in ("data", "blob", "about"):
                    return ""
                if scheme not in ("http", "https"):
                    return f"{scheme or 'that'} links aren't followed"
                key = (scheme, sp.hostname, sp.port)
                if key not in checked:
                    checked[key] = await loop.run_in_executor(None, guard, u)
                return checked[key]

            async def route(r) -> None:
                req = r.request
                try:
                    if req.resource_type in _SKIP_TYPES:
                        return await r.abort("blockedbyclient")
                    nav = req.is_navigation_request()
                    why = await why_not(req.url)
                    if nav and not why:
                        why = _not_to_open(req.method, req.url)
                    if why:
                        if nav:
                            blocked.append(why)
                        return await r.abort("blockedbyclient")
                    await r.continue_()
                except Exception:  # noqa: BLE001 — a URL it can't make sense of, or the page closed
                    with suppress(Exception):
                        await r.abort("blockedbyclient")

            await ctx.route("**/*", route)
            page = await ctx.new_page()
            page.on("popup", lambda p: asyncio.ensure_future(_quietly(p.close())))
            status = [0]

            def on_response(resp) -> None:
                with suppress(Exception):
                    if resp.frame == page.main_frame and resp.request.is_navigation_request():
                        status[0] = resp.status

            page.on("response", on_response)
            try:
                await page.goto(url, wait_until="commit", timeout=max(1.0, deadline - loop.time()) * 1000)
            except Exception as e:  # noqa: BLE001
                if why := (blocked or [*gate.refused.values()] or [""])[0]:
                    raise RenderError(f"not reading it — {why}") from None
                raise RenderError(_first_line(e)) from None
            await _settle(page, deadline, gate)
            snap = None
            for _ in range(3):  # the page may be navigating right then
                try:
                    snap = await page.evaluate(_SNAPSHOT_JS, [MAX_HTML, MAX_TEXT])
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(0.3)
            if not snap:
                raise RenderError("the page kept changing and couldn't be read")
            if why := gate.refused.get((urlsplit(page.url).hostname or "").lower()):
                raise RenderError(f"not reading it — where the page led: {why}")
            return Rendered(
                url=page.url,
                title=str(snap.get("title") or ""),
                html=str(snap.get("html") or ""),
                text=str(snap.get("text") or ""),
                seconds=round(loop.time() - t0, 2),
                status=status[0],
                capped=gate.capped,
            )
        finally:
            with suppress(Exception):
                await asyncio.wait_for(ctx.close(), 3)


def _run_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        with suppress(Exception):
            loop.close()


async def _settle(page, deadline: float, gate: _Gate) -> None:
    """Wait for the page to be built: the network quiet (Playwright's networkidle), or its text
    unchanged for a while — whichever comes first, and never past the deadline."""
    loop = asyncio.get_running_loop()

    def left_ms() -> float:
        return max(1.0, (deadline - loop.time()) * 1000)

    with suppress(Exception):
        await page.wait_for_load_state("domcontentloaded", timeout=left_ms())
    idle = asyncio.ensure_future(page.wait_for_load_state("networkidle", timeout=left_ms()))
    idle.add_done_callback(lambda t: t.cancelled() or t.exception())
    last, since = -1, loop.time()
    try:
        while not idle.done() and loop.time() < deadline and not gate.capped:
            try:
                n = await page.evaluate(_LEN_JS)
            except Exception:  # noqa: BLE001 — navigating
                n = -1
            now = loop.time()
            if n != last:
                last, since = n, now
            elif n >= _SETTLED_CHARS and now - since >= _STABLE_S:
                break
            await asyncio.wait({idle}, timeout=_POLL_S)
    finally:
        if not idle.done():
            idle.cancel()


def _first_line(e: Exception) -> str:
    s = (str(e).strip().splitlines() or [type(e).__name__])[0]
    s = re.sub(r"^\w*Error:\s*", "", s)
    return s[:160] or "the page didn't load"


async def _quietly(aw) -> None:
    with suppress(Exception):
        await aw


_renderer = _Renderer()


async def render(
    url: str,
    guard: Callable[[str], str] | None = None,
    *,
    timeout_s: float = TIMEOUT_S,
    max_bytes: int = MAX_BYTES,
) -> Rendered:
    """Load `url` in the headless browser, let its scripts build it, and return what it shows.
    `guard(url)` says why a URL must not be loaded ("" = fine); web_fetch's own rule by default.
    Raises RenderError."""
    if guard is None:
        from neo2.core.security.policy import private_host as guard
    return await _renderer.render(url, guard, timeout_s=timeout_s, max_bytes=max_bytes)


async def shutdown() -> None:
    """Close the headless browser now (NEO is quitting)."""
    await _renderer.close()
