"""The chat's browser card on a real headless Chromium (opt-in: NEO_TEST_RENDER=1, like the other real
browser check). Local pages only (set_content and 127.0.0.1 servers): what the user types stays
theirs, a dialog names the frame that opened it, and a page's icon is fetched without the profile."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from neo2.core.agent.registry import ToolContext, registry
from neo2.core.events import current_job
from neo2.core.tools import browser, browser_live

pytestmark = [
    pytest.mark.skipif(os.environ.get("NEO_TEST_RENDER") != "1", reason="real headless Chromium: NEO_TEST_RENDER=1"),
    pytest.mark.usefixtures("real_browser"),
]

FORM = """<!doctype html><title>Form</title><main><h1>Profile</h1>
<label>Security answer <input id=a name=answer type=text></label><br>
<label>Nickname <input id=b name=nick type=text></label><br>
<label>Note <div id=c contenteditable=true role=textbox aria-label=Note style="border:1px solid;min-height:40px"></div></label>
<label>Size <select id=s name=size><option>Small</option><option>Medium</option><option>Large</option></select></label>
<p>Some text on the page.</p></main>"""


class Conn:
    def __init__(self) -> None:
        self.json: list[dict] = []
        self.pending = None

    def send_json(self, text: str) -> bool:
        self.json.append(json.loads(text))
        return True

    def offer_frame(self, payload) -> bool:
        r = self.pending is not None
        self.pending = payload
        return r

    def of(self, type_: str) -> list[dict]:
        return [m["data"] for m in self.json if m["type"] == type_]


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, type_: str, data: dict, *, recent: bool = False, replay: dict | None = None) -> None:
        self.sent.append((type_, json.loads(json.dumps(data, default=str))))

    def of(self, type_: str) -> list[dict]:
        return [d for t, d in self.sent if t == type_]


@pytest.fixture
async def real(monkeypatch):
    from neo2.core import config
    from neo2.core.tools import load_all

    monkeypatch.setenv("NEO_BROWSER_VIEW", "chat")
    monkeypatch.setattr(config, "_settings", None)
    browser_live.reset()
    load_all()
    live = browser_live.live
    ch = Channel()
    live.channel = ch
    live.hooks.resume = live.hooks.decline = None
    monkeypatch.setattr(browser, "_b", browser._Browser())
    yield live, ch
    await browser.shutdown()
    browser_live.reset()
    live.channel = None


def _serve(body) -> int:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            status, headers, out = body(self)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


class User:
    """The user's pointer and keys, as the overlay sends them."""

    def __init__(self, live) -> None:
        self.live, self.conn, self.seq = live, Conn(), 0
        live.command(self.conn, {"cmd": "browser_hello", "v": 1, "binary": True, "formats": ["jpeg"], "surface": "overlay"})
        self.view()

    def view(self) -> None:
        self.live.command(self.conn, {"cmd": "browser_view", "sid": self.live.s.sid, "surface": "docked", "css_w": 819,
                                      "css_h": 472, "dpr": 2, "visible": True})  # fmt: skip

    def send(self, **m) -> None:
        self.seq += 1
        self.live.command(self.conn, {"cmd": "browser_input", "sid": self.live.s.sid, "seq": self.seq, **m})

    async def click(self, page, sel: str) -> None:
        x, y = await page.evaluate(f"(() => {{ const r = document.querySelector('{sel}').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()")
        self.send(type="mousedown", x=x, y=y, button="left", buttons=1, clicks=1, mods=0)
        self.send(type="mouseup", x=x, y=y, button="left", buttons=0, clicks=1, mods=0)
        await asyncio.sleep(0.2)

    def type(self, text: str) -> None:
        for c in text:
            code = f"Key{c.upper()}" if c.isalpha() else f"Digit{c}"
            self.send(type="keydown", key=c, code=code, mods=0, text=c)
            self.send(type="keyup", key=c, code=code, mods=0)

    def key(self, key: str) -> None:
        self.send(type="keydown", key=key, code=key, mods=0)
        self.send(type="keyup", key=key, code=key, mods=0)

    def control(self, action: str, **extra) -> None:
        self.live.command(self.conn, {"cmd": "browser_control", "sid": self.live.s.sid, "req": action, "action": action, **extra})


async def _read(job: str) -> str:
    tok = current_job.set(job)
    try:
        return (await registry().invoke("browser_read", {}, ToolContext())).text
    finally:
        current_job.reset(tok)


async def _marked(page) -> list[str]:
    return await page.evaluate("[...document.querySelectorAll('[data-neo-user-filled]')].map(e => e.id)")


async def test_what_the_user_types_stays_theirs_in_every_read_of_the_page(real):
    live, ch = real
    tok = current_job.set("j1")
    page = await browser._b.page()
    await page.set_content(FORM)
    current_job.reset(tok)
    await _read("j1")
    user = User(live)
    await asyncio.sleep(0.4)

    # A normal typist: the answer, then Tab at once (nothing typed in the next field).
    assert await live.take() == ""
    await user.click(page, "#a")
    user.type("tiger42")
    user.key("Tab")
    await asyncio.sleep(0.8)
    assert await page.evaluate("document.querySelector('#a').value") == "tiger42"
    assert await _marked(page) == ["a"]
    user.control("release")
    await asyncio.sleep(0.3)
    assert "tiger42" not in await _read("j1")

    # An editor's words are left out of the page's text too; a <select> choice is theirs as well.
    assert await live.take() == ""
    await user.click(page, "#c")
    user.type("mypinisfour")
    await asyncio.sleep(0.5)
    await user.click(page, "#s")
    (sel,) = [m for t, m in ch.sent if t == "browser_select"]
    user.control("select", select_id=sel["select_id"], value="Large")
    await asyncio.sleep(0.5)
    assert sorted(await _marked(page)) == ["a", "c", "s"]
    user.control("release")
    await asyncio.sleep(0.3)
    text = await _read("j1")
    assert "mypinisfour" not in text and "tiger42" not in text
    assert '"Size" [filled by you]' in text

    # The card folds; the user's next request (a new card) reads the same page: still masked.
    assert await live.take() == ""
    await user.click(page, "#b")
    user.type("zebra77")
    await asyncio.sleep(0.5)
    user.control("release")
    await asyncio.sleep(0.3)
    await live.end("done")
    text = await _read("j2")
    assert live.s is not None and "zebra77" not in text and "tiger42" not in text
    poster = ch.of("browser_close")[-1]["poster"]
    assert poster and poster.startswith("data:image/jpeg")  # a picture of the page, the fields painted over

    # NEO's own typing afterwards is NEO's: it isn't marked.
    tok = current_job.set("j2")
    try:
        await page.evaluate("document.querySelector('#b').value = ''; document.querySelector('#b').removeAttribute('data-neo-user-filled')")
        text = await _read("j2")
        n = next(ln.split("]")[0][1:] for ln in text.splitlines() if 'textbox "Nickname"' in ln)
        await registry().invoke("browser_type", {"target": n, "text": "robot"}, ToolContext())
    finally:
        current_job.reset(tok)
    assert "b" not in await _marked(page)


async def test_a_dialog_from_an_embedded_frame_names_that_frame(real):
    live, ch = real
    ad = _serve(lambda h: (200, {"Content-Type": "text/html"}, b"<script>setTimeout(() => prompt('Your session expired. Re-enter your password:'), 1200)</script>ad"))
    top = _serve(lambda h: (200, {"Content-Type": "text/html"}, f"<title>My Bank</title><h1>My Bank</h1><iframe src='http://127.0.0.1:{ad}/ad'></iframe>".encode()))
    tok = current_job.set("j1")
    try:
        await registry().invoke("browser_open", {"url": f"http://localhost:{top}/"}, ToolContext())
    finally:
        current_job.reset(tok)
    assert await live.take() == ""
    for _ in range(40):
        if ch.of("browser_dialog"):
            break
        await asyncio.sleep(0.1)
    (d,) = ch.of("browser_dialog")
    assert d["origin"] == "127.0.0.1" and d["embedded"] is True


async def test_a_pages_icon_is_fetched_without_the_profiles_cookies(real):
    live, ch = real
    seen: list[dict] = []
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64

    def internal(h):
        seen.append({"path": h.path, "cookie": h.headers.get("Cookie")})
        return 200, {"Content-Type": "image/png"}, png

    b = _serve(internal)
    a = _serve(lambda h: (200, {"Content-Type": "text/html"}, f"<title>Deals</title><link rel=icon sizes=16x16 href='http://127.0.0.1:{b}/admin/icon?x=1'><p>Great deals</p>".encode()))
    tok = current_job.set("j1")
    page = await browser._b.page()
    await page.context.add_cookies([{"name": "session", "value": "ADMIN-SESSION", "domain": "127.0.0.1", "path": "/", "sameSite": "Strict"}])
    try:
        await registry().invoke("browser_open", {"url": f"http://localhost:{a}/"}, ToolContext())
    finally:
        current_job.reset(tok)
    for _ in range(30):
        if seen:
            break
        await asyncio.sleep(0.1)
    assert seen and all(r["cookie"] is None for r in seen)
