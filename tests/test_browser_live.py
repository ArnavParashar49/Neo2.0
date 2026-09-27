"""NEO's browser live in the chat (neo/tools/browser_live.py): the card's protocol, frames and
credits, the screencast's life, the user's input (and every gate on it), take / hand back / stop,
hand-offs and approvals, and the websocket lane — on fake Playwright pages and a fake CDP session:
no Chromium starts, nothing reaches the network."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import struct
import time

import pytest

from neo.agent import confirm
from neo.agent.registry import ToolContext, registry
from neo.events import bus, current_job
from neo.tools import browser, browser_live
from neo.tools.browser_live import key_event, pack_frame, page_viewport, unpack_frame


def _jpeg(w: int = 64, h: int = 40) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (w, h), (200, 220, 255)).save(out, "JPEG", quality=60)
    return out.getvalue()


JPEG = _jpeg()
B64 = base64.b64encode(JPEG).decode()


# ---- fakes ---------------------------------------------------------------------------------------------


class FakeCDP:
    """A CDP session: records what NEO sends, answers what the card asks, emits page events."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.handlers: dict[str, list] = {}
        self.fail: set[str] = set()
        self.history = (0, 1)

    def on(self, event: str, fn) -> None:
        self.handlers.setdefault(event, []).append(fn)

    def emit(self, event: str, params: dict) -> None:
        for fn in self.handlers.get(event, []):
            fn(params)

    async def send(self, method: str, params: dict | None = None):
        self.sent.append((method, params or {}))
        if method in self.fail:
            raise RuntimeError("boom")
        if method == "Page.getLayoutMetrics":
            return {"cssVisualViewport": {"pageX": 0, "pageY": 120, "clientWidth": 1024, "clientHeight": 590}}
        if method == "Page.captureScreenshot":
            return {"data": B64}
        if method == "Page.getNavigationHistory":
            i, n = self.history
            return {"currentIndex": i, "entries": [{"url": "", "title": ""}] * n}
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "MAIN"}}}
        return {}

    async def detach(self) -> None:
        pass

    def calls(self, method: str) -> list[dict]:
        return [p for m, p in self.sent if m == method]


class FakeLoc:
    def __init__(self, page: FakePage, sel: str) -> None:
        self.page, self.sel = page, sel

    @property
    def first(self) -> FakeLoc:
        return self

    async def select_option(self, value=None, timeout=None) -> None:
        self.page.log.append(("select_option", self.sel, value))


class FakeContext:
    def __init__(self) -> None:
        self.cdps: list[FakeCDP] = []

    async def new_cdp_session(self, page) -> FakeCDP:
        cdp = FakeCDP()
        self.cdps.append(cdp)
        return cdp


class FakePage:
    def __init__(self, url: str = "https://shop.example/mug") -> None:
        self.url = url
        self.context = FakeContext()
        self.log: list[tuple] = []
        self.handlers: dict[str, list] = {}
        self.viewport_size = {"width": 1024, "height": 590}
        self.closed = False
        self.scripts: dict[str, object] = {}  # a script's first words → what it returns

    def is_closed(self) -> bool:
        return self.closed

    def on(self, event: str, fn) -> None:
        self.handlers.setdefault(event, []).append(fn)

    async def title(self) -> str:
        return "Blue mug — Shop"

    async def evaluate(self, script, arg=None):
        self.log.append(("evaluate", script[:30], arg))
        for key, value in self.scripts.items():
            if key in script:
                return value(arg) if callable(value) else value
        if "scrollY" in script and len(script) < 20:
            return 120
        return None

    async def set_viewport_size(self, size: dict) -> None:
        self.viewport_size = dict(size)
        self.log.append(("viewport", size["width"], size["height"]))

    def locator(self, sel: str) -> FakeLoc:
        return FakeLoc(self, sel)

    async def go_back(self, **kw):
        self.log.append(("back",))

    async def go_forward(self, **kw):
        self.log.append(("forward",))

    async def reload(self, **kw):
        self.log.append(("reload",))

    async def goto(self, url, **kw):
        self.log.append(("goto", url))

    async def bring_to_front(self) -> None:
        pass

    @property
    def cdp(self) -> FakeCDP:
        return self.context.cdps[-1]


class FakeConn:
    """The websocket server's client: what the card sends it."""

    def __init__(self) -> None:
        self.json: list[dict] = []
        self.frames: list[bytes | str] = []
        self.pending: bytes | str | None = None

    def send_json(self, text: str) -> bool:
        self.json.append(json.loads(text))
        return True

    def offer_frame(self, payload) -> bool:
        replaced = self.pending is not None
        self.pending = payload
        return replaced

    def flush(self) -> None:
        """The drain task sends the slot."""
        if self.pending is not None:
            self.frames.append(self.pending)
            self.pending = None

    def of(self, type_: str) -> list[dict]:
        return [m["data"] for m in self.json if m["type"] == type_]


class Channel:
    """The websocket server's broadcast, as the card sees it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, bool]] = []

    def broadcast(self, type_: str, data: dict, *, recent: bool = False) -> None:
        self.sent.append((type_, json.loads(json.dumps(data, default=str)), recent))

    def of(self, type_: str) -> list[dict]:
        return [d for t, d, _ in self.sent if t == type_]


@pytest.fixture
async def card(monkeypatch):
    """The chat's browser card, a channel that records what it broadcasts, and a fake page."""
    from neo import config

    monkeypatch.setenv("NEO_BROWSER_VIEW", "chat")
    monkeypatch.setattr(config, "_settings", None)
    browser_live.reset()
    live = browser_live.live
    ch = Channel()
    live.channel = ch
    live.hooks.resume = live.hooks.decline = None
    monkeypatch.setattr(browser_live, "_PRESENT_S", 0)
    monkeypatch.setattr(browser_live, "_THUMB_AFTER_S", 0)
    b = browser._Browser()
    monkeypatch.setattr(browser, "_b", b)
    confirm.cancel("test")
    yield live, ch
    confirm.cancel("test")
    pending = list(live._tasks)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    browser_live.reset()
    live.channel = None


async def _open(live, ch, page: FakePage | None = None, *, job: str = "j1"):
    """A session is open on a bound page; one overlay watches it docked."""
    page = page or FakePage()
    browser._b._current = page
    live.launched()
    live.claim(job)
    await live.bind(page)
    conn = FakeConn()
    live.command(conn, {"cmd": "browser_hello", "v": 1, "binary": True, "formats": ["jpeg"], "surface": "overlay"})
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 819, "css_h": 472,
                        "dpr": 2, "visible": True})  # fmt: skip
    await asyncio.sleep(0.01)
    return page, conn


def _frame(cdp: FakeCDP, n: int = 1, sy: float = 0) -> None:
    for i in range(n):
        cdp.emit("Page.screencastFrame", {"data": B64, "sessionId": 100 + i, "metadata": {
            "deviceWidth": 1024, "deviceHeight": 590, "scrollOffsetX": 0, "scrollOffsetY": sy, "timestamp": 1790000000.5}})  # fmt: skip


# ---- the protocol's pieces ------------------------------------------------------------------------------


def test_a_frame_header_is_64_bytes_little_endian_and_round_trips():
    sid, raw = browser_live.new_sid()
    assert len(sid) == 22 and len(raw) == 16
    buf = pack_frame(raw, 812, JPEG, kind=2, ts_ms=1790512345123.4, vw=1024, vh=590, iw=1638, ih=944, dpr=1.6,
                     sx=0, sy=1320.5, flags=4 | 1)  # fmt: skip
    assert browser_live.HEADER_LEN == 64 and buf[:4] == b"NBF1" and buf[4] == 1 and buf[5] == 2
    assert struct.unpack_from("<H", buf, 6)[0] == 64  # header_len
    assert buf[8:24] == raw and struct.unpack_from("<I", buf, 24)[0] == 812  # sid, fid (the UI's offsets)
    assert struct.unpack_from("<d", buf, 28)[0] == 1790512345123.4
    assert struct.unpack_from("<HHHHH", buf, 36) == (1024, 590, 1638, 944, 160)
    assert struct.unpack_from("<ff", buf, 46) == (0.0, 1320.5) and struct.unpack_from("<H", buf, 54)[0] == 5
    assert buf[56] == 1 and buf[57:64] == b"\0" * 7 and buf[64:] == JPEG
    meta, image = unpack_frame(buf)
    assert meta["sid"] == sid and meta["fid"] == 812 and meta["kind"] == 2 and image == JPEG
    assert unpack_frame(b"XXXX" + buf[4:]) is None and unpack_frame(buf[:40]) is None


def test_jpeg_size_reads_the_frame_header_without_decoding():
    assert browser_live.jpeg_size(_jpeg(123, 45)) == (123, 45)
    assert browser_live.jpeg_size(b"not a jpeg") is None


@pytest.mark.parametrize(
    "view, page",
    [
        ((819, 472), (1024, 590)),  # 14" MacBook Pro, docked
        ((940, 544), (1175, 680)),  # 16"
        ((778, 415), (1050, 560)),  # 13" Air: the height hits 560 and the width follows it
        ((919, 532), (1149, 665)),  # compact
        ((972, 544), (1215, 680)),  # external
        ((2400, 1500), (2400, 1500)),  # a very large window: never scaled up past the view
        ((600, 2000), (480, 1600)),  # a tall one: the height limit, the width follows
    ],
)
def test_the_page_follows_the_card_in_its_own_aspect(view, page):
    assert page_viewport(*view) == page


def test_keys_become_cdp_key_events_with_macs_editing_commands():
    assert key_event("keydown", "a", "KeyA", 0, text="a") == {
        "key": "a", "code": "KeyA", "windowsVirtualKeyCode": 65, "nativeVirtualKeyCode": 65, "modifiers": 0,
        "type": "keyDown", "text": "a", "unmodifiedText": "a",
    }  # fmt: skip
    enter = key_event("keydown", "Enter", "Enter", 0)
    assert enter["type"] == "keyDown" and enter["text"] == "\r" and enter["windowsVirtualKeyCode"] == 13
    select_all = key_event("keydown", "a", "KeyA", 4, text="a")  # ⌘A never types an "a"
    assert select_all["type"] == "rawKeyDown" and "text" not in select_all and select_all["commands"] == ["selectAll"]
    back = key_event("keydown", "Backspace", "Backspace", 1)
    assert back["type"] == "rawKeyDown" and back["commands"] == ["deleteWordBackward"]
    assert key_event("keyup", "", "ShiftLeft", 0) == {
        "key": "Shift", "code": "ShiftLeft", "windowsVirtualKeyCode": 16, "nativeVirtualKeyCode": 16,
        "modifiers": 0, "type": "keyUp",
    }  # fmt: skip
    assert key_event("keydown", "é", "", 0, text="é")["text"] == "é"


def test_the_words_on_the_card_come_from_the_real_host():
    assert browser_live.site_name("https://smile.amazon.co.uk/x") == "Amazon"
    assert browser_live.site_name("https://amazon.com.example.io/login") == "Example"
    assert browser_live.secure_of("https://a.example") == "secure"
    assert browser_live.secure_of("http://a.example") == "insecure"
    assert browser_live.secure_of("http://192.168.1.4:8080/") == "local"
    assert browser_live.secure_of("about:blank") == "internal"
    assert browser_live.secure_of("chrome-error://chromewebdata/") == "error"
    assert browser_live.quoted("x" * 40) == "“" + "x" * 32 + "…”"


# ---- opening, state, the screencast ----------------------------------------------------------------------


async def test_the_first_step_of_a_job_opens_the_card_and_the_screencast_follows_who_watches(card):
    live, ch = card
    live.claim("browse1", url="https://shop.example/mug", starting=True)
    (opened,) = ch.of("browser_open")
    assert opened["status"] == "starting" and opened["job"] == "browse1" and opened["url"] == "https://shop.example/mug"
    assert [r for t, _, r in ch.sent if t == "browser_open"] == [True]  # the transcript keeps it
    page = FakePage()
    browser._b._current = page
    live.launched()
    await live.bind(page)
    cdp = page.cdp
    assert not cdp.calls("Page.startScreencast")  # nobody watches yet: no frames at all
    conn = FakeConn()
    live.command(conn, {"cmd": "browser_hello", "v": 1, "binary": True, "formats": ["jpeg"]})
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 819, "css_h": 472, "dpr": 2, "visible": True})
    await asyncio.sleep(0.01)
    (cast,) = cdp.calls("Page.startScreencast")
    assert cast == {"format": "jpeg", "quality": 72, "maxWidth": 1638, "maxHeight": 944, "everyNthFrame": 2}
    conn.flush()
    meta, _ = unpack_frame(conn.frames[0])
    assert meta["kind"] == 2 and meta["flags"] & 4 and meta["sy"] == 120  # a still at once
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "hidden", "css_w": 0, "css_h": 0, "dpr": 2, "visible": False})
    await asyncio.sleep(0.01)
    assert cdp.calls("Page.stopScreencast")  # hidden: stopped
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "mini", "css_w": 88, "css_h": 55, "dpr": 2, "visible": True})
    await asyncio.sleep(0.01)
    assert cdp.calls("Page.startScreencast")[-1] == {"format": "jpeg", "quality": 55, "maxWidth": 480, "maxHeight": 300, "everyNthFrame": 6}
    states = ch.of("browser_state")
    assert states[-1]["status"] == "working" and [s["seq"] for s in states] == sorted({s["seq"] for s in states})


async def test_frames_never_queue_two_credits_and_the_newest_frame_wins(card):
    live, ch = card
    page, conn = await _open(live, ch)
    cdp = page.cdp
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "focus", "css_w": 1100, "css_h": 800, "dpr": 2, "visible": True})
    await asyncio.sleep(0.01)
    conn.flush()
    v = live.viewers[conn]
    v.credits, v.outstanding, conn.frames = 2, {}, []

    def frame(sy):
        v.last_sent = 0  # the frame-rate cap is its own test
        _frame(cdp, 1, sy=sy)
        conn.flush()  # the drain sends what it was offered

    frame(1)
    frame(2)
    frame(3)
    frame(4)
    assert [unpack_frame(f)[0]["sy"] for f in conn.frames] == [1, 2]  # two in flight, no more
    assert v.credits == 0 and v.slot is not None
    await asyncio.sleep(0.01)
    assert len(cdp.calls("Page.screencastFrameAck")) >= 4  # Chromium always gets its ack at once
    live.command(conn, {"cmd": "browser_ack", "sid": live.s.sid, "fid": 99999})  # unknown: ignored
    assert v.credits == 0
    live.command(conn, {"cmd": "browser_ack", "sid": live.s.sid, "fid": unpack_frame(conn.frames[0])[0]["fid"]})
    conn.flush()
    assert [unpack_frame(f)[0]["sy"] for f in conn.frames] == [1, 2, 4]  # the newest, not a queue
    frame(5)
    assert len(conn.frames) == 3
    v.last_ack = time.monotonic() - 3  # a lost ack (or a reloaded view) can't stall it
    frame(6)
    assert unpack_frame(conn.frames[-1])[0]["sy"] == 6
    # A frame that replaces one the drain hasn't sent yet takes over its credit.
    v.credits, v.outstanding = 2, {}
    v.last_sent = 0
    _frame(cdp, 1, sy=7)
    v.last_sent = 0
    _frame(cdp, 1, sy=8)
    assert v.credits == 1 and unpack_frame(conn.pending)[0]["sy"] == 8


async def test_a_frame_the_page_wasnt_is_never_streamed(card):
    live, ch = card
    page, conn = await _open(live, ch)
    conn.flush()
    v = live.viewers[conn]
    v.credits, v.outstanding, conn.frames = 2, {}, []
    v.last_sent = 0
    page.cdp.emit("Page.screencastFrame", {"data": B64, "sessionId": 7, "metadata": {
        "deviceWidth": 1638, "deviceHeight": 944, "scrollOffsetX": 0, "scrollOffsetY": 0, "timestamp": 1.0}})  # fmt: skip
    conn.flush()
    assert conn.frames == []  # the page resized for a moment (a scaled screenshot): dropped
    _frame(page.cdp)
    conn.flush()
    assert len(conn.frames) == 1


async def test_a_slow_view_gets_its_frame_rate_and_a_json_one_gets_base64(card):
    live, ch = card
    page, conn = await _open(live, ch)
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "mini", "css_w": 88, "css_h": 55, "dpr": 2, "visible": True})
    await asyncio.sleep(0.01)
    v = live.viewers[conn]
    conn.flush()
    v.credits, v.last_sent = 2, time.monotonic()
    _frame(page.cdp)
    assert conn.pending is None and v.slot is not None and v.timer is not None  # 2 fps: it waits
    old = FakeConn()
    live.command(old, {"cmd": "browser_hello", "v": 1})  # no binary support
    live.command(old, {"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 800, "css_h": 470, "dpr": 2, "visible": True})
    await asyncio.sleep(0.01)
    old.flush()
    msg = json.loads(old.frames[-1])
    assert msg["type"] == "browser_frame" and base64.b64decode(msg["data"]["b64"]) == JPEG
    assert msg["data"]["kind"] in ("still", "live") and live.viewers[old].fps == 10


async def test_state_tracks_the_page_loading_history_tabs_and_errors(card):
    live, ch = card
    page, conn = await _open(live, ch)
    cdp = page.cdp
    cdp.history = (1, 3)
    cdp.emit("Page.frameStartedLoading", {"frameId": "MAIN"})
    await asyncio.sleep(0.15)
    assert ch.of("browser_state")[-1]["loading"] is True
    cdp.emit("Page.frameStoppedLoading", {"frameId": "OTHER"})  # a frame inside: not the page
    cdp.emit("Page.frameStoppedLoading", {"frameId": "MAIN"})
    await asyncio.sleep(0.45)
    st = ch.of("browser_state")[-1]
    assert st["loading"] is False and st["can_back"] and st["can_forward"] and st["title"] == "Blue mug — Shop"
    assert st["secure"] == "secure" and st["tabs"] == {"count": 1, "active": 0} and st["viewport"]["dpr"] == 2
    cdp.emit("Page.frameNavigated", {"frame": {"id": "MAIN", "url": "chrome-error://chromewebdata/",
                                                "unreachableUrl": "https://nowhere.example/"}})  # fmt: skip
    await asyncio.sleep(0.45)
    assert ch.of("browser_state")[-1]["error"] == {"code": "navigation", "message": "Couldn't reach nowhere.example"}
    page.handlers["crash"][0](page)
    st = ch.of("browser_state")[-1]
    assert st["status"] == "error" and st["error"]["code"] == "crashed"


async def test_the_page_takes_the_cards_size_but_never_under_a_pending_approval(card, monkeypatch):
    live, ch = card
    page, conn = await _open(live, ch)
    await asyncio.sleep(0.45)
    assert ("viewport", 1024, 590) not in page.log  # already that size
    live._vp_last = 0
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "focus", "css_w": 1091, "css_h": 697, "dpr": 2, "visible": True})
    await asyncio.sleep(0.45)
    assert page.log[-1] == ("viewport", 1364, 871) or ("viewport", 1364, 871) in page.log
    assert ch.of("browser_state")[-1]["viewport"] == {"w": 1364, "h": 871, "dpr": 2}
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "mini", "css_w": 88, "css_h": 55, "dpr": 2, "visible": True})
    await asyncio.sleep(0.45)
    assert live.vp == (1364, 871)  # a mini row never resizes the page
    live.s.status = "waiting"
    live._vp_last = 0
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 819, "css_h": 472, "dpr": 2, "visible": True})
    await asyncio.sleep(0.5)
    assert live.vp == (1364, 871)  # the target of the approval stays put
    live.s.status = "thinking"
    await asyncio.sleep(0.35)
    assert live.vp == (1024, 590)


async def test_a_resize_mid_action_re_aims_the_cursor_with_a_new_intent(card):
    live, ch = card
    page, conn = await _open(live, ch)
    await live.enter()
    box = {"x": 10, "y": 20, "width": 100, "height": 30}

    async def remeasure():
        return {"x": 40, "y": 60, "width": 100, "height": 30}

    await live.intent("click", "Clicked “Go”", rect=box, label="Go", remeasure=remeasure)
    first = ch.of("browser_action")[-1]
    live._vp_last = 0
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "focus", "css_w": 1091, "css_h": 697, "dpr": 2, "visible": True})
    await asyncio.sleep(1.0)  # mid-action it waits, but never more than 600 ms
    again = ch.of("browser_action")[-1]
    assert again["phase"] == "intent" and again["aid"] > first["aid"] and again["rect"]["x"] == 40
    live.done()
    live.leave()
    assert ch.of("browser_action")[-1]["phase"] == "done" and ch.of("browser_action")[-1]["aid"] == again["aid"]


# ---- NEO's actions ---------------------------------------------------------------------------------------


async def test_an_action_shows_intent_then_done_with_a_step_and_a_thumbnail(card):
    live, ch = card
    page, conn = await _open(live, ch)
    await live.enter()
    await live.intent("click", "Clicked “Add to cart”", rect={"x": 100, "y": 200, "width": 120, "height": 40}, label="Add to cart", role="button")
    intent = ch.of("browser_action")[-1]
    assert intent["phase"] == "intent" and intent["kind"] == "click" and intent["x"] == 160 and intent["y"] == 220
    assert intent["rect"] == {"x": 100, "y": 200, "w": 120, "h": 40} and intent["label"] == "Add to cart"
    assert intent["vw"] == 1024 and intent["sy"] == 120 and intent["button"] == "left" and "text" not in intent
    assert ch.of("browser_step")[-1]["status"] == "running" and ch.of("browser_state")[-1]["status"] == "working"
    live.done()
    live.leave()
    assert ch.of("browser_action")[-1]["phase"] == "done" and ch.of("browser_action")[-1]["aid"] == intent["aid"]
    await asyncio.sleep(0.05)
    step = ch.of("browser_step")[-1]
    assert step["label"] == "Clicked “Add to cart”" and step["status"] == "ok" and step["by"] == "neo"
    assert step["thumb"].startswith("data:image/jpeg;base64,")
    # Never a scaled clip: Chromium resizes the page for one, and the screencast would stream that.
    assert page.cdp.calls("Page.captureScreenshot") and all("clip" not in p for p in page.cdp.calls("Page.captureScreenshot"))
    await asyncio.sleep(0.12)  # states are coalesced to ten a second
    assert ch.of("browser_state")[-1]["status"] == "thinking"


async def test_scrolls_merge_and_typing_then_enter_is_one_step(card):
    live, ch = card
    await _open(live, ch)
    for _ in range(3):
        await live.intent("scroll", "Scrolled down", direction="down", amount=700, present=False)
        live.done(thumb=False)
    await live.intent("type", "Typed “mugs” in Search", fieldname="Search", text="mugs", present=False)
    live.done(thumb=False)
    await live.intent("key", "Pressed Enter", key="Enter", present=False)
    live.done(thumb=False)
    labels = [s["label"] for s in live.s.steps]
    assert labels == ["Scrolled down 3 times", "Typed “mugs” in Search and pressed Enter"]


def _chat_world(monkeypatch, page_url="https://shop.example/mug"):
    """test_browser's fake shop page, as a chat card sees it."""
    from test_browser import SHOP, TEXT, Ctx, Page

    from neo.tools import load_all

    load_all()
    ctx = Ctx()
    b = browser._Browser()
    b._ctx = ctx
    monkeypatch.setattr(browser, "_b", b)
    for name in ("_FOLLOW_S", "_CHALLENGE_WAIT_S", "_SETTLE_S"):
        monkeypatch.setattr(browser, name, 0)
    page = ctx.add(Page(ctx, page_url, SHOP, TEXT))
    page.search_fields.add("aria-ref=e5")
    page.info["aria-ref=e10"] = {"tag": "button", "label": "Place order", "around": ""}
    page.info["aria-ref=e6"] = {"tag": "button", "label": "Search", "around": ""}
    return b, ctx, page


async def test_a_tool_step_narrates_itself_and_a_secret_field_is_never_typed_or_shown(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    token = current_job.set("browse7")
    try:
        await registry().invoke("browser_read", {}, ToolContext())
        out = await registry().invoke("browser_type", {"target": "2", "text": "blue mugs", "submit": True}, ToolContext())
    finally:
        current_job.reset(token)
    assert out.ok and ch.of("browser_open")[0]["job"] == "browse7"
    typed = [a for a in ch.of("browser_action") if a["kind"] == "type"]
    assert typed[0]["phase"] == "intent" and typed[0]["text"] == "blue mugs" and typed[0]["field"] == "Search the shop"
    assert [s["label"] for s in live.s.steps] == ["Read the page", "Typed “blue mugs” in Search the shop and pressed Enter"]
    page.info["aria-ref=e5"] = {"tag": "input", "type": "password"}
    out = await registry().invoke("browser_type", {"target": "2", "text": "hunter2"}, ToolContext())
    assert out.halt and "log in" in out.text
    assert not [x for x in page.log if x[0] == "fill" and x[2] == "hunter2"]
    assert "hunter2" not in json.dumps(ch.sent)  # never on the card, in any message
    (need,) = ch.of("browser_need_user")
    assert need["reason"] == "login" and need["title"] == "Log in to Shop here, then hand back."
    assert need["sub"] == "NEO won't see what you type." and live.s.controller == "user" and live.s.status == "need_user"


async def test_an_approval_spotlights_its_target_and_a_decline_ends_it(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    await registry().invoke("browser_read", {}, ToolContext())
    out = await registry().invoke("browser_click", {"target": "6"}, ToolContext())
    assert out.halt and out.text.startswith("NEEDS_CONFIRM")
    p = confirm.peek()
    gated = ch.of("browser_action")[-1]
    assert gated["phase"] == "intent" and gated["gated"] is True and gated["label"] == "Place order"
    (need,) = ch.of("browser_need_user")
    assert need["reason"] == "confirm" and need["action_id"] == p.action_id and need["title"] == "Waiting for your OK"
    assert live.s.status == "waiting" and live.s.steps[-1]["label"] == "Asked to click “Place order”"
    assert live.s.steps[-1]["status"] == "waiting"
    confirm.cancel()  # the user said no on the confirm card
    await asyncio.sleep(0.01)
    assert live.s.steps[-1]["status"] == "declined" and live.s.steps[-1]["label"] == "Didn't click “Place order”"
    fail = ch.of("browser_action")[-1]
    assert fail["phase"] == "fail" and fail["reason"] == "declined" and fail["aid"] == gated["aid"]
    assert live.s.need_user is None and live.s.status == "thinking"


async def test_the_approved_click_lands_on_the_same_step(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    await registry().invoke("browser_read", {}, ToolContext())
    await registry().invoke("browser_click", {"target": "6"}, ToolContext())
    p = confirm.peek()
    step_id = live.s.steps[-1]["step_id"]
    with confirm.approved(p.action_id):
        out = await registry().invoke("browser_click", dict(p.params), ToolContext(origin="approval"))
    assert out.ok and ("click", "aria-ref=e10") in page.log
    step = next(s for s in live.s.steps if s["step_id"] == step_id)
    assert step["label"] == "Approved · Clicked “Place order”" and step["status"] == "ok"
    assert live.s.need_user is None and live.s.waiting is None and ch.of("browser_action")[-1]["phase"] == "done"


# ---- the user takes the page -------------------------------------------------------------------------------


async def test_take_holds_neos_next_step_and_hand_back_gives_it_a_fresh_read(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    await registry().invoke("browser_read", {}, ToolContext())
    conn = FakeConn()
    live.command(conn, {"cmd": "browser_hello", "v": 1, "binary": True})
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "c1", "action": "take"})
    await asyncio.sleep(0.01)
    assert conn.of("browser_result") == [{"req": "c1", "ok": True}]
    assert live.s.controller == "user" and live.s.status == "user" and live.s.steps[-1]["label"] == "You took over"
    t0 = browser_live.live.held_seconds()
    step = asyncio.ensure_future(registry().invoke("browser_click", {"target": "7"}, ToolContext()))
    await asyncio.sleep(0.2)
    assert not step.done() and not [x for x in page.log if x[0] == "click"]  # NEO waits for the page
    assert live.held_seconds() - t0 >= 0.15  # …and that's not the run's time
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "c2", "action": "release"})
    out = await asyncio.wait_for(step, 2)
    assert out.text.startswith("The user used the browser while you waited, so NEO didn't do that step.")
    assert "URL: https://shop.example/mug" in out.text and not [x for x in page.log if x[0] == "click"]
    user = next(s for s in live.s.steps if s["by"] == "user")
    assert user["status"] == "ok" and user["label"].startswith("You took over · 0:0")
    assert live.s.controller == "neo"


async def test_take_waits_for_neos_action_in_flight_and_declines_a_pending_approval(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    declined = []

    async def decline(aid):
        declined.append(aid)
        confirm.cancel()

    live.hooks.decline = decline
    await registry().invoke("browser_read", {}, ToolContext())
    await registry().invoke("browser_click", {"target": "6"}, ToolContext())
    aid = confirm.peek().action_id
    await live.enter()  # a NEO action holds the page
    take = asyncio.ensure_future(live.take())
    await asyncio.sleep(0.1)
    assert not take.done() and live.s.controller == "neo"
    live.leave()
    assert await asyncio.wait_for(take, 1) == ""
    await asyncio.sleep(0.01)
    assert declined == [aid] and confirm.peek() is None and live.s.controller == "user"
    assert any(s["status"] == "declined" for s in live.s.steps)  # taking over never approves


async def test_the_users_input_reaches_the_page_only_while_they_have_it(card):
    live, ch = card
    page, conn = await _open(live, ch)
    cdp = page.cdp
    sid = live.s.sid

    def send(seq, **m):
        live.command(conn, {"cmd": "browser_input", "sid": sid, "seq": seq, **m})

    send(1, type="mousedown", x=10, y=10, button="left", buttons=1, clicks=1, mods=0)
    await asyncio.sleep(0.01)
    assert not cdp.calls("Input.dispatchMouseEvent")
    assert conn.of("browser_result")[-1] == {"req": "input", "ok": False, "code": "not_user"}
    await live.take()
    live.command(conn, {"cmd": "browser_input", "sid": "x" * 22, "seq": 2, "type": "mousemove", "x": 1, "y": 1, "buttons": 0, "mods": 0})
    send(3, type="mousedown", x=5000, y=-4, button="left", buttons=1, clicks=1, mods=0)  # clamped to the page
    send(3, type="mouseup", x=5, y=5, button="left", buttons=0, clicks=1, mods=0)  # a replay: dropped
    send(4, type="mouseup", x=5, y=5, button="left", buttons=0, clicks=1, mods=0)
    send(5, type="keydown", key="a", code="KeyA", mods=4, repeat=False, location=0)
    send(6, type="text", text="héllo ✓")
    send(7, type="wheel", x=100, y=100, dx=0, dy=240.5, mods=0)
    send(8, type="keydown", key="<script>alert(1)</script>", code="KeyA", mods=0)  # not a key
    await asyncio.sleep(0.05)
    mouse = cdp.calls("Input.dispatchMouseEvent")
    assert mouse[0] == {"type": "mousePressed", "x": 1024, "y": 0, "button": "left", "buttons": 1, "clickCount": 1, "modifiers": 0}
    assert [m["type"] for m in mouse] == ["mousePressed", "mouseReleased", "mouseWheel"]
    assert mouse[2]["deltaY"] == 240.5
    (key,) = cdp.calls("Input.dispatchKeyEvent")
    assert key["commands"] == ["selectAll"] and key["modifiers"] == 4
    assert cdp.calls("Input.insertText") == [{"text": "héllo ✓"}]
    # A rejection is said, at most once a second (the wrong sid's, a moment later, isn't).
    assert [r.get("code") for r in conn.of("browser_result")] == ["not_user"]


async def test_input_is_rate_limited_and_moves_coalesce(card):
    live, ch = card
    page, conn = await _open(live, ch)
    await live.take()
    sid = live.s.sid
    for i in range(1, 60):
        live.command(conn, {"cmd": "browser_input", "sid": sid, "seq": i, "type": "mousedown", "x": 1, "y": 1,
                            "button": "left", "buttons": 1, "clicks": 1, "mods": 0})  # fmt: skip
    await asyncio.sleep(0.05)
    assert len(page.cdp.calls("Input.dispatchMouseEvent")) <= 41  # 20/s, a burst of 40
    assert {"req": "input", "ok": False, "code": "rate"} in conn.of("browser_result")
    live._inputs.clear()
    live._enqueue({"type": "mousemove", "x": 1, "y": 1, "buttons": 0, "mods": 0, "sid": sid})
    live._enqueue({"type": "mousemove", "x": 9, "y": 9, "buttons": 0, "mods": 0, "sid": sid})
    assert len(live._inputs) == 1 and live._inputs[0]["x"] == 9


async def test_what_the_user_types_is_never_logged(card, capsys):
    live, ch = card
    page, conn = await _open(live, ch)
    await live.take()
    page.cdp.fail.add("Input.insertText")
    live.command(conn, {"cmd": "browser_input", "sid": live.s.sid, "seq": 1, "type": "paste", "text": "my secret words"})
    await asyncio.sleep(0.05)
    out = capsys.readouterr().out
    assert "browser input failed: paste RuntimeError" in out and "secret" not in out
    assert "secret" not in json.dumps(ch.sent)


async def test_while_the_user_drives_a_select_opens_on_the_card_and_a_dialog_asks_them(card):
    live, ch = card
    page, conn = await _open(live, ch)
    await live.take()
    page.scripts["closest('select')"] = lambda arg: {
        "rect": {"x": 40, "y": 120, "w": 90, "h": 24}, "multiple": False,
        "options": [{"value": "s", "label": "Small", "selected": True, "disabled": False},
                    {"value": "m", "label": "Medium", "selected": False, "disabled": False}],
    }  # fmt: skip
    live.command(conn, {"cmd": "browser_input", "sid": live.s.sid, "seq": 1, "type": "mousedown", "x": 50, "y": 130,
                        "button": "left", "buttons": 1, "clicks": 1, "mods": 0})  # fmt: skip
    live.command(conn, {"cmd": "browser_input", "sid": live.s.sid, "seq": 2, "type": "mouseup", "x": 50, "y": 130,
                        "button": "left", "buttons": 0, "clicks": 1, "mods": 0})  # fmt: skip
    await asyncio.sleep(0.05)
    assert not page.cdp.calls("Input.dispatchMouseEvent")  # headless draws no picker: the card does
    (sel,) = ch.of("browser_select")
    assert [o["label"] for o in sel["options"]] == ["Small", "Medium"]
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "c9", "action": "select", "select_id": sel["select_id"], "value": "m"})
    await asyncio.sleep(0.01)
    assert ("select_option", f'[data-neo-select="{sel["select_id"]}"]', ["m"]) in page.log

    class Dialog:
        type, message, default_value = "confirm", "Remove the mug?", ""

        def __init__(self):
            self.did = ""

        async def accept(self, *a):
            self.did = "accept"

        async def dismiss(self):
            self.did = "dismiss"

    d = Dialog()
    assert live.user_dialog(d)
    (shown,) = ch.of("browser_dialog")
    assert shown["message"] == "Remove the mug?" and shown["origin"] == "shop.example" and shown["kind"] == "confirm"
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "d1", "action": "dialog", "dialog_id": shown["dialog_id"], "accept": True})
    await asyncio.sleep(0.01)
    assert d.did == "accept" and ch.of("browser_dialog")[-1] == {"sid": live.s.sid, "dialog_id": shown["dialog_id"], "closed": True}


async def test_copy_goes_only_to_the_client_that_asked(card):
    live, ch = card
    page, conn = await _open(live, ch)
    other = FakeConn()
    live.command(other, {"cmd": "browser_hello", "v": 1, "binary": True})
    await live.take()
    page.scripts["selectionStart"] = "Blue mug"
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "c3", "action": "copy"})
    await asyncio.sleep(0.01)
    assert conn.of("browser_clipboard") == [{"sid": live.s.sid, "req": "c3", "text": "Blue mug"}]
    assert not other.of("browser_clipboard") and not ch.of("browser_clipboard")


# ---- hand-offs and the end of a card ---------------------------------------------------------------------------


async def test_a_login_hand_off_focuses_the_field_and_hand_back_resumes_the_job(card):
    live, ch = card
    page, conn = await _open(live, ch)
    resumed = []

    async def resume(text):
        resumed.append(text)

    live.hooks.resume = resume
    page.scripts["one-time-code"] = {"x": 300, "y": 180, "w": 280, "h": 36, "kind": "text"}
    page.scripts["getComputedStyle(e).cursor"] = {"cursor": None, "focus": None}  # nothing focused (yet)
    await live.need_user("login", page=page)
    (need,) = ch.of("browser_need_user")
    assert need["rect"] == {"x": 300, "y": 180, "w": 280, "h": 36} and need["site"] == "Shop"
    assert live.s.steps[-1] == {**live.s.steps[-1], "label": "Your turn to log in", "by": "user", "status": "waiting"}
    await asyncio.sleep(0.15)
    hint = ch.of("browser_hint")[-1]
    assert hint == {"sid": live.s.sid, "focus": None}  # said explicitly, even when nothing has focus
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "r1", "action": "release"})
    await asyncio.sleep(0.02)
    assert resumed == ["I've finished logging in in the browser. Continue the task."]
    user = live.s.steps[-1]
    assert user["label"].startswith("You logged in · 0:0") and user["status"] == "ok"
    assert live.s.controller == "neo" and live.s.need_user is None and time.time() - live.handed_back_at < 5


async def test_saying_continue_in_the_chat_is_a_hand_back_too(card):
    live, ch = card
    await _open(live, ch, job="browse1")
    await live.need_user("captcha")
    bus()._jobs.pop("browse1", None)
    live.claim("browse2")  # the user said "continue": a new job asks for the page
    assert live.s.controller == "neo" and live.s.need_user is None and live.s.job == "browse2"


async def test_a_finished_job_folds_its_card_but_not_while_the_user_has_the_page(card, monkeypatch):
    live, ch = card
    monkeypatch.setattr(browser_live, "_CLOSE_AFTER_S", 0.05)
    await bus().set_state(__import__("neo.events", fromlist=["NeoState"]).NeoState.THINKING, job="j1")
    await _open(live, ch, job="j1")
    await live.intent("read", "Read the page", present=False)
    live.done(thumb=False)
    await bus().end_job("j1")
    await asyncio.sleep(0.2)
    (closed,) = ch.of("browser_close")
    assert closed["reason"] == "done" and closed["alive"] and closed["summary"] == "Read the page"
    assert closed["poster"].startswith("data:image/jpeg;base64,") and closed["steps"] == 1
    assert [r for t, _, r in ch.sent if t == "browser_close"] == [True] and live.s is None
    live.claim("j2")
    await live.take()
    await bus().set_state(__import__("neo.events", fromlist=["NeoState"]).NeoState.THINKING, job="j2")
    await bus().end_job("j2")
    await asyncio.sleep(0.2)
    assert len(ch.of("browser_close")) == 1 and live.s is not None  # the user is still on it


async def test_stop_cancels_only_the_job_that_owns_the_card(card):
    from neo.events import NeoState

    live, ch = card
    await _open(live, ch, job="")
    started = asyncio.Event()

    async def job(name):
        await bus().set_state(NeoState.WORKING, job=name)
        live.claim(name)
        started.set()
        await asyncio.sleep(30)

    owner = asyncio.ensure_future(job("browse3"))
    await started.wait()
    other = asyncio.ensure_future(asyncio.sleep(30))
    conn = FakeConn()
    live.command(conn, {"cmd": "browser_hello", "v": 1})
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "s1", "action": "stop"})
    await asyncio.sleep(0.02)
    assert owner.cancelled() and not other.done()
    other.cancel()
    assert conn.of("browser_result") == [{"req": "s1", "ok": True}]
    assert live.s.controller == "user" and live.s.job is None and ch.of("browser_state")[-1]["job"] is None


async def test_close_asks_first_while_neo_works_and_reopen_brings_the_warm_page_back(card):
    live, ch = card
    page, conn = await _open(live, ch, job="j5")
    live.s.status = "working"
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "x1", "action": "close"})
    await asyncio.sleep(0.01)
    assert conn.of("browser_result")[-1] == {"req": "x1", "ok": False, "code": "busy"}
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "x2", "action": "close", "force": True})
    await asyncio.sleep(0.02)
    (closed,) = ch.of("browser_close")
    assert closed["reason"] == "closed" and closed["alive"] and live.s is None
    sid = closed["sid"]
    live.command(conn, {"cmd": "browser_control", "sid": sid, "req": "x3", "action": "reopen"})
    await asyncio.sleep(0.02)
    reopened = ch.of("browser_open")[-1]
    assert reopened["sid"] == sid and reopened["reason"] == "reopen" and reopened["controller"] == "user"
    await asyncio.sleep(0.15)
    after = [d for t, d, _ in ch.sent[[i for i, (t, d, _) in enumerate(ch.sent) if t == "browser_open"][-1]:] if t == "browser_state"]
    assert "favicon" in after[0]  # the reopened card gets its icon back (the UI reset it on open)
    assert conn.of("browser_result")[-1] == {"req": "x3", "ok": True}


async def test_popping_out_is_bookkeeping_and_the_window_is_the_largest_view(card):
    live, ch = card
    page, conn = await _open(live, ch)
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "p1", "action": "popout"})
    live.command(conn, {"cmd": "browser_view", "sid": live.s.sid, "surface": "mini", "css_w": 88, "css_h": 55, "dpr": 2, "visible": True})
    win = FakeConn()
    live.command(win, {"cmd": "browser_hello", "v": 1, "binary": True, "surface": "window"})
    live.command(win, {"cmd": "browser_view", "sid": live.s.sid, "surface": "window", "css_w": 1100, "css_h": 700, "dpr": 2, "visible": True})
    await asyncio.sleep(0.2)
    assert ch.of("browser_state")[-1]["popped"] is True
    assert page.cdp.calls("Page.startScreencast")[-1]["quality"] == 78  # the window's frames
    win.flush()
    assert win.frames  # the window got a still at once
    live.command(conn, {"cmd": "browser_control", "sid": live.s.sid, "req": "p2", "action": "popin"})
    live.viewer_gone(win)
    await asyncio.sleep(0.2)
    assert ch.of("browser_state")[-1]["popped"] is False


async def test_after_the_user_had_the_page_neos_read_hides_what_they_filled(card, monkeypatch):
    live, ch = card
    b, ctx, page = _chat_world(monkeypatch)
    page.snapshot = page.snapshot.replace('searchbox "Search the shop" [ref=e5]: mugs', 'searchbox "Search the shop" [ref=e5]: my own words')
    await registry().invoke("browser_read", {}, ToolContext())
    live.s.touched = True  # the user had it

    real = page.evaluate

    async def evaluate(script, arg=None):
        if script is browser._JS_USER_FILLED:
            return ["my own words"]
        return await real(script, arg)

    page.evaluate = evaluate
    out = await registry().invoke("browser_read", {}, ToolContext())
    assert "my own words" not in out.text and '"Search the shop" [filled by you]' in out.text


async def test_window_mode_has_no_card(card, monkeypatch):
    from neo import config

    live, ch = card
    monkeypatch.setenv("NEO_BROWSER_VIEW", "window")
    monkeypatch.setattr(config, "_settings", None)
    b, ctx, page = _chat_world(monkeypatch)
    out = await registry().invoke("browser_read", {}, ToolContext())
    assert out.ok and ch.sent == [] and live.s is None


async def test_erase_ends_the_card_without_a_poster(card):
    live, ch = card
    await _open(live, ch)
    await browser.shutdown("erased")
    (closed,) = ch.of("browser_close")
    assert closed["reason"] == "erased" and not closed["alive"] and closed["poster"] is None and closed["url"] == ""


async def test_the_hello_snapshot_rebuilds_a_card(card):
    live, ch = card
    await _open(live, ch)
    for i in range(70):
        await live.intent("scroll", f"Scrolled {'down' if i % 2 else 'up'}", direction="down" if i % 2 else "up", present=False)
        live.done(thumb=False)
    snap = live.hello_snapshot()
    assert snap["v"] == 1 and snap["live"]["sid"] == live.s.sid and len(snap["steps"]) == 60
    assert snap["open"]["sid"] == live.s.sid and snap["action"]["phase"] == "done" and "favicon" in snap["live"]
    browser_live.reset()
    assert browser_live.live.hello_snapshot()["live"] is None


# ---- the websocket lane ----------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_the_websocket_sends_frames_as_binary_after_json_and_keeps_browser_commands_off_the_bus(card, monkeypatch):
    import websockets

    from neo import config
    from neo.server import ws as ws_mod

    live, _ = card
    port = _free_port()
    monkeypatch.setenv("NEO_WS_PORT", str(port))
    monkeypatch.setattr(config, "_settings", None)
    seen = []

    async def on_command(cmd):
        seen.append(cmd)

    srv = ws_mod.UIServer(on_command)
    await srv.start()
    events = []
    unsub = bus().subscribe(lambda ev: events.append(ev.type))
    try:
        page = FakePage()
        browser._b._current = page
        live.launched()
        live.claim("j1")
        await live.bind(page)
        async with websockets.connect(f"ws://127.0.0.1:{port}/?token={ws_mod.ws_token()}") as c:
            hello = json.loads(await c.recv())
            assert hello["type"] == "hello" and hello["data"]["browser"]["live"]["sid"] == live.s.sid
            await c.send(json.dumps({"cmd": "browser_hello", "v": 1, "binary": True, "formats": ["jpeg"], "surface": "overlay"}))
            await c.send(json.dumps({"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 819, "css_h": 472, "dpr": 2, "visible": True}))
            await c.send(json.dumps({"cmd": "browser_input", "sid": live.s.sid, "seq": 1, "type": "text", "text": "private"}))
            big = {"cmd": "browser_input", "sid": live.s.sid, "seq": 2, "type": "paste", "text": "y" * 200_000}
            await c.send(json.dumps(big))  # over the browser's 128 KB: refused
            await c.send(json.dumps({"cmd": "text", "text": "z" * 150_000}))  # a long chat message still goes
            got = []
            while not any(isinstance(m, bytes) for m in got):
                got.append(await asyncio.wait_for(c.recv(), 3))
            frame = next(m for m in got if isinstance(m, bytes))
            meta, image = unpack_frame(frame)
            assert meta["sid"] == live.s.sid and image == JPEG
            await c.send(json.dumps({"cmd": "browser_ack", "sid": live.s.sid, "fid": meta["fid"]}))
            await asyncio.sleep(0.05)
        assert [c["cmd"] for c in seen] == ["text"]  # browser_* never reach the app's command handler…
        assert not [e for e in events if e.startswith("browser_")]  # …nor the event bus
        assert srv._recent[-1]["type"] == "browser_open" or any(m["type"] == "browser_open" for m in srv._recent)
    finally:
        unsub()
        if srv._server:
            srv._server.close()


async def test_a_step_waiting_for_the_page_is_never_stranded_by_a_second_hand_over(card):
    live, ch = card
    await _open(live, ch)
    await live.take()
    held = asyncio.ensure_future(live.enter())
    await asyncio.sleep(0.05)
    live._user_has_it()  # the page changes hands again while the step waits (a stop, a hand-off…)
    await asyncio.sleep(0.05)
    assert not held.done()
    live._handed_back(resume=False)
    assert await asyncio.wait_for(held, 1) is True  # it waited, and it goes on
    live.leave()


# ---- the rest of NEO that the card leans on --------------------------------------------------------------


async def test_the_bus_knows_each_jobs_task_ends_it_alone_and_says_when_a_job_ends():
    from neo.events import EventBus, NeoState

    ev = EventBus()
    ended = []
    ev.on_job_end(ended.append)
    seen = []

    async def job(name):
        await ev.set_state(NeoState.WORKING, job=name)
        await ev.tool_start("browser_read", {}, job=name)
        seen.append(current_job.get())
        await asyncio.sleep(30)

    a, b = asyncio.ensure_future(job("a")), asyncio.ensure_future(job("b"))
    await asyncio.sleep(0.01)
    assert seen == ["a", "b"]  # the job a tool runs for, without a parameter
    assert ev.cancel_job("a") and not ev.cancel_job("nope")
    await asyncio.sleep(0.01)
    assert a.cancelled() and not b.done()
    await ev.end_job("b")
    await ev.end_job("b")  # a second end is not a second ending
    assert ended == ["b"]
    b.cancel()


async def test_a_runs_clock_stops_while_the_browser_waits_for_the_user(monkeypatch):
    from neo.agent import loop
    from neo.agent.loop import run_agent
    from neo.providers.base import Turn

    held = {"s": 0.0}
    loop.add_pause_clock(lambda: held["s"])

    class Slow:
        name, supports_vision, supports_tools = "slow", False, True

        async def complete(self, messages, **kw):
            held["s"] += 100.0  # a hundred seconds of the user's time went by
            return Turn(text="done")

    res = await run_agent("x", provider=Slow(), system="", tools=[], max_seconds=50, verify=False)
    assert res.stopped == "done" and res.text == "done"


async def test_browse_budget_does_not_count_the_users_time(monkeypatch):
    from neo.agent import loop
    from neo.tools import browse

    held = {"s": 0.0}
    loop.add_pause_clock(lambda: held["s"])

    async def job():
        held["s"] += 5.0  # the user had the page for five seconds…
        await asyncio.sleep(1.5)  # …and the job's own time is inside its 1 s budget plus that
        return "ok"

    assert await browse._within(job(), 1.0) == "ok"

    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(TimeoutError):
        await browse._within(slow(), 0.2)


async def test_with_two_overlays_the_sharpest_sets_the_frames_and_the_largest_the_page(card):
    live, ch = card
    page, conn = await _open(live, ch)  # docked 819 × 472 at 2x
    other = FakeConn()
    live.command(other, {"cmd": "browser_hello", "v": 1, "binary": True})
    live.command(other, {"cmd": "browser_view", "sid": live.s.sid, "surface": "docked", "css_w": 765, "css_h": 443,
                         "dpr": 1, "visible": True})  # fmt: skip
    live._vp_last = 0
    await asyncio.sleep(0.45)
    assert page.cdp.calls("Page.startScreencast")[-1]["maxWidth"] == 1638  # not the 1x view's 765
    assert live.vp == page_viewport(819, 472)  # whichever spoke last, the larger view wins
