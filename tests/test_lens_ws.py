"""Search Screen on the core's websocket and the shell's socket: only NEO.app's own page with the live
ticket the shell sent in lens_freeze is the `lens` client (one at a time, gone when it closes); it
hears nothing of the bus, the replay or the cards and is no viewer; its still goes to it alone as
NBF1 kind 4; its commands from anyone else are refused; an area runs end to end (the on-device pass,
the answer, the rows, the actions, Continue in NEO), older areas are stale, and the core closes it
on Erase, Stop and its own actions — with nothing of the area in any log line. The server, the shell
and the page are this test's own; the still, the brains and the browser are fakes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import pytest
import websockets
from PIL import Image, ImageDraw

from neo2.core import config, events
from neo2.core.lens import ask, capture, live, local, regions, search
from neo2.core.server import ws as ws_mod
from neo2.core.tools.browser_live import unpack_frame
from neo2.core.tools.computer import ocr, screen
from tests.test_v2_ws_security import APP, PAGE, _Peer, _Shell, _with_server

TICKET = "cd" * 32
DISPLAY = {"id": 1, "x": 0, "y": 0, "w": 800, "h": 500, "scale": 2}
SECRET = "Invoice for Meera Shah — total 3,499 — https://pay.example.com/i/77"


class Words:
    def __init__(self, text=SECRET):
        self.text = text

    def languages(self, fast):
        return ["en-US"]

    def recognize(self, image, *, fast, languages, auto, ranges, **kw):
        return [ocr.Raw(self.text, 1.0, (0.1, 0.4, 0.8, 0.2))]


class NoCodes:
    def codes(self, img):
        return []

    def language(self, text):
        return ("en", 0.99)

    def detect(self, text):
        return local.detect_plain(text)


class NoThings:
    def objects(self, img):
        return [(0.2, 0.2, 0.3, 0.3)]

    def instances(self, img):
        return []


def _still() -> screen.Frame:
    img = Image.new("RGB", (1600, 1000), (245, 245, 245))
    ImageDraw.Draw(img).rectangle((200, 200, 900, 500), fill=(30, 90, 200))
    return screen.Frame(img, 0.0, 0.0, 0.5, backend="sck")


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    """Search Screen with fakes for everything outside the core: the still, OCR, Vision, the
    windows, the brains, the search rows, the browser and the clipboard."""
    monkeypatch.setattr(events, "_bus", events.EventBus())
    events.app_bus().subscribe(live._on_event)  # (as at import: an erase closes Search Screen)
    live.reset()
    live.lens_live.hooks = live.Hooks()
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(ocr, "_engine", Words())
    monkeypatch.setattr(ocr, "_preferred", ["en-IN"])
    monkeypatch.setattr(local, "_engine", NoCodes())
    monkeypatch.setattr(regions, "_engine", NoThings())
    monkeypatch.setattr(regions, "windows", lambda sp: [[0.0, 0.0, 800.0, 480.0]])
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    w = {"freeze": [], "asked": [], "copied": [], "opened": [], "rows": []}

    def freeze(sp, timeout=2.0):
        w["freeze"].append(sp)
        return capture.Freeze(_still(), "", 21.0)

    async def answer(part, *, text="", question="", lang="", order=None):
        w["asked"].append({"part": part, "text": text, "question": question})
        return ask.Answer("done", "product", "Blue bag", "A blue tote bag.", "blue tote bag", True, "gemini", "Gemini", False)

    def links(query, n):
        w["rows"].append((query, n))
        return [{"title": "Blue tote — Shop", "host": "shop.example", "href": "https://shop.example/tote?ref=lens"}]

    async def open_url(url, browser_id="", browser_name=""):
        w["opened"].append(url)
        return f"Opened {url}"

    from neo2.core.tools.computer import apps

    monkeypatch.setattr(capture, "freeze", freeze)
    monkeypatch.setattr(ask, "answer", answer)
    monkeypatch.setattr(ask, "eyes", lambda order=None: {"label": "Gemini", "local": False})
    monkeypatch.setattr(search, "links", links)
    monkeypatch.setattr(apps, "open_url", open_url)
    monkeypatch.setattr(live, "copy_text", lambda text: w["copied"].append(text) or True)
    monkeypatch.setattr(live, "copy_image", lambda img: w["copied"].append(img.size) or True)
    monkeypatch.setattr(live, "ASK_AFTER_S", 0.05)
    yield w
    live.reset()


class _Lens(_Peer):
    """Search Screen's overlay: NEO.app's page (origin APP) with the live ticket."""

    def __init__(self, port, ticket=TICKET, *, origin=APP, role=None, extra=""):
        super().__init__(port, origin=origin, role=role)
        self._url += f"&lens={ticket}{extra}"
        self.binary: list[bytes] = []

    async def next(self, type_: str) -> dict:
        while True:
            raw = await asyncio.wait_for(self.c.recv(), 3)
            if isinstance(raw, bytes):
                self.binary.append(raw)
                continue
            msg = json.loads(raw)
            if msg["type"] == type_:
                return msg

    async def until(self, type_: str, **match) -> dict:
        while True:
            msg = await self.next(type_)
            if all(msg["data"].get(k) == v for k, v in match.items()):
                return msg


async def _status(port, ticket=TICKET, **kw) -> str:
    try:
        async with _Lens(port, ticket, **kw):
            return "open"
    except websockets.exceptions.InvalidStatus as e:
        return str(e.response.status_code)


async def _open(shell, ticket=TICKET, **extra):
    await shell.say(t="lens_freeze", ticket=ticket, display=DISPLAY, via="hold", **extra)
    await asyncio.sleep(0.05)


# ---- who gets in ----------------------------------------------------------------------------------------


def test_only_neo_apps_page_with_the_live_ticket_is_search_screen(monkeypatch, capsys):
    async def body(srv, port, seen):
        got = {"before any freeze": await _status(port)}
        async with _Shell(srv, monkeypatch) as shell:
            assert shell.hello["lens"] == {"on": True, "hold_ms": 1000}
            await shell.say(t="lens_freeze", ticket="not-hex", display=DISPLAY)
            await asyncio.sleep(0.02)
            got["a ticket the shell doesn't make"] = await _status(port, "not-hex")
            await _open(shell)
            got |= {
                "a wrong ticket": await _status(port, "ef" * 32),
                "the notch's page": await _status(port, origin=PAGE),
                "a local tool": await _status(port, origin=None),
                "a host": await _status(port, origin=None, role="host"),
                "with Brains & Keys' ticket too": await _status(port, extra="&ticket=" + TICKET),
            }
            async with _Lens(port) as page:
                got["a second overlay"] = await _status(port)
                kinds = sorted(cl.kind for cl in srv._clients.values())
                hello = page.hello
                await shell.say(t="lens_closed", ticket=TICKET, why="esc")
                with pytest.raises(websockets.ConnectionClosed):
                    for _ in range(10):
                        await asyncio.wait_for(page.c.recv(), 3)
            got["after it closed"] = await _status(port)
        return got, kinds, hello

    got, kinds, hello = _with_server(monkeypatch, body)
    assert got == {
        "before any freeze": "403",
        "a ticket the shell doesn't make": "403",
        "a wrong ticket": "403",
        "the notch's page": "403",
        "a local tool": "403",
        "a host": "403",
        "with Brains & Keys' ticket too": "403",
        "a second overlay": "409",
        "after it closed": "403",
    }
    assert kinds == ["lens"]
    assert sorted(hello) == ["lens", "state"]  # (no transcript, no cards, no shell status)
    assert hello["lens"]["display"] == {"w": 800.0, "h": 500.0, "scale": 2.0} and hello["lens"]["search"]["label"] == "Google"
    out = capsys.readouterr().out
    assert TICKET not in out and "ef" * 32 not in out and "its ticket isn't the shell's" in out


def test_the_overlay_hears_nothing_of_the_bus_and_is_no_viewer(monkeypatch):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            async with _Lens(port) as page:
                alone = (srv.viewers(), ws_mod.card_on_screen())
                async with _Peer(port, origin=APP) as app:
                    assert srv.viewers() == 1 and ws_mod.card_on_screen()  # (NEO.app's page counts; the overlay never)
                    await events.bus().say("a private chat line", role="user")
                    srv.camera_send("camera_state", {"sid": "s"})
                    srv.broadcast("browser_state", {"x": 1}, recent=True)
                    srv.camera_frame(b"cam", lambda cl: cl.kind == "lens")  # (even when asked to: never the overlay)
                    await shell.say(t="status", accessibility=True, screen_recording=True, open_at_login=False, version="2", hotkey="⌥⌘")
                    await asyncio.sleep(0.1)
                    app_heard = await app.quiet()
                    page_heard = await page.quiet()
                    lens_cl = next(cl for cl in srv._clients.values() if cl.kind == "lens")
                    return alone, app_heard, page_heard, lens_cl.cam

    alone, app_heard, page_heard, cam = _with_server(monkeypatch, body)
    assert alone == (0, False)  # the overlay alone: no viewer, no card on screen
    assert {"transcript", "camera_state", "browser_state", "shell_status"} <= set(app_heard)
    assert page_heard == [] and cam is None  # …and the overlay heard none of it


def test_search_screens_commands_from_anyone_else_are_refused(monkeypatch, capsys):
    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            for name, kw in {"app": {"origin": APP}, "page": {"origin": PAGE}, "host": {"role": "host"}, "tool": {}}.items():
                async with _Peer(port, **kw) as p:
                    await p.send(cmd="lens_select", sel=1, rect=[0, 0, 100, 100], how="drag")
                    out[name] = (await p.next("lens_refused"))["data"]
        return out, seen

    out, seen = _with_server(monkeypatch, body)
    assert all(v == {"cmd": "lens_select", "error": "forbidden"} for v in out.values()) and len(out) == 4
    assert seen == []  # never the command handler, never the bus
    assert "lens: lens_select refused: from a host" in capsys.readouterr().out


# ---- the still ------------------------------------------------------------------------------------------


def test_the_still_goes_to_the_overlay_alone_as_kind_4(monkeypatch, _world):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, _Peer(port, origin=APP) as app:
            await _open(shell)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                hello = (await page.next("lens_hello"))["data"]
                frozen = (await page.next("lens_frozen"))["data"]
                assert await app.quiet() == []
                return hello, frozen, page.binary

    hello, frozen, binary = _with_server(monkeypatch, body)
    assert hello["frozen"] == "ok" and hello["eyes"] == {"label": "Gemini", "local": False} and hello["ask"] is True
    assert frozen == {"ok": True, "ms": 21}
    [frame] = binary
    meta, jpeg = unpack_frame(frame)
    assert meta["kind"] == live.KIND_LENS == 4 and (meta["vw"], meta["vh"], meta["iw"], meta["ih"], meta["dpr"]) == (800, 500, 1600, 1000, 2.0)
    assert jpeg[:2] == b"\xff\xd8"
    assert _world["freeze"][0] == capture.Spot(1, 0.0, 0.0, 800.0, 500.0, 2.0)


def test_live_mode_when_there_is_no_still(monkeypatch, _world):
    monkeypatch.setattr(capture, "freeze", lambda sp, timeout=2.0: capture.Freeze(None, "no_sck"))
    grabbed = []

    def grab_live(sp, rect):
        grabbed.append(rect)
        return screen.Frame(Image.new("RGB", (200, 100), (200, 200, 200)), 10.0, 10.0, 0.5)

    monkeypatch.setattr(capture, "grab_live", grab_live)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            await shell.say(t="lens_commit", ticket=TICKET)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                frozen = (await page.next("lens_frozen"))["data"]
                regions_ = (await page.next("lens_regions"))["data"]
                await page.send(cmd="lens_select", sel=1, rect=[10, 10, 100, 50], how="drag", live=True)
                captured = (await page.next("lens_captured"))["data"]
                local_ = (await page.next("lens_local"))["data"]
                return frozen, regions_, captured, local_, page.binary

    frozen, regions_, captured, local_, binary = _with_server(monkeypatch, body)
    assert frozen == {"ok": False, "why": "no_sck"} and binary == []
    assert regions_["windows"] and regions_["lines"] == []  # (live mode: windows only)
    assert captured == {"sel": 1, "ok": True} and grabbed == [(10.0, 10.0, 100.0, 50.0)]
    assert local_["thumb"].startswith("data:image/jpeg;base64,")


# ---- an area, end to end --------------------------------------------------------------------------------


def test_an_area_end_to_end(monkeypatch, _world, capsys, caplog):
    caplog.set_level(logging.DEBUG)

    async def body(srv, port, seen):
        out: dict = {}
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            await shell.say(t="lens_commit", ticket=TICKET)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                out["regions"] = (await page.next("lens_regions"))["data"]
                await page.send(cmd="lens_select", sel=1, rect=[100, 100, 350, 150], how="lasso")
                out["local"] = (await page.next("lens_local"))["data"]
                out["thinking"] = (await page.next("lens_answer"))["data"]
                out["answer"] = (await page.next("lens_answer"))["data"]
                out["links"] = (await page.next("lens_links"))["data"]
                acts = {a["kind"]: a["id"] for a in out["local"]["actions"]}
                await page.send(cmd="lens_select", sel=1, rect=[0, 0, 10, 10])  # (the same number again)
                out["stale"] = (await page.next("lens_refused"))["data"]
                await page.send(cmd="lens_act", sel=1, id=acts["copy_text"])
                out["copied"] = (await page.next("lens_done"))["data"]
                await page.send(cmd="lens_act", sel=1, id="a99")
                out["unknown"] = (await page.next("lens_refused"))["data"]
                await page.send(cmd="lens_act", sel=1, id=out["links"]["rows"][0]["id"])
                out["opened"] = (await page.next("lens_done"))["data"]
                out["shell"] = await shell.read()
        return out

    out = _with_server(monkeypatch, body)
    w = _world
    assert out["regions"]["lines"] and out["regions"]["objects"] == [[160.0, 100.0, 240.0, 150.0]]
    loc = out["local"]
    assert loc["sel"] == 1 and loc["text"] == SECRET and loc["lang"]["code"] == "en" and loc["dark"] is False
    assert [a["kind"] for a in loc["actions"]] == ["copy_text", "open", "copy_image", "image_search"]
    link = next(a for a in loc["actions"] if a["kind"] == "open")
    assert link["detail"] == "pay.example.com" and "value" not in link  # (the page never gets the link itself)
    assert out["thinking"] == {"sel": 1, "state": "thinking"}
    ans = out["answer"]
    assert (ans["state"], ans["kind"], ans["title"], ans["brain"]) == ("done", "product", "Blue bag", {"label": "Gemini", "local": False})
    assert ans["actions"][0]["kind"] == "text_search" and ans["actions"][0]["label"] == "Search “blue tote bag”"
    assert out["links"]["rows"] == [{"id": out["links"]["rows"][0]["id"], "title": "Blue tote — Shop", "host": "shop.example"}]
    assert w["rows"] == [("blue tote bag", 5)]
    [asked] = w["asked"]
    assert asked["text"] == SECRET and asked["part"].mime == "image/jpeg"  # the area, and the text read on this Mac
    assert out["stale"] == {"cmd": "lens_select", "error": "stale"}
    assert out["copied"]["ok"] and out["copied"]["note"] == "copied" and not out["copied"]["close"] and w["copied"] == [SECRET]
    assert out["unknown"] == {"cmd": "lens_act", "error": "stale"}
    assert out["opened"]["close"] and w["opened"] == ["https://shop.example/tote?ref=lens"]
    assert out["shell"] == {"t": "lens_close", "why": "action"}  # an action that opened something closes it
    assert live.lens_live.ticket is None and live.lens_live.frame is None  # …and the still is gone
    logs = capsys.readouterr().out + caplog.text
    for word in ("Meera", "3,499", "pay.example", "tote", "Blue bag", "shop.example"):
        assert word not in logs, word
    assert "[lens] area 1 (lasso)" in logs


def test_an_older_areas_commands_are_stale_and_a_new_area_calls_it_off(monkeypatch, _world):
    gate = asyncio.Event()

    async def slow_answer(part, *, text="", question="", lang="", order=None):
        _world["asked"].append(text)
        await gate.wait()
        return ask.Answer("done", "other", "late", "late", "", False, "gemini", "Gemini")

    monkeypatch.setattr(ask, "answer", slow_answer)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                await page.next("lens_frozen")
                await page.send(cmd="lens_select", sel=1, rect=[0, 0, 200, 200])
                await page.until("lens_answer", state="thinking", sel=1)
                await page.send(cmd="lens_select", sel=2, rect=[300, 200, 200, 100], how="pick")
                await page.until("lens_answer", state="thinking", sel=2)
                await page.send(cmd="lens_ask", sel=1, qid=1, q="and this?")
                stale = (await page.next("lens_refused"))["data"]
                gate.set()
                done = await page.until("lens_answer", state="done")
                return stale, done["data"]

    stale, done = _with_server(monkeypatch, body)
    assert stale == {"cmd": "lens_ask", "error": "stale"} and done["sel"] == 2  # area 1's answer never came


def test_continue_in_neo_takes_the_area_to_the_chat(monkeypatch, _world):
    handed = []

    async def continue_(text, part, context):
        handed.append((text, part, context))

    live.lens_live.hooks.continue_ = continue_
    published = []
    events.app_bus().subscribe(lambda ev: published.append((ev.type, dict(ev.data))))

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                await page.next("lens_frozen")
                await page.send(cmd="lens_select", sel=1, rect=[100, 100, 300, 200])
                await page.until("lens_answer", state="done")
                await page.send(cmd="lens_continue", sel=1, text="  where can I buy   this cheaper? ")
                said = await shell.read()
                await asyncio.sleep(0.05)
        return said

    said = _with_server(monkeypatch, body)
    assert said == {"t": "lens_close", "why": "continue"}
    [(text, part, context)] = handed
    assert text == "where can I buy this cheaper?" and part.mime == "image/jpeg"
    assert '<untrusted source="search_screen">' in context and "Blue bag: A blue tote bag." in context
    assert ("lens_continue", {}) in published  # (nothing in it)


def test_the_core_closes_it_on_erase_and_on_stop(monkeypatch):
    async def body(srv, port, seen):
        heard = []
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            await events.app_bus().publish("erased", counts={})
            heard.append(await shell.read())
            await _open(shell, ticket="ab" * 32)
            async with _Peer(port, origin=APP) as app:
                await app.send(cmd="stop")
                heard.append(await shell.read())
        return heard

    heard = _with_server(monkeypatch, body)
    assert heard == [{"t": "lens_close", "why": "erased"}, {"t": "lens_close", "why": "stop"}]


def test_a_hold_let_go_of_early_is_dropped_unanalysed(monkeypatch, _world):
    computed = []
    monkeypatch.setattr(regions, "compute", lambda *a, **k: computed.append(1) or {})

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            await shell.say(t="lens_closed", ticket=TICKET, why="cancel")
            await asyncio.sleep(0.05)
            await shell.say(t="lens_commit", ticket=TICKET)  # (too late: that ticket is gone)
            await asyncio.sleep(0.05)

    _with_server(monkeypatch, body)
    assert computed == [] and live.lens_live.frame is None and live.lens_live.ticket is None


def test_the_still_goes_after_ten_minutes(monkeypatch):
    monkeypatch.setattr(live, "STILL_TTL_S", 0.05)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            return await shell.read()

    assert _with_server(monkeypatch, body) == {"t": "lens_close", "why": "idle"}


# ---- opening it from the core and from a host ------------------------------------------------------------


def test_the_voice_opens_it_with_its_question_and_the_notch_folds_first(monkeypatch, _world):
    published = []
    events.bus().subscribe(lambda ev: published.append((ev.type, dict(ev.data))))

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, _Peer(port, role="host") as host:
            await shell.read()  # (host_attached)
            got = [live.lens_live.open_overlay("how much is this bag")]
            got.append(live.lens_live.open_overlay("again"))  # (within 1.5 s: busy)
            asked = await shell.read()
            await host.send(cmd="shell", action="lens")
            result = (await host.next("shell_result"))["data"]
            await _open(shell)
            async with _Lens(port) as page:
                await page.send(cmd="lens_hello", v=1)
                hello = (await page.next("lens_hello"))["data"]
                await page.send(cmd="lens_select", sel=1, rect=[0, 0, 300, 300])
                await page.until("lens_answer", state="done")
        return got, asked, result, hello

    got, asked, result, hello = _with_server(monkeypatch, body)
    assert got == ["opened", "busy"] and asked == {"t": "shell", "action": "lens"}
    assert result == {"action": "lens", "ok": False, "error": "busy"}  # (one every 1.5 s, whoever asks)
    assert ("lens_opening", {}) in published
    assert hello["prefill"] == "how much is this bag"
    assert _world["asked"][0]["question"] == "how much is this bag"


def test_off_or_no_app_means_no_overlay(monkeypatch):
    monkeypatch.setenv("NEO_LENS", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert live.lens_live.open_overlay() == "off"
    monkeypatch.setenv("NEO_LENS", "on")
    monkeypatch.setattr(config, "_settings", None)
    live.lens_live.channel = None
    assert live.lens_live.open_overlay() == "no_app"

    async def body(srv, port, seen):
        return live.lens_live.open_overlay()  # (a core started by hand: no shell)

    assert _with_server(monkeypatch, body) == "no_app"


def test_the_shells_freeze_needs_a_display_neo_can_read(monkeypatch, _world):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            for bad in (None, {"id": 1}, {**DISPLAY, "w": -5}, {**DISPLAY, "scale": float("inf")}):
                await shell.say(t="lens_freeze", ticket=TICKET, display=bad)
            await asyncio.sleep(0.05)
            return live.lens_live.ticket

    assert _with_server(monkeypatch, body) is None and _world["freeze"] == []


def test_a_restarted_shell_drops_search_screen(monkeypatch):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell:
            await _open(shell)
            assert live.lens_live.ticket == TICKET
        await asyncio.sleep(0.05)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(srv.shell_task, 1)
        return live.lens_live.ticket

    assert _with_server(monkeypatch, body) is None
