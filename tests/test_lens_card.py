"""Search Screen's results in the notch: the overlay hands an area to NotchHub only when NEO lives in
the notch and both NotchHub's connection and NEO's page there said `lens_card_hello`; the core makes
the area a card with an id of its own, NotchHub hears nothing of it but {cid, open, why}, NEO's page
gets a snapshot and then every result tagged with the cid; a refusal or no answer leaves the sheet
complete; once shown, the overlay's still goes and the area stays until the card closes (Esc, an
action that opens something, Continue in NEO, Stop, Erase, a new Search Screen, 10 idle minutes,
nobody left to show it); the card's commands need its live id; the voice is the card's only after
its mic; and nothing of a card reaches a log line. The server, the shell, NotchHub and the pages
are this test's own; the still, the brains and the browser are fakes (tests/test_lens_ws.py)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import threading

import pytest
import websockets
from PIL import Image

from neo2.core import config, events
from neo2.core.lens import ask, live, search
from neo2.core.server import ws as ws_mod
from tests.test_lens_ws import DISPLAY, SECRET, TICKET, _Lens, _open, _world  # noqa: F401 — (the fakes)
from tests.test_v2_ws_security import APP, PAGE, _Peer, _Shell, _with_server

ISLAND = 69734208  # NotchHub's island's display (a CGDirectDisplayID)
RECT = [100, 100, 350, 150]


@pytest.fixture(autouse=True)
def _notch(monkeypatch):
    """NEO's UI lives in the notch (NotchHub hosts it)."""
    monkeypatch.setenv("NEO_UI_HOST", "notch")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(live, "FOLD_S", 0.0)


@pytest.fixture
def held(monkeypatch):
    """The brain answers only once the test lets it — after the hand-off, so the results reach the
    card in the notch (not a race with the overlay). Returns the release."""
    gate: dict = {}
    real = ask.answer  # (the fake: tests/test_lens_ws.py)

    def ev() -> asyncio.Event:
        return gate.setdefault("ev", asyncio.Event())

    async def answer(part, **kw):
        await ev().wait()
        return await real(part, **kw)

    monkeypatch.setattr(ask, "answer", answer)
    return lambda: ev().set()


class _Rec:
    """Keeps every message a client heard; `next` returns the first one of a type (and fields) it
    hasn't returned yet — whether it came before the call or after."""

    def _book(self) -> tuple[list[dict], set[int]]:
        if not hasattr(self, "_heard"):
            self._heard, self._taken = [], set()
        return self._heard, self._taken

    @property
    def heard(self) -> list[dict]:
        return self._book()[0]

    async def next(self, type_: str, **match) -> dict:
        heard, taken = self._book()
        ok = lambda m: m["type"] == type_ and all(m["data"].get(k) == v for k, v in match.items())  # noqa: E731
        for i, m in enumerate(heard):
            if i not in taken and ok(m):
                taken.add(i)
                return m
        while True:
            raw = await asyncio.wait_for(self.c.recv(), 3)
            if isinstance(raw, bytes):
                continue  # (the overlay's still)
            heard.append(json.loads(raw))
            if ok(heard[-1]):
                taken.add(len(heard) - 1)
                return heard[-1]

    async def drain(self) -> list[dict]:
        """Everything that came by the time the core answered a `state` request."""
        await self.send(cmd="state")
        got = []
        while True:
            raw = await asyncio.wait_for(self.c.recv(), 3)
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            self.heard.append(msg)
            if msg["type"] == "state":
                return got
            got.append(msg)


class _Who(_Rec, _Peer):
    """NotchHub's connection (role host) or NEO's page in its panel (origin neo://app)."""


class _Ov(_Rec, _Lens):
    """Search Screen's overlay."""


async def _host(stack, port, *, hello=True, display=ISLAND) -> _Who:
    h = await stack.enter_async_context(_Who(port, role="host"))
    if hello:
        await h.send(cmd="lens_card_hello", **({"display": display} if display is not None else {}))
    return h


async def _page(stack, port, *, hello=True) -> _Who:
    p = await stack.enter_async_context(_Who(port, origin=PAGE))
    if hello:
        await p.send(cmd="lens_card_hello")
    return p


async def _overlay(stack, port, shell) -> _Ov:
    await asyncio.sleep(0.05)  # (the hellos are in)
    await _open(shell)
    ov = await stack.enter_async_context(_Ov(port))
    await ov.send(cmd="lens_hello", v=1)
    ov.hello_ = (await ov.next("lens_hello"))["data"]
    await ov.next("lens_frozen")
    return ov


async def _offer(ov, host, sel=1, rect=RECT) -> str:
    """An area, handed off: the cid NotchHub was offered."""
    await ov.send(cmd="lens_select", sel=sel, rect=rect, how="drag")
    await ov.send(cmd="lens_handoff", sel=sel, to="")
    return (await host.next("lens_card", open=True))["data"]["cid"]


async def _shown(ov, host, page=None, sel=1, release=None):
    """Handed off and shown: NotchHub acks (and the brain, `held`, may answer now)."""
    cid = await _offer(ov, host, sel)
    snap = (await page.next("lens_card", cid=cid))["data"] if page is not None else None
    await host.send(cmd="lens_card_shown", cid=cid)
    ok = (await ov.next("lens_handoff"))["data"]
    if release is not None:
        release()
    return cid, snap, ok


# ---- who can show a card --------------------------------------------------------------------------------


class _Fake:
    def __init__(self, kind, cards):
        self.kind, self.lens_cards, self.lens_display, self.conn = kind, cards, None, object()


def test_a_handoff_needs_the_notch_a_capable_host_and_a_capable_or_no_page(monkeypatch):
    srv = ws_mod.UIServer(lambda cmd: None)

    def can(*clients, ui="notch"):
        monkeypatch.setenv("NEO_UI_HOST", ui)
        monkeypatch.setattr(config, "_settings", None)
        srv._clients = {object(): c for c in clients}
        return srv.can_host_lens_card()

    host, old_host = _Fake("host", True), _Fake("host", False)
    page, old_page = _Fake("page", True), _Fake("page", False)
    assert can(host) is True  # (no page yet: NotchHub makes this install's page as its panel opens)
    assert can(host, page) is True and can(host, page, old_page) is True
    assert can(host, old_page) is False  # an older page still loaded in NotchHub: the sheet
    assert can(old_host, page) is False and can(page) is False  # an older NotchHub, or none
    assert can(host, page, ui="window") is False  # NEO in its own window
    assert can(host, _Fake("app", True), _Fake("tool", True)) is True  # (NEO.app's window is never a page)


def test_the_overlays_hello_says_notch_only_when_a_handoff_could_work(monkeypatch):
    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            await _open(shell)
            async with _Lens(port) as ov:
                out["alone"] = ov.hello["lens"]["notch"]
            await _host(stack, port)
            await _page(stack, port, hello=False)  # (an older page: never said lens_card_hello)
            await asyncio.sleep(0.05)
            await _open(shell, ticket="ef" * 32)
            async with _Lens(port, "ef" * 32) as ov:
                out["older page"] = ov.hello["lens"]["notch"]
            await _page(stack, port)
            await asyncio.sleep(0.05)
            await _open(shell, ticket="12" * 32)
            async with _Lens(port, "12" * 32) as ov:
                out["capable"] = ov.hello["lens"]["notch"]
        return out

    assert _with_server(monkeypatch, body) == {"alone": False, "older page": False, "capable": True}


def test_a_handoff_waits_for_its_area_and_refuses_an_older_sel(monkeypatch):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            ov = await _overlay(stack, port, shell)
            await ov.send(cmd="lens_handoff", sel=1, to="")  # (no area yet)
            early = (await ov.next("lens_handoff"))["data"]
            # Sent together: the hand-off waits for the area being cut.
            await ov.send(cmd="lens_select", sel=1, rect=RECT, how="drag")
            await ov.send(cmd="lens_handoff", sel=1, to="fr")
            offer = (await host.next("lens_card", open=True))["data"]
            await host.send(cmd="lens_card_shown", cid=offer["cid"])
            ok = (await ov.next("lens_handoff"))["data"]
            await ov.send(cmd="lens_select", sel=2, rect=[0, 0, 200, 200], how="drag")
            await ov.send(cmd="lens_handoff", sel=1, to="")
            older = (await ov.next("lens_handoff"))["data"]
            return early, offer, ok, older, live.lens_live._card

    early, offer, ok, older, card = _with_server(monkeypatch, body)
    assert early == {"sel": 1, "ok": False, "why": "stale"}
    assert sorted(offer) == ["cid", "open"] and len(offer["cid"]) == 32
    assert ok == {"sel": 1, "ok": True, "display": ISLAND}
    assert older == {"sel": 1, "ok": False, "why": "stale"}


def test_no_handoff_without_the_notch(monkeypatch):
    monkeypatch.setenv("NEO_UI_HOST", "window")

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            await _host(stack, port)
            ov = await _overlay(stack, port, shell)
            await ov.send(cmd="lens_select", sel=1, rect=RECT, how="drag")
            await ov.send(cmd="lens_handoff", sel=1, to="")
            return ov.hello_["notch"], (await ov.next("lens_handoff"))["data"], (await ov.next("lens_answer", state="done"))["data"]

    notch, no, answer = _with_server(monkeypatch, body)
    assert notch is False and no == {"sel": 1, "ok": False, "why": "no_notch"} and answer["title"] == "Blue bag"


# ---- what each one hears ------------------------------------------------------------------------------------


def test_the_host_hears_only_the_cid_open_and_why(monkeypatch, _world, held):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await page.next("lens_links", cid=cid)  # (everything worked out by now)
            await page.send(cmd="lens_card_close", cid=cid, why="esc")
            await host.next("lens_card", open=False)
            await host.drain()
            return host.heard

    heard = _with_server(monkeypatch, body)
    cards = [m["data"] for m in heard if m["type"] == "lens_card"]
    assert [c["open"] for c in cards] == [True, False] and cards[1]["why"] == "esc"
    for c in cards:
        assert set(c) <= {"cid", "open", "why"}
    text = json.dumps([m for m in heard if m["type"] != "hello"])
    for word in ("Meera", "Blue bag", "tote", "shop.example", "pay.example", "Gemini", "thumb", "data:image"):
        assert word not in text, word
    # (of the bus, only the contentless lens_opening every client hears)
    assert not any(m["type"].startswith("lens_") and m["type"] not in ("lens_card", "lens_opening") for m in heard)


def test_the_page_gets_the_snapshot_then_cid_tagged_results(monkeypatch, _world):
    gate = asyncio.Event()
    real = ask.answer

    async def late_answer(part, **kw):
        await gate.wait()  # (answered after the overlay is gone)
        return await real(part, **kw)

    monkeypatch.setattr(ask, "answer", late_answer)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            await ov.send(cmd="lens_select", sel=1, rect=RECT, how="drag")
            await ov.next("lens_local")  # (worked out before the hand-off)
            await ov.send(cmd="lens_handoff", sel=1, to="de")
            cid = (await host.next("lens_card", open=True))["data"]["cid"]
            snap = (await page.next("lens_card", cid=cid))["data"]
            await host.send(cmd="lens_card_shown", cid=cid)
            ok = (await ov.next("lens_handoff"))["data"]
            gate.set()  # the answer comes during the flight: to the card alone
            answer = (await page.next("lens_answer", state="done"))["data"]
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            with contextlib.suppress(websockets.ConnectionClosed, TimeoutError):
                while True:
                    await ov.next("(until it closes)")
            rest = [m for m in ov.heard[ov.heard.index(next(m for m in ov.heard if m["type"] == "lens_handoff")):]]
            links = (await page.next("lens_links", cid=cid))["data"]
            # A page that (re)loads asks again: the card as it is now.
            async with _Who(port, origin=PAGE) as again:
                await again.send(cmd="lens_card_hello")
                later = (await again.next("lens_card", cid=cid))["data"]
            return cid, snap, ok, answer, rest, links, later, page.heard

    cid, snap, ok, answer, rest, links, later, heard = _with_server(monkeypatch, body)
    assert ok["ok"] is True
    card = snap["card"]
    assert snap["cid"] == cid and snap["open"] is True and card["sel"] == 1 and card["translate_to"] == "de"
    assert card["local"]["text"] == SECRET and card["thumb"].startswith("data:image/jpeg;base64,")
    assert card["answer"] is None or card["answer"]["state"] == "thinking"
    assert card["hello"]["eyes"] == {"label": "Gemini", "local": False} and card["hello"]["search"]["label"] == "Google"
    assert {"code", "name"} <= set(card["hello"]["langs"][0])
    # (an action is its id, label and full host: never the link, a code or the still itself)
    assert all(set(x) <= {"id", "kind", "label", "detail", "warn"} for x in card["local"]["actions"])
    assert "still" not in json.dumps(snap) and "href" not in json.dumps(snap)
    assert answer["cid"] == cid and answer["title"] == "Blue bag" and answer["sel"] == 1
    assert links["cid"] == cid and links["rows"][0]["host"] == "shop.example"
    assert not [m for m in rest if m["type"] in ("lens_links", "lens_reply") or (m["type"] == "lens_answer" and m["data"]["state"] == "done")]
    # (the overlay heard none of it after "shown")
    assert later["card"]["answer"]["title"] == "Blue bag" and later["card"]["links"]["rows"]
    tagged = [m for m in heard if m["type"] in ("lens_answer", "lens_links", "lens_local")]
    assert tagged and all(m["data"]["cid"] == cid for m in tagged)


def test_a_refused_or_late_handoff_leaves_the_sheet_complete(monkeypatch, _world):
    monkeypatch.setattr(live, "HANDOFF_WAIT_S", 0.2)

    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            ov = await _overlay(stack, port, shell)
            cid = await _offer(ov, host)
            await host.send(cmd="lens_card_close", cid=cid, why="unshown")  # NotchHub's camera has the island
            out["refused"] = (await ov.next("lens_handoff"))["data"]
            out["answer"] = (await ov.next("lens_answer", state="done", sel=1))["data"]
            out["host_refused"] = (await host.next("lens_card", open=False))["data"]
            out["kept"] = live.lens_live._area is not None and live.lens_live.ticket == TICKET
            # No answer from NotchHub at all:
            cid2 = await _offer(ov, host, sel=2)
            out["late"] = (await ov.next("lens_handoff", sel=2))["data"]
            out["host_late"] = (await host.next("lens_card", open=False, cid=cid2))["data"]
            out["late_answer"] = (await ov.next("lens_answer", state="done", sel=2))["data"]
            out["card"] = live.lens_live._card
        return out

    out = _with_server(monkeypatch, body)
    assert out["refused"] == {"sel": 1, "ok": False, "why": "unshown"}
    assert out["answer"]["title"] == "Blue bag" and out["kept"]  # the sheet has every result, the area is kept
    assert out["host_refused"]["why"] == "unshown"
    assert out["late"] == {"sel": 2, "ok": False, "why": "timeout"} and out["host_late"]["why"] == "timeout"
    assert out["late_answer"]["title"] == "Blue bag" and out["card"] is None


def test_the_still_goes_at_the_handoff_and_the_area_stays(monkeypatch, _world):
    opened = []

    async def image_search(img):
        opened.append(img.size)
        return "opened"

    monkeypatch.setattr(search, "image_search", image_search)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, snap, _ok = await _shown(ov, host, page)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await asyncio.sleep(0.1)
            lv = live.lens_live
            gone = (lv.ticket, lv.frame, lv._still, lv.regions, lv.spot, lv.client)
            crop = lv._area.crop.image.size if lv._area else None
            loc = snap["card"]["local"] or (await page.next("lens_local", cid=cid))["data"]
            acts = {a["kind"]: a["id"] for a in loc["actions"]}
            await page.send(cmd="lens_card_act", cid=cid, id=acts["copy_image"])
            copied = (await page.next("lens_done", cid=cid))["data"]
            await page.send(cmd="lens_card_act", cid=cid, id=acts["image_search"])
            searched = (await page.next("lens_done", cid=cid, id=acts["image_search"]))["data"]
            closed = (await host.next("lens_card", open=False))["data"]
            return gone, crop, copied, searched, closed, lv._area

    gone, crop, copied, searched, closed, area = _with_server(monkeypatch, body)
    assert gone == (None, None, None, None, None, None)  # the still, its regions, the ticket and the line
    assert crop == (700, 300)  # the area at the still's pixels (350 × 150 pt at 2×)
    assert copied["ok"] and copied["note"] == "copied" and _world["copied"] == [(700, 300)]
    assert searched["ok"] and searched["close"] and opened == [(700, 300)]
    assert closed["why"] == "action" and area is None  # (and the area goes with the card)


def test_card_commands_need_the_live_cid(monkeypatch, _world):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, snap, _ok = await _shown(ov, host, page)
            loc = snap["card"]["local"] or (await page.next("lens_local", cid=cid))["data"]
            copy = next(a["id"] for a in loc["actions"] if a["kind"] == "copy_text")
            out = []
            for bad in ("0" * 32, "not-a-cid", cid.upper(), None):
                await page.send(cmd="lens_card_act", **({"cid": bad} if bad is not None else {}), id=copy)
                out.append((await page.next("lens_refused"))["data"])
            await page.send(cmd="lens_card_act", cid=cid, id="a99")
            out.append((await page.next("lens_refused"))["data"])
            await host.send(cmd="lens_card_shown", cid="1" * 32)
            out.append((await host.next("lens_refused"))["data"])
            return out, live.lens_live._card is not None

    out, still_open = _with_server(monkeypatch, body)
    assert out[:4] == [{"cmd": "lens_card_act", "error": e} for e in ("stale", "unknown", "unknown", "unknown")]
    assert out[4] == {"cmd": "lens_card_act", "error": "stale"}  # (an action id the core doesn't know)
    assert out[5] == {"cmd": "lens_card_shown", "error": "stale"}
    assert _world["copied"] == [] and still_open  # nothing ran, and the card is still up


def test_closing_actions_close_the_card_for_both_and_copies_keep_it(monkeypatch, _world, held):
    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            loc = snap["card"]["local"] or (await page.next("lens_local", cid=cid))["data"]
            acts = {a["kind"]: a["id"] for a in loc["actions"]}
            await page.send(cmd="lens_card_act", cid=cid, id=acts["copy_text"])
            copied = (await page.next("lens_done", cid=cid))["data"]
            await asyncio.sleep(0.05)
            still = live.lens_live._card is not None
            links = (await page.next("lens_links", cid=cid))["data"]
            await page.send(cmd="lens_card_act", cid=cid, id=links["rows"][0]["id"])
            opened = (await page.next("lens_done", cid=cid, id=links["rows"][0]["id"]))["data"]
            on_page = (await page.next("lens_card", open=False))["data"]
            on_host = (await host.next("lens_card", open=False))["data"]
            return copied, still, opened, on_page, on_host, cid

    copied, still, opened, on_page, on_host, cid = _with_server(monkeypatch, body)
    assert copied["note"] == "copied" and not copied["close"] and still and _world["copied"] == [SECRET]
    assert opened["close"] and _world["opened"] == ["https://shop.example/tote?ref=lens"]
    assert on_page == {"cid": cid, "open": False, "why": "action"} == on_host


def test_continue_from_the_card(monkeypatch, _world, held):
    handed = []

    async def continue_(text, part, context):
        handed.append((text, part, context))

    live.lens_live.hooks.continue_ = continue_
    published = []
    events.app_bus().subscribe(lambda ev: published.append((ev.type, dict(ev.data))))

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await page.next("lens_answer", cid=cid, state="done")
            await page.send(cmd="lens_card_continue", cid=cid, text="  is it   cheaper elsewhere? ")
            on_host = (await host.next("lens_card", open=False))["data"]
            await asyncio.sleep(0.05)
            return on_host

    on_host = _with_server(monkeypatch, body)
    assert on_host["why"] == "continue"
    [(text, part, context)] = handed
    assert text == "is it cheaper elsewhere?" and part.mime == "image/jpeg" and "Blue bag: A blue tote bag." in context
    assert ("lens_continue", {}) in published  # (nothing in it)


@pytest.mark.parametrize("how", ["stop", "erased", "replaced", "idle", "gone", "no page", "confirm"])
def test_the_card_closes_on_stop_erase_a_new_freeze_idle_host_gone_and_no_page(monkeypatch, _world, how, held):
    monkeypatch.setattr(live, "CARD_IDLE_S", 0.2 if how == "idle" else 600.0)
    monkeypatch.setattr(live, "CARD_PAGE_WAIT_S", 0.2)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            watch = await stack.enter_async_context(_Who(port, origin=PAGE))
            await watch.send(cmd="lens_card_hello")
            host_stack = contextlib.AsyncExitStack()
            host = await _host(host_stack, port)
            page_stack = contextlib.AsyncExitStack()
            page = await _page(page_stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await page.next("lens_answer", cid=cid, state="done")
            if how == "stop":
                async with _Peer(port, origin=APP) as app:
                    await app.send(cmd="stop")
                    await asyncio.sleep(0.1)
            elif how == "erased":
                await events.app_bus().publish("erased", counts={})
            elif how == "confirm":  # NEO asks for an OK: the chat (its confirm card) takes the panel back
                await events.app_bus().publish("confirm", action_id="act1", summary="send the mail", tool="mail")
            elif how == "replaced":
                await shell.say(t="lens_freeze", ticket="ef" * 32, display=DISPLAY, via="hold")
            elif how == "gone":
                await host_stack.aclose()
            elif how == "no page":
                await page_stack.aclose()
                await watch.c.close()  # (every page that shows cards is gone)
                await asyncio.sleep(0.4)
                return live.lens_live._card is None, live.lens_live._area is None
            closed = (await watch.next("lens_card", open=False))["data"]
            await host_stack.aclose()
            await page_stack.aclose()
            return closed["why"], live.lens_live._area is None

    got = _with_server(monkeypatch, body)
    if how == "no page":
        assert got == (True, True)
    else:
        assert got == ({"gone": "gone", "confirm": "chat"}.get(how, how), True)


def test_no_page_in_eight_seconds_closes_a_card_shown_before_the_page_came(monkeypatch, _world):
    monkeypatch.setattr(live, "CARD_PAGE_WAIT_S", 0.2)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            ov = await _overlay(stack, port, shell)  # (no page at all: NotchHub hasn't made it yet)
            cid, _snap, ok = await _shown(ov, host)
            closed = (await host.next("lens_card", open=False))["data"]
            return ok, closed, cid

    ok, closed, cid = _with_server(monkeypatch, body)
    assert ok["ok"] is True and closed == {"cid": cid, "open": False, "why": "gone"}


def test_a_page_that_comes_in_time_keeps_the_card(monkeypatch, _world):
    monkeypatch.setattr(live, "CARD_PAGE_WAIT_S", 0.3)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host)
            page = await _page(stack, port)  # (NotchHub's web view loaded as its panel opened)
            snap = (await page.next("lens_card", cid=cid))["data"]
            await asyncio.sleep(0.5)
            return snap, live.lens_live._card is not None

    snap, kept = _with_server(monkeypatch, body)
    assert snap["open"] is True and kept


def test_voice_goes_to_the_card_only_after_its_mic(monkeypatch, _world, held):
    armed = []

    async def listen():
        armed.append(1)

    live.lens_live.hooks.listen = listen
    asked = []

    async def follow_up(part, *, text, history, q, first="", sent=None):
        asked.append(q)
        return "It is navy blue.", ask.Answer("done", label="Gemini", local=False)

    monkeypatch.setattr(ask, "follow_up", follow_up)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await page.next("lens_answer", cid=cid, state="done")
            lv = live.lens_live
            before = lv.takes_voice()
            await page.send(cmd="lens_card_listen", cid=cid)
            on = (await page.next("lens_listening", cid=cid))["data"]
            during = lv.takes_voice()
            said = await lv.hear("what colour is it")
            off = (await page.next("lens_listening", cid=cid))["data"]
            reply = (await page.next("lens_reply", cid=cid, state="done"))["data"]
            return before, on, during, said, off, reply, lv.takes_voice()

    before, on, during, said, off, reply, after = _with_server(monkeypatch, body)
    assert before is False and during is True and after is False  # one sentence, after the mic alone
    assert on["on"] is True and off["on"] is False and armed == [1]
    assert said == "It is navy blue." and reply["text"] == said and asked == ["what colour is it"]


def test_lens_local_always_carries_a_thumb_made_off_the_loop(monkeypatch, _world):
    threads = []
    real = live._thumb

    def thumb(img):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(img)

    monkeypatch.setattr(live, "_thumb", thumb)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            ov = await _overlay(stack, port, shell)  # (a still: not live mode)
            await ov.send(cmd="lens_select", sel=1, rect=RECT, how="drag")
            return (await ov.next("lens_local"))["data"]

    loc = _with_server(monkeypatch, body)
    head = "data:image/jpeg;base64,"
    assert loc["thumb"].startswith(head)
    img = Image.open(io.BytesIO(base64.b64decode(loc["thumb"][len(head):])))
    assert img.format == "JPEG" and max(img.size) <= 96
    assert threads == [False]  # in the on-device pass's thread, never on the event loop


def test_nothing_of_a_card_in_the_log(monkeypatch, _world, capsys, caplog, held):
    caplog.set_level(logging.WARNING, logger="websockets")  # (the test's own clients' frames; first: the
    caplog.set_level(logging.DEBUG)  # last call sets the handler's level)

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, _ok = await _shown(ov, host, page, release=held)
            await shell.say(t="lens_closed", ticket=TICKET, why="notch")
            await page.next("lens_links", cid=cid)
            await page.send(cmd="lens_card_ask", cid=cid, qid=1, q="who is Meera Shah")
            await page.next("lens_reply", cid=cid, state="done")
            await page.send(cmd="lens_card_translate", cid=cid, to="fr")
            await page.next("lens_translation", cid=cid)
            await page.send(cmd="lens_card_close", cid=cid, why="x")
            await host.next("lens_card", open=False)
            return cid

    async def follow_up(part, *, text, history, q, first="", sent=None):
        return "Meera Shah sent it.", ask.Answer("done", label="Gemini", local=False)

    async def translate_text(text, name, sent=None):
        return "Facture pour Meera Shah", ""

    monkeypatch.setattr(ask, "follow_up", follow_up)
    monkeypatch.setattr(ask, "translate_text", translate_text)
    cid = _with_server(monkeypatch, body)
    logs = capsys.readouterr().out + caplog.text
    for word in ("Meera", "3,499", "pay.example", "tote", "Blue bag", "shop.example", "Facture", cid):
        assert word not in logs, word
    assert "[lens] lens_card_ask" in logs and "the card in the notch closed (x)" in logs
    assert "the overlay closed (notch): the still is gone, the area is in the notch" in logs


def test_an_overlay_closed_another_way_once_the_card_was_shown_closes_the_card(monkeypatch, _world):
    """Esc (or the pill's ×, a right-click) a moment before NotchHub's "yes" reached the overlay: it
    closes without flying — and the card goes too, never opening in the notch by itself after."""

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            cid, _snap, ok = await _shown(ov, host, page)
            await shell.say(t="lens_closed", ticket=TICKET, why="esc")
            on_host = (await host.next("lens_card", open=False))["data"]
            on_page = (await page.next("lens_card", open=False))["data"]
            lv = live.lens_live
            return ok, cid, on_host, on_page, (lv._card, lv._area, lv.ticket, lv.client)

    ok, cid, on_host, on_page, left = _with_server(monkeypatch, body)
    assert ok["ok"] is True
    assert on_host == {"cid": cid, "open": False, "why": "esc"} and on_page == on_host
    assert left == (None, None, None, None)


def test_an_overlay_that_gave_up_waiting_keeps_its_area_and_gets_its_results_again(monkeypatch, _world, held):
    """The overlay's own clock ran out (it shows the sheet now): `lens_handoff {sel, cancel}` closes the
    card made of that area — offered, or already shown — and whatever only the card heard comes to the
    overlay again, so the sheet is complete and nothing shows in two places."""

    async def body(srv, port, seen):
        out = {}
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            page = await _page(stack, port)
            ov = await _overlay(stack, port, shell)
            # Offered, NotchHub silent: given up before the core's own wait is over.
            cid = await _offer(ov, host)
            await ov.send(cmd="lens_handoff", sel=1, cancel=True)
            out["offered_host"] = (await host.next("lens_card", open=False, cid=cid))["data"]
            out["offered_reply"] = (await ov.next("lens_handoff", sel=1))["data"]
            # Shown, the answer heard by the card alone, then given up on (a late "yes" was ignored).
            cid2, _snap, ok = await _shown(ov, host, page, sel=2, release=held)
            out["ok"] = ok
            out["card_answer"] = (await page.next("lens_answer", cid=cid2, state="done"))["data"]["title"]
            out["overlay_before"] = [m for m in await ov.drain() if m["type"] == "lens_answer"]
            await ov.send(cmd="lens_handoff", sel=2, cancel=True)
            out["shown_host"] = (await host.next("lens_card", open=False, cid=cid2))["data"]
            out["again"] = (await ov.next("lens_answer", sel=2, state="done"))["data"]["title"]
            lv = live.lens_live
            out["left"] = (lv._card, lv._area is not None and lv._area.n == 2, lv.ticket == TICKET)
            # Nothing to cancel (no card, an older area): nothing happens, and no reply.
            await ov.send(cmd="lens_handoff", sel=2, cancel=True)
            out["quiet"] = [m["type"] for m in await ov.drain() if m["type"] in ("lens_handoff", "lens_refused")]
        return out

    out = _with_server(monkeypatch, body)
    assert out["offered_host"]["why"] == "timeout" and out["offered_reply"]["ok"] is False
    assert out["ok"]["ok"] is True and out["card_answer"] == "Blue bag"
    assert out["overlay_before"] == []  # (shown: the results went to the card alone)
    assert out["shown_host"]["why"] == "timeout" and out["again"] == "Blue bag"
    assert out["left"] == (None, True, True)  # the overlay keeps its area and its ticket
    assert out["quiet"] == []


def test_a_confirmation_while_a_card_is_only_offered_leaves_the_sheet(monkeypatch, _world):
    """NEO asks for an OK while NotchHub hasn't answered yet: no card, the sheet by the area."""

    async def body(srv, port, seen):
        async with _Shell(srv, monkeypatch) as shell, contextlib.AsyncExitStack() as stack:
            host = await _host(stack, port)
            ov = await _overlay(stack, port, shell)
            cid = await _offer(ov, host)
            await events.app_bus().publish("confirm", action_id="act1", summary="send the mail", tool="mail")
            closed = (await host.next("lens_card", open=False, cid=cid))["data"]
            reply = (await ov.next("lens_handoff", sel=1))["data"]
            return closed, reply, live.lens_live._card, live.lens_live._area is not None

    closed, reply, card, kept = _with_server(monkeypatch, body)
    assert closed["why"] == "chat" and reply["ok"] is False and card is None and kept
