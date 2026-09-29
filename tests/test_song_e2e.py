"""The song feature end to end, faked only where the Mac or the network would be. The user's typed "Neo,
what song is this?" comes in over NEO's websocket (the real UIServer, its connection faked) to the real
Session; the real song_identify listens through the real microphone source (sounddevice faked: a tone,
faster than real time), the real recogniser makes Shazam's real fingerprint and asks a fake Shazam
(httpx.MockTransport), and the card's events go back out over the same websocket.

What the overlay receives is checked against the UI contract, and against ui/tests/fixtures/song-wire.json:
the same messages, which ui/tests/song-wire.test.ts replays through the card's reducer (ui/src/song/state.ts).
Both sides test one wire. When the core's messages change on purpose, rewrite that file with
``NEO_SONG_WIRE_WRITE=1`` and run the UI's tests (``npm test`` in ui/)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from neo2.core.tools.music import capture, live, recognize

pytest.importorskip("shazamio")

RATE = capture.RATE
SPEED = 10.0  # the fake microphone: seconds of sound per real second
WIRE = Path(__file__).resolve().parents[1] / "ui" / "tests" / "fixtures" / "song-wire.json"
COVER_RE = re.compile(
    r"^data:image/(?:jpeg|png|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$"
)  # ui/src/song/state.ts
APPLE = "https://music.apple.com/us/album/1499378108?i=1499378615"
SHAZAM = "https://www.shazam.com/track/518495306/blinding-lights"
COVER_URL = "https://is1-ssl.mzstatic.com/image/thumb/Music/400x400cc.jpg"


def _music(seconds: float, amp: float) -> np.ndarray:
    """A few notes over a chord (int16): loud enough to be music, never played anywhere."""
    t = np.arange(int(seconds * RATE)) / RATE
    notes = np.array([440.0, 554.4, 659.3, 880.0])[(t * 2).astype(int) % 4]
    sig = (
        np.sin(2 * np.pi * notes * t)
        + 0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 330.0 * t)
    )
    return (sig / np.abs(sig).max() * amp * 32767).astype(np.int16)


class _Stream:
    """sounddevice's RawInputStream, faked: after start() it feeds its signal to the callback from a
    thread of its own, SPEED times faster than real time, until abort()/close()."""

    made: list[_Stream] = []
    amp = 0.08

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback):
        assert (samplerate, channels, dtype) == (RATE, 1, "int16")
        self.blocksize, self.callback = blocksize, callback
        self.on = threading.Event()
        self.did: list[str] = []
        self.signal = _music(20.0, self.amp)
        _Stream.made.append(self)

    def start(self):
        self.did.append("start")
        self.on.set()
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self):
        pos = 0
        while self.on.is_set():
            chunk = self.signal[pos : pos + self.blocksize]
            pos = (pos + self.blocksize) % (self.signal.size - self.blocksize)
            self.callback(chunk.tobytes(), self.blocksize, None, None)
            time.sleep(self.blocksize / RATE / SPEED)

    def abort(self):
        self.did.append("abort")
        self.on.clear()

    def close(self):
        self.did.append("close")
        self.on.clear()


class Conn:
    """One overlay's websocket as UIServer._handle sees it: what it sends up, what comes down."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.got: list[dict] = []

    def send_cmd(self, **cmd) -> None:
        self.inbox.put_nowait(json.dumps(cmd))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        m = await self.inbox.get()
        if m is None:
            raise StopAsyncIteration
        return m

    async def send(self, data) -> None:
        self.got.append(json.loads(data))

    def song(self) -> list[dict]:
        return [m for m in self.got if m["type"].startswith("song_")]

    def hello(self) -> dict:
        return next(m for m in self.got if m["type"] == "hello")["data"]


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((48, 48, 3), (180, 40, 60), np.uint8)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _match() -> dict:
    return {
        "matches": [{"id": "1"}],
        "track": {
            "title": "Blinding Lights",
            "subtitle": "The Weeknd",
            "url": SHAZAM,
            "genres": {"primary": "R&B/Soul"},
            "images": {"coverart": COVER_URL},
            "hub": {"options": [{"actions": [{"type": "applemusicopen", "uri": APPLE}]}]},
            "sections": [
                {
                    "type": "SONG",
                    "metadata": [
                        {"title": "Album", "text": "After Hours"},
                        {"title": "Released", "text": "2020"},
                    ],
                }
            ],
        },
    }


@pytest.fixture
def neo(monkeypatch, tmp_path):
    """NEO's core as NEO.app runs it — websocket server, Session, the real song_identify — with the
    microphone, the Mac's own sound, Shazam and the Mac's link opener faked."""
    import sys

    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod
    from neo2.core.reflex.schema import Decision
    from neo2.core.server.ws import UIServer
    from neo2.core.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", events.EventBus())
    load_all()
    live.reset()
    _Stream.made.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawInputStream=_Stream))
    monkeypatch.setattr(capture, "mac_ready", lambda: "nothing is playing on this Mac")
    asked: list[dict] = []
    answers: list[dict] = []
    gate = asyncio.Event()  # Shazam answers once it is set (a test that looks at the card mid-listen)
    gate.set()

    async def network(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.host.endswith("shazam.com"):
            asked.append(json.loads(request.content))
            await gate.wait()
            return httpx.Response(
                200, json=answers.pop(0) if answers else {"matches": [], "tagid": "t", "retryms": 0}
            )
        if request.method == "GET" and str(request.url) == COVER_URL:
            return httpx.Response(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
        raise AssertionError(f"a request NEO never makes: {request.method} {request.url}")

    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(network))
    opened: list[str] = []
    monkeypatch.setattr(live.song_live, "opener", opened.append)

    class Brain:  # no model is ever asked: the user's words go straight to listening
        async def stream(self, *a, **k):
            raise AssertionError("a model was asked")
            yield ""

        async def complete(self, *a, **k):
            raise AssertionError("a model was asked")

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Brain())

    class Reflex:
        async def decide(self, text):
            return Decision("chat", 0.9, False, 0.0, False, 0.0, "fake")

    session = sess_mod.Session(reflex=Reflex())

    async def on_command(cmd: dict) -> None:  # neo2.core.app's, for what these tests send
        if cmd.get("cmd") == "text":
            await session.handle(str(cmd.get("text") or ""), from_user=True)
        elif cmd.get("cmd") == "forget_all":
            await events.bus().publish("erased", counts={})

    srv = UIServer(on_command)
    monkeypatch.setattr(live.song_live, "channel", srv)
    unsub = [events.bus().subscribe(srv._relay), events.bus().subscribe(live._on_event)]
    yield SimpleNamespace(srv=srv, asked=asked, answers=answers, opened=opened, gate=gate)
    for u in unsub:
        u()
    live.reset()


async def _connect(neo) -> tuple[Conn, asyncio.Task]:
    c = Conn()
    task = asyncio.create_task(neo.srv._handle(c))  # type: ignore[arg-type]
    await _until(lambda: c.got)
    return c, task


async def _until(ok, timeout: float = 15.0) -> None:
    t0 = time.monotonic()
    while not ok():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def _check_contract(msgs: list[dict]) -> None:
    """Every song message exactly as the UI contract has it (ui/src/song/types.ts)."""
    for m in msgs:
        assert set(m) == {"type", "data", "ts"}
        d, t = m["data"], m["type"]
        if t == "song_open":
            assert set(d) == {"sid", "source", "listen_s"} and d["source"] in ("mic", "mac", "both")
            assert isinstance(d["listen_s"], float)
        elif t == "song_state":
            assert set(d) == {"sid", "status", "level", "elapsed_s", "message", "result"}
            assert d["status"] in ("listening", "matching", "found", "not_found", "error")
            assert isinstance(d["level"], float) and 0.0 <= d["level"] <= 1.0
            assert isinstance(d["elapsed_s"], float) and d["elapsed_s"] >= 0
            assert d["message"] is None or isinstance(d["message"], str)
            r = d["result"]
            if r is not None:
                assert set(r) == {"title", "artist", "album", "year", "genre", "cover", "links"}
                assert set(r["links"]) == {"apple_music", "shazam"}
                assert all(v is None or v.startswith("https://") for v in r["links"].values())
                assert r["cover"] is None or (COVER_RE.match(r["cover"]) and len(r["cover"]) < 1_500_000)
                assert r["year"] is None or re.fullmatch(r"\d{4}", r["year"])
        else:
            assert t == "song_close" and set(d) == {"sid", "reason"}
            assert d["reason"] in ("done", "stopped", "error", "erased")


def _skeleton(msgs: list[dict]) -> list[list]:
    """What the card's faces depend on, without the meter's ticks, sids, times or the cover's bytes."""
    out: list[list] = []
    for m in msgs:
        d = m["data"]
        if m["type"] == "song_open":
            item = ["song_open", d["source"], d["listen_s"]]
        elif m["type"] == "song_close":
            item = ["song_close", d["reason"]]
        else:
            r = d["result"]
            r = r and {**r, "cover": "data URL" if r["cover"] else None}
            item = ["song_state", d["status"], d["message"], r]
        if not (out and item == out[-1] and item[:2] == ["song_state", "listening"]):
            out.append(item)
    return out


def _for_wire(msgs: list[dict], sids: dict[str, str]) -> list[dict]:
    """The messages as the fixture keeps them: sids by order of appearance, times from 0."""
    t0 = msgs[0]["ts"] if msgs else 0.0
    return [
        {
            "type": m["type"],
            "data": {**m["data"], "sid": sids.setdefault(m["data"]["sid"], f"s{len(sids) + 1}")},
            "ts": round(m["ts"] - t0, 3),
        }
        for m in msgs
    ]


def _wire(key: str, msgs: list[dict], sids: dict[str, str]) -> None:
    """``msgs`` against the fixture's ``key`` (or, with NEO_SONG_WIRE_WRITE=1, into it)."""
    shown = _for_wire(msgs, sids)
    if os.environ.get("NEO_SONG_WIRE_WRITE") == "1":
        data = json.loads(WIRE.read_text()) if WIRE.exists() else {}
        data[key] = shown
        WIRE.parent.mkdir(parents=True, exist_ok=True)
        WIRE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        return
    kept = json.loads(WIRE.read_text())[key]
    assert _skeleton(shown) == _skeleton(kept), (
        f"the core's {key!r} messages changed: the UI's fixture is out of date"
    )


async def test_what_song_is_this_from_the_users_words_to_the_card(neo):
    neo.answers[:] = [{"matches": [], "tagid": "t1", "retryms": 0}, _match()]  # the 2nd try knows it
    neo.gate.clear()  # (Shazam's first answer waits while the overlay reloads mid-listen)
    a, task_a = await _connect(neo)
    a.send_cmd(cmd="text", text="Neo, what song is this?")
    await _until(lambda: any(m["type"] == "song_state" and m["data"]["level"] > 0 for m in a.song()))
    sid = a.song()[0]["data"]["sid"]

    # The overlay reloads while NEO listens: the replay has the card's opening, and song_hello gets its state.
    b, task_b = await _connect(neo)
    assert [m["type"] for m in b.hello()["recent"] if m["type"].startswith("song_")] == ["song_open"]
    b.send_cmd(cmd="song_hello")
    await _until(lambda: b.song())
    rejoin = [m for m in b.hello()["recent"] if m["type"].startswith("song_")] + b.song()[:1]
    assert rejoin[-1]["data"]["sid"] == sid and rejoin[-1]["data"]["status"] == "listening"
    neo.gate.set()

    await _until(lambda: a.song()[-1]["type"] == "song_close")
    got = a.song()
    _check_contract(got)
    assert {m["data"]["sid"] for m in got} == {sid}
    assert _skeleton(got) == [
        ["song_open", "mic", 12.0],
        ["song_state", "listening", None, None],
        ["song_state", "found", None, {
            "title": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours", "year": "2020",
            "genre": "R&B/Soul", "cover": None, "links": {"apple_music": APPLE, "shazam": SHAZAM},
        }],
        ["song_state", "found", None, {
            "title": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours", "year": "2020",
            "genre": "R&B/Soul", "cover": "data URL", "links": {"apple_music": APPLE, "shazam": SHAZAM},
        }],
        ["song_close", "done"],
    ]  # fmt: skip
    ticks = [m for m in got if m["data"].get("status") == "listening"]
    assert sum(x["data"]["level"] > 0.3 for x in ticks) >= 2  # the meter moved with the music
    span = ticks[-1]["ts"] - ticks[0]["ts"]
    assert len(ticks) <= 10 * span + 2  # at most ten a second
    heard = [x["data"]["elapsed_s"] for x in ticks]
    assert heard == sorted(heard) and heard[-1] <= 12.0
    # Two tries (3.5 s, then 7 s: the match), each only a fingerprint; the microphone off and closed.
    assert len(neo.asked) == 2 and all(set(x["signature"]) == {"uri", "samplems"} for x in neo.asked)
    (stream,) = _Stream.made
    assert stream.did == ["start", "abort", "close"] and not live.song_live.busy
    # NEO says the song (the transcript the overlay shows).
    said = [
        m["data"]["text"] for m in a.got if m["type"] == "transcript" and m["data"].get("role") == "assistant"
    ]
    assert "That's Blinding Lights by The Weeknd, from After Hours, 2020." in said

    # The card's buttons: the core opens that card's own https links, nothing else.
    a.send_cmd(cmd="song_link", sid=sid, which="apple_music")
    a.send_cmd(cmd="song_link", sid=sid, which="shazam")
    a.send_cmd(cmd="song_link", sid="another", which="shazam")
    a.send_cmd(cmd="song_link", sid=sid, which="file")
    await _until(lambda: len(neo.opened) >= 2)
    await asyncio.sleep(0.05)
    assert neo.opened == [APPLE, SHAZAM]

    # A late overlay: the replay holds the card's opening, its song and its close; song_hello says nothing.
    c, task_c = await _connect(neo)
    replay = [m for m in c.hello()["recent"] if m["type"].startswith("song_")]
    _check_contract(replay)
    assert [m["type"] for m in replay] == ["song_open", "song_state", "song_state", "song_close"]
    assert replay[-2]["data"]["result"]["cover"] == got[-2]["data"]["result"]["cover"]
    c.send_cmd(cmd="song_hello")
    await asyncio.sleep(0.1)
    assert c.song() == []

    sids: dict[str, str] = {}
    _wire("found", got, sids)
    _wire("rejoin", rejoin, sids)
    _wire("replay", replay, sids)
    for conn, task in ((a, task_a), (b, task_b), (c, task_c)):
        conn.inbox.put_nowait(None)
        await task


@pytest.mark.parametrize("how", ["song_stop", "stop", "forget_all"])
async def test_the_cards_stop_the_stop_button_and_an_erase_stop_listening_at_once(neo, how):
    a, task = await _connect(neo)
    a.send_cmd(cmd="text", text="what song is playing")
    await _until(lambda: any(m["type"] == "song_state" and m["data"]["level"] > 0 for m in a.song()))
    sid = a.song()[0]["data"]["sid"]
    a.send_cmd(cmd=how, sid=sid) if how == "song_stop" else a.send_cmd(cmd=how)
    await _until(lambda: a.song()[-1]["type"] == "song_close")
    (stream,) = _Stream.made
    assert "abort" in stream.did and not stream.on.is_set() and not live.song_live.busy
    await asyncio.sleep(0.4)  # nothing more for that card: no late try, no meter tick
    got = a.song()
    assert got[-1]["type"] == "song_close"
    _check_contract(got)
    reason = "erased" if how == "forget_all" else "stopped"
    assert _skeleton(got) == [
        ["song_open", "mic", 12.0],
        ["song_state", "listening", None, None],
        ["song_close", reason],
    ]
    assert len(neo.asked) <= 1  # (a try may have been on its way; never another)
    _wire(reason, got, {})
    a.inbox.put_nowait(None)
    await task


async def test_the_macs_own_sound_and_the_room_at_once(neo, monkeypatch):
    """A song in headphones: the room is quiet, the Mac plays it. Both listen; the Mac's clip matches."""

    class MacSound:  # the Mac's own sound (ScreenCaptureKit's source), faked
        name = "mac"

        def __init__(self) -> None:
            self.on = threading.Event()

        def start(self, push) -> None:
            self.on.set()
            signal = _music(20.0, 0.2)

            def feed() -> None:
                pos = 0
                while self.on.is_set():
                    push(signal[pos : pos + 1600])
                    pos = (pos + 1600) % (signal.size - 1600)
                    time.sleep(0.1 / SPEED)

            threading.Thread(target=feed, daemon=True).start()

        def stop(self) -> None:
            self.on.clear()

    mac = MacSound()
    monkeypatch.setattr(capture, "mac_ready", lambda: "")
    monkeypatch.setitem(capture.SOURCES, "mac", lambda: mac)
    monkeypatch.setattr(_Stream, "amp", 0.0)  # the room: silence (never sent to Shazam)
    neo.answers[:] = [_match()]
    a, task = await _connect(neo)
    a.send_cmd(cmd="text", text="ye kaunsa gaana hai")
    await _until(lambda: a.song() and a.song()[-1]["type"] == "song_close")
    got = a.song()
    _check_contract(got)
    assert got[0]["data"]["source"] == "both"
    assert [s[1] for s in _skeleton(got)[1:-1]] == ["listening", "found", "found"]
    assert len(neo.asked) == 1  # the room's silence was never a request
    assert not mac.on.is_set() and not live.song_live.busy
    a.inbox.put_nowait(None)
    await task
