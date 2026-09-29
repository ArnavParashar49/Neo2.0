"""Naming a song: capture (fake sources), Shazam's request (a fake network, the real fingerprint), the
chat's song card per its UI contract, the websocket lane and song_identify's schedule, stops and
privacy gate. Nothing real listens here: effects are off (the microphone and ScreenCaptureKit are
refused), no request leaves the process, no audio is recorded or played."""

from __future__ import annotations

import asyncio
import builtins
import io
import json
import os
import tempfile
import threading
import time

import httpx
import numpy as np
import pytest
from PIL import Image

from neo2.core import effects, runtime
from neo2.core.agent.registry import USERS_WORDS, ToolContext, catalog
from neo2.core.tools.music import capture, live, recognize
from neo2.core.tools.music import tool as song_tool
from neo2.core.tools.music.live import song_live

RATE = capture.RATE


def tone(seconds: float, hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    sig = amp * np.sin(2 * np.pi * hz * t) + 0.5 * amp * np.sin(
        2 * np.pi * hz * 1.5 * t * (1 + 0.02 * np.sin(t))
    )
    return (np.clip(sig, -1, 1) * 32767).astype(np.int16)


class FakeSource:
    """Pushes a synthetic signal from its own thread (like PortAudio's or ScreenCaptureKit's), faster
    than real time: ``speed`` seconds of audio each real second."""

    instances: list[FakeSource] = []

    def __init__(self, name: str, *, amp: float = 0.3, speed: float = 20.0, fail: str = "") -> None:
        self.name, self.amp, self.speed, self.fail = name, amp, speed, fail
        self.started = self.stopped = False
        self._stop = threading.Event()
        FakeSource.instances.append(self)

    def start(self, push) -> None:
        if self.fail:
            raise capture.Unavailable(self.fail)
        self.started = True
        chunk = tone(0.1, amp=self.amp) if self.amp else np.zeros(int(0.1 * RATE), np.int16)

        def run() -> None:
            while not self._stop.is_set():
                push(chunk)
                self._stop.wait(0.1 / self.speed)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self.stopped = True
        self._stop.set()


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, bool]] = []

    def song_send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        self.sent.append((type_, json.loads(json.dumps(data)), recent))

    def types(self) -> list[str]:
        return [t for t, _, _ in self.sent]

    def states(self) -> list[dict]:
        return [d for t, d, _ in self.sent if t == "song_state"]


@pytest.fixture
def card(monkeypatch):
    live.reset()
    ch = FakeChannel()
    FakeSource.instances.clear()
    opened: list[str] = []
    monkeypatch.setattr(song_live, "channel", ch)
    monkeypatch.setattr(song_live, "source_factory", lambda name: FakeSource(name))
    monkeypatch.setattr(song_live, "opener", opened.append)
    monkeypatch.setattr(capture, "plan", lambda: (["mic"], "nothing is playing on this Mac"))
    monkeypatch.setattr(recognize, "available", lambda: "")
    monkeypatch.setattr(recognize, "warm_soon", lambda: None)
    monkeypatch.setattr(song_tool, "TICK_S", 0.01)  # (its tries, at the fake sources' pace)
    ch.opened = opened  # type: ignore[attr-defined]
    yield ch
    live.reset()


def _user(words: str, origin: str = "user") -> ToolContext:
    return ToolContext(user_text=words, extras={USERS_WORDS: words}, origin=origin)  # type: ignore[arg-type]


SONG = recognize.Song(
    title="Blinding Lights",
    artist="The Weeknd",
    album="After Hours",
    year="2020",
    genre="R&B/Soul",
    links={
        "apple_music": "https://music.apple.com/us/album/1499378108?i=1499378615",
        "shazam": "https://www.shazam.com/track/518495306",
    },
    cover_url="https://is1-ssl.mzstatic.com/image/thumb/Music/400x400cc.jpg",
)


class FakeShazam:
    """recognize.identify's stand-in: the answers, in order (a Song, None, or an exception)."""

    def __init__(self, *answers, delay: float = 0.0) -> None:
        self.answers = list(answers)
        self.calls: list[int] = []  # the samples each request was given
        self.delay = delay

    async def __call__(self, pcm: np.ndarray):
        self.calls.append(int(pcm.size))
        if self.delay:
            await asyncio.sleep(self.delay)
        a = self.answers.pop(0) if self.answers else None
        if isinstance(a, BaseException):
            raise a
        return a


# ---- capture -------------------------------------------------------------------------------------------


def test_the_ring_keeps_only_the_latest_seconds_and_wipes():
    r = capture.Ring(seconds=2.0)
    for i in range(50):  # 5 s in 100 ms chunks
        r.write(np.full(1600, i, np.int16))
    assert r.heard_s() == pytest.approx(5.0)
    last = r.last(1.0)
    assert last.size == RATE and set(np.unique(last)) == set(range(40, 50))
    assert r.last(10.0).size == 2 * RATE  # never more than it holds (bounded)
    r.write(tone(0.1))
    assert r.level() > 0.5
    r.wipe()
    assert r.heard_s() == 0 and r.last(2.0).size == 0 and not r._buf.any()
    r.write(tone(0.1))  # a push that was already on its way when listening stopped
    assert r.heard_s() == 0 and not r._buf.any()


def test_a_chunk_bigger_than_the_ring_keeps_its_end():
    r = capture.Ring(seconds=1.0)
    r.write(np.concatenate([np.full(2 * RATE, 3, np.int16), np.arange(RATE, dtype=np.int16)]))
    assert r.heard_s() == 3.0 and np.array_equal(r.last(1.0), np.arange(RATE, dtype=np.int16))


def test_loudness_and_the_meter():
    assert capture.audible(tone(1.0)) and not capture.audible(np.zeros(RATE, np.int16))
    assert not capture.audible(
        (np.random.default_rng(0).standard_normal(RATE) * 30).astype(np.int16)
    )  # a quiet room
    assert capture.level(-60) == 0.0 and capture.level(-12) == 1.0 and 0 < capture.level(-30) < 1


def test_coreaudio_buffers_become_16k_mono():
    frames = 4800  # 0.1 s at 48 kHz
    left = np.sin(np.linspace(0, 60 * np.pi, frames)).astype(np.float32) * 0.5
    planar = np.concatenate([left, left]).tobytes()
    out = capture.to_pcm16k(planar, rate=48000, channels=2, flags=1 | 32, bits=32, frames=frames)
    assert out.dtype == np.int16 and out.size == 1600 and 15000 < np.abs(out).max() <= 16400
    inter = np.stack([left, left], axis=1).ravel().tobytes()
    assert np.array_equal(
        capture.to_pcm16k(inter, rate=48000, channels=2, flags=1, bits=32, frames=frames), out
    )
    ints = (left * 32767).astype(np.int16)
    same = capture.to_pcm16k(ints[:1600].tobytes(), rate=16000, channels=1, flags=4, bits=16, frames=1600)
    assert np.abs(same.astype(int) - ints[:1600].astype(int)).max() <= 2
    odd = capture.to_pcm16k(left.tobytes(), rate=44100, channels=1, flags=1, bits=32, frames=frames)
    assert abs(odd.size - round(frames * 16000 / 44100)) <= 1
    assert capture.to_pcm16k(b"\0" * 16, rate=16000, channels=1, flags=0, bits=24, frames=4).size == 0


async def test_the_listener_starts_what_it_can_and_stops_everything():
    ok, broken = (
        FakeSource("mic"),
        FakeSource("mac", fail="the Mac's own sound can't be captured (no display)"),
    )
    lis = capture.Listener([ok, broken])
    assert await lis.start() == ["mic"] and "no display" in lis.why()
    for _ in range(50):
        if lis.heard_s() >= 1.0:
            break
        await asyncio.sleep(0.02)
    assert lis.heard_s() >= 1.0 and lis.level() > 0 and set(lis.clips(12.0)) == {"mic"}
    lis.stop()
    assert ok.stopped and not lis.capturing and lis.heard_s() == 0.0
    assert not any(r._buf.any() for r in lis.rings.values())  # nothing heard stays in memory


async def test_a_source_that_starts_too_late_is_left_out_and_turned_off():
    class Slow(FakeSource):
        def start(self, push):
            time.sleep(0.3)
            super().start(push)

    slow = Slow("mac")
    lis = capture.Listener([FakeSource("mic"), slow])
    assert await lis.start(timeout=0.05) == ["mic"] and "in time" in lis.why()
    await asyncio.sleep(0.4)
    assert slow.stopped  # its start finished later: off at once
    lis.stop()


async def test_the_real_microphone_is_refused_under_tests():
    # (the Mac's own sound too: tests/test_v2_effects.py's the_macs_own_sound, before ScreenCaptureKit)
    pytest.importorskip("sounddevice")
    with pytest.raises(effects.EffectRefused) as e:
        capture.MicSource().start(lambda pcm: None)
    assert e.value.kind == effects.AUDIO == "audio"
    lis = capture.Listener([capture.MicSource()])
    with pytest.raises(effects.EffectRefused):
        await lis.start()
    assert not lis.capturing


def test_the_macs_sound_only_while_it_plays_and_with_the_permission(monkeypatch):
    monkeypatch.setattr(capture, "output_playing", lambda: False)
    assert capture.plan() == (["mic"], "nothing is playing on this Mac")
    monkeypatch.setattr(capture, "output_playing", lambda: True)
    from neo2.core.tools.computer import sck

    monkeypatch.setattr(sck, "_framework", lambda: object())
    monkeypatch.setattr(capture, "screen_recording", lambda: False)  # never asked for here
    assert capture.plan() == (["mic"], "NEO doesn't have Screen Recording")
    monkeypatch.setattr(capture, "screen_recording", lambda: True)
    assert capture.plan() == (["mic", "mac"], "")


# ---- recognize: the request, the answer, the cover ---------------------------------------------------------


def _shazam_answer() -> dict:
    return {
        "matches": [{"id": "1"}],
        "track": {
            "title": "Blinding Lights",
            "subtitle": "The Weeknd",
            "url": "https://www.shazam.com/track/518495306/blinding-lights",
            "genres": {"primary": "R&B/Soul"},
            "images": {"coverart": "https://is1-ssl.mzstatic.com/image/thumb/Music/400x400cc.jpg"},
            "share": {"href": "https://www.shazam.com/track/518495306/blinding-lights"},
            "hub": {
                "options": [
                    {"actions": [
                        {"type": "applemusicopen", "uri": "https://music.apple.com/us/album/1499378108?i=1499378615"},
                        {"type": "uri", "uri": "https://evil.example.com/x"},
                    ]}
                ],
                "actions": [{"type": "uri", "uri": "https://audio-ssl.itunes.apple.com/preview.m4a"}],
            },
            "sections": [
                {"type": "SONG", "metadata": [
                    {"title": "Album", "text": "After Hours"},
                    {"title": "Label", "text": "Republic Records"},
                    {"title": "Released", "text": "2020"},
                ]}
            ],
        },
    }  # fmt: skip


def test_shazams_answer_becomes_the_cards_result():
    song = recognize.parse(_shazam_answer())
    assert song is not None
    assert song.result("data:image/jpeg;base64,xx") == {
        "title": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours", "year": "2020",
        "genre": "R&B/Soul", "cover": "data:image/jpeg;base64,xx",
        "links": {
            "apple_music": "https://music.apple.com/us/album/1499378108?i=1499378615",
            "shazam": "https://www.shazam.com/track/518495306/blinding-lights",
        },
    }  # fmt: skip
    assert song.cover_url.startswith("https://is1-ssl.mzstatic.com/")
    assert song.sentence() == "That's Blinding Lights by The Weeknd, from After Hours, 2020."
    assert recognize.parse({"matches": [], "tagid": "x"}) is None
    assert recognize.parse({"matches": [{}]}) is None and recognize.parse("nope") is None


def test_only_https_links_on_the_right_hosts_and_a_single_named_plainly():
    answer = _shazam_answer()
    t = answer["track"]
    t["url"] = "http://www.shazam.com/track/1"  # not https
    t["share"]["href"] = "https://shazam.com.evil.example/track/1"
    t["hub"]["options"][0]["actions"][0]["uri"] = "javascript:alert(1)"
    t["images"]["coverart"] = "https://images.example.com/cover.jpg"
    t["sections"][0]["metadata"] = [
        {"title": "Album", "text": "Blinding Lights - Single"},
        {"title": "Released", "text": "2019-11-29"},
    ]
    song = recognize.parse(answer)
    assert song.links == {"apple_music": None, "shazam": None} and song.cover_url is None
    assert song.year == "2019" and song.sentence() == "That's Blinding Lights by The Weeknd, 2019."


def _wav_pcm_marker(pcm: np.ndarray) -> bytes:
    return pcm[RATE : RATE + 64].tobytes()


async def test_identify_sends_only_the_fingerprint_to_shazam(monkeypatch):
    pytest.importorskip("shazamio")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_shazam_answer())

    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(handler))
    pcm = tone(7.0)
    song = await recognize.identify(pcm)
    assert song is not None and song.title == "Blinding Lights"
    (req,) = seen
    assert req.method == "POST" and req.url.scheme == "https" and req.url.host.endswith("shazam.com")
    body = json.loads(req.content)
    assert set(body) == {"timezone", "signature", "timestamp", "context", "geolocation"}
    assert set(body["signature"]) == {"uri", "samplems"} and body["signature"]["samplems"] == 7000
    assert body["signature"]["uri"].startswith("data:audio/vnd.shazam.sig;base64,")
    assert len(req.content) < pcm.nbytes / 10  # a fingerprint, not the audio
    assert _wav_pcm_marker(pcm) not in req.content and b"RIFF" not in req.content


async def test_identify_uses_only_the_latest_12_seconds_and_never_under_one(monkeypatch):
    pytest.importorskip("shazamio")
    seen: list[dict] = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"matches": []})

    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(handler))
    assert await recognize.identify(tone(20.0)) is None
    assert seen[-1]["signature"]["samplems"] == 12000
    assert await recognize.identify(tone(0.5)) is None and len(seen) == 1  # under a second: no request


@pytest.mark.parametrize(
    "respond",
    [
        lambda r: httpx.Response(503, text="busy"),
        lambda r: httpx.Response(429, json={}),
        lambda r: httpx.Response(200, text="<html>not json</html>"),
        lambda r: httpx.Response(200, json={"error": "try later"}),  # an answer that isn't one
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    ],
)
async def test_a_failed_request_is_one_failure_never_retried(monkeypatch, respond):
    pytest.importorskip("shazamio")
    calls: list[int] = []

    def handler(request):
        calls.append(1)
        return respond(request)

    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(handler))
    with pytest.raises(recognize.Failed):
        await recognize.identify(tone(4.0))
    assert calls == [1]  # one attempt (shazamio's own client would try up to 20 times)


async def test_the_audio_never_touches_the_disk(monkeypatch):
    pytest.importorskip("shazamio")
    recognize._shazamio()  # (its modules load from disk once, before the check)
    monkeypatch.setattr(
        recognize, "transport", httpx.MockTransport(lambda r: httpx.Response(200, json={"matches": []}))
    )

    def no_file(*a, **k):
        raise AssertionError(f"a file was opened: {a[:1]}")

    for mod, name in (
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (tempfile, "mkstemp"),
        (tempfile, "NamedTemporaryFile"),
        (tempfile, "TemporaryFile"),
    ):
        monkeypatch.setattr(mod, name, no_file)
    assert await recognize.identify(tone(4.0)) is None


def _jpeg(w: int, h: int, *, noise: bool = False) -> bytes:
    arr = (np.random.default_rng(1).integers(0, 255, (h, w, 3)) if noise else np.full((h, w, 3), 120)).astype(
        np.uint8
    )
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    return buf.getvalue()


async def test_the_cover_is_fetched_by_the_core_and_bounded(monkeypatch):
    big = _jpeg(1200, 900, noise=True)
    asked: list[str] = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200, content=big, headers={"content-type": "image/jpeg"})

    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(handler))
    url = await recognize.cover("https://is1-ssl.mzstatic.com/image/thumb/x/1200x900.jpg")
    assert url and url.startswith("data:image/jpeg;base64,") and len(url) <= recognize.COVER_DATA_MAX
    import base64

    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(img.size) <= recognize.COVER_PX
    # never a host that isn't Shazam's, never plain http: not even asked
    assert await recognize.cover("https://evil.example.com/a.jpg") is None
    assert await recognize.cover("http://is1-ssl.mzstatic.com/a.jpg") is None
    assert len(asked) == 1


async def test_a_cover_too_big_or_not_a_picture_is_no_cover(monkeypatch):
    huge = b"\xff\xd8" + b"0" * (recognize.COVER_MAX_BYTES + 10)
    answers = iter([
        httpx.Response(200, content=huge, headers={"content-type": "image/jpeg"}),
        httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}),
        httpx.Response(302, headers={"location": "https://evil.example.com/a.jpg"}),
        httpx.Response(200, content=b"not a jpeg at all", headers={"content-type": "image/jpeg"}),
    ])  # fmt: skip
    monkeypatch.setattr(recognize, "transport", httpx.MockTransport(lambda r: next(answers)))
    for _ in range(4):
        assert await recognize.cover("https://is1-ssl.mzstatic.com/a.jpg") is None


def test_missing_shazamio_says_what_to_install(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert "music extra" in recognize.available()


# ---- the card ------------------------------------------------------------------------------------------


async def test_the_card_follows_the_ui_contract(card):
    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    sid = song_live.open(lis, "mic", 12.0)
    assert card.sent[0] == ("song_open", {"sid": sid, "source": "mic", "listen_s": 12.0}, True)
    st = card.states()[0]
    assert st == {
        "sid": sid,
        "status": "listening",
        "level": 0.0,
        "elapsed_s": 0.0,
        "message": None,
        "result": None,
    }
    song_live.finish("not_found", message="I couldn't recognise it.")
    assert card.sent[-2] == (
        "song_state",
        {
            "sid": sid,
            "status": "not_found",
            "level": 0.0,
            "elapsed_s": 0.0,
            "message": "I couldn't recognise it.",
            "result": None,
        },
        True,  # a card's last state is kept in the replay
    )
    assert card.sent[-1] == ("song_close", {"sid": sid, "reason": "done"}, True)
    assert not lis.capturing and not song_live.busy and song_live.state() is None


async def test_the_meter_is_sent_at_most_ten_times_a_second(card):
    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    song_live.open(lis, "mic", 12.0)
    n = len(card.sent)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        song_live.update(level=0.5, elapsed=1.0)
        await asyncio.sleep(0.005)
    sent = len(card.sent) - n
    assert 3 <= sent <= 7  # ~0.5 s at ≤ 10 a second
    song_live.update("matching")  # a new status goes at once
    assert card.states()[-1]["status"] == "matching"
    song_live.close("stopped")


async def test_found_shows_the_song_then_its_cover_then_closes(card):
    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    sid = song_live.open(lis, "both", 12.0)

    async def fetch(url):
        assert url == SONG.cover_url
        return "data:image/jpeg;base64,/9j/"

    song_live.found(SONG.result(), SONG.cover_url, fetch)
    first = card.states()[-1]
    assert (
        first["status"] == "found"
        and first["result"]["cover"] is None
        and first["result"]["title"] == "Blinding Lights"
    )
    assert not lis.capturing  # listening is over the moment it is found
    for _ in range(50):
        if card.types()[-1] == "song_close":
            break
        await asyncio.sleep(0.01)
    assert card.sent[-2][0] == "song_state" and card.sent[-2][2] is True
    assert card.sent[-2][1]["result"]["cover"] == "data:image/jpeg;base64,/9j/"
    assert card.sent[-1] == ("song_close", {"sid": sid, "reason": "done"}, True)
    # the card's links open with the Mac's default handler, only its own, only https
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "apple_music"})
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "shazam"})
    song_live.command(None, {"cmd": "song_link", "sid": "another-card", "which": "shazam"})
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "javascript"})
    assert card.opened == [SONG.links["apple_music"], SONG.links["shazam"]]


async def test_a_links_that_isnt_https_is_never_kept(card):
    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    sid = song_live.open(lis, "mic", 12.0)
    song_live.finish(
        "found",
        result={
            **SONG.result(),
            "links": {"apple_music": "http://music.apple.com/x", "shazam": "file:///etc/passwd"},
        },
    )
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "apple_music"})
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "shazam"})
    assert card.opened == []


async def test_card_commands_stop_and_hello(card):
    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    sid = song_live.open(lis, "mic", 12.0)

    class Conn:
        def __init__(self):
            self.got = []

        def send_json(self, text):
            self.got.append(json.loads(text))

    conn = Conn()
    song_live.command(conn, {"cmd": "song_hello"})
    assert conn.got[0]["type"] == "song_state" and conn.got[0]["data"]["sid"] == sid
    song_live.command(conn, {"cmd": "song_stop", "sid": "an-old-card"})
    assert song_live.busy and lis.capturing  # a stale card's stop is for listening that's over
    song_live.command(conn, {"cmd": "song_stop", "sid": sid})
    assert not song_live.busy and not lis.capturing
    assert card.sent[-1] == ("song_close", {"sid": sid, "reason": "stopped"}, True)
    song_live.command(conn, {"cmd": "song_hello"})
    assert len(conn.got) == 1  # no card live: nothing to say


async def test_erase_stops_listening_and_forgets_the_links(card):
    from neo2.core.events import app_bus

    lis = capture.Listener([FakeSource("mic")])
    await lis.start()
    song_live.open(lis, "mic", 12.0)
    await app_bus().publish("erased", counts={})
    assert not song_live.busy and not lis.capturing
    lis2 = capture.Listener([FakeSource("mic")])
    await lis2.start()
    sid = song_live.open(lis2, "mic", 12.0)
    song_live.finish("found", result=SONG.result())
    await app_bus().publish("erased", counts={})
    song_live.command(None, {"cmd": "song_link", "sid": sid, "which": "shazam"})
    assert card.opened == []


async def test_ws_song_lane():
    from neo2.core.server.ws import UIServer, _Client

    class Conn:
        pass

    srv = UIServer(on_command=lambda cmd: asyncio.sleep(0))
    a, b = _Client(Conn()), _Client(Conn())  # type: ignore[arg-type]
    srv._clients = {a.conn: a, b.conn: b}
    srv.song_send("song_state", {"sid": "s", "status": "listening", "level": 0.4})
    assert json.loads(a.q[-1])["type"] == "song_state" and b.q
    assert not srv._recent  # the meter isn't replayed
    srv.song_send("song_open", {"sid": "s", "source": "mic", "listen_s": 12.0}, recent=True)
    assert srv._recent[-1]["type"] == "song_open"


# ---- song_identify --------------------------------------------------------------------------------------


def test_the_schedule():
    assert song_tool.schedule(12.0) == [3.5, 7.0, 11.0]
    assert song_tool.schedule(20.0) == [3.5, 7.0, 11.0, 15.0, 19.0]
    assert song_tool.schedule(4.0) == [3.5]
    assert song_tool.schedule(3.0) == [3.0]


def test_registered_as_a_music_tool():
    from neo2.core.tools import PRESETS, REFUSED

    t = catalog().get("song_identify")
    assert t and t.category == "music" and "macos" in t.requires and t.untrusted and t.slow
    assert not t.parameters.get("required")
    assert all("song_identify" not in names for names in PRESETS.values()) and "song_identify" in REFUSED


async def test_a_match_at_the_second_try_stops_listening(card, monkeypatch):
    shazam = FakeShazam(None, SONG, SONG)
    monkeypatch.setattr(recognize, "identify", shazam)
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, "data:image/jpeg;base64,/9j/"))
    out = await catalog().invoke("song_identify", {}, _user("what song is this?"))
    assert out.ok and out.text == "That's Blinding Lights by The Weeknd, from After Hours, 2020." and out.say
    assert len(shazam.calls) == 2  # the first try missed, the second matched: no third
    assert 3.4 * RATE <= shazam.calls[0] <= 4.5 * RATE and 6.9 * RATE <= shazam.calls[1] <= 8 * RATE
    (src,) = FakeSource.instances
    assert src.stopped  # the microphone off at once
    statuses = [d["status"] for d in card.states()]
    assert statuses[0] == "listening" and "found" in statuses and "not_found" not in statuses
    levels = [d["level"] for d in card.states() if d["status"] == "listening"]
    assert max(levels) > 0.3  # the meter moved with what it heard
    for _ in range(50):
        if card.types()[-1] == "song_close":
            break
        await asyncio.sleep(0.01)
    assert card.types()[0] == "song_open" and card.sent[-1][1]["reason"] == "done"
    assert card.sent[-2][1]["result"]["cover"] == "data:image/jpeg;base64,/9j/"


async def test_both_sources_at_once_the_first_match_wins(card, monkeypatch):
    monkeypatch.setattr(capture, "plan", lambda: (["mic", "mac"], ""))
    monkeypatch.setattr(song_live, "source_factory", lambda n: FakeSource(n, amp=0.3 if n == "mac" else 0.0))
    shazam = FakeShazam(SONG)
    monkeypatch.setattr(recognize, "identify", shazam)
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, None))
    out = await catalog().invoke("song_identify", {}, _user("ye kaunsa gaana hai"))
    assert out.ok and "Blinding Lights" in out.text
    assert card.sent[0][1]["source"] == "both"
    assert len(shazam.calls) == 1  # the silent microphone wasn't worth a request


async def test_no_match_after_listening_says_so(card, monkeypatch):
    shazam = FakeShazam(None, None, None)
    monkeypatch.setattr(recognize, "identify", shazam)
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert out.ok and out.text == "I couldn't recognise it."
    assert len(shazam.calls) == 3
    assert card.states()[-2]["status"] == "matching"  # the last try, with the microphone already off
    assert card.states()[-1]["status"] == "not_found" and card.sent[-1][1]["reason"] == "done"
    assert FakeSource.instances[0].stopped


async def test_silence_is_never_sent_and_is_said_as_silence(card, monkeypatch):
    monkeypatch.setattr(song_live, "source_factory", lambda n: FakeSource(n, amp=0.0))
    shazam = FakeShazam()
    monkeypatch.setattr(recognize, "identify", shazam)
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert out.text == "I couldn't hear any music." and shazam.calls == []


async def test_no_answer_from_shazam_is_an_error(card, monkeypatch):
    shazam = FakeShazam(recognize.Failed("ConnectError"), recognize.Failed("ConnectError"))
    monkeypatch.setattr(recognize, "identify", shazam)
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert not out.ok and "Shazam" in out.text
    assert len(shazam.calls) == 2  # it gives up after two tries in a row with no answer
    assert card.states()[-1]["status"] == "error" and card.sent[-1][1]["reason"] == "error"


async def test_one_failed_try_then_an_answer_is_no_error(card, monkeypatch):
    shazam = FakeShazam(recognize.Failed("timed out"), None, SONG)
    monkeypatch.setattr(recognize, "identify", shazam)
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, None))
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert out.ok and "Blinding Lights" in out.text and len(shazam.calls) == 3


async def test_the_cards_stop_mid_listen_stops_at_once(card, monkeypatch):
    monkeypatch.setattr(song_live, "source_factory", lambda n: FakeSource(n, speed=1.0))  # real time
    shazam = FakeShazam()
    monkeypatch.setattr(recognize, "identify", shazam)
    task = asyncio.create_task(catalog().invoke("song_identify", {}, _user("what song is this")))
    for _ in range(100):
        if song_live.busy:
            break
        await asyncio.sleep(0.01)
    sid = song_live.sid
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    song_live.command(None, {"cmd": "song_stop", "sid": sid})
    out = await asyncio.wait_for(task, 2.0)
    assert time.monotonic() - t0 < 0.5
    assert out.text == "Stopped listening." and out.say is False
    assert FakeSource.instances[0].stopped and shazam.calls == []
    assert card.sent[-1] == ("song_close", {"sid": sid, "reason": "stopped"}, True)


async def test_a_stop_while_shazam_is_asked_cancels_the_request(card, monkeypatch):
    shazam = FakeShazam(delay=5.0)
    monkeypatch.setattr(recognize, "identify", shazam)
    task = asyncio.create_task(catalog().invoke("song_identify", {}, _user("what song is this")))
    for _ in range(200):
        if shazam.calls:
            break
        await asyncio.sleep(0.01)
    assert shazam.calls
    assert song_live.stop("stopped")
    out = await asyncio.wait_for(task, 1.0)
    assert out.text == "Stopped listening."


async def test_the_session_stop_cancels_listening(card, monkeypatch):
    monkeypatch.setattr(song_live, "source_factory", lambda n: FakeSource(n, speed=1.0))
    monkeypatch.setattr(recognize, "identify", FakeShazam())
    task = asyncio.create_task(catalog().invoke("song_identify", {}, _user("what song is this")))
    for _ in range(100):
        if song_live.busy:
            break
        await asyncio.sleep(0.01)
    task.cancel()  # the stop button / "stop": the request is cancelled
    with pytest.raises(asyncio.CancelledError):
        await task
    assert FakeSource.instances[0].stopped and not song_live.busy
    assert card.sent[-1][0] == "song_close" and card.sent[-1][1]["reason"] == "stopped"


async def test_a_second_ask_while_listening_waits_for_the_first(card, monkeypatch):
    monkeypatch.setattr(song_live, "source_factory", lambda n: FakeSource(n, speed=1.0))
    monkeypatch.setattr(recognize, "identify", FakeShazam())
    task = asyncio.create_task(catalog().invoke("song_identify", {}, _user("what song is this")))
    for _ in range(100):
        if song_live.busy:
            break
        await asyncio.sleep(0.01)
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert out.ok and "already listening" in out.text and len(FakeSource.instances) == 1
    song_live.stop()
    await task


async def test_nothing_to_listen_with(card, monkeypatch):
    monkeypatch.setattr(
        song_live, "source_factory", lambda n: FakeSource(n, fail="the microphone couldn't be opened")
    )
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert not out.ok and "microphone couldn't be opened" in out.text and card.sent == []


async def test_the_macs_sound_failing_leaves_the_microphone(card, monkeypatch):
    monkeypatch.setattr(capture, "plan", lambda: (["mic", "mac"], ""))
    monkeypatch.setattr(
        song_live, "source_factory", lambda n: FakeSource(n, fail="no display" if n == "mac" else "")
    )
    monkeypatch.setattr(recognize, "identify", FakeShazam(SONG))
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, None))
    out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    assert out.ok and card.sent[0][1]["source"] == "mic"
    assert any("microphone" in (d["message"] or "") for d in card.states())


@pytest.mark.parametrize(
    "ctx",
    [
        ToolContext(user_text="what song is this?", origin="agent"),  # a model's call, no words of the user's
        _user("summarize this page", origin="agent"),  # the user asked something else: the page said "listen"
        _user("open safari", origin="live"),
        _user("play this song"),  # (even the user's own words, when they don't ask what's playing)
    ],
)
async def test_only_on_the_users_own_request(card, monkeypatch, ctx):
    monkeypatch.setattr(recognize, "identify", FakeShazam(SONG))
    out = await catalog().invoke("song_identify", {}, ctx)
    assert not out.ok and "ask" in out.text
    assert not FakeSource.instances and not card.sent  # never listened, no card


async def test_never_for_a_model_once_a_page_was_read(card, monkeypatch):
    monkeypatch.setattr(recognize, "identify", FakeShazam(SONG))
    ctx = ToolContext(
        user_text="what song is this",
        extras={USERS_WORDS: "what song is this"},
        origin="agent",
        taint=["web_fetch"],
    )
    out = await catalog().invoke("song_identify", {}, ctx)
    assert not out.ok and not FakeSource.instances
    # a model doing what the user's whole sentence asked, with nothing read: it listens
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, None))
    ok = ToolContext(
        user_text="x", extras={USERS_WORDS: "what's the song playing in this video"}, origin="agent"
    )
    out = await catalog().invoke("song_identify", {}, ok)
    assert out.ok and "Blinding Lights" in out.text


async def test_refused_in_library_mode_and_when_off(card, monkeypatch):
    monkeypatch.setattr(runtime, "_library", True)
    out = await song_tool.song_identify({}, _user("what song is this"))
    assert not out.ok
    monkeypatch.setattr(runtime, "_library", False)
    monkeypatch.setenv("NEO_SONG_ID", "false")
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    out = await song_tool.song_identify({}, _user("what song is this"))
    assert not out.ok and "settings" in out.text
    assert not FakeSource.instances


def test_an_embedded_agent_never_listens_unless_its_host_turns_it_on(tmp_path):
    from neo2.core.config import LibrarySettings

    assert LibrarySettings(data_dir=tmp_path / "agent").song_id is False
    assert LibrarySettings(data_dir=tmp_path / "agent2", song_id=True).song_id is True


async def test_whats_heard_never_reaches_the_bus_a_log_or_the_card(card, monkeypatch, caplog):
    from neo2.core.events import app_bus, bus

    events: list = []
    un1, un2 = bus().subscribe(events.append), app_bus().subscribe(events.append)
    clips: list[np.ndarray] = []

    async def fake(pcm):
        clips.append(pcm.copy())
        return SONG if len(clips) == 2 else None

    monkeypatch.setattr(recognize, "identify", fake)
    monkeypatch.setattr(recognize, "cover", lambda url: asyncio.sleep(0, None))
    caplog.set_level("DEBUG")
    try:
        out = await catalog().invoke("song_identify", {}, _user("what song is this"))
    finally:
        un1()
        un2()
    assert out.ok
    marker = clips[0][:32].tobytes()

    def carries_audio(v) -> bool:
        if isinstance(v, np.ndarray | bytes | bytearray | memoryview):
            return True
        if isinstance(v, dict):
            return any(carries_audio(x) for x in v.values())
        if isinstance(v, list | tuple):
            return any(carries_audio(x) for x in v)
        return False

    assert not any(carries_audio(e.data) for e in events)
    for _, data, _ in card.sent:
        assert not carries_audio(data) and set(data) <= {
            "sid",
            "source",
            "listen_s",
            "status",
            "level",
            "elapsed_s",
            "message",
            "result",
            "reason",
        }
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Blinding" not in text and marker.hex() not in text and len(text) < 2000
    (src,) = FakeSource.instances
    assert src.stopped
