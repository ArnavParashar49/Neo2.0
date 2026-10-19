"""The on-device voice is the user's to ask for (neo2/core/voice/pipeline.py, neo2/core/app.py): with no
Gemini key and its 2.9 GB of models not on this Mac, NEO starts no download — it says what the voice
needs and types meanwhile; Brains & Keys and the chat's card ask for it (`voice_download`, from the
windows the user has, never a tool) and everyone hears how far it is; once it is here, or a key
arrives, the voice starts without a restart. And `say`, the last resort, speaks with the voice this
Mac has. Everything is faked: no model is loaded, nothing is downloaded or spoken."""

from __future__ import annotations

import asyncio
import json

import pytest

from neo2.core import brains, config, events, fetch, health
from neo2.core.voice import pipeline
from tests.test_v2_ws_security import APP, PAGE, _Peer, _with_server

KEY = "AIzaTEST-0000-abcdefghijklmnopqrstuv"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_started", None)
    monkeypatch.setattr(brains, "voice_kick", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())


def _voice(monkeypatch, mode="auto", key=""):
    monkeypatch.setenv("NEO_VOICE", mode)
    if key:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setattr(config, "_settings", None)


def _no_local(monkeypatch):
    made: list = []

    async def make_local(on_text):
        made.append("local")
        return "the local voice"

    monkeypatch.setattr(pipeline, "make_local", make_local)
    return made


# ---- make_voice: no download unasked -------------------------------------------------------------------


def test_no_key_and_no_local_voice_starts_no_download(monkeypatch):
    _voice(monkeypatch)
    made = _no_local(monkeypatch)
    with pytest.raises(pipeline.VoiceNotSetUp) as e:
        asyncio.run(pipeline.make_voice(lambda t: None, None))
    assert made == []  # the 2.9 GB download that make_local's loaders would start: never
    assert "Gemini key or the on-device voice" in e.value.problem and "2.9 GB" in e.value.problem
    assert "Brains & Keys" in e.value.problem and "Typing works" in e.value.problem


def test_the_voice_set_to_this_mac_but_not_downloaded_says_so(monkeypatch):
    _voice(monkeypatch, "local", KEY)
    made = _no_local(monkeypatch)
    with pytest.raises(pipeline.VoiceNotSetUp) as e:
        asyncio.run(pipeline.make_voice(lambda t: None, None))
    assert made == [] and "isn't downloaded yet" in e.value.problem and "Gemini Live there" in e.value.problem


def test_a_key_live_refused_and_no_local_voice(monkeypatch):
    from neo2.core.voice import live as live_mod

    _voice(monkeypatch, "auto", KEY)
    made = _no_local(monkeypatch)

    class Refused:
        def __init__(self, *a):
            pass

        async def probe(self):
            raise RuntimeError("403 permission denied")

    monkeypatch.setattr(live_mod, "LiveVoice", Refused)
    with pytest.raises(pipeline.VoiceNotSetUp) as e:
        asyncio.run(pipeline.make_voice(lambda t: None, None))
    assert made == [] and "refused this key" in e.value.problem


def test_with_the_local_voice_downloaded_it_starts_as_before(monkeypatch):
    _voice(monkeypatch, "local")
    made = _no_local(monkeypatch)
    monkeypatch.setattr(pipeline, "local_ready", lambda: True)
    assert asyncio.run(pipeline.make_voice(lambda t: None, None)) == "the local voice" and made == ["local"]


# ---- the supervisor: says it once, tries again when something arrives ----------------------------------------


class _App:
    """What the voice supervisor needs of the App."""

    def __init__(self):
        from neo2.core.app import App

        self.voice = None
        self.voice_kick = asyncio.Event()
        self.session = None
        self.kick_voice = App.kick_voice.__get__(self)

    async def handle_text(self, *a, **kw):
        pass

    async def handle_speech(self, *a, **kw):
        pass


def test_no_voice_yet_is_said_once_and_a_kick_brings_it_up(monkeypatch, capsys):
    from neo2.core import app as app_mod
    from neo2.core.voice import devices

    _voice(monkeypatch)
    monkeypatch.setattr(devices, "mic_status", lambda: devices.AUTHORIZED)
    made = _no_local(monkeypatch)
    published: list = []

    async def publish(s=None):
        published.append("brains")

    monkeypatch.setattr(brains, "publish", publish)

    async def go():
        app, stop = _App(), asyncio.Event()
        task = asyncio.create_task(app_mod._voice_supervisor(app, stop))
        await asyncio.sleep(0.05)
        first = dict(health._live)
        app.kick_voice()  # (a brains_reload while nothing changed: the same answer, not said again)
        await asyncio.sleep(0.05)
        monkeypatch.setattr(pipeline, "local_ready", lambda: True)  # the user's download finished
        app.kick_voice()
        await asyncio.sleep(0.05)
        stop.set()
        task.cancel()
        return app, first

    app, first = asyncio.run(go())
    assert "Gemini key or the on-device voice" in first["voice"]
    assert app.voice == "the local voice" and made == ["local"] and "voice" not in health._live
    out = capsys.readouterr().out
    assert out.count("[voice] no voice yet") == 1
    assert brains._started == brains._voice_print(config.settings()) and published == ["brains"]


def test_a_key_saved_while_no_voice_runs_starts_it_without_a_restart(monkeypatch):
    """Brains & Keys saves a Gemini key: the files are applied, the voice is kicked, and the voice that
    comes up uses the key — so nothing says "restart NEO" afterwards."""
    kicked: list = []
    monkeypatch.setattr(brains, "voice_kick", lambda: kicked.append(1))
    monkeypatch.setattr(config, "KEY_STORE_PATH", None)
    monkeypatch.setattr(config, "SAVED_PATH", None)
    asyncio.run(brains._apply(tell=False))
    assert kicked == [1]
    brains.started(config.settings())
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    config.settings().gemini_api_key = KEY
    assert brains.restart_needed() is True
    brains.started(config.settings())  # (what the supervisor does once the voice is up)
    assert brains.restart_needed() is False


def test_the_download_is_heard_by_everyone_and_a_finished_one_kicks_the_voice(monkeypatch):
    from neo2.core.app import App

    seen: list = []
    events.bus().subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))
    published: list = []

    async def publish(s=None):
        published.append("brains")

    monkeypatch.setattr(brains, "publish", publish)

    async def go():
        app = App.__new__(App)
        app.voice, app.voice_kick, app._local_voice_told = None, asyncio.Event(), 0.0
        fetch.downloads._spawn = _scripted()
        fetch.downloads.listen(app.download_changed)
        fetch.downloads.start("local_voice")  # (scripted: ends at once, without "done")
        await fetch.downloads.wait("local_voice")
        await asyncio.sleep(0.01)
        failed = [d for t, d in seen if t == "local_voice"][-1]
        fetch.downloads._spawn = _scripted({"event": "progress", "all_done": 5, "all_total": 10},
                                           {"event": "done", "ok": True})  # fmt: skip
        monkeypatch.setattr(pipeline, "local_ready", lambda: True)
        fetch.downloads.start("local_voice")
        await fetch.downloads.wait("local_voice")
        await asyncio.sleep(0.01)
        return app, failed

    app, failed = asyncio.run(go())
    assert failed["state"] == "failed"
    states = [d["state"] for t, d in seen if t == "local_voice"]
    assert "downloading" in states and states[-1] == "ready"
    assert app.voice_kick.is_set()  # no voice ran: it starts now
    assert len(published) >= 3  # started, failed, started again, done: the summary each time it changed


def _scripted(*evs):
    async def spawn(group):
        for ev in evs:
            yield ev

    return spawn


# ---- voice_download on the websocket -------------------------------------------------------------------------


def _wired():
    """The App's listener on the downloads (neo2.core.app.run wires it): what tells every client."""
    from neo2.core.app import App

    app = App.__new__(App)
    app.voice, app.voice_kick, app._local_voice_told = None, asyncio.Event(), 0.0
    fetch.downloads.listen(app.download_changed)
    return app


def test_voice_download_is_the_users_from_their_windows_and_never_a_tools(monkeypatch):
    async def body(srv, port, seen):
        hold = asyncio.Event()

        async def spawn(group):
            yield {"event": "progress", "all_done": 1, "all_total": 4}
            await hold.wait()
            yield {"event": "done", "ok": True}

        fetch.downloads._spawn = spawn
        _wired()
        out: dict = {}
        async with _Peer(port) as tool:
            await tool.send(cmd="voice_download")
            out["tool"] = (await tool.next("voice_download_result"))["data"]
        assert not fetch.downloads.running("local_voice")
        async with _Peer(port, origin=APP) as app, _Peer(port, origin=PAGE) as page:
            assert app.hello["brains"]["local_voice"]["state"] == "missing"
            await app.send(cmd="voice_download")
            out["app"] = (await app.next("voice_download_result"))["data"]
            out["heard"] = (await page.next("local_voice"))["data"]  # every window hears how far it is
            await page.send(cmd="voice_download")
            out["page"] = (await page.next("voice_download_result"))["data"]
            await page.send(cmd="voice_download", cancel=True)
            out["cancel"] = (await page.next("voice_download_result"))["data"]
            hold.set()
            await fetch.downloads.wait("local_voice")
            await app.send(cmd="voice_download", cancel=True)
            out["nothing"] = (await app.next("voice_download_result"))["data"]
        return out

    out = _with_server(monkeypatch, body)
    assert out["tool"]["ok"] is False and out["tool"]["error"] == "forbidden"
    assert out["app"]["ok"] is True and out["app"]["local_voice"]["state"] == "downloading"
    assert out["heard"]["state"] == "downloading" and out["heard"]["size"] == fetch.LOCAL_VOICE_BYTES
    assert out["page"] == {**out["page"], "ok": False, "error": "busy"}
    assert out["cancel"]["ok"] is True and out["nothing"]["error"] == "not_running"


def test_brains_and_keys_asks_with_its_id_and_hears_the_progress(monkeypatch):
    from tests.test_ws_keys import TICKET, _Keys

    async def body(srv, port, seen):
        srv.shell_said({"t": "keys_ticket", "ticket": TICKET})
        _wired()
        monkeypatch.setattr(pipeline, "local_ready", lambda: True)
        monkeypatch.setattr(fetch, "_ready", None)
        async with _Keys(port) as window:
            await window.send(cmd="voice_download", id="r7")
            here = (await window.next("voice_download_result"))["data"]
            monkeypatch.setattr(pipeline, "local_ready", lambda: False)
            monkeypatch.setattr(fetch, "_ready", None)
            fetch.downloads._spawn = _scripted({"event": "progress", "all_done": 2, "all_total": 4}, {"event": "done", "ok": False})
            await window.send(cmd="voice_download", id="r8")
            started = (await window.next("voice_download_result"))["data"]
            heard = (await window.next("local_voice"))["data"]
        return here, started, heard

    here, started, heard = _with_server(monkeypatch, body)
    assert here == {"ok": False, "error": "ready", "local_voice": {"size": fetch.LOCAL_VOICE_BYTES, "state": "ready"}, "id": "r7"}
    assert started["ok"] is True and started["id"] == "r8"
    assert heard["state"] in ("downloading", "failed")
    assert json.dumps(heard).count(KEY) == 0


# ---- say: the voice this Mac has ------------------------------------------------------------------------


def test_say_uses_the_premium_voice_only_when_this_mac_has_it(monkeypatch):
    from neo2.core import app as app_mod

    monkeypatch.setattr(app_mod, "_say_voice", None)
    monkeypatch.setattr(app_mod, "_installed_voices", lambda: {"com.apple.voice.compact.en-US.Samantha"})
    assert app_mod._say_args() == []  # a new Mac: the user's own system voice
    monkeypatch.setattr(app_mod, "_installed_voices", lambda: {"com.apple.voice.Aman.premium"})
    assert app_mod._say_args() == []  # (looked up once)
    monkeypatch.setattr(app_mod, "_say_voice", None)
    assert app_mod._say_args() == ["-v", "com.apple.voice.Aman.premium"]
    ran: list = []

    class Popen:
        def __init__(self, cmd, **kw):
            ran.append(cmd)

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", Popen)
    monkeypatch.setattr(app_mod, "_say_voice", [])
    app_mod._say_aloud("hello")
    assert ran == [["say", "hello"]]
