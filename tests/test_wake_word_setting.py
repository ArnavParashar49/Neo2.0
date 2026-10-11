""""Listen for 'Neo'" (settings().wake_word). On: the microphone stream stays open for the wake word.
Off: no stream while NEO is idle — a wake by hand opens it, NEO listens as after a spoken "Neo", and
idle closes it again. Switched while NEO runs, kept in ~/.neo/.env, and a host's or NEO.app's page's
alone to switch.

No real device anywhere: sounddevice's streams are stand-ins whose frames a test pushes itself."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types

import pytest

from neo2.core.events import EventBus
from tests.test_v2_ws_security import APP, PAGE, _Peer, _with_server
from tests.test_voice_wake import FakeLive, heard

sd = pytest.importorskip("sounddevice")
audio = pytest.importorskip("neo2.core.voice.audio")

TIMEOUT_S = 0.3  # listen_timeout_s here (the silence watch ticks every 0.25 s)
_FRAME = b"\x01\x00" * 320  # 20 ms


class _Stream:
    """sounddevice's stream, faked: counts what was opened, and never touches a device."""

    opened: list[_Stream] = []

    def __init__(self, *a, callback=None, **kw):
        self.cb, self.closed, self.kind = callback, False, "in" if "callback" in kw or callback else "out"
        _Stream.opened.append(self)

    def start(self):
        pass

    def abort(self):
        pass

    def close(self):
        self.closed = True

    def write(self, chunk):
        pass


def _inputs() -> list[_Stream]:
    """The input streams open now."""
    return [s for s in _Stream.opened if s.kind == "in" and not s.closed]


class _Spotter:
    """The wake spotter, faked: says what the test tells it to, and counts the frames it was fed."""

    def __init__(self, *a):
        self.fed, self.say, self.resets = 0, [], 0

    def feed(self, frame):
        self.fed += 1
        return self.say.pop(0) if self.say else None

    def reset(self):
        self.resets += 1


@pytest.fixture
def devices(monkeypatch):
    """No real audio: fake streams, PortAudio's refresh a no-op, no watcher thread."""
    _Stream.opened = []
    calls: list[str] = []
    monkeypatch.setattr(sd, "RawInputStream", _Stream)
    monkeypatch.setattr(sd, "OutputStream", _Stream)
    monkeypatch.setattr(sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(sd, "_initialize", lambda: calls.append("initialize"))
    monkeypatch.setattr(audio, "_register", lambda o: audio._streams.add(o))
    monkeypatch.setattr(audio, "_taps", ())
    monkeypatch.setattr(audio, "_listeners", ())
    monkeypatch.setattr(audio, "_yielders", ())
    yield calls
    for o in list(audio._streams):
        audio._streams.discard(o)


def _settings(monkeypatch, wake_word: bool) -> None:
    from neo2.core import config, events

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("NEO_LISTEN_TIMEOUT_S", str(TIMEOUT_S))
    monkeypatch.setenv("NEO_WAKE_WORD", "true" if wake_word else "false")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    assert config.settings().wake_word is wake_word


def _switch(on: bool) -> None:
    """What config.set_wake_word does to the live settings (its file is tested by itself below)."""
    from neo2.core import config

    config.settings().wake_word = on


async def _live(monkeypatch):
    """A LiveVoice started for real — its Mic, its loop — over fake streams, a fake spotter and
    sessions that open without the network (with the real silence watch)."""
    from neo2.core.tools import load_all
    from neo2.core.voice import live as live_mod

    load_all()
    monkeypatch.setattr(live_mod, "WakeSpotter", _Spotter)
    monkeypatch.setitem(sys.modules, "silero_vad", types.ModuleType("silero_vad"))  # no VAD model: the energy gate

    class Session:
        memory_context = ""
        voice_owned = False

    async def on_text(text):
        pass

    v = live_mod.LiveVoice(on_text, Session())
    v.opened = 0

    async def ensure():
        async with v._connect_lock:
            if v._live:
                return
            v._closing = False
            v._live, v._live_cm = FakeLive(), None
            v._last_activity = v._session_open = v._last_user_speech = time.time()
            v._words_at = 0.0
            v._turn_open, v._utterance, v._turn_audio = False, "", False
            v.opened += 1
            v._tasks.append(asyncio.create_task(v._silence_watch(v._live)))

    monkeypatch.setattr(v, "_ensure_session", ensure)
    await v.start()
    return v


async def _frames(v, n: int = 3) -> None:
    """The microphone delivers `n` frames (as PortAudio would), and the voice's loop takes them."""
    for _ in range(n):
        stream = v.mic._stream
        if stream is None:
            break
        stream.cb(_FRAME, 320, None, None)
        await asyncio.sleep(0.03)


async def _until(cond, s: float = 3.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < s:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


# ---- Gemini Live ---------------------------------------------------------------------------------


def test_on_the_microphone_is_open_from_the_start_as_before(devices, monkeypatch):
    _settings(monkeypatch, True)

    async def run():
        v = await _live(monkeypatch)
        assert len(_inputs()) == 1 and v.mic.active and audio.mic_running() and not v._mic_parked
        await _frames(v)
        assert v._wake.fed == 3  # the spotter hears the room
        v._wake.say = ["wake"]
        await _frames(v, 1)
        assert await _until(lambda: v._live is not None) and v._listening  # "Neo" wakes it
        assert await _until(lambda: v._live is None)  # nothing said: idle again…
        await _frames(v)
        assert len(_inputs()) == 1 and v.mic.active  # …and the microphone is still open
        await v.close()

    asyncio.run(run())


def test_off_no_microphone_at_idle_a_wake_by_hand_opens_it_and_idle_closes_it(devices, monkeypatch, capsys):
    _settings(monkeypatch, False)

    async def run():
        v = await _live(monkeypatch)
        # Idle: no stream was ever opened, nothing is fed to the spotter, nothing to tap.
        await asyncio.sleep(0.1)
        assert _inputs() == [] and not v.mic.active and not audio.mic_running() and v._mic_parked
        assert v._wake.fed == 0
        # Typed words open a session, never the microphone.
        await v.send_text("what's the capital of France?")
        assert v._live is not None and not v._listening and _inputs() == []
        v._closing = True
        await v._close_session()
        # The summon keys (the shell's `wake`, a host's {"cmd":"wake"}): the microphone opens, NEO listens.
        await v.wake("ui")
        assert len(_inputs()) == 1 and v.mic.active and audio.mic_running()
        assert v._live is not None and v._listening
        live = v._live
        await _frames(v, 4)
        assert len(live.audio) == 4 and v._wake.fed == 4  # the room goes up; the spotter runs (for "stop")
        # Nothing said within listen_timeout_s: idle, and the stream is closed again.
        assert await _until(lambda: v._live is None)
        assert await _until(lambda: not v.mic.active)
        assert _inputs() == [] and not audio.mic_running() and v._mic_parked and not v._listening
        fed = v._wake.fed
        await asyncio.sleep(0.1)
        assert v._wake.fed == fed
        # …and the next summon opens it again: the loop was waiting, not gone.
        await v.wake("ui")
        live = v._live
        await _frames(v, 2)
        assert len(_inputs()) == 1 and len(live.audio) == 2
        await v.close()
        assert _inputs() == []

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "[voice] wake word off: the microphone opens only when summoned" in out
    # One line per transition (the second session was ended by close(), not by going idle).
    assert out.count("[voice] microphone on (ui)") == 2 and out.count("[voice] microphone off (idle)") == 1
    assert out.index("microphone on (ui)") < out.index("woke (ui)") < out.index("microphone off (idle)")


def test_off_the_follow_up_window_keeps_the_microphone_until_the_session_closes(devices, monkeypatch):
    """After a wake by hand the rules are the session's own: words keep it open, quiet after the
    answer closes it — and the microphone with it, not before."""
    _settings(monkeypatch, False)
    from tests.test_voice_wake import DONE

    async def run():
        v = await _live(monkeypatch)
        await v.wake("ui")
        ub, mb = [], []
        await v._on_message(heard("what time is it", finished=False), ub, mb, v._live)
        await asyncio.sleep(TIMEOUT_S + 0.4)
        await _frames(v, 2)
        assert v._live is not None and v.mic.active  # the model is answering: all still open
        await v._on_message(DONE, ub, mb, v._live)
        assert await _until(lambda: v._live is None, 4.0)
        assert await _until(lambda: not v.mic.active) and _inputs() == []
        await v.close()

    asyncio.run(run())


def test_off_push_to_talk_opens_the_microphone_too(devices, monkeypatch):
    _settings(monkeypatch, False)

    async def run():
        v = await _live(monkeypatch)
        await v.push_to_talk(True)
        assert len(_inputs()) == 1 and v._listening and v._ptt
        await asyncio.sleep(TIMEOUT_S + 0.5)
        await _frames(v, 2)
        assert v._live is not None and v.mic.active  # held: the clock is paused
        await v.push_to_talk(False)
        assert await _until(lambda: v._live is None) and await _until(lambda: not v.mic.active)
        await v.close()

    asyncio.run(run())


def test_off_a_spoken_neo_wakes_nothing(devices, monkeypatch):
    """The microphone can be open with the wake word off (a song being named through it): "Neo" said
    then is not a wake."""
    _settings(monkeypatch, False)

    async def run():
        v = await _live(monkeypatch)
        await v.wake("ui")
        untap = audio.tap(lambda frame: None)  # NEO is naming a song through the voice's microphone
        v._closing = True
        await v._close_session()
        assert v.mic.active  # idle, but the tap keeps it
        v._wake.say = ["wake", "wake_final"]
        await _frames(v, 3)
        assert v._live is None and not v._listening and v.opened == 1
        untap()
        await _frames(v, 1)  # the song is done: the next frame finds nothing left to listen for
        assert await _until(lambda: not v.mic.active) and _inputs() == []
        await v.close()

    asyncio.run(run())


def test_switched_while_running_both_ways(devices, monkeypatch, capsys):
    _settings(monkeypatch, True)

    async def run():
        v = await _live(monkeypatch)
        assert len(_inputs()) == 1
        # Off while idle: closed at once.
        _switch(False)
        await v.set_wake_word(False)
        assert _inputs() == [] and not v.mic.active and v._mic_parked
        # On again: open at once, and "Neo" works.
        _switch(True)
        await v.set_wake_word(True)
        assert len(_inputs()) == 1 and v.mic.active
        v._wake.say = ["wake"]
        await _frames(v, 1)
        assert await _until(lambda: v._live is not None) and v._listening
        # Off in the middle of that session: nothing changes until NEO is idle…
        _switch(False)
        await v.set_wake_word(False)
        await _frames(v, 2)
        assert v.mic.active and v._live is not None and len(v._live.audio) == 2
        # …then the microphone closes with the session.
        assert await _until(lambda: v._live is None) and await _until(lambda: not v.mic.active)
        assert _inputs() == []
        # On in the middle of a session a wake by hand opened: it simply stays open afterwards.
        await v.wake("ui")
        _switch(True)
        await v.set_wake_word(True)
        assert await _until(lambda: v._live is None)
        await _frames(v, 2)
        assert v.mic.active and len(_inputs()) == 1
        await v.close()

    asyncio.run(run())
    out = capsys.readouterr().out
    assert out.count("[voice] wake word off: the microphone opens only when summoned") == 2
    assert "[voice] microphone on (wake word)" in out and "[voice] wake word on" in out


def test_a_microphone_that_cannot_open_is_said_and_nothing_listens(devices, monkeypatch, capsys):
    _settings(monkeypatch, False)
    from neo2.core.events import bus

    notes = []

    async def run():
        v = await _live(monkeypatch)
        bus().subscribe(lambda ev: notes.append(ev.data.get("text") or ev.data.get("message")) if ev.type == "note" else None)

        def refuse(**kw):
            raise RuntimeError("Error querying device -1")

        monkeypatch.setattr(sd, "RawInputStream", refuse)
        await v.wake("ui")
        assert not v._listening and not v.mic.active and v._mic_parked
        await v.close()

    asyncio.run(run())
    assert "[voice] couldn't open the microphone" in capsys.readouterr().out
    assert any("microphone" in str(n) for n in notes)


# ---- the device watcher --------------------------------------------------------------------------


def test_the_device_watcher_leaves_a_microphone_closed_on_purpose_alone(devices, monkeypatch):
    _settings(monkeypatch, False)
    calls = devices

    async def run():
        v = await _live(monkeypatch)  # the speaker is open, the microphone is not
        assert v.spk.active and not v.mic.active and not v.mic.stalled()
        assert audio.reopen_all("the input device changed") is True
        assert _inputs() == [] and v.mic._stream is None  # the speaker came back; no microphone appeared
        await v.wake("ui")
        assert len(_inputs()) == 1
        audio.reopen_all("the input device changed")  # while it is wanted it follows the device, as before
        assert len(_inputs()) == 1 and v.mic.active
        assert await _until(lambda: not v.mic.active)  # idle: closed again
        monkeypatch.setattr(audio, "_STALL_S", 0.0)  # "no frame for this long = dead"
        assert not v.mic.stalled()  # closed on purpose is not broken: the watcher has nothing to repair
        assert not any(isinstance(o, audio.Mic) and o.stalled() for o in list(audio._streams))
        audio.reopen_all("the audio stopped working")
        assert _inputs() == [] and v.mic._stream is None
        await v.close()

    asyncio.run(run())
    assert "initialize" in calls


def test_a_restarted_microphone_drops_what_the_last_session_left(devices):
    """stop() leaves a marker (and maybe frames) in the queue: a later start() begins clean."""
    mic = audio.Mic()
    mic.start()
    mic._stream.cb(_FRAME, 320, None, None)
    mic.stop()
    assert mic.q.qsize() == 2 and not mic.active
    mic.start()
    assert mic.q.qsize() == 0 and mic.active and len(_inputs()) == 1
    mic._stream.cb(_FRAME, 320, None, None)
    assert asyncio.run(mic.read()) == _FRAME
    mic.stop()


# ---- naming a song -------------------------------------------------------------------------------


def test_off_the_song_opens_its_own_stream_and_hands_it_over_when_neo_is_summoned(devices, monkeypatch):
    _settings(monkeypatch, False)
    from neo2.core.tools.music import capture

    async def run():
        v = await _live(monkeypatch)
        got = []
        src = capture.MicSource()
        src.start(got.append)  # NEO idle, its microphone closed: the song listens on a stream of its own
        own = src._stream
        assert own is not None and src._untap is None and _inputs() == [own]
        assert not v.mic.active  # …which is not the voice's: no spotter, nothing of it reaches the model
        own.cb(_FRAME, 320, None, None)
        assert len(got) == 1 and v._wake.fed == 0
        # Summoned meanwhile: the voice opens the device, the song's own stream goes first and it
        # hears through a tap from then on — never two input streams at once.
        await v.wake("ui")
        assert own.closed and src._stream is None and src._untap is not None
        assert _inputs() == [v.mic._stream]
        await _frames(v, 2)
        assert len(got) == 3
        # The session ends while the song still listens: the microphone stays until it is done.
        v._closing = True
        await v._close_session()
        await _frames(v, 2)
        assert v.mic.active and len(got) == 5
        src.stop()
        await _frames(v, 1)
        assert await _until(lambda: not v.mic.active) and _inputs() == []
        assert audio._taps == () and audio._yielders == ()
        await v.close()

    asyncio.run(run())


class _FakeAnalyzer:
    """Apple's sound classifier, faked: takes buffers, classifies nothing."""

    def __init__(self):
        self.fed = 0

    def analyzeAudioBuffer_atAudioFramePosition_(self, buffer, position):
        self.fed += 1


class _FakeBuffer:
    def __init__(self):
        self.data = bytearray(640)

    def int16ChannelData(self):
        data = self.data
        return [types.SimpleNamespace(as_buffer=lambda n: memoryview(data))]

    def setFrameLength_(self, n):
        pass


def _monitor(monkeypatch):
    """Gentle check-ins switched on for real (its tap, its worker), Apple's classifier faked."""
    from neo2.core.voice import wellbeing

    analyzer = _FakeAnalyzer()
    monkeypatch.setattr(
        wellbeing.WellbeingMonitor,
        "_make_analyzer",
        staticmethod(lambda: (analyzer, object(), types.SimpleNamespace(), _FakeBuffer())),
    )
    m = wellbeing.WellbeingMonitor(loop=asyncio.get_running_loop(), on_event=lambda kind: None, is_ignored=lambda: False)
    m.set_enabled(True)
    m.analyzer = analyzer
    return m


def test_off_check_ins_never_hold_the_microphone_open(devices, monkeypatch):
    """Gentle check-ins hear only frames a microphone NEO opened for something else captures: with
    the wake word off, a summon opens it, the monitor hears that session, and idle closes it — the
    check-ins' tap never keeps it open (macOS's orange dot goes off)."""
    _settings(monkeypatch, False)

    async def run():
        v = await _live(monkeypatch)
        m = _monitor(monkeypatch)
        try:
            assert _inputs() == [] and v._mic_parked and not audio.tapped()  # switching check-ins on opens nothing
            await v.wake("ui")
            await _frames(v, 3)
            assert await _until(lambda: m.analyzer.fed >= 1)  # the monitor hears the session the user opened
            assert await _until(lambda: v._live is None)  # nothing said: idle…
            assert await _until(lambda: not v.mic.active)  # …and the microphone closes, the tap notwithstanding
            assert _inputs() == [] and v._mic_parked and not audio.mic_running()
        finally:
            await asyncio.to_thread(m.close)
            await v.close()

    asyncio.run(run())


def test_local_off_check_ins_never_hold_the_microphone_open(devices, monkeypatch):
    _settings(monkeypatch, False)

    async def run():
        v = await _local(monkeypatch, [""])
        m = _monitor(monkeypatch)
        try:
            await v.wake("ui")
            assert len(_inputs()) == 1
            await _frames(v, 1)
            assert await _until(lambda: not v._armed.is_set() and not v._capturing)
            await _frames(v, 1)
            assert await _until(lambda: not v.mic.active)
            assert _inputs() == [] and v._mic_parked
        finally:
            await asyncio.to_thread(m.close)
            await v.close()

    asyncio.run(run())


def test_a_listen_only_tap_hears_frames_but_never_counts_as_using_the_microphone(devices):
    heard: list[bytes] = []
    untap = audio.tap(heard.append, hold=False)
    assert not audio.tapped()  # the voice may close its microphone whenever it likes
    mic = audio.Mic()
    mic.start()
    mic._stream.cb(_FRAME, 320, None, None)
    assert heard == [_FRAME]
    hold = audio.tap(lambda frame: None)  # naming a song: that one does keep it
    assert audio.tapped()
    hold()
    untap()
    assert not audio.tapped() and audio._listeners == () and audio._taps == ()
    mic.stop()


def test_off_a_song_named_and_over_at_idle_never_involves_the_voice(devices, monkeypatch):
    _settings(monkeypatch, False)
    from neo2.core.tools.music import capture

    async def run():
        v = await _live(monkeypatch)
        src = capture.MicSource()
        src.start(lambda pcm: None)
        own = src._stream
        src.stop()
        assert own.closed and _inputs() == [] and audio._yielders == ()
        await v.wake("ui")  # a later summon finds nobody to ask
        assert len(_inputs()) == 1
        await v.close()

    asyncio.run(run())


# ---- the on-device voice -------------------------------------------------------------------------


async def _local(monkeypatch, utterances):
    """A LocalVoice started for real over fake streams; its models aren't loaded, and each utterance
    it captures is the next of `utterances` ("" = nobody spoke)."""
    from neo2.core.voice import local as local_mod

    said = list(utterances)
    handled: list[str] = []

    async def on_text(text):
        handled.append(text)

    v = local_mod.LocalVoice(on_text)
    monkeypatch.setattr(v, "_load_models", lambda: setattr(v, "_wake", _Spotter()))

    async def capture():
        v._capturing = True
        await asyncio.sleep(0.05)
        v._capturing = False
        return said.pop(0) if said else ""

    monkeypatch.setattr(v, "_capture_utterance", capture)
    await v.start()
    v.handled = handled
    return v


def test_local_off_no_microphone_at_idle_a_wake_opens_it_and_idle_closes_it(devices, monkeypatch, capsys):
    _settings(monkeypatch, False)

    async def run():
        v = await _local(monkeypatch, ["", "note this down", ""])
        await asyncio.sleep(0.1)
        assert _inputs() == [] and not v.mic.active and v._wake.fed == 0
        # Summoned, nothing said: open, then closed again.
        await v.wake("ui")
        assert len(_inputs()) == 1 and v.mic.active
        await _frames(v, 1)
        assert await _until(lambda: not v._armed.is_set() and not v._capturing)
        await _frames(v, 1)
        assert await _until(lambda: not v.mic.active) and _inputs() == []
        # Summoned, something said: it stays open through the answer and the follow-up window…
        await v.wake("ui")
        await _frames(v, 1)
        assert await _until(lambda: v.handled == ["note this down"])
        assert await _until(lambda: v._followup_until > 0)
        await _frames(v, 1)  # the follow-up capture begins: nobody speaks
        assert v.mic.active
        assert await _until(lambda: v._followup_until == 0.0 and not v._capturing)
        # …and closes when the loop has nothing left to do.
        await _frames(v, 1)
        assert await _until(lambda: not v.mic.active) and _inputs() == []
        # "Neo" said with the wake word off (push-to-talk has the microphone open) is not a wake.
        await v.close()

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "[voice] wake word off: the microphone opens only when summoned" in out
    assert out.count("[voice] microphone on (ui)") == 2 and out.count("[voice] microphone off (idle)") == 2


def test_local_switched_while_running_and_on_keeps_it_open(devices, monkeypatch):
    _settings(monkeypatch, True)

    async def run():
        v = await _local(monkeypatch, [""])
        assert len(_inputs()) == 1 and v.mic.active
        await _frames(v, 3)
        assert v.mic.active and v._wake.fed == 3  # idle with the wake word on: open, the spotter fed
        _switch(False)
        await v.set_wake_word(False)
        assert not v.mic.active and _inputs() == []
        _switch(True)
        await v.set_wake_word(True)
        assert v.mic.active and len(_inputs()) == 1
        v._wake.say = ["wake"]
        await _frames(v, 1)  # "Neo" — heard again
        assert await _until(lambda: v._capturing)
        _switch(False)
        await v.set_wake_word(False)  # mid-utterance: not now…
        assert v.mic.active
        assert await _until(lambda: not v._capturing)
        await _frames(v, 2)
        assert await _until(lambda: not v.mic.active)  # …at the next idle
        await v.close()

    asyncio.run(run())


# ---- kept in ~/.neo/.env -------------------------------------------------------------------------


def _reloaded(monkeypatch, env):
    """The settings a new NEO.app process would read from this env file (effects on for this thread
    only, as NEO.app runs: under the suite no .env file is ever read)."""
    from neo2.core import config, effects

    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    token = effects._as_app.set(True)
    try:
        return config.Settings(_env_file=env)
    finally:
        effects._as_app.reset(token)


def test_set_wake_word_writes_one_line_and_leaves_the_rest_of_the_file_alone(tmp_path, monkeypatch):
    from neo2.core import config

    env = tmp_path / ".env"
    theirs = "# my keys\nGEMINI_API_KEY=keep-me\n\n  NEO_LISTEN_TIMEOUT_S = 7 \nNEO_UI_HOST=notch\n"
    env.write_text(theirs)
    config.set_wake_word(False, env_path=env)
    assert env.read_text() == theirs + "\nNEO_WAKE_WORD=false\n"  # added at the end; nothing else moved
    assert oct(os.stat(env).st_mode & 0o777) == "0o600"
    assert config.settings().wake_word is False
    # A new process reads it back (what NEO does at its next start).
    again = _reloaded(monkeypatch, env)
    assert again.wake_word is False and again.ui_host == "notch" and again.listen_timeout_s == 7
    # Switched back: the line is replaced where it stands, once.
    config.set_wake_word(True, env_path=env)
    assert env.read_text() == theirs + "\nNEO_WAKE_WORD=true\n"
    assert config.settings().wake_word is True and _reloaded(monkeypatch, env).wake_word is True
    # In the middle of the user's file it stays in the middle; a second such line goes.
    env.write_text("A=1\nNEO_WAKE_WORD=true\n# note\nB=2\nNEO_WAKE_WORD=true\n")
    config.set_wake_word(False, env_path=env)
    assert env.read_text() == "A=1\nNEO_WAKE_WORD=false\n# note\nB=2\n"
    # The other setting NEO writes there doesn't disturb it, nor it the other.
    config.set_ui_host("window", env_path=env)
    config.set_wake_word(True, env_path=env)
    assert env.read_text() == "A=1\nNEO_WAKE_WORD=true\n# note\nB=2\n\nNEO_UI_HOST=window\n"
    # No file yet: it is made, private.
    fresh = tmp_path / "new" / ".env"
    config.set_wake_word(False, env_path=fresh)
    assert fresh.read_text() == "NEO_WAKE_WORD=false\n" and oct(os.stat(fresh).st_mode & 0o777) == "0o600"
    # Only true or false; and never the user's real file while effects are off.
    for bad in ("false", 0, None):
        with pytest.raises(ValueError):
            config.set_wake_word(bad, env_path=env)
    monkeypatch.setenv("NEO_NO_EFFECTS", "1")
    with pytest.raises(RuntimeError):
        config.set_wake_word(False)


def test_wake_word_defaults_on_and_reads_its_env(monkeypatch):
    from neo2.core import config

    monkeypatch.delenv("NEO_WAKE_WORD", raising=False)
    assert config.Settings().wake_word is True
    monkeypatch.setenv("NEO_WAKE_WORD", "0")
    assert config.Settings().wake_word is False


# ---- who may switch it ---------------------------------------------------------------------------


def test_the_hello_says_whether_neo_listens_for_its_name(monkeypatch):
    async def body(srv, port, seen):
        async with _Peer(port, role="host") as host:
            first = host.hello["wake_word"]
        _switch(False)
        async with _Peer(port, origin=PAGE) as page:
            return first, page.hello["wake_word"]

    assert _with_server(monkeypatch, body) == (True, False)


def test_set_is_a_hosts_and_neo_apps_pages_alone(monkeypatch):
    """A tool that read the token, and NEO's page in the notch: refused, and the core's command
    handler never hears of it. A host and NEO.app's own page: done."""

    async def body(srv, port, seen):
        got = {}
        for who, kw in (("tool", {}), ("page", {"origin": PAGE}), ("host", {"role": "host"}), ("app", {"origin": APP})):
            async with _Peer(port, **kw) as peer:
                await peer.send(cmd="set", name="wake_word", value=False)
                got[who] = (await peer.next("set_result"))["data"]
        return got, list(seen)

    got, seen = _with_server(monkeypatch, body)
    refused = {"name": "wake_word", "ok": False, "error": "forbidden"}
    assert got == {"tool": refused, "page": refused, "host": {"name": "wake_word", "ok": True}, "app": {"name": "wake_word", "ok": True}}
    assert seen == [{"cmd": "set", "name": "wake_word", "value": False}] * 2


def test_set_takes_only_the_settings_it_knows_with_values_of_their_type(monkeypatch):
    async def body(srv, port, seen):
        got = []
        async with _Peer(port, role="host") as host:
            for name, value in (("confirm_destructive", False), ("gemini_api_key", "x"), ("", True), ("wake_word", "false"), ("wake_word", 0), ("wake_word", None)):
                await host.send(cmd="set", name=name, value=value)
                got.append((await host.next("set_result"))["data"].get("error"))
            await host.send(cmd="set", name="wake_word")
            got.append((await host.next("set_result"))["data"].get("error"))
        return got, list(seen)

    got, seen = _with_server(monkeypatch, body)
    assert got == ["unknown", "unknown", "unknown", "invalid", "invalid", "invalid", "invalid"]
    assert seen == []
    assert ws_settable() == {"wake_word": bool, "wellbeing_checkins": bool}


def ws_settable():
    from neo2.core.server import ws as ws_mod

    return ws_mod.SETTABLE


def test_a_set_that_fails_says_so(monkeypatch):
    async def failing(cmd):
        raise RuntimeError("the file can't be written")

    async def body(srv, port, seen):
        async with _Peer(port, role="host") as host:
            await host.send(cmd="set", name="wake_word", value=False)
            return (await host.next("set_result"))["data"]

    assert _with_server(monkeypatch, body, on_command=failing) == {"name": "wake_word", "ok": False, "error": "failed"}


def test_the_core_keeps_it_applies_it_to_the_voice_and_tells_every_client(monkeypatch, tmp_path):
    """{"cmd":"set","name":"wake_word"} end to end, minus the user's own env file: written, the live
    settings, the running voice, a `settings` event to everyone connected and the next hello."""
    from neo2.core import app as app_mod
    from neo2.core import config

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=keep-me\n")
    real = config.set_wake_word
    monkeypatch.setattr(app_mod, "set_wake_word", lambda on: real(on, env_path=env))

    class Voice:
        calls: list[bool] = []

        async def set_wake_word(self, on):
            self.calls.append(on)

    async def body(srv, port, seen):
        app = app_mod.App.__new__(app_mod.App)
        app.voice = Voice()
        srv._on_command = app.on_command
        async with _Peer(port, origin=PAGE) as page, _Peer(port, role="host") as host:
            await host.send(cmd="set", name="wake_word", value=False)
            told = [(await page.next("settings"))["data"], (await host.next("settings"))["data"]]
            result = (await host.next("set_result"))["data"]
            async with _Peer(port) as later:
                hello = later.hello["wake_word"]
            await host.send(cmd="set", name="wake_word", value=True)
            told.append((await page.next("settings"))["data"])
            # A tool's attempt changes nothing.
            async with _Peer(port) as tool:
                await tool.send(cmd="set", name="wake_word", value=False)
                await tool.next("set_result")
            return told, result, hello, config.settings().wake_word

    told, result, hello, now = _with_server(monkeypatch, body)
    off = {"wellbeing_checkins": False}  # (check-ins: off by default, told alongside)
    assert told == [{"wake_word": False, **off}, {"wake_word": False, **off}, {"wake_word": True, **off}]
    assert result == {"name": "wake_word", "ok": True} and hello is False and now is True
    assert Voice.calls == [False, True]
    assert env.read_text() == "GEMINI_API_KEY=keep-me\n\nNEO_WAKE_WORD=true\n"


def test_without_a_voice_the_setting_is_still_kept_and_told(monkeypatch, tmp_path):
    from neo2.core import app as app_mod
    from neo2.core import config
    from neo2.core.events import bus

    monkeypatch.setattr("neo2.core.events._bus", EventBus())
    env = tmp_path / ".env"
    real = config.set_wake_word
    monkeypatch.setattr(app_mod, "set_wake_word", lambda on: real(on, env_path=env))
    seen = []
    bus().subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))
    app = app_mod.App.__new__(app_mod.App)
    app.voice = None
    asyncio.run(app.on_command({"cmd": "set", "name": "wake_word", "value": False}))
    asyncio.run(app.on_command({"cmd": "set", "name": "wake_word", "value": "true"}))  # not a bool: nothing
    asyncio.run(app.on_command({"cmd": "set", "name": "voice", "value": False}))  # not settable: nothing
    assert env.read_text() == "NEO_WAKE_WORD=false\n" and config.settings().wake_word is False
    assert seen == [("settings", {"wake_word": False, "wellbeing_checkins": False})]


def test_gentle_check_ins_are_switched_the_same_way(monkeypatch, tmp_path):
    """{"cmd":"set","name":"wellbeing_checkins"} end to end, minus the user's own env file: written, the
    live settings, the running monitor (it never opens the microphone), a `settings` event to everyone
    connected and the next hello. A tool, or NEO's page in the notch, can't."""
    from neo2.core import app as app_mod
    from neo2.core import config

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=keep-me\n")
    real = config.set_wellbeing_checkins
    monkeypatch.setattr(app_mod, "set_wellbeing_checkins", lambda on: real(on, env_path=env))

    class Monitor:
        calls: list[bool] = []

        def set_enabled(self, on):
            self.calls.append(on)

    async def body(srv, port, seen):
        app = app_mod.App.__new__(app_mod.App)
        app.voice, app.wellbeing = None, Monitor()
        srv._on_command = app.on_command
        async with _Peer(port, origin=PAGE) as page, _Peer(port, role="host") as host:
            await page.send(cmd="set", name="wellbeing_checkins", value=True)  # the notch's page: no
            refused = (await page.next("set_result"))["data"]
            await host.send(cmd="set", name="wellbeing_checkins", value=True)
            told = [(await page.next("settings"))["data"], (await host.next("settings"))["data"]]
            result = (await host.next("set_result"))["data"]
            await host.send(cmd="set", name="wellbeing_checkins", value="yes")  # not a bool
            invalid = (await host.next("set_result"))["data"]
            async with _Peer(port) as later:
                hello = later.hello.get("wellbeing_checkins")
            return refused, told, result, invalid, hello, config.settings().wellbeing_checkins

    refused, told, result, invalid, hello, now = _with_server(monkeypatch, body)
    assert refused == {"name": "wellbeing_checkins", "ok": False, "error": "forbidden"}
    on = {"wake_word": True, "wellbeing_checkins": True}
    assert told == [on, on] and result == {"name": "wellbeing_checkins", "ok": True}
    assert invalid == {"name": "wellbeing_checkins", "ok": False, "error": "invalid"}
    assert hello is True and now is True and Monitor.calls == [True]
    assert env.read_text() == "GEMINI_API_KEY=keep-me\n\nNEO_WELLBEING_CHECKINS=true\n"
