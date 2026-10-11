"""Voice wake: the mic goes up to the model only after "Neo" (or the hotkey / push-to-talk), and
NEO goes back to idle when nothing is said to it within `listen_timeout_s`."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from neo2.core.events import EventBus, NeoState

TIMEOUT_S = 0.3  # listen_timeout_s for these tests (the watch ticks every 0.25 s)


class FakeSpk:
    def __init__(self):
        self.played, self.playing, self.last_stop = [], threading.Event(), 0.0
        self.interrupts = 0

    def play_pcm16(self, data):
        self.played.append(data)

    def interrupt(self):
        self.interrupts += 1

    async def wait_done(self):
        return None


class FakeLive:
    def __init__(self):
        self.responses, self.texts, self.audio = [], [], []

    async def send_tool_response(self, function_responses):
        self.responses.extend(function_responses)

    async def send_client_content(self, turns):
        self.texts.append(turns)

    async def send_realtime_input(self, audio):
        self.audio.append(audio)


def sc(**kw):
    base = dict(
        interrupted=False, input_transcription=None, interim_input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=False,
    )
    base.update(kw)
    return NS(server_content=NS(**base), tool_call=None, tool_call_cancellation=None, go_away=None)


def heard(text, finished=True):
    return sc(input_transcription=NS(text=text, finished=finished))


_TONE = (np.sin(np.arange(2400) * 2 * np.pi * 220 / 24000) * 6000).astype(np.int16).tobytes()  # 0.1 s of voice
_LOUD = (np.sin(np.arange(320) * 2 * np.pi * 220 / 16000) * 6000).astype(np.int16).tobytes()  # 20 ms, a voice in the room
_QUIET = b"\x00\x00" * 320


def audio(n=1):
    return sc(model_turn=NS(parts=[NS(inline_data=NS(data=_TONE)) for _ in range(n)]))


DONE = sc(turn_complete=True)


@pytest.fixture
def voice(monkeypatch):
    """A LiveVoice whose sessions open without the network (a FakeLive + the real silence watch)."""
    from neo2.core import config, events
    from neo2.core.tools import load_all
    from neo2.core.voice import live as live_mod

    load_all()
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("NEO_LISTEN_TIMEOUT_S", str(TIMEOUT_S))
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())

    class Session:
        memory_context = ""
        voice_owned = False

        async def command(self, text, steps):
            from neo2.core.agent.session import Reply

            return Reply("did it", "quick", silent=True)

    typed: list[str] = []

    async def on_text(text):
        typed.append(text)

    v = live_mod.LiveVoice(on_text, Session())
    v.spk = FakeSpk()
    v.opened, v.typed = 0, typed

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
    yield v
    for t in v._tasks:
        t.cancel()


async def _closed_within(v, s: float) -> float | None:
    """Seconds until the session closed, or None if it was still open after `s`."""
    t0 = time.time()
    while time.time() - t0 < s:
        if v._live is None:
            return time.time() - t0
        await asyncio.sleep(0.05)
    return None


async def _mic_frames(v, frames, hits=()):
    """Drive _mic_loop with these frames (and the spotter's verdict on each)."""
    frames, hits = list(frames) + [b""], iter(hits)

    class Mic:
        async def read(self):
            return frames.pop(0)

    class Wake:
        def feed(self, frame):
            return next(hits, None)

    v.mic, v._wake = Mic(), Wake()
    await v._mic_loop()


# ---- Live --------------------------------------------------------------------------------------


def test_typed_words_never_turn_the_mic_on(voice):
    """Typed text gets Live's spoken answer, but the session it opens is deaf: nothing from the
    room goes up until the wake word."""
    from neo2.core.events import bus

    async def run():
        await voice.send_text("what's the capital of France?")
        assert voice._live is not None and voice.opened == 1
        assert not voice._listening and voice._words_at > 0
        await voice._refresh_state()
        assert bus().state != NeoState.LISTENING  # the orb doesn't claim to listen
        live = voice._live
        await _mic_frames(voice, [_LOUD] * 5)  # a voice in the room, no "Neo"
        assert live.audio == [] and voice._last_user_speech <= voice._words_at
        # "Neo" while that session is open: now it listens, and the room goes up
        await _mic_frames(voice, [_LOUD] * 3, hits=["wake"])
        assert voice._listening and len(live.audio) == 2 and voice._live is live

    asyncio.run(run())
    assert voice.typed == []  # Live answered, as before; no second brain was asked


def test_a_wake_with_nothing_said_goes_idle_after_listen_timeout(voice, capsys):
    async def run():
        await voice.wake("voice")
        assert voice._live is not None and voice._listening
        took = await _closed_within(voice, TIMEOUT_S + 1.0)
        assert took is not None and took >= TIMEOUT_S
        assert not voice._listening  # the next "Neo" starts fresh

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "[voice] woke (voice)" in out and "nothing said for" in out and "idle" in out


def test_a_voice_in_the_room_alone_doesnt_keep_it_open(voice):
    """The TV, someone else talking: the local detector hears a voice the whole time, but the
    server never transcribes words. The session closes anyway — a short grace for the server
    to transcribe, never an extension."""
    from neo2.core.voice import live as live_mod

    async def run():
        await voice.wake("voice")
        t0 = time.time()
        while voice._live is not None and time.time() - t0 < 3.0:
            voice._hear(_LOUD)  # (energy gate: silero isn't loaded in tests)
            await asyncio.sleep(0.05)
        return time.time() - t0

    took = asyncio.run(run())
    assert voice._live is None
    assert TIMEOUT_S <= took <= TIMEOUT_S + live_mod._TRANSCRIBE_GRACE_S + 0.4
    assert voice._last_voice_at > 0 and voice._last_user_speech <= voice._session_open  # heard, not "said"


def test_words_transcribed_keep_it_open_and_quiet_after_the_answer_closes_it(voice, capsys):
    from neo2.core.voice import live as live_mod

    async def run():
        await voice.wake("ui")
        ub, mb = [], []
        await asyncio.sleep(0.1)
        await voice._on_message(heard("what time is it", finished=False), ub, mb, voice._live)
        assert voice._words_at > 0 and voice._turn_open
        assert await _closed_within(voice, TIMEOUT_S + 0.6) is None  # the model is answering: held open
        # mid-sentence, the local detector counts as the user (a pause isn't silence)
        before = voice._last_user_speech
        await asyncio.sleep(0.05)
        voice._hear(_LOUD)
        assert voice._last_user_speech > before
        await voice._on_message(DONE, ub, mb, voice._live)  # answered
        assert not voice._turn_open
        # The follow-up window, then idle. (That voice a moment ago buys the server a grace to
        # transcribe it — never more.)
        took = await _closed_within(voice, TIMEOUT_S + live_mod._TRANSCRIBE_GRACE_S + 0.6)
        assert took is not None and TIMEOUT_S - 0.05 <= took <= TIMEOUT_S + live_mod._TRANSCRIBE_GRACE_S + 0.3

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "quiet for" in out and "nothing said" not in out


def test_the_stall_hold_needs_words_first(voice):
    """A turn the model opened on its own (nothing transcribed yet) doesn't hold the session."""

    async def run():
        await voice.wake("voice")
        voice._turn_open = True  # as a tool result would set it — but no words this session
        voice._last_activity = time.time()
        assert await _closed_within(voice, TIMEOUT_S + 1.0) is not None

    asyncio.run(run())


def test_a_dropped_socket_reconnects_only_mid_turn(voice, capsys):
    async def run():
        await voice.wake("voice")
        await voice._lost(voice._live, "receive ended")
        assert voice._live is None and voice.opened == 1  # idle: nothing was going on
        await voice.wake("voice")
        await voice._on_message(heard("what time is it"), [], [], voice._live)
        await voice._lost(voice._live, "send failed")
        assert voice._live is not None and voice.opened == 3 and voice._listening  # mid-turn: back
        await voice.send_text("and the date?")
        await voice._on_message(DONE, [], [], voice._live)
        await asyncio.sleep(0.05)
        await voice._lost(voice._live, "receive ended")
        assert voice._live is None and voice.opened == 3  # answered already: idle

    asyncio.run(run())
    out = capsys.readouterr().out
    assert out.count("reconnecting") == 1 and "receive ended; idle" in out


def test_the_mic_is_muted_from_the_first_voice_in_a_reply(voice):
    """While the gate still holds the model's reply, nothing has played — but the room must not
    reach the server, or its own activity detection cuts the reply off unheard. A command's
    narration (gate "mute") is the exception: the user may talk over that."""

    async def run():
        await voice.wake("voice")
        await voice._on_message(heard("thanks neo"), [], [], voice._live)
        assert not voice._mic_muted()
        await voice._model_audio(_TONE, [], [])  # the first chunk, held by the gate
        assert voice._gate == "hold" and voice.spk.played == [] and voice._turn_audio
        assert voice._mic_muted()
        voice._gate = "mute"  # it turned out to be a command: the user may speak again
        assert not voice._mic_muted()
        voice._gate = "hold"
        await voice.interrupt()  # a committed "stop" during the hold
        assert not voice._mic_muted() and voice._held == [] and voice._gate == "mute"
        await voice._on_message(DONE, [], [], voice._live)
        assert not voice._turn_audio

    asyncio.run(run())


def test_quiet_since_still_counts_typed_text_and_model_turns(voice):
    from neo2.core.voice import live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v.spk, v._session_open, v._last_user_speech, v._last_turn_end = NS(last_stop=0.0), 100.0, 0.0, 0.0
    assert v._quiet_since() == 100.0
    v._last_turn_end = 130.0
    assert v._quiet_since() == 130.0


# ---- a spoken routine said with pauses -----------------------------------------------------------


def test_live_ends_a_turn_after_700_ms_of_silence():
    """Every reply waits this long after the user stops: it stays short (a routine's pauses are the
    collector's to bridge, below)."""
    from neo2.core.voice import live as live_mod

    assert live_mod._LIVE_VAD_SILENCE_MS == 700


def test_a_routine_said_with_a_pause_is_saved_whole(voice, monkeypatch):
    """"When I say I want to study, open Notion" … pause … "Safari and youtube.com": the server ends
    the first turn at the pause; the collector keeps the definition open until the user has been quiet
    for a moment, claims the second sentence too (the model's calls for it are skipped, nothing of it
    runs by itself) and saves the whole thing once."""
    from neo2.core.agent.session import Reply
    from neo2.core.config import settings
    from neo2.core.voice import live as live_mod

    monkeypatch.setattr(live_mod, "_ROUTINE_SETUP_SETTLE_S", 0.4)
    settings().listen_timeout_s = 5.0  # (as it is by default: the session outlives the pause)
    ran = []

    async def command(text, steps):
        ran.append((text, steps))
        return Reply("", "quick", silent=True)

    monkeypatch.setattr(voice._agent_session, "command", command, raising=False)

    async def run():
        await voice.wake("ui")
        ub, mb = [], []
        await voice._on_message(heard("When I say I want to study, open Notion."), ub, mb, voice._live)
        await voice._on_message(DONE, ub, mb, voice._live)  # the server ends the turn at the pause
        await asyncio.sleep(0.2)
        await voice._on_message(heard("Safari and youtube.com."), ub, mb, voice._live)
        assert voice._lane_owns() and voice._lane_tried_utt == voice._utt_id  # claimed with the first
        await voice._on_message(DONE, ub, mb, voice._live)
        t0 = time.time()
        while not ran and time.time() - t0 < 3:
            await asyncio.sleep(0.05)

    asyncio.run(run())
    ((text, steps),) = ran
    ((tool, args),) = steps
    assert text == "When I say I want to study, open Notion, Safari and youtube.com"
    assert tool == "routine_manage" and args["trigger"] == "I want to study"
    targets = [a["target"] for a in args["actions"]]
    assert targets[:2] == ["Notion", "Safari"] and "youtube.com" in targets[2] and len(targets) == 3


def test_routine_parts_join_as_said():
    from neo2.core.voice.live import _join_parts

    assert _join_parts(["When I say I want to study, open Notion.", "Safari and YouTube."]) == (
        "When I say I want to study, open Notion, Safari and YouTube"
    )
    assert _join_parts(["When I say", "time to start working, open VS Code"]) == (
        "When I say time to start working, open VS Code"
    )
    assert _join_parts(["When I say focus, open Notion and", "Safari"]) == "When I say focus, open Notion and Safari"
    assert _join_parts(["", "  "]) == ""


@pytest.mark.parametrize(
    "local_trigger, replaced",
    [("Time to start working!", True), ("time to", False)],
)
def test_the_models_routine_is_replaced_only_by_a_parse_of_the_same_phrase(voice, monkeypatch, local_trigger, replaced):
    """The model's routine_manage call is replaced by the local parse of the whole definition only when
    both heard the same phrase: a parse that misread it never overrides the model, which heard it too."""
    from neo2.core.agent.registry import Registry, ToolOutput
    from neo2.core.tools import routines

    mine = {"action": "create", "trigger": local_trigger, "actions": [{"kind": "open_app", "target": "VS Code"}, {"kind": "open_app", "target": "Slack"}]}
    model = {"action": "create", "trigger": "time to start working", "actions": [{"kind": "open_app", "target": "VS Code"}]}
    monkeypatch.setattr(routines, "is_setup_request", lambda text: True)
    monkeypatch.setattr(routines, "parse_local_setup", lambda text: dict(mine))
    got = []

    async def invoke(self, name, args, ctx):
        got.append(args)
        return ToolOutput("Saved.")

    monkeypatch.setattr(Registry, "invoke", invoke)  # (the class: an instance attribute would outlive the test)
    said = "When I say time to start working, open VS Code and Slack"

    async def run():
        await voice._ensure_session()
        voice._last_heard_at = 0.0  # (quiet long since: the definition is complete)
        fc = NS(name="routine_manage", args=dict(model), id="r1")
        await voice._run_tool(voice._live, fc, said, utt=voice._utt_id, said=said)

    asyncio.run(run())
    assert got == [mine if replaced else model]


# ---- the local cascade -------------------------------------------------------------------------


def _local(monkeypatch):
    from neo2.core import config, events
    from neo2.core.voice import local as local_mod

    monkeypatch.setenv("NEO_LISTEN_TIMEOUT_S", "0.5")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    v = local_mod.LocalVoice.__new__(local_mod.LocalVoice)
    v._vad = v._wake = v._stt = v._handler = None
    v._ptt = v._capturing = v._speaking = False
    v._speaking_since = v._followup_until = 0.0
    v.spk = FakeSpk()
    return v, local_mod


def test_local_nothing_said_after_wake_goes_idle_after_listen_timeout(monkeypatch, capsys):
    v, local_mod = _local(monkeypatch)
    clock = [1000.0]
    reads = []

    class Mic:
        async def read(self):
            clock[0] += 0.02  # one 20 ms frame of silence
            reads.append(1)
            return _QUIET

    v.mic = Mic()
    monkeypatch.setattr(local_mod, "time", NS(time=lambda: clock[0]))
    assert asyncio.run(v._capture_utterance()) == ""
    assert 20 <= len(reads) <= 30  # 0.5 s of silence, not the 5 s that was hard-coded
    assert "nothing said for" in capsys.readouterr().out


def test_local_followup_uses_the_setting_and_a_stop_clears_it(monkeypatch, capsys):
    v, _ = _local(monkeypatch)
    v._arm_followup()
    assert 0.4 <= v._followup_until - time.time() <= 0.5
    assert "answered; listening" in capsys.readouterr().out
    asyncio.run(v.interrupt())
    assert v._followup_until == 0.0  # never arms again from a stale window
