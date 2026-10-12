"""The wake gate (neo2.core.voice.wakegate): after a spoken "Neo", does a voice follow?

Shadow by default: the audio goes up exactly as before and only the verdict is logged and counted.
On: the audio after a spoken "Neo" waits for a voice, then all of it goes up in order; a wake nobody
spoke after closes without anything of the room leaving the Mac. A wake by hand, a check-in and a
reconnect are never gated.

No real microphone, model or network: frames are made here, the spotter and the session are fakes
(the harness of tests/test_voice_wake.py), and the detector is the energy fallback (no silero)."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from neo2.core.events import EventBus
from tests.test_voice_wake import FakeLive, FakeSpk, _mic_frames, heard

_RATE = 16000
_F = 320  # one 20 ms frame
_SINE = (np.sin(np.arange(_F) * 2 * np.pi * 220 / _RATE) * 6000).astype(np.int16)


def quiet(i: int) -> bytes:
    """A silent frame that is still told apart from every other one (so order can be checked)."""
    return np.full(_F, i % 300, dtype=np.int16).tobytes()


def loud(i: int) -> bytes:
    """A frame the energy fallback hears as a voice, told apart from every other one."""
    return (_SINE + np.int16(i % 300)).astype(np.int16).tobytes()


HIT = quiet(299)  # the frame the spotter said "neo" on (today it never goes up)


def sent(live) -> list[bytes]:
    return [b.data for b in live.audio]


def _counts() -> dict[str, int]:
    from neo2.core.reflex import learn

    return {k: v for k, v in learn._counts.items() if k.startswith("wake_")}


def _gate_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("[voice] wake gate") or "(gate)" in ln]


_LINE = re.compile(
    r"^\[voice\] wake gate \((shadow|on)\): (pass|drop); words=(yes|no); speech_ms=\d+; first_voice_ms=(\d+|-)$"
)


# ---- the gate itself ---------------------------------------------------------------------------


def _gate(**kw):
    from neo2.core.voice.wakegate import WakeGate

    kw.setdefault("opened_at", 0.0)
    return WakeGate(kw.pop("mode", "on"), kw.pop("gate_s", 3.0), kw.pop("speech_ms", 300), **kw)


def _run(gate, frames: list[bool]) -> list[str]:
    """Feed 20 ms frames (True = voiced); the verdict after each."""
    return [gate.feed(_F if v else 0, _F) for v in frames]


@pytest.fixture
def clock(monkeypatch):
    """The gate's wall clock, set by the test (the audio is what moves in these tests)."""
    from neo2.core.voice import wakegate

    now = [1000.0]
    monkeypatch.setattr(wakegate, "_clock", lambda: now[0])
    return now


def test_silence_is_rejected_at_the_deadline_and_not_a_frame_before(clock):
    g = _gate()
    out = _run(g, [False] * 160)
    assert out[148] == "pending" and out[149] == "reject"  # the 150th frame ends 3.0 s of audio
    assert set(out[150:]) == {"reject"} and g.settled and not g.passed
    assert g.voiced == 0 and g.first_voice == -1


def test_speech_at_1_2_s_confirms_once_300_ms_of_it_are_heard(clock):
    g = _gate()
    out = _run(g, [False] * 60 + [True] * 40 + [False] * 60)  # a voice from 1.2 s to 2.0 s
    assert out[73] == "pending" and out[74] == "confirm"  # 15 voiced frames = 300 ms, at 1.5 s
    assert set(out[74:]) == {"confirm"} and g.passed
    assert g.first_voice * 1000 // _RATE == 1200
    assert g.voiced * 1000 // _RATE == 800  # still counted after confirming, until the deadline


def test_the_tail_of_neo_alone_never_confirms(clock):
    """The end of "Neo" itself is voiced right after the hit: 250 ms of it count for nothing."""
    g = _gate()
    out = _run(g, [True] * 12 + [False] * 150)  # 240 ms of voice, then silence
    assert out[-1] == "reject" and g.voiced == 0 and g.first_voice == -1
    # A voice that runs across the end of the tail counts from the tail on only.
    g = _gate()
    _run(g, [True] * 13)  # 0-260 ms
    assert g.voiced == 160 and g.first_voice * 1000 // _RATE == 250


def test_a_slow_connect_extends_the_deadline_to_gate_s_after_the_session_opened(clock):
    """Frames waited in the microphone's queue while Live connected, so 3 s of audio can be
    through before the user even saw the rim: the deadline is also 3 s of wall clock after it."""
    g = _gate(opened_at=999.0)  # the session opened 1 s ago (wall clock)
    out = _run(g, [False] * 200)  # 4 s of queued silence, drained at once
    assert set(out) == {"pending"}
    clock[0] = 1002.0  # 3 s after the session opened
    assert g.feed(0, _F) == "reject"
    # …and a voice inside that extension confirms it.
    clock[0] = 1000.0
    g = _gate(opened_at=999.0)
    _run(g, [False] * 200)
    assert _run(g, [True] * 15)[-1] == "confirm" and g.first_voice * 1000 // _RATE == 4000


def test_counting_stops_at_the_deadline(clock):
    g = _gate()
    _run(g, [False] * 150 + [True] * 50)  # a voice only after 3 s
    assert g.verdict == "reject" and g.voiced == 0


def test_the_thresholds_come_from_the_settings(clock):
    g = _gate(speech_ms=100, gate_s=1.0)
    assert _run(g, [False] * 13 + [True] * 5)[-1] == "confirm"
    g = _gate(speech_ms=100, gate_s=1.0)
    assert _run(g, [False] * 50)[-1] == "reject"


def test_a_wake_by_hand_or_a_reconnect_settles_it(clock):
    g = _gate()
    _run(g, [False] * 20)
    g.confirm()
    assert g.verdict == "confirm" and g.forced and g.passed
    assert _run(g, [False] * 200)[-1] == "confirm"  # the deadline doesn't take it back
    g = _gate()
    _run(g, [False] * 150)
    g.confirm()
    assert g.verdict == "reject" and not g.forced  # (a settled verdict stays)


def test_a_gate_that_never_settled_is_a_drop_and_its_line_is_numbers_only(clock):
    g = _gate(mode="shadow")
    _run(g, [False] * 10)
    assert g.pending() and not g.passed
    line = g.line(words=False)
    assert line == "[voice] wake gate (shadow): drop; words=no; speech_ms=0; first_voice_ms=-"
    assert g.counters(words=False) == ["wake_gate_drop"]
    assert g.counters(words=True) == ["wake_gate_drop", "wake_gate_drop_words"]
    _run(g, [True] * 20)
    assert (
        g.line(words=True) == "[voice] wake gate (shadow): pass; words=yes; speech_ms=350; first_voice_ms=250"
    )
    assert _LINE.match(g.line(words=True))
    assert g.counters(words=True) == ["wake_gate_pass"]
    assert g.counters(words=False) == ["wake_gate_pass", "wake_gate_pass_nowords"]


def test_the_counter_names_are_the_quality_reports():
    from neo2.core.reflex import learn
    from neo2.core.voice import wakegate

    src = Path(wakegate.__file__).read_text(encoding="utf-8")
    named = set(re.findall(r'"(wake_gate_[a-z_]+)"', src))
    assert named == set(learn._WAKE_GATE)


def test_a_mode_it_doesnt_know_is_refused():
    from neo2.core.voice.wakegate import WakeGate

    with pytest.raises(ValueError):
        WakeGate("maybe")


# ---- the settings ------------------------------------------------------------------------------


def _fresh(monkeypatch, **env):
    from neo2.core import config

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    return config.settings()


def test_the_gate_is_shadow_by_default(monkeypatch):
    s = _fresh(monkeypatch)
    assert (s.wake_gate, s.wake_gate_s, s.wake_gate_speech_ms) == ("shadow", 3.0, 300)


@pytest.mark.parametrize(
    "raw, mode",
    [
        ("on", "on"),
        (" OFF ", "off"),
        ("Shadow", "shadow"),
        ("onn", "shadow"),
        ("true", "shadow"),
        ("", "shadow"),
    ],
)
def test_a_typo_never_switches_the_gate_on(monkeypatch, raw, mode):
    assert _fresh(monkeypatch, NEO_WAKE_GATE=raw).wake_gate == mode


@pytest.mark.parametrize(
    "gate_s, speech_ms, want",
    [("2.5", "450", (2.5, 450)), ("0", "5", (1.0, 100)), ("99", "99999", (10.0, 2000)), ("abc", "x", (3.0, 300)),
     ("nan", "nan", (3.0, 300))],
)  # fmt: skip
def test_the_numbers_stay_within_reason(monkeypatch, gate_s, speech_ms, want):
    s = _fresh(monkeypatch, NEO_WAKE_GATE_S=gate_s, NEO_WAKE_GATE_SPEECH_MS=speech_ms)
    assert (s.wake_gate_s, s.wake_gate_speech_ms) == want


# ---- the voice: what goes up ------------------------------------------------------------------


@pytest.fixture
def make(monkeypatch):
    """A LiveVoice with the gate in `mode`, sessions that open without the network (a FakeLive each,
    kept in `v.lives`, with the real silence watch) and the energy detector. `clock_ahead`: the
    gate's wall-clock half of the deadline is long past, so the audio alone decides."""
    from neo2.core import config, events
    from neo2.core.tools import load_all
    from neo2.core.voice import live as live_mod
    from neo2.core.voice import wakegate

    load_all()
    built = []

    def build(mode: str = "shadow", *, listen: float = 5.0, clock_ahead: bool = True):
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("NEO_LISTEN_TIMEOUT_S", str(listen))
        monkeypatch.setenv("NEO_WAKE_GATE", mode)
        monkeypatch.setattr(config, "_settings", None)
        monkeypatch.setattr(events, "_bus", EventBus())
        if clock_ahead:
            monkeypatch.setattr(wakegate, "_clock", lambda: time.time() + 3600)

        class Session:
            memory_context = ""
            voice_owned = False

            async def command(self, text, steps):
                from neo2.core.agent.session import Reply

                return Reply("did it", "quick", silent=True)

        async def on_text(text):
            pass

        v = live_mod.LiveVoice(on_text, Session())
        v.spk = FakeSpk()
        v.lives = []

        async def ensure():
            async with v._connect_lock:
                if v._live:
                    return
                v._closing = False
                v._live, v._live_cm = FakeLive(), None
                v.lives.append(v._live)
                v._last_activity = v._session_open = v._last_user_speech = time.time()
                v._words_at = 0.0
                v._turn_open, v._utterance, v._turn_audio = False, "", False
                v._tasks.append(asyncio.create_task(v._silence_watch(v._live)))

        monkeypatch.setattr(v, "_ensure_session", ensure)
        built.append(v)
        return v

    yield build
    for v in built:
        for t in v._tasks:
            t.cancel()


async def _wake_and(v, frames: list[bytes]) -> None:
    """A spoken "Neo" on HIT, then these frames."""
    await _mic_frames(v, [HIT, *frames], hits=["wake"])


_SPEECH = (
    [quiet(i) for i in range(10)] + [loud(i) for i in range(10, 40)] + [quiet(i) for i in range(40, 200)]
)
_SILENCE = [quiet(i) for i in range(200)]  # 4 s


@pytest.mark.parametrize("frames", [_SPEECH, _SILENCE], ids=["speech", "silence"])
def test_off_and_shadow_send_exactly_what_today_sends(make, frames):
    """Today every frame after the hit goes up, in order (the hit itself never does). Shadow only
    watches: the same bytes, the same session."""
    got = {}
    for mode in ("off", "shadow"):
        v = make(mode)
        asyncio.run(_wake_and(v, frames))
        assert len(v.lives) == 1 and v._live is v.lives[0] and v._listening
        got[mode] = sent(v.lives[0])
    assert got["off"] == got["shadow"] == frames


def test_on_with_speech_sends_every_frame_since_the_hit_in_order_then_streams(make, capsys):
    v = make("on")

    async def run():
        await _wake_and(v, _SPEECH[:27])  # up to 540 ms: 290 ms of voice past the tail — not yet
        live = v.lives[0]
        assert sent(live) == [] and v._wake_gate.pending()
        await _mic_frames(v, _SPEECH[27:28])  # the 15th voiced frame confirms it
        assert sent(live) == _SPEECH[:28]  # all of it, in order
        await _mic_frames(v, _SPEECH[28:])
        assert sent(live) == _SPEECH and v._live is live  # …then streaming as today
        await v._close_session()

    asyncio.run(run())
    (line,) = _gate_lines(capsys.readouterr().out)
    assert line == "[voice] wake gate (on): pass; words=no; speech_ms=550; first_voice_ms=250"
    assert _counts() == {"wake_voice": 1, "wake_gate_pass": 1, "wake_gate_pass_nowords": 1}


def test_on_with_silence_sends_nothing_and_closes_the_session(make, capsys):
    v = make("on")

    async def run():
        await _wake_and(v, _SILENCE[:149])
        live = v.lives[0]
        assert v._live is live and sent(live) == []
        await _mic_frames(v, _SILENCE[149:])  # 3 s of audio after the hit: rejected
        assert v._live is None and not v._listening and sent(live) == [] and len(v.lives) == 1
        assert v._wake_gate is None and not v._gate_held

    asyncio.run(run())
    out = capsys.readouterr().out
    assert _gate_lines(out) == [
        "[voice] wake without speech; dropped (gate)",
        "[voice] wake gate (on): drop; words=no; speech_ms=0; first_voice_ms=-",
    ]
    assert _counts() == {"wake_voice": 1, "wake_gate_drop": 1}


def test_on_holds_queued_frames_until_the_session_has_been_open_for_gate_s(make, monkeypatch):
    """With the real clock: 4 s of audio that queued during a connect is through the gate at once,
    but the user only just saw the rim. Nothing goes up, and the session stays."""
    from neo2.core.voice import wakegate

    v = make("on", clock_ahead=False)

    async def run():
        await _wake_and(v, _SILENCE)
        assert v._live is v.lives[0] and sent(v.lives[0]) == [] and v._wake_gate.pending()
        monkeypatch.setattr(wakegate, "_clock", lambda: time.time() + 3600)
        await _mic_frames(v, [quiet(1)])
        assert v._live is None and sent(v.lives[0]) == []

    asyncio.run(run())


def test_on_a_reject_never_closes_a_session_answering_typed_words(make, capsys):
    """A "Neo" said while a typed question is being answered: nobody spoke after it, so the room stays
    on the Mac — and the session goes on, deaf again, as it was."""
    v = make("on")

    async def run():
        await v.send_text("what's the capital of France?")
        live = v._live
        assert not v._listening
        await _wake_and(v, _SILENCE[:160])
        assert v._live is live and not v._listening and sent(live) == []
        await _mic_frames(v, [loud(1)] * 20)  # the room goes on: none of it goes up now
        assert sent(live) == []
        await v._close_session()

    asyncio.run(run())
    assert _gate_lines(capsys.readouterr().out) == [
        "[voice] wake without speech; not listening (gate)",
        "[voice] wake gate (on): drop; words=no; speech_ms=0; first_voice_ms=-",
    ]


def test_on_after_a_reject_that_left_the_session_deaf_a_wake_by_hand_is_never_gated(make, capsys):
    v = make("on")

    async def run():
        await v.send_text("what's the capital of France?")
        live = v._live
        await _wake_and(v, _SILENCE[:160])
        assert not v._listening and v._wake_gate is None  # (its line is out: the gate is done)
        await v.wake("ui")
        await _mic_frames(v, _SILENCE[:5])
        assert v._live is live and v._listening and sent(live) == _SILENCE[:5]

    asyncio.run(run())
    lines = _gate_lines(capsys.readouterr().out)
    assert lines == [
        "[voice] wake without speech; not listening (gate)",
        "[voice] wake gate (on): drop; words=no; speech_ms=0; first_voice_ms=-",
    ]


@pytest.mark.parametrize("at", [3, 27], ids=["mid-way", "the-last-held-frame"])
def test_on_a_session_that_closes_while_the_held_audio_goes_up_gets_no_more(make, capsys, at):
    v = make("on")

    async def run():
        await _wake_and(v, _SPEECH[:27])
        live = v.lives[0]
        real = live.send_realtime_input

        async def closing(audio):
            await real(audio)
            if len(live.audio) == at:
                await v._close_session()  # the silence watch, a drop, NEO quitting…

        live.send_realtime_input = closing
        await _mic_frames(v, _SPEECH[27:30])
        assert sent(live) == _SPEECH[:at] and v._live is None

    asyncio.run(run())
    assert "send failed" not in capsys.readouterr().out


def test_shadow_measures_a_false_drop_when_words_come_after_all(make, capsys):
    """Shadow's whole point: the gate says "no voice" but the server transcribes words anyway."""
    v = make("shadow")

    async def run():
        await _wake_and(v, _SILENCE)  # a would-drop, still streamed as today
        assert sent(v.lives[0]) == _SILENCE and v._wake_gate.verdict == "reject"
        await v._on_message(heard("what time is it"), [], [], v._live)
        await v._close_session()

    asyncio.run(run())
    (line,) = _gate_lines(capsys.readouterr().out)
    assert line == "[voice] wake gate (shadow): drop; words=yes; speech_ms=0; first_voice_ms=-"
    assert _counts() == {"wake_voice": 1, "wake_gate_drop": 1, "wake_gate_drop_words": 1}


def test_shadow_counts_a_voice_with_no_words_after_it(make, capsys):
    """The TV case: a voice confirms the gate, the server transcribes nothing. Closed by the real
    silence watch, as NEO does after listen_timeout_s."""
    v = make("shadow", listen=0.3)

    async def run():
        await _wake_and(v, _SPEECH)
        t0 = time.time()
        while v._live is not None and time.time() - t0 < 3:
            await asyncio.sleep(0.05)
        assert v._live is None

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "nothing said for" in out
    (line,) = _gate_lines(out)
    assert line == "[voice] wake gate (shadow): pass; words=no; speech_ms=550; first_voice_ms=250"
    assert _counts() == {"wake_voice": 1, "wake_gate_pass": 1, "wake_gate_pass_nowords": 1}


def test_off_counts_the_wake_and_nothing_else(make, capsys):
    v = make("off")

    async def run():
        await _wake_and(v, _SILENCE)
        assert v._wake_gate is None
        await v._close_session()

    asyncio.run(run())
    assert _gate_lines(capsys.readouterr().out) == []
    assert _counts() == {"wake_voice": 1}


@pytest.mark.parametrize("source", ["ptt", "ui", "check-in"])
def test_a_wake_by_hand_or_a_checkin_is_never_gated(make, source, capsys):
    v = make("on")

    async def run():
        await v.wake(source)
        assert v._wake_gate is None and v._listening
        await _mic_frames(v, _SILENCE)
        assert sent(v.lives[0]) == _SILENCE
        await v._close_session()

    asyncio.run(run())
    assert _gate_lines(capsys.readouterr().out) == [] and _counts() == {}


def test_a_wake_by_hand_while_the_gate_waits_sends_what_it_held(make, capsys):
    v = make("on")

    async def run():
        await _wake_and(v, _SILENCE[:20])
        assert sent(v.lives[0]) == []
        await v.wake("ui")  # the user summoned NEO by hand: they are there
        await _mic_frames(v, _SILENCE[20:25])
        assert sent(v.lives[0]) == _SILENCE[:25] and v._live is v.lives[0]
        await v._close_session()

    asyncio.run(run())
    (line,) = _gate_lines(capsys.readouterr().out)
    assert line.startswith("[voice] wake gate (on): pass;")


def test_a_reconnect_mid_turn_is_never_gated_and_never_gets_the_old_sessions_audio(make, capsys):
    v = make("on")

    async def run():
        await _wake_and(v, _SILENCE[:20])
        old = v.lives[0]
        v._tool_tasks["t"] = asyncio.create_task(asyncio.sleep(30))  # a tool is running: mid-turn
        await v._lost(old, "receive ended")
        assert len(v.lives) == 2 and v._live is v.lives[1] and v._listening and v._wake_gate is None
        assert v._gate_held is None  # what the old session held back is gone
        await _mic_frames(v, _SILENCE[20:30])
        assert sent(old) == [] and sent(v.lives[1]) == _SILENCE[20:30]  # (held frames went nowhere)
        v._tool_tasks.pop("t").cancel()
        await v._close_session()

    asyncio.run(run())
    lines = _gate_lines(capsys.readouterr().out)
    assert lines == ["[voice] wake gate (on): pass; words=no; speech_ms=0; first_voice_ms=-"]


def test_on_a_session_closed_while_the_gate_waits_lets_go_of_what_it_held(make, capsys):
    """The silence watch, the idle watch, a dropped socket, NEO quitting: whatever closes the session,
    the frames held back go with it — never up, and not kept in RAM."""
    v = make("on")

    async def run():
        await _wake_and(v, _SILENCE[:20])
        assert len(v._gate_held) == 20
        await v._close_session()
        assert v._gate_held is None and v._wake_gate is None and sent(v.lives[0]) == []

    asyncio.run(run())
    (line,) = _gate_lines(capsys.readouterr().out)
    assert line == "[voice] wake gate (on): drop; words=no; speech_ms=0; first_voice_ms=-"


def test_on_a_send_that_fails_while_flushing_stops_there(make, capsys):
    v = make("on")

    async def run():
        await _wake_and(v, _SPEECH[:27])
        live = v.lives[0]
        real = live.send_realtime_input

        failed = []

        async def flaky(audio):
            if len(live.audio) == 5 and not failed:  # the socket fails once, on the 6th frame
                failed.append(1)
                raise ConnectionError("gone")
            await real(audio)

        live.send_realtime_input = flaky
        await _mic_frames(v, _SPEECH[27:40])
        assert sent(live) == _SPEECH[:5] and v._live is None and not v._gate_held

    asyncio.run(run())
    assert "send failed: gone; idle" in capsys.readouterr().out


def test_every_gate_line_is_numbers_and_fixed_words(make, capsys):
    for mode, frames in (("shadow", _SPEECH), ("shadow", _SILENCE), ("on", _SPEECH), ("on", _SILENCE)):
        v = make(mode)

        async def run(v=v, frames=frames):
            await _wake_and(v, frames)
            if v._live is not None:
                await v._close_session()

        asyncio.run(run())
    lines = [ln for ln in _gate_lines(capsys.readouterr().out) if ln.startswith("[voice] wake gate")]
    assert len(lines) == 4 and all(_LINE.match(ln) for ln in lines)


# ---- the detector's measure ----------------------------------------------------------------------


def test_hear_says_how_much_of_a_frame_was_a_voice(make):
    """The energy fallback: the whole frame or nothing. Silero: whole 32 ms windows."""
    v = make("shadow")
    assert v._vad is None
    assert v._hear(loud(0)) == _F and v._hear(quiet(0)) == 0

    class Silero:
        def __init__(self, p):
            self.p = p

        def __call__(self, chunk, rate):
            return NS(item=lambda: self.p)

    pytest.importorskip("torch")
    v._vad = Silero(0.9)
    got = [v._hear(quiet(i)) for i in range(8)]  # 8 × 320 = 2560 samples = 5 windows of 512
    assert sum(got) == 5 * 512 and set(got) <= {0, 512}
    v._vad = Silero(0.1)
    assert sum(v._hear(quiet(i)) for i in range(8)) == 0
