"""Gentle check-ins (neo2.core.voice.wellbeing, neo2.core.app, the voices' check-in turn).

* One sound counts once: Apple's classifier judges 3 s windows every 1.5 s, so one cough is in two
  results; old hits are pruned on every result; a cue the policy refuses doesn't pile up.
* How often: the same kind waits 10 min → 1 h → 4 h, at most 3 a day, quiet for the rest of the day
  after a "no" or two unanswered check-ins.
* When: only while NEO is idle (no session, no follow-up window, not just busy) and no other app plays.
* What is kept: the check-in line and the reply to it never reach the journal, and the reply is an
  answer, never a command (no fast-path lane, no early actor, no camera or song); the listen has a hard
  deadline and isn't tried when Live can't be reached or the microphone is closed.
* The monitor never needs the audio stack to stay off, and a failure inside Apple's callback is
  flagged, never raised into Apple's thread.

Fakes only: no microphone, no Apple classifier (except one optional check of the cached Objective-C
class), no network."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from types import SimpleNamespace as NS

import pytest

from neo2.core.events import EventBus, NeoState
from neo2.core.voice import wellbeing
from neo2.core.voice.wellbeing import CheckinPolicy, _CueState, declines
from tests.test_voice_wake import _TONE, DONE, heard, sc, voice  # noqa: F401 — (voice: a fixture)
from tests.test_wake_word_setting import devices  # noqa: F401 — (a fixture: fake audio streams)

HOP = 1.5  # a new SoundAnalysis result every 1.5 s, each about the last 3 s


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.n = 0  # results so far: where in the analyzed stream the next window starts

    def __call__(self) -> float:
        return self.t


def _cues(**kw):
    clock = _Clock()
    out: list[str] = []
    state = _CueState(out.append, lambda: False, clock=clock, **kw)
    return state, clock, out


def _feed(state, clock, scores_at: dict[int, dict], n: int, *, windows: bool = True) -> None:
    """`n` more results at the real spacing, each about the 3 s of the stream before it (the k-th
    result of the whole stream covers [k*1.5, k*1.5+3] s); `scores_at` by i, from 0 for this call."""
    for i in range(n):
        k = clock.n
        state.update(scores_at.get(i, {}), (k * HOP, k * HOP + 3.0) if windows else None)
        clock.t += HOP
        clock.n += 1


# ---- one sound counts once ----------------------------------------------------------------------


def test_one_cough_in_two_overlapping_windows_is_one_cough():
    state, clock, out = _cues()
    # One cough at ~5 s of the stream: Apple sees it in [3, 6] and in [4.5, 7.5].
    _feed(state, clock, {2: {"cough": 0.9}, 3: {"cough": 0.85}}, 6)
    assert out == [] and len(state._hits["cough"]) == 1


def test_two_coughs_a_few_seconds_apart_ask():
    state, clock, out = _cues()
    # Coughs at ~5 s and ~11 s: windows [3, 6] + [4.5, 7.5], then [9, 12] + [10.5, 13.5].
    _feed(state, clock, {2: {"cough": 0.9}, 3: {"cough": 0.9}, 6: {"cough": 0.9}, 7: {"cough": 0.8}}, 9)
    assert out == ["cough"]


def test_a_nose_blow_and_its_echo_window_is_not_sniffling():
    state, clock, out = _cues()
    _feed(state, clock, {4: {"nose_blowing": 0.7}, 5: {"nose_blowing": 0.6}}, 8)
    assert out == []
    _feed(state, clock, {2: {"sniffling": 0.7}}, 3)  # a second, separate one: now it asks
    assert out == ["sniffling"]


def test_without_window_times_one_result_spacing_is_one_sound():
    state, clock, out = _cues()
    _feed(state, clock, {0: {"cough": 0.9}, 1: {"cough": 0.9}}, 2, windows=False)
    assert out == []
    _feed(state, clock, {1: {"cough": 0.9}}, 2, windows=False)  # 3 s after the first: a second cough
    assert out == ["cough"]


def test_old_hits_are_pruned_on_every_result():
    state, clock, out = _cues()
    _feed(state, clock, {0: {"cough": 0.9}}, 1)
    _feed(state, clock, {}, 60)  # 90 s of nothing (the cough span is 75 s)
    assert len(state._hits["cough"]) == 0
    _feed(state, clock, {0: {"cough": 0.9}}, 1)
    assert out == []


def test_a_refused_cue_does_not_pile_up_for_after_the_cooldown():
    allowed = [False]
    state, clock, out = _cues(may=lambda kind: allowed[0])
    _feed(state, clock, {0: {"cough": 0.9}, 4: {"cough": 0.9}}, 6)
    assert out == [] and len(state._hits["cough"]) == 0  # refused: its hits went with it
    allowed[0] = True
    _feed(state, clock, {0: {"cough": 0.9}}, 3)  # one fresh cough after the cooldown: not enough
    assert out == []
    _feed(state, clock, {0: {"cough": 0.9}}, 3)
    assert out == ["cough"]


def test_crying_asks_once_per_episode_after_about_five_seconds():
    state, clock, out = _cues()
    _feed(state, clock, {i: {"crying_sobbing": 0.9} for i in range(10)}, 10)
    assert out == ["crying"]
    _feed(state, clock, {}, 3)  # it stops…
    _feed(state, clock, {i: {"crying_sobbing": 0.9} for i in range(5)}, 5)  # …and starts again later
    assert out == ["crying", "crying"]


def test_ignored_while_neo_speaks():
    clock = _Clock()
    out: list[str] = []
    speaking = [True]
    state = _CueState(out.append, lambda: speaking[0], clock=clock)
    _feed(state, clock, {0: {"cough": 0.9}, 2: {"cough": 0.9}, 4: {"cough": 0.9}}, 6)
    assert out == [] and not state._hits["cough"]


# ---- how often ----------------------------------------------------------------------------------


def _policy(day: list[str] | None = None):
    clock = _Clock(0.0)
    day = day or ["mon"]
    return CheckinPolicy(clock=clock, day_of=lambda t: day[0]), clock, day


def test_the_same_kind_backs_off_10_minutes_then_an_hour_then_four():
    p, clock, day = _policy()
    assert p.allows("cough")
    p.given("cough")
    clock.t = 9 * 60
    assert not p.allows("cough") and not p.allows("sneezing")  # nothing at all within 10 min
    clock.t = 10 * 60 + 1
    assert p.allows("cough") and p.allows("cough_or_throat")  # (one family)
    p.given("cough_or_throat")
    clock.t += 59 * 60
    assert not p.allows("throat_clearing") and p.allows("sneezing")  # an hour for coughs; colds are apart
    clock.t += 61
    assert p.allows("cough")
    p.given("cough")
    day[0] = "tue"  # (a new day: the daily count starts again, the 24 h back-off doesn't)
    clock.t += 3 * 3600 + 59 * 60
    assert not p.allows("cough")
    clock.t += 61
    assert p.allows("cough")


def test_at_most_three_a_day():
    p, clock, day = _policy()
    for kind in ("cough", "sneezing", "crying"):
        assert p.allows(kind)
        p.given(kind)
        clock.t += 11 * 60
    assert not p.allows("throat_clearing") and not p.allows("sniffling")
    day[0] = "tue"
    assert p.allows("sniffling")


def test_a_no_quiets_it_for_the_rest_of_the_day():
    p, clock, day = _policy()
    p.given("cough")
    p.replied("no thanks")
    clock.t += 5 * 3600
    assert not p.allows("crying")
    day[0] = "tue"
    assert p.allows("crying")


def test_two_unanswered_check_ins_quiet_it_for_the_day_one_does_not():
    p, clock, day = _policy()
    p.given("cough")
    p.replied("")
    clock.t += 2 * 3600
    assert p.allows("sneezing")
    p.given("sneezing")
    p.replied("   ")
    clock.t += 5 * 3600
    assert not p.allows("crying")
    day[0] = "tue"
    assert p.allows("crying")


def test_an_answer_keeps_it_going():
    p, clock, _ = _policy()
    p.given("cough")
    p.replied("yes please, some tips would help")
    clock.t += 2 * 3600
    assert p.allows("cough")


@pytest.mark.parametrize(
    "said",
    ["no", "No thanks.", "nope", "nah I'm good", "I'm fine", "i am okay", "not now", "leave me alone",
     "stop asking", "Neo, no", "nahi", "nahin yaar", "main theek hoon", "नहीं", "मैं ठीक हूँ", "don't ask"],
)
def test_declines(said):
    assert declines(said)


@pytest.mark.parametrize(
    "said",
    ["yes", "yeah, a few tips please", "not really, I've got a cold", "I don't feel great", "sure",
     "I'm not okay", "now that you ask, yes", "nothing much", "nobody asked but yes", ""],
)
def test_not_declines(said):
    assert not declines(said)  # ("I'm not okay": they are talking — NEO answers, and comes back later)


# ---- the monitor ----------------------------------------------------------------------------------


def _no_audio_stack(monkeypatch) -> None:
    """A Mac without the voice extra: `from neo2.core.voice import audio` fails."""
    import neo2.core.voice

    monkeypatch.setitem(sys.modules, "neo2.core.voice.audio", None)
    monkeypatch.delattr(neo2.core.voice, "audio", raising=False)


def test_switched_off_it_never_imports_the_audio_stack(monkeypatch):
    _no_audio_stack(monkeypatch)
    loop = asyncio.new_event_loop()
    try:
        m = wellbeing.WellbeingMonitor(loop=loop, on_event=lambda k: None, is_ignored=lambda: False)
        m.set_enabled(False)
        m.close()
    finally:
        loop.close()


def test_switched_on_without_the_audio_stack_it_says_so_and_stays_off(monkeypatch, capsys):
    _no_audio_stack(monkeypatch)
    errors: list[str] = []

    async def run():
        m = wellbeing.WellbeingMonitor(
            loop=asyncio.get_running_loop(), on_event=lambda k: None, is_ignored=lambda: False, on_error=errors.append
        )
        m.set_enabled(True)  # never raises into NEO's startup
        await asyncio.sleep(0.05)
        assert m._failed and not m._enabled.is_set() and m._worker is None

    asyncio.run(run())
    assert errors and errors[0].endswith("Error") and "[wellbeing] local sound classifier unavailable" in capsys.readouterr().out


class _Buffer:
    def __init__(self):
        self.data = bytearray(640)

    def int16ChannelData(self):  # noqa: N802
        data = self.data
        return [types.SimpleNamespace(as_buffer=lambda n: memoryview(data))]

    def setFrameLength_(self, n):  # noqa: N802
        pass


def test_a_failure_inside_apples_callback_switches_check_ins_off_cleanly(monkeypatch):
    """Apple calls the observer on its own thread inside analyzeAudioBuffer: an error there is only
    flagged (a Python error raised into it aborted the whole core); the worker then stops this one
    feature, untaps, and says so."""
    audio = pytest.importorskip("neo2.core.voice.audio")
    monkeypatch.setattr(audio, "_taps", ())
    monkeypatch.setattr(audio, "_listeners", ())
    observer = types.SimpleNamespace()

    class Analyzer:
        def analyzeAudioBuffer_atAudioFramePosition_(self, buffer, position):  # noqa: N802
            observer._neo_owner._callback_failed = True  # what the real observer does on an error

    monkeypatch.setattr(
        wellbeing.WellbeingMonitor, "_make_analyzer", staticmethod(lambda: (Analyzer(), object(), observer, _Buffer()))
    )
    errors: list[str] = []

    async def run():
        m = wellbeing.WellbeingMonitor(
            loop=asyncio.get_running_loop(), on_event=lambda k: None, is_ignored=lambda: False, on_error=errors.append
        )
        m.set_enabled(True)
        assert audio._listeners == (m._capture,) and audio._taps == () and not audio.tapped()  # listen-only
        m._capture(b"\x00\x00" * 320)
        t0 = time.time()
        while not errors and time.time() - t0 < 3:
            await asyncio.sleep(0.02)
        assert errors == ["RuntimeError"] and m._failed and audio._listeners == ()
        assert observer._neo_owner is None  # a late callback finds nobody
        m.close()

    asyncio.run(run())


def test_the_observer_class_is_made_once(monkeypatch):
    """Switching check-ins back on after a failure defines the Objective-C class again — which raised,
    and turned them off for good. Now there is one per process."""
    if sys.platform != "darwin":
        pytest.skip("macOS only")
    objc = pytest.importorskip("objc")
    framework = "/System/Library/Frameworks/SoundAnalysis.framework"  # (its protocol; no audio involved)
    objc.loadBundle("SoundAnalysis", {}, bundle_path=framework)
    first = wellbeing._observer_class(objc)
    assert first is wellbeing._observer_class(objc) and first.__name__ == "NEOWellbeingObserver"


# ---- the app: when -----------------------------------------------------------------------------


class _Voice:
    """A Live-like voice: says the line (or not), listens (or not), records what it was asked."""

    def __init__(self, *, spoke=True, reply="yes please"):
        self._live = None
        self._speaking = self._turn_open = self._ptt = self._mic_parked = False
        self.spk = NS(playing=threading.Event())
        self.spoke, self.reply = spoke, reply
        self.said: list[str] = []
        self.listened: list[dict] = []

    async def speak_checkin(self, text):
        self.said.append(text)
        return self.spoke

    async def listen_after_checkin(self, **kw):
        self.listened.append(kw)
        return self.reply


@pytest.fixture
def app(monkeypatch):
    from neo2.core import app as app_mod
    from neo2.core import config, events

    monkeypatch.setenv("NEO_WELLBEING_CHECKINS", "true")
    monkeypatch.setenv("NEO_WAKE_WORD", "true")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(app_mod, "_other_media_playing", lambda: False)
    local: list[str] = []
    monkeypatch.setattr(app_mod, "_say_aloud_wait", local.append)
    a = app_mod.App.__new__(app_mod.App)
    a._tasks, a._wellbeing_speaking_until, a._wellbeing_reply_context = set(), 0.0, None
    a._last_active = time.monotonic() - 3600
    a.voice = _Voice()
    a.wellbeing = NS(policy=CheckinPolicy())
    a.local_said = local
    return a


def _checkin(app, kind="cough"):
    asyncio.run(app._wellbeing_checkin(kind))


def test_idle_it_asks_listens_and_counts_the_answer(app):
    _checkin(app)
    assert app.voice.said == ["A bit of a cough? You okay? Want a few simple tips?"]
    assert app.voice.listened and app.voice.listened[0]["seconds"] == 5.0
    assert not app.wellbeing.policy.allows("cough")  # (given: the back-off runs)


@pytest.mark.parametrize(
    "busy",
    ["listening", "session", "followup", "just_talked", "media", "task", "turn", "policy", "off"],
)
def test_never_unless_neo_is_idle_and_nothing_else_plays(app, monkeypatch, busy):
    from neo2.core import app as app_mod
    from neo2.core import config
    from neo2.core.events import bus

    if busy == "listening":
        asyncio.run(bus().set_state(NeoState.LISTENING))
    elif busy == "session":
        app.voice._live = object()  # a Live session is open: the conversation or its follow-up window
    elif busy == "followup":
        app.voice._followup_until = time.time() + 5  # the on-device voice's follow-up window
    elif busy == "just_talked":
        app._last_active = time.monotonic() - 5
    elif busy == "media":
        monkeypatch.setattr(app_mod, "_other_media_playing", lambda: True)  # a call, a video, music
    elif busy == "task":
        asyncio.run(bus().set_state(NeoState.WORKING, job="j1"))
    elif busy == "turn":
        app.voice._turn_open = True
    elif busy == "policy":
        app.wellbeing.policy.given("cough")
    elif busy == "off":
        config.settings().wellbeing_checkins = False
    _checkin(app)
    assert app.voice.said == [] and app.voice.listened == [] and app.local_said == []


def test_it_tracks_when_neo_was_last_busy(app):
    from neo2.core.events import Event

    app._last_active = 0.0
    app._track_activity(Event(type="state", data={"state": "idle"}))
    assert app._last_active == 0.0
    app._track_activity(Event(type="state", data={"state": "thinking"}))
    assert time.monotonic() - app._last_active < 1


def test_one_check_in_at_a_time(app):
    async def run():
        await asyncio.gather(app._wellbeing_checkin("cough"), app._wellbeing_checkin("sneezing"))

    asyncio.run(run())
    assert len(app.voice.said) == 1


@pytest.mark.parametrize("why", ["wake_word_off", "parked"])
def test_with_the_microphone_closed_it_says_the_line_but_never_listens(app, why):
    from neo2.core import config

    if why == "wake_word_off":
        config.settings().wake_word = False
    else:
        app.voice._mic_parked = True
    _checkin(app)
    assert app.voice.said and app.voice.listened == []


def test_offline_it_says_it_locally_and_doesnt_try_to_listen(app):
    app.voice.spoke = None  # Live couldn't be reached
    _checkin(app)
    assert app.local_said == ["A bit of a cough? You okay? Want a few simple tips?"] and app.voice.listened == []


def test_cut_short_it_says_it_locally_and_still_listens(app):
    app.voice.spoke = False  # Live was reached but didn't finish: a fresh session can hear the reply
    _checkin(app)
    assert app.local_said and app.voice.listened


@pytest.mark.parametrize("reply, quiet", [("no thanks", True), ("yes please", False)])
def test_a_no_quiets_it_for_the_day(app, reply, quiet):
    app.voice.reply = reply
    _checkin(app)
    app.wellbeing.policy._last = None  # (past the 10 min gap)
    app.wellbeing.policy._given.clear()
    assert app.wellbeing.policy.allows("crying") is not quiet


def test_two_unanswered_quiet_it_for_the_day(app):
    app.voice.reply = ""
    for kind in ("cough", "sneezing"):
        _checkin(app, kind)
        app.wellbeing.policy._last = None
    assert not app.wellbeing.policy.allows("crying")


# ---- Live: the check-in line and its reply ---------------------------------------------------------


def _journal(monkeypatch) -> list[tuple]:
    from neo2.core.memory import journal

    lines: list[tuple] = []
    monkeypatch.setattr(journal, "log_turn", lambda who, text, **kw: lines.append((who, text)))
    return lines


def _wake_word_on() -> None:
    from neo2.core.config import settings

    settings().wake_word = True  # (the microphone is open anyway: the only time NEO listens for a reply)


def test_the_check_in_line_is_said_and_shown_but_never_journaled(voice, monkeypatch):  # noqa: F811
    from neo2.core.events import bus

    lines = _journal(monkeypatch)
    shown: list[str] = []
    line = "A bit of a cough? You okay? Want a few simple tips?"

    async def run():
        bus().subscribe(lambda ev: shown.append(ev.data["text"]) if ev.type == "transcript" and ev.data.get("final") else None)
        await voice._ensure_session()
        task = asyncio.create_task(voice.speak_checkin(line))
        await asyncio.sleep(0.05)
        assert voice._checkin_turn
        ub, mb = [], []
        await voice._on_message(sc(output_transcription=NS(text=line), model_turn=NS(parts=[NS(inline_data=NS(data=_TONE))])), ub, mb, voice._live)
        await voice._on_message(DONE, ub, mb, voice._live)
        assert await asyncio.wait_for(task, 3) is True
        assert not voice._checkin_turn and voice._dialog == []
        # (A turn of the user's own afterwards is journaled as ever.)
        await voice._on_message(heard("thanks neo"), ub, mb, voice._live)
        await voice._on_message(DONE, ub, mb, voice._live)

    asyncio.run(run())
    assert line in shown and lines == [("user", "thanks neo")]


def test_the_reply_is_an_answer_never_a_command_and_never_journaled(voice, monkeypatch):  # noqa: F811
    _wake_word_on()
    from neo2.core.agent.early import early

    lines = _journal(monkeypatch)
    fed: list[str] = []
    monkeypatch.setattr(type(early()), "feed", lambda self, text: fed.append(text))  # (the class: see test_voice_wake)

    async def run():
        listen = asyncio.create_task(voice.listen_after_checkin(seconds=2.0, context="NEO just offered tips."))
        await asyncio.sleep(0.05)
        assert voice._listening and voice._checkin_listening
        ub, mb = [], []
        await voice._on_message(heard("open safari"), ub, mb, voice._live)  # the TV, or the user: either way
        await asyncio.sleep(0.05)
        assert not any(k.startswith("lane:") for k in voice._tool_tasks)  # no command lane
        assert not voice._camera_asked("what am I holding", voice._utt_id)
        assert not voice._song_asked("what song is this", voice._utt_id)
        await voice._on_message(DONE, ub, mb, voice._live)
        return await asyncio.wait_for(listen, 3)

    assert asyncio.run(run()) == "open safari"
    assert fed == [] and lines == []


def test_the_reply_window_has_a_hard_deadline(voice, monkeypatch):  # noqa: F811
    """A TV or a call keeps "words" coming: the listen NEO opened itself still ends on time."""
    _wake_word_on()
    from neo2.core.voice import live as live_mod

    monkeypatch.setattr(live_mod, "_CHECKIN_REPLY_S", 0.4)

    async def run():
        listen = asyncio.create_task(voice.listen_after_checkin(seconds=0.3))
        await asyncio.sleep(0.05)
        t0 = time.time()
        while not listen.done() and time.time() - t0 < 4:
            await voice._on_message(sc(interim_input_transcription=NS(text="and now, the weather")), [], [], voice._live)
            await asyncio.sleep(0.1)
        return listen.done(), time.time() - t0, voice._live

    done, took, live = asyncio.run(run())
    assert done and live is None and took < 1.6


def test_no_listen_when_live_cant_be_reached_or_the_microphone_is_closed(voice, monkeypatch):  # noqa: F811
    _wake_word_on()
    async def nothing():
        return None

    async def run():
        voice._mic_parked = True
        parked = await voice.listen_after_checkin()
        voice._mic_parked = False
        monkeypatch.setattr(voice, "_ensure_session", nothing)  # offline
        offline = await voice.listen_after_checkin()
        return parked, offline

    assert asyncio.run(run()) == (None, None)
    assert voice.opened == 0 and not voice._listening


def test_a_summon_during_the_reply_window_is_the_users_own_again(voice):  # noqa: F811
    _wake_word_on()
    async def run():
        listen = asyncio.create_task(voice.listen_after_checkin(seconds=2.0))
        await asyncio.sleep(0.05)
        await voice.wake("ui")
        assert not voice._checkin_listening and voice._checkin_until == 0.0
        voice._closing = True
        await voice._close_session()
        return await asyncio.wait_for(listen, 3)

    assert asyncio.run(run()) == ""


# ---- the on-device voice ------------------------------------------------------------------------------


def test_local_the_reply_is_returned_and_never_run_from_the_transcript(devices, monkeypatch):  # noqa: F811
    """The on-device voice hands the reply to the session (which answers it with the check-in's
    context), never to the early actor that runs fast paths straight from a transcript."""
    pytest.importorskip("sounddevice")
    from neo2.core.agent.early import early
    from tests.test_wake_word_setting import _frames, _local, _settings, _until

    _settings(monkeypatch, True)
    fed: list[str] = []
    monkeypatch.setattr(type(early()), "feed", lambda self, text: fed.append(text))  # (the class: see test_voice_wake)

    async def run():
        v = await _local(monkeypatch, ["open safari"])
        try:
            listen = asyncio.create_task(v.listen_after_checkin(seconds=1.0))
            await asyncio.sleep(0.02)
            await _frames(v, 1)
            reply = await asyncio.wait_for(listen, 3)
            assert await _until(lambda: v.handled == ["open safari"])
            return reply
        finally:
            await v.close()

    assert asyncio.run(run()) == "open safari" and fed == []


def test_local_with_the_wake_word_off_it_never_listens(devices, monkeypatch):  # noqa: F811
    pytest.importorskip("sounddevice")
    from tests.test_wake_word_setting import _inputs, _local, _settings

    _settings(monkeypatch, False)

    async def run():
        v = await _local(monkeypatch, ["yes"])
        try:
            return await v.listen_after_checkin(), _inputs()
        finally:
            await v.close()

    assert asyncio.run(run()) == (None, [])
