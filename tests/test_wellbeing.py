"""Gentle check-ins (neo2.core.voice.wellbeing).

* One sound counts once: Apple's classifier judges 3 s windows every 1.5 s, so one cough is in two
  results; old hits are pruned on every result; a cue the policy refuses doesn't pile up.
* How often: the same kind waits 10 min → 1 h → 4 h, at most 3 a day, quiet for the rest of the day
  after a "no" or two unanswered check-ins.
* The monitor never needs the audio stack to stay off, and a failure inside Apple's callback is
  flagged, never raised into Apple's thread.

Fakes only: no microphone, no Apple classifier (except one optional check of the cached Objective-C
class), no network."""

from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from neo2.core.voice import wellbeing
from neo2.core.voice.wellbeing import CheckinPolicy, _CueState, declines

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
