"""The microphone heals itself or says so (neo2.core.voice.audio, NEO_MIC_ESCALATE).

The device watcher reopens a microphone that stopped delivering, and until now a reopen counted as
working when it merely opened: core.log 6002-6038 has 37 "reopened" lines in a row after a restart,
every one of them silent, about 18 minutes of a NEO that couldn't hear and never said so. Now sound
has to come in after a reopen (5 frames within 2 s, checked outside the reopen lock). When it
doesn't, the watcher tries again after 5, 10, 20, 40, then every 60 s; on the 3rd silent try in a row
it tells the user once (NEO's page through health, one macOS notification), and when frames come
back it says so and takes that back. A microphone closed on purpose (the wake word off, NEO idle) is
never checked. NEO_MIC_ESCALATE=false is the watcher as it was: a test runs today's loop beside it.

No real device anywhere: sounddevice's streams are stand-ins, the clock is fake, and the "device"
delivers frames only while a test says it works.
"""

from __future__ import annotations

import re
import threading
import weakref

import pytest

sd = pytest.importorskip("sounddevice")
audio = pytest.importorskip("neo2.core.voice.audio")

_FRAME = b"\x01\x00" * 320  # 20 ms


class _World:
    """The Mac's audio, faked: the default devices, whether a microphone stream opens and then
    delivers frames, and a clock whose sleeps deliver them (a frame every 20 ms, while it works)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.mode = "silent"  # "works": open input streams deliver; "silent": they don't; "fail": no open
        self.dev = (1, 2)  # (default input, default output)
        self.streams: list[_Stream] = []
        self.reopens: list[tuple[float, str]] = []  # every reopen_all: (when, why)
        self.on_sleep = None  # a test's hook, called on every sleep (after the frames)

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s
        if self.mode == "works":
            for st in self.streams:
                if st.cb is not None and st.started and not st.closed:
                    for _ in range(min(10, max(1, round(s / 0.02)))):
                        st.cb(_FRAME, 320, None, None)
        if self.on_sleep is not None:
            self.on_sleep()

    def inputs(self) -> list[_Stream]:
        """The input streams opened so far (closed ones too)."""
        return [s for s in self.streams if s.cb is not None]


_world = _World()


class _Stream:
    """sounddevice's stream, faked: never touches a device; the world says whether it opens."""

    def __init__(self, *a, callback=None, **kw):
        if callback is not None and _world.mode == "fail":
            raise RuntimeError("Error opening RawInputStream: Internal PortAudio error [PaErrorCode -9986]")
        self.cb, self.closed, self.started = callback, False, False
        _world.streams.append(self)

    def start(self):
        self.started = True

    def abort(self):
        pass

    def close(self):
        self.closed = True

    def write(self, chunk):
        pass


@pytest.fixture
def world(monkeypatch):
    """A fresh fake world and a clean audio module: its own stream set, no watcher thread (a real
    one an earlier test started steps aside: it watches only while it is the registered one), the
    notification and NEO's page's problem list recorded instead of shown."""
    global _world
    from neo2.core import health
    from neo2.core.voice import devices

    _world = _World()
    monkeypatch.setattr(sd, "RawInputStream", _Stream)
    monkeypatch.setattr(sd, "OutputStream", _Stream)
    monkeypatch.setattr(sd, "_terminate", lambda: None)
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    monkeypatch.setattr(audio, "_streams", weakref.WeakSet())
    monkeypatch.setattr(audio, "_register", lambda o: audio._streams.add(o))
    monkeypatch.setattr(audio, "_watcher", None)
    monkeypatch.setattr(audio, "_taps", ())
    monkeypatch.setattr(audio, "_listeners", ())
    monkeypatch.setattr(audio, "_clock", _world.clock)
    monkeypatch.setattr(audio, "_sleep", _world.sleep)
    monkeypatch.setattr(devices, "default_input", lambda: _world.dev[0])
    monkeypatch.setattr(devices, "default_output", lambda: _world.dev[1])
    real = audio.reopen_all

    def spy(reason: str) -> bool:
        _world.reopens.append((_world.now, reason))
        return real(reason)

    monkeypatch.setattr(audio, "reopen_all", spy)
    notes: list[str] = []
    monkeypatch.setattr(audio, "_notify", notes.append)
    monkeypatch.setattr(health, "_live", {})
    _world.notes = notes
    yield _world
    for o in list(audio._streams):
        if isinstance(o, audio.Mic):
            o.stop()
        else:
            o.close()


def _escalate(monkeypatch, on: bool) -> None:
    from neo2.core.config import settings

    monkeypatch.setattr(settings(), "mic_escalate", on)


def _run(w, seconds: float) -> None:
    """The watcher thread, driven by hand: a poll a second."""
    for _ in range(int(seconds)):
        _world.sleep(1.0)
        w.step()


def _lines(capsys) -> list[str]:
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[audio]")]


def _silent_tries(lines: list[str]) -> list[str]:
    return [ln for ln in lines if "no sound came in" in ln or "didn't open" in ln]


def _counts() -> dict[str, int]:
    from neo2.core.reflex import learn

    return {k: v for k, v in learn._counts.items() if k.startswith("mic_")}


def _problem() -> str | None:
    from neo2.core import health

    return health._live.get("mic")


# ---- the microphone counts what it delivers --------------------------------------------------------


def test_a_microphone_counts_its_frames_since_it_was_last_opened(world):
    mic = audio.Mic()
    mic.start()
    assert mic.frames_since_open == 0
    for _ in range(7):
        mic._stream.cb(_FRAME, 320, None, None)
    assert mic.frames_since_open == 7 and mic.q.qsize() == 7
    audio.reopen_all("the input device changed")
    assert mic.frames_since_open == 0  # a new stream starts from nothing: its own frames prove it


# ---- silent reopens: sooner, then less often, and said once -----------------------------------------


def test_a_silent_reopen_is_tried_again_after_5_10_20_40_then_every_60_s(world, capsys):
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 400)
    tries = _silent_tries(_lines(capsys))
    nexts = [int(re.search(r"next in (\d+)s", ln).group(1)) for ln in tries]
    assert nexts[:7] == [5, 10, 20, 40, 60, 60, 60]
    assert [int(re.search(r"try (\d+);", ln).group(1)) for ln in tries] == list(range(1, len(tries) + 1))
    # each next try comes that long after the last one ended (its 2 s check), within a poll and the
    # 0.4 s settle the watcher always gives macOS
    times = [t for t, _ in world.reopens]
    for gap, wait in zip([b - a for a, b in zip(times, times[1:], strict=False)], nexts, strict=False):
        assert wait + 2.0 + 0.4 < gap <= wait + 2.0 + 0.4 + 1.0 + 0.1
    assert mic.active  # still wanted the whole time: never given up on


def test_the_user_is_told_once_on_the_third_silent_try_and_never_before(world, capsys):
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    lines: list[str] = []
    told_at = None
    for _ in range(600):
        _run(w, 1)
        lines += _lines(capsys)
        if world.notes and told_at is None:
            told_at = len(_silent_tries(lines))
    assert told_at == 3  # on the 3rd silent try, about half a minute after the stall
    assert world.notes == [audio.MIC_PROBLEM]  # one notification in ten minutes
    assert _problem() == audio.MIC_PROBLEM  # and NEO's page lists it
    c = _counts()
    assert c["mic_stall"] == 1 and c["mic_notice"] == 1 and c["mic_reopen_silent"] >= 10
    assert "mic_recovered" not in c


def test_the_notice_comes_about_half_a_minute_into_the_silence(world):
    mic = audio.Mic()
    mic.start()
    start = world.now
    w = audio._Watcher()
    while not world.notes:
        _run(w, 1)
        assert world.now - start < 60
    assert 20 <= world.now - start <= 40


def test_frames_coming_back_take_the_notice_back_and_count_a_recovery(world, capsys):
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    while not world.notes:
        _run(w, 1)
    capsys.readouterr()
    world.mode = "works"
    _run(w, 70)  # the next try (at most 60 s away) brings sound
    lines = _lines(capsys)
    assert "[audio] the microphone works again" in lines
    assert _problem() is None
    assert _counts()["mic_recovered"] == 1
    assert not _silent_tries(lines)
    n = len(world.reopens)
    _run(w, 120)
    assert len(world.reopens) == n  # healthy again: nothing more to do
    # a later episode is a new one: told again, once
    world.mode = "silent"  # the device goes quiet under the open stream
    _run(w, 200)
    assert world.notes == [audio.MIC_PROBLEM, audio.MIC_PROBLEM]
    assert _counts()["mic_stall"] == 2 and _counts()["mic_notice"] == 2


def test_frames_that_come_late_end_the_episode_without_another_reopen(world, capsys):
    """A Bluetooth microphone can take a few seconds to deliver after a switch: the 2 s check
    misses it, and the next poll sees it working, with no reopen in between."""
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 6)
    assert len(world.reopens) == 1 and len(_silent_tries(_lines(capsys))) == 1
    world.mode = "works"
    _run(w, 30)
    assert len(world.reopens) == 1
    assert "[audio] the microphone works again" in _lines(capsys)
    assert _counts() == {"mic_stall": 1, "mic_reopen_silent": 1, "mic_recovered": 1}
    assert world.notes == [] and _problem() is None


def test_a_reopen_that_brings_sound_is_todays(world, capsys):
    world.mode = "works"
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 5)
    assert world.reopens == []  # delivering: nothing to do
    world.dev = (7, 2)  # AirPods in
    before = world.now
    _run(w, 1)
    assert [r for _, r in world.reopens] == ["the input device changed"]
    assert world.now - before < 1.0 + 0.4 + 0.2  # the check is over as soon as frames arrive
    assert _lines(capsys) == ["[audio] the input device changed — reopened on the current devices"]
    assert _counts() == {} and world.notes == [] and _problem() is None
    _run(w, 120)
    assert len(world.reopens) == 1


def test_a_microphone_that_wont_open_is_a_silent_try_too(world, capsys):
    mic = audio.Mic()
    mic.start()
    world.mode = "fail"
    w = audio._Watcher()
    _run(w, 60)
    tries = _silent_tries(_lines(capsys))
    assert tries[0].startswith("[audio] the microphone didn't open (try 1; next in 5s;")
    assert world.notes == [audio.MIC_PROBLEM]
    world.mode = "works"
    _run(w, 70)
    assert _problem() is None and _counts()["mic_recovered"] == 1


def test_the_log_line_says_whether_a_song_or_a_check_in_listens(world, capsys):
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 6)
    (line,) = _silent_tries(_lines(capsys))
    assert line == (
        "[audio] reopened, but no sound came in (try 1; next in 5s; song tap: no; check-in listener: no)"
    )
    untap = audio.tap(lambda f: None)
    unlisten = audio.tap(lambda f: None, hold=False)
    _run(w, 10)
    (line,) = _silent_tries(_lines(capsys))
    assert line.endswith("(try 2; next in 10s; song tap: yes; check-in listener: yes)")
    untap()
    unlisten()


def test_the_check_never_holds_the_reopen_lock(world):
    mic = audio.Mic()
    mic.start()
    held: list[bool] = []
    world.on_sleep = lambda: held.append(audio._reopen_lock.locked())
    w = audio._Watcher()
    _run(w, 6)
    assert len(world.reopens) == 1 and len(held) > 20  # the 2 s check slept in small steps…
    assert not any(held)  # …and never inside the lock: the voice can stop or start meanwhile


def test_a_stall_is_counted_once_an_episode(world):
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 300)
    c = _counts()
    assert c["mic_stall"] == 1 and c["mic_reopen_silent"] == len(world.reopens) >= 7


# ---- a microphone closed on purpose ----------------------------------------------------------------


def test_a_stopped_microphone_is_never_checked_or_reopened(world, capsys):
    spk = audio.Speaker()
    spk.start()
    mic = audio.Mic()
    mic.start()
    mic.stop()  # the wake word off and NEO idle: closed on purpose
    opened = len(world.inputs())
    w = audio._Watcher()
    _run(w, 30)
    world.dev = (1, 9)  # the speaker follows the output device…
    _run(w, 200)
    assert [r for _, r in world.reopens] == ["the output device changed"]
    assert len(world.inputs()) == opened and mic._stream is None  # …and no microphone appears
    assert _counts() == {} and world.notes == [] and not _silent_tries(_lines(capsys))


def test_an_episode_waits_while_the_microphone_is_closed_on_purpose(world, capsys):
    spk = audio.Speaker()
    spk.start()
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 12)
    assert len(_silent_tries(_lines(capsys))) == 2
    mic.stop()  # NEO went idle with the wake word off
    n = len(world.reopens)
    _run(w, 300)
    assert len(world.reopens) == n and world.notes == []  # nothing to hear: no tries, nothing said
    world.dev = (1, 9)  # the speaker follows a new output device meanwhile…
    _run(w, 3)
    n += 1
    assert len(world.reopens) == n
    lines = _lines(capsys)
    assert "[audio] the microphone works again" not in lines  # …which proves nothing about the microphone
    assert not _silent_tries(lines) and "mic_recovered" not in _counts()
    world.mode = "works"
    mic.start()  # summoned: the fresh stream delivers
    _run(w, 3)
    assert "[audio] the microphone works again" in _lines(capsys)
    assert len(world.reopens) == n


# ---- off is today's watcher --------------------------------------------------------------------------


class _Today:
    """Today's watcher loop body (before NEO_MIC_ESCALATE), word for word but for the fake clock:
    the reference the setting's off position has to match."""

    def __init__(self) -> None:
        from neo2.core.voice import devices

        self.devices = devices
        self.last = (devices.default_input(), devices.default_output())
        self.last_stall_fix = 0.0
        self.broken = False

    def step(self) -> None:
        devices = self.devices
        if not any(o.active for o in list(audio._streams)):
            return
        now_dev = (devices.default_input(), devices.default_output())
        reason = ""
        if now_dev[0] != self.last[0]:
            reason = "the input device changed"
        elif now_dev[1] != self.last[1]:
            reason = "the output device changed"
        elif (
            self.broken
            or any(
                (isinstance(o, audio.Mic) and o.stalled())
                or (isinstance(o, audio.Speaker) and o.active and o._stream is None)
                for o in list(audio._streams)
            )
        ) and audio._clock() - self.last_stall_fix > (audio._STALL_RETRY_S if not self.broken else 5.0):
            reason, self.last_stall_fix = "the audio stopped working", audio._clock()
        self.last = now_dev
        if reason:
            audio._sleep(0.4)
            self.broken = not audio.reopen_all(reason)
            self.last = (devices.default_input(), devices.default_output())


def _play(make, script: dict[int, object], seconds: int, capsys) -> tuple:
    """Run a watcher through a scripted half hour: `script` maps a second to what happens then."""
    global _world
    for o in list(audio._streams):
        if isinstance(o, audio.Mic):
            o.stop()
        else:
            o.close()
    audio._streams.clear()
    fresh = _World()
    fresh.notes = _world.notes
    _world.__dict__.update(fresh.__dict__)  # the same world object the fixture patched in, reset
    spk = audio.Speaker()
    spk.start()
    mic = audio.Mic()
    mic.start()
    w = make()
    capsys.readouterr()
    for t in range(seconds):
        if t in script:
            script[t](mic)
        _world.sleep(1.0)
        w.step()
    # (how many input and output streams were opened: the order within one reopen is the stream
    # set's, which no watcher chooses)
    opened = (len(_world.inputs()), len(_world.streams) - len(_world.inputs()))
    out = (list(_world.reopens), _lines(capsys), opened, mic._stream is None)
    spk.close()
    mic.stop()
    return out


def _set(**kw):
    return lambda mic: _world.__dict__.update(kw)


_SCRIPTS = {
    "silent for half an hour": {},
    "won't open, then comes back": {0: _set(mode="fail"), 400: _set(mode="works")},
    "devices change while it works": {0: _set(mode="works"), 50: _set(dev=(5, 2)), 90: _set(dev=(5, 6))},
    "goes quiet, a device change, back": {
        0: _set(mode="works"),
        30: _set(mode="silent"),
        200: _set(dev=(3, 2)),
        400: _set(mode="works"),
    },
    "closed on purpose midway": {100: lambda mic: mic.stop(), 300: lambda mic: mic.start()},
}


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_off_is_todays_watcher_exactly(world, monkeypatch, capsys, name):
    _escalate(monkeypatch, False)
    today = _play(_Today, _SCRIPTS[name], 1800, capsys)
    now = _play(audio._Watcher, _SCRIPTS[name], 1800, capsys)
    assert now == today
    assert world.notes == [] and _problem() is None
    assert set(_counts()) <= {"mic_stall"}  # the stalls are still counted (content-free), nothing else


def test_off_retries_a_silent_microphone_every_30_s_and_never_tells(world, monkeypatch, capsys):
    _escalate(monkeypatch, False)
    mic = audio.Mic()
    mic.start()
    w = audio._Watcher()
    _run(w, 300)
    times = [t for t, _ in world.reopens]
    assert len(times) >= 8 and all(
        30 < b - a <= 30 + 1.0 + 0.4 + 0.1 for a, b in zip(times, times[1:], strict=False)
    )
    assert not _silent_tries(_lines(capsys)) and world.notes == [] and _problem() is None
    assert _counts() == {"mic_stall": len(times)}  # each of today's "reopened" lines was a new stall


# ---- the thread, the setting, the notification -----------------------------------------------------


def test_the_thread_is_a_thin_loop_that_outlives_a_bad_poll_and_steps_aside_when_replaced(
    world, monkeypatch, capsys
):
    steps: list[int] = []

    def step(self):
        steps.append(1)
        if len(steps) == 2:
            raise RuntimeError("PortAudio said no")
        if len(steps) == 4:
            audio._watcher = None  # another watcher took over: this one stops

    monkeypatch.setattr(audio._Watcher, "step", step)
    monkeypatch.setattr(audio, "_sleep", lambda s: None)
    t = threading.Thread(target=audio._watch, daemon=True)
    monkeypatch.setattr(audio, "_watcher", t)
    t.start()
    t.join(5)
    assert not t.is_alive() and len(steps) == 4
    assert "[audio] device watch hiccup: PortAudio said no" in capsys.readouterr().out


def test_the_setting_is_on_by_default_and_can_be_switched_off(monkeypatch):
    from neo2.core import config

    assert config.Settings.model_fields["mic_escalate"].default is True
    monkeypatch.setenv("NEO_MIC_ESCALATE", "false")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().mic_escalate is False


def test_the_notification_is_one_osascript_not_waited_for_with_the_words_as_an_argument(monkeypatch):
    import subprocess

    started: list[tuple] = []

    class Popen:
        def __init__(self, args, **kw):
            started.append((args, kw))

        def wait(self, *a, **k):
            raise AssertionError("the watcher never waits for the notification")

    monkeypatch.setattr(subprocess, "Popen", Popen)
    audio._notify(audio.MIC_PROBLEM)
    ((args, kw),) = started
    assert args[0] == "osascript" and args[-1] == audio.MIC_PROBLEM
    assert not any(audio.MIC_PROBLEM in a for a in args[:-1])  # never spliced into the script itself
    assert "display notification" in " ".join(args[:-1])
    assert kw.get("stdout") == subprocess.DEVNULL and kw.get("stderr") == subprocess.DEVNULL


def test_the_counters_are_the_ones_the_quality_report_reads():
    import inspect

    from neo2.core.reflex import learn

    src = inspect.getsource(audio)
    assert set(re.findall(r'_count\("(mic_[a-z_]+)"\)', src)) == set(learn._MIC)
