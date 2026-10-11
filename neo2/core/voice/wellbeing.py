"""Gentle check-ins: opt-in sound cues that make NEO ask a short, kind question.

Detection stays on-device. Apple's built-in sound classifier (SoundAnalysis, no model to download)
hears the 20 ms frames NEO's own microphone captures, through a listen-only tap
(neo2.core.voice.audio.tap(hold=False)). It never opens the microphone and never keeps it open: with
"Listen for 'Neo'" off it hears only the sessions the user opens themselves. Frames wait in a small
bounded queue in RAM; nothing about them — audio, labels, scores, times — goes to disk, the journal
or a log.

What leaves the Mac: with Gemini Live, the check-in line and about 5 s of the user's reply audio go
to Gemini (Live speaks the line in NEO's voice and hears the answer, as in any turn). NEO listens for
that reply only while "Listen for 'Neo'" has the microphone open anyway. With the on-device voice the
line is spoken and the reply heard on the Mac, and the reply's words (with a note that NEO just
checked in) go to the chat brain like any request (the README's "what you say and type"). Neither the
line nor the reply goes to NEO's memory journal (neo2.core.voice.live, neo2.core.agent.session).

How often (CheckinPolicy, RAM only): the same kind of check-in waits 10 min, then 1 h, then 4 h; at
most 3 a day; none for the rest of the day after a "no" or after two check-ins nobody answered. NEO
asks only while it is idle and no other app is playing sound (neo2.core.app).
"""

from __future__ import annotations

import asyncio
import ctypes
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

_FRAME_BYTES = 320 * 2  # the shared microphone: 320 int16 samples at 16 kHz, every 20 ms
_QUEUE_FRAMES = 100  # bounded to two seconds while Apple's model initializes or briefly falls behind
_NASAL_LABELS = ("nose_blowing", "sniffing", "sniffle", "sniffling")
_SNEEZE_LABELS = ("sneeze", "sneezing")
_CRY_LABELS = ("crying_sobbing", "crying", "sobbing")
_THROAT_LABELS = ("throat_clearing", "throat_clearing_cough", "throat_clear")
_LABELS = frozenset(("cough", *_CRY_LABELS, *_NASAL_LABELS, *_SNEEZE_LABELS, *_THROAT_LABELS))
# Apple's classifier judges 3 s windows, a new one every 1.5 s: one cough is in two of them. Without
# the window's time range (an older macOS, a test), two hits of one family this close are one sound.
_WINDOW_S = 3.0


class WellbeingMonitor:
    """Classify cough/crying cues only while NEO's own microphone is already capturing."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[str], None],
        is_ignored: Callable[[], bool],
        on_error: Callable[[str], None] | None = None,
        policy: CheckinPolicy | None = None,
    ) -> None:
        self._loop = loop
        self._on_event = on_event
        self._on_error = on_error
        self._is_ignored = is_ignored
        self.policy = policy or CheckinPolicy()
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._frames: queue.Queue[bytes | None] = queue.Queue(maxsize=_QUEUE_FRAMES)
        self._worker: threading.Thread | None = None
        self._untap: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._failed = False
        self._callback_failed = False  # set from inside Apple's callback, acted on by _run

    def set_enabled(self, enabled: bool) -> None:
        """Start or stop listening in on the existing microphone; never opens or holds an audio
        device. (The audio module is imported only when switched on: a Mac without the voice extra
        must still start NEO with check-ins off.)"""
        with self._lock:
            if enabled:
                try:
                    from neo2.core.voice import audio
                except Exception as error:  # noqa: BLE001 — no sounddevice / PortAudio on this Mac
                    self._failed = True
                    self._enabled.clear()
                    self._report_error(error)
                    return
                self._failed = False
                self._stop.clear()
                self._enabled.set()
                if self._worker is None or not self._worker.is_alive():
                    self._callback_failed = False
                    self._worker = threading.Thread(
                        target=self._run, name="neo-local-sound-checkins", daemon=True
                    )
                    self._worker.start()
                if self._untap is None:
                    # Listen-only: the voice closes its microphone whenever it would without us.
                    self._untap = audio.tap(self._capture, hold=False)
                return

            self._enabled.clear()
            if self._untap is not None:
                self._untap()
                self._untap = None
            self._clear_frames()

    def close(self) -> None:
        self.set_enabled(False)
        self._stop.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            self._clear_frames()
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                pass
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def _capture(self, frame: bytes) -> None:
        """PortAudio callback: bounded nonblocking enqueue only; no analysis or disk work here."""
        if not self._enabled.is_set() or len(frame) != _FRAME_BYTES:
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass  # keep the live audio path real-time if Apple's classifier falls behind

    def _clear_frames(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _emit(self, kind: str) -> None:
        if self._enabled.is_set():
            self._loop.call_soon_threadsafe(self._on_event, kind)

    def _report_error(self, error: Exception) -> None:
        # Avoid logging any frame contents, class confidence, or personal health detail.
        print(f"[wellbeing] local sound classifier unavailable: {type(error).__name__}")
        if self._on_error is not None:
            try:
                self._loop.call_soon_threadsafe(self._on_error, type(error).__name__)
            except RuntimeError:  # the loop has closed: NEO is quitting
                pass

    def _give_up(self, error: Exception) -> None:
        with self._lock:
            self._failed = True
            self._enabled.clear()
            if self._untap is not None:
                self._untap()
                self._untap = None
        self._clear_frames()
        self._report_error(error)

    def _run(self) -> None:
        try:
            analyzer, _request, observer, buffer = self._make_analyzer()
        except Exception as error:  # noqa: BLE001 — an unavailable OS framework must not affect voice
            self._give_up(error)
            return

        frame_position = 0
        state = _CueState(self._emit, self._is_ignored, may=self.policy.allows)
        observer._neo_state = state
        observer._neo_owner = self
        try:
            while not self._stop.is_set():
                try:
                    frame = self._frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                if not self._enabled.is_set():
                    continue
                if self._is_ignored():
                    state.reset()
                    continue
                buffer.int16ChannelData()[0].as_buffer(320)[:] = frame
                buffer.setFrameLength_(320)
                analyzer.analyzeAudioBuffer_atAudioFramePosition_(buffer, frame_position)
                frame_position += 320
                if self._callback_failed:  # (Apple calls back inside analyzeAudioBuffer: just now)
                    raise RuntimeError("the sound classifier's callback failed")
        except Exception as error:  # noqa: BLE001 — stop only this optional feature
            self._give_up(error)
        finally:
            observer._neo_owner = observer._neo_state = None  # a late callback finds nobody to tell

    @staticmethod
    def _make_analyzer() -> tuple[Any, Any, Any, Any]:
        """Create Apple's built-in classifier through PyObjC without a bundled model or download."""
        import objc
        from AVFoundation import AVAudioFormat, AVAudioPCMBuffer, AVAudioPCMFormatInt16

        framework = "/System/Library/Frameworks/SoundAnalysis.framework"
        classes: dict[str, Any] = {}
        objc.loadBundle("SoundAnalysis", classes, bundle_path=framework)
        library = ctypes.CDLL(framework + "/SoundAnalysis")
        identifier_pointer = ctypes.c_void_p.in_dll(library, "SNClassifierIdentifierVersion1").value
        identifier = objc.objc_object(c_void_p=identifier_pointer)
        request = classes["SNClassifySoundRequest"].alloc().initWithClassifierIdentifier_error_(identifier, None)
        if request is None:
            raise RuntimeError("Apple sound classifier did not initialize")
        labels = set(request.knownClassifications())
        if not {"cough", "crying_sobbing"}.issubset(labels):
            raise RuntimeError("Apple sound classifier is missing supported labels")

        observer = _observer_class(objc).alloc().init()
        audio_format = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVAudioPCMFormatInt16, 16000.0, 1, False
        )
        analyzer = classes["SNAudioStreamAnalyzer"].alloc().initWithFormat_(audio_format)
        if not analyzer.addRequest_withObserver_error_(request, observer, None):
            raise RuntimeError("Apple sound classifier could not attach to the stream")
        buffer = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(audio_format, 320)
        if buffer is None:
            raise RuntimeError("Could not allocate the audio analysis buffer")
        return analyzer, request, observer, buffer


_observer_cls: Any = None


def _observer_class(objc: Any) -> Any:
    """The SNResultsObserving class (one Objective-C class per process: defining it a second time —
    check-ins switched back on after a failure — raises, and that turned them off for good).

    Nothing may raise out of its callback: Apple calls it on its own audio thread, and a Python
    error there aborts the whole NEO core. A failure is flagged for the worker (`_callback_failed`),
    which then switches check-ins off the ordinary way."""
    global _observer_cls
    if _observer_cls is None:
        from Foundation import NSObject  # type: ignore[import-not-found]

        protocol = objc.protocolNamed("SNResultsObserving")

        class NEOWellbeingObserver(NSObject, protocols=[protocol]):  # type: ignore[misc,valid-type,call-arg]
            def request_didProduceResult_(self, _request, result) -> None:  # noqa: N802
                with objc.autorelease_pool():
                    owner = getattr(self, "_neo_owner", None)
                    state = getattr(self, "_neo_state", None)
                    if owner is None or state is None:
                        return
                    try:
                        state.update(_scores(result), _window(result))
                    except Exception:  # noqa: BLE001 — never into Apple's thread (see above)
                        owner._callback_failed = True

            def request_didFailWithError_(self, _request, _error) -> None:  # noqa: N802
                # The request is over (Apple sends nothing more): check-ins switch off. No audio,
                # error text or labels are logged.
                owner = getattr(self, "_neo_owner", None)
                if owner is not None:
                    owner._callback_failed = True

        _observer_cls = NEOWellbeingObserver
    return _observer_cls


def _scores(result: Any) -> dict[str, float]:
    """The confidences of the labels check-ins look at, from one classification result."""
    scores: dict[str, float] = {}
    for classification in result.classifications():
        label = str(classification.identifier())
        if label in _LABELS:
            scores[label] = float(classification.confidence())
    return scores


def _window(result: Any) -> tuple[float, float] | None:
    """(start, end) of the audio the result judged, in seconds of the analyzed stream (a CMTimeRange:
    (start, duration), each (value, timescale, …)). None when PyObjC can't read it."""
    try:
        time_range = result.timeRange()
        start, duration = time_range[0], time_range[1]
        begin = float(start[0]) / float(start[1])
        return begin, begin + float(duration[0]) / float(duration[1])
    except Exception:  # noqa: BLE001
        return None


# ---- how often NEO may check in -------------------------------------------------------------------
_FAMILY = {
    "cough": "cough",
    "cough_or_throat": "cough",
    "throat_clearing": "cough",
    "sniffling": "cold",
    "sneezing": "cold",
    "crying": "crying",
}
_BACKOFF_S = (10 * 60.0, 60 * 60.0, 4 * 60 * 60.0)  # after the 1st, 2nd, 3rd+ of a kind within a day
_MIN_GAP_S = 10 * 60.0  # between any two check-ins
_DAILY_CAP = 3
_UNANSWERED_PER_DAY = 2  # this many check-ins nobody answered: quiet until tomorrow
_DAY_S = 24 * 3600.0
# "No" to a check-in: quiet for the rest of the day. Only how the answer starts counts — "no, I'm
# not okay" too (NEO answers what they said; it just doesn't come back today unasked).
_DECLINE = re.compile(
    r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?neo[\s,!.]*)?(?:"
    r"no+|nope|nah|naa|not\s+now|stop(?:\s+(?:it|asking))?|leave\s+(?:me|it)(?:\s+alone)?|"
    r"(?:please\s+)?don'?t\s+ask|i'?m\s+(?:fine|ok(?:ay)?|good|alright|all\s+right)|"
    r"i\s+am\s+(?:fine|ok(?:ay)?|good|alright|all\s+right)|"
    r"nahi+n?|nahin|mat\s+pucho|rehne\s+do|(?:main\s+)?th(?:ee|i)k\s+(?:hoon|hu|hun|hai)|"
    r"नहीं|नही|रहने\s+दो|(?:मैं\s+)?ठीक\s+(?:हूँ|हूं|है))"
    r"(?![\w'])",
    re.I,
)


def declines(text: str) -> bool:
    """Does this reply to a check-in say "no" / "I'm fine" / "leave it"?"""
    return bool(_DECLINE.match(" ".join(str(text or "").split())))


def _local_day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))


class CheckinPolicy:
    """When NEO may check in. Asked from the classifier's thread and told the outcome on the event
    loop, so it locks. Kept in RAM only: nothing about the user's health is written anywhere to
    remember it by — a restart of NEO starts the day's count afresh."""

    def __init__(self, *, clock: Callable[[], float] = time.time, day_of: Callable[[float], str] = _local_day) -> None:
        self._clock = clock
        self._day_of = day_of
        self._lock = threading.Lock()
        self._given: dict[str, deque[float]] = {}  # family → its check-ins in the last 24 h
        self._last: float | None = None  # the latest check-in
        self._day = ""
        self._today = 0
        self._unanswered = 0
        self._quiet_day = ""

    def _roll(self, now: float) -> None:
        day = self._day_of(now)
        if day != self._day:
            self._day, self._today, self._unanswered = day, 0, 0
        for times in self._given.values():
            while times and now - times[0] > _DAY_S:
                times.popleft()

    def allows(self, kind: str) -> bool:
        """May NEO check in about `kind` now (the rules in the module doc)?"""
        with self._lock:
            now = self._clock()
            self._roll(now)
            if self._quiet_day == self._day or self._today >= _DAILY_CAP:
                return False
            if self._last is not None and now - self._last < _MIN_GAP_S:
                return False
            times = self._given.get(_FAMILY.get(kind, kind))
            if times:
                wait = _BACKOFF_S[min(len(times), len(_BACKOFF_S)) - 1]
                if now - times[-1] < wait:
                    return False
            return True

    def given(self, kind: str) -> None:
        """NEO just checked in about `kind`."""
        with self._lock:
            now = self._clock()
            self._roll(now)
            self._last = now
            self._today += 1
            self._given.setdefault(_FAMILY.get(kind, kind), deque()).append(now)

    def replied(self, text: str) -> None:
        """What the user said back ("" = nothing in the listen window)."""
        with self._lock:
            self._roll(self._clock())
            if not str(text or "").strip():
                self._unanswered += 1
                if self._unanswered >= _UNANSWERED_PER_DAY:
                    self._quiet_day = self._day
            elif declines(text):
                self._quiet_day = self._day


class _CueState:
    """Turns classification results into cues. Short-lived debounce state: RAM only, discarded with
    the monitor.

    One sound counts once. Apple judges overlapping 3 s windows, so one cough shows up in two
    results 1.5 s apart: a hit counts only when its window starts at or after the end of the window
    that gave the family's last hit (with no window known, `_WINDOW_S` apart). Hits older than their
    group's span are pruned on every result, and a cue the policy refuses (`may`) clears its hits:
    a cooldown never ends with yesterday's coughs already counted."""

    # group: (family for de-duplication, how long a hit stays counted, hits needed, the cue)
    _GROUPS = {
        "cough": ("throat", 75.0, 2, "cough"),
        "throat": ("throat", 40.0, 2, "throat_clearing"),
        "throat_proxy": ("throat", 40.0, 3, "cough_or_throat"),
        "nasal": ("nasal", 45.0, 2, "sniffling"),
        "sneeze": ("sneeze", 75.0, 3, "sneezing"),
    }
    _ORDER = ("cough", "nasal", "sneeze", "throat", "throat_proxy")  # the cue checked first wins

    def __init__(
        self,
        emit: Callable[[str], None],
        is_ignored: Callable[[], bool],
        *,
        may: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._is_ignored = is_ignored
        self._may = may or (lambda kind: True)
        self._clock = clock
        self._hits: dict[str, deque[float]] = {g: deque() for g in self._GROUPS}
        self._family_end: dict[str, float] = {}  # family → end of the window that gave its last hit
        self._family_at: dict[str, float] = {}  # family → when its last hit came (no window known)
        self._cry_started: float | None = None
        self._last_cry_seen = 0.0
        self._cry_prompted_in_episode = False

    def reset(self) -> None:
        for hits in self._hits.values():
            hits.clear()
        self._family_end.clear()
        self._family_at.clear()
        self._cry_started = None
        self._last_cry_seen = 0.0
        self._cry_prompted_in_episode = False

    def _hit(self, group: str, now: float, window: tuple[float, float] | None) -> None:
        family = self._GROUPS[group][0]
        if window is not None:
            if window[0] < self._family_end.get(family, float("-inf")):
                return  # the same sound, seen again in the next overlapping window
            self._family_end[family] = window[1]
        elif now - self._family_at.get(family, float("-inf")) < _WINDOW_S:
            return
        self._family_at[family] = now
        self._hits[group].append(now)

    def update(self, scores: dict[str, float], window: tuple[float, float] | None = None) -> None:
        if self._is_ignored():
            self.reset()
            return
        now = self._clock()
        for group, (_, span, _, _) in self._GROUPS.items():
            hits = self._hits[group]
            while hits and now - hits[0] > span:
                hits.popleft()

        cough = scores.get("cough", 0.0)
        # SoundAnalysis has no throat-clearing label on some macOS versions. Repeated lower-confidence
        # cough classifications are the closest built-in local cue; newer classifiers can provide
        # an explicit throat_clearing label through the same path.
        explicit_throat = max((scores.get(k, 0.0) for k in _THROAT_LABELS), default=0.0)
        if cough >= 0.78:
            self._hit("cough", now, window)
        elif explicit_throat >= 0.52:
            self._hit("throat", now, window)
        elif cough >= 0.52:
            self._hit("throat_proxy", now, window)
        if max((scores.get(k, 0.0) for k in _NASAL_LABELS), default=0.0) >= 0.52:
            self._hit("nasal", now, window)
        if max((scores.get(k, 0.0) for k in _SNEEZE_LABELS), default=0.0) >= 0.60:
            self._hit("sneeze", now, window)

        for group in self._ORDER:
            _, _, needed, kind = self._GROUPS[group]
            if len(self._hits[group]) >= needed:
                self._hits[group].clear()
                if self._may(kind):
                    self._emit(kind)
                return

        crying = max((scores.get(k, 0.0) for k in _CRY_LABELS), default=0.0)
        if crying >= 0.82:
            self._cry_started = self._cry_started if self._cry_started is not None else now
            self._last_cry_seen = now
            if not self._cry_prompted_in_episode and now - self._cry_started >= 5.0:
                self._cry_prompted_in_episode = True  # once per episode, asked or not
                if self._may("crying"):
                    self._emit("crying")
        elif self._cry_started is not None and now - self._last_cry_seen > 1.5:
            self._cry_started = None
            self._cry_prompted_in_episode = False
