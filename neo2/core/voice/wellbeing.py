"""Opt-in, on-device sound cues for gentle check-ins.

This module never opens the microphone. It taps the existing voice microphone's 20 ms PCM frames,
passes them directly to Apple's SoundAnalysis framework, and keeps only short-lived frames in a
bounded queue. It does not write audio, labels, or detection times to disk, the journal, or a model.
"""

from __future__ import annotations

import asyncio
import ctypes
import queue
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


class WellbeingMonitor:
    """Classify cough/crying cues only while NEO's own microphone is already capturing."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[str], None],
        is_ignored: Callable[[], bool],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._loop = loop
        self._on_event = on_event
        self._on_error = on_error
        self._is_ignored = is_ignored
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._frames: queue.Queue[bytes | None] = queue.Queue(maxsize=_QUEUE_FRAMES)
        self._worker: threading.Thread | None = None
        self._untap: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._failed = False

    def set_enabled(self, enabled: bool) -> None:
        """Start or stop observing the existing mic; never opens or owns an audio device."""
        from neo2.core.voice import audio

        with self._lock:
            if enabled:
                if self._failed:
                    self._failed = False
                self._stop.clear()
                self._enabled.set()
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._run, name="neo-local-sound-checkins", daemon=True
                    )
                    self._worker.start()
                if self._untap is None:
                    self._untap = audio.tap(self._capture)
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
            self._loop.call_soon_threadsafe(self._on_error, type(error).__name__)

    def _run(self) -> None:
        try:
            analyzer, request, observer, buffer = self._make_analyzer()
            observer.owner = self
        except Exception as error:  # noqa: BLE001 — an unavailable OS framework must not affect voice
            with self._lock:
                self._failed = True
                self._enabled.clear()
                if self._untap is not None:
                    self._untap()
                    self._untap = None
            self._clear_frames()
            self._report_error(error)
            return

        frame_position = 0
        state = _CueState(self._emit, self._is_ignored)
        observer.state = state
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
        except Exception as error:  # noqa: BLE001 — stop only this optional feature
            with self._lock:
                self._failed = True
                self._enabled.clear()
                if self._untap is not None:
                    self._untap()
                    self._untap = None
            self._clear_frames()
            self._report_error(error)

    @staticmethod
    def _make_analyzer() -> tuple[Any, Any, Any, Any]:
        """Create Apple's built-in classifier through PyObjC without a bundled model or download."""
        import objc
        from AVFoundation import AVAudioFormat, AVAudioPCMBuffer, AVAudioPCMFormatInt16
        from Foundation import NSObject

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

        protocol = objc.protocolNamed("SNResultsObserving")

        class Observer(NSObject, protocols=[protocol]):
            owner: WellbeingMonitor | None = None
            state: _CueState | None = None

            def request_didProduceResult_(self, _request, result) -> None:
                if self.owner is None or self.state is None:
                    return
                scores: dict[str, float] = {}
                for classification in result.classifications():
                    label = str(classification.identifier())
                    if label in (
                        "cough", *_CRY_LABELS, *_NASAL_LABELS, *_SNEEZE_LABELS, *_THROAT_LABELS
                    ):
                        scores[label] = float(classification.confidence())
                self.state.update(scores)

            def request_didFailWithError_(self, _request, _error) -> None:
                # The stream loop owns recovery. No audio, error text, or labels are logged here.
                return

        observer = Observer.alloc().init()
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


class _CueState:
    """Short-lived debounce state: it exists in RAM only and is discarded with the monitor."""

    def __init__(self, emit: Callable[[str], None], is_ignored: Callable[[], bool]) -> None:
        self._emit = emit
        self._is_ignored = is_ignored
        self._cough_hits: deque[float] = deque()
        self._last_cough_hit = 0.0
        self._throat_hits: deque[float] = deque()
        self._throat_proxy_hits: deque[float] = deque()
        self._nasal_hits: deque[float] = deque()
        self._sneeze_hits: deque[float] = deque()
        self._last_throat_hit = 0.0
        self._last_nasal_hit = 0.0
        self._last_sneeze_hit = 0.0
        self._last_physical_prompt = 0.0
        self._cry_started: float | None = None
        self._last_cry_seen = 0.0
        self._last_cry_prompt = 0.0
        self._cry_prompted_in_episode = False

    def reset(self) -> None:
        self._cough_hits.clear()
        self._last_cough_hit = 0.0
        self._throat_hits.clear()
        self._throat_proxy_hits.clear()
        self._nasal_hits.clear()
        self._sneeze_hits.clear()
        self._last_throat_hit = self._last_nasal_hit = self._last_sneeze_hit = 0.0
        self._cry_started = None
        self._last_cry_seen = 0.0
        self._cry_prompted_in_episode = False

    def update(self, scores: dict[str, float]) -> None:
        if self._is_ignored():
            self.reset()
            return
        now = time.monotonic()
        cough = scores.get("cough", 0.0)
        if cough >= 0.78 and now - self._last_cough_hit >= 1.4:
            self._last_cough_hit = now
            self._push_recent(self._cough_hits, now, 75.0)
        # SoundAnalysis has no throat-clearing label on some macOS versions. Repeated lower-confidence
        # cough classifications are the closest built-in local cue; newer classifiers can provide
        # an explicit throat_clearing label through the same path.
        explicit_throat = max((scores.get(k, 0.0) for k in _THROAT_LABELS), default=0.0)
        proxy_throat = cough if cough < 0.78 else 0.0
        if explicit_throat >= 0.52 and now - self._last_throat_hit >= 1.0:
            self._last_throat_hit = now
            self._push_recent(self._throat_hits, now, 40.0)
        elif proxy_throat >= 0.52 and now - self._last_throat_hit >= 1.0:
            self._last_throat_hit = now
            self._push_recent(self._throat_proxy_hits, now, 40.0)

        if (
            len(self._cough_hits) >= 2
            and now - self._last_physical_prompt >= 10 * 60
        ):
            self._physical_prompt("cough", now)
            self._cough_hits.clear()
            return

        nasal = max((scores.get(k, 0.0) for k in _NASAL_LABELS), default=0.0)
        if nasal >= 0.52 and now - self._last_nasal_hit >= 1.1:
            self._last_nasal_hit = now
            self._push_recent(self._nasal_hits, now, 45.0)
        sneeze = max((scores.get(k, 0.0) for k in _SNEEZE_LABELS), default=0.0)
        if sneeze >= 0.60 and now - self._last_sneeze_hit >= 0.8:
            self._last_sneeze_hit = now
            self._push_recent(self._sneeze_hits, now, 75.0)

        if (
            len(self._nasal_hits) >= 2
            and now - self._last_physical_prompt >= 10 * 60
        ):
            self._physical_prompt("sniffling", now)
            self._nasal_hits.clear()
            return
        if (
            len(self._sneeze_hits) >= 3
            and now - self._last_physical_prompt >= 10 * 60
        ):
            self._physical_prompt("sneezing", now)
            self._sneeze_hits.clear()
            return
        if (
            len(self._throat_hits) >= 2
            and now - self._last_physical_prompt >= 10 * 60
        ):
            self._physical_prompt("throat_clearing", now)
            self._throat_hits.clear()
            return
        if (
            len(self._throat_proxy_hits) >= 3
            and now - self._last_physical_prompt >= 10 * 60
        ):
            self._physical_prompt("cough_or_throat", now)
            self._throat_proxy_hits.clear()
            return

        crying = max((scores.get(k, 0.0) for k in _CRY_LABELS), default=0.0)
        if crying >= 0.82:
            self._cry_started = self._cry_started or now
            self._last_cry_seen = now
            if (
                not self._cry_prompted_in_episode
                and now - self._cry_started >= 5.0
                and now - self._last_cry_prompt >= 45 * 60
                and now - self._last_physical_prompt >= 10 * 60
            ):
                self._last_cry_prompt = now
                self._last_physical_prompt = now
                self._cry_prompted_in_episode = True
                self._emit("crying")
        elif self._cry_started is not None and now - self._last_cry_seen > 1.5:
            self._cry_started = None
            self._cry_prompted_in_episode = False

    @staticmethod
    def _push_recent(hits: deque[float], now: float, window: float) -> None:
        hits.append(now)
        while hits and now - hits[0] > window:
            hits.popleft()

    def _physical_prompt(self, kind: str, now: float) -> None:
        self._last_physical_prompt = now
        self._emit(kind)
