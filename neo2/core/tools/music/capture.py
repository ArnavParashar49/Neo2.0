"""What NEO hears when it names a song: the microphone and the Mac's own sound, in memory only.

Two sources, both 16 kHz mono int16 like the voice's microphone (neo2.core.voice.audio):

* ``mic`` — a sounddevice input stream of its own. macOS lets a second one open beside the voice's,
  so the voice pipeline's Mic and its queue are never touched.
* ``mac`` — what the Mac itself plays (a song in headphones the microphone can't hear), through a
  ScreenCaptureKit stream with audio: NEO's own sound and windows left out, its 2×2-pixel picture
  thrown away. Only while the default output device is running (CoreAudio's
  kAudioDevicePropertyDeviceIsRunningSomewhere) and NEO already has Screen Recording: it is never
  asked for here (asking would bring up macOS's prompt), the microphone alone listens then.

Each source fills a ring of at most ``RING_S`` seconds (about 420 KB) and knows how loud the latest
tenth of a second was. Nothing is written to disk, logged or put on the event bus, and ``stop()``
zeroes the rings.

Opening either is a real effect: ``effects.guard("audio", …)`` right before the stream opens, refused
under NEO's tests and NEO_NO_EFFECTS=1 unless a fake stands in for the real API.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from neo2.core import effects

RATE = 16000
BLOCK = 1600  # the microphone's callback: 100 ms
RING_S = 13.0  # Shazam's signature covers 12 s: the latest 13 s is all that is kept
START_S = 2.0  # a source that hasn't started by then is left out
SILENT_DB = -50.0  # quieter than this (RMS, dBFS) is a quiet room, not music worth a request
_FLOOR_DB, _TOP_DB = -60.0, -12.0  # the card's meter: 0 at the floor, 1 at the top
_FLOAT, _NON_INTERLEAVED = 1, 32  # CoreAudio's kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved

Push = Callable[[np.ndarray], None]


class Unavailable(RuntimeError):
    """A source can't listen; the message is for the user."""


def rms_db(pcm: np.ndarray) -> float:
    """The loudness of int16 samples, in dBFS (-120 for none)."""
    if pcm.size == 0:
        return -120.0
    x = pcm.astype(np.float32) / 32768.0
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-6))


def level(db: float) -> float:
    """dBFS as the card's meter shows it (0..1)."""
    return min(1.0, max(0.0, (db - _FLOOR_DB) / (_TOP_DB - _FLOOR_DB)))


def audible(pcm: np.ndarray) -> bool:
    """Loud enough to be music someone wants named (a silent Mac, a quiet room are not)."""
    return rms_db(pcm) > SILENT_DB


class Ring:
    """The latest ``seconds`` of one source's audio (int16), written from its own thread."""

    def __init__(self, seconds: float = RING_S, rate: int = RATE) -> None:
        self.rate = rate
        self._buf = np.zeros(int(seconds * rate), np.int16)
        self._pos = 0  # where the next sample goes
        self._total = 0  # samples heard since the start
        self._db = -120.0  # the latest chunk's loudness
        self._wiped = False  # once wiped it takes nothing more (a push already on its way included)
        self._mu = threading.Lock()

    def write(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.int16).ravel()
        n = pcm.size
        if n == 0:
            return
        db = rms_db(pcm[-self.rate // 10 :])
        cap = self._buf.size
        with self._mu:
            if self._wiped:
                return
            if n >= cap:
                pcm = pcm[-cap:]
            k = min(pcm.size, cap - self._pos)
            self._buf[self._pos : self._pos + k] = pcm[:k]
            self._buf[: pcm.size - k] = pcm[k:]
            self._pos = (self._pos + pcm.size) % cap
            self._total += n
            self._db = db

    def last(self, seconds: float) -> np.ndarray:
        """A copy of the latest ``seconds`` (fewer when less was heard)."""
        with self._mu:
            n = min(int(seconds * self.rate), self._total, self._buf.size)
            if n <= 0:
                return np.zeros(0, np.int16)
            return self._buf[np.arange(self._pos - n, self._pos) % self._buf.size]

    def heard_s(self) -> float:
        with self._mu:
            return self._total / self.rate

    def level(self) -> float:
        with self._mu:
            return level(self._db)

    def wipe(self) -> None:
        """Nothing heard stays in memory, and nothing more comes in."""
        with self._mu:
            self._wiped = True
            self._buf[:] = 0
            self._pos = self._total = 0
            self._db = -120.0


class Source(Protocol):
    """What listens: ``start(push)`` calls ``push(int16 samples at 16 kHz)`` from any thread until
    ``stop()`` (which may come before start returns, from another thread)."""

    name: str

    def start(self, push: Push) -> None: ...

    def stop(self) -> None: ...


# ---- the microphone ------------------------------------------------------------------------------


class MicSource:
    """The default input device, through a sounddevice stream of its own."""

    name = "mic"

    def __init__(self) -> None:
        self._stream: Any = None
        self._stopped = False
        self._mu = threading.Lock()

    def start(self, push: Push) -> None:
        try:
            import sounddevice as sd
        except Exception as e:  # noqa: BLE001 — (the voice extra isn't installed, or no PortAudio)
            raise Unavailable("the microphone can't be used here (sounddevice isn't installed)") from e
        effects.guard(effects.AUDIO, "open the microphone to recognise a song", via=sd.RawInputStream)

        def on_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            push(np.frombuffer(bytes(indata), dtype=np.int16))

        try:
            stream = sd.RawInputStream(
                samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK, callback=on_audio
            )
        except Exception as e:  # noqa: BLE001 — no input device, or it's busy
            raise Unavailable("the microphone couldn't be opened") from e
        with self._mu:
            if self._stopped:  # stopped while it opened: never started
                with contextlib.suppress(Exception):
                    stream.close()
                return
            self._stream = stream
        stream.start()

    def stop(self) -> None:
        with self._mu:
            self._stopped = True
            stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):  # the device may be gone already
                stream.abort()
                stream.close()


# ---- the Mac's own sound -------------------------------------------------------------------------


def _property(ca: Any, objc: Any, obj: int, selector: int) -> int:
    """One 32-bit CoreAudio property of an audio object (0 when it can't be read)."""
    addr = ca.AudioObjectPropertyAddress(
        selector, ca.kAudioObjectPropertyScopeGlobal, ca.kAudioObjectPropertyElementMain
    )
    status, _size, data = ca.AudioObjectGetPropertyData(obj, addr, 0, objc.NULL, 4, None)
    if status != 0 or data is None:
        return 0
    return int.from_bytes(bytes(data)[:4], sys.byteorder)


def output_playing() -> bool:
    """Is the Mac playing sound now: its default output device running, for any app? (False when
    it can't tell.) A read: nothing is captured to find out."""
    try:
        import CoreAudio as ca  # type: ignore[import-not-found]
        import objc

        dev = _property(ca, objc, ca.kAudioObjectSystemObject, ca.kAudioHardwarePropertyDefaultOutputDevice)
        return bool(dev) and bool(_property(ca, objc, dev, ca.kAudioDevicePropertyDeviceIsRunningSomewhere))
    except Exception:  # noqa: BLE001 — no CoreAudio bindings, or an API that moved
        return False


def screen_recording() -> bool:
    """NEO already has Screen Recording (the preflight check: it never brings up macOS's prompt)."""
    try:
        import Quartz as Q

        return bool(Q.CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001
        return False


def mac_ready() -> str:
    """ "" when the Mac's own sound can be listened to now, else why not."""
    if not output_playing():
        return "nothing is playing on this Mac"
    try:
        from neo2.core.tools.computer import sck

        sck._framework()
    except Exception:  # noqa: BLE001 — (sck.Failed, or no Quartz)
        return "ScreenCaptureKit isn't installed"
    if not screen_recording():
        return "NEO doesn't have Screen Recording"
    return ""


def plan() -> tuple[list[str], str]:
    """The sources to listen to now: always the microphone, and the Mac's own sound while it plays
    something NEO may capture. With why the Mac's sound is left out ("" when it isn't)."""
    why = mac_ready()
    return (["mic"] if why else ["mic", "mac"]), why


def to_pcm16k(raw: bytes, *, rate: float, channels: int, flags: int, bits: int, frames: int) -> np.ndarray:
    """One CoreAudio linear-PCM buffer (float or integer, interleaved or one plane per channel) as
    16 kHz mono int16. ScreenCaptureKit is asked for 16 kHz mono already; anything else is mixed
    down and resampled here."""
    channels = max(1, int(channels))
    if flags & _FLOAT:
        dtype: Any = {32: np.float32, 64: np.float64}.get(bits)
    else:
        dtype = {16: np.int16, 32: np.int32}.get(bits)
    if dtype is None or frames <= 0:
        return np.zeros(0, np.int16)
    x = np.frombuffer(raw[: len(raw) - len(raw) % np.dtype(dtype).itemsize], dtype=dtype)
    n = min(frames, x.size // channels)
    if channels > 1:
        planes = (
            x[: n * channels].reshape(channels, n)
            if flags & _NON_INTERLEAVED
            else x[: n * channels].reshape(n, channels).T
        )
        y = planes.astype(np.float32).mean(axis=0)
    else:
        y = x[:n].astype(np.float32)
    if not flags & _FLOAT:
        y /= 32768.0 if bits == 16 else 2147483648.0
    rate = float(rate or RATE)
    if y.size and abs(rate - RATE) > 1:
        step = rate / RATE
        if abs(step - round(step)) < 1e-6 and step >= 2:  # 48 kHz → 16 kHz: each three averaged
            k = int(round(step))
            y = y[: y.size - y.size % k].reshape(-1, k).mean(axis=1)
        else:
            out = max(1, int(round(y.size / step)))
            y = np.interp(np.linspace(0, y.size - 1, out), np.arange(y.size), y).astype(np.float32)
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)


def _pcm_of(cm: Any, buf: Any) -> np.ndarray | None:
    """A ScreenCaptureKit audio sample buffer's samples, copied out (16 kHz mono int16)."""
    asbd = cm.CMAudioFormatDescriptionGetStreamBasicDescription(cm.CMSampleBufferGetFormatDescription(buf))
    block = cm.CMSampleBufferGetDataBuffer(buf)
    frames = int(cm.CMSampleBufferGetNumSamples(buf))
    if asbd is None or block is None or frames <= 0:
        return None
    size = int(cm.CMBlockBufferGetDataLength(block))
    status, data = cm.CMBlockBufferCopyDataBytes(block, 0, size, None)
    if status != 0 or data is None:
        return None
    return to_pcm16k(
        bytes(data), rate=float(asbd.mSampleRate), channels=int(asbd.mChannelsPerFrame),
        flags=int(asbd.mFormatFlags), bits=int(asbd.mBitsPerChannel), frames=frames,
    )  # fmt: skip


_output_cls: Any = None


def _output_class(objc: Any) -> Any:
    """The SCStreamOutput that hands each sample buffer to Python (one ObjC class per process)."""
    global _output_cls
    if _output_cls is None:
        from Foundation import NSObject  # type: ignore[import-not-found]

        protocols = []
        with contextlib.suppress(Exception):  # the method's exact signature (its type is an NSInteger)
            protocols = [objc.protocolNamed("SCStreamOutput")]

        class NEOSongAudioOutput(NSObject, protocols=protocols):  # type: ignore[misc,valid-type,call-arg]
            def stream_didOutputSampleBuffer_ofType_(self, stream, buf, kind):  # noqa: N802
                with objc.autorelease_pool():
                    fn = getattr(self, "_neo_on_buffer", None)
                    if fn is not None:
                        with contextlib.suppress(Exception):  # a bad buffer never kills the queue
                            fn(buf, int(kind))

        _output_cls = NEOSongAudioOutput
    return _output_cls


class MacSource:
    """What the Mac itself plays, through a ScreenCaptureKit stream with audio (macOS 13+)."""

    name = "mac"

    def __init__(self) -> None:
        self._stream: Any = None
        self._out: Any = None
        self._queue: Any = None
        self._stopped = False
        self._mu = threading.Lock()

    def start(self, push: Push) -> None:
        from neo2.core.tools.computer import sck

        try:
            sc = sck._framework()
        except sck.Failed:
            raise Unavailable(
                "the Mac's own sound can't be captured (ScreenCaptureKit isn't installed)"
            ) from None
        effects.guard(effects.AUDIO, "capture the Mac's sound (ScreenCaptureKit)", via=sc.SCStream)
        import CoreMedia as cm  # type: ignore[import-not-found]
        import objc

        from neo2.core.tools.camera.capture import _dispatch_queue

        deadline = time.monotonic() + START_S
        try:
            content = sck._shareable(deadline)
        except sck.Failed as e:
            raise Unavailable(f"the Mac's own sound can't be captured ({e})") from None
        displays = list(content.displays() or [])
        if not displays:
            raise Unavailable("the Mac's own sound can't be captured (no display to capture it with)")
        flt = sc.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            displays[0], sck._neo_apps(content), []
        )
        cfg = sc.SCStreamConfiguration.alloc().init()
        cfg.setCapturesAudio_(True)
        cfg.setExcludesCurrentProcessAudio_(True)  # NEO's own voice is never part of the song
        cfg.setSampleRate_(RATE)
        cfg.setChannelCount_(1)
        cfg.setWidth_(2)  # a stream always has a picture: as small and rare as it goes, and dropped
        cfg.setHeight_(2)
        cfg.setMinimumFrameInterval_(cm.CMTimeMake(1, 1))
        cfg.setShowsCursor_(False)
        stream = sc.SCStream.alloc().initWithFilter_configuration_delegate_(flt, cfg, None)
        audio = int(sc.SCStreamOutputTypeAudio)

        def on_buffer(buf: Any, kind: int) -> None:
            if kind == audio and (pcm := _pcm_of(cm, buf)) is not None and pcm.size:
                push(pcm)

        out = _output_class(objc).alloc().init()
        out._neo_on_buffer = on_buffer
        queue = _dispatch_queue(objc, b"neo.song.audio")
        for kind in (sc.SCStreamOutputTypeScreen, sc.SCStreamOutputTypeAudio):  # (a screen output too:
            # without one ScreenCaptureKit complains about every picture it drops)
            ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(out, kind, queue, None)
            if not ok:
                out._neo_on_buffer = None
                raise Unavailable(f"the Mac's own sound can't be captured ({sck._ns_error(err)})")
        started = threading.Event()
        box: dict[str, Any] = {}

        def on_start(error: Any) -> None:
            box["error"] = error
            started.set()

        with self._mu:
            if self._stopped:  # stopped while it was being set up: never started
                out._neo_on_buffer = None
                return
            self._stream, self._out, self._queue = stream, out, queue
        stream.startCaptureWithCompletionHandler_(on_start)
        if not started.wait(max(0.1, deadline - time.monotonic())):
            self.stop()
            raise Unavailable("the Mac's own sound can't be captured (ScreenCaptureKit didn't start in time)")
        if box.get("error") is not None:
            self.stop()
            raise Unavailable(f"the Mac's own sound can't be captured ({sck._ns_error(box['error'])})")

    def stop(self) -> None:
        with self._mu:
            self._stopped = True
            stream, self._stream = self._stream, None
            out, self._out = self._out, None
        if out is not None:
            out._neo_on_buffer = None
        if stream is not None:
            with contextlib.suppress(Exception):  # (never waited for: it stops on its own queue)
                stream.stopCaptureWithCompletionHandler_(lambda error: None)


SOURCES: dict[str, Callable[[], Source]] = {"mic": MicSource, "mac": MacSource}


def source(name: str) -> Source:
    return SOURCES[name]()


# ---- listening to them together ------------------------------------------------------------------


class Listener:
    """One listen: its sources, each into a ring of its own. ``stop()`` (idempotent) stops them all
    and wipes what they heard."""

    def __init__(self, sources: list[Source], *, ring_s: float = RING_S) -> None:
        self.sources = list(sources)
        self.rings = {s.name: Ring(ring_s) for s in self.sources}
        self.started: list[str] = []
        self.failed: dict[str, str] = {}
        self.capturing = False

    async def start(self, timeout: float = START_S) -> list[str]:
        """Start every source at once (each on a thread of its own: an ObjC or PortAudio start can't
        be cancelled). Returns the names of those that started; one that fails or takes longer than
        ``timeout`` is left out (and stopped once its start is over)."""
        self.capturing = True

        def pusher(ring: Ring) -> Push:
            def push(pcm: np.ndarray) -> None:
                if self.capturing:
                    ring.write(pcm)

            return push

        futs = {
            asyncio.ensure_future(asyncio.to_thread(s.start, pusher(self.rings[s.name]))): s
            for s in self.sources
        }
        try:
            done, pending = await asyncio.wait(futs, timeout=timeout)
        except BaseException:  # cancelled while starting (the user's stop): nothing stays on
            self.stop()
            for f, s in futs.items():
                f.add_done_callback(lambda _f, s=s: s.stop())
            raise
        refused: BaseException | None = None
        for f, s in futs.items():
            if f in pending:
                self.failed[s.name] = "it didn't start in time"
                s.stop()
                f.add_done_callback(lambda _f, s=s: s.stop())  # (a start that finishes later: off at once)
                continue
            e = f.exception()
            if e is None:
                self.started.append(s.name)
            elif isinstance(e, Exception):
                self.failed[s.name] = str(e) if isinstance(e, Unavailable) else f"{type(e).__name__}"
                s.stop()
            else:  # an effect refused (tests): it goes through, whatever else started goes off
                refused = e
        if refused is not None:
            self.stop()
            raise refused
        if not self.started:
            self.capturing = False
        return list(self.started)

    def why(self) -> str:
        """Why the sources that didn't start didn't (for the user)."""
        return "; ".join(dict.fromkeys(self.failed.values())) or "nothing to listen with"

    def heard_s(self) -> float:
        return max((r.heard_s() for n, r in self.rings.items() if n in self.started), default=0.0)

    def level(self) -> float:
        return max((r.level() for n, r in self.rings.items() if n in self.started), default=0.0)

    def clips(self, seconds: float) -> dict[str, np.ndarray]:
        """The latest ``seconds`` each started source heard (copies)."""
        return {n: r.last(seconds) for n, r in self.rings.items() if n in self.started}

    def stop(self) -> None:
        self.capturing = False
        for s in self.sources:
            with contextlib.suppress(Exception):
                s.stop()
        for r in self.rings.values():
            r.wipe()
