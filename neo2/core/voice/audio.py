"""Shared audio I/O: a 16 kHz mono mic stream and a 24 kHz playback sink with interrupt."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import weakref
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from neo2.core import effects

MIC_RATE = 16000
OUT_RATE = 24000
FRAME = 320  # 20 ms at 16 kHz


class Mic:
    """Continuously captures int16 PCM frames into an asyncio-friendly queue. Follows the default
    input device (AirPods in or out) and reopens itself if the stream goes silent."""

    def __init__(self, rate: int = MIC_RATE) -> None:
        self.rate = rate
        self.q: queue.Queue[bytes | None] = queue.Queue(maxsize=400)
        self._stream: sd.RawInputStream | None = None
        self._closed = False
        self._wanted = False  # started and not stopped: it should be capturing, stream or not
        self.last_frame = 0.0  # monotonic time of the latest callback

    def _cb(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self.last_frame = time.monotonic()
        frame = bytes(indata)
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass
        for fn in _taps:  # (a tuple: a tap added or removed meanwhile is a new one)
            try:
                fn(frame)
            except Exception:  # noqa: BLE001 — a listener's trouble never breaks the voice
                pass

    @property
    def active(self) -> bool:
        return self._wanted and not self._closed

    def stalled(self) -> bool:
        # No stream at all (a reopen failed: the device vanished) counts too, so it's retried.
        return self.active and (self._stream is None or time.monotonic() - self.last_frame > _STALL_S)

    def _open_stream(self) -> None:
        effects.guard("audio", "open the microphone", via=sd.RawInputStream)
        self._stream = sd.RawInputStream(
            samplerate=self.rate, channels=1, dtype="int16", blocksize=FRAME, callback=self._cb
        )
        self.last_frame = time.monotonic()
        self._stream.start()

    def _close_stream(self) -> None:
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.abort()
                st.close()
            except Exception:  # noqa: BLE001 — the device may already be gone
                pass

    def start(self) -> None:
        if self._stream:
            return
        self._closed = False
        self._open_stream()
        self._wanted = True
        _register(self)

    def stop(self) -> None:
        self._closed, self._wanted = True, False
        with _reopen_lock:  # never while the device watcher is reopening it
            self._close_stream()
        try:
            self.q.put_nowait(None)  # wake any reader so no executor thread is left blocked at exit
        except queue.Full:
            pass

    def _get(self) -> bytes:
        while not self._closed:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            return item or b""
        return b""

    async def read(self) -> bytes:
        """Next 20 ms frame; returns b"" once the mic is stopped (callers should exit their loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get)

    def drain(self) -> None:
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class Speaker:
    """Plays float32/int16 chunks in order; `interrupt()` drops what's queued. Follows the default
    output device."""

    def __init__(self, rate: int = OUT_RATE) -> None:
        self.rate = rate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._io = threading.Lock()  # the writer thread vs. a device switch
        self._thread: threading.Thread | None = None
        self._gen = 0
        self._closed = False
        self.playing = threading.Event()
        self.last_stop = 0.0  # when playback last went quiet (for a half-duplex mic tail)

    @property
    def active(self) -> bool:
        return self._thread is not None and not self._closed

    def _open_stream(self) -> None:
        effects.guard("audio", "open the speaker", via=sd.OutputStream)
        with self._io:
            self._stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="float32", blocksize=1024)
            self._stream.start()

    def _close_stream(self) -> None:
        st = self._stream
        if st is not None:
            try:
                st.abort()  # unblocks a write stuck on a device that went away
            except Exception:  # noqa: BLE001
                pass
        with self._io:
            st, self._stream = self._stream, None
            if st is not None:
                try:
                    st.close()
                except Exception:  # noqa: BLE001
                    pass

    def start(self) -> None:
        if self._thread:
            return
        self._closed = False
        self._open_stream()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _register(self)

    def _run(self) -> None:
        # `playing` stays set across the small gaps between streamed chunks; it only clears after
        # the queue has been empty for a moment, so listeners see one continuous utterance.
        item = self._q.get()
        while item is not None:
            gen, chunk = item  # type: ignore[misc]
            if gen == self._gen:
                self.playing.set()
                try:
                    with self._io:
                        if self._stream is not None:
                            self._stream.write(chunk)
                except Exception:  # noqa: BLE001 — a device switch mid-write: drop the chunk
                    pass
            try:
                item = self._q.get(timeout=0.5)  # tolerate network jitter between streamed chunks
            except queue.Empty:
                self.playing.clear()
                self.last_stop = time.time()
                item = self._q.get()
        self.playing.clear()

    def play(self, chunk: np.ndarray) -> None:
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0
        self._q.put((self._gen, np.ascontiguousarray(chunk, dtype=np.float32)))  # type: ignore[arg-type]

    def play_pcm16(self, data: bytes) -> None:
        self.play(np.frombuffer(data, dtype=np.int16))

    def interrupt(self) -> None:
        self._gen += 1
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        st = self._stream
        if st is not None:
            try:
                st.abort()  # stop the sound now (a write in progress returns)
            except Exception:  # noqa: BLE001
                pass
            with self._io:
                try:
                    if self._stream is st:
                        st.start()
                except Exception:  # noqa: BLE001
                    pass
        self.playing.clear()
        self.last_stop = time.time()

    async def wait_done(self) -> None:
        while not self._q.empty() or self.playing.is_set():
            await asyncio.sleep(0.05)

    def close(self) -> None:
        self._closed = True
        self._q.put(None)
        with _reopen_lock:  # never while the device watcher is reopening it
            self._close_stream()


# ---- following the devices ---------------------------------------------------------------------
_STALL_S = 3.0  # a running input stream delivers a frame every 20 ms; this long without one = dead
_STALL_RETRY_S = 30.0  # don't reopen for a stall more often than this (a broken device stays broken)
_streams: weakref.WeakSet = weakref.WeakSet()
_reopen_lock = threading.Lock()
_watcher: threading.Thread | None = None
_taps: tuple[Callable[[bytes], None], ...] = ()  # each running Mic's frames go to these too
_taps_lock = threading.Lock()


def mic_running() -> bool:
    """Is a voice microphone stream capturing now (one a :func:`tap` would hear)?"""
    return any(isinstance(o, Mic) and o.active and o._stream is not None for o in list(_streams))


def tap(fn: Callable[[bytes], None]) -> Callable[[], None]:
    """Hear every 20 ms int16 frame the voice's microphone captures (on PortAudio's thread: be quick,
    never block), without touching its queue. Returns the call that removes the tap."""
    global _taps
    with _taps_lock:
        _taps = (*_taps, fn)

    def untap() -> None:
        global _taps
        with _taps_lock:
            _taps = tuple(t for t in _taps if t is not fn)

    return untap


def _register(obj: Mic | Speaker) -> None:
    global _watcher
    _streams.add(obj)
    if _watcher is None or not _watcher.is_alive():
        _watcher = threading.Thread(target=_watch, daemon=True, name="neo-audio-devices")
        _watcher.start()


def reopen_all(reason: str) -> bool:
    """Close every stream that should be running, have PortAudio re-read the device list, and
    reopen on the current defaults. Returns whether everything came back."""
    with _reopen_lock:
        objs = [o for o in list(_streams) if o.active]
        if not objs:
            return True
        for o in objs:
            o._close_stream()
        # Separately: after a failed initialise, terminating raises "not initialized" and would
        # keep PortAudio down for good.
        if getattr(sd, "_initialized", 1) > 0:
            try:
                sd._terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            sd._initialize()
        except Exception as e:  # noqa: BLE001
            print(f"[audio] couldn't refresh the device list: {e}")
            return False
        ok = True
        for o in objs:
            if not o.active:
                continue  # stopped while we were busy: stays closed
            try:
                o._open_stream()
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[audio] couldn't reopen the {'microphone' if isinstance(o, Mic) else 'speaker'}: {e}")
        print(f"[audio] {reason} — reopened on the current devices" if ok else f"[audio] {reason} — will retry")
        return ok


def _watch() -> None:
    from neo2.core.voice import devices

    last = (devices.default_input(), devices.default_output())
    last_stall_fix = 0.0
    broken = False  # the last reopen didn't bring everything back: keep trying
    while True:
        time.sleep(1.0)
        try:
            if not any(o.active for o in list(_streams)):
                continue
            now_dev = (devices.default_input(), devices.default_output())
            reason = ""
            if now_dev[0] != last[0]:
                reason = "the input device changed"
            elif now_dev[1] != last[1]:
                reason = "the output device changed"
            elif (
                broken
                or any(
                    (isinstance(o, Mic) and o.stalled()) or (isinstance(o, Speaker) and o.active and o._stream is None)
                    for o in list(_streams)
                )
            ) and time.monotonic() - last_stall_fix > (_STALL_RETRY_S if not broken else 5.0):
                reason, last_stall_fix = "the audio stopped working", time.monotonic()
            last = now_dev
            if reason:
                time.sleep(0.4)  # let macOS finish switching (AirPods announce themselves in steps)
                broken = not reopen_all(reason)
                last = (devices.default_input(), devices.default_output())
        except Exception as e:  # noqa: BLE001 — the watcher must outlive any one bad poll
            print(f"[audio] device watch hiccup: {e}")


def rms(frame: bytes) -> float:
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)) / 32768.0) if len(a) else 0.0
