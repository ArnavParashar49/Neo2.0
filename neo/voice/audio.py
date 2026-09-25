"""Shared audio I/O: a 16 kHz mono mic stream and a 24 kHz playback sink with interrupt."""

from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd

MIC_RATE = 16000
OUT_RATE = 24000
FRAME = 320  # 20 ms at 16 kHz


class Mic:
    """Continuously captures int16 PCM frames into an asyncio-friendly queue."""

    def __init__(self, rate: int = MIC_RATE) -> None:
        self.rate = rate
        self.q: queue.Queue[bytes | None] = queue.Queue(maxsize=400)
        self._stream: sd.RawInputStream | None = None
        self._closed = False

    def _cb(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def start(self) -> None:
        if self._stream:
            return
        self._stream = sd.RawInputStream(
            samplerate=self.rate, channels=1, dtype="int16", blocksize=FRAME, callback=self._cb
        )
        self._stream.start()

    def stop(self) -> None:
        self._closed = True
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
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
    """Plays float32/int16 chunks in order; `interrupt()` drops what's queued."""

    def __init__(self, rate: int = OUT_RATE) -> None:
        self.rate = rate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._gen = 0
        self.playing = threading.Event()

    def start(self) -> None:
        if self._thread:
            return
        self._stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="float32", blocksize=1024)
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            gen, chunk = item  # type: ignore[misc]
            if gen != self._gen:
                continue
            self.playing.set()
            try:
                self._stream.write(chunk)  # type: ignore[union-attr]
            except Exception:
                pass
            if self._q.empty():
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
        if self._stream:
            try:
                self._stream.abort()
                self._stream.start()
            except Exception:
                pass
        self.playing.clear()

    async def wait_done(self) -> None:
        while not self._q.empty() or self.playing.is_set():
            await asyncio.sleep(0.05)

    def close(self) -> None:
        self._q.put(None)
        if self._stream:
            self._stream.stop()
            self._stream.close()


def rms(frame: bytes) -> float:
    a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)) / 32768.0) if len(a) else 0.0
