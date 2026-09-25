"""Local voice cascade: wake word → silero VAD → Parakeet STT → brain → Kokoro TTS.

Fully offline, no quota. The mic loop never blocks on the brain: each utterance is handled in
its own task, so the loop keeps feeding the wake spotter while NEO thinks or speaks — saying
"Neo" or "stop" mid-sentence cuts the speech off (barge-in) and re-arms listening.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable

import numpy as np

from neo.agent.early import early
from neo.config import settings
from neo.events import NeoState, bus
from neo.voice.audio import MIC_RATE, Mic, Speaker, rms
from neo.voice.wake import WakeSpotter

TextHandler = Callable[[str], Awaitable[None]]

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_FOLLOWUP_S = 6.0  # keep listening this long after a reply without needing the wake word
_MAX_UTTER_S = 20.0
_SILENCE_MS = 800  # trailing silence that ends an utterance (silero)
_NO_SPEECH_S = 5.0
_VAD_CHUNK = 512  # silero works on 32 ms windows at 16 kHz
_PARTIAL_EVERY_S = 1.2  # re-transcribe the growing buffer this often for live partials


class LocalVoice:
    handles_text = False  # typed text goes through the session; we only speak replies

    def __init__(self, on_text: TextHandler) -> None:
        self._on_text = on_text
        self.mic = Mic()
        self.spk = Speaker()
        self._wake: WakeSpotter | None = None
        self._stt = None
        self._tts = None
        self._loop_task: asyncio.Task | None = None
        self._handler: asyncio.Task | None = None
        self._ptt = False
        self._armed = asyncio.Event()
        self._speaking = False
        self._capturing = False
        self._followup_until = 0.0
        self._vad = None
        self._vad_model = None

    # ---- lifecycle -------------------------------------------------------------------
    async def start(self) -> None:
        await asyncio.to_thread(self._load_models)
        self.mic.start()
        self.spk.start()
        self._loop_task = asyncio.create_task(self._loop())

    def _load_models(self) -> None:
        s = settings()
        self._wake = WakeSpotter(MIC_RATE)
        from mlx_audio.stt.utils import load as stt_load
        from mlx_audio.tts.utils import load_model as tts_load

        self._stt = stt_load(s.stt_model)
        self._tts = tts_load(s.tts_model)
        try:
            from silero_vad import VADIterator, load_silero_vad

            self._vad_model = load_silero_vad()
            self._vad = VADIterator(
                self._vad_model,
                threshold=0.5,
                sampling_rate=MIC_RATE,
                min_silence_duration_ms=_SILENCE_MS,
                speech_pad_ms=200,
            )
        except Exception as e:  # noqa: BLE001 — fall back to the energy gate
            print(f"[voice] silero VAD unavailable ({e}); using energy gate")
            self._vad = None
        # Warm both so the first real utterance is fast.
        self._tts_chunks("Ready.")
        import mlx.core as mx

        self._stt.generate(mx.array(np.zeros(MIC_RATE, dtype=np.float32)))

    async def close(self) -> None:
        for t in (self._loop_task, self._handler):
            if t:
                t.cancel()
        self.mic.stop()
        self.spk.close()

    # ---- external controls -----------------------------------------------------------
    async def wake(self, source: str = "ui") -> None:
        self._armed.set()

    async def push_to_talk(self, on: bool) -> None:
        self._ptt = on
        if on:
            self._armed.set()

    async def interrupt(self) -> None:
        """Stop speaking and abandon the in-flight request (barge-in / stop button)."""
        self.spk.interrupt()
        self._speaking = False
        if self._handler and not self._handler.done():
            self._handler.cancel()

    async def speak(self, text: str) -> None:
        if not text.strip() or self._tts is None:
            return
        self._speaking = True
        await bus().set_state(NeoState.SPEAKING)
        try:
            for sent in _SENT_RE.split(text.strip()):
                if not self._speaking:
                    break
                for chunk in await asyncio.to_thread(self._tts_chunks, sent):
                    if not self._speaking:
                        break
                    self.spk.play(chunk)
            await self.spk.wait_done()
        finally:
            self._speaking = False
            if bus().state == NeoState.SPEAKING:
                await bus().set_state(NeoState.IDLE)

    def _tts_chunks(self, text: str) -> list[np.ndarray]:
        s = settings()
        out = []
        for r in self._tts.generate(text=text, voice=s.tts_voice, speed=1.05, lang_code="a", stream=True):
            out.append(np.array(r.audio, dtype=np.float32))
        return out

    # ---- main loop -------------------------------------------------------------------
    async def _loop(self) -> None:
        assert self._wake
        while True:
            frame = await self.mic.read()
            if not frame:  # mic stopped
                return
            if self._capturing:
                continue  # _capture_utterance owns the frames right now
            hit = self._wake.feed(frame)
            busy = self._speaking or (self._handler is not None and not self._handler.done())
            if busy:
                if (
                    hit == "stop_final"
                ):  # only a committed "stop" may cut NEO off (its own voice is in the mic)
                    await self.interrupt()
                    await bus().set_state(NeoState.IDLE)
                else:
                    continue
            armed = (
                self._armed.is_set()
                or (hit or "").startswith("wake")
                or self._ptt
                or time.time() < self._followup_until
            )
            if not armed:
                continue
            self._armed.clear()
            self.mic.drain()
            text = await self._capture_utterance()
            early().feed(text)
            await early().tick(final=True)  # last clause of the utterance, if it's a quick command
            text = early().remaining()
            early().new_utterance()
            if not text:
                self._followup_until = time.time() + _FOLLOWUP_S if early().executed else 0.0
                await bus().set_state(NeoState.LISTENING if early().executed else NeoState.IDLE)
                continue
            self._handler = asyncio.create_task(self._handle(text))

    async def _handle(self, text: str) -> None:
        try:
            await self._on_text(text)
            self._followup_until = time.time() + _FOLLOWUP_S
            await bus().set_state(NeoState.LISTENING)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 — one bad turn must not kill the voice loop
            print(f"[voice] turn failed: {e}")
            await bus().set_state(NeoState.IDLE)

    async def _capture_utterance(self) -> str:
        """Record one utterance (silero VAD; energy gate as fallback), streaming partial transcripts
        to the overlay while the user is still talking, then return the final transcript."""
        self._capturing = True
        partial_task: asyncio.Task | None = None
        try:
            await bus().set_state(NeoState.LISTENING)
            frames: list[bytes] = []
            started = False
            t0 = time.time()
            last_voice = t0
            last_partial = t0
            noise = 0.004
            vad_buf = np.zeros(0, dtype=np.float32)
            if self._vad is not None:
                self._vad.reset_states()
            while time.time() - t0 < _MAX_UTTER_S:
                f = await self.mic.read()
                if not f:
                    return ""
                ended = False
                if self._vad is not None:
                    vad_buf = np.concatenate(
                        [vad_buf, np.frombuffer(f, dtype=np.int16).astype(np.float32) / 32768.0]
                    )
                    while len(vad_buf) >= _VAD_CHUNK:
                        chunk, vad_buf = vad_buf[:_VAD_CHUNK], vad_buf[_VAD_CHUNK:]
                        import torch

                        ev = self._vad(torch.from_numpy(chunk.copy()), return_seconds=False)
                        if ev and "start" in ev:
                            started = True
                            last_voice = time.time()
                        elif ev and "end" in ev and started and not self._ptt:
                            ended = True
                    if started:
                        frames.append(f)
                        last_voice = time.time()
                else:  # energy gate
                    level = rms(f)
                    if not started:
                        noise = 0.9 * noise + 0.1 * level
                    if level > max(0.012, noise * 3.5):
                        started = True
                        last_voice = time.time()
                    if started:
                        frames.append(f)
                        if time.time() - last_voice > _SILENCE_MS / 1000 and not self._ptt:
                            ended = True
                if ended:
                    break
                if not started and time.time() - t0 > _NO_SPEECH_S and not self._ptt:
                    return ""  # nobody spoke
                # Live partials: re-transcribe the buffer so far, off the loop, at most every ~1 s.
                if (
                    started
                    and time.time() - last_partial > _PARTIAL_EVERY_S
                    and (partial_task is None or partial_task.done())
                ):
                    last_partial = time.time()
                    partial_task = asyncio.create_task(self._emit_partial(b"".join(frames)))
            if not frames:
                return ""
            await bus().set_state(NeoState.THINKING)
            if partial_task and not partial_task.done():
                partial_task.cancel()
            pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            if len(pcm) < MIC_RATE * 0.3:
                return ""
            import mlx.core as mx

            res = await asyncio.to_thread(self._stt.generate, mx.array(pcm))
            return (res.text or "").strip()
        finally:
            self._capturing = False
            if self._wake:
                self._wake.reset()

    async def _emit_partial(self, raw: bytes) -> None:
        try:
            import mlx.core as mx

            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            res = await asyncio.to_thread(self._stt.generate, mx.array(pcm))
            if res.text and self._capturing:
                await bus().say(res.text.strip(), role="user", final=False)
                early().feed(res.text.strip())
                await early().tick()  # act on a finished clause while the user keeps talking
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — partials are cosmetic
            pass
