"""Gemini Live voice backend: native speech-to-speech; delegates real work to the agent loop.

Live is the ear + mouth. It gets the cheap, instant tools directly (volume, clock, open app,
timer, memory, calendar) and ONE `agent_task` tool that runs NEO's own loop for anything
multi-step. The session opens on wake and closes after a quiet period to save quota.

Design notes that came out of real use:
- Half-duplex: while NEO's own audio is playing (plus a short tail) the mic is NOT streamed to
  Live, otherwise the model hears itself, interrupts itself and the transcript fills with echoes.
  The local wake/stop spotter keeps listening during that time so "stop" still works.
- One state machine (`_refresh_state`) decides what the orb shows — speaking > working >
  listening — instead of every event racing to set it. The agent loop is told not to drop to
  idle while a voice session owns the orb.
- The websocket keepalive is relaxed: Google's endpoint is slow to answer pings while it is
  synthesising audio, and the default 20 s timeout killed every session mid-conversation.
- An unexpected disconnect during an active conversation reconnects once, silently.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable

import numpy as np
from google import genai
from google.genai import types as gt

from neo.agent.early import early
from neo.agent.prompt import build as build_prompt
from neo.agent.registry import ToolContext, registry
from neo.config import settings
from neo.events import NeoState, bus
from neo.voice.audio import Mic, Speaker, rms
from neo.voice.wake import WakeSpotter

TextHandler = Callable[[str], Awaitable[None]]

_DIRECT_TOOLS = {
    "volume",
    "brightness",
    "clock",
    "timer",
    "open_app",
    "memory",
    "calendar_today",
    "web_search",
    "reminder_add",
    "apps_running",
    "mail_unread",
    "notes_create",
    "safari_open",
}
_IDLE_CLOSE_S = 90.0
_CONNECT_TIMEOUT_S = 12.0
_RECONNECT_WINDOW_S = 60.0  # reopen automatically if the socket dies within this long of activity
_PLAYBACK_TAIL_S = 0.35  # keep the mic muted this long after NEO stops talking (room echo)
_BARGE_IN_GRACE_S = 1.5  # ignore "stop" for the first moment of NEO's own speech (echo onset)
_TURN_STALL_S = 20.0  # an unanswered turn keeps the session open this long, then silence rules apply
_VAD_CHUNK = 512  # silero works on 32 ms windows at 16 kHz

_LIVE_EXTRA = """You are speaking aloud, so keep replies short and natural. For anything that needs
several steps or looking at the screen, files, mail or the web, call agent_task with a clear goal and
then relay its result in one or two sentences. Don't narrate tool use. When the user is working in
an app that is already open (a note, a document, a message) and asks you to type or add something,
continue in that same place via agent_task — don't create a new note or file elsewhere."""


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _declarations() -> list[gt.FunctionDeclaration]:
    decls = []
    for t in registry().all():
        if t.name in _DIRECT_TOOLS:
            decls.append(
                gt.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                    # Slow tools don't block the conversation: the model acknowledges, keeps
                    # listening, and gets the result later (scheduled WHEN_IDLE).
                    behavior=gt.Behavior.NON_BLOCKING if t.slow else gt.Behavior.BLOCKING,
                )
            )
    decls.append(
        gt.FunctionDeclaration(
            name="agent_task",
            description="Hand a multi-step or on-screen task to NEO's agent (it can see the screen, control apps, "
            "read/write files, browse, send mail). Returns a short result to relay.",
            behavior=gt.Behavior.NON_BLOCKING,
            parameters_json_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
            },
        )
    )
    return decls


class LiveVoice:
    handles_text = True  # typed text is sent into the Live session so it answers aloud

    def __init__(self, on_text: TextHandler, session_for_agent) -> None:
        self._on_text = on_text
        self._agent_session = session_for_agent
        s = settings()
        self._client = genai.Client(api_key=s.gemini_api_key)
        # websockets kwargs ride along with the SDK's ssl context dict (see google.genai.live).
        self._client._api_client._websocket_ssl_ctx.update({"ping_interval": 20, "ping_timeout": 120})
        self.mic = Mic()
        self.spk = Speaker()
        self._wake: WakeSpotter | None = None
        self._live = None
        self._live_cm = None
        self._connect_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._tool_tasks: dict[str, asyncio.Task] = {}  # function-call id → running tool task
        self._goals: dict[str, str] = {}  # function-call id → agent_task goal in flight
        self._last_activity = 0.0
        self._ptt = False
        self._speaking = False  # audio for the current model turn is playing / still arriving
        self._closing = False  # a close we asked for (idle timeout / shutdown) → don't reconnect
        self._speaking_since = 0.0
        self._vad = None  # silero: local "is the user talking?" so we know when to stop listening
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._last_user_speech = 0.0
        self._last_turn_end = 0.0  # when the model last finished a turn or got a tool result
        self._turn_open = False  # the user has said something the model hasn't finished answering
        self._session_open = 0.0

    async def start(self) -> None:
        self._wake = await asyncio.to_thread(WakeSpotter)
        try:
            from silero_vad import load_silero_vad

            self._vad = await asyncio.to_thread(load_silero_vad)
        except Exception as e:  # noqa: BLE001 — fall back to the energy gate
            print(f"[live] silero VAD unavailable ({e}); using energy gate for the listen timeout")
        self.mic.start()
        self.spk.start()
        self._tasks.append(asyncio.create_task(self._mic_loop()))

    async def probe(self) -> None:
        """Open + close a session once to verify the free tier allows Live."""
        cfg = gt.LiveConnectConfig(response_modalities=["TEXT"])
        async with self._client.aio.live.connect(model=settings().gemini_live_model, config=cfg):
            pass

    async def close(self) -> None:
        self._closing = True
        for t in self._tasks + list(self._tool_tasks.values()):
            t.cancel()
        await self._close_session()
        self.mic.stop()
        self.spk.close()

    # ---- controls ----------------------------------------------------------------------
    async def wake(self, source: str = "ui") -> None:
        await self._ensure_session()

    async def push_to_talk(self, on: bool) -> None:
        self._ptt = on
        if on:
            await self.wake("ptt")

    async def interrupt(self) -> None:
        """Stop playback and cancel any agent task the model kicked off."""
        self.spk.interrupt()
        self._speaking = False
        for t in list(self._tool_tasks.values()):
            t.cancel()
        await self._refresh_state()

    async def speak(self, text: str) -> None:  # Live speaks for itself
        return

    async def send_text(self, text: str) -> None:
        await self._ensure_session()
        if not self._live:
            await self._on_text(text)  # Live is down: fall back to the text session
            return
        await self._live.send_client_content(
            turns=gt.Content(role="user", parts=[gt.Part.from_text(text=text)])
        )
        self._last_activity = self._last_user_speech = time.time()  # typed = said
        self._turn_open = True

    # ---- state machine -----------------------------------------------------------------
    async def _refresh_state(self) -> None:
        """The orb follows the conversation: speaking > working (agent busy) > listening > idle."""
        if not self._live:
            await bus().set_state(NeoState.IDLE)
        elif self._speaking or self.spk.playing.is_set():
            await bus().set_state(NeoState.SPEAKING)
        elif any(not t.done() for t in self._tool_tasks.values()):
            pass  # the agent loop is narrating (thinking / working / searching)
        else:
            await bus().set_state(NeoState.LISTENING)

    # ---- session -----------------------------------------------------------------------
    async def _ensure_session(self) -> None:
        async with self._connect_lock:  # concurrent wake/ptt/send_text must share one session
            if self._live:
                return
            s = settings()
            self._closing = False
            await bus().set_state(NeoState.CONNECTING)
            cfg = gt.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=build_prompt(self._agent_session.memory_context, _LIVE_EXTRA),
                tools=[gt.Tool(function_declarations=_declarations())],
                speech_config=gt.SpeechConfig(
                    voice_config=gt.VoiceConfig(
                        prebuilt_voice_config=gt.PrebuiltVoiceConfig(voice_name=s.live_voice)
                    )
                ),
                input_audio_transcription=gt.AudioTranscriptionConfig(),
                output_audio_transcription=gt.AudioTranscriptionConfig(),
                realtime_input_config=gt.RealtimeInputConfig(
                    automatic_activity_detection=gt.AutomaticActivityDetection(
                        silence_duration_ms=700, prefix_padding_ms=200
                    )
                ),
            )
            cm = self._client.aio.live.connect(model=s.gemini_live_model, config=cfg)
            try:
                live = await asyncio.wait_for(cm.__aenter__(), _CONNECT_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001 — quota/network/throttle; stay usable, retry on next wake
                why = "timed out" if isinstance(e, TimeoutError) else str(e)[:120]
                print(f"[live] connect failed: {why}")
                await bus().note(f"Voice session failed ({why[:60]}) — say “Hey Neo” to retry")
                await bus().set_state(NeoState.IDLE)
                return
            self._live_cm, self._live = cm, live
            self._last_activity = self._session_open = self._last_user_speech = time.time()
            if self._vad is not None:
                self._vad.reset_states()
            self._vad_buf = np.zeros(0, dtype=np.float32)
            self._agent_session.voice_owned = True
            self._tasks.append(asyncio.create_task(self._recv_loop(live)))
            self._tasks.append(asyncio.create_task(self._idle_watch(live)))
            self._tasks.append(asyncio.create_task(self._silence_watch(live)))
            self._tasks.append(asyncio.create_task(self._early_ticker(live)))
            await self._refresh_state()

    async def _close_session(self) -> None:
        cm, self._live_cm, self._live = self._live_cm, None, None
        self._speaking = False
        self._agent_session.voice_owned = False
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        await bus().set_state(NeoState.IDLE)

    async def _lost(self, live, why: str) -> None:
        """The socket died under us. Reconnect if the user was mid-conversation."""
        if self._live is not live:
            return
        recent = time.time() - self._last_activity < _RECONNECT_WINDOW_S
        await self._close_session()
        if recent and not self._closing:
            print(f"[live] {why}; reconnecting")
            await bus().note("Voice session dropped — reconnecting")
            await self._ensure_session()
        else:
            print(f"[live] {why}")

    async def _idle_watch(self, live) -> None:
        while self._live is live:
            await asyncio.sleep(5)
            busy = self.spk.playing.is_set() or any(not t.done() for t in self._tool_tasks.values())
            if not busy and time.time() - self._last_activity > _IDLE_CLOSE_S:
                self._closing = True
                await self._close_session()
                return

    # ---- audio in ---------------------------------------------------------------------
    async def _early_ticker(self, live) -> None:
        """Let the early actor notice a clause that has gone quiet even if no new words arrive."""
        while self._live is live:
            await asyncio.sleep(0.15)
            try:
                await early().tick()
            except Exception as e:  # noqa: BLE001
                print(f"[early] {str(e)[:80]}")

    async def _silence_watch(self, live) -> None:
        """Stop listening after `listen_timeout_s` of no speech from the user (background noise
        doesn't count — silero decides what is speech). The wake word re-arms the session."""
        timeout = settings().listen_timeout_s
        while self._live is live:
            await asyncio.sleep(0.25)
            if self._ptt or self._speaking or self.spk.playing.is_set():
                continue
            if any(not t.done() for t in self._tool_tasks.values()):
                continue
            if self._turn_open and time.time() - self._last_activity < _TURN_STALL_S:
                continue  # the model is still composing its answer
            if time.time() - self._quiet_since() > timeout:
                print(f"[live] no speech for {timeout:.0f}s; closing session")
                self._closing = True
                await self._close_session()
                return

    def _quiet_since(self) -> float:
        """The last moment either side of the conversation did something."""
        return max(
            self._last_user_speech, self.spk.last_stop, self._session_open, self._last_turn_end
        )

    def _hear(self, frame: bytes) -> None:
        """Feed the local VAD; remember when the user last spoke."""
        if self._vad is None:
            if rms(frame) > 0.02:
                self._last_user_speech = time.time()
            return
        self._vad_buf = np.concatenate(
            [self._vad_buf, np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0]
        )
        while len(self._vad_buf) >= _VAD_CHUNK:
            chunk, self._vad_buf = self._vad_buf[:_VAD_CHUNK], self._vad_buf[_VAD_CHUNK:]
            try:
                import torch

                if float(self._vad(torch.from_numpy(chunk.copy()), 16000).item()) > 0.6:
                    self._last_user_speech = time.time()
            except Exception:  # noqa: BLE001
                pass

    def _mic_muted(self) -> bool:
        """Half-duplex: don't feed NEO its own voice."""
        if self._ptt:
            return False
        return (
            self._speaking
            or self.spk.playing.is_set()
            or (time.time() - self.spk.last_stop) < _PLAYBACK_TAIL_S
        )

    async def _mic_loop(self) -> None:
        assert self._wake
        while True:
            frame = await self.mic.read()
            if not frame:
                return
            hit = self._wake.feed(frame)
            if not self._live:
                if hit and hit.startswith("wake"):
                    try:
                        await self.wake("voice")
                    except Exception as e:  # noqa: BLE001
                        print(f"[live] wake failed: {e}")
                continue
            if self._mic_muted():
                # NEO's own voice is in the mic now. Only a *committed* "stop", after the first
                # moment of speech, may cut it off — partial hits and "neo" are its own echo.
                if hit == "stop_final" and time.time() - self._speaking_since > _BARGE_IN_GRACE_S:
                    print("[live] barge-in: user said stop")
                    await self.interrupt()
                continue
            self._hear(frame)
            try:
                await self._live.send_realtime_input(
                    audio=gt.Blob(data=frame, mime_type="audio/pcm;rate=16000")
                )
            except Exception as e:  # noqa: BLE001
                await self._lost(self._live, f"send failed: {str(e)[:100]}")

    # ---- server events ----------------------------------------------------------------
    async def _recv_loop(self, live) -> None:
        user_buf: list[str] = []
        model_buf: list[str] = []
        try:
            # `receive()` yields ONE model turn and then stops (it is meant to be called per turn).
            # Re-enter it for as long as the session lives; a real disconnect raises instead.
            while self._live is live:
                async for msg in live.receive():
                    await self._on_message(msg, user_buf, model_buf, live)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            await self._lost(live, f"receive ended: {str(e)[:100]}")

    async def _on_message(self, msg, user_buf: list[str], model_buf: list[str], live) -> None:
        self._last_activity = time.time()
        sc = msg.server_content
        if sc:
            if sc.interrupted:
                print("[live] server-side interruption")
                self.spk.interrupt()
                self._speaking = False
                model_buf.clear()
                await self._refresh_state()
            if sc.input_transcription and sc.input_transcription.text:
                self._turn_open = True
                user_buf.append(sc.input_transcription.text)
                await bus().say("".join(user_buf), role="user", final=False)
                early().feed("".join(user_buf))
                await early().tick()
            interim = getattr(sc, "interim_input_transcription", None)
            if interim and interim.text:  # words as they're being said, before they're committed
                early().feed("".join(user_buf) + " " + interim.text)
                await early().tick()
            if sc.output_transcription and sc.output_transcription.text:
                model_buf.append(sc.output_transcription.text)
                await bus().say("".join(model_buf), final=False)
            if sc.model_turn:
                for part in sc.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        if not self._speaking:
                            self._speaking = True
                            self._speaking_since = time.time()
                            await self._refresh_state()
                        self.spk.play_pcm16(part.inline_data.data)
            if sc.turn_complete:
                self._turn_open = False
                self._last_turn_end = time.time()
                if user_buf:
                    await bus().say("".join(user_buf), role="user", final=True)
                    user_buf.clear()
                await early().tick(final=True)
                early().new_utterance()
                if model_buf:
                    await bus().say("".join(model_buf), final=True)
                    model_buf.clear()
                # Don't block the receive loop on playback — server-side events must keep flowing.
                asyncio.create_task(self._after_playback())
        if msg.tool_call:
            for fc in msg.tool_call.function_calls or []:
                task = asyncio.create_task(self._run_tool(live, fc))
                self._tool_tasks[fc.id or fc.name] = task
            await self._refresh_state()
        if msg.tool_call_cancellation:
            for cid in msg.tool_call_cancellation.ids or []:
                t = self._tool_tasks.pop(cid, None)
                if t:
                    t.cancel()
        if msg.go_away:
            print("[live] server asked us to go away; will reconnect on next activity")

    async def _after_playback(self) -> None:
        await self.spk.wait_done()
        await asyncio.sleep(0.4)  # let the tail settle so the orb doesn't blink between sentences
        if not self.spk.playing.is_set():
            self._speaking = False
            await self._refresh_state()

    async def _run_tool(self, live, fc) -> None:
        args = dict(fc.args or {})
        key = fc.id or fc.name
        try:
            if fc.name == "agent_task":
                goal = str(args.get("goal", "")).strip()
                if self._agent_task_running(goal, except_key=key):
                    # The model re-asked for the same thing (it got an error, or the user repeated
                    # themselves) while the first attempt is still working: never run it twice.
                    result = "Still working on that — I'll tell you when it's done."
                else:
                    self._goals[key] = goal
                    reply = await self._agent_session.handle(goal)
                    result = reply.text
            elif (done := early().recently_done(fc.name, args)) is not None:
                result = done  # already ran while the user was still talking
            else:
                await bus().set_state(NeoState.WORKING, job=key)
                await bus().tool_start(fc.name, args, job=key)
                out = await registry().invoke(fc.name, args, ToolContext(user_text=str(args)))
                await bus().tool_end(fc.name, out.ok, out.text, job=key)
                result = out.text
        except asyncio.CancelledError:
            result = "cancelled by the user"
        except Exception as e:  # noqa: BLE001 — the model must always get a response back
            result = f"Error: {e}"
        finally:
            self._tool_tasks.pop(key, None)
            self._goals.pop(key, None)
            await bus().end_job(key)
        self._last_activity = time.time()
        t = registry().get(fc.name)
        non_blocking = fc.name == "agent_task" or bool(t and t.slow)
        resp = gt.FunctionResponse(
            id=fc.id,
            name=fc.name,
            response={"result": result[:4000]},
            # For non-blocking calls, say the result when the user isn't talking.
            scheduling=gt.FunctionResponseScheduling.WHEN_IDLE if non_blocking else None,
        )
        if self._live is live:
            try:
                await live.send_tool_response(function_responses=[resp])
                self._last_turn_end = time.time()
                self._turn_open = True  # the model owes a spoken reply for this result
            except Exception as e:  # noqa: BLE001
                print(f"[live] tool response failed: {e}")
            # The model now composes its spoken reply: show "thinking" until audio arrives
            # instead of blinking through "listening" for the half-second in between.
            if not any(not t.done() for t in self._tool_tasks.values()):
                await bus().set_state(NeoState.THINKING)
                asyncio.create_task(self._settle_if_quiet(2.5))
        else:  # session went away mid-task: don't lose the result
            await bus().say(result[:600], final=True)
            await self._refresh_state()

    def _agent_task_running(self, goal: str, *, except_key: str) -> bool:
        g = _canon(goal)
        return any(
            k != except_key and _canon(v) == g and not self._tool_tasks[k].done()
            for k, v in self._goals.items()
            if k in self._tool_tasks
        )

    async def _settle_if_quiet(self, after_s: float) -> None:
        await asyncio.sleep(after_s)
        if not self._speaking and not self.spk.playing.is_set():
            await self._refresh_state()
