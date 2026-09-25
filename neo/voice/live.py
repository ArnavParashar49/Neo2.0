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
import time
from collections.abc import Awaitable, Callable

from google import genai
from google.genai import types as gt

from neo.agent.prompt import build as build_prompt
from neo.agent.registry import ToolContext, registry
from neo.config import settings
from neo.events import NeoState, bus
from neo.voice.audio import Mic, Speaker
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
}
_IDLE_CLOSE_S = 90.0
_CONNECT_TIMEOUT_S = 12.0
_RECONNECT_WINDOW_S = 60.0  # reopen automatically if the socket dies within this long of activity
_PLAYBACK_TAIL_S = 0.35  # keep the mic muted this long after NEO stops talking (room echo)

_LIVE_EXTRA = """You are speaking aloud, so keep replies short and natural. For anything that needs
several steps or looking at the screen, files, mail or the web, call agent_task with a clear goal and
then relay its result in one or two sentences. Don't narrate tool use."""


def _declarations() -> list[gt.FunctionDeclaration]:
    decls = []
    for t in registry().all():
        if t.name in _DIRECT_TOOLS:
            decls.append(
                gt.FunctionDeclaration(
                    name=t.name, description=t.description, parameters_json_schema=t.parameters
                )
            )
    decls.append(
        gt.FunctionDeclaration(
            name="agent_task",
            description="Hand a multi-step or on-screen task to NEO's agent (it can see the screen, control apps, "
            "read/write files, browse, send mail). Returns a short result to relay.",
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
        self._last_activity = 0.0
        self._ptt = False
        self._speaking = False  # audio for the current model turn is playing / still arriving
        self._closing = False  # a close we asked for (idle timeout / shutdown) → don't reconnect

    async def start(self) -> None:
        self._wake = await asyncio.to_thread(WakeSpotter)
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
        self._last_activity = time.time()

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
            self._last_activity = time.time()
            self._agent_session.voice_owned = True
            self._tasks.append(asyncio.create_task(self._recv_loop(live)))
            self._tasks.append(asyncio.create_task(self._idle_watch(live)))
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
                if hit == "wake":
                    try:
                        await self.wake("voice")
                    except Exception as e:  # noqa: BLE001
                        print(f"[live] wake failed: {e}")
                continue
            if self._mic_muted():
                if hit == "stop" or hit == "wake":
                    await self.interrupt()
                continue
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
                self.spk.interrupt()
                self._speaking = False
                model_buf.clear()
                await self._refresh_state()
            if sc.input_transcription and sc.input_transcription.text:
                user_buf.append(sc.input_transcription.text)
                await bus().say("".join(user_buf), role="user", final=False)
            if sc.output_transcription and sc.output_transcription.text:
                model_buf.append(sc.output_transcription.text)
                await bus().say("".join(model_buf), final=False)
            if sc.model_turn:
                for part in sc.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        if not self._speaking:
                            self._speaking = True
                            await self._refresh_state()
                        self.spk.play_pcm16(part.inline_data.data)
            if sc.turn_complete:
                if user_buf:
                    await bus().say("".join(user_buf), role="user", final=True)
                    user_buf.clear()
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
                reply = await self._agent_session.handle(str(args.get("goal", "")))
                result = reply.text
            else:
                await bus().set_state(NeoState.WORKING)
                out = await registry().invoke(fc.name, args, ToolContext(user_text=str(args)))
                result = out.text
        except asyncio.CancelledError:
            result = "cancelled by the user"
        except Exception as e:  # noqa: BLE001 — the model must always get a response back
            result = f"Error: {e}"
        finally:
            self._tool_tasks.pop(key, None)
        self._last_activity = time.time()
        resp = gt.FunctionResponse(id=fc.id, name=fc.name, response={"result": result[:4000]})
        if self._live is live:
            try:
                await live.send_tool_response(function_responses=[resp])
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

    async def _settle_if_quiet(self, after_s: float) -> None:
        await asyncio.sleep(after_s)
        if not self._speaking and not self.spk.playing.is_set():
            await self._refresh_state()
