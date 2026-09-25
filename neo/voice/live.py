"""Gemini Live voice backend: native speech-to-speech; delegates real work to the agent loop.

Live is the ear + mouth. It gets the cheap, instant tools directly (volume, clock, open app,
timer, memory, calendar) and ONE `agent_task` tool that runs NEO's own loop for anything
multi-step. The session opens on wake and closes after a quiet period to save quota.
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
        for t in list(self._tool_tasks.values()):
            t.cancel()

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

    # ---- session -----------------------------------------------------------------------
    async def _ensure_session(self) -> None:
        async with self._connect_lock:  # concurrent wake/ptt/send_text must share one session
            if self._live:
                return
            s = settings()
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
                live = await cm.__aenter__()
            except Exception as e:  # noqa: BLE001 — quota/network; stay usable, retry on next wake
                print(f"[live] connect failed: {str(e)[:120]}")
                await bus().publish("error", message=f"Voice session failed: {str(e)[:120]}")
                await bus().set_state(NeoState.IDLE)
                return
            self._live_cm, self._live = cm, live
            self._last_activity = time.time()
            self._tasks.append(asyncio.create_task(self._recv_loop(live)))
            self._tasks.append(asyncio.create_task(self._idle_watch()))
            await bus().set_state(NeoState.LISTENING)

    async def _close_session(self) -> None:
        cm, self._live_cm, self._live = self._live_cm, None, None
        if cm:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        await bus().set_state(NeoState.IDLE)

    async def _idle_watch(self) -> None:
        while self._live:
            await asyncio.sleep(5)
            busy = self.spk.playing.is_set() or any(not t.done() for t in self._tool_tasks.values())
            if not busy and time.time() - self._last_activity > _IDLE_CLOSE_S:
                await self._close_session()
                return

    # ---- audio in ---------------------------------------------------------------------
    async def _mic_loop(self) -> None:
        assert self._wake
        while True:
            frame = await self.mic.read()
            if not frame:
                return
            if not self._live:
                if self._wake.feed(frame) == "wake":
                    try:
                        await self.wake("voice")
                    except Exception as e:  # noqa: BLE001
                        print(f"[live] wake failed: {e}")
                continue
            if self.spk.playing.is_set() and self._wake.feed(frame) == "stop":
                await self.interrupt()
            try:
                await self._live.send_realtime_input(
                    audio=gt.Blob(data=frame, mime_type="audio/pcm;rate=16000")
                )
            except Exception as e:  # noqa: BLE001
                print(f"[live] send failed: {e}")
                await self._close_session()

    # ---- server events ----------------------------------------------------------------
    async def _recv_loop(self, live) -> None:
        user_buf: list[str] = []
        model_buf: list[str] = []
        try:
            async for msg in live.receive():
                self._last_activity = time.time()
                sc = msg.server_content
                if sc:
                    if sc.interrupted:
                        self.spk.interrupt()
                        model_buf.clear()
                    if sc.input_transcription and sc.input_transcription.text:
                        user_buf.append(sc.input_transcription.text)
                        await bus().say("".join(user_buf), role="user", final=False)
                    if sc.output_transcription and sc.output_transcription.text:
                        model_buf.append(sc.output_transcription.text)
                        await bus().say("".join(model_buf), final=False)
                    if sc.model_turn:
                        for part in sc.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                await bus().set_state(NeoState.SPEAKING)
                                self.spk.play_pcm16(part.inline_data.data)
                    if sc.turn_complete:
                        if user_buf:
                            await bus().say("".join(user_buf), role="user", final=True)
                            user_buf.clear()
                        if model_buf:
                            await bus().say("".join(model_buf), final=True)
                            model_buf.clear()
                        # Don't block the receive loop on playback — server-side barge-in must land.
                        asyncio.create_task(self._after_playback())
                if msg.tool_call:
                    for fc in msg.tool_call.function_calls or []:
                        task = asyncio.create_task(self._run_tool(live, fc))
                        self._tool_tasks[fc.id or fc.name] = task
                if msg.tool_call_cancellation:
                    for cid in msg.tool_call_cancellation.ids or []:
                        t = self._tool_tasks.pop(cid, None)
                        if t:
                            t.cancel()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[live] receive ended: {str(e)[:120]}")
            if self._live is live:
                await self._close_session()

    async def _after_playback(self) -> None:
        await self.spk.wait_done()
        if self._live and bus().state == NeoState.SPEAKING:
            await bus().set_state(NeoState.LISTENING)

    async def _run_tool(self, live, fc) -> None:
        args = dict(fc.args or {})
        key = fc.id or fc.name
        try:
            if fc.name == "agent_task":
                await bus().set_state(NeoState.WORKING)
                reply = await self._agent_session.handle(str(args.get("goal", "")))
                result = reply.text
            else:
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
        else:  # session went away mid-task: don't lose the result
            await bus().say(result[:600], final=True)
