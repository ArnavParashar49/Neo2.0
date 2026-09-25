"""Wires everything together: tools, session, voice, UI server. `python -m neo` lands here."""

from __future__ import annotations

import asyncio
import signal
from typing import Any

from neo.agent.session import Session
from neo.config import settings
from neo.events import NeoState, bus
from neo.memory.store import store
from neo.reflex import warmup
from neo.server.ws import UIServer
from neo.tools import load_all


class App:
    def __init__(self) -> None:
        self.session = Session()
        self.voice = None  # set in run() when a backend is available
        self._busy = asyncio.Lock()
        self._current: asyncio.Task | None = None

    async def handle_text(self, text: str, *, speak: bool = True) -> None:
        async with self._busy:
            self._current = asyncio.current_task()
            try:
                self.session.memory_context = store().prompt_context(text)
                reply = await self.session.handle(text)
                if speak and self.voice and reply.text:
                    await self.voice.speak(reply.text)
            except asyncio.CancelledError:
                await bus().say("Stopped.", final=True)
                await bus().set_state(NeoState.IDLE)
            finally:
                self._current = None

    def cancel_current(self) -> bool:
        if self._current and not self._current.done():
            self._current.cancel()
            return True
        return False

    async def on_command(self, cmd: dict[str, Any]) -> None:
        kind = cmd.get("cmd")
        if kind == "text":
            text = str(cmd.get("text", ""))
            if self.voice and getattr(self.voice, "handles_text", False):
                await self.voice.send_text(text)  # Live answers aloud + in the transcript
            else:
                await self.handle_text(text)
        elif kind == "confirm":
            await self.handle_text("yes" if cmd.get("yes") else "no")
        elif kind == "stop":
            if self.voice:
                await self.voice.interrupt()
            if not self.cancel_current():
                await bus().set_state(NeoState.IDLE)
        elif kind == "ptt":
            if self.voice:
                await self.voice.push_to_talk(bool(cmd.get("on")))
        elif kind == "wake":
            if self.voice:
                await self.voice.wake("ui")


async def run() -> None:
    s = settings()
    load_all()
    app = App()
    ui = UIServer(app.on_command)
    await ui.start()
    warmup(app.session.reflex)

    if s.voice != "off":
        try:
            from neo.voice.pipeline import make_voice

            app.voice = await make_voice(app.handle_text, app.session)
        except Exception as e:  # noqa: BLE001 — text-only is still useful
            print(f"[voice] unavailable: {e}")

    await bus().set_state(NeoState.IDLE)
    print("NEO is running. Say 'Hey Neo', or type in the overlay.")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    if app.voice:
        await app.voice.close()
