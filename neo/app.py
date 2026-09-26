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
        import time

        self.started = time.time()
        self.session = Session()
        self.voice = None  # set in run() when a backend is available
        self._tasks: set[asyncio.Task] = set()

    async def handle_text(self, text: str, *, speak: bool = True) -> None:
        """Handle one request. Several may be in flight at once (the session runs them as jobs)."""
        self.session.memory_context = await asyncio.to_thread(store().prompt_context, text)
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        try:
            reply = await self.session.handle(text)
            if speak and self.voice and reply.text and not reply.silent:
                await self.voice.speak(reply.text)
        except asyncio.CancelledError:
            await bus().say("Stopped.", final=True)
        finally:
            if task:
                self._tasks.discard(task)

    def cancel_current(self) -> bool:
        """Stop everything in flight (the stop button / "stop")."""
        n = 0
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
                n += 1
        return n > 0

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


def _log_activity(ev) -> None:
    """One line per tool call / route decision in the core log (what the REPL's -v shows)."""
    d = ev.data
    if ev.type == "tool_start":
        print(f"  ▶ {d['name']} {str(d.get('args', {}))[:120]}")
    elif ev.type == "tool_end":
        print(f"  ◀ {d['name']} {'ok' if d['ok'] else 'FAIL'}: {str(d.get('summary', ''))[:140]}")
    elif ev.type == "reflex":
        print(f"  ~ {d['intent']} ({d['intent_confidence']:.2f}) via {d['source']} {d['latency_ms']:.0f}ms")
    elif ev.type == "turn":
        print(
            f"  = {d['route']} via {d.get('model') or d.get('brain') or '-'} in {d['ms']}ms, {d['tools']} tools"
        )


async def _lag_monitor() -> None:
    """Log when the event loop stalls — a stalled loop silently kills websockets (keepalive)."""
    import time

    while True:
        t0 = time.monotonic()
        await asyncio.sleep(0.5)
        lag = time.monotonic() - t0 - 0.5
        if lag > 0.4:
            print(f"[loop] stalled for {lag:.1f}s — something blocked the event loop")


async def _cocoa_pump() -> None:
    """Let Cocoa deliver its notifications. NSWorkspace's running-app list and frontmost app are
    only refreshed by the main thread's run loop, which an asyncio process never runs — without
    this NEO can't see apps launched after it started, or which one is in front."""
    try:
        from AppKit import NSDate, NSRunLoop
    except ImportError:  # not macOS
        return
    while True:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.002))
        await asyncio.sleep(0.2)


async def run() -> None:
    s = settings()
    from neo.permissions import request_missing

    if missing := request_missing():
        print(f"[neo] asked macOS for: {', '.join(missing)} — allow NEO in System Settings → Privacy & Security")
    load_all()
    app = App()
    ui = UIServer(app.on_command)
    await ui.start()
    warmup(app.session.reflex)  # Laya, then the memory embedder, sequentially

    if s.voice != "off":
        try:
            from neo.voice.pipeline import make_voice

            app.voice = await make_voice(app.handle_text, app.session)
        except Exception as e:  # noqa: BLE001 — text-only is still useful
            print(f"[voice] unavailable: {e}")

    await bus().set_state(NeoState.IDLE)
    asyncio.create_task(_lag_monitor())
    asyncio.create_task(_cocoa_pump())
    from neo.reflex.learn import learn_loop

    def _idle() -> bool:
        from neo.events import bus

        voice_busy = bool(app.voice and getattr(app.voice, "_live", None))
        return not voice_busy and not bus().active_jobs and not app._tasks

    asyncio.create_task(learn_loop(_idle))
    bus().subscribe(_log_activity)
    print("NEO is running. Say 'Hey Neo', or type in the overlay.")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    if app.voice:
        await app.voice.close()
    try:
        from neo.memory.summarize import summarize_session
        from neo.tools.browser import shutdown as close_browser

        await close_browser()
        if s := await summarize_session(app.session.history, app.started):
            print(f"[memory] session saved: {s[:100]}")
    except Exception as e:  # noqa: BLE001
        print(f"[memory] shutdown summary skipped: {str(e)[:80]}")
