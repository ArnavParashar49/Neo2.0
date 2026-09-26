"""Wires everything together: tools, session, voice, UI server. `python -m neo` lands here."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from neo.agent import confirm
from neo.agent.session import Session, answer_of
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
            reply = await self.session.handle(text, from_user=True)  # typed, or the user's own speech
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

    async def answer(self, yes: bool, *, action_id: str = "", said: str = "") -> None:
        """The user answered NEO's pending question (the card's buttons, or a typed yes/no)."""
        reply = await self.session.answer(yes, action_id=action_id, text=said or ("yes" if yes else "no"))
        if reply is None:
            await bus().note("That question had already expired — nothing was done.")
            return
        if self.voice and reply.text:
            if hasattr(self.voice, "inform"):
                await self.voice.inform(reply.text)  # Live says how it went, in the user's language
            elif not reply.silent:
                await self.voice.speak(reply.text)

    async def on_command(self, cmd: dict[str, Any]) -> None:
        kind = cmd.get("cmd")
        if kind == "text":
            text = str(cmd.get("text", ""))
            if confirm.peek() is not None and (ans := answer_of(text)) is not None:
                await bus().say(text, role="user")
                await self.answer(ans, said=text)  # typed straight to the question, in any voice mode
            elif self.voice and getattr(self.voice, "handles_text", False):
                await self.voice.send_text(text)  # Live answers aloud + in the transcript
            else:
                await self.handle_text(text)
        elif kind == "confirm":
            # The card's buttons carry the id of the exact action they show: a stale card (or a
            # client that doesn't know the id) can't approve whatever happens to be pending now.
            action_id = str(cmd.get("action_id") or "")
            if not action_id:
                await bus().note("Couldn't tell which question that answer was for — nothing was done.")
                return
            await self.answer(bool(cmd.get("yes")), action_id=action_id)
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
        print(f"[neo] missing: {', '.join(missing)} — allow NEO in System Settings → Privacy & Security (NEO's menu has shortcuts)")
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
    if os.environ.get("NEO_LAUNCHED_BY_APP"):
        asyncio.create_task(_exit_with_parent(stop))
    await stop.wait()
    # Everything below must finish well inside NEO.app's grace period (it SIGKILLs after 12 s).
    if app.voice:
        try:
            await asyncio.wait_for(app.voice.close(), 2)
        except Exception:  # noqa: BLE001
            pass
    try:
        from neo.memory.summarize import summarize_session
        from neo.tools.browser import shutdown as close_browser

        await asyncio.wait_for(close_browser(), 2)
        if s := await asyncio.wait_for(summarize_session(app.session.history, app.started), _SUMMARY_BUDGET_S):
            print(f"[memory] session saved: {s[:100]}")
    except TimeoutError:
        print("[memory] shutdown summary took too long — skipped")
    except Exception as e:  # noqa: BLE001
        print(f"[memory] shutdown summary skipped: {str(e)[:80]}")


_SUMMARY_BUDGET_S = 7.0  # the model gets 5 s inside; this covers the crude fallback + the write


async def _exit_with_parent(stop: asyncio.Event) -> None:
    """Started by NEO.app: if the app goes away without stopping us (force quit, crash), stop
    too — a core with the mic open and nobody to show it must never be left behind."""
    parent = os.getppid()
    while not stop.is_set():
        await asyncio.sleep(1.0)
        if os.getppid() != parent:  # re-parented to launchd: the app is gone
            print("[neo] NEO.app went away — shutting down")
            stop.set()
