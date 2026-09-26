"""Wires everything together: tools, session, voice, UI server. `python -m neo` lands here."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from neo import health
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
        if self.voice and reply.text and not reply.silent:  # approved and done: nothing to say
            if hasattr(self.voice, "inform"):
                await self.voice.inform(reply.text)  # Live says what went wrong, in the user's language
            else:
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
        elif kind == "forget_all":
            # From the overlay's own "Erase everything?" confirmation (never a model's tool).
            from neo.privacy import forget_everything

            await self.session.cancel_all()
            self.session.history.clear()
            self.session._summarized = 0
            self.session.memory_context = ""
            if self.voice is not None and hasattr(self.voice, "_dialog"):
                self.voice._dialog.clear()  # the spoken conversation it keeps for context
            counts = await forget_everything()
            print("[privacy] erased everything NEO kept about the user")
            await bus().publish("erased", counts=counts)
            await bus().note("Erased: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
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


# Argument values safe to keep in a log file people share for troubleshooting: app names, keys,
# settings — never typed text, message bodies, commands, queries or file contents.
_LOGGABLE = {"name", "app", "site", "direction", "level", "keys", "minutes", "action", "kind", "news", "button", "count", "id", "role", "new_tab", "send", "submit"}


def _safe_args(args: dict) -> str:
    from urllib.parse import urlsplit

    out = []
    for k, v in (args or {}).items():
        s = str(v)
        if k == "url" or (k == "target" and "://" in s):
            out.append(f"{k}=<{urlsplit(s).hostname or 'link'}>")
        elif k == "path" or (k == "target" and s.startswith(("/", "~"))):
            out.append(f"{k}=<file>")
        elif k in _LOGGABLE or k == "target":
            out.append(f"{k}={s[:40]!r}")
        else:
            out.append(f"{k}=<{len(s)} chars>")
    return " ".join(out)


def _log_activity(ev) -> None:
    """One line per tool call / route decision in the core log — what ran and whether it worked,
    never what was typed, read or sent."""
    d = ev.data
    if ev.type == "tool_start":
        print(f"  ▶ {d['name']} {_safe_args(d.get('args', {}))}")
    elif ev.type == "tool_end":
        summary = str(d.get("summary", ""))
        detail = f": {summary[:80]}" if not d["ok"] else f" ({len(summary)} chars)"
        print(f"  ◀ {d['name']} {'ok' if d['ok'] else 'FAIL'}{detail}")
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

    # Signals first: quitting NEO during a first-run download must still shut down cleanly.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    if os.environ.get("NEO_LAUNCHED_BY_APP"):
        asyncio.create_task(_exit_with_parent(stop))

    if missing := request_missing():
        print(f"[neo] missing: {', '.join(missing)} — allow NEO in System Settings → Privacy & Security (NEO's menu has shortcuts)")
    load_all()
    app = App()
    ui = UIServer(app.on_command)
    await ui.start()
    warmup(app.session.reflex)  # Laya, then the memory embedder, sequentially
    from neo.tools.system import restore_timers

    restore_timers()  # timers set before a restart still go off

    # Text works from here on; voice comes up alongside (it may be downloading models).
    await bus().set_state(NeoState.IDLE)
    asyncio.create_task(_lag_monitor())
    asyncio.create_task(_cocoa_pump())
    if s.voice != "off":
        asyncio.create_task(_voice_supervisor(app, stop))
    from neo.reflex.learn import learn_loop

    def _idle() -> bool:
        from neo.events import bus

        voice_busy = bool(app.voice and getattr(app.voice, "_live", None))
        return not voice_busy and not bus().active_jobs and not app._tasks

    asyncio.create_task(learn_loop(_idle))
    asyncio.create_task(_checkpoint_loop(app, _idle))
    bus().subscribe(_log_activity)
    print("NEO is running. Say 'Hey Neo', or type in the overlay.")
    await stop.wait()
    # Everything below must finish well inside NEO.app's grace period (it SIGKILLs after 12 s).
    if app.voice:
        try:
            await asyncio.wait_for(app.voice.close(), 2)
        except Exception:  # noqa: BLE001
            pass
    try:
        from neo.tools.browser import shutdown as close_browser

        await asyncio.wait_for(close_browser(), 2)
        if s := await asyncio.wait_for(app.session.checkpoint(final=True, budget_s=5.0), _SUMMARY_BUDGET_S):
            print(f"[memory] session saved: {s[:100]}")
    except TimeoutError:
        print("[memory] shutdown summary took too long — skipped")
    except Exception as e:  # noqa: BLE001
        print(f"[memory] shutdown summary skipped: {str(e)[:80]}")


_CHECKPOINT_EVERY_S = 20 * 60


async def _checkpoint_loop(app: App, is_idle) -> None:
    """Summarise the conversation every so often while idle, so a crash or a power cut loses at
    most the last stretch instead of the whole day."""
    while True:
        await asyncio.sleep(_CHECKPOINT_EVERY_S)
        if is_idle():
            try:
                await app.session.checkpoint()
            except Exception as e:  # noqa: BLE001
                print(f"[memory] checkpoint skipped: {str(e)[:80]}")


_VOICE_RETRY_S = 60.0
_LIVE_RETRY_S = {"quota": 3600.0, "network": 180.0, "other": 600.0}


def _say_aloud(text: str) -> None:
    """Last-resort speech when no voice backend can talk (macOS's own voice)."""
    import subprocess

    try:
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


async def _voice_supervisor(app: App, stop: asyncio.Event) -> None:
    """Bring voice up, and keep it up: retry if it can't start, and when Gemini Live keeps
    failing (quota used up, network down) say so and switch to the on-device voice until Live
    works again."""
    from neo.voice import devices
    from neo.voice.pipeline import local_ready, make_local, make_voice

    from neo import health

    if devices.mic_status() in (devices.DENIED, devices.RESTRICTED):
        print("[voice] microphone access is off for NEO")
    down: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    async def on_down(kind: str, why: str) -> None:
        await down.put((kind, why))

    while not stop.is_set():
        if app.voice is None:
            try:
                app.voice = await make_voice(app.handle_text, app.session)
                health.report("voice", None)
            except Exception as e:  # noqa: BLE001
                print(f"[voice] unavailable: {str(e)[:160]}")
                health.report("voice", f"Voice is off ({str(e)[:80]}). You can type; NEO keeps trying.")
                await asyncio.sleep(_VOICE_RETRY_S)
                continue
            if hasattr(app.voice, "on_down"):
                app.voice.on_down = on_down
        kind, why = await down.get()
        live = app.voice
        reason = "today's free voice minutes are used up" if kind == "quota" else "it can't reach Google right now"
        if local_ready():
            print(f"[voice] Gemini Live is down ({why}); switching to the on-device voice")
            try:
                await asyncio.wait_for(live.close(), 3)
            except Exception:  # noqa: BLE001
                pass
            app.voice = None
            try:
                app.voice = await make_local(app.handle_text)
                health.report("voice", f"Gemini Live is unavailable ({reason}) — using the on-device voice for now.")
                await app.voice.speak(f"Online voice is unavailable because {reason}. I'm using the on-device voice for now.")
            except Exception as e:  # noqa: BLE001
                print(f"[voice] on-device voice failed too: {str(e)[:120]}")
                app.voice = None  # back to Live on the next pass: it may work on the next wake
                _say_aloud(f"Voice is unavailable because {reason}. You can still type to me.")
                continue
            asyncio.create_task(_back_to_live(app, stop, kind, on_down))
        else:
            health.report("voice", f"Voice is unavailable: {reason}. You can still type; NEO keeps trying.")
            _say_aloud(f"Voice is unavailable because {reason}. You can still type to me.")


async def _back_to_live(app: App, stop: asyncio.Event, kind: str, on_down) -> None:
    """While on the on-device voice, check now and then whether Gemini Live works again."""
    from neo.voice.live import LiveVoice

    while not stop.is_set():
        await asyncio.sleep(_LIVE_RETRY_S.get(kind, 600.0))
        if app._tasks or bus().active_jobs:
            continue  # never swap voices mid-request
        v = LiveVoice(app.handle_text, app.session)
        try:
            await asyncio.wait_for(v.probe(), 15)
        except Exception:  # noqa: BLE001
            continue
        local = app.voice
        try:
            if local is not None:
                await asyncio.wait_for(local.close(), 3)
            await v.start()
            v.on_down = on_down
            app.voice = v
            print("[voice] Gemini Live is back")
            health.report("voice", None)
            await bus().note("Gemini Live voice is back.")
        except Exception as e:  # noqa: BLE001
            print(f"[voice] couldn't switch back to Live: {str(e)[:120]}")
            app.voice = local
        return


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
