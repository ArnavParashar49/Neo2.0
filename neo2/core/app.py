"""Wires everything together: tools, session, voice, UI server. `python -m neo2` lands here."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from typing import Any

from neo2.core import health
from neo2.core.agent import confirm, grants
from neo2.core.agent.session import Session, answer_of
from neo2.core.config import set_ui_host, set_wake_word, set_wellbeing_checkins, settings
from neo2.core.events import NeoState, bus
from neo2.core.memory.store import store
from neo2.core.reflex import warmup
from neo2.core.server.ws import UIServer
from neo2.core.tools import load_all


class App:
    _checkin_busy = False  # a check-in is being said or answered: one at a time
    _last_active = 0.0  # monotonic: NEO was last not idle (see _checkin_idle)

    def __init__(self) -> None:
        import time

        self.started = time.time()
        self.session = Session()
        self.voice = None  # set in run() when a backend is available
        self._tasks: set[asyncio.Task] = set()
        self.wellbeing = None  # local sound monitor; it never opens the microphone
        self._wellbeing_speaking_until = 0.0
        self._wellbeing_reply_context: tuple[float, str] | None = None
        self._last_active = time.monotonic()  # (starting up counts as busy)
        self.ui: UIServer | None = None  # the websocket server (set in run()): its link to NEO.app's shell
        self._keys_opened_at: float | None = None  # monotonic: Brains & Keys last opened for a request

    async def handle_text(
        self, text: str, *, speak: bool = True, source: str = "typed", prompt_context: str = "", checkin: bool = False
    ) -> None:
        """Handle one request. Several may be in flight at once (the session runs them as jobs;
        each works out its own memory context: neo2.core.agent.session). `source`: "typed", or
        "voice" for the on-device voice's transcription — what the journal records. `checkin`: the
        reply to a check-in NEO started itself (handle_speech): answered, never routed or kept."""
        if await self._key_guard(text, speak=speak):
            return
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        try:
            # typed, or the user's own speech
            reply = await self.session.handle(
                text,
                from_user=True,
                source="voice" if source == "voice" else "typed",
                prompt_context=prompt_context,
                checkin=checkin,
            )
            if speak and self.voice and reply.text and not reply.silent:
                await self.voice.speak(reply.spoken or reply.text)
        except asyncio.CancelledError:
            await bus().say("Stopped.", final=True)
        finally:
            if task:
                self._tasks.discard(task)

    async def _key_guard(self, text: str, *, speak: bool = True) -> bool:
        """Typed text that starts with an API key goes to no model, no journal and no transcript: NEO
        says so instead (True: it was one). Narrow on purpose — the known key prefixes only
        (neo2.core.keyshape.looks_like_key), so a long hash is still just text."""
        from neo2.core.keyshape import looks_like_key

        if not looks_like_key(text):
            return False
        from neo2.core.onboarding import KEY_TYPED

        print("[keys] a message that starts with an API key was kept from every model")
        await bus().say(KEY_TYPED, role="assistant", final=True)
        voice = self.voice
        if speak and voice is not None and not getattr(voice, "handles_text", False) and hasattr(voice, "speak"):
            await voice.speak(KEY_TYPED)  # (the on-device voice: nothing leaves the Mac to say it)
        return True

    def open_keys(self, open_now: bool = True) -> str:
        """Brains & Keys for a request that needs a model and has none (neo2.core.onboarding): opened —
        at most once every 10 minutes — through NEO.app's shell. "opened", "busy" (not now) or
        "no_app" (no NEO.app at the other end: a core started by hand)."""
        ui = self.ui
        if ui is None or not ui.has_shell():
            return "no_app"
        now = time.monotonic()
        if not open_now or (self._keys_opened_at is not None and now - self._keys_opened_at < _KEYS_AGAIN_S):
            return "busy"
        if not ui.open_keys():
            return "no_app"
        self._keys_opened_at = now
        return "opened"

    def open_lens(self, question: str = "") -> str:
        """Search Screen, asked for by the voice or the agent's tool: NEO.app's shell opens its
        overlay (neo2.core.lens.live: one every 1.5 s). "opened", "off", "no_app" or "busy"."""
        from neo2.core.lens.live import lens_live

        return lens_live.open_overlay(question)

    async def handle_lens(self, text: str, image, context: str) -> None:
        """Search Screen's "Continue in NEO": the user's words (or "Tell me more about this.") with the
        area's picture, as their own request — tainted as screen content (what it shows is data,
        never instructions), its title and questions so far as this turn's context only. The chat
        line shows a "Screen area" chip; no pixels go into the transcript, the replay or the journal."""
        said = text.strip() or "Tell me more about this."
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        try:
            reply = await self.session.handle(
                said, images=[image] if image is not None else None, from_user=True, taint=("search_screen",),
                prompt_context=context, source="typed", attach="screen",
            )  # fmt: skip
            if self.voice and reply.text and not reply.silent and not getattr(self.voice, "handles_text", False):
                await self.voice.speak(reply.spoken or reply.text)
        except asyncio.CancelledError:
            await bus().say("Stopped.", final=True)
        finally:
            if task:
                self._tasks.discard(task)

    async def lens_listen(self) -> None:
        """Search Screen's mic button: the voice listens, as the summon keys make it, for a question
        about the area (the session's lens lane takes the sentence)."""
        await bus().publish("woke", source="ui")
        if self.voice:
            await self.voice.wake("ui")

    def voice_kind(self) -> str | None:
        """Which voice runs now: "live" (Gemini Live), "local" (on this Mac), None (none yet)."""
        v = self.voice
        if v is None:
            return None
        return "live" if hasattr(v, "_dialog") else "local"

    async def handle_speech(self, text: str) -> None:
        """What the on-device voice heard the user say. The first words it heard in a check-in's
        reply window are the reply to it: the session answers them with the check-in's context and
        never runs or keeps them (Session._checkin_answer); later words are the user's own again."""
        context = self._wellbeing_reply_context
        self._wellbeing_reply_context = None
        started = getattr(self.voice, "_last_utterance_started_at", None)
        timely = (
            context is not None
            and started is not None
            and started <= context[0]
            and time.monotonic() <= context[0] + 15.0  # allow local transcription to finish
        )
        if timely and context is not None:
            await self.handle_text(text, source="voice", prompt_context=context[1], checkin=True)
        else:
            await self.handle_text(text, source="voice")

    async def erase(self) -> dict[str, int]:
        """Erase everything NEO kept about the user. Everything that could write a piece of the
        conversation back afterwards is stopped first: requests in flight, summaries being
        written, the open voice session and what it keeps for context and for Laya."""
        from neo2.core.agent.early import early
        from neo2.core.privacy import forget_everything

        self.cancel_current()
        await self.session.cancel_all()
        for t in list(self.session._bg):
            t.cancel()
        v = self.voice
        if v is not None:
            await v.interrupt()  # cancels the agent tasks a voice session started
            if hasattr(v, "_dialog"):  # Gemini Live
                v._label, v._utterance = None, ""
                v._call_labels.clear()
                v._dialog.clear()
                if v._live is not None:
                    v._closing = True  # a fresh session: the server forgets the conversation too
                    await v._close_session()
        early().new_utterance()
        early()._claims.clear()
        grants.reset()  # every "yes for this task" given so far ends with what it was given in
        counts = await forget_everything()  # bumps the erase generation before anything else
        self.session.history.clear()
        self.session._summarized = 0
        self.session._chunk_started = time.time()
        self.session.memory_context = ""
        print("[privacy] erased everything NEO kept about the user")
        return counts

    async def resume_browsing(self, text: str) -> None:
        """The user handed NEO's browser back after doing their part (a login, a code): the job
        goes on, as if they had said so — they did, on the card."""
        if self.voice and getattr(self.voice, "handles_text", False):
            await bus().say(text, role="user")
            await self.voice.send_text(text)
        else:
            await self.handle_text(text)

    async def decline(self, action_id: str) -> None:
        """The user took NEO's browser over while it waited for their OK: that's a no."""
        await self.answer(False, action_id=action_id, said="no")

    def cancel_current(self) -> bool:
        """Stop everything in flight (the stop button / "stop")."""
        n = 0
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
                n += 1
        return n > 0

    async def answer(self, yes: bool, *, action_id: str = "", said: str = "", scope: str = "") -> None:
        """The user answered NEO's pending question (the card's buttons, or a typed yes/no).
        `scope` "task": the card's "yes for this task" (typed or said, the words carry it)."""
        reply = await self.session.answer(
            yes, action_id=action_id, text=said or ("yes" if yes else "no"), scope="task" if scope == "task" else None
        )
        if reply is None:
            await bus().note("That question had already expired — nothing was done.")
            return
        live = self.voice is not None and hasattr(self.voice, "inform")
        if reply.text and not reply.silent and not live:
            # What NEO answers is in the journal (Live writes down what it then says itself).
            from neo2.core.memory import journal

            journal.log_turn("assistant", reply.text, route=reply.route, tainted=bool(reply.taint))
        if self.voice and reply.text and not reply.silent:  # approved and done: nothing to say
            if live:
                await self.voice.inform(reply.text)  # Live says what went wrong, in the user's language
            else:
                await self.voice.speak(reply.text)

    async def on_command(self, cmd: dict[str, Any]) -> None:
        kind = cmd.get("cmd")
        if kind == "text":
            text = str(cmd.get("text", ""))
            if await self._key_guard(text):
                return  # (before Gemini Live too: it would be sent to Google)
            if confirm.peek() is not None and (ans := answer_of(text)) is not None:
                await bus().say(text, role="user")
                from neo2.core.memory import journal

                journal.log_turn("user", text, route="confirm", source="typed")
                await self.answer(ans, said=text)  # typed straight to the question, in any voice mode
            elif self.voice and getattr(self.voice, "handles_text", False):
                from neo2.core.tools.browser_live import live

                live.user_said()  # the user's own words (a typed request has its job only in Live's calls)
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
            await self.answer(bool(cmd.get("yes")), action_id=action_id, scope=str(cmd.get("scope") or ""))
        elif kind == "forget_all":
            # From the overlay's own "Erase everything?" confirmation, or `neo2 --forget-all`
            # while NEO runs — never a model's tool.
            counts = await self.erase()
            await bus().publish("erased", counts=counts)
            await bus().note("Erased: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
        elif kind == "stop":
            from neo2.core.tools.browser_live import live

            live.user_stopped()  # the page NEO was on stays, and it's the user's now
            if self.voice:
                await self.voice.interrupt()  # tells Laya's outcome ledger, knowing how long NEO spoke
            else:
                from neo2.core.reflex import outcomes

                outcomes.stop()  # right after NEO answered, a stop says the answer was wrong
            if not self.cancel_current():
                await bus().set_state(NeoState.IDLE)
        elif kind == "ptt":
            if self.voice:
                await self.voice.push_to_talk(bool(cmd.get("on")))
        elif kind == "wake":
            # Summoned by hand (the hotkey, the menu-bar item, the orb) — never the wake word. Told
            # to every client: NotchHub opens its panel for this and not for a voice turn.
            await bus().publish("woke", source="ui")
            if self.voice:
                await self.voice.wake("ui")
        elif kind == "ui_host":
            # NotchHub takes NEO's UI into the notch ("notch") and gives it back when it quits
            # ("window"); NEO.app's shell follows the event (its window hidden, or shown again).
            host = str(cmd.get("host") or "")
            if host not in ("window", "notch"):
                return
            set_ui_host(host)
            await bus().publish("ui_host", host=host)
        elif kind == "set":
            # A setting switched while NEO runs. The websocket lets only a host and NEO.app's own
            # page send this (neo2/core/server/ws.py), and has checked the name and the value.
            if cmd.get("name") == "wake_word" and isinstance(cmd.get("value"), bool):
                await self.set_wake_word(cmd["value"])
            elif cmd.get("name") == "wellbeing_checkins" and isinstance(cmd.get("value"), bool):
                await self.set_wellbeing_checkins(cmd["value"])

    async def set_wake_word(self, on: bool) -> None:
        """"Listen for 'Neo'" switched: kept in ~/.neo/.env, applied to the running voice at once
        (on: its microphone opens; off: it closes now if NEO is idle, else when it next is — a
        voice that starts later reads the setting), and told to every client."""
        if settings().wake_word != on:
            set_wake_word(on)
        if self.voice is not None and hasattr(self.voice, "set_wake_word"):
            await self.voice.set_wake_word(on)
        elif not on:
            print("[voice] wake word off: the microphone opens only when summoned")
        await bus().publish(
            "settings", wake_word=on, wellbeing_checkins=settings().wellbeing_checkins
        )

    async def set_wellbeing_checkins(self, on: bool) -> None:
        """Switch the local sound check-in monitor without opening or retaining microphone audio."""
        if settings().wellbeing_checkins != on:
            set_wellbeing_checkins(on)
        if self.wellbeing is not None:
            self.wellbeing.set_enabled(on)
        await bus().publish(
            "settings", wake_word=settings().wake_word, wellbeing_checkins=on
        )

    def wellbeing_event(self, kind: str) -> None:
        """Deliver a local cue as a short check-in through the active voice backend."""
        asyncio.create_task(self._wellbeing_checkin(kind))

    def _checkin_idle(self, *, busy_ok: bool = False) -> bool:
        """Is NEO idle enough for an unasked question? Nothing running or waiting, no voice turn or
        follow-up window open, and quiet for a while: never between turns of the user's own
        conversation, never over NEO's own voice. (`busy_ok`: the check-in asking is this one.)"""
        if bus().state != NeoState.IDLE or (self._checkin_busy and not busy_ok):
            return False
        voice = self.voice
        if _busy(self, voice):
            return False
        if time.monotonic() - self._last_active < _CHECKIN_QUIET_S:
            return False  # the user was just talking with NEO: maybe they aren't done
        if voice is None:
            return True
        speaker = getattr(voice, "spk", None)
        if getattr(voice, "_speaking", False) or (speaker is not None and speaker.playing.is_set()):
            return False
        if getattr(voice, "_turn_open", False) or getattr(voice, "_ptt", False):
            return False  # let an active user sentence finish before offering an unsolicited check-in
        if getattr(voice, "_live", None) is not None:
            return False  # a Live session is open: the conversation, or its follow-up window
        armed = getattr(voice, "_armed", None)
        listening = (armed is not None and armed.is_set()) or time.time() < getattr(voice, "_followup_until", 0.0)
        return not listening  # (the on-device voice listening for the user, after a summon or a reply)

    def _track_activity(self, ev) -> None:
        """When NEO was last anything but idle (a check-in waits a while after that)."""
        if ev.type == "state" and ev.data.get("state") != NeoState.IDLE.value:
            self._last_active = time.monotonic()

    async def _wellbeing_checkin(self, kind: str) -> None:
        # Detection stays local; Live receives only the short personalized sentence so it can speak
        # it in the configured voice. How often (the monitor's policy): 10 min → 1 h → 4 h per
        # kind, at most 3 a day, none for the rest of the day after a "no" or two unanswered ones.
        # When: only while NEO is idle and no other app is playing sound (a call, a video, music).
        if not settings().wellbeing_checkins:
            return
        policy = getattr(self.wellbeing, "policy", None)
        if (policy is not None and not policy.allows(kind)) or not self._checkin_idle():
            return
        self._checkin_busy = True
        try:
            if await asyncio.to_thread(_other_media_playing) or not self._checkin_idle(busy_ok=True):
                return
            await self._checkin(kind, policy)
        finally:
            self._checkin_busy = False

    async def _checkin(self, kind: str, policy) -> None:
        text = {
            "cough": "A bit of a cough? You okay? Want a few simple tips?",
            "cough_or_throat": "A cough or throat tickle? Everything okay? Want a few simple tips?",
            "sniffling": "Some sniffles? Feeling okay, or catching a cold? Want a few simple tips?",
            "sneezing": "A few sneezes? You alright? Want a few simple tips?",
            "throat_clearing": "Clearing your throat a lot? Everything okay? Want a few simple tips?",
            "crying": "Hey, are you okay? I'm here if you want to talk.",
        }.get(kind, "Hey, are you doing okay? I'm here if you need anything.")
        if policy is not None:
            policy.given(kind)
        # Use the active NEO voice. Live speaks the personalized line through Charon; the local
        # classifier label itself remains on-device. The on-device voice keeps using its own TTS.
        self._wellbeing_speaking_until = time.monotonic() + max(6.0, len(text) / 8.0)
        reachable = True
        speak_checkin = getattr(self.voice, "speak_checkin", None)
        if callable(speak_checkin):
            try:
                spoke = await speak_checkin(text)
            except Exception as error:  # noqa: BLE001 — check-ins must not affect the voice loop
                print(f"[wellbeing] configured voice unavailable: {type(error).__name__}")
                spoke = None
            if not spoke:
                # None: Live couldn't be reached — say it here, and don't try to listen through it
                # (a second and third failed connect, then "online voice unavailable", for a cough).
                # False: Live was reached but didn't finish — a fresh session can still listen.
                reachable = spoke is not None
                await bus().say(text, role="assistant")
                await asyncio.to_thread(_say_aloud_wait, text)
        elif self.voice is not None and hasattr(self.voice, "speak"):
            await bus().say(text, role="assistant")
            await self.voice.speak(text)  # local Kokoro output goes through NEO's normal playback gate
        else:
            await bus().say(text, role="assistant")
            await asyncio.to_thread(_say_aloud_wait, text)

        listen = getattr(self.voice, "listen_after_checkin", None)
        # Only while the microphone is open anyway: with "Listen for 'Neo'" off a check-in never
        # opens it (the user decides when NEO listens).
        if not callable(listen) or not reachable or not settings().wake_word or getattr(self.voice, "_mic_parked", False):
            return
        # This ephemeral context lets a plain "yes" make sense to the answer model. The
        # classifier's internal label and the audio sample itself are never attached to it.
        reply_context = (
            "NEO just asked if the user is okay and is listening to the answer. Respond gently "
            "to what they actually say; do not infer a condition."
            if kind == "crying"
            else "NEO just offered a few simple comfort tips and is listening to the answer. "
            "If the user accepts, give brief, low-risk general suggestions; do not diagnose."
        )
        if not hasattr(self.voice, "_live"):
            self._wellbeing_reply_context = (time.monotonic() + 5.0, reply_context)
        try:
            reply = await listen(seconds=5.0, context=reply_context)
        except Exception as error:  # noqa: BLE001 — check-ins must not affect the voice loop
            self._wellbeing_reply_context = None
            print(f"[wellbeing] follow-up listening unavailable: {type(error).__name__}")
            return
        if reply is not None and policy is not None:
            policy.replied(reply)  # nothing said, or a "no": NEO backs off (it never logs the words)

    def wellbeing_unavailable(self, _reason: str) -> None:
        asyncio.create_task(self._wellbeing_unavailable())

    async def _wellbeing_unavailable(self) -> None:
        if not settings().wellbeing_checkins:
            return
        await bus().note("Local sound check-ins couldn't start on this Mac, so they were turned off.")
        await self.set_wellbeing_checkins(False)


_CHECKIN_QUIET_S = 60.0  # a check-in waits this long after NEO was last busy (the user may go on)
_KEYS_AGAIN_S = 600.0  # Brains & Keys opens by itself for a request at most this often


def _other_media_playing() -> bool:
    """Is another app playing sound now — a call, a video, music? Then a check-in would talk over it
    (and a cough in a video is not the user's). CoreAudio's per-process output list
    (neo2.core.decide.media.AudioWatcher, macOS 14.2+, no permission; NEO's own voice excluded). Where
    that list can't be read, nothing is known to play."""
    try:
        from neo2.core.decide import media

        return media.audio_confirmable() and bool(media.AudioWatcher().playing())
    except Exception:  # noqa: BLE001 — no CoreAudio list on this Mac
        return False


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
        from neo2.core.agent.failures import KINDS

        summary = str(d.get("summary", ""))
        # Failures too: an error can quote a path, a URL with a token, or a file's first line. A
        # failure's kind is one of NEO's fixed names (neo2.core.agent.failures), never anything else.
        kind = f" [{d['kind']}]" if d.get("kind") in KINDS else ""
        print(f"  ◀ {d['name']} {'ok' if d['ok'] else 'FAIL'}{kind} ({len(summary)} chars)")
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
    from neo2.core import log_to_stdout
    from neo2.core.server import shell_link

    log_to_stdout()
    app_socket = shell_link.inherited()  # NEO.app's shell, if it started us: claimed before any child starts
    s = settings()
    from neo2.core.permissions import request_missing

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
    from neo2.core.tools.browser_live import live

    live.hooks.resume = app.resume_browsing
    live.hooks.decline = app.decline
    ui = UIServer(app.on_command)
    app.ui = ui
    from neo2.core import brains, onboarding

    brains.started(s)  # what the voice starts with: a key changed later says "restart NEO"
    brains.voice_now = app.voice_kind
    onboarding.set_opener(app.open_keys)  # a request with no model to answer it opens Brains & Keys
    from neo2.core.lens import search as lens_search
    from neo2.core.lens.live import lens_live
    from neo2.core.tools import lens as lens_tool

    lens_live.hooks.continue_ = app.handle_lens  # Search Screen's Continue in NEO, its Brains & Keys…, its mic
    lens_live.hooks.open_keys = ui.open_keys
    lens_live.hooks.listen = app.lens_listen
    lens_tool.start(app_socket is not None)  # search_screen: hidden while it's off or no NEO.app runs us
    with contextlib.suppress(OSError):
        lens_search.sweep()  # an image search's page an earlier run left behind
    await ui.start()
    ui.shell_task = shell_link.start(ui, app_socket)  # the menu, the summon keys, the status: no page needed
    bus().subscribe(app._track_activity)
    if sys.platform == "darwin":
        # Gentle check-ins are optional: whatever goes wrong setting them up, NEO starts without them.
        try:
            from neo2.core.voice.wellbeing import WellbeingMonitor

            app.wellbeing = WellbeingMonitor(
                loop=loop,
                on_event=app.wellbeing_event,
                on_error=app.wellbeing_unavailable,
                is_ignored=lambda: (
                    time.monotonic() < app._wellbeing_speaking_until
                    or bool(getattr(app.voice, "_speaking", False))
                    or bool(getattr(getattr(app.voice, "spk", None), "playing", None) and app.voice.spk.playing.is_set())
                ),
            )
            app.wellbeing.set_enabled(s.wellbeing_checkins)
        except Exception as e:  # noqa: BLE001
            print(f"[wellbeing] check-ins unavailable: {type(e).__name__}")
    warmup(app.session.reflex)  # Laya, then the memory embedder, sequentially
    from neo2.core.tools.system import restore_timers

    restore_timers()  # timers set before a restart still go off
    if sys.platform == "darwin":
        from neo2.core.decide import engine as decide
        from neo2.core.decide.kinds import schedule

        decide.warm()  # scan the apps now, so "play …" doesn't wait for it the first time
        schedule.prune()  # calendar files an earlier run handed to a calendar app
        try:
            from neo2.core.tools import shortcuts

            # The user's Shortcuts list, read in the background (RAM only), so "run my … shortcut" is
            # known the first time; hidden from every model when NEO_SHORTCUTS is off.
            shortcuts.start()
        except Exception as e:  # noqa: BLE001 — NEO starts without them
            print(f"[shortcuts] unavailable: {type(e).__name__}")
        try:
            from neo2.core.tools import context_now

            context_now.start()  # in_front hidden from every model when NEO_CONTEXT_CAPTURE is off
        except Exception as e:  # noqa: BLE001
            print(f"[in_front] unavailable: {type(e).__name__}")

    # Text works from here on; voice comes up alongside (it may be downloading models).
    await bus().set_state(NeoState.IDLE)
    asyncio.create_task(_lag_monitor())
    asyncio.create_task(_cocoa_pump())
    asyncio.create_task(health.watch())
    if s.voice != "off":
        asyncio.create_task(_voice_supervisor(app, stop))
    from neo2.core.reflex.learn import learn_loop
    from neo2.core.reflex.teacher import teacher_loop

    def _idle() -> bool:
        from neo2.core.events import bus

        voice_busy = bool(app.voice and getattr(app.voice, "_live", None))
        return not voice_busy and not bus().active_jobs and not app._tasks

    asyncio.create_task(learn_loop(_idle))
    asyncio.create_task(teacher_loop(_idle))  # returns at once unless NEO_LEARN_TEACHER is on
    if sys.platform == "darwin":
        # The OS's own English↔Hindi UI strings, built once per macOS build when NEO is idle (paused
        # while it is busy): what the gate reads a Hindi label by (decision 44). NEO.app only.
        from neo2.core.decide import uistrings

        asyncio.create_task(uistrings.idle_loop(_idle))
        # When the user starts talking (the wake word, the start of speech: prefetch.prewarm), text
        # recognition loads too, so a screenshot's first read of the words on it doesn't pay for it.
        # (The memory embedder the resolver's meaning stage uses is warmed above, at start.)
        from neo2.core.agent import prefetch
        from neo2.core.tools.computer import ocr

        prefetch.register_warmer(ocr.warm)
    asyncio.create_task(_memory_loop(app, _idle))
    bus().subscribe(_log_activity)
    print(
        "NEO is running. Say 'Neo', or type in the overlay."
        if settings().wake_word or settings().voice == "off"
        else "NEO is running. Summon it with its keys, or type in the overlay."
    )
    await stop.wait()
    if app.wellbeing is not None:
        await asyncio.to_thread(app.wellbeing.close)
    # Everything below must finish well inside NEO.app's grace period (it SIGKILLs after 12 s).
    if app.voice:
        try:
            await asyncio.wait_for(app.voice.close(), 2)
        except Exception:  # noqa: BLE001
            pass
    # What was said is on disk already (the journal commits within a quarter of a second): this
    # only writes the last lines out. Nothing about memory depends on the rest of the shutdown.
    from neo2.core.memory import journal

    with contextlib.suppress(Exception):
        await asyncio.wait_for(asyncio.to_thread(journal.flush, 0.8), 1.0)
    try:
        from neo2.core.tools import render
        from neo2.core.tools.browser import shutdown as close_browser

        await asyncio.wait_for(close_browser(), 2)
        with contextlib.suppress(Exception):  # the invisible page renderer (atexit would also do it)
            await asyncio.wait_for(render.shutdown(), 2)
    except Exception as e:  # noqa: BLE001
        print(f"[neo] browser shutdown skipped: {str(e)[:80]}")
    # A short best effort at the open conversation's summary; one that doesn't make it is
    # written at the next start (neo2.core.memory.summarize.catch_up).
    try:
        if s := await asyncio.wait_for(app.session.checkpoint(final=True, budget_s=2.5), _SUMMARY_BUDGET_S):
            print(f"[memory] summaries saved ({s})")  # not what was said: logs stay content-free
    except TimeoutError:
        print("[memory] shutdown summary took too long — it is written at the next start")
    except Exception as e:  # noqa: BLE001
        print(f"[memory] shutdown summary skipped: {str(e)[:80]}")
    with contextlib.suppress(Exception):
        await asyncio.to_thread(journal.close, 0.5)


_CHECKPOINT_EVERY_S = 20 * 60
_MEMORY_TICK_S = 30.0  # how often the memory's background work looks for something to do
_EXTRACT_IDLE_S = 60.0  # the model reads the journal for facts only after this long without a line
_PRUNE_EVERY_S = 86400.0


async def _memory_loop(app: App, is_idle, *, tick_s: float = _MEMORY_TICK_S) -> None:
    """Memory's background work, only ever while NEO is idle — never on a request's path:

    * at start, the conversations an earlier run left without a summary (it was killed, or quit
      mid-conversation) get one, and journal lines past their keep-by date go;
    * a conversation that has gone quiet gets its summary (written again if it goes on);
    * after a minute without a line, the fast brain reads the new journal lines for facts
      (neo2.core.memory.extract — nothing with NEO_CLOUD_MEMORY=off or NEO_MEMORY_EXTRACT=rules);
    * the vectors that couldn't be made while the embedding model was loading are made, and the
      journal's writer is nudged to write the recall counters.

    With the journal off, the session's history is summarised now and then instead, as before."""
    from neo2.core.memory import embed, extract, journal, summarize

    s = settings()
    if not s.memory_enabled:
        return
    last_checkpoint, last_prune = time.time(), 0.0
    first = True
    while True:
        await asyncio.sleep(2.0 if first else tick_s)
        try:
            if not is_idle():
                continue
            now = time.time()
            if embed.ready() and not store()._backfilled:
                # vectors for what was written while the embedding model was still loading
                await asyncio.to_thread(store().backfill)
            j = journal.journal()
            if j is None:  # no journal: summaries come from the session's history, every so often
                first = False
                if now - last_checkpoint >= _CHECKPOINT_EVERY_S:
                    last_checkpoint = now
                    await app.session.checkpoint()
                continue
            if first or now - last_prune >= _PRUNE_EVERY_S:
                last_prune = now
                gone = await asyncio.to_thread(store().prune_turns, s.memory_retention_days)
                if gone:
                    print(f"[memory] {gone} journal lines older than {s.memory_retention_days} days removed")
            if first:
                first = False
                if s.summaries and (n := await summarize.catch_up()):
                    print(f"[memory] caught up: {n} conversation(s) from before summarised")
            elif s.summaries and (n := await summarize.summarize_due()):
                print(f"[memory] {n} conversation(s) summarised")
            if is_idle() and now - j.last_turn_at >= _EXTRACT_IDLE_S:
                await extract.idle_pass()
            j.kick()  # (the recall counters)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[memory] background work skipped: {str(e)[:80]}")


_VOICE_RETRY_S = 60.0
_LIVE_RETRY_S = {"quota": 3600.0, "network": 180.0, "other": 600.0}
_MACOS_SYSTEM_VOICE = "com.apple.voice.Aman.premium"  # natural masculine Siri voice, stays on-device


def _say_aloud(text: str) -> None:
    """Last-resort speech when no voice backend can talk (macOS's own voice)."""
    import subprocess

    try:
        subprocess.Popen(
            ["say", "-v", _MACOS_SYSTEM_VOICE, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        pass


def _say_aloud_wait(text: str) -> None:
    """Speak a short local prompt and wait so voice capture remains muted through its audio tail."""
    import subprocess

    try:
        subprocess.run(
            ["say", "-v", _MACOS_SYSTEM_VOICE, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception:  # noqa: BLE001 — an unavailable system voice shouldn't break NEO
        pass


async def _voice_supervisor(app: App, stop: asyncio.Event) -> None:
    """Bring voice up, and keep it up: retry if it can't start, and when Gemini Live keeps
    failing (quota used up, network down) say so and switch to the on-device voice until Live
    works again."""
    from neo2.core import health
    from neo2.core.voice import devices
    from neo2.core.voice.pipeline import local_ready, make_local, make_voice

    if devices.mic_status() in (devices.DENIED, devices.RESTRICTED):
        print("[voice] microphone access is off for NEO")
    down: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    async def on_down(kind: str, why: str) -> None:
        await down.put((kind, why))

    while not stop.is_set():
        if app.voice is None:
            try:
                app.voice = await make_voice(app.handle_text, app.session, on_speech=app.handle_speech)
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
            # Let whatever Live is doing finish first (an agent task mid-way through files, a reply
            # being spoken): closing Live cancels it.
            await _wait_idle(app, live, limit_s=120.0)
            print(f"[voice] Gemini Live is down ({why}); switching to the on-device voice")
            try:
                local = await make_local(app.handle_speech)  # up before Live goes: never a gap
            except Exception as e:  # noqa: BLE001
                print(f"[voice] on-device voice failed too: {str(e)[:120]}")
                health.report("voice", f"Voice is unavailable: {reason}. You can still type; NEO keeps trying.")
                _say_aloud(f"Voice is unavailable because {reason}. You can still type to me.")
                continue  # keep Live: it may connect on the next wake
            app.voice = local
            try:
                await asyncio.wait_for(live.close(), 3)
            except Exception:  # noqa: BLE001
                pass
            health.report("voice", f"Gemini Live is unavailable ({reason}) — using the on-device voice for now.")
            await local.speak(f"Online voice is unavailable because {reason}. I'm using the on-device voice for now.")
            asyncio.create_task(_back_to_live(app, stop, kind, on_down))
        else:
            health.report("voice", f"Voice is unavailable: {reason}. You can still type; NEO keeps trying.")
            _say_aloud(f"Voice is unavailable because {reason}. You can still type to me.")


def _busy(app: App, voice) -> bool:
    """Is a request in flight (typed, spoken, or an agent task the voice started)?"""
    if any(not t.done() for t in app._tasks) or bus().active_jobs:
        return True
    if voice is None:
        return False
    if any(not t.done() for t in getattr(voice, "_tool_tasks", {}).values()):
        return True
    handler = getattr(voice, "_handler", None)
    return bool(getattr(voice, "_capturing", False) or (handler is not None and not handler.done()))


async def _wait_idle(app: App, voice, *, limit_s: float) -> bool:
    deadline = time.monotonic() + limit_s
    while _busy(app, voice) and time.monotonic() < deadline:
        await asyncio.sleep(1.0)
    return not _busy(app, voice)


async def _back_to_live(app: App, stop: asyncio.Event, kind: str, on_down) -> None:
    """While on the on-device voice, check now and then whether Gemini Live works again."""
    from neo2.core.voice.live import LiveVoice

    while not stop.is_set():
        await asyncio.sleep(_LIVE_RETRY_S.get(kind, 600.0))
        local = app.voice
        if _busy(app, local):
            continue  # never swap voices mid-request
        v = LiveVoice(app.handle_text, app.session)
        try:
            await asyncio.wait_for(v.probe(), 15)
        except Exception:  # noqa: BLE001
            continue
        if _busy(app, local):
            continue  # something started while the probe ran
        try:
            await v.start()  # Live first; the on-device voice stays in charge until this works
        except Exception as e:  # noqa: BLE001
            print(f"[voice] couldn't switch back to Live: {str(e)[:120]}")
            try:
                await v.close()
            except Exception:  # noqa: BLE001
                pass
            continue
        v.on_down = on_down
        app.voice = v
        if local is not None:
            try:
                await asyncio.wait_for(local.close(), 3)
            except Exception:  # noqa: BLE001
                pass
        print("[voice] Gemini Live is back")
        health.report("voice", None)
        await bus().note("Gemini Live voice is back.")
        return


_SUMMARY_BUDGET_S = 3.0  # the model gets 2.5 s inside; this covers the crude fallback + the write


async def _exit_with_parent(stop: asyncio.Event) -> None:
    """Started by NEO.app: if the app goes away without stopping us (force quit, crash), stop
    too — a core with the mic open and nobody to show it must never be left behind."""
    parent = os.getppid()
    while not stop.is_set():
        await asyncio.sleep(1.0)
        if os.getppid() != parent:  # re-parented to launchd: the app is gone
            print("[neo] NEO.app went away — shutting down")
            stop.set()
