"""System controls with zero-LLM fast paths: volume, mute, brightness, clock, sleep."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Collection
from typing import Any

from neo2.core import logger
from neo2.core.agent.registry import ToolContext, tool

log = logger(__name__)

_MAC = frozenset({"macos"})


# The Mac's own scripting, loaded on first use: `clock` needs none of it, so importing this module
# never loads the Mac tools (or AppKit) — the web preset uses the clock off macOS too.
async def osascript(script: str, timeout: float = 30) -> str:
    from neo2.core.tools.computer.apps import osascript as run

    return await run(script, timeout)


def _q(s: str) -> str:
    from neo2.core.tools.computer.apps import _q as quote

    return quote(s)

# Fast paths are whole-utterance commands: anchored, with an explicit object and direction, so an
# ordinary sentence that merely mentions "volume" or "dim" never triggers them.
_P = r"^\s*(?:please\s+|can you\s+|could you\s+)?"
_S = r"(?:\s+(?:a\s+bit|a\s+little|a\s+notch|please|now))*[\s.!]*$"
_VOL_RE = (
    _P
    + r"(?:turn|set|make|put)?\s*(?:the\s+)?(?:volume|sound)\s*(?:to|at)?\s*(?P<level>\d{1,3})\s*(?:%|percent)?"
    + _S
)
_UP_RE = (
    _P
    + r"(?:(?:turn|crank|bump)\s+(?:the\s+)?(?:volume|sound|it)\s+up|(?:volume|sound)\s+up|turn\s+up\s+(?:the\s+)?(?:volume|sound)|louder)"
    + _S
)
_DOWN_RE = (
    _P
    + r"(?:(?:turn|bring)\s+(?:the\s+)?(?:volume|sound|it)\s+down|(?:volume|sound)\s+(?:down|lower)|turn\s+down\s+(?:the\s+)?(?:volume|sound)|quieter)"
    + _S
)
_MUTE_RE = _P + r"(?:mute|silence)(?:\s+(?:the\s+)?(?:sound|volume|audio|mac|computer|it))?" + _S
_UNMUTE_RE = _P + r"unmute(?:\s+(?:the\s+)?(?:sound|volume|audio|mac|computer|it))?" + _S
_BR_UP = (
    _P
    + r"(?:(?:turn|make|bump)\s+(?:the\s+)?(?:brightness|screen)\s+(?:up|brighter)|brightness\s+up|turn\s+up\s+(?:the\s+)?brightness|(?:raise|increase)\s+(?:the\s+)?brightness|(?:make\s+(?:the\s+)?screen\s+)?brighter)"
    + _S
)
_BR_DOWN = (
    _P
    + r"(?:(?:turn|make)\s+(?:the\s+)?(?:brightness|screen)\s+(?:down|dimmer|lower)|brightness\s+down|turn\s+down\s+(?:the\s+)?brightness|(?:lower|reduce|decrease|dim)\s+(?:the\s+)?(?:brightness|screen)|(?:make\s+(?:the\s+)?screen\s+)?dimmer)"
    + _S
)
_TIME_RE = _P + r"(?:what(?:'s| is) the time|what time is it)(?:\s+(?:right\s+)?now)?[\s?.!]*$"
_TIMER_RE = (
    _P + r"(?:set|start)\s+(?:a\s+)?timer\s+(?:for\s+)?(?P<minutes>\d+(?:\.\d+)?)\s*(?:min|minute)s?" + _S
)
_TIMER2_RE = _P + r"(?:set|start)\s+(?:a\s+)?(?P<minutes>\d+(?:\.\d+)?)\s*(?:min|minute)s?\s+timer" + _S


@tool(
    "volume",
    "Set system volume 0-100, nudge it up/down, or mute/unmute. action: set|up|down|mute|unmute.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["set", "up", "down", "mute", "unmute"]},
            "level": {"type": "integer"},
        },
        "required": ["action"],
    },
    category="system",
    parallel_safe=True,
    fast_path=[
        (_VOL_RE, {"action": "set", "level": "<level>"}),
        (_UNMUTE_RE, {"action": "unmute"}),
        (_MUTE_RE, {"action": "mute"}),
        (_UP_RE, {"action": "up"}),
        (_DOWN_RE, {"action": "down"}),
    ],
    quiet=True,
    early=True,
    requires=_MAC,
)
async def volume(a: dict, c: ToolContext) -> str:
    act = a.get("action", "set")
    if act == "mute":
        await osascript("set volume with output muted")
        return "Muted."
    if act == "unmute":
        await osascript("set volume without output muted")
        return "Unmuted."
    cur = await osascript("output volume of (get volume settings)")
    cur_i = int(cur) if cur.isdigit() else 50
    if act == "set":
        lvl = a.get("level")
        level = 50 if lvl in (None, "") else max(0, min(100, int(lvl)))
    elif act == "up":
        level = min(100, cur_i + 15)
    else:
        level = max(0, cur_i - 15)
    await osascript(f"set volume output volume {level}")
    return f"Volume {level}%."


@tool(
    "brightness",
    "Nudge screen brightness up or down.",
    {
        "type": "object",
        "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, "steps": {"type": "integer"}},
        "required": ["direction"],
    },
    category="system",
    fast_path=[(_BR_DOWN, {"direction": "down"}), (_BR_UP, {"direction": "up"})],
    quiet=True,
    early=True,
    requires=_MAC,
)
async def brightness(a: dict, c: ToolContext) -> str:
    key = 144 if a.get("direction") == "up" else 145
    steps = max(1, min(16, int(a.get("steps") or 3)))
    await osascript(f'tell application "System Events" to repeat {steps} times\nkey code {key}\nend repeat')
    return f"Brightness {a.get('direction')}."


@tool(
    "clock",
    "Current local date and time.",
    {"type": "object", "properties": {}},
    category="system",
    parallel_safe=True,
    fast_path=[(_TIME_RE, {})],
    read_only=True,
)
async def clock(a: dict, c: ToolContext) -> str:
    return _dt.datetime.now().strftime("It's %-I:%M %p on %A, %B %-d.")


@tool(
    "sleep_display",
    "Put the display to sleep / lock the screen.",
    {"type": "object", "properties": {}},
    category="system",
    quiet=True,
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:lock\s+(?:the\s+|my\s+)?(?:screen|mac|computer|laptop)|"
            r"(?:put\s+)?(?:the\s+)?(?:display|screen)\s+to\s+sleep|sleep\s+(?:the\s+)?(?:display|screen)|"
            r"turn\s+(?:off|of)\s+the\s+(?:screen|display))\s*[.!]*\s*$",
            {},
        )
    ],
    requires=_MAC,
)
async def sleep_display(a: dict, c: ToolContext) -> str:
    p = await asyncio.create_subprocess_exec("pmset", "displaysleepnow")
    await p.wait()
    return "Display sleeping."


@tool(
    "timer",
    "Start a countdown timer that notifies when done. minutes may be fractional.",
    {
        "type": "object",
        "properties": {"minutes": {"type": "number"}, "label": {"type": "string"}},
        "required": ["minutes"],
    },
    category="system",
    fast_path=[
        (
            _TIMER_RE,
            {"minutes": "<minutes>"},
        ),
        (
            _TIMER2_RE,
            {"minutes": "<minutes>"},
        ),
    ],
    quiet=True,
    requires=_MAC,
)
async def timer(a: dict, c: ToolContext) -> str:
    raw = a.get("minutes")
    mins = 1.0 if raw in (None, "") else max(0.05, min(float(raw), _MAX_TIMER_MIN))
    label = " ".join(str(a.get("label") or "Timer").split())[:80] or "Timer"
    t = {"id": uuid.uuid4().hex[:12], "label": label, "due": time.time() + mins * 60}
    with _timers_lock:
        _save_timers(_load_timers() + [t])
    _start(t)
    return f"Timer set for {mins:g} minute{'s' if mins != 1 else ''}."


# Timers count wall-clock time (asyncio's clock stops while the Mac sleeps) and are kept in
# ~/.neo/timers.json, so a restart or a closed lid doesn't lose one.
_MAX_TIMER_MIN = 24 * 60
_STALE_S = 12 * 3600  # a timer that should have gone off this long ago isn't worth announcing
_timers_lock = threading.Lock()
_timer_tasks: dict[str, asyncio.Task] = {}


def _timers_path():
    from neo2.core.config import settings

    return settings().data_dir / "timers.json"


def _load_timers() -> list[dict]:
    try:
        data = json.loads(_timers_path().read_text())
        return [t for t in data if isinstance(t, dict) and {"id", "label", "due"} <= t.keys()]
    except (OSError, ValueError):
        return []


def _save_timers(ts: list[dict]) -> None:
    p = _timers_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ts))
    os.replace(tmp, p)


def _start(t: dict) -> None:
    if t["id"] not in _timer_tasks:
        _timer_tasks[t["id"]] = asyncio.get_running_loop().create_task(_run_timer(t))


async def _run_timer(t: dict) -> None:
    try:
        await _announce("present", t)
        while (left := t["due"] - time.time()) > 0:
            await asyncio.sleep(min(left, 5.0))  # re-check the wall clock: wakes up on time after sleep
        late = time.time() - t["due"] > 90
        with _timers_lock:
            _save_timers([x for x in _load_timers() if x["id"] != t["id"]])
        await _announce("end", t, done=True)
        note = f"{t['label']} is done" + (" (late — your Mac was asleep)" if late else "")
        await osascript(f'display notification {_q(note)} with title "NEO" sound name "Glass"')
        from neo2.core.events import bus

        await bus().say(f"{t['label']}: time's up." + (" (A bit late — your Mac was asleep.)" if late else ""))
    except asyncio.CancelledError:
        # Cancelled before it went off (the user's "cancel the timer", or NEO quitting): no
        # notification, no "time's up". Whoever cancelled it took it out of timers.json and told
        # the clients (cancel_timers); quitting leaves it in the file, for the next start to restore.
        raise
    finally:
        _timer_tasks.pop(t["id"], None)


# ---- what the clients see: NEO's running timers ----------------------------------------------
# Each timer that starts counting down (set, or restored after a restart) and each one that ends
# (it went off: done=true; it was cancelled: done=false) is an `activity` event on the bus, and a
# client's hello carries the running ones (server/ws.py), so NotchHub can show a countdown. It
# counts down from `ends_at` itself: nothing is sent per second.
_TITLE_MAX = 40


def timer_title(t: dict) -> str:
    """A timer's name as a client shows it (and NEO says it): one line, at most 40 characters."""
    return " ".join(str(t.get("label") or "").split())[:_TITLE_MAX] or "Timer"


def _activity(op: str, t: dict, *, done: bool = False) -> dict[str, Any]:
    """The `activity` event's data for one timer. Always the same keys, so a client can read every
    one alike; `id` is "timer:" + the timer's own id, so other kinds of activity never collide."""
    return {
        "op": op, "id": f"timer:{t['id']}", "kind": "timer", "title": timer_title(t), "ends_at": float(t["due"]), "done": done,
    }  # fmt: skip


async def _announce(op: str, t: dict, *, done: bool = False) -> None:
    """Tell the clients; never a reason for a timer not to go off."""
    try:
        from neo2.core.events import bus

        await bus().publish("activity", **_activity(op, t, done=done))
    except Exception as e:  # noqa: BLE001 — (the bus already keeps a bad subscriber to itself)
        log.warning(f"[timer] activity not announced: {type(e).__name__}")


def running_timers() -> list[dict]:
    """The timers in timers.json, the soonest first."""
    with _timers_lock:
        return sorted(_load_timers(), key=lambda t: t["due"])


def timers_snapshot() -> list[dict[str, Any]]:
    """The timers counting down in this process, the soonest first, as a client's hello carries
    them: [{id, title, ends_at}] (an `activity` event's keys). One in timers.json that nothing counts
    down (a core that never restored it) would never go off, so it isn't shown."""
    out = []
    for t in running_timers():
        task = _timer_tasks.get(t["id"])
        if task is not None and not task.done():
            a = _activity("present", t)
            out.append({"id": a["id"], "title": a["title"], "ends_at": a["ends_at"]})
    return out


async def cancel_timers(ids: Collection[str]) -> list[dict]:
    """Stop these timers before they go off: out of timers.json, their countdown cancelled (it
    never notifies or speaks), and an `activity` end (done=false) told to the clients. Returns the
    ones that were still set. (No await before the file is written: a countdown can't go off in
    between — it takes its timer out of the file before it says anything.)"""
    want = set(ids)
    with _timers_lock:
        ts = _load_timers()
        gone = [t for t in ts if t["id"] in want]
        if gone:
            _save_timers([t for t in ts if t["id"] not in want])
    for t in gone:
        task = _timer_tasks.pop(t["id"], None)
        if task is not None and not task.done():
            with contextlib.suppress(RuntimeError):  # its loop is already closed: nothing to stop
                task.cancel()
        await _announce("end", t, done=False)
    return gone


def restore_timers() -> int:
    """Re-arm the timers that were running before NEO restarted. Returns how many."""
    now = time.time()
    with _timers_lock:
        ts = _load_timers()
        keep = [t for t in ts if t["due"] > now - _STALE_S]
        if len(keep) != len(ts):
            _save_timers(keep)
    for t in keep:
        _start(t)
    return len(keep)


_ = re  # keep import for the regex constants above
