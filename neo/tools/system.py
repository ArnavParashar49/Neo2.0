"""System controls with zero-LLM fast paths: volume, mute, brightness, clock, sleep."""

from __future__ import annotations

import asyncio
import datetime as _dt
import re

from neo.agent.registry import ToolContext, tool
from neo.tools.computer.apps import osascript

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
)
async def timer(a: dict, c: ToolContext) -> str:
    raw = a.get("minutes")
    mins = 1.0 if raw in (None, "") else max(0.05, float(raw))
    label = a.get("label") or "Timer"

    async def fire() -> None:
        await asyncio.sleep(mins * 60)
        await osascript(f'display notification "{label} is done" with title "NEO" sound name "Glass"')
        from neo.events import bus

        await bus().say(f"{label}: time's up.")

    asyncio.get_event_loop().create_task(fire())
    return f"Timer set for {mins:g} minute{'s' if mins != 1 else ''}."


_ = re  # keep import for the regex constants above
