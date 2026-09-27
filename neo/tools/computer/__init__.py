"""Registers the computer-control tools. Import this module to make them available."""

from __future__ import annotations

import asyncio
import re
import time

from neo.agent import confirm, grants
from neo.agent.registry import ToolContext, ToolOutput, fence, tool
from neo.config import decide_mode
from neo.tools.computer import apps, ax, pointer, policy, screen, shell
from neo.tools.computer import input as inp

_OBJ = {"type": "object", "properties": {}}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


# ---- Accessibility ------------------------------------------------------------------------
# Seeing and acting in other apps (the AX tree, clicks, keys, typing, System Events' UI
# scripting) all take the Accessibility permission. Without it every such call fails, and a
# model that keeps getting failures tries its way around them (the Chess run: six AppleScripts,
# each a card). So the first one in a task stops the run and tells the user how to turn it on.
AX_OFF = (
    "NEO needs Accessibility to see and click in apps — turn NEO on in System Settings › Privacy & "
    "Security › Accessibility, then ask again"
)
_AX_STILL_OFF = (
    "Error: Accessibility is still off for NEO, and the user has already been told how to turn it on. "
    "Stop here and tell them; don't try AppleScript or any other way into the app."
)
# What osascript says when System Events may not drive another app's UI.
_AX_DENIED = re.compile(r"assistive access|not allowed to send keystrokes|\(-1719\)|\(-25211\)", re.I)


def _ax_trusted() -> bool:
    try:
        return bool(ax.is_trusted())
    except Exception:  # noqa: BLE001
        return False


def _ax_off_reply() -> str:
    """NEEDS_USER the first time in a task (the run stops, the user hears how to fix it); after
    that an error that tells the model to stop rather than look for another way in."""
    return confirm.ask_user(AX_OFF) if grants.first("accessibility-off") else _AX_STILL_OFF


def _needs_ax() -> str | None:
    """None with Accessibility on, else what the tool answers instead of trying."""
    return None if _ax_trusted() else _ax_off_reply()


# macOS checks separately whether a process may post input events, and drops them silently when
# it may not: a click NEO "made" would never arrive. Asked before any click, key or typing.
INPUT_OFF = (
    "macOS isn't letting NEO send clicks or keystrokes — in System Settings › Privacy & Security › "
    "Accessibility, turn NEO off and on again, then ask again"
)
_INPUT_STILL_OFF = (
    "Error: macOS still isn't letting NEO send clicks or keystrokes, and the user has already been told. "
    "Stop here and tell them."
)


def _needs_input() -> str | None:
    """None when NEO may both read the UI and post input events; else what the tool answers."""
    if err := _needs_ax():
        return err
    try:
        allowed = inp.can_post()
    except Exception:  # noqa: BLE001
        allowed = True
    if allowed:
        return None
    return confirm.ask_user(INPUT_OFF) if grants.first("input-off") else _INPUT_STILL_OFF


def _dialog_text(els=None, app: str = "") -> str:
    """The words in the app's focused window (a sheet or alert says what its "OK" does)."""
    if els is None:
        try:
            _, _, els = ax.window_elements(app or None)
        except Exception:  # noqa: BLE001
            return ""
    return " ".join((e.value or e.title) for e in els if e.role in ("AXStaticText", "AXTextArea"))[:2000]


def _ask_before_click(tool_name: str, a: dict, label: str, app: str = "", els=None) -> str | None:
    """A button that sends, buys, deletes, grants or publishes is the user's call — and so is a
    plain "OK" / "Continue" in a dialog that says it deletes or sends something."""
    context = _dialog_text(els, app) if policy._GENERIC_OK.match(label or "") else ""
    if not policy.commits(label, context):
        return None
    where = f" in {app}" if app else ""
    gate = confirm.gated(tool_name, a, f"click “{label.strip()[:60]}”{where}")
    return gate if isinstance(gate, str) else None


async def _ask_first(tool_name: str, a: dict, label: str, app: str = "", els=None) -> str | None:
    """:func:`_ask_before_click` from a handler: the dialog's words (a walk of its window, slow on
    a busy app) are read off the event loop; the question itself is staged on it (its card)."""
    if els is None and policy._GENERIC_OK.match(label or ""):
        try:
            _, _, els = await asyncio.to_thread(ax.window_elements, app or None)
        except Exception:  # noqa: BLE001
            els = []
    return _ask_before_click(tool_name, a, label, app, els)


# ---- seeing ---------------------------------------------------------------------------


@tool(
    "ax_tree",
    "Read the frontmost (or named) app's UI as a text tree of elements with ids, roles, titles, "
    "values and screen positions. Use this FIRST to see what's on screen — it's fast and exact. "
    "Element ids (e12) can be used with ax_press, ax_set_value, and click. When it shows little "
    "(a game board like Chess, a map, a canvas), take a screenshot and click at x,y instead.",
    _p({"app": {"type": "string", "description": "App name; omit for the frontmost app"}}),
    parallel_safe=True,
    category="computer",
    untrusted=True,
)
async def ax_tree(a: dict, c: ToolContext) -> str:
    if err := _needs_ax():
        return err
    return await asyncio.to_thread(ax.snapshot, a.get("app"))


@tool(
    "ax_find",
    "Search the current UI tree for elements whose title or value contains the text.",
    _p({"query": {"type": "string"}, "app": {"type": "string"}}, ["query"]),
    parallel_safe=True,
    category="computer",
    untrusted=True,
)
async def ax_find(a: dict, c: ToolContext) -> str:
    if err := _needs_ax():
        return err
    return await asyncio.to_thread(ax.find, a["query"], a.get("app"))


_TAIL = r"\s*[.!]*\s*$"
_LEAD = r"^\s*(?:and\s+|now\s+|please\s+|then\s+)*(?:can\s+you\s+|could\s+you\s+)?"
# Voice scroll at the pointer. "scroll down", "scroll down a bit", "scroll up some more" — but a
# place ("scroll down in Safari", "scroll to the comments") or a task ("… and tell me") goes to
# the agent, which can scroll the right window.
_SCROLL_CTX = r"(?:\s+(?:a\s+(?:bit|little|lot)|some|more|further|again|a\s+page|please|now))*"
_SCROLL = [
    (rf"{_LEAD}scroll\s+down{_SCROLL_CTX}{_TAIL}", {"at_pointer": True, "dy": -8}),
    (rf"{_LEAD}scroll\s+up{_SCROLL_CTX}{_TAIL}", {"at_pointer": True, "dy": 8}),
    (rf"{_LEAD}page\s+down{_TAIL}", {"at_pointer": True, "dy": -30}),
    (rf"{_LEAD}page\s+up{_TAIL}", {"at_pointer": True, "dy": 30}),
]


@tool(
    "screenshot",
    "Take a screenshot: a whole display (display: its number, 1 = main), or a region x,y,w,h in screen "
    "points. Use when ax_tree is not enough (game boards like Chess, maps, drawing and other canvas "
    "apps, Electron apps, images, layout questions). Its note gives the image size and a shot id: to "
    "click, drag or scroll at something in the picture, pass its pixel position in the image with that "
    "shot id and NEO maps it to the screen.",
    _p(
        {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "w": {"type": "integer"},
            "h": {"type": "integer"},
            "display": {"type": "integer"},
        }
    ),
    category="computer",
    slow=True,
    untrusted=True,
)
async def screenshot(a: dict, c: ToolContext) -> ToolOutput:
    import io

    from PIL import Image

    region = None
    if all(a.get(k) is not None for k in ("x", "y", "w", "h")):
        try:
            region = tuple(int(float(a[k])) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            return ToolOutput("Error: x, y, w, h must be integers", ok=False)
        if region[2] <= 0 or region[3] <= 0:
            return ToolOutput("Error: region width and height must be positive", ok=False)
    shown = await asyncio.to_thread(screen.displays)
    display = shown[0]
    if a.get("display") not in (None, "") and not region:
        try:
            want = int(a["display"])
        except (TypeError, ValueError):
            return ToolOutput("Error: display must be a number (1 = the main display)", ok=False)
        display = next((d for d in shown if d.index == want), None)
        if display is None:
            return ToolOutput(f"Error: there's no display {want} ({'; '.join(d.words() for d in shown)})", ok=False)
    try:
        img, scale = await asyncio.to_thread(screen.capture, region, None if region else display)
    except PermissionError as e:
        return ToolOutput(confirm.ask_user(str(e)), ok=False)
    w, h = Image.open(io.BytesIO(img.data)).size
    x0, y0 = (region[0], region[1]) if region else (display.x, display.y)
    shot = pointer.remember(x0, y0, scale, w, h)
    if region:
        what = f"region {region[2]}×{region[3]} pt at screen ({region[0]},{region[1]})"
        others: list = []
    else:
        what = f"all of {display.words()}"
        others = [d for d in shown if d.index != display.index]
    # NEO's own words (the mapping) aren't fenced; the picture is what came from the screen.
    return ToolOutput(pointer.note(shot, what, others), images=[img], fence=False)


@tool(
    "apps_running",
    "List running apps and which one is frontmost.",
    _OBJ,
    parallel_safe=True,
    category="apps",
    fast_path=[
        (
            r"^\s*(?:what(?:'s| is)?\s+(?:apps?\s+(?:are|is)\s+)?(?:open|running)(?:\s+right\s+now)?|"
            r"which\s+apps?\s+(?:are|is)\s+(?:open|running)|list\s+(?:the\s+)?(?:open|running)\s+apps?|"
            r"what\s+apps?\s+do\s+i\s+have\s+open)\s*[?.!]*\s*$",
            {},
        )
    ],
)
async def apps_running(a: dict, c: ToolContext) -> str:
    front, _ = ax.frontmost_app()
    return f"Frontmost: {front}\nRunning: " + ", ".join(ax.running_apps())


# ---- acting ---------------------------------------------------------------------------


def _taint(c: ToolContext, name: str) -> None:
    """Words read off the screen went into this result: the request has read untrusted content."""
    if name not in c.taint:
        c.taint.append(name)


async def _watch(c: ToolContext, pid: int, pts: list[tuple[float, float]]) -> pointer.Look | None:
    """The look before an input a model chose (the user's own commands aren't second-guessed:
    they are watching)."""
    return await pointer.look(pid, pts) if c.origin == "agent" else None


def _element_pid(e: ax.Element) -> int:
    return ax.pid_of(e.ref) if e.ref is not None else 0


@tool(
    "ax_press",
    "Activate a UI element by id from ax_tree (button, link, menu item, checkbox, tab).",
    _p({"id": {"type": "string"}}, ["id"]),
    category="computer",
    quiet=True,
)
async def ax_press(a: dict, c: ToolContext) -> str:
    if err := _needs_ax():
        return err
    e = ax.get(str(a.get("id") or ""))  # resolved once: the element gated is the element pressed
    if not e:
        return f"Error: unknown element {a.get('id')} — ids come from the latest ax_tree; call it again."
    pid = await asyncio.to_thread(_element_pid, e)
    owner = await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""
    if ax.is_neo(pid, owner):
        return "Error: that element is NEO's own window — NEO never presses its own buttons."
    # The app that owns the element, not whatever is in front: clicking NEO's own card to
    # approve brings NEO forward, and the question must stay the same.
    if gate := await _ask_first("ax_press", a, e.title or e.value, owner):
        return gate
    before = await _watch(c, pid, [])
    out = await asyncio.to_thread(ax.press_el, e)
    if before is not None and not out.startswith("Error"):
        out += await _changes(before, pid, c)
    return out


async def _changes(before: pointer.Look, pid: int, c: ToolContext) -> str:
    """What visibly followed a key press or an AX press — only what changed (a press may rightly
    change nothing anyone can see)."""
    now = await pointer.after(before, pid, [])
    out = pointer.compare(before, now, [])
    if out.changed is not True:
        return ""
    text = " " + out.said
    if out.details:
        _taint(c, "screen")
        text += "\n" + fence("screen", out.details)
    return text


@tool(
    "ax_set_value",
    "Put text into a text field or area by element id (read back; typed into it when the field "
    "ignores a direct set). Only for text inputs — press or click checkboxes, sliders and popups.",
    _p({"id": {"type": "string"}, "value": {"type": "string"}}, ["id", "value"]),
    category="computer",
    quiet=True,
)
async def ax_set_value(a: dict, c: ToolContext) -> str:
    if err := _needs_input():
        return err
    e = ax.get(str(a.get("id") or ""))
    if not e:
        return f"Error: unknown element {a.get('id')} — ids come from the latest ax_tree; call it again."
    pid = await asyncio.to_thread(_element_pid, e)
    if ax.is_neo(pid, await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""):
        return "Error: that field is NEO's own window — NEO never types into itself."
    return await asyncio.to_thread(ax.set_el_value, e, str(a.get("value") or ""))


_CLICK_ARGS = {
    "x": {"type": "number"},
    "y": {"type": "number"},
    "shot": {"type": "string", "description": "screenshot id: x,y are pixels of that image"},
    "id": {"type": "string", "description": "element id from ax_tree, instead of x,y"},
    "button": {"type": "string", "enum": ["left", "right", "middle"]},
    "count": {"type": "integer", "description": "1, 2 (double-click) or 3"},
    "modifiers": {"type": "array", "items": {"type": "string", "enum": ["cmd", "shift", "alt", "ctrl"]}},
}


async def _gate_click(a: dict, labels: list[str], owner: str, verb: str = "click") -> str | None:
    for label in dict.fromkeys(x for x in labels if x):
        if gate := await _ask_first(verb, a, label, owner):
            return gate
    return None


def _at(x: float, y: float) -> str:
    return f"({int(round(x))}, {int(round(y))})"


@tool(
    "click",
    "Click on an element id from ax_tree, or at a point: pixels of a screenshot with its shot id, or "
    "screen points. Brings the app there to the front first, never clicks NEO's own window, and "
    "says what changed. button: left|right|middle; count: 1-3; modifiers: cmd, shift, alt, ctrl.",
    _p(_CLICK_ARGS),
    category="computer",
    quiet=True,
)
async def click(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := _needs_input():
        return err
    button, count, mods = str(a.get("button") or "left"), a.get("count", 1), a.get("modifiers") or []
    if count in (None, ""):
        count = 1
    try:
        inp.mods_flags(mods)
        if button not in ("left", "right", "middle"):
            raise ValueError(f"button must be left, right or middle, not {button!r}")
        if isinstance(count, bool) or str(count).strip() not in ("1", "2", "3"):
            raise ValueError(f"count must be 1, 2 or 3, not {count!r}")
    except ValueError as e:
        return f"Error: {e}"
    count = int(str(count).strip())
    try:
        if a.get("id"):
            e = ax.get(str(a["id"]))
            if not e:
                return f"Error: unknown element {a['id']} — call ax_tree first."
            pid = await asyncio.to_thread(_element_pid, e)
            owner = await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""
            if ax.is_neo(pid, owner):
                return "Error: that element is NEO's own window — NEO never clicks its own buttons."
            frame = await asyncio.to_thread(ax.visible_frame, e.ref) if e.ref is not None else (e.x, e.y, e.w, e.h)
            if frame is None:
                return (
                    f"Error: {e.id} isn't on screen now (scrolled out of view, or gone) — scroll to it or look "
                    "again (ax_tree)."
                )
            pts = [(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)]
            t = await pointer.probe(pts)
            labels, where = [e.title or e.value, t.hits[0].label], owner or t.hits[0].app
            head = f"Clicked {e.id} ({e.role.removeprefix('AX')}) at {_at(*pts[0])}"
        else:
            pid = 0
            pts, shot = pointer.points(a, ("x", "y"), c)
            t = await pointer.probe(pts)
            labels, where = [t.hits[0].label], t.hits[0].app
            head = f"{pointer.click_word(button, count)} at {_at(*pts[0])}"
            if shot is not None:
                head += f" (pixel ({float(a['x']):g}, {float(a['y']):g}) of {shot.id})"
        if gate := await _gate_click(a, labels, where):
            return gate
        await pointer.bring(t, pts, want_pid=pid)
    except pointer.Refused as r:
        return ToolOutput(str(r), ok=False)
    h = t.hits[0]
    pid = pid or h.pid
    if h.app:
        head += f" in {h.app}"
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    before = await _watch(c, pid, pts)
    await asyncio.to_thread(inp.click, *pts[0], button=button, count=count, mods=mods)
    if before is None:
        return head
    out = pointer.compare(before, await pointer.after(before, pid, pts), [_at(*pts[0])])
    under = [pointer.words(h)] if not a.get("id") else []
    text, ok = pointer.report(head, out, under)
    if any(under) or out.details:
        _taint(c, "screen")
    return ToolOutput(text, ok=ok)


@tool(
    "drag",
    "Drag from (x1,y1) to (x2,y2): pixels of a screenshot with its shot id, or screen points. Brings "
    "the app there to the front first and says what changed at both ends.",
    _p(
        {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            "shot": {"type": "string", "description": "screenshot id: the points are pixels of that image"},
        },
        ["x1", "y1", "x2", "y2"],
    ),
    category="computer",
    quiet=True,
)
async def drag(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := _needs_input():
        return err
    try:
        pts, shot = pointer.points(a, ("x1", "y1", "x2", "y2"), c)
        t = await pointer.probe(pts)
        src, dst = t.hits
        # A drop can delete (onto the Trash) or send (onto a chat, AirDrop): the same gate as a click.
        what = f"drag “{(src.label or 'that')[:40]}” onto “{(dst.label or 'that spot')[:40]}”"
        commits = [lb for lb in (dst.label, src.label) if lb and policy.commits(lb)]
        if commits:
            where = f" in {dst.app}" if dst.app else ""
            gate = confirm.gated("drag", a, what + where)
            if isinstance(gate, str):
                return gate
        await pointer.bring(t, pts)
    except pointer.Refused as r:
        return ToolOutput(str(r), ok=False)
    pid = t.hits[0].pid
    before = await _watch(c, pid, pts)
    await asyncio.to_thread(inp.drag, *pts[0], *pts[1])
    head = f"Dragged {_at(*pts[0])} → {_at(*pts[1])}"
    if shot is not None:
        head += f" (pixels of {shot.id})"
    if t.hits[0].app:
        head += f" in {t.hits[0].app}"
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    if before is None:
        return head
    out = pointer.compare(before, await pointer.after(before, pid, pts), [f"the start {_at(*pts[0])}", f"the end {_at(*pts[1])}"])
    under = [f"start {pointer.words(src)}" if pointer.words(src) else "", f"end {pointer.words(dst)}" if pointer.words(dst) else ""]
    text, ok = pointer.report(head, out, under)
    if any(under) or out.details:
        _taint(c, "screen")
    return ToolOutput(text, ok=ok)


@tool(
    "scroll",
    "Scroll the content under the pointer (or at x,y: screenshot pixels with its shot id, or screen "
    "points). dy<0 scrolls down, dy>0 up (lines).",
    _p(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "shot": {"type": "string", "description": "screenshot id: x,y are pixels of that image"},
            "dy": {"type": "integer"},
            "dx": {"type": "integer"},
            "at_pointer": {"type": "boolean"},
        }
    ),
    category="computer",
    fast_path=_SCROLL,
    quiet=True,
)
async def scroll(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := _needs_input():
        return err
    try:
        dy = int(float(a.get("dy", -5) if a.get("dy") not in (None, "") else -5))
        dx = int(float(a.get("dx", 0) or 0))
    except (TypeError, ValueError):
        return "Error: dy and dx must be whole numbers (lines)."
    try:
        if not a.get("at_pointer") and a.get("x") is not None and a.get("y") is not None:
            pts, _shot = pointer.points(a, ("x", "y"), c)
        else:
            pts = [await asyncio.to_thread(inp.mouse_position)]  # "scroll down": wherever the pointer is
            h = await asyncio.to_thread(ax.hit, *pts[0])
            if h.pid and ax.is_neo(h.pid, h.app):
                # The pointer rests on NEO's window (the user just clicked the orb): scroll the app
                # they're working in, not NEO's chat.
                frame = await asyncio.to_thread(ax.work_window_frame)
                if frame is None:
                    return ToolOutput("Error: the pointer is over NEO's own window, and no other window is in front to scroll.", ok=False)
                pts = [(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)]
        t = await pointer.probe(pts)
    except pointer.Refused as r:
        return ToolOutput(str(r), ok=False)
    pid = t.hits[0].pid
    before = await _watch(c, pid, pts)
    text = await asyncio.to_thread(inp.scroll, pts[0][0], pts[0][1], dy, dx)
    if before is None:
        return text
    out = pointer.compare(before, await pointer.after(before, pid, pts), [_at(*pts[0])])
    text, ok = pointer.report(str(text) + (f" in {t.hits[0].app}." if t.hits[0].app else "."), out, [])
    return ToolOutput(text, ok=ok)


# ---- dictation lane -------------------------------------------------------------------------
# "type Laptops in the heading", "new line", "type some points about laptops": the words *are*
# the payload, so no model is needed to understand them. These regexes make typing and keys
# zero-LLM fast paths (the reflex still gates them), and composing a brief is one small chat
# call plus typing — never the multi-step agent loop. Anything ambiguous is left to a model:
# a pronoun ("write that down"), a message to someone ("write an email to Sam"), a thing
# ("write the report"). The model knows what "that" is; a regex doesn't.

# "type" is unambiguous. "write" is dictation only for short or quoted text, or with a place
# ("write buy milk in the note"). "put", "enter", "insert" are task verbs.
_TYPE_VERB = r"type"
_WRITE_VERB = r"(?:jot\s+down|write\s+down|write)"
_NOT_DICTATION = (
    # a message to someone is a task for Mail/Messages
    r"(?!(?:an?\s+|the\s+|new\s+|my\s+)*(?:e-?mails?|mails?|messages?|texts?|letters?|repl(?:y|ies)|responses?|dms?|notes?)\s+to\b)"
    r"(?!(?:an?\s+|new\s+)*(?:e-?mails?|mail|messages?|letters?|dms?)\b)"
    r"(?!to\s+\w+)"
    r"(?!(?:me|us)\b)"  # "write me a poem" wants an answer, not dictation
    # pronouns and particles: "write that down", "type it in", "type up the notes"
    r"(?!(?:it|this|that|these|those|them|what|whatever|everything|up|in|out|down|over|back)\b)"
)
_WRITE_NOT_THING = r"(?!(?:the|my|our|his|her|their|your|a|an)\b)"  # "write the report" is a task
_COUNT_NOUN = (
    r"(?:bullet\s+)?(?:points?|bullets?|reasons?|ideas?|lines?|sentences?|tips?|ways?|things?|items?|"
    r"facts?|examples?|questions?|paragraphs?|options?|steps?|names?|titles?|headlines?|taglines?|"
    r"slogans?|jokes?|thoughts?|notes?|words?|pros?|cons?|benefits?|features?)"
)
# Payloads that are a *brief* to write, not text to type verbatim.
_COMPOSE_CUE = (
    r"(?:(?:some|a\s+few|several|a\s+couple(?:\s+of)?|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    rf"\s+(?:\w+\s+){{0,2}}{_COUNT_NOUN}\b"
    rf"|{_COUNT_NOUN}\s+(?:about|on|for|of|related\s+to|regarding)\b"
    r"|an?\s+(?:short|quick|brief|long|nice|polite|formal|casual|simple|small|little|funny|good)?\s*"
    r"(?:list|paragraph|summary|essay|poem|story|sentence|description|caption|bio|intro|introduction|"
    r"outline|recipe|plan|draft|note|haiku|joke|title|heading|tagline|slogan|toast|limerick)s?\b"
    r"|something\b|anything\b)"
)
# Apps a dictation may name as its place (patterns run case-insensitively, so no capital trick).
_APP = (
    r"(?:the\s+)?(?:notes|textedit|pages|numbers|keynote|word|excel|docs|google\s+docs|sheets|safari|"
    r"chrome|firefox|mail|messages|slack|discord|whatsapp|telegram|terminal|vs\s*code|xcode|notion|"
    r"obsidian|reminders|stickies|freeform)(?:\s+app)?"
)
# UI places only (never "the box", "the page", "the line" — those occur in dictated prose).
_PLACE = (
    r"(?:heading|title|header|subject(?:\s+line)?|search\s+(?:bar|box|field)|address\s+bar|url\s+bar|"
    r"body|(?:text\s+)?field|text\s+box|input(?:\s+box)?|editor|cell|note|document|form|"
    r"first\s+line|comment\s+box|message\s+box|chat\s+box)"
)
# "… in the heading", "… into the search bar", "… as the title of the current note (in Notes)".
_TYPE_TARGET = (
    rf"(?:\s+(?:in|into|as|for|on)\s+(?:the|this|my)\s+(?:current\s+|open\s+|new\s+|same\s+)?{_PLACE}"
    r"(?:\s+(?:of|in)\s+(?:the|this|my)\s+(?:current\s+|open\s+|new\s+)?(?:note|document|page|file|email|message|window))?"
    rf"(?:\s+in\s+{_APP})?"
    rf"|\s+in\s+{_APP})?"
)
# "type in the search bar hello": the place may come first as well as last.
_TYPE_PLACE_FIRST = rf"(?:(?:in|into|on)\s+(?:the|this|my)\s+{_PLACE}\s+)?"
# A quoted payload ends at its closing quote: type 'Laptops' in the heading of the note in Notes.
_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|.+?)"
_SHORT_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|\S+(?:\s+\S+){0,5}?)"
_LITERAL_TYPE = (
    rf"{_LEAD}{_TYPE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?!{_COMPOSE_CUE}){_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_LITERAL_WRITE = (
    rf"{_LEAD}{_WRITE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"{_WRITE_NOT_THING}(?!{_COMPOSE_CUE}){_SHORT_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_COMPOSE_TYPE = (
    rf"{_LEAD}(?:{_TYPE_VERB}|{_WRITE_VERB}|draft|compose)\s+{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?P<brief>{_COMPOSE_CUE}.*?){_TYPE_TARGET}{_TAIL}"
)

# ---- keys -------------------------------------------------------------------------------------
# Some chords only mean the right thing in some apps: cmd+r is "reload" in a browser but "reply"
# in Mail; cmd+up is "top of page" in a browser but "parent folder" in Finder. Those carry an
# `app` guard and fail (→ the agent takes over) anywhere else.
_TAB_APPS = {"finder", "terminal", "iterm2", "code", "visual studio code", "xcode", "warp"}  # + every browser


def _browser_names() -> set[str]:
    """The browsers on this Mac, lower-case: the default one, plus every app the inventory finds
    opening web pages (by evidence, not by name)."""
    names: set[str] = set()
    try:
        from neo.decide.inventory import inventory

        names = {a.name.lower() for a in inventory() if "https" in a.handles or a.media_evidence[:1] == ("web browser",)}
    except Exception:  # noqa: BLE001
        pass
    if d := apps.default_browser():
        names.add(d.lower())
    return names
_HOTKEYS: list[tuple[str, dict]] = [
    (
        rf"{_LEAD}(?:press|hit|push)\s+(?:the\s+)?(?P<keys>enter|return|tab|escape|esc|delete|"
        rf"backspace|space|up|down|left|right|home|end)(?:\s+(?:key|arrow|bar))?{_TAIL}",
        {"keys": "<keys>"},
    ),
    (rf"{_LEAD}(?:new|next)\s+line{_TAIL}", {"keys": "return"}),
    (rf"{_LEAD}select\s+all{_TAIL}", {"keys": "cmd+a"}),
    # undo/redo only right after NEO itself typed or pressed something
    (rf"{_LEAD}(?:undo(?:\s+that)?|scratch\s+that){_TAIL}", {"keys": "cmd+z", "after": "input"}),
    (rf"{_LEAD}redo(?:\s+that)?{_TAIL}", {"keys": "cmd+shift+z", "after": "input"}),
    (rf"{_LEAD}copy\s+(?:that|it|this){_TAIL}", {"keys": "cmd+c"}),
    (rf"{_LEAD}paste(?:\s+(?:that|it|this))?{_TAIL}", {"keys": "cmd+v"}),
    (rf"{_LEAD}save(?:\s+(?:that|it|this|the\s+(?:file|note|document)))?{_TAIL}", {"keys": "cmd+s"}),
    # the notes app this user uses (by evidence) while write_note is on; else the legacy guard
    (
        rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+note{_TAIL}",
        {"keys": "cmd+n", "app": "note"},
        lambda f: decide_mode("write_note") == "on",
    ),
    (rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+note{_TAIL}", {"keys": "cmd+n", "app": "Notes"}),
    (rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+document{_TAIL}", {"keys": "cmd+n", "app": "document"}),
    (rf"{_LEAD}(?:open\s+)?(?:a\s+)?new\s+window{_TAIL}", {"keys": "cmd+n"}),
    (rf"{_LEAD}(?:open\s+)?(?:a\s+)?new\s+tab{_TAIL}", {"keys": "cmd+t", "app": "tabs"}),
    (rf"{_LEAD}close\s+(?:this|the|that)\s+(?:window|tab|note|document)(?:\s+please)?{_TAIL}", {"keys": "cmd+w"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?next\s+tab{_TAIL}", {"keys": "ctrl+tab"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?(?:previous|last|prev)\s+tab{_TAIL}", {"keys": "ctrl+shift+tab"}),
    # a browser's own undo for tabs: ⌘⇧T brings back what was closed, most recent first
    (
        rf"{_LEAD}(?:re-?open|bring\s+back|restore|undo\s+clos(?:e|ing))\s+(?:the\s+)?(?:last\s+|previous\s+|recently\s+)?"
        rf"(?:closed\s+)?tab(?:\s+(?:i|you)\s+(?:just\s+)?closed)?{_TAIL}",
        {"keys": "cmd+shift+t", "app": "browser"},
    ),
    *[
        (rf"{_LEAD}(?:go\s+to|switch\s+to|open)\s+(?:the\s+)?(?:{w})\s+tab{_TAIL}", {"keys": f"cmd+{i}", "app": "browser"})
        for i, w in enumerate(
            ("first|1st", "second|2nd", "third|3rd", "fourth|4th", "fifth|5th", "sixth|6th", "seventh|7th", "eighth|8th"), 1
        )
    ],
    *[
        (rf"{_LEAD}(?:go\s+to|switch\s+to)\s+tab\s+(?:number\s+)?{i}{_TAIL}", {"keys": f"cmd+{i}", "app": "browser"})
        for i in range(1, 9)
    ],
    (rf"{_LEAD}(?:open\s+)?(?:an?\s+)?(?:new\s+)?(?:private|incognito)\s+(?:window|tab){_TAIL}", {"keys": "cmd+shift+n", "app": "browser"}),
    (rf"{_LEAD}close\s+(?:all\s+(?:the\s+)?tabs|the\s+whole\s+window|this\s+whole\s+window){_TAIL}", {"keys": "cmd+shift+w", "app": "browser"}),
    (rf"{_LEAD}(?:go\s+to|focus|select|click)\s+(?:on\s+)?(?:the\s+)?(?:address|url|search)\s+bar{_TAIL}", {"keys": "cmd+l", "app": "browser"}),
    (rf"{_LEAD}bookmark\s+(?:this|the)\s+(?:page|tab|site){_TAIL}", {"keys": "cmd+d", "app": "browser"}),
    (
        rf"{_LEAD}(?:hard\s+(?:reload|refresh)|(?:reload|refresh)\s+(?:the\s+page\s+)?without\s+(?:the\s+)?cache){_TAIL}",
        {"keys": "cmd+shift+r", "app": "browser"},
    ),
    # Hinglish
    (rf"{_LEAD}(?:ek\s+)?naya\s+tab\s+(?:kholo|khol\s+do|open\s+karo){_TAIL}", {"keys": "cmd+t", "app": "tabs"}),
    (rf"{_LEAD}(?:ye|yeh|is|iss)\s+tab\s+(?:ko\s+)?(?:band\s+karo|band\s+kar\s+do|close\s+karo|close\s+kar\s+do){_TAIL}", {"keys": "cmd+w"}),
    (
        rf"{_LEAD}(?:(?:band|close)\s+(?:kiya\s+hua|hua)|pichla\s+band|last\s+closed)\s+tab\s+(?:wapas|phir\s+se)\s+"
        rf"(?:kholo|khol\s+do|lao|le\s+aao){_TAIL}",
        {"keys": "cmd+shift+t", "app": "browser"},
    ),
    (rf"{_LEAD}agla\s+tab(?:\s+(?:kholo|dikhao|pe\s+jao))?{_TAIL}", {"keys": "ctrl+tab"}),
    (rf"{_LEAD}pichla\s+tab(?:\s+(?:kholo|dikhao|pe\s+jao))?{_TAIL}", {"keys": "ctrl+shift+tab"}),
    (rf"{_LEAD}(?:page\s+)?(?:refresh|reload)\s+(?:karo|kar\s+do){_TAIL}", {"keys": "cmd+r", "app": "browser"}),
    (rf"{_LEAD}go\s+back(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+[", "app": "browser|finder"}),
    (rf"{_LEAD}go\s+forward(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+]", "app": "browser|finder"}),
    (rf"{_LEAD}(?:reload|refresh)(?:\s+(?:the|this)\s+page)?{_TAIL}", {"keys": "cmd+r", "app": "browser"}),
    (rf"{_LEAD}zoom\s+in{_TAIL}", {"keys": "cmd+="}),
    (rf"{_LEAD}zoom\s+out{_TAIL}", {"keys": "cmd+-"}),
    (rf"{_LEAD}(?:reset\s+(?:the\s+)?zoom|actual\s+size){_TAIL}", {"keys": "cmd+0"}),
    (rf"{_LEAD}(?:(?:enter|exit|leave|toggle)\s+)?full\s*screen{_TAIL}", {"keys": "ctrl+cmd+f"}),
    (rf"{_LEAD}minimi[sz]e(?:\s+(?:this|the|that)\s+window)?{_TAIL}", {"keys": "cmd+m"}),
    (rf"{_LEAD}hide\s+(?:this|it|that|the\s+app|this\s+app){_TAIL}", {"keys": "cmd+h"}),
    (rf"{_LEAD}(?:find|search)\s+(?:on|in)\s+(?:this|the)\s+page{_TAIL}", {"keys": "cmd+f"}),
    (rf"{_LEAD}(?:take|grab|capture)\s+(?:a\s+)?screenshot(?:\s+of\s+(?:this|the\s+screen|my\s+screen)?)?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}screenshot(?:\s+(?:this|the\s+screen|my\s+screen))?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?top{_TAIL}", {"keys": "cmd+up", "app": "browser"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?bottom{_TAIL}", {"keys": "cmd+down", "app": "browser"}),
    (rf"{_LEAD}(?:open\s+)?spotlight{_TAIL}", {"keys": "cmd+space"}),
    (rf"{_LEAD}(?:quit|close)\s+(?:this|the|the\s+current)\s+app{_TAIL}", {"keys": "cmd+q"}),
    (rf"{_LEAD}(?:switch|next)\s+app{_TAIL}", {"keys": "cmd+tab"}),
]

# ---- click by label -----------------------------------------------------------------------------
_KEYNAMES = (
    r"(?:enter|return|tab|escape|esc|delete|backspace|space(?:\s*bar)?|up|down|left|right|home|end|all|"
    r"command|cmd|control|ctrl|option|alt|shift|fn|caps\s+lock|page\s+(?:up|down)|f\d+)"
)
_VAGUE_LABELS = {
    "it", "this", "that", "these", "those", "them", "here", "there", "on", "in", "the", "a", "an",
    "one", "none", "me", "us", "you", "him", "her", "something", "anything", "everything",
    "nothing", "wisely", "again", "away", "around", "through", "twice", "once",
}
_CONTROL_NOUN = r"(?:button|link|tab|icon|menu\s+item|menu|option|checkbox|item)"
_NOT_VAGUE = rf"(?!(?:on\s+)?(?:the\s+)?(?:{'|'.join(sorted(_VAGUE_LABELS))})\s*[.!]*\s*$)"
_CLICK_PATHS: list[tuple[str, dict]] = [
    # quoted: click "Sign in"
    (rf"{_LEAD}(?:click|tap|press|hit|select|choose)\s+(?:on\s+)?(?:the\s+)?[\"“'](?P<label>[^\"”']+)[\"”'](?:\s+{_CONTROL_NOUN})?{_TAIL}", {"label": "<label>"}),
    # click/tap <label> [button] — no key names, no "and …" (that's two things)
    (
        rf"{_LEAD}(?:click|tap)\s+{_NOT_VAGUE}(?:on\s+)?(?:the\s+)?(?!(?:the\s+)?{_KEYNAMES}\b)(?!.*\b(?:and|then)\b)"
        rf"(?P<label>[\w'’&.+-]+(?:\s+[\w'’&.+-]+){{0,4}}?)(?:\s+{_CONTROL_NOUN})?{_TAIL}",
        {"label": "<label>"},
    ),
    # press/hit/select/choose only with a control noun: "press the send button"
    (
        rf"{_LEAD}(?:press|hit|select|choose)\s+(?:on\s+)?(?:the\s+)?(?!(?:the\s+)?{_KEYNAMES}\b)(?!.*\b(?:and|then)\b)"
        rf"(?P<label>[\w'’&.+-]+(?:\s+[\w'’&.+-]+){{0,4}}?)\s+{_CONTROL_NOUN}{_TAIL}",
        {"label": "<label>"},
    ),
]
_DOC_APPS = {"notes", "textedit", "pages", "stickies", "numbers", "keynote", "freeform"}  # legacy's list


def _doc_like(name: str, pid: int | None) -> bool:
    """The front app makes a fresh note or document with ⌘N: by evidence while write_note is on
    (neo.decide.kinds.note.doc_like), else by legacy's list of names."""
    if decide_mode("write_note") != "on":
        return name.lower() in _DOC_APPS
    from neo.decide import inventory
    from neo.decide.kinds import note

    app = inventory.app_for_pid(pid) if pid else None
    return app is not None and note.doc_like(app)


_DICTATE_SYSTEM = (
    "You type on the user's behalf into the document they have open. Reply with ONLY the text to "
    "type: no preamble, no quotes, no markdown headings or bold. For points or a list, one item "
    "per line starting with '- '. Be concise (under 120 words unless asked otherwise) and write in "
    "the user's language."
)
_INPUT_TOOLS = ("type_text", "dictate", "hotkey", "click_text", "ax_set_value")
_UNDO_WINDOW_S = 120.0


_NEVER_TYPE_INTO = {"neo-ui", "neo"}  # NEO's own overlay


async def _guard_front(want: str = "") -> str | None:
    """Keystrokes only ever go to the app NEO is working in.

    `want` (or else the app NEO last opened/activated, if that was recent) must be frontmost —
    brought forward if needed. If macOS won't, nothing is sent and the user is told, instead of
    typing into whatever they happen to be using."""
    goal = want or apps.target()
    front = apps.front_name()
    if goal and front.lower() != goal.lower():
        ok, real = await apps.bring_front(goal, wait=0.3)
        if not ok:
            return confirm.ask_user(
                f"{real} isn't in front ({apps.front_name() or 'another app'} is), "
                f"so I didn't type or press anything — click {real} and say it again."
            )
        front = real
    if front.lower() in _NEVER_TYPE_INTO:
        return confirm.ask_user("click where you want me to type first.")
    return None


async def _ensure_text_focus() -> tuple[str | None, bool]:
    """Put the keyboard focus somewhere typeable in the front window of NEO's target app.

    Returns (error, created) — `created` when a fresh note/document had to be made."""
    if err := await _guard_front():
        return err, False
    if await asyncio.to_thread(ax.focused_is_secure):
        return "Error: the cursor is in a password field — I won't type there.", False
    if await asyncio.to_thread(ax.focused_editable):
        return None, False
    app, pid = ax.frontmost_app()
    area = await asyncio.to_thread(ax.main_text_area)
    created = False
    if area is None and await asyncio.to_thread(_doc_like, app, pid):
        await asyncio.to_thread(inp.hotkey, "cmd+n")  # the front window shows no editor at all: a fresh note/document
        created = True
        await asyncio.sleep(0.7)
        if await asyncio.to_thread(ax.focused_editable):
            return None, created
        area = await asyncio.to_thread(ax.main_text_area)
    if area is None:
        return f"Error: nothing to type into is focused in {app or 'the front app'} — click into a text field first.", created
    if area.focused:  # the editor already has the caret (Notes reports focus on the area, not system-wide)
        return None, created
    # Focus it through Accessibility; only if the app won't, click it — in the part that's on screen
    # (a long document's text area reports its whole length, most of it scrolled out of view).
    if area.ref is None or not await asyncio.to_thread(ax.focus_el, area.ref):
        frame = await asyncio.to_thread(ax.visible_frame, area.ref) if area.ref is not None else (area.x, area.y, area.w, area.h)
        if frame is None:
            return f"Error: the text area in {app or 'the front app'} isn't on screen — click into it first.", created
        spot = (frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)
        try:
            await pointer.probe([spot])  # never NEO's own window
        except pointer.Refused as r:
            return str(r), created
        await asyncio.to_thread(inp.click, *spot)
        await asyncio.sleep(0.2)
        if not await asyncio.to_thread(ax.focused_editable):
            return f"Error: I couldn't put the cursor in {app or 'the front app'}'s text — click into it first.", created
    await asyncio.to_thread(inp.hotkey, "cmd+down")  # dictation appends at the end
    return None, created


async def _line_break(app: str, pid: int, role: str) -> str:
    """How a line break is typed where the cursor is — decided by what's there, never by the
    app's name: "" joins the lines (a single-line field; an app running a shell, where any Return
    runs the line), "return" in a document editor, else shift+return — a new line in a chat box
    or a web editor, never a "send"."""
    if role in ax._SINGLE_LINE:
        return ""
    from neo.tools.computer.focus import hosts_shell

    if await asyncio.to_thread(hosts_shell, pid):
        return ""
    if await asyncio.to_thread(_doc_like, app, pid):
        return "return"
    return "shift+return"


async def _type(text: str) -> None:
    """Type text so every line break is a real new line — and never a "send" or a command run."""
    app, pid = ax.frontmost_app()
    role, _ = await asyncio.to_thread(ax.focused_role)
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")  # a typed \r is a Return
    newline = await _line_break(app, pid, role) if "\n" in text else "return"
    if not newline:
        text = " ".join(part.strip() for part in text.split("\n") if part.strip())
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line:
            await asyncio.to_thread(inp.type_text, line)
        if i < len(lines) - 1:
            await asyncio.to_thread(inp.hotkey, newline)
            await asyncio.sleep(0.03)


@tool(
    "type_text",
    "Type text where the cursor is (dictation). Use for literal text the user wants typed.",
    _p({"text": {"type": "string"}}, ["text"]),
    category="computer",
    fast_path=[(_LITERAL_TYPE, {"text": "<text>"}), (_LITERAL_WRITE, {"text": "<text>"})],
    quiet=True,
    payload=True,
)
async def type_text(a: dict, c: ToolContext) -> str:
    if err := _needs_input():
        return err
    text = str(a["text"])
    err, created = await _ensure_text_focus()
    if err:
        return err
    await _type(text)
    return f"Typed “{text[:80]}”" + (" into a new note." if created else ".")


@tool(
    "dictate",
    "Write short text from a brief (bullet points, a paragraph, a caption…) and type it where "
    "the cursor is. Use when the user wants NEO to come up with the words.",
    _p({"brief": {"type": "string"}}, ["brief"]),
    category="computer",
    slow=True,
    fast_path=[(_COMPOSE_TYPE, {"brief": "<brief>"})],
    quiet=True,
    payload=True,
)
async def dictate(a: dict, c: ToolContext) -> str:
    from neo.providers import brain
    from neo.providers.base import Message

    if ax_err := _needs_input():
        return ax_err
    brief = str(a["brief"]).strip()
    err, created = await _ensure_text_focus()  # check before spending a model call
    if err:
        return err
    turn = await brain("fast").complete([Message.user(brief)], system=_DICTATE_SYSTEM, max_tokens=400)
    text = (turn.text or "").strip().strip("\"“”")
    if not text:
        return "Error: I couldn't come up with anything to type."
    area = await asyncio.to_thread(ax.main_text_area)
    if area is not None and area.value.strip() and not area.value.endswith("\n"):
        text = "\n" + text  # the heading stays a heading; the points start on their own line
    await _type(text)
    first = text.strip().splitlines()[0][:80]
    return f"Typed {len(text.strip().splitlines())} line(s), starting “{first}”" + (" (new note)." if created else ".")


@tool(
    "hotkey",
    "Press a key or chord, e.g. 'return', 'cmd+s', 'escape', 'tab'. Optional app: bring that app to "
    "the front first. The Mac's standard shortcuts - use them instead of clicking or scripting: "
    "browser: new tab cmd+t, close tab cmd+w, reopen closed tab cmd+shift+t, next / previous tab "
    "ctrl+tab / ctrl+shift+tab, tab 1-8 cmd+1..cmd+8, last tab cmd+9, address bar cmd+l, reload cmd+r "
    "(hard reload cmd+shift+r), back / forward cmd+[ / cmd+], new window cmd+n, private window "
    "cmd+shift+n, close window cmd+shift+w, bookmark cmd+d, find on page cmd+f, zoom cmd+= / cmd+- / "
    "cmd+0; any app: select all cmd+a, copy / cut / paste cmd+c / cmd+x / cmd+v, undo / redo cmd+z / "
    "cmd+shift+z, save cmd+s, new document cmd+n, close window cmd+w, quit app cmd+q, hide app cmd+h, "
    "minimize cmd+m, full screen ctrl+cmd+f, switch app cmd+tab, Spotlight cmd+space, screenshot "
    "cmd+shift+3 (an area cmd+shift+4).",
    _p({"keys": {"type": "string"}, "app": {"type": "string"}}, ["keys"]),
    category="computer",
    fast_path=_HOTKEYS,
    quiet=True,
)
async def hotkey(a: dict, c: ToolContext) -> str:
    if err := _needs_input():
        return err
    keys, want = str(a["keys"]), str(a.get("app") or "")
    if a.get("after") == "input":
        from neo.agent.registry import registry as _reg

        last = _reg().last
        if not last or last[0] not in _INPUT_TOOLS or time.time() - last[1] > _UNDO_WINDOW_S:
            return "Error: undo what? I haven't typed anything just now."
    # only "new note" while write_note is on: any other call naming "note" (a model's, a playbook
    # saved before a rollback) presses its own keys, as legacy does
    if want == "note" and _chord(keys) == {"cmd", "n"} and decide_mode("write_note") == "on":
        return await _new_note(c)
    specific = want not in ("", "browser", "browser|finder", "document", "tabs")
    if err := await _guard_front(want if specific else ""):
        return err
    front_name, front_pid = ax.frontmost_app()
    front = front_name.lower()
    browsers = await asyncio.to_thread(_browser_names) if want in ("browser", "browser|finder", "tabs") else set()
    if want == "browser" and front not in browsers:
        return f"Error: '{keys}' only makes sense in a browser, and {front or 'nothing'} is in front."
    if want == "browser|finder" and front not in browsers | {"finder"}:
        return f"Error: going back only makes sense in a browser or Finder, and {front or 'nothing'} is in front."
    if want == "document" and not await asyncio.to_thread(_doc_like, front, front_pid):
        return "Error: which app should the new document be in?"
    # "new tab" from an app without tabs: the user's default browser
    if want == "tabs" and front not in _TAB_APPS | browsers:
        if not (browser := apps.default_browser()):
            return "Error: there's no default browser to open a tab in."
        if err := await _guard_front(browser):
            return err
    try:
        inp.parse_chord(keys)
    except ValueError as e:
        return f"Error: {e}"  # an unknown modifier or key: nothing pressed, never the bare key
    name, pid = ax.frontmost_app()
    if gate := await _ask_before_keys(a, keys, name, c, pid):
        return gate
    before = await _watch(c, pid, [])
    out = await asyncio.to_thread(inp.hotkey, keys)
    if before is not None and not out.startswith("Error"):
        out += await _changes(before, pid, c)  # ⌘N may open a window, or a sheet that waits for an answer
    return out


async def _new_note(c: ToolContext) -> str:
    """"new note" (write_note on): the New command of the notes app this user uses, found by its
    ⌘N shortcut in the app's own menus; the chord itself when the menu doesn't show one."""
    from neo.decide import axkit
    from neo.decide.kinds import note

    app = await note.top_app(c.user_text)
    if app is None:
        return "Error: there's no notes app on this Mac."
    if err := await _guard_front(app.name):
        return err
    pid = await axkit.window_pid(app, 5.0)
    item = await asyncio.to_thread(axkit.menu_item, pid, "n", 0) if pid else None
    if item and await asyncio.to_thread(axkit.perform, item[0], "AXPress"):
        return f"New note in {app.name}."
    return await asyncio.to_thread(inp.hotkey, "cmd+n")


def _chord(keys: str) -> set[str]:
    """The modifiers and key of a chord, by name ({"cmd", "shift", "d"}), however it was written."""
    try:
        flags, key = inp.parse_chord(keys)
    except ValueError:
        return {p.strip().lower() for p in keys.replace(" ", "").split("+") if p.strip()}
    return {name for name, f in inp._MODS.items() if flags & f} | {key}


async def _return_runs_or_sends(pid: int, name: str) -> str:
    """What a Return does where the keyboard focus is, by evidence rather than the app's name:
    "runs" in an app running an interactive shell (a terminal, an editor's terminal), "sends" in
    a multi-line input outside a document editor (a chat box, a web composer), "" otherwise (a
    document, an address or search field)."""
    from neo.tools.computer.focus import hosts_shell

    if await asyncio.to_thread(hosts_shell, pid):
        return "runs"
    role, _ = await asyncio.to_thread(ax.focused_role)
    multi = role == "AXTextArea" or (role not in ax._EDITABLE and role != "" and await asyncio.to_thread(ax.focused_editable))
    if multi and not await asyncio.to_thread(_doc_like, name, pid):
        return "sends"
    return ""


async def _ask_before_keys(a: dict, keys: str, app: str, c: ToolContext, pid: int = 0) -> str | None:
    """Keys that run or send something: Return in a terminal runs whatever is on its command
    line; ⌘⇧D sends the open Mail message; Return in a chat box sends the message — asked first
    once this request has read something untrusted (a page or an email could have asked for it)."""
    ch, low = _chord(keys), app.lower()
    submit = bool(ch & {"return", "enter"}) or ("ctrl" in ch and bool(ch & {"m", "j"}))
    what, detail = "", ""
    does = await _return_runs_or_sends(pid, app) if submit else ""
    if does == "runs":
        text = await asyncio.to_thread(ax.focused_element_text)
        line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        if len(line) > 2000:
            line = "…" + line[-2000:] + "\n(the start of this line is cut off here)"
        what, detail = f"press Return in {app} — it runs the command on its line", line or "(couldn't read the line)"
    elif low == "mail" and ch == {"cmd", "shift", "d"}:
        what = "send the message that's open in Mail"
        _, win = await asyncio.to_thread(ax.focused_window, app)
        detail = f"Window: {ax._str(ax._attr(win, 'AXTitle', ''))}" if win is not None else ""
    elif does == "sends" and c.tainted():
        what = f"press Return in {app} — it sends the message"
    if not what:
        return None
    gate = confirm.gated("hotkey", a, what, detail=detail)
    return gate if isinstance(gate, str) else None


_CLICK_ROLES = ("AXButton", "AXLink", "AXMenuItem", "AXTab", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXCell", "AXRow", "AXStaticText", "AXImage")


@tool(
    "click_text",
    "Click the on-screen element with this label (button, link, tab, menu item…) in the front "
    "window.",
    _p({"label": {"type": "string"}}, ["label"]),
    category="computer",
    quiet=True,
    fast_path=_CLICK_PATHS,
    payload=True,
)
async def click_text(a: dict, c: ToolContext) -> str:
    label = " ".join(str(a["label"]).lower().split())
    if not label or len(label) < 2 or all(w in _VAGUE_LABELS for w in label.split()):
        return f"Error: click what? '{a['label']}' isn't a label I can find."
    if err := _needs_input():
        return err
    if err := await _guard_front():
        return err
    app, frame, els = await asyncio.to_thread(ax.window_elements)
    if frame is None:
        return f"Error: {app or 'the front app'} has no window in front."
    fx, fy, fw, fh = frame
    word = re.compile(rf"(?<![\w]){re.escape(label)}(?![\w])", re.I)

    def visible(e) -> bool:
        return e.w > 0 and e.h > 0 and e.x + e.w > fx and e.y + e.h > fy and e.x < fx + fw and e.y < fy + fh

    cands = [e for e in els if e.enabled and visible(e)]
    exact = [e for e in cands if e.title.lower().strip() == label or (not e.title and e.value.lower().strip() == label)]
    hits = exact or [e for e in cands if word.search(e.title) or word.search(e.value)]
    if not hits:
        return f"Error: nothing in the front window of {app} is called '{a['label']}'."
    if not exact and len({(e.title or e.value).lower() for e in hits}) > 1:
        names = ", ".join(sorted({(e.title or e.value)[:30] for e in hits})[:4])
        return f"Error: several things match '{a['label']}' ({names}) — which one?"
    hits.sort(
        key=lambda e: (
            _CLICK_ROLES.index(e.role) if e.role in _CLICK_ROLES else len(_CLICK_ROLES),
            e.w * e.h,
        )
    )
    e = hits[0]
    name = (e.title or e.value)[:40]
    if gate := await _ask_first("click_text", a, e.title or e.value, app, els):
        return gate
    if "AXPress" in e.actions:
        pressed = await asyncio.to_thread(ax.press_element, e)
        if not pressed.startswith("Error"):
            return f"Clicked '{name}' ({e.role.removeprefix('AX')})"
        code = _AX_CODE.search(pressed)
        # Only a press the app says it doesn't support falls back to the mouse. "Cannot complete"
        # (-25204) comes back when the press happened but the app is busy (it opened a dialog):
        # a second, mouse click would press it twice or land on that dialog.
        if code and int(code.group(1)) == ax._CANNOT_COMPLETE:
            return f"Clicked '{name}' ({e.role.removeprefix('AX')}), but the app didn't answer in time — look before clicking again."
        if code and int(code.group(1)) not in ax._UNSUPPORTED:
            return f"Error: couldn't press '{name}' ({pressed.removeprefix('Error: ')})."
    spot = e.center
    try:
        await pointer.probe([spot])  # never NEO's own window over it
    except pointer.Refused as r:
        return str(r)
    await asyncio.to_thread(inp.click, *spot)
    return f"Clicked '{name}' ({e.role.removeprefix('AX')}) at {tuple(map(int, spot))}"


_AX_CODE = re.compile(r"\((-?\d+)\)")


# ---- apps -----------------------------------------------------------------------------


@tool(
    "open_app",
    "Open an app by name, a URL, or a file/folder path.",
    _p({"target": {"type": "string"}}, ["target"]),
    category="apps",
    # 1–3 word app/site names only ("open safari", "launch vs code", "open github.com");
    # anything with articles, "for", "and", timers etc. is a sentence for the agent.
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:open|launch|switch\s+to|bring\s+up|focus(?:\s+on)?)\s+(?!(?:a|an|the|my|up|it|this|that)\b)"
            r"(?!.*\b(?:please|now|for|and|with|then|to|in|on)\b)"
            r"(?P<target>[\w.+-]+(?:\s+[\w.+-]+){0,2})[\s.!]*$",
            {"target": "<target>"},
        ),
        (  # "<app> kholo", "mera mail khol do"; like the English one, never "<app> mein <site>"
            r"^\s*(?:(?:mera|meri|mere)\s+)?(?!(?:ye|yeh|isko|usko|woh|wo)\b)"
            r"(?!.*\b(?:mein|me|pe|par|se|ka|ki|ke|aur|phir|wala|wali|naya|nayi|naye)\b)"
            r"(?P<target>[\w.+-]+(?:\s+[\w.+-]+){0,2}?)\s+(?:kholo|khol\s+do|khol\s+dijiye)\s*[.!।]*\s*$",
            {"target": "<target>"},
            lambda f: decide_mode("open_app") == "on",
        ),
    ],
    quiet=True,
    early=True,
    idempotent=True,
)
async def open_app(a: dict, c: ToolContext):
    target = str(a["target"]).strip()
    why = policy.open_risk(target)
    if not why and c.tainted() and ("://" in target or "." in target) and policy.url_carries_data(target):
        why = f"the link carries data, after reading {c.tainted()}"
    if why:
        gate = confirm.gated("open_app", a, f"open {target[:80]} — {why}", detail=target)
        if isinstance(gate, str):
            return gate
    m = decide_mode("open_app")
    if m == "on":
        from neo.decide import run
        from neo.decide.kinds import launch

        return run.to_output(await launch.open_target(target))
    out = await apps.open_app(target)
    if m == "shadow":
        from neo.decide import shadow

        with shadow.starting("open_app"):
            from neo.decide.kinds import launch

            ok = not out.startswith(("Error", "NEEDS_"))
            shadow.later("open_app", launch.shadow(target, out), legacy_ok=ok)
    return out


@tool(
    "activate_app",
    "Bring a running app to the front.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
    quiet=True,
    idempotent=True,
)
async def activate_app(a: dict, c: ToolContext) -> str:
    return await apps.activate(a["name"])


@tool(
    "quit_app",
    "Quit an app gracefully.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
    quiet=True,
)
async def quit_app(a: dict, c: ToolContext) -> str:
    return await apps.quit_app(a["name"])


@tool(
    "close",
    "Close what the user names, no questions: a running app (quits it) or browser tabs by site or "
    "title ('close YouTube' closes every YouTube tab). Use this for any 'close / quit X' - never "
    "write AppleScript to close something.",
    _p({"target": {"type": "string"}}, ["target"]),
    category="apps",
    quiet=True,
    idempotent=True,
    # "close youtube", "quit spotify", "close the netflix tab" — not "close this tab/window" or
    # "close the app" (a key press on what's in front: see hotkey)
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:quit|close|kill|exit)\s+(?!(?:this|that|it|all|every)\b)"
            r"(?!(?:the|the\s+current)\s+(?:window|tab|note|document|app)\s*[.!]*\s*$)"
            r"(?P<target>[\w.+'-]+(?:\s+[\w.+'-]+){0,3}?)\s*[.!]*\s*$",
            {"target": "<target>"},
        )
    ],
)
async def close(a: dict, c: ToolContext) -> str:
    from neo.decide.close import close_target

    return await close_target(str(a.get("target") or ""))


_SCRIPT_TRIES = 2  # failed scripts on one app in one task before NEO stops trying that way


@tool(
    "applescript",
    "Run AppleScript and return its result. Only for what an app exposes to scripting that no "
    "dedicated tool covers - not for seeing or clicking in an app's window (ax_tree, screenshot, "
    "click, type_text and hotkey do that). Scripts that only read (get/count/exists/set a camelCase variable) "
    "run straight away; the user approves anything else. After two failures on an app, stop "
    "scripting it.",
    _p({"script": {"type": "string"}}, ["script"]),
    risk="destructive",
    category="apps",
    slow=True,
    untrusted=True,  # file contents and app data it prints are content, never orders
)
async def applescript(a: dict, c: ToolContext) -> str:
    script = str(a.get("script") or "")
    facts = policy.script_facts(script)
    if facts.ui and (err := _needs_ax()):
        return err  # System Events' UI scripting takes Accessibility too
    failed = f"applescript-failed:{(facts.app or '?').casefold()}"
    if grants.count(failed) >= _SCRIPT_TRIES:
        return (
            f"Error: AppleScript already failed {_SCRIPT_TRIES} times on {facts.app or 'this'} in this task, so I "
            "didn't run another. Stop scripting it: look with screenshot and act with click, type_text or "
            "hotkey, or tell the user what's in the way."
        )
    # A script that only reads runs straight away — unless this request has read something
    # untrusted (a page, a mail, the screen), which could be what asked for it.
    if not facts.read_only or c.tainted():
        where = f" in {facts.app}" if facts.app else ""
        what = f"read {facts.app} with an AppleScript" if facts.read_only and facts.app else f"run an AppleScript{where}"
        target = grants.Target(facts.app, facts.danger, tuple(c.taint))
        gate = confirm.gated("applescript", a, what, detail=script, target=target)
        if isinstance(gate, str):
            return gate
    out = await apps.osascript(script, timeout=60)
    if out.startswith("Error"):
        if _AX_DENIED.search(out):
            return _ax_off_reply()
        grants.note(failed)
    return out


def _named_browser(name: str):
    """The installed browser the user named ("chrome", "the arc browser"), by the inventory's
    evidence (it opens web pages); None for anything else ("google", "the internet")."""
    try:
        from neo.decide.inventory import inventory, named_in

        app = named_in(name, inventory())
    except Exception:  # noqa: BLE001
        return None
    if app is None:
        return None
    if "https" in app.handles or (app.media_evidence[:1] == ("web browser",)):
        return app
    return None


@tool(
    "open_url",
    "Open a web page in the user's default browser, or in the browser they named (browser: the "
    "name they said). Use this for any website - never assume a particular browser.",
    _p({"url": {"type": "string"}, "browser": {"type": "string"}}, ["url"]),
    category="web",
    quiet=True,
    idempotent=True,
)
async def open_url(a: dict, c: ToolContext) -> str:
    url = str(a["url"]).strip()
    scheme = url.split(":", 1)[0].lower() if re.match(r"^[a-z][\w+.-]*:", url, re.I) else "https"
    why = "" if scheme in ("http", "https") else f"{scheme}: links run code or open other apps"
    if not why and c.tainted() and policy.url_carries_data(url):
        why = f"the link carries data, after reading {c.tainted()}"
    if why:
        gate = confirm.gated("open_url", a, f"open {url[:80]} — {why}", detail=url)
        if isinstance(gate, str):
            return gate
    browser = str(a.get("browser") or "").strip()
    app = await asyncio.to_thread(_named_browser, browser) if browser else None
    return await apps.open_url(url, app.bundle_id if app else "", app.name if app else "")


# ---- the page the user is looking at -----------------------------------------------------------------
# "summarize this page", "what's on this page", "read me the comments", "is page pe kya hai", "ye page
# summarize karo": the page in the user's own browser, which NEO then answers them from (page_read).
_Q_TAIL = r"(?:\s+(?:for\s+me|please|pls|now))*\s*[.!?]*\s*$"
_PAGE_NOUN = r"(?:web\s*)?(?:page|site|website|tab|article|post|blog(?:\s+post)?)"
# "this page", and "the page" / "the tab" (on screen, the one in front): the page the user is looking
# at. "the article", "the post", "the site" alone may be one NEO fetched or the conversation named,
# which only a model can tell.
_THIS_PAGE = rf"(?:(?:this|that|current|the\s+current)\s+{_PAGE_NOUN}|the\s+(?:web\s*)?(?:page|tab))"
# The comments and replies are a page's (the browser in front: page_read reads no other app).
_COMMENTS = r"(?:comments?|replies)"  # a page's, wherever they are said
_REPLIES = r"(?:comments?|replies|reviews?|responses)"  # a page's when the page is named
_ON_PAGE = rf"\s+(?:on|in|under|of|below|from)\s+(?:this|that|the)\s+(?:{_PAGE_NOUN}|video)"
_HI_PAGE = r"(?:ye|yeh|is|iss|isi)\s+(?:web\s*)?(?:page|site|website|tab|article|post)"
_HI_DO = r"(?:karo|kar\s+do|kardo|kijiye|kar\s+dijiye|do|dijiye|batao|bata\s+do|sunao|suna\s+do)"
_PAGE_PATHS = [
    # summarize / sum up / recap / tl;dr this page, give me a summary of it, what's the gist of it
    (
        rf"{_LEAD}(?:summari[sz]e|sum\s+up|recap|tl;?\s*dr|give\s+me\s+(?:a\s+|the\s+)?(?:quick\s+|short\s+|brief\s+)?"
        rf"(?:summary|gist|tl;?\s*dr|rundown|recap)\s+of|what(?:'s|\s+is)\s+the\s+gist\s+of)\s+"
        rf"(?:{_THIS_PAGE}|(?:the\s+)?{_COMMENTS}(?:{_ON_PAGE})?|(?:the\s+)?{_REPLIES}{_ON_PAGE}){_Q_TAIL}",
        {},
    ),
    # what's on / in this page, what's this page about, what does this article say
    (
        rf"{_LEAD}(?:what(?:'s|\s+is)\s+(?:on|in)\s+{_THIS_PAGE}|"
        rf"what(?:'s|\s+is)\s+(?:this|that|current|the\s+current)\s+{_PAGE_NOUN}(?:\s+(?:all\s+)?about)?|"
        rf"what(?:'s|\s+is)\s+{_THIS_PAGE}\s+(?:all\s+)?about|what\s+does\s+{_THIS_PAGE}\s+say|"
        rf"tell\s+me\s+(?:about|what(?:'s|\s+is)\s+(?:on|in))\s+{_THIS_PAGE}){_Q_TAIL}",
        {},
    ),
    # read me this page, read this article out loud, read me the comments
    (
        rf"{_LEAD}read\s+(?:me\s+|out\s+|aloud\s+)?"
        rf"(?:{_THIS_PAGE}|(?:the\s+)?(?:top\s+)?{_COMMENTS}(?:{_ON_PAGE})?|(?:the\s+)?(?:top\s+)?{_REPLIES}{_ON_PAGE})"
        rf"(?:\s+(?:to\s+me|out(?:\s+loud)?|aloud))?{_Q_TAIL}",
        {},
    ),
    # what are people saying in the comments, what do the reviews on this page say
    (
        rf"{_LEAD}(?:what\s+are\s+(?:people|they)\s+saying\s+in\s+the\s+{_COMMENTS}(?:{_ON_PAGE})?|"
        rf"what\s+(?:are|do)\s+the\s+{_COMMENTS}(?:{_ON_PAGE})?\s+(?:saying|say)|"
        rf"what\s+(?:are|do)\s+the\s+{_REPLIES}{_ON_PAGE}\s+(?:saying|say)){_Q_TAIL}",
        {},
    ),
    # what's the price here / on this page
    (rf"{_LEAD}what(?:'s|\s+is)\s+the\s+(?:price|cost)\s+(?:here|on\s+{_THIS_PAGE}){_Q_TAIL}", {}),
    # Hinglish: "is page pe kya hai", "is page mein kya likha hai", "ye page summarize karo", "is page
    # ka summary do", "ye article padh ke sunao", "comments padh ke sunao", "comments mein kya likha hai"
    (
        rf"{_LEAD}{_HI_PAGE}\s+(?:pe|par|mein|me|main)\s+kya\s+(?:hai|likha\s+(?:hua\s+)?hai|chal\s+raha\s+hai|bola\s+hai)"
        rf"{_Q_TAIL}",
        {},
    ),
    (rf"{_LEAD}{_HI_PAGE}\s+(?:ko\s+|ka\s+|ki\s+)?(?:summari[sz]e|summary|saar|saransh)\s+{_HI_DO}{_Q_TAIL}", {}),
    (
        rf"{_LEAD}(?:{_HI_PAGE}|(?:is\s+page\s+(?:ke|ki)\s+)?{_COMMENTS})\s+(?:ko\s+)?"
        rf"(?:padh\s+ke\s+sunao|padhkar\s+sunao|padh\s+kar\s+sunao|padho|padh\s+do|sunao){_Q_TAIL}",
        {},
    ),
    (
        rf"{_LEAD}(?:log\s+)?{_COMMENTS}\s+(?:mein|me|main)\s+kya\s+(?:likha\s+hai|hai|keh\s+rahe\s+hain|bol\s+rahe\s+hain)"
        rf"{_Q_TAIL}",
        {},
    ),
    (rf"{_LEAD}(?:yahan|yaha|{_HI_PAGE}\s+(?:pe|par))\s+(?:price|daam|keemat|kimat)\s+kya\s+hai{_Q_TAIL}", {}),
    # Devanagari: "इस पेज पर क्या है", "ये पेज समराइज़ करो"
    (r"^\s*(?:इस|ये|यह)\s+पेज\s+(?:पे|पर|में)\s+क्या\s+(?:है|लिखा\s+है)\s*[?।.!]*\s*$", {}),
    (r"^\s*(?:इस|ये|यह)\s+पेज\s+(?:को\s+|का\s+)?(?:समराइज़|समराइज|सारांश)\s+(?:करो|कर\s+दो|बताओ)\s*[?।.!]*\s*$", {}),
]


@tool(
    "page_read",
    "Read the web page the user is looking at in their own browser (the front tab of whichever "
    "browser is in front): its URL, title and main text, or, when it has little text (a video, a "
    "picture, a PDF) or Accessibility is off, what a screenshot of it shows. Use it for 'summarize "
    "this page', 'what's on this page', 'read me the comments', 'what's the price here', then answer "
    "from it. With another app in front it reads nothing and says so: look at that app instead. Not "
    "for NEO's own browser window, and not for a link the user isn't looking at (web_fetch).",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    slow=True,  # a wait for the page's tree, a walk of it, maybe a vision call: Live talks meanwhile
    chain=False,  # "open X and summarize it" would read the page before it loads
    fast_path=_PAGE_PATHS,
)
async def page_read(a: dict, c: ToolContext):
    from neo.decide import run
    from neo.decide.kinds import site

    # The user's own words chose it (a fast path, or Laya's pick): no model is waiting to read the
    # page, so NEO answers them from it. A model's call gets the page itself.
    ask = " ".join(c.user_text.split()) if c.origin == "user" else ""
    return run.to_output(await site.page_read(ask=ask))


@tool(
    "safari_read",
    "Read the web page the user is looking at in their browser (the old name of page_read).",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    hidden=True,  # page_read reads whichever browser is in front; this name stays for old playbooks
)
async def safari_read(a: dict, c: ToolContext):
    return await page_read(a, c)


# The task tools (neo/tools/tasks.py) were registered right here before they moved: loading them
# at this point keeps the registry's order, which the tool picker's keyword fallback breaks ties
# by and Live declares its tools in.
from neo.tools import tasks  # noqa: E402, F401


@tool(
    "finder_reveal",
    "Reveal a file or folder in Finder.",
    _p({"path": {"type": "string"}}, ["path"]),
    category="apps",
    quiet=True,
)
async def finder_reveal(a: dict, c: ToolContext) -> str:
    return await apps.finder_reveal(a["path"])


# ---- shell & files ----------------------------------------------------------------------


@tool(
    "shell",
    "Run a zsh command (cwd defaults to home). Read-only commands (ls, cat, grep, git status…) run "
    "straight away; anything that changes something is shown to the user to approve first.",
    _p(
        {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number"},
        },
        ["command"],
    ),
    risk="destructive",
    category="files",
    slow=True,
    untrusted=True,  # file contents and app data it prints are content, never orders
)
async def shell_tool(a: dict, c: ToolContext) -> str:
    return await shell.run(a["command"], cwd=a.get("cwd"), timeout=float(a.get("timeout", 60)), args=a)


@tool(
    "read_file",
    "Read a text file (numbered lines), a PDF, or list a directory.",
    _p({"path": {"type": "string"}, "start": {"type": "integer"}, "lines": {"type": "integer"}}, ["path"]),
    parallel_safe=True,
    category="files",
    untrusted=True,
)
async def read_file(a: dict, c: ToolContext) -> str:
    return shell.read_file(a["path"], start=int(a.get("start") or 1), lines=int(a.get("lines") or 400))


@tool(
    "write_file",
    "Create a file with content. New files in Desktop/Documents/Downloads are written straight away; "
    "overwrites and anything elsewhere are shown to the user to approve first.",
    _p({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    risk="destructive",
    category="files",
)
async def write_file(a: dict, c: ToolContext) -> str:
    return shell.write_file(a["path"], a["content"], a)


@tool(
    "edit_file",
    "Replace one exact occurrence of old_string with new_string in a file (the user approves the change).",
    _p(
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    risk="destructive",
    category="files",
)
async def edit_file(a: dict, c: ToolContext) -> str:
    return shell.edit_file(a["path"], a["old_string"], a["new_string"], a)


@tool(
    "trash",
    "Move a file or folder to the Trash (the user approves first).",
    _p({"path": {"type": "string"}}, ["path"]),
    risk="destructive",
    category="files",
)
async def trash(a: dict, c: ToolContext) -> str:
    return shell.trash(a["path"], a)
