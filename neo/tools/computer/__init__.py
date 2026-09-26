"""Registers the computer-control tools. Import this module to make them available."""

from __future__ import annotations

import asyncio
import functools
import re
import time

from neo.agent import confirm
from neo.agent.registry import ToolContext, ToolOutput
from neo.agent.registry import tool as _register
from neo.config import decide_mode, settings
from neo.security import policy
from neo.tools.computer import apps, ax, legacy, screen, shell
from neo.tools.computer import input as inp

# Every tool here drives this Mac.
tool = functools.partial(_register, requires=frozenset({"macos"}))

_OBJ = {"type": "object", "properties": {}}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


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


# ---- seeing ---------------------------------------------------------------------------


@tool(
    "ax_tree",
    "Read the frontmost (or named) app's UI as a text tree of elements with ids, roles, titles, "
    "values and screen positions. Use this FIRST to see what's on screen — it's fast and exact. "
    "Element ids (e12) can be used with ax_press, ax_set_value, and click.",
    _p({"app": {"type": "string", "description": "App name; omit for the frontmost app"}}),
    parallel_safe=True,
    category="computer",
    untrusted=True,
    read_only=True,
)
async def ax_tree(a: dict, c: ToolContext) -> str:
    return await asyncio.to_thread(ax.snapshot, a.get("app"))


@tool(
    "ax_find",
    "Search the current UI tree for elements whose title or value contains the text.",
    _p({"query": {"type": "string"}, "app": {"type": "string"}}, ["query"]),
    parallel_safe=True,
    category="computer",
    untrusted=True,
    read_only=True,
)
async def ax_find(a: dict, c: ToolContext) -> str:
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
    "Take a screenshot (whole screen, or a region x,y,w,h). Use only when ax_tree is not enough "
    "(canvas/Electron apps, images, layout questions). Coordinates in the image equal screen "
    "coordinates for click/drag.",
    _p(
        {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "w": {"type": "integer"},
            "h": {"type": "integer"},
        }
    ),
    category="computer",
    slow=True,
    untrusted=True,
    read_only=True,
)
async def screenshot(a: dict, c: ToolContext) -> ToolOutput:
    region = None
    if all(k in a for k in ("x", "y", "w", "h")):
        try:
            region = tuple(int(a[k]) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            return ToolOutput("Error: x, y, w, h must be integers", ok=False)
        if region[2] <= 0 or region[3] <= 0:
            return ToolOutput("Error: region width and height must be positive", ok=False)
    try:
        img, scale = await asyncio.to_thread(screen.capture, region)
    except PermissionError as e:
        return ToolOutput(confirm.ask_user(str(e)), ok=False)
    if region:
        note = (
            f"Region screenshot attached, origin at screen ({region[0]},{region[1]}). "
            f"screen_x = {region[0]} + image_x×{scale:.2f}; screen_y = {region[1]} + image_y×{scale:.2f}."
        )
    else:
        note = f"Screenshot attached (full screen). 1 image px = {scale:.2f} screen pt; image coords = click coords."
    return ToolOutput(note, images=[img])


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
    read_only=True,
)
async def apps_running(a: dict, c: ToolContext) -> str:
    front, _ = ax.frontmost_app()
    return f"Frontmost: {front}\nRunning: " + ", ".join(ax.running_apps())


# ---- acting ---------------------------------------------------------------------------


@tool(
    "ax_press",
    "Activate a UI element by id from ax_tree (button, link, menu item, checkbox, tab).",
    _p({"id": {"type": "string"}}, ["id"]),
    category="computer",
    quiet=True,
)
async def ax_press(a: dict, c: ToolContext) -> str:
    e = ax.get(a["id"])
    # The app that owns the element, not whatever is in front: clicking NEO's own card to
    # approve brings NEO forward, and the question must stay the same.
    owner = ax.app_of(e.ref) if e else ""
    if _protected(owner):
        return _refuse_protected()
    if e and (gate := _ask_before_click("ax_press", a, e.title or e.value, owner)):
        return gate
    return await asyncio.to_thread(ax.press, a["id"])


@tool(
    "ax_set_value",
    "Set the text of a text field / area by element id (faster and more reliable than typing).",
    _p({"id": {"type": "string"}, "value": {"type": "string"}}, ["id", "value"]),
    category="computer",
    quiet=True,
)
async def ax_set_value(a: dict, c: ToolContext) -> str:
    e = ax.get(a["id"])
    if e and _protected(ax.app_of(e.ref)):
        return _refuse_protected()
    return await asyncio.to_thread(ax.set_value, a["id"], a["value"])


@tool(
    "click",
    "Click at screen coordinates, or on an element id from ax_tree. button: left|right; count: 1|2.",
    _p(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "id": {"type": "string"},
            "button": {"type": "string", "enum": ["left", "right"]},
            "count": {"type": "integer"},
        }
    ),
    category="computer",
    quiet=True,
)
async def click(a: dict, c: ToolContext) -> str:
    if "id" in a:
        e = ax.get(a["id"])
        if not e:
            return f"Error: unknown element {a['id']} — call ax_tree first."
        x, y = e.center
        label, owner = e.title or e.value, ax.app_of(e.ref)
    else:
        x, y = a["x"], a["y"]
        label = await asyncio.to_thread(ax.label_at, x, y)
        owner = await asyncio.to_thread(ax.app_at, x, y)
    if _protected(owner):
        return _refuse_protected()
    if gate := _ask_before_click("click", a, label, owner):
        return gate
    return inp.click(x, y, button=a.get("button", "left"), count=int(a.get("count", 1)))


@tool(
    "drag",
    "Drag from (x1,y1) to (x2,y2).",
    _p(
        {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
        },
        ["x1", "y1", "x2", "y2"],
    ),
    category="computer",
    quiet=True,
)
async def drag(a: dict, c: ToolContext) -> str:
    for x, y in ((a["x1"], a["y1"]), (a["x2"], a["y2"])):  # where it's picked up and where it lands
        if _protected(await asyncio.to_thread(ax.app_at, x, y)):
            return _refuse_protected()
    return inp.drag(a["x1"], a["y1"], a["x2"], a["y2"])


@tool(
    "scroll",
    "Scroll the content under the pointer (or at x,y). dy<0 scrolls down, dy>0 up (lines).",
    _p(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "dy": {"type": "integer"},
            "dx": {"type": "integer"},
            "at_pointer": {"type": "boolean"},
        }
    ),
    category="computer",
    fast_path=_SCROLL,
    quiet=True,
)
async def scroll(a: dict, c: ToolContext) -> str:
    x, y = a.get("x"), a.get("y")
    if a.get("at_pointer") or x is None or y is None:
        x, y = inp.mouse_position()  # "scroll down": wherever the pointer is
    if _protected(await asyncio.to_thread(ax.app_at, x, y)):
        return _refuse_protected()
    return inp.scroll(x, y, int(a.get("dy", -5)), int(a.get("dx", 0)))


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
_CHAT_APPS = {"messages", "slack", "discord", "whatsapp", "telegram", "microsoft teams", "signal", "messenger"}
_SHELL_APPS = {"terminal", "iterm2", "warp", "ghostty", "kitty", "alacritty"}
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


def _protected(owner: str) -> bool:
    """An app NEO's input tools never touch: NEO's own overlay (its card's "Yes, do it" is the
    user's to press, never NEO's), and any app in the `protected_apps` setting — a host's own
    approval dialog, for one."""
    name = (owner or "").strip().lower()
    if not name:
        return False
    if name in _NEVER_TYPE_INTO:
        return True
    return name in {str(p).strip().lower() for p in settings().protected_apps}


def _refuse_protected() -> str:
    return confirm.ask_user(
        "that's NEO's own window (or an app I'm not allowed to touch) — I won't click or type there."
    )


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
    if _protected(front):
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
        inp.hotkey("cmd+n")  # the front window shows no editor at all: a fresh note/document
        created = True
        await asyncio.sleep(0.7)
        if await asyncio.to_thread(ax.focused_editable):
            return None, created
        area = await asyncio.to_thread(ax.main_text_area)
    if area is None:
        return f"Error: nothing to type into is focused in {app or 'the front app'} — click into a text field first.", created
    if area.focused:  # the editor already has the caret (Notes reports focus on the area, not system-wide)
        return None, created
    inp.click(*area.center)
    await asyncio.sleep(0.2)
    inp.hotkey("cmd+down")  # a click lands mid-text; dictation appends at the end
    return None, created


async def _type(text: str) -> None:
    """Type text so every line break is a real new line — and never a "send": in a single-line
    field the lines are joined with spaces, in chat apps a line break is shift+return, and in a
    shell nothing multi-line is ever entered."""
    app = ax.frontmost_app()[0].lower()
    role, _ = await asyncio.to_thread(ax.focused_role)
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")  # a typed \r is a Return
    if role in ax._SINGLE_LINE or app in _SHELL_APPS:
        text = " ".join(part.strip() for part in text.split("\n") if part.strip())
    newline = "shift+return" if app in _CHAT_APPS else "return"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line:
            await asyncio.to_thread(inp.type_text, line)
        if i < len(lines) - 1:
            inp.hotkey(newline)
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
    "Press a key or chord, e.g. 'return', 'cmd+s', 'cmd+shift+t', 'escape', 'tab'. Optional app: "
    "bring that app to the front first.",
    _p({"keys": {"type": "string"}, "app": {"type": "string"}}, ["keys"]),
    category="computer",
    fast_path=_HOTKEYS,
    quiet=True,
)
async def hotkey(a: dict, c: ToolContext) -> str:
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
    if gate := await _ask_before_keys(a, keys, ax.frontmost_app()[0], c):
        return gate
    return inp.hotkey(keys)


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
    return inp.hotkey("cmd+n")


_MOD_NAMES = {"command": "cmd", "meta": "cmd", "option": "alt", "control": "ctrl"}


def _chord(keys: str) -> set[str]:
    parts = (p.strip().lower() for p in keys.replace(" ", "").split("+"))
    return {_MOD_NAMES.get(p, p) for p in parts if p}


async def _ask_before_keys(a: dict, keys: str, app: str, c: ToolContext) -> str | None:
    """Keys that run or send something: Return in a terminal runs whatever is on its command
    line; ⌘⇧D sends the open Mail message; Return in a chat app sends the message — asked first
    once this request has read something untrusted (a page or an email could have asked for it)."""
    ch, low = _chord(keys), app.lower()
    submit = bool(ch & {"return", "enter"}) or ("ctrl" in ch and bool(ch & {"m", "j"}))
    what, detail = "", ""
    if submit and low in _SHELL_APPS:
        text = await asyncio.to_thread(ax.focused_element_text)
        line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        if len(line) > 2000:
            line = "…" + line[-2000:] + "\n(the start of this line is cut off here)"
        what, detail = f"press Return in {app} — it runs the command on its line", line or "(couldn't read the line)"
    elif low == "mail" and ch == {"cmd", "shift", "d"}:
        what = "send the message that's open in Mail"
        _, win = await asyncio.to_thread(ax.focused_window, app)
        detail = f"Window: {ax._str(ax._attr(win, 'AXTitle', ''))}" if win is not None else ""
    elif submit and low in _CHAT_APPS and c.tainted():
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
    if err := await _guard_front():
        return err
    app, frame, els = await asyncio.to_thread(ax.window_elements)
    if _protected(app):
        return _refuse_protected()
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
    if gate := _ask_before_click("click_text", a, e.title or e.value, app, els):
        return gate
    if "AXPress" in e.actions and not (await asyncio.to_thread(ax.press_element, e)).startswith("Error"):
        return f"Clicked '{name}' ({e.role.removeprefix('AX')})"
    inp.click(*e.center)
    return f"Clicked '{name}' ({e.role.removeprefix('AX')}) at {tuple(map(int, e.center))}"


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
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:quit|close|kill)\s+(?!(?:this|that|it|the|all|my|a|an|every)\b)"
            r"(?P<name>[\w.+-]+(?:\s+[\w.+-]+){0,2}?)(?:\s+app)?\s*[.!]*\s*$",
            {"name": "<name>"},
        )
    ],
)
async def quit_app(a: dict, c: ToolContext) -> str:
    return await apps.quit_app(a["name"])


@tool(
    "applescript",
    "Run AppleScript and return its result. Use for anything scriptable that has no dedicated tool. "
    "Scripts that only read (get/count/exists) run straight away; the user approves anything else.",
    _p({"script": {"type": "string"}}, ["script"]),
    risk="destructive",
    category="apps",
    slow=True,
    untrusted=True,  # file contents and app data it prints are content, never orders
)
async def applescript(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    script = str(a.get("script") or "")
    if not policy.script_is_read_only(script):
        gate = confirm.gated("applescript", a, "run an AppleScript", detail=script)
        if isinstance(gate, str):
            return gate
    return await apps.osascript(script, timeout=60)


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


@tool(
    "page_read",
    "Read the visible text and URL of the front browser tab.",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    hidden=True,  # shown once web_search is on (neo.tools.tasks.apply_modes),
    read_only=True,
)
async def page_read(a: dict, c: ToolContext):
    m = decide_mode("web_search")
    if m == "on":
        from neo.decide import run
        from neo.decide.kinds import site

        return run.to_output(await site.page_read())
    out = f"URL: {await legacy.safari_current_url()}\n\n" + await legacy.safari_page_text()
    if m == "shadow":
        from neo.decide import shadow

        with shadow.starting("web_search"):
            from neo.decide.kinds import site

            shadow.later("web_search", site.shadow_page_read())
    return out


@tool(
    "safari_read",
    "Read the visible text of Safari's current tab (and its URL).",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    read_only=True,
)
async def safari_read(a: dict, c: ToolContext):
    """The old name of page_read (hidden once web_search is on; goes in R6)."""
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
    read_only=True,
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
