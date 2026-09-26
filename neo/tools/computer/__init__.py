"""Registers the computer-control tools. Import this module to make them available."""

from __future__ import annotations

import asyncio

from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.tools.computer import apps, ax, screen, shell
from neo.tools.computer import input as inp

_OBJ = {"type": "object", "properties": {}}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


# ---- seeing ---------------------------------------------------------------------------


@tool(
    "ax_tree",
    "Read the frontmost (or named) app's UI as a text tree of elements with ids, roles, titles, "
    "values and screen positions. Use this FIRST to see what's on screen — it's fast and exact. "
    "Element ids (e12) can be used with ax_press, ax_set_value, and click.",
    _p({"app": {"type": "string", "description": "App name; omit for the frontmost app"}}),
    parallel_safe=True,
    category="computer",
)
async def ax_tree(a: dict, c: ToolContext) -> str:
    return await asyncio.to_thread(ax.snapshot, a.get("app"))


@tool(
    "ax_find",
    "Search the current UI tree for elements whose title or value contains the text.",
    _p({"query": {"type": "string"}, "app": {"type": "string"}}, ["query"]),
    parallel_safe=True,
    category="computer",
)
async def ax_find(a: dict, c: ToolContext) -> str:
    return await asyncio.to_thread(ax.find, a["query"], a.get("app"))


_TAIL = r"\s*[.!]*\s*$"
_LEAD = r"^\s*(?:and\s+|now\s+|please\s+|then\s+)*(?:can\s+you\s+|could\s+you\s+)?"
# Voice scroll at the pointer (x/y = -1 → wherever the mouse is).
# "scroll down", "scroll down a bit", "scroll down the page in safari" — but not "scroll down and
# tell me…" (a task).
_SCROLL_CTX = r"(?:\s+(?!.*\b(?:and|then|until|till)\b)[\w' ]{0,40})?"
_SCROLL = [
    (rf"{_LEAD}scroll\s+down{_SCROLL_CTX}{_TAIL}", {"x": -1, "y": -1, "dy": -8}),
    (rf"{_LEAD}scroll\s+up{_SCROLL_CTX}{_TAIL}", {"x": -1, "y": -1, "dy": 8}),
    (rf"{_LEAD}page\s+down{_TAIL}", {"x": -1, "y": -1, "dy": -30}),
    (rf"{_LEAD}page\s+up{_TAIL}", {"x": -1, "y": -1, "dy": 30}),
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
        return ToolOutput(f"NEEDS_USER: {e}", ok=False)
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
    return await asyncio.to_thread(ax.press, a["id"])


@tool(
    "ax_set_value",
    "Set the text of a text field / area by element id (faster and more reliable than typing).",
    _p({"id": {"type": "string"}, "value": {"type": "string"}}, ["id", "value"]),
    category="computer",
    quiet=True,
)
async def ax_set_value(a: dict, c: ToolContext) -> str:
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
    else:
        x, y = a["x"], a["y"]
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
        }
    ),
    category="computer",
    fast_path=_SCROLL,
    quiet=True,
)
async def scroll(a: dict, c: ToolContext) -> str:
    x, y = a.get("x"), a.get("y")
    if x is None or y is None or x < 0 or y < 0:
        x, y = inp.mouse_position()  # "scroll down": wherever the pointer is
    return inp.scroll(x, y, int(a.get("dy", -5)), int(a.get("dx", 0)))


# ---- dictation lane -------------------------------------------------------------------------
# "type Laptops in the heading", "new line", "type some points about laptops": the words *are*
# the payload, so no model is needed to understand them. These regexes make typing and keys
# zero-LLM fast paths (the reflex still gates them), and composing a brief is one small chat
# call plus typing — never the multi-step agent loop.

# "type" is unambiguous. "write" is dictation only for short or quoted text, or with a place
# ("write buy milk in the note"); "write the report and email it to Sam" is a task. "put",
# "enter", "insert" are task verbs ("put the highlights in a note").
_TYPE_VERB = r"(?:type)"
_WRITE_VERB = r"(?:jot\s+down|write\s+down|write)"
# "write an email to Sam …" is a task for Mail, not words to type where the cursor is.
_NOT_DICTATION = (
    r"(?!an?\s+(?:new\s+)?(?:e-?mail|mail|message|text|letter|reply|response|note|dm)\s+to\b)"
    r"(?!(?:me|us)\b)"  # "write me a poem" is a request for an answer, not dictation
)
# Payloads that are a *brief* to write, not text to type verbatim.
_COMPOSE_CUE = (
    r"(?:some|a\s+few|several|a\s+couple(?:\s+of)?|bullet(?:\s+points?)?|points?|"
    r"(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+\w+|"
    r"an?\s+(?:short|quick|brief|long|nice|polite|formal|casual|simple|small|little)?\s*"
    r"(?:list|paragraph|summary|essay|poem|story|sentence|line|description|caption|bio|"
    r"intro|introduction|outline|recipe|plan|draft|note|haiku|joke|title|heading)s?\b|"
    r"an?\s+\w+\s+(?:about|on)\b|something|anything)"
)
# Apps a dictation may name as its place (patterns run case-insensitively, so no capital trick).
_APP = (
    r"(?:the\s+)?(?:notes|textedit|pages|numbers|keynote|word|excel|docs|google\s+docs|sheets|safari|"
    r"chrome|firefox|mail|messages|slack|discord|whatsapp|telegram|terminal|vs\s*code|xcode|notion|"
    r"obsidian|reminders|stickies|freeform|\w+\s+app)(?:\s+app)?"
)
# "… in the heading", "… into the search bar", "… as the title of the current note".
_TYPE_TARGET = (
    r"(?:\s+(?:in|into|as|for|on)\s+(?:the\s+|this\s+|my\s+)?(?:\w+\s+)?"
    r"(?:heading|title|header|subject|body|field|box|bar|notes?|document|editor|input|line|text|"
    r"search|page|cell|form|message|comment|reply)"
    r"(?:\s+(?:of|in)\s+(?:the\s+|this\s+|my\s+)?(?:current\s+|open\s+|new\s+)?\w+)?"
    rf"(?:\s+in\s+{_APP})?"  # "… in Notes", "… in the Notes app"
    rf"|\s+in\s+{_APP})?"  # or just "… in Notes"
)
# A quoted payload ends at its closing quote: type 'Laptops' in the heading of the note in Notes.
_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|.+?)"
# "type in the search bar hello": the place may come first as well as last.
_TYPE_PLACE_FIRST = (
    r"(?:(?:in|into|on)\s+(?:the\s+|this\s+|my\s+)?(?:\w+\s+)?"
    r"(?:heading|title|header|subject|body|field|box|bar|notes?|document|editor|input|search|form|"
    r"cell|comment|reply)\s+)?"
)
_SHORT_PAYLOAD = r"(?P<text>\"[^\"]+\"|“[^”]+”|'[^']+'|\S+(?:\s+\S+){0,5}?)"
_LITERAL_TYPE = (
    rf"{_LEAD}{_TYPE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?!{_COMPOSE_CUE}){_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_LITERAL_WRITE = (
    rf"{_LEAD}{_WRITE_VERB}\s+(?:the\s+(?:words?|text)\s+)?{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?!{_COMPOSE_CUE}){_SHORT_PAYLOAD}{_TYPE_TARGET}{_TAIL}"
)
_COMPOSE_TYPE = (
    rf"{_LEAD}(?:{_TYPE_VERB}|{_WRITE_VERB}|draft|compose)\s+{_TYPE_PLACE_FIRST}{_NOT_DICTATION}"
    rf"(?P<brief>{_COMPOSE_CUE}.*?){_TYPE_TARGET}{_TAIL}"
)
_HOTKEYS: list[tuple[str, dict]] = [
    (
        rf"{_LEAD}(?:press|hit|push)\s+(?:the\s+)?(?P<keys>enter|return|tab|escape|esc|delete|"
        rf"backspace|space|up|down|left|right|home|end)(?:\s+(?:key|arrow))?{_TAIL}",
        {"keys": "<keys>"},
    ),
    (rf"{_LEAD}(?:new|next)\s+line{_TAIL}", {"keys": "return"}),
    (rf"{_LEAD}select\s+all{_TAIL}", {"keys": "cmd+a"}),
    (rf"{_LEAD}(?:undo(?:\s+that)?|scratch\s+that){_TAIL}", {"keys": "cmd+z"}),
    (rf"{_LEAD}redo(?:\s+that)?{_TAIL}", {"keys": "cmd+shift+z"}),
    (rf"{_LEAD}copy\s+(?:that|it|this){_TAIL}", {"keys": "cmd+c"}),
    (rf"{_LEAD}paste(?:\s+(?:that|it|this))?{_TAIL}", {"keys": "cmd+v"}),
    (rf"{_LEAD}save(?:\s+(?:that|it|this|the\s+(?:file|note|document)))?{_TAIL}", {"keys": "cmd+s"}),
    (rf"{_LEAD}(?:make\s+|create\s+|start\s+|open\s+)?(?:a\s+)?new\s+(?:note|document|file|window){_TAIL}", {"keys": "cmd+n"}),
    (rf"{_LEAD}(?:open\s+)?(?:a\s+)?new\s+tab{_TAIL}", {"keys": "cmd+t"}),
    (rf"{_LEAD}close\s+(?:this|the|that)\s+(?:window|tab|note|document)(?:\s+please)?{_TAIL}", {"keys": "cmd+w"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?next\s+tab{_TAIL}", {"keys": "ctrl+tab"}),
    (rf"{_LEAD}(?:go\s+to\s+the\s+)?(?:previous|last|prev)\s+tab{_TAIL}", {"keys": "ctrl+shift+tab"}),
    (rf"{_LEAD}go\s+back(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+["}),
    (rf"{_LEAD}go\s+forward(?:\s+a\s+page)?{_TAIL}", {"keys": "cmd+]"}),
    (rf"{_LEAD}(?:reload|refresh)(?:\s+(?:the|this)\s+page)?{_TAIL}", {"keys": "cmd+r"}),
    (rf"{_LEAD}zoom\s+in{_TAIL}", {"keys": "cmd+="}),
    (rf"{_LEAD}zoom\s+out{_TAIL}", {"keys": "cmd+-"}),
    (rf"{_LEAD}(?:reset\s+(?:the\s+)?zoom|actual\s+size){_TAIL}", {"keys": "cmd+0"}),
    (rf"{_LEAD}(?:(?:enter|exit|leave|toggle)\s+)?full\s*screen{_TAIL}", {"keys": "ctrl+cmd+f"}),
    (rf"{_LEAD}minimi[sz]e(?:\s+(?:this|the|that)\s+window)?{_TAIL}", {"keys": "cmd+m"}),
    (rf"{_LEAD}hide\s+(?:this|it|that|the\s+app|this\s+app){_TAIL}", {"keys": "cmd+h"}),
    (rf"{_LEAD}(?:find|search)\s+(?:on|in)\s+(?:this|the)\s+page{_TAIL}", {"keys": "cmd+f"}),
    (rf"{_LEAD}(?:take|grab|capture)\s+(?:a\s+)?screenshot(?:\s+of\s+(?:this|the\s+screen|my\s+screen)?)?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}screenshot(?:\s+(?:this|the\s+screen|my\s+screen))?{_TAIL}", {"keys": "cmd+shift+3"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?top{_TAIL}", {"keys": "cmd+up"}),
    (rf"{_LEAD}scroll\s+to\s+(?:the\s+)?bottom{_TAIL}", {"keys": "cmd+down"}),
    (rf"{_LEAD}(?:open\s+)?spotlight{_TAIL}", {"keys": "cmd+space"}),
    (rf"{_LEAD}(?:quit|close)\s+(?:this|the\s+current)\s+app{_TAIL}", {"keys": "cmd+q"}),
    (rf"{_LEAD}(?:switch|next)\s+app{_TAIL}", {"keys": "cmd+tab"}),
]
# Click whatever on screen is called that: "click play", "press the send button", "tap next".
_CLICK_LABEL = (
    rf"{_LEAD}(?:click|tap|press|hit|select|choose)\s+(?:on\s+)?(?:the\s+)?"
    r"(?!(?:enter|return|tab|escape|esc|delete|backspace|space|up|down|left|right|home|end|all)\b)"
    r"(?P<label>.+?)(?:\s+(?:button|link|tab|icon|menu|item|option|checkbox))?"
    rf"{_TAIL}"
)
_DOC_APPS = {"notes", "textedit", "pages", "stickies", "numbers", "keynote", "freeform"}
_DICTATE_SYSTEM = (
    "You type on the user's behalf into the document they have open. Reply with ONLY the text to "
    "type: no preamble, no quotes, no markdown headings or bold. For points or a list, one item "
    "per line starting with '- '. Be concise (under 120 words unless asked otherwise) and write in "
    "the user's language."
)


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] in "\"“'" and text[-1] in "\"”'":
        return text[1:-1]
    return text


async def _ensure_text_focus() -> str | None:
    """Put the keyboard focus somewhere typeable. Returns an error message if that's impossible."""
    if await asyncio.to_thread(ax.focused_editable):
        return None
    app, _ = ax.frontmost_app()
    area = await asyncio.to_thread(ax.main_text_area)
    if area is None and app.lower() in _DOC_APPS:
        inp.hotkey("cmd+n")  # nothing open to type into: a fresh note/document
        await asyncio.sleep(0.7)
        if await asyncio.to_thread(ax.focused_editable):
            return None
        area = await asyncio.to_thread(ax.main_text_area)
    if area is None:
        return f"Error: nothing to type into is focused in {app or 'the front app'} — click into a text field first."
    if area.focused:  # the editor already has the caret (Notes reports focus on the area, not system-wide)
        return None
    inp.click(*area.center)
    await asyncio.sleep(0.2)
    inp.hotkey("cmd+down")  # a click lands mid-text; dictation appends at the end
    return None


async def _type(text: str) -> None:
    """Type text; line breaks are pressed as Return so every app makes a real new line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line:
            await asyncio.to_thread(inp.type_text, line)
        if i < len(lines) - 1:
            inp.hotkey("return")
            await asyncio.sleep(0.03)


@tool(
    "type_text",
    "Type text where the cursor is (dictation). Use for literal text the user wants typed.",
    _p({"text": {"type": "string"}}, ["text"]),
    category="computer",
    fast_path=[(_LITERAL_TYPE, {"text": "<text>"}), (_LITERAL_WRITE, {"text": "<text>"})],
    early=False,  # the words are the payload: wait for the whole sentence
    quiet=True,
)
async def type_text(a: dict, c: ToolContext) -> str:
    text = _unquote(str(a["text"]))
    err = await _ensure_text_focus()
    if err:
        return err
    await _type(text)
    return f"Typed “{text[:80]}”."


@tool(
    "dictate",
    "Write short text from a brief (bullet points, a paragraph, a caption…) and type it where "
    "the cursor is. Use when the user wants NEO to come up with the words.",
    _p({"brief": {"type": "string"}}, ["brief"]),
    category="computer",
    slow=True,
    fast_path=[(_COMPOSE_TYPE, {"brief": "<brief>"})],
    early=False,
    quiet=True,
)
async def dictate(a: dict, c: ToolContext) -> str:
    from neo.providers import brain
    from neo.providers.base import Message

    brief = str(a["brief"]).strip()
    turn = await brain("fast").complete([Message.user(brief)], system=_DICTATE_SYSTEM, max_tokens=400)
    text = (turn.text or "").strip().strip("\"“”")
    if not text:
        return "Error: I couldn't come up with anything to type."
    err = await _ensure_text_focus()
    if err:
        return err
    area = await asyncio.to_thread(ax.main_text_area)
    if area is not None and area.value.strip():
        text = "\n" + text  # the heading stays a heading; the points start on their own line
    await _type(text)
    first = text.strip().splitlines()[0][:80]
    return f"Typed {len(text.splitlines())} line(s), starting “{first}”."


@tool(
    "hotkey",
    "Press a key or chord, e.g. 'return', 'cmd+s', 'cmd+shift+t', 'escape', 'tab'.",
    _p({"keys": {"type": "string"}}, ["keys"]),
    category="computer",
    fast_path=_HOTKEYS,
    quiet=True,
)
async def hotkey(a: dict, c: ToolContext) -> str:
    return inp.hotkey(a["keys"])


_CLICK_ROLES = ("AXButton", "AXLink", "AXMenuItem", "AXTab", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXCell", "AXRow", "AXStaticText", "AXImage")


@tool(
    "click_text",
    "Click the on-screen element whose label contains the text (button, link, tab, menu item…) "
    "in the frontmost app.",
    _p({"label": {"type": "string"}}, ["label"]),
    category="computer",
    quiet=True,
    fast_path=[(_CLICK_LABEL, {"label": "<label>"})],
)
async def click_text(a: dict, c: ToolContext) -> str:
    label = str(a["label"]).strip().lower()
    if not label:
        return "Error: nothing to click."
    await asyncio.to_thread(ax.snapshot)  # fresh tree of the frontmost app
    hits = [
        e
        for e in ax._last_tree.values()
        if e.enabled and e.w > 0 and (label in e.title.lower() or label in e.value.lower())
    ]
    if not hits:
        return f"Error: nothing on screen called '{a['label']}'."
    # exact title, then the most button-like role, then the smallest thing that matched
    hits.sort(
        key=lambda e: (
            e.title.lower() != label,
            _CLICK_ROLES.index(e.role) if e.role in _CLICK_ROLES else len(_CLICK_ROLES),
            e.w * e.h,
        )
    )
    e = hits[0]
    if "AXPress" in e.actions:
        out = ax.press(e.id)
        if not out.startswith("Error"):
            return f"Clicked '{e.title or e.value}' ({e.role.removeprefix('AX')})"
    inp.click(*e.center)
    return f"Clicked '{e.title or e.value}' ({e.role.removeprefix('AX')}) at {tuple(map(int, e.center))}"


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
        )
    ],
    quiet=True,
)
async def open_app(a: dict, c: ToolContext) -> str:
    return await apps.open_app(a["target"])


@tool(
    "activate_app",
    "Bring a running app to the front.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
    quiet=True,
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
    "Run AppleScript and return its result. Use for anything scriptable that has no dedicated tool.",
    _p({"script": {"type": "string"}}, ["script"]),
    category="apps",
    slow=True,
)
async def applescript(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    script = str(a.get("script") or "")
    if apps.script_is_risky(script):
        gate = confirm.gated("osa:" + script[:60], "applescript", a, f"run AppleScript: {script[:100]!r}")
        if isinstance(gate, str):
            return gate
    return await apps.osascript(script, timeout=60)


@tool(
    "safari_open",
    "Open a URL in Safari.",
    _p({"url": {"type": "string"}, "new_tab": {"type": "boolean"}}, ["url"]),
    category="web",
    quiet=True,
)
async def safari_open(a: dict, c: ToolContext) -> str:
    return await apps.safari_open(a["url"], bool(a.get("new_tab", True)))


@tool(
    "safari_read",
    "Read the visible text of Safari's current tab (and its URL).",
    _OBJ,
    parallel_safe=True,
    category="web",
)
async def safari_read(a: dict, c: ToolContext) -> str:
    url = await apps.safari_current_url()
    return f"URL: {url}\n\n" + await apps.safari_page_text()


@tool(
    "mail_send",
    "Compose an email in Mail. send=false opens a draft for the user; send=true sends it.",
    _p(
        {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "send": {"type": "boolean"},
        },
        ["to", "subject", "body"],
    ),
    risk="destructive",
    category="apps",
    slow=True,
)
async def mail_send(a: dict, c: ToolContext) -> str:
    from neo.agent import confirm

    if a.get("send"):
        gate = confirm.gated(
            f"mail:{a['to']}:{a['subject'][:30]}",
            "mail_send",
            a,
            f"send an email to {a['to']} — “{a['subject']}”",
        )
        if isinstance(gate, str):
            return gate
    return await apps.mail_send(a["to"], a["subject"], a["body"], bool(a.get("send")))


@tool(
    "mail_unread",
    "List unread inbox messages (sender | subject | date).",
    _p({"limit": {"type": "integer"}}),
    parallel_safe=True,
    category="apps",
    slow=True,
    timeout=50,
    fast_path=[
        (
            r"^\s*(?:(?:check|read|show(?:\s+me)?|list|get|open)\s+(?:my\s+)?(?:new\s+|unread\s+|latest\s+)?"
            r"(?:e-?mails?|mail|inbox)|(?:do\s+i\s+have\s+)?(?:any\s+)?(?:new|unread)\s+(?:e-?mails?|mail|messages)|"
            r"what(?:'s| is)\s+(?:new\s+)?in\s+my\s+inbox|any\s+(?:new\s+)?(?:e-?mails?|mail))\s*[?.!]*\s*$",
            {},
        )
    ],
)
async def mail_unread(a: dict, c: ToolContext) -> str:
    return await apps.mail_unread(int(a.get("limit") or 10))


@tool(
    "calendar_today",
    "List today's calendar events.",
    _OBJ,
    parallel_safe=True,
    category="apps",
    fast_path=[
        (
            r"^\s*(?:what(?:'s| is)\s+on\s+(?:my\s+)?(?:calendar|schedule|agenda)(?:\s+(?:for\s+)?today)?|"
            r"(?:do\s+i\s+have\s+)?(?:any\s+)?(?:meetings?|events?|appointments?)\s+today|"
            r"(?:show|check|read|open)\s+(?:me\s+)?(?:my\s+)?(?:calendar|schedule|agenda)(?:\s+(?:for\s+)?today)?|"
            r"what(?:'s| is)\s+my\s+(?:schedule|agenda|day)(?:\s+like)?(?:\s+today)?|what\s+do\s+i\s+have\s+(?:on\s+)?today)\s*[?.!]*\s*$",
            {},
        )
    ],
)
async def calendar_today(a: dict, c: ToolContext) -> str:
    return await apps.calendar_today()


@tool(
    "calendar_add",
    "Add a calendar event. Times are ISO local 'YYYY-MM-DDTHH:MM'.",
    _p(
        {
            "title": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "calendar": {"type": "string"},
        },
        ["title", "start", "end"],
    ),
    risk="reversible",
    category="apps",
    quiet=True,
)
async def calendar_add(a: dict, c: ToolContext) -> str:
    return await apps.calendar_add(a["title"], a["start"], a["end"], a.get("calendar", ""))


@tool(
    "notes_create",
    "Create a note in Apple Notes.",
    _p(
        {"title": {"type": "string"}, "body": {"type": "string"}, "folder": {"type": "string"}},
        ["title", "body"],
    ),
    risk="reversible",
    category="apps",
    quiet=True,
    chain=False,  # only as the whole utterance: a later clause means 'in that app'
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:make|create|add|start)\s+(?:a\s+)?(?:new\s+)?note\s+(?:called|titled|named)\s+"
            r"[\"“']?(?P<title>.+?)[\"”']?(?:\s+(?:saying|that\s+says|with(?:\s+the\s+text)?|containing)\s+"
            r"[\"“']?(?P<body>.+?)[\"”']?)?\s*[.!]*\s*$",
            {"title": "<title>", "body": "<body>"},
        ),
        (
            r"^\s*(?:please\s+)?(?:make|create|add|start)\s+(?:a\s+)?(?:new\s+)?note\s+(?:saying|that\s+says)\s+"
            r"[\"“']?(?P<body>.+?)[\"”']?\s*[.!]*\s*$",
            {"title": "", "body": "<body>"},
        )
    ],
)
async def notes_create(a: dict, c: ToolContext) -> str:
    title, body = (a.get("title") or "").strip(), (a.get("body") or "").strip()
    if not title:  # "make a note saying buy milk": the first words become the title
        title = " ".join(body.split()[:6]) or "New note"
    return await apps.notes_create(title, body, a.get("folder", ""))


@tool(
    "reminder_add",
    "Add a reminder, optionally due at ISO local 'YYYY-MM-DDTHH:MM'.",
    _p({"title": {"type": "string"}, "due": {"type": "string"}}, ["title"]),
    risk="reversible",
    category="apps",
    quiet=True,
)
async def reminder_add(a: dict, c: ToolContext) -> str:
    return await apps.reminder_add(a["title"], a.get("due", ""))


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
    "Run a zsh command (cwd defaults to home). Destructive commands are confirmed with the user first.",
    _p(
        {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number"},
            "confirm": {"type": "boolean"},
        },
        ["command"],
    ),
    risk="reversible",
    category="files",
    slow=True,
)
async def shell_tool(a: dict, c: ToolContext) -> str:
    return await shell.run(a["command"], cwd=a.get("cwd"), timeout=float(a.get("timeout", 60)), confirm_arg=a)


@tool(
    "read_file",
    "Read a text file (numbered lines), a PDF, or list a directory.",
    _p({"path": {"type": "string"}, "start": {"type": "integer"}, "lines": {"type": "integer"}}, ["path"]),
    parallel_safe=True,
    category="files",
)
async def read_file(a: dict, c: ToolContext) -> str:
    return shell.read_file(a["path"], start=int(a.get("start") or 1), lines=int(a.get("lines") or 400))


@tool(
    "write_file",
    "Create or overwrite a file with content (overwrites are confirmed).",
    _p(
        {"path": {"type": "string"}, "content": {"type": "string"}, "confirm": {"type": "boolean"}},
        ["path", "content"],
    ),
    risk="destructive",
    category="files",
)
async def write_file(a: dict, c: ToolContext) -> str:
    return shell.write_file(a["path"], a["content"], a)


@tool(
    "edit_file",
    "Replace one exact occurrence of old_string with new_string in a file.",
    _p(
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    risk="reversible",
    category="files",
)
async def edit_file(a: dict, c: ToolContext) -> str:
    return shell.edit_file(a["path"], a["old_string"], a["new_string"])


@tool(
    "trash",
    "Move a file or folder to the Trash (confirmed).",
    _p({"path": {"type": "string"}, "confirm": {"type": "boolean"}}, ["path"]),
    risk="destructive",
    category="files",
)
async def trash(a: dict, c: ToolContext) -> str:
    return shell.trash(a["path"], a)
