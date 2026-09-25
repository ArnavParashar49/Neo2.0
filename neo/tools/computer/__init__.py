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
)
async def ax_press(a: dict, c: ToolContext) -> str:
    return await asyncio.to_thread(ax.press, a["id"])


@tool(
    "ax_set_value",
    "Set the text of a text field / area by element id (faster and more reliable than typing).",
    _p({"id": {"type": "string"}, "value": {"type": "string"}}, ["id", "value"]),
    category="computer",
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
)
async def drag(a: dict, c: ToolContext) -> str:
    return inp.drag(a["x1"], a["y1"], a["x2"], a["y2"])


@tool(
    "scroll",
    "Scroll at (x,y). dy<0 scrolls down, dy>0 up (lines).",
    _p(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "dy": {"type": "integer"},
            "dx": {"type": "integer"},
        },
        ["x", "y"],
    ),
    category="computer",
)
async def scroll(a: dict, c: ToolContext) -> str:
    return inp.scroll(a["x"], a["y"], int(a.get("dy", -5)), int(a.get("dx", 0)))


@tool(
    "type_text",
    "Type text into the focused element.",
    _p({"text": {"type": "string"}}, ["text"]),
    category="computer",
)
async def type_text(a: dict, c: ToolContext) -> str:
    return await asyncio.to_thread(inp.type_text, a["text"])


@tool(
    "hotkey",
    "Press a key or chord, e.g. 'return', 'cmd+s', 'cmd+shift+t', 'escape', 'tab'.",
    _p({"keys": {"type": "string"}}, ["keys"]),
    category="computer",
)
async def hotkey(a: dict, c: ToolContext) -> str:
    return inp.hotkey(a["keys"])


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
            r"^\s*(?:please\s+)?(?:open|launch)\s+(?!(?:a|an|the|my|up|it|this|that)\b)"
            r"(?P<target>[\w.+-]+(?:\s+[\w.+-]+){0,2})[\s.!]*$",
            {"target": "<target>"},
        )
    ],
)
async def open_app(a: dict, c: ToolContext) -> str:
    return await apps.open_app(a["target"])


@tool(
    "activate_app",
    "Bring a running app to the front.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
)
async def activate_app(a: dict, c: ToolContext) -> str:
    return apps.activate(a["name"])


@tool("quit_app", "Quit an app gracefully.", _p({"name": {"type": "string"}}, ["name"]), category="apps")
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
)
async def mail_unread(a: dict, c: ToolContext) -> str:
    return await apps.mail_unread(int(a.get("limit", 10)))


@tool("calendar_today", "List today's calendar events.", _OBJ, parallel_safe=True, category="apps")
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
)
async def notes_create(a: dict, c: ToolContext) -> str:
    return await apps.notes_create(a["title"], a["body"], a.get("folder", ""))


@tool(
    "reminder_add",
    "Add a reminder, optionally due at ISO local 'YYYY-MM-DDTHH:MM'.",
    _p({"title": {"type": "string"}, "due": {"type": "string"}}, ["title"]),
    risk="reversible",
    category="apps",
)
async def reminder_add(a: dict, c: ToolContext) -> str:
    return await apps.reminder_add(a["title"], a.get("due", ""))


@tool(
    "finder_reveal",
    "Reveal a file or folder in Finder.",
    _p({"path": {"type": "string"}}, ["path"]),
    category="apps",
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
