"""The keyboard tools: type_text, dictate and hotkey — typing that never sends or runs by
accident, keys asked about when they send or run something. Every gate they pass is in gates.py."""

from __future__ import annotations

import asyncio
import time

from neo2.core.agent.registry import ToolContext, ToolOutput
from neo2.core.config import decide_mode
from neo2.core.tools.computer import apps, ax, gates, menus, phrases, pointer
from neo2.core.tools.computer import input as inp
from neo2.core.tools.computer._catalog import _p, tool

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
        from neo2.core.decide.inventory import inventory

        names = {a.name.lower() for a in inventory() if "https" in a.handles or a.media_evidence[:1] == ("web browser",)}
    except Exception:  # noqa: BLE001
        pass
    if d := apps.default_browser():
        names.add(d.lower())
    return names


_DICTATE_SYSTEM = (
    "You type on the user's behalf into the document they have open. Reply with ONLY the text to "
    "type: no preamble, no quotes, no markdown headings or bold. For points or a list, one item "
    "per line starting with '- '. Be concise (under 120 words unless asked otherwise) and write in "
    "the user's language."
)
_INPUT_TOOLS = ("type_text", "dictate", "hotkey", "click_text", "ax_set_value")
_UNDO_WINDOW_S = 120.0


def _layout() -> None:
    """The keyboard layout's letter keys, read afresh when it changed (cheap when it didn't): here,
    on the event loop's thread, the one where macOS allows reading it; a chord sent from a worker
    thread then presses the key that types its letter on this layout (⌘Z on AZERTY)."""
    try:
        inp.refresh_layout()
    except Exception:  # noqa: BLE001 - can't read it: the US positions stay
        pass


async def _ensure_text_focus() -> tuple[str | None, bool]:
    """Put the keyboard focus somewhere typeable in the front window of NEO's target app.

    Returns (error, created) — `created` when a fresh note/document had to be made."""
    if err := await gates._guard_front():
        return err, False
    if await asyncio.to_thread(ax.focused_is_secure):
        return "Error: the cursor is in a password field — I won't type there.", False
    if await asyncio.to_thread(ax.focused_editable):
        return None, False
    app, pid = ax.frontmost_app()
    area = await asyncio.to_thread(ax.main_text_area)
    created = False
    if area is None and await asyncio.to_thread(gates._doc_like, app, pid):
        _layout()
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
        spot = await asyncio.to_thread(pointer.spot, frame) if frame is not None else None
        if spot is None:
            return f"Error: the text area in {app or 'the front app'} isn't on screen — click into it first.", created
        try:
            await pointer.probe([spot])  # never NEO's own window
        except pointer.Refused as r:
            return (gates._refuse_protected() if r.protected else str(r)), created
        await asyncio.to_thread(inp.click, *spot)
        await asyncio.sleep(0.2)
        if not await asyncio.to_thread(ax.focused_editable):
            return f"Error: I couldn't put the cursor in {app or 'the front app'}'s text — click into it first.", created
    _layout()
    await asyncio.to_thread(inp.hotkey, "cmd+down")  # dictation appends at the end
    return None, created


_SHELL_JOINED = " (as one line: the cursor is on a command line, where a line break would run each line)"


async def _line_break(app: str, pid: int, role: str) -> tuple[str, str]:
    """How a line break is typed where the cursor is — decided by what's there, never by the
    app's name — and what to tell about it: "" joins the lines (a single-line field; a command
    line, where any Return runs the line), "return" in a document editor, else shift+return — a
    new line in a chat box or a web editor, never a "send"."""
    if role in ax._SINGLE_LINE:
        return "", ""
    if await gates._command_line(pid):
        return "", _SHELL_JOINED
    if await asyncio.to_thread(gates._doc_like, app, pid):
        return "return", ""
    return "shift+return", ""


async def _type(text: str) -> str:
    """Type text so every line break is a real new line — and never a "send" or a command run.
    Returns what the result should add ("" or why the lines were joined)."""
    app, pid = ax.frontmost_app()
    role, _ = await asyncio.to_thread(ax.focused_role)
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")  # a typed \r is a Return
    newline, note = await _line_break(app, pid, role) if "\n" in text else ("return", "")
    if not newline:
        text = " ".join(part.strip() for part in text.split("\n") if part.strip())
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line:
            await asyncio.to_thread(inp.type_text, line)
        if i < len(lines) - 1:
            await asyncio.to_thread(inp.hotkey, newline)
            await asyncio.sleep(0.03)
    return note


@tool(
    "type_text",
    "Type text where the cursor is (dictation). Use for literal text the user wants typed. field: a text "
    "field's label to type into instead (found by its label in the front or named app's window).",
    _p(
        {
            "text": {"type": "string"},
            "field": {"type": "string", "description": "the label of the field to type into (optional)"},
            "app": {"type": "string", "description": "the app whose field it is (optional; the front app)"},
            "replace": {"type": "boolean", "description": "with field: replace what the field already holds"},
        },
        ["text"],
    ),
    category="computer",
    # (a named field first: "type hello in the search field" keeps the place it names)
    fast_path=[*menus.FIELD_FAST_PATHS, (phrases._LITERAL_TYPE, {"text": "<text>"}), (phrases._LITERAL_WRITE, {"text": "<text>"})],
    quiet=True,
    payload=True,
)
async def type_text(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := gates._needs_input():
        return err
    if a.get("field"):  # a field by its label: found, gated, set through Accessibility and read back
        return await menus.fill_field(
            str(a["text"]), str(a["field"]), c, app=str(a.get("app") or ""), replace=bool(a.get("replace"))
        )
    text = str(a["text"])
    err, created = await _ensure_text_focus()
    if err:
        return gates._read_from_screen(c, err)
    # A model's typing is checked: the text where the cursor is should have changed — looked at
    # twice, since a web or Electron field reports its new text a moment after it shows it.
    watch = c.origin == "agent" and not created
    before = await asyncio.to_thread(ax.focused_element_text) if watch else ""
    note = await _type(text)
    if watch and before:
        for wait in (pointer.LOOK_S, pointer.LOOK_AGAIN_S):
            await asyncio.sleep(wait)
            if await asyncio.to_thread(ax.focused_element_text) != before:
                break
        else:
            return ToolOutput(
                f"Typed “{text[:80]}”, but I couldn't confirm it arrived: the text where the cursor is looks "
                f"the same after {pointer.LOOK_S + pointer.LOOK_AGAIN_S:g}s. Look (ax_tree or a screenshot) "
                "before typing it again — typing twice would put it in twice.",
                ok=False,
            )
    return f"Typed “{text[:80]}”{note}" + (" into a new note." if created else ".") + gates._input_doubt()


@tool(
    "dictate",
    "Write short text from a brief (bullet points, a paragraph, a caption…) and type it where "
    "the cursor is. Use when the user wants NEO to come up with the words.",
    _p({"brief": {"type": "string"}}, ["brief"]),
    category="computer",
    slow=True,
    fast_path=[(phrases._COMPOSE_TYPE, {"brief": "<brief>"})],
    quiet=True,
    payload=True,
)
async def dictate(a: dict, c: ToolContext) -> str:
    from neo2.core.providers import brain
    from neo2.core.providers.base import Message

    if ax_err := gates._needs_input():
        return ax_err
    brief = str(a["brief"]).strip()
    err, created = await _ensure_text_focus()  # check before spending a model call
    if err:
        return gates._read_from_screen(c, err)
    turn = await brain("fast").complete([Message.user(brief)], system=_DICTATE_SYSTEM, max_tokens=400)
    text = (turn.text or "").strip().strip("\"“”")
    if not text:
        return "Error: I couldn't come up with anything to type."
    area = await asyncio.to_thread(ax.main_text_area)
    if area is not None and area.value.strip() and not area.value.endswith("\n"):
        text = "\n" + text  # the heading stays a heading; the points start on their own line
    note = await _type(text)
    first = text.strip().splitlines()[0][:80]
    return (
        f"Typed {len(text.strip().splitlines())} line(s), starting “{first}”{note}"
        + (" (new note)." if created else ".")
        + gates._input_doubt()
    )


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
    fast_path=phrases._HOTKEYS,
    quiet=True,
)
async def hotkey(a: dict, c: ToolContext) -> str:
    if err := gates._needs_input():
        return err
    keys, want = str(a["keys"]), str(a.get("app") or "")
    if a.get("after") == "input":
        from neo2.core.agent.registry import registry as _reg

        last = _reg().last
        if not last or last[0] not in _INPUT_TOOLS or time.time() - last[1] > _UNDO_WINDOW_S:
            return "Error: undo what? I haven't typed anything just now."
    # only "new note" while write_note is on: any other call naming "note" (a model's, a playbook
    # saved before a rollback) presses its own keys, as legacy does
    if want == "note" and gates._chord(keys) == {"cmd", "n"} and decide_mode("write_note") == "on":
        return await _new_note(c)
    specific = want not in ("", "browser", "browser|finder", "document", "tabs")
    if err := await gates._guard_front(want if specific else ""):
        return err
    front_name, front_pid = ax.frontmost_app()
    front = front_name.lower()
    browsers = await asyncio.to_thread(_browser_names) if want in ("browser", "browser|finder", "tabs") else set()
    if want == "browser" and front not in browsers:
        return f"Error: '{keys}' only makes sense in a browser, and {front or 'nothing'} is in front."
    if want == "browser|finder" and front not in browsers | {"finder"}:
        return f"Error: going back only makes sense in a browser or Finder, and {front or 'nothing'} is in front."
    if want == "document" and not await asyncio.to_thread(gates._doc_like, front, front_pid):
        return "Error: which app should the new document be in?"
    # "new tab" from an app without tabs: the user's default browser
    if want == "tabs" and front not in _TAB_APPS | browsers:
        if not (browser := apps.default_browser()):
            return "Error: there's no default browser to open a tab in."
        if err := await gates._guard_front(browser):
            return err
    try:
        inp.parse_chord(keys)
    except ValueError as e:
        return f"Error: {e}"  # an unknown modifier or key: nothing pressed, never the bare key
    name, pid = ax.frontmost_app()
    if gate := await gates._ask_before_keys(a, keys, name, c, pid):
        return gate
    before = await gates._watch(c, pid, []) if _may_change_scene(keys) else None
    _layout()
    out = await asyncio.to_thread(inp.hotkey, keys)
    if before is not None and not out.startswith("Error"):
        out += await gates._changes(before, pid, c)  # ⌘N may open a window, or a sheet that waits for an answer
    return out + ("" if out.startswith("Error") else gates._input_doubt())


async def _new_note(c: ToolContext) -> str:
    """"new note" (write_note on): the New command of the notes app this user uses, found by its
    ⌘N shortcut in the app's own menus; the chord itself when the menu doesn't show one."""
    from neo2.core.decide import axkit
    from neo2.core.decide.kinds import note

    app = await note.top_app(c.user_text)
    if app is None:
        return "Error: there's no notes app on this Mac."
    if err := await gates._guard_front(app.name):
        return err
    pid = await axkit.window_pid(app, 5.0)
    item = await asyncio.to_thread(axkit.menu_item, pid, "n", 0) if pid else None
    if item and await asyncio.to_thread(axkit.perform, item[0], "AXPress"):
        return f"New note in {app.name}."
    _layout()
    return await asyncio.to_thread(inp.hotkey, "cmd+n")


# Keys that only move the caret or edit the text where it is: no window, sheet or focus to wait for.
_CARET_KEYS = frozenset({"left", "right", "up", "down", "home", "end", "pageup", "pagedown", "delete", "backspace", "forwarddelete"})


def _may_change_scene(keys: str) -> bool:
    """Could this key open or close a window or a sheet, or move the focus? (A ⌘ or ⌃ chord,
    Return, Escape, Tab, a letter…: looked at afterwards. A caret move: not.)"""
    try:
        flags, key = inp.parse_chord(keys)
    except ValueError:
        return True
    return bool(flags & (inp._MODS["cmd"] | inp._MODS["ctrl"])) or key not in _CARET_KEYS
