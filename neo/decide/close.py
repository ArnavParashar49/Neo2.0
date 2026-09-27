"""'Close YouTube', 'quit Spotify', 'close the netflix tab': close what the user named, no questions.

Closing is undoable (reopen the app; a browser brings a tab back with ⌘⇧T) and an app asks about
its own unsaved work, so nothing here needs the user's OK. What makes that safe is that no model
writes a script: NEO builds it from the browser's own scripting dictionary (raw «class»/«property»
codes, never a browser's name) and passes the user's words as argv, never as script source.

Order: a running app the words name (quit it), then open tabs in running browsers whose address
or title has the words (close them). Nothing is launched to look: only apps already running are
asked.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neo.decide._objc import pooled

if TYPE_CHECKING:
    from neo.decide.inventory import App

# Words that say what kind of thing it is, not which: "close the youtube tab" closes "youtube".
_FILLER = re.compile(r"\b(?:the|a|an|my|this|that|all|every|of|tabs?|windows?|apps?|application|page|site|website)\b", re.I)
_SCRIPT_S = 8.0


@dataclass(frozen=True)
class _TabCodes:
    window: str
    tab: str
    url: str
    title: str
    close: str


def words(target: str) -> str:
    """The name in 'close the YouTube tab': 'YouTube'."""
    return " ".join(_FILLER.sub(" ", target or "").split()).strip(" .,!?\"'")


def _tab_codes(app: App) -> _TabCodes | None:
    """The browser's own codes for its windows, tabs, a tab's address and title, and 'close' —
    from its scripting dictionary; None if it doesn't script its tabs."""
    d = getattr(app, "script", None)
    if d is None:
        return None
    window = (d.classes.get("window") or ("",))[0]
    tab = (d.classes.get("tab") or ("",))[0]
    props = {k.lower(): v for k, v in d.properties.items()}
    url = props.get("url", "")
    title = props.get("title") or props.get("name", "")
    close = getattr(d.commands.get("close"), "code", "") or ""
    from neo.decide.media import _ok4, _ok8  # a code goes into script source: exactly a code

    if not (all(_ok4(c) for c in (window, tab, url, title)) and _ok8(close)):
        return None
    return _TabCodes(window, tab, url, title, close)


def _close_tabs_script(c: _TabCodes) -> str:
    # Newest last: walk each window's tabs backwards so closing one doesn't skip the next.
    return (
        "on run argv\n\tset bid to item 1 of argv\n\tset q to item 2 of argv\n\tset n to 0\n"
        "\ttell application id bid\n"
        f"\t\trepeat with w in (every «class {c.window}»)\n"
        f"\t\t\tset ts to every «class {c.tab}» of w\n"
        "\t\t\trepeat with i from (count of ts) to 1 by -1\n"
        "\t\t\t\tset t to item i of ts\n\t\t\t\tset u to \"\"\n\t\t\t\tset nm to \"\"\n"
        f"\t\t\t\ttry\n\t\t\t\t\tset u to («property {c.url}» of t) as text\n\t\t\t\tend try\n"
        f"\t\t\t\ttry\n\t\t\t\t\tset nm to («property {c.title}» of t) as text\n\t\t\t\tend try\n"
        "\t\t\t\tif u contains q or nm contains q then\n"
        f"\t\t\t\t\t«event {c.close}» t\n\t\t\t\t\tset n to n + 1\n"
        "\t\t\t\tend if\n\t\t\tend repeat\n\t\tend repeat\n\tend tell\n\treturn n\nend run"
    )


@pooled
def _running_apps() -> list[App]:
    from neo.decide import inventory

    return [a for a in inventory.inventory() if a.running]


def _is_browser(app: App) -> bool:
    return "https" in app.handles or app.media_evidence[:1] == ("web browser",)


async def close_target(target: str) -> str:
    """Close what `target` names. Returns what was closed, or an 'Error: …' saying what isn't open."""
    from neo.decide import inventory, media
    from neo.tools.computer import apps as mac

    what = words(target)
    if not what:
        return "Error: close what?"
    running = await asyncio.to_thread(_running_apps)
    app = await asyncio.to_thread(inventory.named_in, what, running)
    if app is not None:
        out = await mac.quit_app(app.name)
        return out if out.startswith("Error") else f"Quit {app.name}"
    closed: list[str] = []
    blocked: list[str] = []
    for b in [a for a in running if _is_browser(a)]:
        codes = _tab_codes(b)
        if codes is None:
            continue
        gate = await media._control_gate(b)  # macOS's one-time "allow NEO to control <browser>?"
        if gate is not None:
            blocked.append(b.name)
            continue
        code, out, err = await media._osascript(_close_tabs_script(codes), [b.bundle_id, what], _SCRIPT_S)
        n = int(out) if code == 0 and out.strip().isdigit() else 0
        if code != 0:
            print(f"[close] {b.name} didn't answer ({err[-80:]})")
        if n:
            closed.append(f"{n} {what} tab{'s' if n != 1 else ''} in {b.name}")
    if closed:
        return "Closed " + " and ".join(closed)
    if blocked:
        return f"Error: {' and '.join(blocked)} need{'s' if len(blocked) == 1 else ''} your OK before NEO can close tabs there"
    installed = await asyncio.to_thread(inventory.named_in, what, None)
    if installed is not None:
        return f"Error: {installed.name} isn't open"
    return f"Error: nothing called {what!r} is open"
