"""An app's UI through Accessibility: the reads and writes every kind's window routes share.

The primitives (attribute reads, sets, actions, the search-field finder, the web-tree switch)
are :mod:`neo.decide.media`'s own, looked up on each call: until they move here, media and its
tests keep working unchanged, and a patch of ``media._attr`` applies here too. The helpers below
find things by role, keyboard shortcut and the OS's own localized words, never by an app's name
or a label written in English only.

Keystrokes go only to the process that owns the element, and nothing here touches the pasteboard.
"""

from __future__ import annotations

import plistlib
import re
import unicodedata
from functools import lru_cache
from typing import Any

from neo.decide._objc import pooled

__all__ = [
    "attr",
    "set_attr",
    "perform",
    "actions",
    "app_el",
    "trusted",
    "post_return",
    "lexicon",
    "find_search",
    "enable_web_tree",
    "restore_web_tree",
    "runtime",
    "window",
    "menu_item",
    "focused_editable",
    "composer",
    "rows_matching",
    "header_title",
    "transcript_count",
    "set_text",
    "clear_text",
    "commit_button",
    "web_url",
    "web_text",
    "dock_badge",
]


def _media():
    from neo.decide import media

    return media


# ---- media's primitives, resolved at call time -----------------------------------------------------


def attr(el: Any, name: str, default: Any = None) -> Any:
    return _media()._attr(el, name, default)


def set_attr(el: Any, name: str, value: Any) -> bool:
    return _media()._set(el, name, value)


def perform(el: Any, action: str) -> bool:
    return _media()._perform(el, action)


def actions(el: Any) -> list[str]:
    return _media()._actions(el)


def app_el(pid: int) -> Any:
    return _media()._ax_app(pid)


def trusted() -> bool:
    return _media()._trusted()


def post_return(pid: int) -> None:
    _media()._post_return(pid)


def find_search(app: Any) -> list:
    return _media().find_search(app)


def enable_web_tree(app: Any, rt: str) -> list[tuple[str, bool]]:
    return _media()._enable_web_tree(app, rt)


def restore_web_tree(app: Any, changed: list[tuple[str, bool]]) -> None:
    _media()._restore(app, changed)


def runtime(app_path: str) -> str:
    return _media()._runtime(app_path)


def window(app: Any) -> Any:
    return _media()._window(app)


def _bfs(root: Any, max_nodes: int = 4000, max_depth: int = 60):
    return _media()._bfs(root, max_nodes, max_depth)


def _s(v: Any) -> str:
    return _media()._s(v)


# ---- words -------------------------------------------------------------------------------------------

# OS string tables that carry "Send" (Intents) and "New" (the standard menu), besides media's own.
_MORE_SOURCES = (
    "/System/Library/Frameworks/Intents.framework/Versions/A/Resources/Localizable.loctable",
    "/System/Library/Frameworks/SwiftUI.framework/Versions/A/Resources/MainMenu.loctable",
)
_MORE_WORDS = {"send": ("Send",), "new": ("New",)}


@lru_cache(maxsize=1)
@pooled
def lexicon() -> dict[str, frozenset[str]]:
    """media's words ('play', 'search', 'find', 'pause') plus 'send' and 'new', in English and the
    user's preferred languages, from the OS's own string tables."""
    words = {k: set(v) for k, v in _MORE_WORDS.items()}
    try:
        from Foundation import NSLocale

        prefs = [str(x) for x in (NSLocale.preferredLanguages() or [])]
    except Exception:  # noqa: BLE001
        prefs = []
    langs = {"en", "hi", *prefs, *(p.split("-")[0] for p in prefs)}
    for f in _MORE_SOURCES:
        try:
            with open(f, "rb") as fh:
                table = plistlib.load(fh)
        except (OSError, plistlib.InvalidFileException, ValueError):
            continue
        en = table.get("en") if isinstance(table, dict) else None
        for key, val in (en or {}).items():
            w = val.strip().rstrip("…").strip().lower() if isinstance(val, str) else ""
            if w not in words:
                continue
            for lang in langs:
                tr = table.get(lang) or table.get(lang.replace("-", "_"))
                tr = tr.get(key) if isinstance(tr, dict) else None
                if isinstance(tr, str) and 0 < len(tr.strip()) < 25:
                    words[w].add(tr.strip().rstrip("…").strip())
    return {**_media().lexicon(), **{k: frozenset(v) for k, v in words.items()}}


def _norm(s: str) -> str:
    """Casefolded, NFC, newlines as \\n, runs of spaces as one: how two texts are compared."""
    s = unicodedata.normalize("NFC", s or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in s.split("\n")).strip().casefold()


# ---- finding things ------------------------------------------------------------------------------------

_EDIT_ROLES = ("AXTextArea", "AXTextField")


def menu_item(pid: int, char: str, mods: int = 0) -> tuple[Any, str] | None:
    """(item, its title) of the menu-bar command with this keyboard shortcut: found by the key
    (AXMenuItemCmdChar, AXMenuItemCmdModifiers; 0 = command alone), never by its title."""
    bar = attr(app_el(pid), "AXMenuBar")
    if bar is None:
        return None
    for el, _d, _p in _bfs(bar, 6000, 8):
        if _s(attr(el, "AXRole")) != "AXMenuItem":
            continue
        key = _s(attr(el, "AXMenuItemCmdChar"))
        same = key.lower() == char.lower() and (attr(el, "AXMenuItemCmdModifiers", 0) or 0) == mods
        if key and same and attr(el, "AXEnabled", True) is not False:
            return el, _s(attr(el, "AXTitle"))
    return None


def _search_like(el: Any) -> bool:
    role, sub = _s(attr(el, "AXRole")), _s(attr(el, "AXSubrole"))
    if "AXSearchField" in (role, sub):
        return True
    words = lexicon()["search"] | lexicon()["find"]
    label = " ".join(_s(attr(el, a)) for a in ("AXPlaceholderValue", "AXDescription", "AXTitle"))
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", label, re.I) for w in words)


def _editable(el: Any) -> bool:
    """A text input NEO may type into: never a password field, never a search box."""
    if el is None or "AXSecureTextField" in (_s(attr(el, "AXRole")), _s(attr(el, "AXSubrole"))):
        return False
    web_editable = attr(el, "AXEditableAncestor") is not None  # contenteditable in a web view
    return (_s(attr(el, "AXRole")) in _EDIT_ROLES or web_editable) and not _search_like(el)


def focused_editable(pid: int) -> Any | None:
    """The app's focused element when it is a text input NEO may type into."""
    el = attr(app_el(pid), "AXFocusedUIElement")
    return el if _editable(el) else None


def composer(win: Any, exclude: tuple = ()) -> Any | None:
    """The window's message composer: the last editable text input (not a search box, not a
    password field), which is the one below a conversation."""
    found = None
    for el, _d, _p in _bfs(win, 4000, 60):
        if el not in exclude and _editable(el) and attr(el, "AXEnabled", True) is not False:
            found = el
    return found


def _label(el: Any) -> str:
    own = _s(attr(el, "AXTitle")) or _s(attr(el, "AXDescription")) or _s(attr(el, "AXValue"))
    return own or _media()._subtree_text(el)


def rows_matching(win: Any, needle: str) -> list[tuple[Any, str]]:
    """Rows, cells, links and buttons whose text holds every word of ``needle`` as whole words
    (casefolded), one per distinct label."""
    words = _norm(needle).split()
    if not words:
        return []
    rxs = [re.compile(rf"(?<!\w){re.escape(w)}(?!\w)") for w in words]
    out: dict[str, Any] = {}
    for el, _d, _p in _bfs(win, 4000, 60):
        if _s(attr(el, "AXRole")) not in ("AXRow", "AXCell", "AXLink", "AXButton"):
            continue
        label = " ".join(_label(el).split())
        if label and all(rx.search(label.casefold()) for rx in rxs):
            out.setdefault(label, el)
    return [(el, label) for label, el in out.items()]


def header_title(win: Any) -> str:
    """Who a conversation window is with: its title, else its first heading or static text."""
    if title := _s(attr(win, "AXTitle")).strip():
        return title
    for role in ("AXHeading", "AXStaticText"):
        for el, _d, _p in _bfs(win, 1500, 30):
            if _s(attr(el, "AXRole")) == role and (t := (_s(attr(el, "AXValue")) or _s(attr(el, "AXTitle"))).strip()):
                return t
    return ""


def transcript_count(win: Any, text: str) -> int:
    """How many non-editable texts in the window hold ``text`` (a sent message shows up as one)."""
    want = _norm(text)
    if not want:
        return 0
    n = 0
    for el, _d, _p in _bfs(win, 8000, 80):
        role = _s(attr(el, "AXRole"))
        if role in ("AXStaticText", "AXTextArea") and not _editable(el) and want in _norm(_s(attr(el, "AXValue"))):
            n += 1
    return n


def commit_button(win: Any) -> Any | None:
    """The window's one enabled button labelled with the OS's own word for "Send" (None when there
    is none, or more than one)."""
    send = {w.casefold() for w in lexicon()["send"]}
    hits = [
        el
        for el, _d, _p in _bfs(win, 4000, 60)
        if _s(attr(el, "AXRole")) == "AXButton"
        and attr(el, "AXEnabled", True) is not False
        and (_s(attr(el, "AXTitle")) or _s(attr(el, "AXDescription"))).strip().casefold() in send
    ]
    return hits[0] if len(hits) == 1 else None


# ---- typing --------------------------------------------------------------------------------------------

_NEWLINE_KEYS = {"return": (36, 0), "shift+return": (36, 1 << 17), "option+return": (36, 1 << 19)}


@pooled
def _pid_of(el: Any) -> int | None:
    try:
        import ApplicationServices as AS

        err, pid = AS.AXUIElementGetPid(el, None)
        return int(pid) if err == 0 else None
    except Exception:  # noqa: BLE001
        return None


@pooled
def _type(pid: int, text: str, newline: str) -> None:
    """Type ``text`` into process ``pid`` only (CGEvent Unicode strings), each line break as the
    ``newline`` chord: a plain Return would send the message in a chat app."""
    import Quartz as Q

    key, flags = _NEWLINE_KEYS.get(newline, _NEWLINE_KEYS["return"])
    for n, line in enumerate(text.split("\n")):
        if n:
            for down in (True, False):
                ev = Q.CGEventCreateKeyboardEvent(None, key, down)
                Q.CGEventSetFlags(ev, flags)
                Q.CGEventPostToPid(pid, ev)
        for i in range(0, len(line), 20):
            piece = line[i : i + 20]
            n16 = len(piece.encode("utf-16-le")) // 2
            for down in (True, False):
                ev = Q.CGEventCreateKeyboardEvent(None, 0, down)
                Q.CGEventKeyboardSetUnicodeString(ev, n16, piece)
                Q.CGEventPostToPid(pid, ev)


def set_text(el: Any, text: str, *, newline: str = "return") -> bool:
    """Put ``text`` into the input and check it reads back the same: AXValue first; if the input
    ignores that (many web inputs), focus it and type into its process."""
    if not _editable(el):
        return False
    set_attr(el, "AXFocused", True)
    if set_attr(el, "AXValue", text) and _norm(_s(attr(el, "AXValue"))) == _norm(text):
        return True
    pid = _pid_of(el)
    if pid is None or _norm(_s(attr(el, "AXValue"))):
        return False  # can't type into it, or a half-set value is there: don't type on top of it
    _type(pid, text, newline)
    return _norm(_s(attr(el, "AXValue"))) == _norm(text)


def clear_text(el: Any, expected: str) -> bool:
    """Take NEO's own text back out before giving up on a send - only when the input still holds
    exactly that text, so nothing the user typed is ever lost."""
    if _norm(_s(attr(el, "AXValue"))) != _norm(expected):
        return False
    return set_attr(el, "AXValue", "") and not _norm(_s(attr(el, "AXValue")))


# ---- web views and the Dock -------------------------------------------------------------------------------


def _web_area(pid: int) -> Any | None:
    win = window(app_el(pid))
    if win is None:
        return None
    return next((el for el, _d, _p in _bfs(win, 3000, 40) if _s(attr(el, "AXRole")) == "AXWebArea"), None)


def web_url(pid: int) -> str:
    """The address of the page in the app's front window ('' if it shows none)."""
    area = _web_area(pid)
    url = attr(area, "AXURL") if area is not None else None
    if url is None:
        return ""
    try:
        return str(url.absoluteString())
    except AttributeError:
        return _s(url)


def web_text(pid: int, limit: int = 8000) -> str:
    """The visible text of the page in the app's front window, up to ``limit`` characters."""
    area = _web_area(pid)
    parts: list[str] = []
    size = 0
    for el, _d, _p in _bfs(area, 20000, 120) if area is not None else ():
        if _s(attr(el, "AXRole")) == "AXStaticText" and (t := _s(attr(el, "AXValue")).strip()):
            parts.append(t)
            size += len(t) + 1
            if size >= limit:
                break
    return "\n".join(parts)[:limit]


@pooled
def _dock_pid() -> int | None:
    try:
        from AppKit import NSRunningApplication

        from neo.decide.inventory import _DOCK_DOMAIN

        apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(_DOCK_DOMAIN) or []
        return int(apps[0].processIdentifier()) if apps else None
    except Exception:  # noqa: BLE001
        return None


def dock_badge(app_name: str) -> str | None:
    """The badge on the app's Dock tile (an unread count, say); None when it has none."""
    pid = _dock_pid()
    if pid is None:
        return None
    want = app_name.strip().casefold()
    for el, _d, _p in _bfs(app_el(pid), 2000, 6):
        if _s(attr(el, "AXSubrole")) == "AXApplicationDockItem" and _s(attr(el, "AXTitle")).strip().casefold() == want:
            return _s(attr(el, "AXStatusLabel")) or None
    return None


# ---- a new note or document through the window (write_note) ------------------------------------------

__all__ += ["window_pid", "value", "is_empty", "multiline"]


async def window_pid(app: Any, ready_s: float) -> int | None:
    """The app's pid once it shows a window, opening it first when it has none (media's launch)."""
    return await _media()._launch(app, ready_s)


def value(el: Any) -> str:
    """The text an input holds now."""
    return _s(attr(el, "AXValue"))


def is_empty(el: Any) -> bool:
    return not _norm(value(el))


def multiline(el: Any) -> bool:
    """A text area or a web editable: a line break is text there, not a submit."""
    return _s(attr(el, "AXRole")) == "AXTextArea" or attr(el, "AXEditableAncestor") is not None
