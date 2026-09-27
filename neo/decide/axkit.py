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
import threading
import unicodedata
from functools import lru_cache
from typing import Any

from neo import effects
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
    "pid_of",
    "needs_sign_in",
    "menu_item",
    "focused_editable",
    "composer",
    "rows_matching",
    "header_title",
    "transcript_count",
    "set_text",
    "set_search",
    "clear_text",
    "commit_button",
    "web_area",
    "in_web",
    "web_url",
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


# An app's web-tree switches are shared by every flow that reads it (a page read, a site search's
# watch): NEO switches them back off when the last of those ends, not the first. app -> [flows
# holding them, the switches NEO turned on (name, previous value)].
_WEB_LOCK = threading.Lock()
_WEB_HELD: dict[Any, list] = {}


def _app_key(app: Any) -> Any:
    return _pid_of(app) or id(app)


def _hold(app: Any, switch: Any) -> list[tuple[str, bool]]:
    """Switch ``app``'s web tree on with ``switch`` (returns what it changed) as one more flow
    holding it."""
    with _WEB_LOCK:
        changed = list(switch())
        held = _WEB_HELD.setdefault(_app_key(app), [0, []])
        held[0] += 1
        held[1] += changed
        return changed


def enable_web_tree(app: Any, rt: str) -> list[tuple[str, bool]]:
    if rt == "native":
        return []  # a native app's web views are always in the tree: no switch to share
    return _hold(app, lambda: _media()._enable_web_tree(app, rt))


def restore_web_tree(app: Any, changed: list[tuple[str, bool]]) -> None:
    """Put back what NEO switched on for ``app``'s web tree, once no other flow still reads it."""
    with _WEB_LOCK:
        key = _app_key(app)
        held = _WEB_HELD.get(key)
        if held is not None:
            held[0] -= 1
            if held[0] > 0:
                return
            del _WEB_HELD[key]
            changed = held[1]
        _media()._restore(app, changed)


def runtime(app_path: str) -> str:
    return _media()._runtime(app_path)


def window(app: Any) -> Any:
    return _media()._window(app)


def pid_of(bundle_id: str) -> int | None:
    """The running app's pid, asked fresh (None when it isn't running)."""
    return _media()._pid_of(bundle_id)


def needs_sign_in(win: Any) -> bool:
    """The window shows a password field (a sign-in wall)."""
    return _media()._needs_sign_in(win)


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

    from neo.tools.computer.input import chunks

    effects.guard("input", "type into another app", via=Q.CGEventPostToPid)
    key, flags = _NEWLINE_KEYS.get(newline, _NEWLINE_KEYS["return"])
    for n, line in enumerate(text.split("\n")):
        if n:
            for down in (True, False):
                ev = Q.CGEventCreateKeyboardEvent(None, key, down)
                Q.CGEventSetFlags(ev, flags)
                Q.CGEventPostToPid(pid, ev)
        for piece in chunks(line):  # at most 20 UTF-16 units, never half a character
            n16 = len(piece.encode("utf-16-le")) // 2
            for down in (True, False):
                ev = Q.CGEventCreateKeyboardEvent(None, 0, down)
                Q.CGEventKeyboardSetUnicodeString(ev, n16, piece)
                Q.CGEventPostToPid(pid, ev)


def set_text(el: Any, text: str, *, newline: str = "return") -> bool:
    """Put ``text`` into the input and check it reads back the same: AXValue first; if the input
    ignores that (many web inputs), focus it and type into its process."""
    return _editable(el) and _put(el, text, newline)


def set_search(el: Any, text: str) -> bool:
    """:func:`set_text` for a search or address field (which :func:`set_text` refuses, so a
    message never lands in one), never a password field."""
    if el is None or "AXSecureTextField" in (_s(attr(el, "AXRole")), _s(attr(el, "AXSubrole"))):
        return False
    return _s(attr(el, "AXRole")) in (*_EDIT_ROLES, "AXComboBox", "AXSearchField") and _put(el, text, "return")


# A field may show what was set, or typed, a moment later (an app that applies it asynchronously,
# keystrokes still on their way): its value is read back every 20 ms for up to 400 ms before NEO
# decides it didn't take — never a false "didn't take the text", never the text typed a second time.
_READBACK_S = 0.4
_READBACK_EVERY_S = 0.02


def _reads(el: Any, text: str) -> bool:
    """The input now reads ``text``, looked at until it does (at most _READBACK_S)."""
    from neo.tools.computer import wait

    want = _norm(text)
    return bool(wait.until_sync(lambda: _norm(_s(attr(el, "AXValue"))) == want, _READBACK_S, every=_READBACK_EVERY_S))


def _put(el: Any, text: str, newline: str) -> bool:
    set_attr(el, "AXFocused", True)
    if set_attr(el, "AXValue", text) and _reads(el, text):
        return True
    pid = _pid_of(el)
    if pid is None or _norm(_s(attr(el, "AXValue"))):
        return False  # can't type into it, or a half-set value is there: don't type on top of it
    _type(pid, text, newline)
    return _reads(el, text)


def clear_text(el: Any, expected: str) -> bool:
    """Take NEO's own text back out before giving up on a send - only when the input still holds
    exactly that text, so nothing the user typed is ever lost."""
    if _norm(_s(attr(el, "AXValue"))) != _norm(expected):
        return False
    return set_attr(el, "AXValue", "") and not _norm(_s(attr(el, "AXValue")))


# ---- web views and the Dock -------------------------------------------------------------------------------


def web_area(pid: int) -> Any | None:
    """The web content of the app's front window (a browser tab, a web view)."""
    win = window(app_el(pid))
    if win is None:
        return None
    return next((el for el, _d, _p in _bfs(win, 3000, 40) if _s(attr(el, "AXRole")) == "AXWebArea"), None)


def in_web(el: Any, max_depth: int = 60) -> bool:
    """Is ``el`` web content (a page's own field, say) rather than one of the app's controls? A
    chain of parents that doesn't end within ``max_depth`` can't be told, so it counts as web."""
    for _ in range(max_depth):
        if el is None:
            return False
        if _s(attr(el, "AXRole")) == "AXWebArea":
            return True
        el = attr(el, "AXParent")
    return True


def web_url(pid: int) -> str:
    """The address of the page in the app's front window ('' if it shows none)."""
    return _url_of(web_area(pid))


def _url_of(area: Any) -> str:
    url = attr(area, "AXURL") if area is not None else None
    if url is None:
        return ""
    try:
        return str(url.absoluteString())
    except AttributeError:
        return _s(url)


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


# ---- a whole page, read for the user (page_read) ------------------------------------------------------

__all__ += ["enable_web_reading", "web_title", "page_text", "web_page", "web_view"]

# Controls say what can be done on a page, not what it says: their words are skipped. An image's
# words are left to a screenshot (see neo.decide.kinds.site).
_NOT_CONTENT = frozenset({
    "AXButton", "AXPopUpButton", "AXMenuButton", "AXMenu", "AXMenuBar", "AXMenuItem", "AXToolbar", "AXTextField",
    "AXSecureTextField", "AXSearchField", "AXComboBox", "AXCheckBox", "AXRadioButton", "AXSlider", "AXIncrementor",
    "AXScrollBar", "AXDisclosureTriangle", "AXTabGroup", "AXImage",
})  # fmt: skip
# The landmarks every page of a site repeats: its header, navigation, footer and search box.
_BOILERPLATE = frozenset({"AXLandmarkBanner", "AXLandmarkNavigation", "AXLandmarkContentInfo", "AXLandmarkSearch"})
_MAIN = frozenset({"AXLandmarkMain", "AXDocumentArticle"})
_ASIDE = frozenset({"AXLandmarkComplementary"})
# The same parts told by the class or id its authors gave them ("footer", "site-nav", "cookie-banner"):
# a name made only of these words and of place words, one of them at least. A name with any other
# word in it ("content-sidebar-wrap", "no-ads", "cookie-recipe") says nothing. Only outside the main
# content: inside it, a word like these may be the content itself.
_BOILER_WORDS = frozenset({
    "nav", "navbar", "navigation", "footer", "sidebar", "cookie", "cookies", "consent", "breadcrumb",
    "breadcrumbs", "newsletter", "advert", "advertisement", "ad", "ads",
})  # fmt: skip
_PLACE_WORDS = frozenset({
    "site", "page", "main", "top", "bottom", "global", "primary", "secondary", "left", "right", "menu",
    "bar", "banner", "notice", "links", "widget", "widgets", "area", "wrap", "wrapper", "container",
    "inner", "outer", "section", "block", "region",
})  # fmt: skip
_HINT_DEPTH = 30  # page furniture sits near the top of the tree: deeper containers aren't asked
_EDITOR_MIN = 80  # an editable area this long is a document in a web editor, not a form's field
_MAIN_MIN = 400  # a main landmark with less text than this isn't where the page's content is
_THIN = 200  # a read this short after the filters is read again without them


def enable_web_reading(app: Any, rt: str) -> list[tuple[str, bool]]:
    """:func:`enable_web_tree` for reading a whole page. A Chromium-based app is given both
    switches, AXManualAccessibility (which newer versions honour) and AXEnhancedUserInterface
    (which older ones need); an Electron app the first, and the second only when that is refused.
    Either switch already on (the user's own assistive setup, or another NEO flow's) means the tree
    is there: nothing more is switched on. Returns what this call changed; :func:`restore_web_tree`
    puts the switches back when the last flow reading the app ends."""
    if rt == "native":
        return []
    return _hold(app, lambda: _switch_for_reading(app, rt))


def _switch_for_reading(app: Any, rt: str) -> list[tuple[str, bool]]:
    names = ("AXManualAccessibility", "AXEnhancedUserInterface")
    prevs = [attr(app, n) for n in names]
    if any(v in (True, 1) for v in prevs):
        return []  # the tree is already exposed: nothing more is switched on
    changed: list[tuple[str, bool]] = []
    for name, prev in zip(names, prevs, strict=True):
        if set_attr(app, name, True):
            changed.append((name, bool(prev)))
            if rt == "electron":
                break
    return changed


def web_title(pid: int, area: Any = None) -> str:
    """The title of the page in the app's front window (``area``, when already found): the web
    area's own, else the window's."""
    area = web_area(pid) if area is None else area
    for el, name in ((area, "AXTitle"), (area, "AXDescription"), (window(app_el(pid)), "AXTitle")):
        if el is not None and (t := " ".join(_s(attr(el, name)).split())):
            return t
    return ""


def _furniture_name(name: str) -> bool:
    words = [w for w in re.split(r"[-_]+", name.casefold()) if w and not w.isdigit()]
    return bool(words) and all(w in _BOILER_WORDS or w in _PLACE_WORDS for w in words) and not _BOILER_WORDS.isdisjoint(words)


def _boilerplate(el: Any) -> bool:
    """The container's class names or id say it is page furniture (a navigation bar, a footer…)."""
    names = attr(el, "AXDOMClassList")
    try:
        classes = [_s(n) for n in names] if names is not None and not isinstance(names, str) else []
    except TypeError:
        classes = []
    return any(_furniture_name(n) for n in " ".join([_s(attr(el, "AXDOMIdentifier")), *classes]).split())


def _heading(el: Any) -> tuple[int, str]:
    """(level 1-6, text) of a heading: its level is its AXValue; its words, its texts' or its label."""
    v = attr(el, "AXValue")
    level = v if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 6 else 2
    words = [
        _s(attr(c, "AXValue"))
        for c, _d, _p in _bfs(el, 200, 12)
        if c is not el and _s(attr(c, "AXRole")) == "AXStaticText"
    ]
    text = " ".join(" ".join(words).split()) or " ".join((_s(attr(el, "AXTitle")) or _s(attr(el, "AXDescription"))).split())
    return level, text


def _pieces(
    area: Any, budget: int, max_nodes: int, filters: bool, stop: threading.Event | None = None
) -> list[tuple[str, int, int, str]]:
    """The page's words in reading order (depth first): (where, block, cell, text) per text, where
    is 'main' (a main landmark or an article), 'aside' (a complementary one), 'furniture' (a
    container named like page furniture, walked after the rest: only a main landmark or a title in
    it may be content) or 'body'. Texts of one block (a paragraph, a list item, a table row:
    anything but a link) belong on one line; ``cell`` tells a row's cells apart. A heading is one
    piece, its text marked with its level. ``filters``: skip the site's header, navigation, footer
    and search landmarks, and mark furniture. ``stop`` set: the read ends there."""
    out: list[tuple[tuple[int, ...], tuple[str, int, int, str]]] = []  # (its place in the page, piece)
    size = seq = n = 0
    stack: list[tuple[Any, str, int, int, int, tuple[int, ...]]] = [(area, "body", 0, 0, 0, ())]
    later: list[tuple[Any, str, int, int, int, tuple[int, ...]]] = []  # furniture, in page order
    while n < max_nodes and size < budget and not (stop is not None and stop.is_set()):
        if not stack:
            if not later:
                break
            stack, later = later[::-1], []
        el, where, block, cell, depth, path = stack.pop()
        n += 1
        role = _s(attr(el, "AXRole"))
        if role == "AXStaticText":
            if t := " ".join(_s(attr(el, "AXValue")).split()):
                out.append((path, (where, block, cell, t)))
                size += len(t) + 1
            continue
        if role in _NOT_CONTENT:
            continue
        if role == "AXTextArea":
            t = _s(attr(el, "AXValue")).strip()
            if len(t) >= _EDITOR_MIN:
                seq += 1
                out.append((path, (where, seq, 0, t)))
                size += len(t) + 1
            continue
        sub = _s(attr(el, "AXSubrole"))
        if filters and sub in _BOILERPLATE:
            continue
        if (
            filters
            and where in ("body", "aside")
            and depth <= _HINT_DEPTH
            and role in ("AXGroup", "AXList")
            and _boilerplate(el)
        ):
            later.append((el, "furniture", block, cell, depth, path))  # walked after the rest
            continue
        if sub in _MAIN:
            where = "main"
        elif sub in _ASIDE and where != "main":
            where = "aside" if where != "furniture" else where
        if role == "AXHeading":
            level, t = _heading(el)
            if t:
                seq += 1
                # the page's title, even in a container named like furniture
                out.append((path, ("body" if where == "furniture" and level == 1 else where, seq, 0, "#" * level + " " + t)))
                size += len(t) + level + 2
            continue
        seq += 1
        if role == "AXCell":
            cell = seq  # the row stays the block: its cells share a line
        elif role != "AXLink":  # a link's words are part of the sentence around it
            block, cell = seq, 0
        if depth < 150:
            kids = list(attr(el, "AXChildren") or [])
            stack.extend((k, where, block, cell, depth + 1, (*path, i)) for i, k in reversed(list(enumerate(kids))))
    return [piece for _path, piece in sorted(out, key=lambda x: x[0])]


_CLOSERS = tuple(",.;:!?)]}%»”’")


def _joins(prev: str, text: str) -> bool:
    """Does ``text`` carry on the sentence ``prev`` left open (words an inline element, a bold
    word say, split off)? A text that starts in lower case or with closing punctuation does."""
    return not prev.startswith("#") and not prev.endswith((".", "!", "?", ":")) and (
        text[0].islower() or text.startswith(_CLOSERS)
    )


def _lines(pieces: list[tuple[str, int, int, str]]) -> list[str]:
    """One line per block, its texts joined; a table row's cells joined with ' | '."""
    lines: list[str] = []
    last: tuple[int, int] | None = None
    for _w, block, cell, text in pieces:
        if lines and last is not None and last[0] == block and cell and last[1] and cell != last[1]:
            lines[-1] += " | " + text
        elif lines and ((last is not None and last[0] == block) or _joins(lines[-1], text)):
            lines[-1] += ("" if text.startswith(_CLOSERS) else " ") + text
        elif not lines or lines[-1] != text:  # a text repeated right after itself is said once
            lines.append(text)
        last = (block, cell)
    return lines


def _compose(pieces: list[tuple[str, int, int, str]], limit: int) -> str:
    """The main content first, then the rest of the page (a sidebar last), cut at ``limit``. The
    furniture is left out, unless the page would be (nearly) empty without it."""
    kept = [p for p in pieces if p[0] != "furniture"]
    if sum(len(p[3]) + 1 for p in kept) >= _THIN:
        pieces = kept
    else:
        pieces = [("body", *p[1:]) if p[0] == "furniture" else p for p in pieces]
    main = _lines([p for p in pieces if p[0] == "main"])
    if sum(len(x) for x in main) >= _MAIN_MIN:
        rest = _lines([p for p in pieces if p[0] == "body"]) + _lines([p for p in pieces if p[0] == "aside"])
        text = "\n".join(main) + ("\n\n[elsewhere on the page]\n" + "\n".join(rest) if rest else "")
    else:
        text = "\n".join(_lines([p for p in pieces if p[0] != "aside"]) + _lines([p for p in pieces if p[0] == "aside"]))
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip() + "\n…"


def page_text(area: Any, limit: int = 8000, max_nodes: int = 6000, *, stop: threading.Event | None = None) -> str:
    """What a page says, for a person: headings (marked '#', '##'…) and the main content first,
    each paragraph on a line, without the site's header, navigation, footer, search box or
    controls where the tree tells them apart, cut at ``limit`` characters. A page those filters
    leave (nearly) empty is read again without them. ``stop`` set: what was read so far."""
    if area is None:
        return ""
    text = _compose(_pieces(area, limit * 3, max_nodes, True, stop), limit)
    if len(text) < _THIN and not (stop is not None and stop.is_set()):
        whole = _compose(_pieces(area, limit * 3, max_nodes, False, stop), limit)
        text = whole if len(whole) > len(text) else text
    return text


def web_page(
    pid: int, limit: int = 8000, *, area: Any = None, stop: threading.Event | None = None
) -> tuple[str, str, str]:
    """(address, title, text) of the page in the app's front window (``area``, when already
    found; see :func:`page_text`)."""
    area = web_area(pid) if area is None else area
    if area is None:
        return "", "", ""
    return _url_of(area), web_title(pid, area), page_text(area, limit, stop=stop)


def _pair(v: Any, kind: str) -> tuple[float, float] | None:
    """An AXPosition (x, y) or AXSize (width, height) as numbers."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return float(v[0]), float(v[1])
    try:
        import ApplicationServices as AS

        point = kind == "point"
        ok, p = AS.AXValueGetValue(v, AS.kAXValueCGPointType if point else AS.kAXValueCGSizeType, None)
    except Exception:  # noqa: BLE001 - not a geometry value
        return None
    if not ok:
        return None
    return (float(p.x), float(p.y)) if point else (float(p.width), float(p.height))


def _frame(el: Any) -> tuple[float, float, float, float] | None:
    pos, size = _pair(attr(el, "AXPosition"), "point"), _pair(attr(el, "AXSize"), "size")
    return (*pos, *size) if pos is not None and size is not None else None


def web_view(area: Any) -> tuple[float, float, float, float] | None:
    """The part of the screen the page shows, (x, y, width, height) in points from the top left:
    the web area's frame, cut to the scroll areas around it (a web area may be the whole document,
    most of it scrolled out of view). None when that can't be told."""
    box = _frame(area) if area is not None else None
    el = area
    for _ in range(6):
        el = attr(el, "AXParent") if el is not None else None
        if el is None or box is None:
            break
        if _s(attr(el, "AXRole")) == "AXScrollArea" and (clip := _frame(el)) is not None:
            left, top = max(box[0], clip[0]), max(box[1], clip[1])
            right, bottom = min(box[0] + box[2], clip[0] + clip[2]), min(box[1] + box[3], clip[1] + clip[3])
            box = (left, top, right - left, bottom - top)
    return box if box is not None and box[2] >= 1 and box[3] >= 1 else None


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
