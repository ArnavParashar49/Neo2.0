"""in_front: "this" — what the user selected, copied or has in front of them, read on their own request.

"Summarize what I copied", "translate the selected text", "explain this paragraph", "add this to my
calendar": the model gets the words themselves instead of hunting for them on the screen. One tool,
four places to look (``what``):

* **selection** — the selected text in the app the user works in: its focused element's
  ``AXSelectedText``, or, on a web page (Safari, Electron), the text of its selected text-marker range.
  Chromium and Electron apps get ax.py's reading switches first. That app is never NEO's own window or
  an accessory panel the user typed into (a notch app's, a menu-bar extra's): it is the regular app
  behind them. Never a password field, never a protected app. A secret's value in it is left out.

  **Copied, when a web page keeps it to itself** (``NEO_CONTEXT_COPY_FALLBACK``, on by default).
  Google Chrome — and so the Chromium browsers and Electron apps — keeps a page's selection to itself:
  the focused web area (or the group in it) has no ``AXSelectedText`` at all, and the string of its
  selected text-marker range comes back empty. ⌘C is the one way to those words, so there — and only
  there: web content whose focused element has no ``AXSelectedText`` — NEO copies the selection and
  puts the clipboard back, as Raycast and PopClip do (:func:`copy_selection`). Never where
  Accessibility answers that nothing is selected (a text view, a terminal, a page's text field), nor
  outside a page (a password manager's list, whose ⌘C copies the password; a terminal, where it may
  interrupt):

  1. never while the clipboard holds a private copy (the markers below: a password manager's is never
     touched, not even kept and put back), a copy from another device (Universal Clipboard: fetched
     only on demand), more than 32 MB, or anything else that can't be kept whole in a moment (a type
     whose bytes don't come); nor on a Mac set to ask before, or never to let, apps read it; never in a
     password field (refused before anything is copied) or with no focused element to tell;
  2. every item on the clipboard is kept, with every type's bytes, and its change count;
  3. ⌘C goes to the user's app alone (CGEventPostToPid, through input.py's door: its watcher, never
     NEO's own processes or a protected app);
  4. the app has up to half a second to copy (the change count moves);
  5. the copy's plain text is read — unless it carries a private marker: then nothing is;
  6. and whatever happened (nothing copied, an error on the way), the clipboard is put back: cleared,
     then the kept items written, every type's bytes as they were — an empty one stays empty. One that
     nothing changed is left alone: it already is what was kept. Then, for a second more (in a thread
     of its own, the reply not kept waiting), a write that comes late — a slow app's copy, a page's
     own second write — is put back over too (a few times at most, and never a private copy). A
     clipboard that couldn't be written back is said, never just logged.

  The words are then the selection's: a secret's value left out, fenced, cut to length. One copy at a
  time, and no clipboard read while one is out (or its late watch). Caveats: an app that isn't the one
  in front (the user typed to NEO in its window) may copy nothing — its Copy works on its active
  window, and it has none — so then NEO says it isn't in front rather than that nothing is selected
  (asked by voice, or from a notch panel, the app stays in front); a page may copy something of its
  own for ⌘C (a web code editor's current line); an app slower still (past the late watch) leaves its
  copy on the clipboard; a clipboard-history app may keep the copy.
* **clipboard** — only when the user's own words name it ("what I copied", "my clipboard"). The
  pasteboard's types are checked before anything is read: a copy marked private
  (https://nspasteboard.org — a password manager's "concealed" copy, a "transient" or an
  "auto-generated" one, or an older vendor marker) is refused unread, and so is anything on a Mac set to never let apps read the
  clipboard (macOS 26's ``accessBehavior``). Text that looks like a password, a key or a code is
  refused. Files are their names only; a picture is its size.
* **window** — the front window's app, title and document *name* (never its folders). A browser's says
  to call page_read for the page itself, which reads it properly.
* **files** — the names of the items selected in Finder, through a read-only AppleScript.

``auto`` (the default): the place the user's words name ("copied" → the clipboard, "selected" → the
selection, "these files" → Finder's); otherwise the selection when there is one, else the window.

**Who may ask.** Only the user's own words, checked here whoever calls (neo2.core.agent.context_words):
Gemini Live's own direct call for the user's sentence (LIVE_CALL); the agent's call in a request the
Session marked as the user's own (USERS_WORDS); never a model's goal (a request Live hands the Session),
an approval's replay, an embedded agent, or the call's own arguments. ``NEO_CONTEXT_CAPTURE=0`` turns it
all off (NEO.app hides the tool from every model at start: :func:`start`).

**What happens to it.** The words are third-party content: fenced as data for the model (untrusted),
and the request is tainted, so a send or a link that follows asks first and nothing is learned from the
run. Nothing is written anywhere: no journal (only the user's line and NEO's reply are, marked as
quoting), no log (the core log has the tool's name and the length of its result), no file — and a
clipboard a copy borrowed is put back as it was. They reach a cloud brain only as the reply's own
brain, on the user's request — as page_read's page does.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from neo2.core import effects, logger, runtime
from neo2.core.agent import context_words
from neo2.core.agent.registry import LIVE_CALL, USERS_WORDS, ToolContext, ToolOutput, catalog, tool
from neo2.core.config import settings
from neo2.core.decide._objc import pooled
from neo2.core.memory import redact

log = logger(__name__)

MAX_CHARS = 4000  # the most of what was read that reaches the model, NEO's own words included
# What is looked at, at most: a select-all in a long document is cut before its secrets are looked for
# (twice what is sent, so one that straddles the cut is still seen whole).
_LOOK_CHARS = 2 * MAX_CHARS
_MAC = frozenset({"macos"})
_WHATS = ("auto", "selection", "clipboard", "window", "files")
_NAMES_MAX = 40  # file names listed; the rest are counted
_WEB_UP = 6  # parents walked up from the focused element to the web area its page's selection is on

# The markers apps put on what they copy (https://nspasteboard.org): a password manager's copy
# (concealed), a value meant for a moment (transient), one an app put there by itself (auto-generated)
# — and the older and vendor ones clipboard tools still honour: Butler's and TextExpander's transient
# one, 1Password's, KeeWeb's, TypeIt4Me's expansion, a password generator's.
PRIVATE_TYPES = frozenset(
    {
        "org.nspasteboard.ConcealedType",
        "org.nspasteboard.TransientType",
        "org.nspasteboard.AutoGeneratedType",
        "de.petermaurer.TransientPasteboardType",
        "com.agilebits.onepassword",
        "net.antelle.keeweb",
        "com.typeit4me.clipping",
        "Pasteboard generator type",
    }
)
_STRING = "public.utf8-plain-text"  # NSPasteboardTypeString
_FILE_URL = "public.file-url"  # NSPasteboardTypeFileURL
_IMAGES = frozenset({"public.png", "public.tiff", "public.jpeg", "public.heic"})
_REMOTE = "com.apple.is-remote-clipboard"  # Universal Clipboard: another device's copy, fetched on demand

# Copying a selection Accessibility doesn't give (copy_selection).
COPY_WAIT_S = 0.5  # the longest an app has to put its copy on the clipboard
LATE_WATCH_S = 1.0  # then how long a late write (a slow copy, a page's second one) is put back over too
_LATE_PUTS = 2  # the most times it is, meanwhile (a tool that rewrites every copy: no tug of war)
_COPY_POLL_S = 0.02  # how often the clipboard's change count is looked at meanwhile
CLIPBOARD_CAP_BYTES = 32 * 1024 * 1024  # a clipboard bigger than this is never copied over (kept, put back)
_KEEP_S = 2.0  # the longest keeping it may take (an app slow to give what it promised: not copied over)
# One look at the clipboard at a time: in_front runs beside itself, and a read in the middle of a copy
# would read the copy, a second copy would keep the first one's and put that back. A look waits for it
# _LOCK_WAIT_S at most (a copy whose app hangs never parks every later one).
_CLIPBOARD = threading.Lock()
_LOCK_WAIT_S = 5.0

# The Finder selection's names, one per line. Read-only (policy.script_is_read_only says so: a test
# holds it to that), one app, nothing it starts or changes.
FINDER_SCRIPT = """tell application "Finder"
	set theNames to {}
	repeat with theItem in (get selection)
		set end of theNames to (name of theItem)
	end repeat
end tell
set AppleScript's text item delimiters to linefeed
return theNames as text"""

# (Said when the words this call answers didn't read as a request about the selection or the clipboard:
# the user may well have asked for something — just not, as far as NEO can tell, for that.)
_NOT_ASKED = (
    "That request didn't read as one about your selection or clipboard, so NEO didn't look: it reads them "
    "only when you ask about them in so many words (“what I copied”, “the selected text”, “translate this”)."
)
_CLIPBOARD_ONLY = (
    'NEO reads the clipboard only when the user says so ("what I copied", "my clipboard"): call in_front '
    "with what=auto for what they selected or have in front of them."
)


class _No(Exception):
    """Why nothing was read: NEO's own words for the user (never anything it read)."""


class _NoCopy(Exception):
    """Why the selection couldn't be copied to be read (mostly before any key was pressed): NEO's own
    words, never anything of the clipboard's."""


_PRIVATE_CLIPBOARD = (
    "your clipboard holds a private copy (a password manager's, say), which NEO never touches, so it "
    "didn't copy the selection to read it"
)
_REMOTE_CLIPBOARD = (
    "your clipboard holds a copy from another device, which NEO would have to fetch to put back, so it "
    "didn't copy the selection to read it"
)
_BIG_CLIPBOARD = (
    "what's on your clipboard is too big for NEO to put back, so it didn't copy the selection to read it"
)
_UNKEPT_CLIPBOARD = (
    "something on your clipboard couldn't be kept to put back, so NEO didn't copy the selection to read it"
)
_SLOW_CLIPBOARD = (
    "keeping what's on your clipboard to put back took too long, so NEO didn't copy the selection to read it"
)
_DENIED_CLIPBOARD = "this Mac never lets NEO read the clipboard, so it couldn't copy the selection to read it"
_ASKS_CLIPBOARD = (
    "this Mac asks before NEO reads the clipboard, so it didn't copy the selection to read it (that would "
    "ask you about your clipboard)"
)
_MOVED_CLIPBOARD = "your clipboard changed just then, so NEO didn't copy the selection to read it"
_BUSY_CLIPBOARD = "your clipboard is busy with another look of NEO's just now: ask again in a moment"
_BEHIND = (
    "copying it gave nothing — maybe because it isn't the app in front, and an app may copy only then. "
    "Ask again with it in front (by voice, for one)"
)
_NOT_PUT_BACK = (
    "NEO copied the selection to read it but couldn't put your clipboard back as it was afterwards: what "
    "you had copied before may be gone"
)


# ---- who may ask ----------------------------------------------------------------------------------


def users_words(ctx: ToolContext) -> str:
    """The user's own sentence this call answers ("" = none): what the Session marked as theirs (a
    fast path, an agent run for their request), or the sentence Gemini Live hands its own direct call
    (LIVE_CALL) — never a model's goal (a request Live hands the Session has origin "live" but no
    direct call), an approval's replay, or the call's own arguments (context_words.wants never reads
    those as a request)."""
    if ctx.origin in ("user", "agent"):
        return str(ctx.get(USERS_WORDS, "") or "")
    if ctx.origin == "live" and ctx.get(LIVE_CALL) is True:
        return str(ctx.user_text or "")
    return ""


def enabled() -> bool:
    """NEO_CONTEXT_CAPTURE: on unless the user switched it off."""
    return bool(getattr(settings(), "context_capture", True))


def copy_fallback() -> bool:
    """NEO_CONTEXT_COPY_FALLBACK: a selection Accessibility doesn't give is copied (on unless switched
    off)."""
    return bool(getattr(settings(), "context_copy_fallback", True))


def start() -> None:
    """NEO.app is starting: switched off, in_front is hidden from every model — Gemini Live's
    declarations, the agent's tool selector, the Session's hook — so off is NEO as it was before it,
    not a tool that is offered and only refuses. (Read at start, like the setting itself.)"""
    t = catalog().get("in_front")
    if t is not None:
        t.hidden = not enabled()


def refusal(ctx: ToolContext) -> str:
    """Why this call may read nothing ("" = it may)."""
    if runtime.library_mode() or runtime.library_bound():
        return "in_front reads only NEO.app's own user's selection and clipboard."
    if not enabled():
        return "Reading what you selected or copied is off in NEO's settings."
    if not context_words.wants(users_words(ctx)):
        return _NOT_ASKED
    return ""


# ---- the app the user works in ------------------------------------------------------------------------


@pooled
def _front() -> tuple[str, int]:
    """(name, pid) of the app in front, as the input tools see it."""
    effects.guard(effects.CAPTURE, "read which app is in front")
    from neo2.core.tools.computer import ax

    return ax.frontmost_app()


@pooled
def _window_owners() -> list[tuple[str, int]]:
    """(owner name, pid) of the on-screen app windows (layer 0), front to back, each app once."""
    effects.guard(effects.CAPTURE, "read which windows are on screen")
    import Quartz as Q

    out: dict[int, str] = {}
    opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
    for w in Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or []:
        p = int(w.get("kCGWindowOwnerPID", 0) or 0)
        if p and int(w.get("kCGWindowLayer", 1)) == 0 and p not in out:
            out[p] = str(w.get("kCGWindowOwnerName", "") or "")
    return [(n, p) for p, n in out.items()]


def _the_users(name: str, pid: int) -> bool:
    """A regular app (windows and a menu bar) that isn't NEO: an accessory panel the user typed into —
    a notch app's, a menu-bar extra's — is told apart by how it runs, never by its name."""
    from neo2.core.tools.computer import ax

    return bool(pid) and not ax.is_neo(pid, name) and ax.regular(pid)


def user_app() -> tuple[str, int]:
    """The app the user works in: the one in front — or, while NEO's own window or an accessory panel
    is in front (the user just typed to NEO there), the regular app whose window is frontmost behind it.
    ("", 0): none."""
    name, pid = _front()
    if _the_users(name, pid):
        return name, pid
    for n, p in _window_owners():
        if p != pid and _the_users(n, p):
            return n, p
    return "", 0


def _check(name: str, pid: int) -> None:
    from neo2.core.tools.computer import gates

    if not pid:
        raise _No("I can't tell which app you're working in.")
    if gates._protected(name, pid):
        raise _No(f"{name or 'That app'} is an app NEO never reads or touches.")


# ---- the selection --------------------------------------------------------------------------------------


@pooled
def _focused_in(pid: int) -> Any:
    """The app's focused element (None), its web tree switched on first when it is a Chromium or
    Electron app (ax.py's reading switches: nothing anyone sees changes)."""
    effects.guard(effects.CAPTURE, "read what is selected in another app")
    import ApplicationServices as AS

    from neo2.core.tools.computer import ax

    root = ax._app_el(pid)
    if ax._expose_web(pid, root):
        ax._web_settle(ax._attr(root, AS.kAXFocusedWindowAttribute) or root)
    el = ax._attr(root, AS.kAXFocusedUIElementAttribute)
    return None if el is None else ax._bounded(el)


def _roles(el: Any) -> tuple[str, str]:
    from neo2.core.tools.computer import ax

    v = ax._attrs(el, ("AXRole", "AXSubrole"))
    return ax._str(v.get("AXRole")), ax._str(v.get("AXSubrole"))


def _marker_text(el: Any) -> str:
    """The words of a web page's selected text-marker range on this element ("" = none)."""
    from neo2.core.tools.computer import ax

    rng = ax._attr(el, "AXSelectedTextMarkerRange")
    if rng is None:
        return ""
    try:
        import ApplicationServices as AS

        err, s = AS.AXUIElementCopyParameterizedAttributeValue(el, "AXStringForTextMarkerRange", rng, None)
    except Exception:  # noqa: BLE001 - an element that can't be asked has nothing selected
        return ""
    return ax._str(s) if err == 0 else ""


def _selected_text(el: Any) -> tuple[str, bool]:
    """(the selected text of the focused element — or, on a web page, of its text-marker range, on the
    element or the web area it is in; whether that is a page's selection Accessibility keeps to itself:
    none read, the focused element has no AXSelectedText at all — not an empty one, which says nothing
    is selected there — and it is, or is in, a web area. The one place a copy is tried.)"""
    from neo2.core.tools.computer import ax

    sel = ax._attrs(el, ("AXSelectedText", "AXRole")).get("AXSelectedText")
    text = ax._str(sel)
    if text.strip():
        return text, False
    node = el
    for _ in range(_WEB_UP):
        if node is None:
            break
        got = _marker_text(node)
        if got.strip():
            return got, False
        if ax._str(ax._attr(node, "AXRole")) == "AXWebArea":
            return "", sel is None
        node = ax._attr(node, "AXParent")
    return "", False


@pooled
def read_selection() -> tuple[str, str, str]:
    """(the app, its selected text — "" when nothing is selected, why NEO couldn't copy it to read it —
    "" when it could, or didn't need to). What Accessibility gives first; with none on a web page that
    keeps its selection to itself, a copy (:func:`copy_selection`) — never in a password field, never
    with no focused element to tell."""
    from neo2.core.tools.computer import ax

    if not ax.is_trusted():
        raise _No("Accessibility is off for NEO, so I can't read what's selected.")
    name, pid = user_app()
    _check(name, pid)
    el = _focused_in(pid)
    if el is None:
        return name, "", ""
    if "AXSecureTextField" in _roles(el):
        raise _No("That's a password field: NEO never reads those.")
    text, kept_to_itself = _selected_text(el)
    if text.strip() or not kept_to_itself or not copy_fallback():
        return name, text, ""
    try:
        return name, copy_selection(pid), ""
    except _NoCopy as why:
        return name, "", str(why)


# ---- the window -------------------------------------------------------------------------------------------


@pooled
def _window_facts(pid: int) -> tuple[str, str]:
    """(title, AXDocument) of the app's focused (else main) window."""
    effects.guard(effects.CAPTURE, "read the front window's title")
    from neo2.core.tools.computer import ax

    v = ax._attrs(ax._app_el(pid), ("AXFocusedWindow", "AXMainWindow"))
    win = v.get("AXFocusedWindow") or v.get("AXMainWindow")
    if win is None:
        return "", ""
    w = ax._attrs(win, ("AXTitle", "AXDocument"))
    return ax._str(w.get("AXTitle")), ax._str(w.get("AXDocument"))


def _shows_web(pid: int) -> bool:
    """Does this app show web pages (a browser, a web app)? By what the app is (the inventory's
    evidence), never its name."""
    try:
        from neo2.core.decide import inventory
        from neo2.core.decide.kinds import site

        return site._shows_web(inventory.app_for_pid(pid))
    except Exception:  # noqa: BLE001 - can't tell: no page_read pointer
        return False


def _document(doc: str) -> str:
    """A window's document as its name: a file's name (never its folders), a page's site."""
    if not doc:
        return ""
    u = urlparse(doc)
    if u.scheme == "file":
        return PurePosixPath(unquote(u.path)).name
    if u.scheme in ("http", "https"):
        return f"a page on {u.hostname}" if u.hostname else ""
    return ""


@pooled
def read_window() -> str:
    name, pid = user_app()
    _check(name, pid)
    title, doc = _window_facts(pid)
    parts = [f"In front: {name}"]
    if title:
        parts.append(f"its window “{title}”")
    if doc_name := _document(doc):
        parts.append(f"its document {doc_name}")
    out = ", ".join(parts) + "."
    if _shows_web(pid):
        out += " It shows web pages: for the page itself, call page_read."
    return out


# ---- the clipboard ----------------------------------------------------------------------------------------


@pooled
def _pasteboard() -> Any:
    from AppKit import NSPasteboard

    effects.guard("clipboard", "read the clipboard", via=NSPasteboard)
    return NSPasteboard.generalPasteboard()


def _behavior(name: str, sdk: int) -> int:
    try:
        import AppKit

        return int(getattr(AppKit, f"NSPasteboardAccessBehavior{name}"))
    except Exception:  # noqa: BLE001 - an older PyObjC: the SDK's value
        return sdk


def _always_deny() -> int:
    """NSPasteboardAccessBehaviorAlwaysDeny (macOS 26: the user never lets apps read the clipboard)."""
    return _behavior("AlwaysDeny", 3)


def _asks() -> int:
    """NSPasteboardAccessBehaviorAsk (macOS 26: the user is asked each time an app reads it)."""
    return _behavior("Ask", 1)


def _access(pb: Any) -> int | None:
    get = getattr(pb, "accessBehavior", None)
    try:
        return int(get()) if callable(get) else None
    except Exception:  # noqa: BLE001 - before macOS 26: no setting to honour
        return None


@pooled
def _file_names(pb: Any) -> list[str]:
    from AppKit import NSURL, NSPasteboardURLReadingFileURLsOnlyKey

    urls = pb.readObjectsForClasses_options_([NSURL], {NSPasteboardURLReadingFileURLsOnlyKey: True}) or []
    return [str(u.lastPathComponent()) for u in urls if u is not None and u.lastPathComponent()]


@pooled
def _image_size(pb: Any) -> tuple[int, int] | None:
    from AppKit import NSImage

    img = NSImage.alloc().initWithPasteboard_(pb)
    if img is None:
        return None
    reps = list(img.representations() or [])
    w = max((int(r.pixelsWide()) for r in reps), default=0)
    h = max((int(r.pixelsHigh()) for r in reps), default=0)
    if not (w and h):
        size = img.size()
        w, h = int(size.width), int(size.height)
    return (w, h) if w and h else None


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:_NAMES_MAX])
    more = len(names) - _NAMES_MAX
    return shown + (f" and {more} more" if more > 0 else "")


@pooled
def read_clipboard() -> str:
    if not _CLIPBOARD.acquire(timeout=_LOCK_WAIT_S):  # (never while a copy has it)
        raise _No("Your clipboard is busy with another look of NEO's just now: ask again in a moment.")
    try:
        return _clipboard_text(_pasteboard())
    finally:
        _CLIPBOARD.release()


def _clipboard_text(pb: Any) -> str:
    if _access(pb) == _always_deny():
        raise _No("This Mac is set to never let NEO read the clipboard.")
    types = {str(t) for t in (pb.types() or [])}  # (the types only: nothing copied is read yet)
    if not types:
        raise _No("The clipboard is empty.")
    if types & PRIVATE_TYPES:
        raise _No("What you copied is marked private (a password manager's, say), so I won't read it.")
    if _FILE_URL in types and (names := _file_names(pb)):
        n = len(names)
        return f"The clipboard holds {n} copied file{'s' if n != 1 else ''}: {_names(names)}."
    if _STRING in types:
        text = str(pb.stringForType_(_STRING) or "")[:_LOOK_CHARS]
        if text.strip():
            if redact.has_secret(text):
                raise _No("What you copied looks like a password, a key or a code, so I won't read it.")
            return "What the user copied:\n" + text
    if types & _IMAGES:
        size = _image_size(pb)
        dims = f", {size[0]}×{size[1]} pixels" if size else ""
        return f"The clipboard holds an image{dims} (its picture isn't read here)."
    raise _No("The clipboard holds something that isn't text, files or a picture.")


# ---- the selection, copied: when Accessibility gives none --------------------------------------------------


@dataclass(frozen=True)
class _Kept:
    """The clipboard as it was: its change count, and each item's types in order with their bytes."""

    count: int
    items: tuple[tuple[tuple[str, Any], ...], ...]


def _types_on(pb: Any) -> set[str]:
    """Every type on the clipboard, every item's (the types only: nothing copied is read)."""
    out = {str(t) for t in (pb.types() or [])}
    for item in pb.pasteboardItems() or []:
        out |= {str(t) for t in (item.types() or [])}
    return out


def _keep(pb: Any) -> _Kept:
    """Every item on the clipboard, every type's bytes — all of it, or _NoCopy before any key is
    pressed: a private copy, or another device's (refused before any of it is read); one over
    CLIPBOARD_CAP_BYTES; one with something that doesn't come (a type whose bytes its app doesn't give
    — a promise it can't keep now, a paste prompt answered "Don't Allow" — or an item with no type: put
    back short, it would be lost); one that takes longer than _KEEP_S to keep; or one that changed while
    it was kept."""
    count = int(pb.changeCount())
    types = _types_on(pb)
    if types & PRIVATE_TYPES:
        raise _NoCopy(_PRIVATE_CLIPBOARD)
    if _REMOTE in types:
        raise _NoCopy(_REMOTE_CLIPBOARD)
    end = time.monotonic() + _KEEP_S
    items: list[tuple[tuple[str, Any], ...]] = []
    size = 0
    for item in pb.pasteboardItems() or []:
        pairs: list[tuple[str, Any]] = []
        for t in [str(t) for t in (item.types() or [])]:
            try:
                data = item.dataForType_(t)
            except Exception:  # noqa: BLE001 - a type its app can't give now: as one that doesn't come
                data = None
            if data is None:
                raise _NoCopy(_UNKEPT_CLIPBOARD)
            size += len(data)
            if size > CLIPBOARD_CAP_BYTES:
                raise _NoCopy(_BIG_CLIPBOARD)
            if time.monotonic() > end:
                raise _NoCopy(_SLOW_CLIPBOARD)
            pairs.append((t, data))
        if not pairs:
            raise _NoCopy(_UNKEPT_CLIPBOARD)
        items.append(tuple(pairs))
    if int(pb.changeCount()) != count:
        raise _NoCopy(_MOVED_CLIPBOARD)
    return _Kept(count, tuple(items))


def _item(pairs: tuple[tuple[str, Any], ...]) -> Any:
    """A kept item, made again: every type's bytes, in their order."""
    from AppKit import NSPasteboardItem

    item = NSPasteboardItem.alloc().init()
    for t, data in pairs:
        item.setData_forType_(data, t)
    return item


def _write(pb: Any, kept: _Kept) -> int:
    """The kept items written over whatever is on the clipboard now (an empty one: left empty) — cleared
    and written once more if the write didn't take. The change count it is left at (clearing moves it,
    writing doesn't). The items are made first: nothing is cleared that can't be written. _No: it
    couldn't be put back (said to the user, never only logged)."""
    try:
        items = [_item(pairs) for pairs in kept.items]
        for _ in range(2):
            count = int(pb.clearContents())
            if not items or bool(pb.writeObjects_(items)):
                return count
        why = "the write didn't take"
    except Exception as e:  # noqa: BLE001 - its kind only: nothing of what was on it reaches the log
        why = type(e).__name__
    log.warning(f"[in_front] the clipboard couldn't be put back: {why}")
    raise _No(_NOT_PUT_BACK)


def _put_back(pb: Any, kept: _Kept) -> int:
    """The clipboard as it was kept (:func:`_write`) — or, when nothing changed it since, left alone: it
    already is what was kept, with anything its app still gives on demand. The change count it is left
    at (what a later write is told apart by)."""
    if int(pb.changeCount()) == kept.count:
        return kept.count
    return _write(pb, kept)


def _press_copy(pid: int) -> None:
    """⌘C for app ``pid`` alone (CGEventPostToPid, through input.py's door: its watcher, never NEO's
    own processes or a protected app), C being the key that types it on the keyboard's layout."""
    import Quartz as Q

    from neo2.core.tools.computer import input as inp

    effects.guard("input", "press ⌘C in another app", via=Q.CGEventPostToPid)
    code, flags = inp.keycode("c"), inp.mods_flags("cmd")
    for down in (True, False):
        ev = Q.CGEventCreateKeyboardEvent(None, code, down)
        Q.CGEventSetFlags(ev, flags)
        inp._post_to_pid(pid, ev)
        if down:
            inp._pause(inp._KEY_HOLD_S)


def _copied(pb: Any, count: int) -> bool:
    """Did the app copy within COPY_WAIT_S? Its change count moved — and what it wrote is there (an
    app clears the clipboard a moment before it writes)."""
    end = time.monotonic() + COPY_WAIT_S
    while True:
        moved = int(pb.changeCount()) != count
        if moved and len(pb.types() or ()):
            return True
        if time.monotonic() >= end:
            return moved
        time.sleep(_COPY_POLL_S)


@pooled
def _watch_late(pb: Any, kept: _Kept, since: int) -> None:
    """Holding the clipboard (_CLIPBOARD, handed over by copy_selection: released here), watch it
    LATE_WATCH_S longer for a write that comes late — a slow app's copy, a page's own second one — and
    put what was kept back over it, _LATE_PUTS times at most. Never over a private copy: that isn't a
    page's (a password manager's, made just then), and it is never touched."""
    try:
        end = time.monotonic() + LATE_WATCH_S
        puts = 0
        while puts < _LATE_PUTS:
            now = time.monotonic()
            if int(pb.changeCount()) != since and (len(pb.types() or ()) or now >= end):
                if _types_on(pb) & PRIVATE_TYPES:
                    return
                since = _write(pb, kept)
                puts += 1
            elif now >= end:
                return
            else:
                time.sleep(_COPY_POLL_S)
    except _No:
        pass  # (_write has said so in the log: the reply is out)
    except Exception as e:  # noqa: BLE001 - its kind only: nothing of what was on it reaches the log
        log.warning(f"[in_front] the clipboard's late watch failed: {type(e).__name__}")
    finally:
        _CLIPBOARD.release()


def _hand_to_late_watch(pb: Any, kept: _Kept, since: int) -> bool:
    """Start the late watch (:func:`_watch_late`) in a thread of its own, the reply not kept waiting,
    handing it the clipboard lock. False: it couldn't start, and the lock is still the caller's."""
    try:
        threading.Thread(
            target=_watch_late, args=(pb, kept, since), name="neo-clipboard-late", daemon=True
        ).start()
    except Exception:  # noqa: BLE001 - no thread to be had: no late watch
        return False
    return True


@pooled
def copy_selection(pid: int) -> str:
    """The selection of app ``pid``, copied with ⌘C and read as plain text ("" = nothing was copied),
    the clipboard put back exactly as it was, whatever happens — and, for LATE_WATCH_S more, put back
    over a write that comes late. _NoCopy: why it wasn't copied (before any key: a private copy on the
    clipboard, another device's, one that can't be kept whole, a Mac that asks or never lets apps read
    it, another look that has it; after: an app not in front that copied nothing); _No: the copy is
    marked private, so it wasn't read — or the clipboard couldn't be put back."""
    if not _CLIPBOARD.acquire(timeout=_LOCK_WAIT_S):
        raise _NoCopy(_BUSY_CLIPBOARD)
    handed = False
    try:
        pb = _pasteboard()
        access = _access(pb)
        if access == _always_deny():
            raise _NoCopy(_DENIED_CLIPBOARD)
        if access == _asks():
            raise _NoCopy(_ASKS_CLIPBOARD)
        kept = _keep(pb)
        ahead = _front()[1] == pid  # (an app behind NEO's window may not copy at all)
        try:
            _press_copy(pid)
            if not _copied(pb, kept.count):
                if not ahead:
                    raise _NoCopy(_BEHIND)
                return ""
            if _types_on(pb) & PRIVATE_TYPES:
                raise _No(
                    "What that copied is marked private (a password manager's, say), so I won't read it."
                )
            return str(pb.stringForType_(_STRING) or "")[:_LOOK_CHARS]
        finally:
            since = _put_back(pb, kept)
            handed = _hand_to_late_watch(pb, kept, since)
    finally:
        if not handed:
            _CLIPBOARD.release()


# ---- Finder's selection ------------------------------------------------------------------------------------


async def finder_selection() -> str:
    from neo2.core.tools.computer import apps

    out = await apps.osascript(FINDER_SCRIPT, timeout=8)
    if out.startswith("Error:"):
        low = out.lower()
        if "not authori" in low or "-1743" in low:
            raise _No("NEO isn't allowed to ask Finder what's selected (Privacy & Security, Automation).")
        raise _No("I couldn't read what's selected in Finder.")
    names = [n.strip() for n in out.splitlines() if n.strip()]
    if not names:
        raise _No("Nothing is selected in Finder.")
    return f"Selected in Finder ({len(names)}): {_names(names)}."


# ---- the tool --------------------------------------------------------------------------------------------------


def _cap(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    note = "\n… (cut here: the rest is longer than NEO reads)"
    return text[: MAX_CHARS - len(note)].rstrip() + note


def _selection_text(app: str, text: str) -> str:
    clean, hidden = redact.redact(text[:_LOOK_CHARS])
    left_out = " (a password, key or code in it was left out)" if hidden else ""
    return f"Selected in {app}{left_out}:\n{clean}"


def _where(asked: str, said: tuple[str, ...]) -> str:
    """Where to look: a window is always the model's to ask for (its name and title only); the
    clipboard only on the user's word for it; any other place the user's words name wins over a
    model's guess."""
    what = asked if asked in _WHATS else "auto"
    if what == "window":
        return what
    if what == "auto":
        return said[0] if said else "auto"
    if what == "clipboard" and "clipboard" not in said:
        raise _No(_CLIPBOARD_ONLY)
    if said and what not in said:
        return said[0]
    return what


@tool(
    "in_front",
    "Read what the user has in front of them, only when they ask about it in their own words: the text they "
    'selected ("translate the selected text", "explain this paragraph"), what they copied ("what I '
    'copied", "my clipboard"), the front window (app, title, document name) or the files selected in '
    "Finder (names). what: auto (their words decide; else the selection, else the window) | selection | "
    "clipboard | window | files. Never on your own initiative, never for something held up to the camera, "
    "and not for a web page (page_read).",
    {
        "type": "object",
        "properties": {"what": {"type": "string", "enum": list(_WHATS)}},
    },
    category="context",
    parallel_safe=True,  # (nothing on screen changes; the clipboard is one call's at a time: _CLIPBOARD)
    read_only=False,  # a selection Accessibility doesn't give is copied: ⌘C pressed in the user's app
    untrusted=True,  # what the user selected or copied is third-party text: data, never instructions
    requires=_MAC,
    timeout=20.0,
)
async def in_front(a: dict, c: ToolContext) -> ToolOutput:
    try:
        why = refusal(c)
        if why:
            raise _No(why)
        what = _where(str(a.get("what") or "auto").strip().lower(), context_words.kinds(users_words(c)))
        if what == "clipboard":
            text = await asyncio.to_thread(read_clipboard)
        elif what == "files":
            text = await finder_selection()
        elif what == "window":
            text = await asyncio.to_thread(read_window)
        else:
            app, sel, why = await asyncio.to_thread(read_selection)
            none = f"I couldn't read what's selected in {app}: {why}." if why else ""
            if sel.strip():
                text = _selection_text(app, sel)
            elif what == "auto":  # nothing selected: "this" is what is in front, then
                text = (none or "Nothing is selected.") + " " + await asyncio.to_thread(read_window)
            else:
                raise _No(none or f"Nothing is selected in {app}.")
    except _No as no:
        # NEO's own words, not content: never fenced, and nothing was read to taint the request.
        return ToolOutput(str(no), ok=False, fence=False)
    return ToolOutput(_cap(text))
