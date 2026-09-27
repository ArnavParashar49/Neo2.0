"""macOS Accessibility (AX) tree — NEO's primary way of *seeing* the screen.

Fast, private, text-only: no screenshots, no vision model. Each interesting element gets
a short id (``e12``) that the model can click / type into / press via the input tools.

Ids are never reused: every walk hands out new ones, and an id the model holds from an earlier
look still means the element it was issued for (never whatever a newer walk numbered the same).

Fast and bounded:

* one round trip per element: an element's role, frame, words, state and children come in one
  ``AXUIElementCopyMultipleAttributeValues`` (per-attribute reads only for the toolkits that answer
  those alone); its action names only when something asks for them;
* only what can be in view is visited: a closed menu is never opened, a background window is one
  line, a long list is entered where it is on screen (its visible rows, or found by halving);
* one Accessibility timeout for the whole process (NEO.app; never in a host's process), a
  wall-clock limit per walk, and a cancel event: a hung app costs 1.5 s, not a minute, and a walk
  nobody waits for any more stops;
* truthful: a partial tree says it is partial and why; a sheet, a modal window or another app's
  window over this one is named first.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ApplicationServices as AS
from AppKit import NSWorkspace

from neo2.core import effects
from neo2.core.decide._objc import pooled
from neo2.core.tools.computer import wait

# ---- the one door for Accessibility actions -------------------------------------------------------
# Every AX action NEO takes in another app — a press, a value or the focus set, a window raised —
# goes through perform_action / set_attribute, as every event goes through input._post (or, for one
# app only, input._post_to_pid). While the controls self-test runs, its watcher (_watch) sees each
# first and raises instead when the action wouldn't stay in its test window; then the target must be
# neither one of NEO's own processes nor a protected app (OffLimits); then the effects switch
# (neo2/core/effects.py) has its say. An app that fails, or an element that can't be asked, is an error
# code: the door itself raises only to refuse, and no caller takes a refusal for a plain failure
# that another way around (typing where a value set failed) could make up for.
_watch: Callable[[str, Any, str, Any], None] | None = None
# What makes an app's web content readable (neo2.core.decide.axkit.enable_web_reading), switched on and
# back on its application element: nothing anyone sees changes, so it is no action in its UI.
_READING_SWITCHES = frozenset({"AXManualAccessibility", "AXEnhancedUserInterface"})
_ILLEGAL = -25201  # kAXErrorIllegalArgument: the answer for an element that can't be asked


class OffLimits(Exception):
    """The door won't act there: one of NEO's own processes, or an app NEO never touches."""


def refuse_off_limits(pid: int) -> None:
    """Raise OffLimits for NEO's own processes and protected apps. The tools and task kinds check
    first, with better answers; this is the last word for any path that didn't."""
    if not pid:
        return
    from neo2.core.tools.computer import gates  # (the frozen gates, looked up at call time)

    if is_neo(pid) or gates._protected("", pid):
        raise OffLimits("that's NEO's own window or an app NEO never touches — nothing was done there")


def _through(kind: str, el: Any, name: str, value: Any) -> None:
    if _watch is not None:
        _watch(kind, el, name, value)
    if not (kind == "set" and name in _READING_SWITCHES):
        refuse_off_limits(pid_of(el))


def perform_action(
    el: Any, action: str, *, kind: str = "accessibility", what: str = "", timeout: float | None = None
) -> int:
    """AXUIElementPerformAction through the one door: the AX error code (0 = done). The element
    answers within _PRESS_TIMEOUT_S (a press may take a moment); the app's kept looks go."""
    _through("perform", el, action, None)
    effects.guard(kind, what or f"press {action} in another app's UI", via=AS.AXUIElementPerformAction)
    _own_timeout(el, _PRESS_TIMEOUT_S if timeout is None else timeout)
    _forget_el(el)
    try:
        return AS.AXUIElementPerformAction(el, action)
    except Exception:  # noqa: BLE001 - an element that can't be asked: a failure, not a refusal
        return _ILLEGAL


def set_attribute(el: Any, name: str, value: Any, *, kind: str = "accessibility", what: str = "") -> int:
    """AXUIElementSetAttributeValue through the one door: the AX error code (0 = done)."""
    _through("set", el, name, value)
    effects.guard(kind, what or f"set {name} in another app's UI", via=AS.AXUIElementSetAttributeValue)
    _forget_el(el)
    try:
        return AS.AXUIElementSetAttributeValue(el, name, value)
    except Exception:  # noqa: BLE001
        return _ILLEGAL


def _forget_el(el: Any) -> None:
    try:
        forget(pid_of(el) or None)  # (an app that can't be told: every kept look goes)
    except Exception:  # noqa: BLE001
        pass


# Roles worth showing to the model. Containers are walked but not listed unless titled.
_INTERACTIVE = {
    "AXButton",
    "AXTextField",
    "AXTextArea",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXMenuButton",
    "AXComboBox",
    "AXLink",
    "AXMenuItem",
    "AXMenuBarItem",
    "AXTab",
    "AXSlider",
    "AXIncrementor",
    "AXDisclosureTriangle",
    "AXCell",
    "AXRow",
    "AXStaticText",
    "AXImage",
    "AXHeading",
    "AXSearchField",
    "AXSecureTextField",
    "AXToolbar",
    "AXTabGroup",
    "AXWebArea",
}
_SKIP_ROLES = {"AXUnknown", "AXSplitter", "AXScrollBar", "AXValueIndicator", "AXGrowArea"}
# Listed even when empty: an untitled, empty text area (a new note, a blank document, a search
# box) is exactly the element the model needs to type into.
_ALWAYS_SHOW = {"AXTextArea", "AXTextField", "AXSearchField", "AXComboBox", "AXWebArea"}
# What the model acts on sorts ahead of what it only reads, when a tree is too big to show whole.
_READ_ONLY_ROLES = {"AXStaticText", "AXImage", "AXHeading"}
# Frames that clip what's inside them: an element wholly outside its window or scroll area is
# scrolled out of view (a web area or a long list reports its whole document's frame).
_CLIPS = {"AXWindow", "AXSheet", "AXDrawer", "AXScrollArea"}
# …except what floats over its window: an open menu or popup list, a popover, a drawer beside it.
_FLOATING = {"AXMenu", "AXPopover", "AXDrawer"}
_FIND_MAX_AGE_S = 30.0  # ax_find reuses the last tree (and its ids) unless it is older than this
_MAX_ELEMENTS = 400
_MAX_NODES = 6000  # elements visited in one walk, listed or not
_MAX_DEPTH = 40
_AX_TIMEOUT_S = 1.5  # a hung app answers each AX call in this long, not the default ~6 s
_PRESS_TIMEOUT_S = 3.0  # …except the element being pressed: a press may rightly take a moment
_CANNOT_COMPLETE = -25204  # kAXErrorCannotComplete: the app didn't answer (busy, hung)
_INVALID = -25202  # kAXErrorInvalidUIElement: the element is gone
_TREE_S = 1.5  # wall-clock limit of one ax_tree walk…
_LOOK_S = 1.0  # …and of a hit, a scene, a window's elements
# A -25204 that took this share of the timeout was a real one (the app isn't answering); a quick
# one is an odd element or attribute — only this many of those in a row end a walk.
_REAL_TIMEOUT = 0.8
_HUNG_AFTER = 3
_LONG_KIDS = 48  # a list longer than this is entered where it is on screen, not walked whole
_CANVAS_SHARE = 0.35  # a childless, unpressable leaf this big is a board or a canvas
_DIALOG_SUBROLES = {"AXDialog", "AXSystemDialog"}
_AX_ERROR_TYPE = 5  # kAXValueAXErrorType: what a batched read puts where a value failed
# An element's text (its value, its title) as kept: every reader shows far less (a line 80 characters,
# a mark 50, the gate's context 2000 in all). A terminal's scrollback or a long file in an editor is
# never copied whole into every look.
_TEXT_MAX = 2000
_SHEET_KIDS = 16  # a sheet is added last among a window's children: only this many, from the last, are asked

# One element's whole look, in one round trip (the role first: the per-attribute fallback asks it
# first, and a hung app is noticed on it).
_WALK_ATTRS = (
    "AXRole",
    "AXSubrole",
    "AXPosition",
    "AXSize",
    "AXTitle",
    "AXDescription",
    "AXPlaceholderValue",
    "AXLabel",
    "AXValue",
    "AXEnabled",
    "AXFocused",
    "AXModal",
    "AXChildren",
)


class _LazyActions(list):
    """An element's action names, asked the first time anything looks at them: most listed
    elements are never pressed, and each ask is one round trip to the app."""

    __slots__ = ("_el", "_done")

    def __init__(self, el: Any) -> None:
        super().__init__()
        self._el = el
        self._done = el is None

    def _load(self) -> None:
        if not self._done:
            self._done = True
            super().extend(_action_names(self._el))

    def __contains__(self, x: object) -> bool:
        self._load()
        return super().__contains__(x)

    def __iter__(self):
        self._load()
        return super().__iter__()

    def __len__(self) -> int:
        self._load()
        return super().__len__()

    def __getitem__(self, i):
        self._load()
        return super().__getitem__(i)

    def __eq__(self, other: object) -> bool:
        self._load()
        return list(self) == other

    def __repr__(self) -> str:
        self._load()
        return super().__repr__()

    __hash__ = None  # type: ignore[assignment]


@dataclass
class Element:
    id: str
    role: str
    title: str
    value: str
    x: float
    y: float
    w: float
    h: float
    enabled: bool = True
    focused: bool = False
    actions: list[str] = field(default_factory=list)
    ref: Any = None  # the AXUIElementRef
    visible: tuple[float, float, float, float] | None = None  # the part on screen (cut to its window)
    note: str = ""  # NEO's own words after the line ("in the background: …")
    subrole: str = ""

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def line(self, coords: bool = True) -> str:
        """The element in one line; ``coords`` False: without where it is (for a brain that can't
        see — it never gets coordinates in any form)."""
        role = self.role.removeprefix("AX")
        bits = [f"{self.id}", role]
        if self.title:
            bits.append(f'"{self.title[:80]}"')
        if self.value and self.value != self.title:
            bits.append(f"= {self.value[:80]!r}")
        if not self.enabled:
            bits.append("(disabled)")
        if self.focused:
            bits.append("(focused)")
        if coords:
            bits.append(f"@({int(self.x)},{int(self.y)} {int(self.w)}x{int(self.h)})")
        if self.note:
            bits.append(self.note)
        return " ".join(bits)

    def short(self) -> str:
        """``e31 Start``: an id and what it says, for one line listing several."""
        words = (self.title or self.value or self.role.removeprefix("AX"))[:40]
        return f"{self.id} {words}"


# The latest walk (what ax_find searches), and every id handed out lately (what an id resolves
# to). A walk that finishes after a newer one started — a slow app's, timed out long ago — adds
# its ids but never replaces the newer tree.
_last_tree: dict[str, Element] = {}
_last_tree_ts = 0.0
_last_tree_app = ""
_last_tree_pid = 0
_issued: dict[str, Element] = {}
_ISSUED_MAX = 4000
_ISSUED_CHARS = 2_000_000  # …and at most this much of their text (the oldest ids go first)
_ids = itertools.count(1)
_walks = itertools.count(1)
_latest_walk = 0
_lock = threading.Lock()


def _new_id() -> str:
    return f"e{next(_ids)}"


def is_trusted() -> bool:
    return bool(AS.AXIsProcessTrusted())


# ---- the process-wide timeout ------------------------------------------------------------------------
# macOS takes a messaging timeout set on the system-wide element as this process's default for every
# element with no timeout of its own: children, parents, hit results — the per-app 2 s that came
# before left all of those at the ~6 s default. Set once, on first use, in NEO.app only: embedded in
# another app (library mode) the process is the host's, and so is its timeout.
_timeout_for: Any = None  # the ApplicationServices the default was set through (tests swap it)


def _library() -> bool:
    """Embedded in another app's process (or an embedded agent's request is running): nothing
    process-wide is changed here (runtime.process_wide_ok)."""
    try:
        from neo2.core import runtime

        return not runtime.process_wide_ok()
    except Exception:  # noqa: BLE001 - can't tell: act as a guest, never change process-wide state
        return True


def _ensure_timeout() -> bool:
    """The process-wide timeout is set (True), or may not be here (a host's process)."""
    global _timeout_for
    if _timeout_for is AS:
        return True
    if _library():
        return False
    try:
        AS.AXUIElementSetMessagingTimeout(AS.AXUIElementCreateSystemWide(), _AX_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        return False
    _timeout_for = AS
    return True


def _own_timeout(el: Any, seconds: float = _AX_TIMEOUT_S) -> None:
    try:
        AS.AXUIElementSetMessagingTimeout(el, seconds)
    except Exception:  # noqa: BLE001
        pass


def _system_wide():
    """The system-wide AX element (whatever has the keyboard focus, whatever is under a point),
    answering within _AX_TIMEOUT_S like every element NEO asks (the timeout set on it is the
    process's default: NEO.app only)."""
    global _timeout_for
    el = AS.AXUIElementCreateSystemWide()
    if not _library():
        _own_timeout(el)
        _timeout_for = AS
    return el


def _app_el(pid: int):
    """The app's AX element, answering each call within _AX_TIMEOUT_S even when the app hangs (the
    process-wide default; in a host's process, a timeout of its own)."""
    el = AS.AXUIElementCreateApplication(pid)
    if not _ensure_timeout():
        _own_timeout(el)
    return el


def _bounded(el):
    """The element, answering each AX call within _AX_TIMEOUT_S even when its app hangs (a timeout
    of its own: in a host's process there is no process-wide one of NEO's)."""
    _own_timeout(el)
    return el


def _focused():
    """The element with the keyboard focus (None), answering within _AX_TIMEOUT_S."""
    el = _attr(_system_wide(), AS.kAXFocusedUIElementAttribute)
    return None if el is None else _bounded(el)


# ---- reading ------------------------------------------------------------------------------------------


def _attr(el, name: str, default=None):
    try:
        err, val = AS.AXUIElementCopyAttributeValue(el, name, None)
    except Exception:  # noqa: BLE001 - an element that can't be asked has nothing to say
        return default
    return val if err == 0 else default


def _ax_error(v: Any) -> bool:
    """A batched read's stand-in for a value that failed (an AXValue of the error type)."""
    if getattr(v, "__ax_error__", False):
        return True
    if type(v).__name__ != "AXValueRef":
        return False
    try:
        return AS.AXValueGetType(v) == _AX_ERROR_TYPE
    except Exception:  # noqa: BLE001
        return False


def _attrs_err(el, names: tuple[str, ...] | list[str]) -> tuple[dict[str, Any], int]:
    """({name: value, or None when it has none}, 0) in one round trip — or the error that ended
    the asking (-25204: the app didn't answer; -25202: the element is gone). Per-attribute reads
    when the batch isn't there or answers nothing but errors (some Java and Qt builds)."""
    fn = getattr(AS, "AXUIElementCopyMultipleAttributeValues", None)
    if fn is not None:
        try:
            err, vals = fn(el, list(names), 0, None)
        except Exception:  # noqa: BLE001
            err, vals = -1, None
        if err in (_CANNOT_COMPLETE, _INVALID):
            return dict.fromkeys(names), err
        if err == 0 and vals is not None:
            out: dict[str, Any] = {}
            good = 0
            for n, v in zip(names, vals, strict=False):
                if v is None or _ax_error(v):
                    out[n] = None
                else:
                    out[n] = v
                    good += 1
            if good:
                return out, 0
    out = dict.fromkeys(names)
    refused = 0
    for n in names:
        t0 = wait.now()
        try:
            err, v = AS.AXUIElementCopyAttributeValue(el, n, None)
        except Exception:  # noqa: BLE001
            continue
        if err == 0:
            out[n] = v
        elif err == _INVALID:
            return out, err
        elif err == _CANNOT_COMPLETE:
            refused += 1
            if wait.now() - t0 >= _REAL_TIMEOUT * _AX_TIMEOUT_S:
                return out, err  # it really isn't answering: don't ask it the rest one by one
    return out, (_CANNOT_COMPLETE if refused == len(names) else 0)


def _attrs(el, names) -> dict[str, Any]:
    """``names`` of ``el`` in one round trip: {name: value, or None}."""
    return _attrs_err(el, names)[0]


class _Walk:
    """One look's limits (a wall-clock deadline, a cancel event, the node budget), and why it ended
    early if it did: "deadline", "cancelled", "hung" (the app stopped answering), "budget",
    "found" (an early exit)."""

    def __init__(self, deadline_s: float | None = None, cancel: threading.Event | None = None) -> None:
        self.deadline = wait.now() + deadline_s if deadline_s else None
        self.limit_s = deadline_s or 0.0
        self.cancel = cancel
        self.stopped = ""
        self.quick = 0  # quick -25204s in a row
        self.timed_out = False  # a call waited out the timeout: the app isn't answering (asking more costs as much)
        self.blockers: list[_Within] = []
        self.canvas: list[tuple[Any, str, tuple[float, float, float, float], Element | None]] = []
        self.canvas_min = 0.0  # a childless leaf at least this big (35% of the window) is a candidate
        self.eager = False  # read action names while walking (the gate and click_text look at them)
        self.early: Callable[[Element], bool] | None = None

    def over(self) -> bool:
        if self.stopped:
            return True
        if self.cancel is not None and self.cancel.is_set():
            self.stopped = "cancelled"
        elif self.deadline is not None and wait.now() >= self.deadline:
            self.stopped = "deadline"
        return bool(self.stopped)

    def refused(self, took: float) -> None:
        if took >= _REAL_TIMEOUT * _AX_TIMEOUT_S:
            self.stopped = self.stopped or "hung"
            self.timed_out = True
            return
        self.quick += 1
        if self.quick >= _HUNG_AFTER:
            self.stopped = self.stopped or "hung"

    def said(self) -> str:
        """Why the look is partial, in NEO's words ('' when it is whole)."""
        if self.stopped == "deadline":
            return f"(stopped after {self.limit_s:g} s: the rest of this window wasn't read — ax_find, or a look at one window, can help)"
        if self.stopped == "hung":
            return "(the app isn't answering: this is what it showed before it stopped — try again in a moment)"
        if self.stopped == "cancelled":
            return "(stopped: nobody was waiting for this look any more)"
        return ""


def _read(el, names, w: _Walk | None = None) -> dict[str, Any] | None:
    """``names`` of ``el`` in one round trip, or None when it didn't answer (or is gone). A real
    timeout — or a few quick refusals in a row — ends the walk ``w``."""
    t0 = wait.now()
    vals, err = _attrs_err(el, names)
    if err:
        if err == _CANNOT_COMPLETE and w is not None:
            w.refused(wait.now() - t0)
        return None
    if w is not None:
        w.quick = 0
    return vals


# ---- long texts -----------------------------------------------------------------------------------------
# An element found holding a long text once (a terminal's scrollback, a big file in an editor) is
# asked from then on for everything but its value, and for the value's first _TEXT_MAX characters on
# their own (AXStringForRange): its whole text isn't copied across on every look.
_BIG_TEXT = 1 << 15  # characters: a value longer than this is read in part from then on
_BIG_KEEP = 64
_big: dict[int, Any] = {}  # hash → the element itself (CFEqual tells it apart from another with that hash)
_big_lock = threading.Lock()


def _clip_text(s: str) -> str:
    return s if len(s) <= _TEXT_MAX else s[:_TEXT_MAX]


def _big_key(el: Any) -> int | None:
    try:
        return hash(el)
    except TypeError:
        return None


def _known_big(el: Any) -> bool:
    k = _big_key(el)
    if k is None:
        return False
    with _big_lock:
        ref = _big.get(k)
    return ref is not None and _same(ref, el)


def _mark_big(el: Any, big: bool) -> None:
    k = _big_key(el)
    if k is None:
        return
    with _big_lock:
        _big.pop(k, None)
        if big:
            _big[k] = el
            while len(_big) > _BIG_KEEP:
                _big.pop(next(iter(_big)))


def _head(el: Any, count: Any, w: _Walk | None) -> str:
    """The first _TEXT_MAX characters of a long text, in a read that can't be longer. '' when it
    can't be read in part (a text this long shows only its first words anyway)."""
    fn = getattr(AS, "AXUIElementCopyParameterizedAttributeValue", None)
    make, kind = getattr(AS, "AXValueCreate", None), getattr(AS, "kAXValueCFRangeType", None)
    try:
        n = max(0, min(int(count), _TEXT_MAX)) if count is not None else _TEXT_MAX
    except (TypeError, ValueError):
        n = _TEXT_MAX
    if fn is None or make is None or kind is None or n == 0 or (w is not None and w.over()):
        return ""
    span = AS.CFRangeMake(0, n) if hasattr(AS, "CFRangeMake") else (0, n)
    t0 = wait.now()
    try:
        err, got = fn(el, "AXStringForRange", make(kind, span), None)
    except Exception:  # noqa: BLE001 - not readable in part
        err, got = -1, None
    if err == _CANNOT_COMPLETE and w is not None:
        w.refused(wait.now() - t0)
    return str(got[:_TEXT_MAX]) if not err and isinstance(got, str) else ""


def _read_text(el: Any, names: tuple[str, ...], w: _Walk | None = None) -> dict[str, Any] | None:
    """:func:`_read`, with a long AXValue kept to its first _TEXT_MAX characters — and, for an element
    known to hold a long text, never read whole (its head alone, in part)."""
    if "AXValue" not in names:
        return _read(el, names, w)
    if _known_big(el):
        v = _read(el, (*(n for n in names if n != "AXValue"), "AXNumberOfCharacters"), w)
        if v is None:
            return None
        count = v.pop("AXNumberOfCharacters", None)
        v["AXValue"] = _head(el, count, w)
        try:
            if count is not None and int(count) <= _BIG_TEXT:
                _mark_big(el, False)  # (it got short: read whole from now on)
        except (TypeError, ValueError):
            pass
        return v
    v = _read(el, names, w)
    raw = v.get("AXValue") if v is not None else None
    if isinstance(raw, str) and len(raw) > _TEXT_MAX:
        if len(raw) > _BIG_TEXT:
            _mark_big(el, True)
        v["AXValue"] = str(raw[:_TEXT_MAX])  # type: ignore[index]
    return v


def _seq(v: Any) -> list:
    """An AX array as a list ([] for nothing, or for something that isn't one)."""
    if v is None or isinstance(v, (str, bytes)):
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _action_names(el: Any) -> list[str]:
    try:
        err, actions = AS.AXUIElementCopyActionNames(el, None)
    except Exception:  # noqa: BLE001
        return []
    return [str(a) for a in (actions or [])] if err == 0 else []


def _pair(v, point: bool) -> tuple[float, float] | None:
    if v is None:
        return None
    try:
        ok, p = AS.AXValueGetValue(v, AS.kAXValueCGPointType if point else AS.kAXValueCGSizeType, None)
    except Exception:  # noqa: BLE001 - not a geometry value (some Java/Qt builds)
        return None
    if not ok:
        return None
    try:
        return (float(p.x), float(p.y)) if point else (float(p.width), float(p.height))
    except (AttributeError, TypeError, ValueError):
        return None


def _frame_of(v: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not v:
        return (0.0, 0.0, 0.0, 0.0)
    pos = _pair(v.get("AXPosition"), True) or (0.0, 0.0)
    size = _pair(v.get("AXSize"), False) or (0.0, 0.0)
    return pos[0], pos[1], size[0], size[1]


def _point_size(el) -> tuple[float, float, float, float]:
    return _frame_of(_attrs(el, ("AXPosition", "AXSize")))


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return ""


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    if a is b:
        return True
    try:
        from CoreFoundation import CFEqual

        return bool(CFEqual(a, b))
    except Exception:  # noqa: BLE001
        return a == b


def _label_words(v: dict[str, Any]) -> str:
    return (
        _str(v.get("AXTitle"))
        or _str(v.get("AXDescription"))
        or _str(v.get("AXPlaceholderValue"))
        or _str(v.get("AXLabel"))
    )


# ---- NEO's own UI --------------------------------------------------------------------------------
# NEO's overlay floats over everything, takes the first click, and hides from screen capture:
# the model never sees it in a screenshot, so a click aimed at what's under it would land on NEO.
# No input tool ever acts on it.
_OWN_NAMES = frozenset({"neo", "neo-ui"})  # the overlay, by the names it runs under
# While the controls self-test runs, its test window's pid (neo2.core.tools.computer.allow_test_window):
# NEO started it, so macOS holds NEO responsible for it — but it is the one app the test acts in.
_test_window_pid = 0


def is_neo(pid: int, name: str = "") -> bool:
    """One of NEO's own processes (the app that launched this core, its overlay), by pid — or its
    overlay by name (a development run starts it apart). Never the self-test's own test window."""
    if pid and pid == _test_window_pid:
        return False
    if " ".join(str(name or "").split()).casefold() in _OWN_NAMES:
        return True
    if not pid or pid <= 0:
        return False
    try:
        from neo2.core.decide import media

        family = media._self_family()
        return pid in family or media.responsible_pid(pid) in family
    except Exception:  # noqa: BLE001 - can't tell: not NEO's
        return False


@pooled
def frontmost_app() -> tuple[str, int]:
    """The app the user is working in. Something frontmost for a moment that isn't a regular app
    (a notification, Spotlight, the Control Center — told by how it runs, not by its name) is
    skipped for the menu-bar app, unless it has a focused window the user is in (a system dialog)."""
    from neo2.core.tools.computer.focus import front_app

    name, pid = front_app()  # always current, unlike NSWorkspace.frontmostApplication
    if not pid or not regular(pid):
        has_window = bool(pid) and _attr(_app_el(pid), AS.kAXFocusedWindowAttribute) is not None
        if not has_window:
            app = NSWorkspace.sharedWorkspace().menuBarOwningApplication()
            if app is not None:
                return app.localizedName() or "", int(app.processIdentifier())
    return name, pid


@pooled
def working_app() -> tuple[str, int]:
    """The app to look at when none is named: the frontmost one — or, while NEO's own window is
    in front (the user just talked to it), the app whose window is frontmost behind it."""
    name, pid = frontmost_app()
    if not is_neo(pid, name):
        return name, pid
    try:
        import Quartz as Q

        opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
        for w in Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or []:  # front to back
            p, n = int(w.get("kCGWindowOwnerPID", 0)), str(w.get("kCGWindowOwnerName", ""))
            if int(w.get("kCGWindowLayer", 1)) == 0 and p and not is_neo(p, n):
                return n, p
    except Exception:  # noqa: BLE001
        pass
    return name, pid


def running_apps() -> list[str]:
    """Normal windowed apps, asked fresh from LaunchServices."""
    from neo2.core.tools.computer.focus import running

    names = sorted({n for n, _, t in running() if t == "Foreground" and n})
    if names:
        return names
    out: list[str] = []
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if a.activationPolicy() == 0 and a.localizedName():  # regular (has a UI)
            out.append(a.localizedName())
    return sorted(set(out))


# name → (pid, when): an app looked up a moment ago is the same app, while it still runs.
_PID_TTL_S = 5.0
_pid_cache: dict[str, tuple[int, float]] = {}


def _still(pid: int, name_l: str) -> bool:
    """The app with this pid still runs, under the name it was found by."""
    try:
        from AppKit import NSRunningApplication

        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
        if a is None or a.isTerminated():
            return False
        return str(a.localizedName() or "").lower() == name_l
    except Exception:  # noqa: BLE001
        return False


@pooled
def pid_for_app(name: str) -> int | None:
    """The app's pid. An exact name wins over a substring match, and a real app (menu bar,
    windows) wins over a helper process that shares the name — "Notes" must not resolve to
    "LinkedNotesUIService", whose AX tree is empty. Asked fresh (apps launched after NEO started
    are found too), and kept for a few seconds while that app still runs — an app called exactly
    that only: a name found by part is asked afresh, so the app called that is found once it runs."""
    from AppKit import NSRunningApplication

    from neo2.core.tools.computer.focus import pid_running

    name_l = name.lower()
    kept = _pid_cache.get(name_l)
    if kept is not None and time.monotonic() - kept[1] < _PID_TTL_S and _still(kept[0], name_l):
        return kept[0]
    best: tuple[int, int] | None = None  # (rank, pid) — lower rank is better
    fresh = [NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) for _, pid in pid_running(name_l)]
    for a in [x for x in fresh if x is not None] or NSWorkspace.sharedWorkspace().runningApplications():
        n = (a.localizedName() or "").lower()
        if not n or (n != name_l and name_l not in n):
            continue
        rank = (0 if n == name_l else 2) + (0 if a.activationPolicy() == 0 else 1)
        if best is None or rank < best[0]:
            best = (rank, int(a.processIdentifier()))
    if best is None:
        _pid_cache.pop(name_l, None)
        return None
    if best[0] == 0:  # the app called exactly that, a regular one
        _pid_cache[name_l] = (best[1], time.monotonic())
    else:
        _pid_cache.pop(name_l, None)
    return best[1]


def _clip(box, frame) -> tuple[float, float, float, float] | None:
    """``frame`` cut to ``box`` (None: no box to cut to)."""
    if box is None:
        return frame
    left, top = max(box[0], frame[0]), max(box[1], frame[1])
    right, bottom = min(box[0] + box[2], frame[0] + frame[2]), min(box[1] + box[3], frame[1] + frame[3])
    return (left, top, max(0.0, right - left), max(0.0, bottom - top))


def _outside(frame, box) -> bool:
    """A real frame (not a zero-sized container) wholly outside the visible box."""
    if box is None or frame[2] <= 0 or frame[3] <= 0:
        return False
    return (
        frame[0] >= box[0] + box[2]
        or frame[1] >= box[1] + box[3]
        or frame[0] + frame[2] <= box[0]
        or frame[1] + frame[3] <= box[1]
    )


@dataclass
class _Within:
    """A sheet or modal dialog met during a walk, and the elements listed inside it."""

    kind: str
    ref: Any
    title: str
    frame: tuple[float, float, float, float]
    els: list[Element] = field(default_factory=list)


def _walk(
    el,
    depth: int,
    out: list[Element],
    new_id,
    *,
    box=None,
    budget: list[int] | None = None,
    skip=None,
    clip: bool = True,
    w: _Walk | None = None,
    within: _Within | None = None,
) -> None:
    """List the interesting elements under ``el``, depth first. ``box``: the visible part of the
    screen here (the window, the scroll area); whatever lies wholly outside it is scrolled out of
    view and skipped with everything under it (``clip=False``: nothing is). One odd element never
    fails the walk; ``w`` (its deadline, cancel event, a hung app) may end it early."""
    if depth > _MAX_DEPTH or len(out) >= _MAX_ELEMENTS:
        return
    if w is not None and w.over():
        return
    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            if w is not None and not w.stopped:
                w.stopped = "budget"
            return
    if skip is not None and _same(el, skip):
        return
    try:
        v = _read_text(el, _WALK_ATTRS, w)
        if v is None:
            return
        role = _str(v.get("AXRole"))
        if role in _SKIP_ROLES:
            return
        if role in _FLOATING or not clip:
            box = None
        x, y, fw, fh = frame = _frame_of(v)
        if _outside(frame, box):
            return
        if w is not None and w.canvas_min < 0 and role == "AXWindow":
            w.canvas_min = _CANVAS_SHARE * fw * fh if fw > 0 and fh > 0 else 0.0
        seen = _clip(box, frame) if fw > 0 and fh > 0 else None
        if clip and role in _CLIPS and fw > 0 and fh > 0:
            box = _clip(box, frame)
        title = _clip_text(_label_words(v))
        value = _clip_text(_str(v.get("AXValue")))
        subrole = _str(v.get("AXSubrole"))
        shown = fw > 0 and fh > 0
        interesting = role in _INTERACTIVE and (title or value or role in _ALWAYS_SHOW) and shown
        if role in ("AXWindow", "AXSheet", "AXDialog"):
            interesting = True
        elif role == "AXMenu":
            interesting = shown  # an open menu; a closed one is noise
        kids = _seq(v.get("AXChildren"))
        if role == "AXMenu" and not shown:
            kids = []  # a closed menu is never walked: its items can't be pressed from here
        if w is not None and shown:
            if role == "AXSheet":
                within = _Within("sheet", el, title, frame)
                w.blockers.append(within)
            elif role == "AXWindow" and depth <= 1 and (v.get("AXModal") is True or subrole in _DIALOG_SUBROLES):
                kind = "modal" if v.get("AXModal") is True or subrole == "AXSystemDialog" else "dialog"
                within = _Within(kind, el, title, frame)
                w.blockers.append(within)
        e = None
        if interesting:
            en, fo = v.get("AXEnabled"), v.get("AXFocused")
            eager = w is not None and w.eager and role not in _READ_ONLY_ROLES
            e = Element(
                id=new_id(),
                role=role,
                title=title,
                value=value,
                x=x,
                y=y,
                w=fw,
                h=fh,
                enabled=True if en is None else bool(en),
                focused=False if fo is None else bool(fo),
                actions=_action_names(el) if eager else _LazyActions(el),
                ref=el,
                visible=seen,
                subrole=subrole,
            )
            out.append(e)
            if within is not None and e.ref is not within.ref:
                within.els.append(e)
            if w is not None and w.early is not None and w.early(e):
                w.stopped = "found"
                return
        big = w is not None and w.canvas_min > 0 and seen is not None and seen[2] * seen[3] >= w.canvas_min
        if big and not kids and role not in _EDITABLE and role not in ("AXWebArea", "AXScrollArea", "AXWindow", "AXSheet"):
            w.canvas.append((el, role, seen, e))  # type: ignore[union-attr]
        kids = _in_view(el, role, kids, box, w)
    except Exception:  # noqa: BLE001 - skip this one element, keep the tree
        return
    for c in kids:
        if w is not None and w.stopped:
            return
        _walk(c, depth + 1, out, new_id, box=box, budget=budget, skip=skip, clip=clip, w=w, within=within)


def _in_view(el, role: str, kids: list, box, w: _Walk | None) -> list:
    """The children of a long list that can be on screen: its visible rows when it says which,
    else found by halving (a list runs top to bottom). Short lists, and lists whose order can't be
    trusted (frames missing, not top to bottom), are walked whole."""
    n = len(kids)
    if box is None or n <= _LONG_KIDS:
        return kids
    if role in ("AXTable", "AXOutline", "AXList", "AXGrid"):
        name = "AXVisibleRows" if role in ("AXTable", "AXOutline") else "AXVisibleChildren"
        got = _read(el, (name, "AXHeader"), w)
        rows = _seq(got.get(name)) if got else []
        if rows:
            head = got.get("AXHeader") if got else None
            return ([head] if head is not None else []) + rows
    frames: dict[int, tuple[float, float, float, float] | None] = {}

    def at(i: int) -> tuple[float, float, float, float] | None:
        if i not in frames:
            v = _read(kids[i], ("AXPosition", "AXSize"), w)
            f = _frame_of(v) if v else None
            frames[i] = f if f is not None and f[2] > 0 and f[3] > 0 else None
        return frames[i]

    sample = sorted({round(k * (n - 1) / 7) for k in range(8)})
    fs = [at(i) for i in sample]
    if any(f is None for f in fs) or any(a[1] > b[1] for a, b in zip(fs, fs[1:], strict=False)):  # type: ignore[index]
        return kids
    top, bottom = box[1], box[1] + box[3]
    lo, hi = 0, n  # the first child whose bottom is below the top of the view
    while lo < hi:
        mid = (lo + hi) // 2
        f = at(mid)
        if f is None:
            return kids
        if f[1] + f[3] <= top:
            lo = mid + 1
        else:
            hi = mid
    first = lo
    lo, hi = first, n  # the first child that starts below the bottom of the view
    while lo < hi:
        mid = (lo + hi) // 2
        f = at(mid)
        if f is None:
            return kids
        if f[1] < bottom:
            lo = mid + 1
        else:
            hi = mid
    return kids[max(0, first - 2) : min(n, lo + 2)]  # (a little slack on both sides)


# Chromium and Electron apps show their web content (a page, Slack's messages, VS Code's editor)
# to Accessibility only once asked. NEO asks when it looks at one, and keeps the switch on while the
# ids it handed out may still be used — then puts it back (neo2.core.decide.axkit counts every flow).
_WEB_HOLD_S = 120.0
_WEB_SETTLE_S = 0.4  # at most this long for a web tree just switched on to fill in
_web_lock = threading.Lock()
_web_held: dict[int, tuple[Any, list, threading.Timer]] = {}


@pooled
def _release_web(pid: int) -> None:
    with _web_lock:
        held = _web_held.pop(pid, None)
    if held is not None:
        held[2].cancel()
        from neo2.core.decide import axkit

        try:
            axkit.restore_web_tree(held[0], held[1])
        except Exception:  # noqa: BLE001
            pass


def _expose_web(pid: int, root) -> bool:
    """Switch the app's web tree on if it is a Chromium or Electron app (by what its bundle holds,
    not its name). True when it was just switched on (it needs a moment to fill in). Only as
    NEO.app: embedded in another app's process, NEO switches no app's accessibility mode and
    leaves no timer thread behind (runtime.process_wide_ok)."""
    from neo2.core import runtime

    if not runtime.process_wide_ok():
        return False
    with _web_lock:
        held = _web_held.get(pid)
        if held is not None:
            held[2].cancel()
            timer = threading.Timer(_WEB_HOLD_S, _release_web, (pid,))
            timer.daemon = True
            timer.start()
            _web_held[pid] = (held[0], held[1], timer)
            return False
    app = _running(pid)
    try:
        path = str(app.bundleURL().path()) if app is not None and app.bundleURL() is not None else ""
    except Exception:  # noqa: BLE001
        path = ""
    if not path:
        return False
    from neo2.core.decide import axkit

    rt = axkit.runtime(path)
    if rt == "native":
        return False
    changed = axkit.enable_web_reading(root, rt)
    timer = threading.Timer(_WEB_HOLD_S, _release_web, (pid,))
    timer.daemon = True
    with _web_lock:
        _web_held[pid] = (root, changed, timer)
    timer.start()
    return bool(changed)


def _child_count(el) -> int:
    try:
        err, n = AS.AXUIElementGetAttributeValueCount(el, "AXChildren", None)
        if err == 0:
            return int(n or 0)
    except Exception:  # noqa: BLE001 - not there (an old fake, an odd toolkit): count them instead
        pass
    return len(_seq(_attr(el, AS.kAXChildrenAttribute)))


def _find_role(start, role: str, max_nodes: int = 60, max_depth: int = 8, w: _Walk | None = None):
    """The first ``role`` under ``start``, breadth first, within a few dozen elements — and within
    ``w``'s limits (its deadline; an app that stopped answering ends the search)."""
    todo, seen = [(start, 0)], 0
    while todo and seen < max_nodes:
        if w is not None and w.over():
            return None
        node, d = todo.pop(0)
        seen += 1
        v = _read(node, ("AXRole", "AXChildren"), w)
        if v is None:
            continue
        if _str(v.get("AXRole")) == role:
            return node
        if d < max_depth:
            todo.extend((c, d + 1) for c in _seq(v.get("AXChildren")))
    return None


def _web_settle(start, cancel: threading.Event | None = None) -> None:
    """After a Chromium or Electron app's web tree was switched on: wait until its web area has
    something in it — at most _WEB_SETTLE_S (one AX call already asked may run past it), instead of
    always that long; an app that stops answering ends the wait."""
    if start is None:
        return
    area: list[Any] = [None, 0.0]
    w = _Walk(_WEB_SETTLE_S, cancel)

    def ready() -> bool:
        if w.over():
            return True  # (out of time, or the app isn't answering: go on without waiting)
        if area[0] is None and wait.now() >= area[1]:
            area[0] = _find_role(start, "AXWebArea", w=w)
            area[1] = wait.now() + 0.1  # (not found: looked for again in a moment, not every poll)
        if area[0] is None or w.over():
            return w.over()
        t0 = wait.now()
        if _child_count(area[0]) > 0:
            return True
        if wait.now() - t0 >= _REAL_TIMEOUT * _AX_TIMEOUT_S:
            w.refused(wait.now() - t0)  # (the count waited out the timeout: the app isn't answering)
        return w.over()

    wait.until_sync(ready, _WEB_SETTLE_S, every=0.03, stop=cancel)


def _lines_budget() -> int:
    """Characters of tree that reach the model whole (the agent loop cuts longer results)."""
    try:
        from neo2.core.config import settings

        return max(1500, int(settings().max_tool_result_chars) - 400)
    except Exception:  # noqa: BLE001
        return 5600


# ---- a look at the app ---------------------------------------------------------------------------------

AX_TREE_DESCRIPTION = (
    "Read the frontmost (or named) app's UI as a text tree of elements with ids, roles, titles, "
    "values and screen positions. Use this FIRST to see what's on screen — it's fast and exact. "
    "Element ids (e12) can be used with ax_press, ax_set_value, and click. A sheet or dialog that "
    "must be answered first is named at the top. When it says an area has no accessible parts (a "
    "board, a map, a canvas), take a screenshot and click at pixel positions in it (with its shot "
    "id) instead. Background windows are one line each: pass window (its title) to look inside one."
)
AX_TREE_PARAMS = {
    "type": "object",
    "properties": {
        "app": {"type": "string", "description": "App name; omit for the frontmost app"},
        "window": {"type": "string", "description": "A background window's title (or its id) to look inside"},
    },
}


def _issue(els: list[Element]) -> None:
    """These elements' ids resolve to them from now on (a look that isn't the tree ax_find reads).
    The oldest ids are let go past _ISSUED_MAX of them, or past _ISSUED_CHARS of their text."""
    with _lock:
        for e in els:
            _issued.pop(e.id, None)
            _issued[e.id] = e
        _trim_issued()


def _text_size(e: Any) -> int:
    return len(getattr(e, "title", "") or "") + len(getattr(e, "value", "") or "")


def _trim_issued() -> None:
    """(Under _lock.) The registry back within its count and its size, the oldest ids first."""
    while len(_issued) > _ISSUED_MAX:
        _issued.pop(next(iter(_issued)))
    total = sum(_text_size(e) for e in _issued.values())
    while _issued and total > _ISSUED_CHARS:
        total -= _text_size(_issued.pop(next(iter(_issued))))


def _register(els: list[Element], walk: int, name: str, pid: int) -> None:
    global _last_tree, _last_tree_ts, _last_tree_app, _last_tree_pid, _latest_walk
    _issue(els)
    with _lock:
        if walk > _latest_walk:
            _latest_walk = walk
            _last_tree = {e.id: e for e in els}
            _last_tree_ts = time.time()
            _last_tree_app = name.lower()
            _last_tree_pid = int(pid or 0)


def _menu_bar(bar, new_id, w: _Walk, open_menus: list[Element]) -> list[Element]:
    """The menu bar's items, for one compact line (never the system menu: the bar's first item),
    and — into ``open_menus`` — whatever is in a menu that is open right now."""
    v = _read(bar, ("AXChildren",), w)
    items = _seq(v.get("AXChildren")) if v else []
    out: list[Element] = []
    for i, it in enumerate(items):
        if i == 0 or w.over():
            continue  # (the system menu, however many items the bar has)
        iv = _read(it, ("AXRole", "AXTitle", "AXPosition", "AXSize", "AXEnabled", "AXSelected", "AXChildren"), w)
        if iv is None or _str(iv.get("AXRole")) != "AXMenuBarItem":
            continue
        x, y, fw, fh = _frame_of(iv)
        en = iv.get("AXEnabled")
        e = Element(new_id(), "AXMenuBarItem", _str(iv.get("AXTitle")), "", x, y, fw, fh,
                    enabled=True if en is None else bool(en), actions=_LazyActions(it), ref=it)  # fmt: skip
        out.append(e)
        menus = _seq(iv.get("AXChildren"))
        if iv.get("AXSelected") is False or not menus:
            continue
        for m in menus:  # an open menu (its bar item selected, or a size of its own): walked whole
            _walk(m, 1, open_menus, new_id, w=w)
    return out


def _canvas_line(w: _Walk, blind: bool = False) -> str:
    """'This area has no accessible parts…' — when one childless leaf with no press action covers
    much of the window (a board, a map, a canvas drawn by the app itself). ``blind``: for a brain
    that can't see — no coordinates, and no screenshot it couldn't look at."""
    if not w.canvas:
        return ""
    el, role, seen, e = max(w.canvas, key=lambda c: c[2][2] * c[2][3])
    if "AXPress" in _action_names(el):
        return ""
    x, y, fw, fh = seen
    what = f"{e.id} " if e is not None else ""
    if blind:
        return (
            f"This area has no accessible parts (a board or canvas): {what}{role.removeprefix('AX')} — "
            "working inside it needs a model that can see the screen."
        )
    return (
        f"This area has no accessible parts (a board or canvas): {what}{role.removeprefix('AX')} "
        f"@({int(x)},{int(y)} {int(fw)}x{int(fh)}) — take a screenshot of it to see and click inside it."
    )


def _blocker_lines(found: list[_Within], covering: list[Blocker]) -> list[str]:
    lines = []
    for b in sorted(found, key=lambda b: b.kind != "sheet"):
        words = b.title or next((e.value or e.title for e in b.els if e.role == "AXStaticText" and (e.value or e.title)), "")
        ctl = [e for e in b.els if e.role not in _READ_ONLY_ROLES and e.role not in ("AXGroup", "AXScrollArea")]
        offers = " · ".join(e.short() for e in ctl[:8])
        what = {"sheet": "A sheet", "modal": "A dialog", "dialog": "A dialog window"}.get(b.kind, "A dialog")
        said = f"“{words[:80]}”" if words else "(no title)"
        tail = " — answer it first." if b.kind in ("sheet", "modal") else "."
        lines.append(f"{what} {said} is open{': ' + offers if offers else ''}{tail}")
    for c in covering:
        lines.append(f"Another app's window covers this one ({c.title[:60] or 'untitled'}): clicks there would reach it, not this app.")
    return lines


@pooled
def snapshot(
    app: str | None = None,
    *,
    max_lines: int = 250,
    compact: bool = False,
    deadline_s: float = _TREE_S,
    cancel: threading.Event | None = None,
    window: str | None = None,
    blind: bool = False,
) -> str:
    """Return a compact text tree of the frontmost (or named) app's UI: a sheet or dialog to
    answer first, then its focused window (or ``window``, one of its others by title or id), what's
    in view only, controls ahead of plain text when it's too big to show whole; its other windows
    one line each and its menu bar in one line. At most ``deadline_s`` of walking (a partial tree
    says so); ``cancel`` set: it stops where it is. ``compact``: the controls only, shorter.
    ``blind``: for a brain that can't see — no coordinates anywhere in it."""
    if not is_trusted():
        from neo2.core.agent.confirm import ask_user

        return ask_user(
            "Accessibility permission is off. Allow NEO in System Settings → Privacy & Security → "
            "Accessibility (NEO's menu has a shortcut), then try again."
        )
    walk = next(_walks)
    if app:
        pid = pid_for_app(app)
        if pid is None:
            return f"Error: app '{app}' is not running."
        name = app
    else:
        name, pid = working_app()
    root = _app_el(pid)
    rv, err = _attrs_err(root, ("AXRole", "AXFocusedWindow", "AXMainWindow", "AXChildren"))
    if err == _CANNOT_COMPLETE:
        return f"Error: {name} isn't responding to Accessibility right now (busy or hung) — try again in a moment."
    w = _Walk(deadline_s, cancel)
    # The window the user works in: the focused one, else the main one (an app that isn't active,
    # some Java and Qt builds), else the only one on screen.
    win = rv.get("AXFocusedWindow") or rv.get("AXMainWindow") or None
    kids = _seq(rv.get("AXChildren"))
    if window:
        chosen = _pick_window(kids, window, w, int(pid or 0))
        if chosen is None:
            return f"Error: {name} has no window called “{window}” — ax_tree lists its windows."
        win = chosen
    elif win is None:
        win = _only_window(kids, w)
    if not compact and window is None:
        cached = _cached(pid, win, name, walk, blind)
        if cached is not None:
            return cached
    gen = _watch_gen(pid)
    if _expose_web(int(pid), root):
        _web_settle(win or root, cancel)
    els: list[Element] = []
    budget = [_MAX_NODES]
    wframe: tuple[float, float, float, float] | None = None
    if win is not None:  # the window the user works in first: a big sidebar can't crowd it out
        w.canvas_min = -1.0  # (set from the window's own frame as the walk reads it)
        _walk(win, 1, els, _new_id, budget=budget, w=w)
        first = els[0] if els and els[0].role == "AXWindow" and els[0].ref is win else None
        wframe = (first.x, first.y, first.w, first.h) if first is not None else _point_size(win)
        if len(els) <= 1 and not w.stopped:
            # Nothing but the window itself in view: an app whose frames don't add up (some Java or
            # Qt builds) — look again without cutting anything to the window.
            els, budget = [], [_MAX_NODES]
            w.blockers.clear()
            w.canvas.clear()
            _walk(win, 1, els, _new_id, budget=budget, clip=False, w=w)
    bar_items: list[Element] = []
    open_menus: list[Element] = []
    background: list[Element] = []
    for c in kids:
        if w.over():
            break
        if win is not None and _same(c, win):
            continue
        cv = _read(c, ("AXRole", "AXSubrole", "AXTitle", "AXPosition", "AXSize", "AXModal", "AXMinimized"), w)
        if cv is None:
            continue
        role = _str(cv.get("AXRole"))
        if role == "AXMenuBar":
            bar_items = _menu_bar(c, _new_id, w, open_menus)
        elif role == "AXWindow" and cv.get("AXModal") is True:
            _walk(c, 1, els, _new_id, budget=budget, w=w)  # a modal window blocks the rest: shown whole
        elif role == "AXWindow":
            x, y, fw, fh = _frame_of(cv)
            if fw <= 0 or fh <= 0:
                continue  # (a window that isn't on screen, closing)
            t = _str(cv.get("AXTitle"))
            how = "minimized" if cv.get("AXMinimized") is True else "another window"
            background.append(
                Element(_new_id(), "AXWindow", t, "", x, y, fw, fh, actions=_LazyActions(c), ref=c,
                        subrole=_str(cv.get("AXSubrole")),
                        note=f'({how}: ax_tree window="{t[:40]}" shows it)' if t else f"({how})")  # fmt: skip
            )
        else:
            _walk(c, 1, els, _new_id, budget=budget, w=w)
    covering = _covering(int(pid or 0), wframe) if wframe is not None else []
    allx = els + open_menus + bar_items + background
    _register(allx, walk, name, int(pid or 0))
    wtitle = els[0].title if els and els[0].ref is win else _str(_attr(win, AS.kAXTitleAttribute)) if win is not None else ""
    head = f"App: {name}" + (f' — window "{wtitle}"' if wtitle else "")
    fixed = _blocker_lines(w.blockers, covering)
    tail: list[str] = []
    if bar_items:
        tail.append("  Menu bar: " + " · ".join(f'{e.id} MenuBarItem "{e.title[:30]}"' for e in bar_items))
    tail += ["  " + e.line(not blind) for e in background]
    if said := w.said():
        tail.append("  " + said)
    elif budget[0] < 0:
        tail.append("  (the walk stopped early: this window is very large)")
    if canvas := _canvas_line(w, blind):
        tail.append("  " + canvas)
    body = els + open_menus
    if compact:
        inside = {id(e) for b in w.blockers for e in b.els}
        body = [e for e in body if e.role not in _READ_ONLY_ROLES or id(e) in inside]
    shown = body[:max_lines]
    budget_chars = min(_lines_budget(), 2500) if compact else _lines_budget()
    left = budget_chars - len(head) - sum(len(x) + 1 for x in fixed) - sum(len(x) + 1 for x in tail)
    if sum(len(e.line(not blind)) + 3 for e in shown) > left:
        # Too much for one look: what can be acted on first, in screen order within each group.
        order = sorted(range(len(shown)), key=lambda i: (shown[i].role in _READ_ONLY_ROLES, i))
        keep: set[int] = set()
        for i in order:
            cost = len(shown[i].line(not blind)) + 3
            if cost > left:
                continue
            keep.add(i)
            left -= cost
        shown = [e for i, e in enumerate(shown) if i in keep]
    lines = [head, *fixed] + ["  " + e.line(not blind) for e in shown]
    if len(body) > len(shown):
        lines.append(f"  … {len(body) - len(shown)} more elements (use ax_find to search)")
    text = "\n".join(lines + tail)
    whole = not w.stopped and budget[0] >= 0
    if whole:
        _maybe_watch(int(pid or 0))
    if not compact and window is None and (whole or gen is None):
        _keep(pid, win, gen if whole else None, text, allx, name, covering, blind)
    return text


def _pick_window(kids: list, want: str, w: _Walk, pid: int = 0):
    """One of the app's windows, by an id from an earlier look (of this app's: never another's
    window under this app's name) or by (part of) its title."""
    e = get(want.strip())
    if e is not None and e.role == "AXWindow" and e.ref is not None:
        return e.ref if pid and pid_of(e.ref) == pid else None
    want_l = want.strip().casefold()
    for c in kids:
        cv = _read(c, ("AXRole", "AXTitle"), w)
        if cv and _str(cv.get("AXRole")) == "AXWindow" and want_l and want_l in _str(cv.get("AXTitle")).casefold():
            return c
    return None


def _only_window(kids: list, w: _Walk):
    """The app's one window on screen (not minimized), when it has exactly one — else None."""
    found = []
    for c in kids:
        cv = _read(c, ("AXRole", "AXSize", "AXMinimized"), w)
        if not cv or _str(cv.get("AXRole")) != "AXWindow" or cv.get("AXMinimized") is True:
            continue
        size = _pair(cv.get("AXSize"), False)
        if size is not None and size[0] > 0 and size[1] > 0:
            found.append(c)
    return found[0] if len(found) == 1 else None


async def tree(
    app: str | None = None, window: str | None = None, *, compact: bool = False, blind: bool = False
) -> str:
    """:func:`snapshot` off the event loop, stopped when the call awaiting it is cancelled."""
    return await wait.in_thread(snapshot, app, window=window, compact=compact, blind=blind)


# ---- the repeat look (axwatch) -------------------------------------------------------------------------
# While NEO.app watches the app's Accessibility notifications, a look at a window nothing has changed
# in since the last look is that look again: the same text, the same ids, no walk.
_CACHE_MAX_S = 5.0
_cache: dict[int, tuple[Any, int, str, list[Element], float, str, tuple, bool]] = {}


def _maybe_watch(pid: int) -> None:
    """NEO.app watches the notifications of the app it looks at, when that app sends them well (a
    native one: Chromium and Electron apps send few, so they are always walked afresh)."""
    try:
        from neo2.core.tools.computer import axwatch

        if not pid or not axwatch.enabled():
            return
        if axwatch.watched(pid):
            axwatch.watch(pid)  # (the app looked at last now: the last to be let go)
            return
        app = _running(pid)
        path = str(app.bundleURL().path()) if app is not None and app.bundleURL() is not None else ""
        from neo2.core.decide import axkit

        if path and axkit.runtime(path) == "native":
            axwatch.watch(pid)
    except Exception:  # noqa: BLE001 - no watching: every look walks
        pass


def _watch_gen(pid: int) -> int | None:
    try:
        from neo2.core.tools.computer import axwatch

        return axwatch.generation(pid)
    except Exception:  # noqa: BLE001
        return None


def _events_sent() -> int:
    try:
        from neo2.core.tools.computer import input as inp

        return int(getattr(inp, "_sent", 0))
    except Exception:  # noqa: BLE001
        return -1


def _keep(pid: int, win, gen: int | None, text: str, els: list[Element], name: str, covering, blind: bool) -> None:
    if not pid:
        return
    if gen is None or win is None or _watch_gen(pid) != gen:
        # Not watched while walking (or something changed meanwhile): nothing can be kept, and
        # what was kept from this app before is older than this look.
        _cache.pop(int(pid), None)
        return
    _cache[int(pid)] = (win, gen, text, els, time.monotonic(), name.lower(), (tuple(c.owner_pid for c in covering), _events_sent()), blind)
    _let_go(_cache, 4)


def _let_go(cache: dict, at: int) -> None:
    """Kept looks too old to be used again (older than _CACHE_MAX_S: ``at`` is where an entry says
    when it was kept) are let go — never a look per app ever looked at, each holding its elements."""
    now = time.monotonic()
    for pid, kept in list(cache.items()):
        if now - kept[at] > _CACHE_MAX_S:
            cache.pop(pid, None)


def _cached(pid: int, win, name: str, walk: int, blind: bool) -> str | None:
    """The last look at this window again, when nothing has changed in it since: the app sent no
    notification, NEO sent no input, no other app's window came over it, and it is recent (and
    for the same kind of brain: a blind one is never handed a look with coordinates in it)."""
    kept = _cache.get(int(pid or 0))
    if kept is None or win is None:
        return None
    kwin, gen, text, els, when, kname, (cover, sent), kblind = kept
    if time.monotonic() - when > _CACHE_MAX_S or kname != name.lower() or not _same(kwin, win) or kblind != blind:
        return None
    if _watch_gen(pid) != gen or _events_sent() != sent:
        return None
    if tuple(c.owner_pid for c in _covering(int(pid), _point_size(win))) != cover:
        return None
    _register(els, walk, name, pid)
    return text


def forget(pid: int | None = None) -> None:
    """Drop the kept looks (of one app, or all): the next look walks again."""
    if pid is None:
        _cache.clear()
        _win_cache.clear()
    else:
        _cache.pop(int(pid), None)
        _win_cache.pop(int(pid), None)


@pooled
def find(query: str, app: str | None = None, *, cancel: threading.Event | None = None, blind: bool = False) -> str:
    """Elements whose title/value/role contains `query` (case-insensitive).

    Searches the tree the model is already holding ids for; a fresh walk (new ids) only when
    that tree is stale or belongs to a different app than asked for."""
    wanted = (app or working_app()[0]).lower()
    same = _last_tree_app == wanted
    if not same and _last_tree_pid:  # "Chrome" and "Google Chrome" are one app: compare pids
        want_pid = pid_for_app(app) if app else working_app()[1]
        same = bool(want_pid) and want_pid == _last_tree_pid
    looked = ""
    if not _last_tree or not same or time.time() - _last_tree_ts > _FIND_MAX_AGE_S:
        out = snapshot(app, cancel=cancel, blind=blind)
        if isinstance(out, str) and out.startswith(("Error", "NEEDS_")):
            return out
        looked = f"(looked at {app or 'the front app'} again: these ids are new)\n"
    q = query.lower()
    hits = [
        e
        for e in _last_tree.values()
        if q in e.title.lower() or q in e.value.lower() or q in e.role.lower()
    ]
    if not hits:
        return f"{looked}No element matching '{query}' (tree from {int(time.time() - _last_tree_ts)}s ago)."
    return looked + "\n".join(e.line(not blind) for e in hits[:40])


async def find_async(query: str, app: str | None = None, *, blind: bool = False) -> str:
    """:func:`find` off the event loop, stopped when the call awaiting it is cancelled."""
    return await wait.in_thread(find, query, app, blind=blind)


def get(eid: str) -> Element | None:
    """The element an id was issued for (from the latest look, or an earlier one)."""
    return _last_tree.get(eid) or _issued.get(eid)


# ---- acting ----------------------------------------------------------------------------------------------
# AXPress and its plain equivalents. Anything else an element offers (AXConfirm — a Return that
# sends —, AXIncrement, AXShowMenu, AXCancel…) is not a "press".
_PRESS_ACTIONS = ("AXPress", "AXPick", "AXOpen")
_UNSUPPORTED = (-25206, -25208, -25200)  # action unsupported, not implemented, plain failure
_SYSTEM_MENU = "Error: that's in the system menu (the Apple menu) — NEO never uses it."
_MENU_UNSURE = (
    "Error: that menu isn't answering, so NEO can't tell it apart from the system menu (the Apple "
    "menu), which it never uses — nothing was pressed; look again in a moment."
)


def _system_menu_refusal(el) -> str:
    """Why a menu element is never pressed ('' when it may be): the system menu, or can't tell."""
    place = _menu_place(el)
    return _SYSTEM_MENU if place == "system" else _MENU_UNSURE if place == "unknown" else ""


def press_action(e: Element) -> str:
    return next((a for a in _PRESS_ACTIONS if a in e.actions), "")


def in_system_menu(el) -> bool:
    """Is this menu element the system (Apple) menu or inside it — the menu bar's first item, present
    in every app's menu bar: Restart…, Shut Down…, Log Out, Force Quit — or can't that be told (a
    menu or menu bar that doesn't answer)? Told by where it sits, never by its words; when its place
    can't be read, it is taken to be there (never pressed on a guess)."""
    return _menu_place(el) != "app"


def _menu_place(el) -> str:
    """"system": the system menu or in it; "app": anywhere else (an app's own menu, a popup or
    context menu, not in a menu at all); "unknown": its menus or menu bar didn't say. The first item
    of any menu bar is the system menu, however many items the bar has (a bar holding the Apple menu
    alone is no app's): as gates._in_system_menu tells it. A status-bar menu's first item (an app's
    AXExtrasMenuBar) is refused the same way — never pressed on a guess that it isn't."""
    p = el
    for _ in range(12):
        if p is None:
            return "app"  # the chain ended before any menu bar: a popup or context menu
        v, err = _attrs_err(p, ("AXRole", "AXParent"))
        role = _str(v.get("AXRole"))
        if err == _INVALID and p is el:
            return "app"  # gone: nothing left to press (the press says so)
        if err or not role:
            return "unknown"
        if role == "AXMenuBarItem":
            bar = v.get("AXParent")
            if bar is None:
                return "unknown"
            bv, berr = _attrs_err(bar, ("AXChildren",))
            items = _seq(bv.get("AXChildren"))
            if berr or not items:
                return "unknown"
            return "system" if _same(items[0], p) else "app"
        if role not in ("AXMenu", "AXMenuItem"):
            return "app"  # not (or no longer) in a menu
        p = v.get("AXParent")
    return "unknown"  # nested deeper than any app's menus: not told apart


def _menu_role(e: Element) -> bool:
    return e.role in ("AXMenuItem", "AXMenuBarItem")


@pooled
def press_el(e: Element) -> str:
    """Press the element in hand."""
    action = press_action(e)
    what = f"{e.id} ({e.role.removeprefix('AX')} {e.title!r})"
    if not action:
        offers = ", ".join(a for a in e.actions) or "none"
        return f"Error: {e.id} can't be pressed (its actions: {offers}) — click it by id instead."
    if _menu_role(e) and e.ref is not None and (no := _system_menu_refusal(e.ref)):
        return no
    err = perform_action(e.ref, action)
    if err == 0:
        return f"Pressed {what}"
    if err == _CANNOT_COMPLETE:
        return (
            f"Pressed {what}, but the app didn't answer in time (busy, or it opened a dialog) — look "
            "before pressing again."
        )
    return f"Error: AX action failed ({err})"


def press(eid: str) -> str:
    e = get(eid)
    if not e:
        return f"Error: unknown element {eid} — ids come from the latest ax_tree; call it again."
    return press_el(e)


_EDITABLE = {"AXTextArea", "AXTextField", "AXSearchField", "AXComboBox"}
_SINGLE_LINE = {"AXTextField", "AXSearchField", "AXComboBox"}


def secure(el) -> bool:
    v = _attrs(el, ("AXRole", "AXSubrole"))
    return "AXSecureTextField" in (_str(v.get("AXRole")), _str(v.get("AXSubrole")))


@pooled
def set_el_value(e: Element, value: str) -> str:
    """Put text into a text field / area and read it back: AXValue first; a field that ignores it
    (many web and framework inputs) is focused and typed into — never a line break (a Return there
    could send or run it: type_text knows how to type one where the cursor is). Never a password
    field, never a checkbox, slider or popup (those are pressed, through the click gate)."""
    if e.role == "AXSecureTextField" or secure(e.ref):
        return "Error: that's a password field — NEO never fills those."
    web = _attr(e.ref, "AXEditableAncestor") is not None
    if e.role not in _EDITABLE and not web:
        return f"Error: {e.id} is a {e.role.removeprefix('AX')}, not a text field — press or click it instead."
    from neo2.core.decide import axkit

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if axkit._put(e.ref, value, ""):  # "": no chord for a line break — those are never typed here
        return f"Set {e.id} to “{value[:60]}”."
    now = _str(_attr(e.ref, AS.kAXValueAttribute))
    if not _norm_text(now) and axkit._CONTROL.search(value):
        return (
            f"Error: {e.id} didn't take the text, and typing it in would press Return at each line break "
            "(which could send or run it) — click into it and use type_text, which types line breaks "
            "the safe way for where the cursor is."
        )
    return f"Error: {e.id} didn't take the text (it reads “{now[:60]}”) — click into it and use type_text."


def _norm_text(s: str) -> str:
    return " ".join(s.split())


def set_value(eid: str, value: str) -> str:
    e = get(eid)
    if not e:
        return f"Error: unknown element {eid} — ids come from the latest ax_tree; call it again."
    return set_el_value(e, value)


def focused_role() -> tuple[str, str]:
    """(role, subrole) of whatever has the keyboard focus, or ("", "")."""
    el = _focused()
    if el is None:
        return "", ""
    v = _attrs(el, ("AXRole", "AXSubrole"))
    return _str(v.get("AXRole")), _str(v.get("AXSubrole"))


def focused_is_secure() -> bool:
    """Is the caret in a password field? NEO never types there."""
    role, sub = focused_role()
    return "AXSecureTextField" in (role, sub)


def focused_editable() -> bool:
    """Is the keyboard focus in something you can type into (never a password field)?"""
    el = _focused()
    if el is None:
        return False
    v = _attrs(el, ("AXRole", "AXSubrole"))
    role = _str(v.get("AXRole"))
    if "AXSecureTextField" in (role, _str(v.get("AXSubrole"))):
        return False
    if role in _EDITABLE:
        return True
    # Web editors (Gmail, Docs) report other roles; a settable AXValue is the practical test.
    err, settable = AS.AXUIElementIsAttributeSettable(el, AS.kAXValueAttribute, None)
    return err == 0 and bool(settable) and role not in ("AXSlider", "AXCheckBox", "AXRadioButton")


@pooled
def focused_window(app: str | None = None):
    """(app name, AX window element) of the frontmost (or named) app's focused window."""
    if app:
        pid, name = pid_for_app(app), app
    else:
        name, pid = frontmost_app()
    if not pid:
        return name, None
    v = _attrs(_app_el(pid), ("AXFocusedWindow", "AXMainWindow"))
    return name, v.get("AXFocusedWindow") or v.get("AXMainWindow")


@pooled
def window_elements(
    app: str | None = None,
    *,
    early_exit: str | None = None,
    deadline_s: float = _LOOK_S,
    cancel: threading.Event | None = None,
) -> tuple[str, tuple[float, float, float, float] | None, list[Element]]:
    """Elements of the focused window only (not every window of the app), plus its frame.
    ``early_exit``: a label — the walk ends at the first enabled element on screen called exactly
    that (click_text's target, a gate's one word)."""
    name, win = focused_window(app)
    if win is None:
        return name, None, []
    pid = pid_of(win)
    gen = _watch_gen(pid)
    if early_exit is None and (kept := _kept_window(pid, win, gen)) is not None:
        return name, kept[0], kept[1]
    if pid and _expose_web(pid, _app_el(pid)):
        _web_settle(win, cancel)
    els: list[Element] = []
    n = itertools.count(1)
    w = _Walk(deadline_s, cancel)
    w.eager = True
    if early_exit:
        want = " ".join(early_exit.split()).casefold()

        def exact(e: Element) -> bool:
            said = (e.title or e.value).strip().casefold()
            return said == want and e.enabled and e.visible is not None and e.visible[2] > 0
        w.early = exact
    _walk(win, 0, els, lambda: f"w{next(n)}", budget=[_MAX_NODES], w=w)
    frame = _point_size(win)
    if early_exit is None and not w.stopped and gen is not None and _watch_gen(pid) == gen:
        _win_cache[int(pid)] = (win, gen, _events_sent(), time.monotonic(), frame, els)
        _let_go(_win_cache, 3)
    elif pid and (gen is None or _watch_gen(pid) != gen):
        _win_cache.pop(int(pid), None)  # what was kept from this app is older than this walk
    return name, frame, els


# The focused window's elements, kept while nothing has changed in it (see _cached): the gate's and
# click_text's walk of a window just looked at costs nothing.
_win_cache: dict[int, tuple[Any, int, int, float, tuple[float, float, float, float], list[Element]]] = {}


def _kept_window(pid: int, win, gen: int | None):
    kept = _win_cache.get(int(pid or 0))
    if kept is None or gen is None:
        return None
    kwin, kgen, sent, when, frame, els = kept
    if kgen != gen or sent != _events_sent() or time.monotonic() - when > _CACHE_MAX_S or not _same(kwin, win):
        return None
    return frame, els


def _text_areas(el, depth: int, out: list[Element], stop: float = 0.0) -> None:
    """Every text area under `el` — its own walk, so a busy sidebar can't hit an element cap."""
    if depth > _MAX_DEPTH or len(out) > 50 or (stop and wait.now() > stop):
        return
    v = _attrs(el, ("AXRole", "AXPosition", "AXSize", "AXValue", "AXEnabled", "AXFocused", "AXChildren"))
    role = _str(v.get("AXRole"))
    if role == "AXTextArea":
        x, y, w, h = _frame_of(v)
        en, fo = v.get("AXEnabled"), v.get("AXFocused")
        out.append(
            Element(
                id=f"t{len(out)}",
                role=role,
                title="",
                value=_str(v.get("AXValue")),
                x=x,
                y=y,
                w=w,
                h=h,
                enabled=True if en is None else bool(en),
                focused=False if fo is None else bool(fo),
                ref=el,
            )
        )
        return
    for c in _seq(v.get("AXChildren")):
        _text_areas(c, depth + 1, out, stop)


@pooled
def main_text_area(app: str | None = None) -> Element | None:
    """The document body of the focused window: the focused text area, else the biggest one.
    None when that window shows no editor (e.g. Notes with no note selected)."""
    _, win = focused_window(app)
    if win is None:
        return None
    areas: list[Element] = []
    _text_areas(win, 0, areas, wait.now() + _LOOK_S)
    areas = [e for e in areas if e.enabled and e.w > 40 and e.h > 20]
    if not areas:
        return None
    focused = [e for e in areas if e.focused]
    return focused[0] if focused else max(areas, key=lambda e: e.w * e.h)


@pooled
def work_window_frame() -> tuple[float, float, float, float] | None:
    """The frame of the focused window of the app the user works in (never NEO's)."""
    name, pid = working_app()
    if not pid or is_neo(pid, name):
        return None
    v = _attrs(_app_el(pid), ("AXFocusedWindow", "AXMainWindow"))
    win = v.get("AXFocusedWindow") or v.get("AXMainWindow")
    if win is None:
        return None
    f = _point_size(win)
    return f if f[2] > 0 and f[3] > 0 else None


def visible_frame(el) -> tuple[float, float, float, float] | None:
    """The part of an element that is on screen: its frame cut to the scroll areas and window
    around it (a text area or a web area reports its whole document). None when none of it is.
    Within a look's limits: an app that stops answering on the way up ends it there."""
    w = _Walk(_LOOK_S)
    v = _read(el, ("AXPosition", "AXSize", "AXParent"), w) or {}
    box = _frame_of(v)
    if box[2] <= 0 or box[3] <= 0:
        return None
    p = v.get("AXParent")
    for _ in range(12):
        if p is None or w.over():
            break
        pv = _read(p, ("AXRole", "AXPosition", "AXSize", "AXParent"), w)
        if pv is None:
            break
        role = _str(pv.get("AXRole"))
        if role in _FLOATING:
            break  # an open menu or popover floats over its window: the window doesn't cut it
        if role in _CLIPS:
            clip = _frame_of(pv)
            if clip[2] > 0 and clip[3] > 0:
                box = _clip(box, clip)
        p = pv.get("AXParent")
    return box if box[2] >= 1 and box[3] >= 1 else None


def focus_el(el) -> bool:
    """Give the element the keyboard focus through AX (no click), and say whether it took it. (A
    refusal at the door goes on up: nothing may type in its place.)"""
    if set_attribute(el, AS.kAXFocusedAttribute, True, what="focus a text area in another app's UI") != 0:
        return False
    return bool(_attr(el, AS.kAXFocusedAttribute, False))


@pooled
def press_element(e: Element) -> str:
    """Press the element in hand (not by id through the shared tree another call may replace)."""
    action = "AXPress" if "AXPress" in e.actions else ""
    if not action:
        return "Error: no press action"
    if _menu_role(e) and e.ref is not None and (no := _system_menu_refusal(e.ref)):
        return no
    err = perform_action(e.ref, action)
    return "Pressed" if err == 0 else f"Error: AX action failed ({err})"


def focused_element_text() -> str:
    el = _focused()
    if el is None:
        return ""
    v = _attrs(el, ("AXValue", "AXSelectedText"))
    return _str(v.get("AXValue")) or _str(v.get("AXSelectedText"))


# What a terminal view says it is: an editor's terminal pane "Terminal 1, zsh", a web terminal's
# input "Terminal input" (never a file named terminal.py).
_TERMINAL = re.compile(r"\bterminal\b(?!\.\w)", re.I)


@pooled
def focused_terminal(pid: int = 0) -> bool:
    """Is the keyboard focus in a terminal view — told by what the focused element, or the pane
    around it, says it is? For an app that runs a shell in a pane (focus.hosts_shell: "pane"), a
    Return runs a command only there; its chat box or editor is still a chat box or an editor.
    ``pid``: that app, whose web tree is switched on first if it is a Chromium or Electron app (an
    editor's terminal pane is a web view: without it, nothing in it can be seen)."""
    if pid:
        root = _app_el(pid)
        if _expose_web(pid, root):
            _web_settle(_attr(root, AS.kAXFocusedWindowAttribute) or root)
    return is_terminal(_focused())


def is_terminal(el) -> bool:
    """Does this element, or the pane around it, say it is a terminal view?"""
    for _ in range(2):
        if el is None:
            return False
        v = _attrs(el, ("AXDescription", "AXTitle", "AXHelp", "AXParent"))
        said = " ".join(_str(v.get(a)) for a in ("AXDescription", "AXTitle", "AXHelp"))
        if _TERMINAL.search(said):
            return True
        el = v.get("AXParent")
        el = None if el is None else _bounded(el)
    return False


def _element_at(x: float, y: float):
    try:
        err, el = AS.AXUIElementCopyElementAtPosition(_system_wide(), float(x), float(y), None)
    except Exception:  # noqa: BLE001
        return None
    if err or el is None:
        return None
    if _library():  # (in NEO.app the process-wide timeout already covers it)
        _own_timeout(el)  # its app answers the questions that follow briefly, even when hung
    return el


_LABEL_ATTRS = ("AXTitle", "AXDescription", "AXValue", "AXParent")


def _label_of(el, v: dict[str, Any] | None = None, w: _Walk | None = None) -> str:
    """What an element says, and its parent's (text inside a button: the button's too)."""
    parts: list[str] = []
    for _ in range(2):
        if el is None or (w is not None and w.over()):
            break
        v = v if v is not None else (_read_text(el, _LABEL_ATTRS, w) or dict.fromkeys(_LABEL_ATTRS))
        for a in ("AXTitle", "AXDescription", "AXValue"):
            s = _str(v.get(a))
            if s:
                parts.append(s)
                break
        el, v = v.get("AXParent"), None
    return " ".join(parts)[:200]


@pooled
def label_at(x: float, y: float) -> str:
    """The label of whatever is under a screen point — and its parent's, for text inside a button."""
    el = _element_at(x, y)
    return "" if el is None else _label_of(el)


def pid_of(el) -> int:
    try:
        err, pid = AS.AXUIElementGetPid(el, None)
        return int(pid) if err == 0 and pid else 0
    except Exception:  # noqa: BLE001
        return 0


def _running(pid: int):
    try:
        from AppKit import NSRunningApplication

        return NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if pid else None
    except Exception:  # noqa: BLE001
        return None


@pooled
def app_of(el) -> str:
    """The name of the app an accessibility element belongs to."""
    app = _running(pid_of(el))
    return str(app.localizedName() or "") if app is not None else ""


@pooled
def app_at(x: float, y: float) -> str:
    """The app whose element is under a screen point."""
    el = _element_at(x, y)
    return "" if el is None else app_of(el)


@dataclass
class Hit:
    """What is under a screen point, as Accessibility tells it (all empty: nothing could be told)."""

    pid: int = 0
    app: str = ""
    label: str = ""
    role: str = ""
    regular: bool = False  # an app with windows and a menu bar, which can be brought forward
    sheet: str = ""  # the point is on a window a sheet blocks (not on the sheet): the sheet's words
    el: Any = None
    window: Any = None


_WORDS_ATTRS = ("AXRole", "AXTitle", "AXDescription", "AXValue", "AXChildren")


def _words(el, limit: int = 3) -> tuple[list[str], list[str]]:
    """(texts, button titles) inside a sheet or dialog: what it says and what it offers — within
    _WORDS_BUDGET_S, and never past an app that stopped answering."""
    texts: list[str] = []
    buttons: list[str] = []
    todo, seen = [(el, 0)], 0
    w = _Walk(_WORDS_BUDGET_S)
    while todo and seen < 120 and not w.over():
        node, d = todo.pop(0)
        seen += 1
        v = _read_text(node, _WORDS_ATTRS, w)
        if v is None:
            continue
        role = _str(v.get("AXRole"))
        if role == "AXButton":
            t = _str(v.get("AXTitle")) or _str(v.get("AXDescription"))
            if t and len(buttons) < 6:
                buttons.append(t[:30])
        elif role in ("AXStaticText", "AXTextArea") and len(texts) < limit:
            t = _str(v.get("AXValue")) or _str(v.get("AXTitle"))
            if t.strip():
                texts.append(" ".join(t.split())[:80])
        if d < 6:
            todo.extend((c, d + 1) for c in _seq(v.get("AXChildren")))
    return texts, buttons


def summary(el) -> str:
    """A sheet or dialog in a few words: “Start a New Game?” (buttons: Start, Cancel)."""
    t0 = wait.now()
    title = _str(_attr(el, AS.kAXTitleAttribute))
    if not title and wait.now() - t0 >= _REAL_TIMEOUT * _AX_TIMEOUT_S:
        return "(no text)"  # (its app isn't answering: asking for its words would wait as long again)
    texts, buttons = _words(el)
    said = " ".join(f"“{t}”" for t in ([title] if title else []) + texts[:2])
    return (said or "(no text)") + (f" (buttons: {', '.join(buttons)})" if buttons else "")


_WORDS_BUDGET_S = 0.5


@pooled
def window_words(h: Hit) -> str:
    """What the part of the window under a point says it is for — the element's own description,
    the texts beside it at each level up to its window, and the window's title (an AirDrop window:
    "AirDrop lets you share instantly with people nearby" next to its recipients) — for judging
    what dropping something there does. Not buttons: every window has some."""
    said: list[str] = []
    el = h.el
    stop = wait.now() + _WORDS_BUDGET_S  # a busy app answers each question slowly: enough is enough
    for _ in range(8):
        if el is None or len(said) >= 16 or wait.now() > stop:
            break
        v = _attrs(el, ("AXDescription", "AXHelp", "AXRole", "AXTitle", "AXChildren", "AXParent"))
        said += [s[:80] for a in ("AXDescription", "AXHelp") if (s := _str(v.get(a)).strip())]
        if _str(v.get("AXRole")) in ("AXWindow", "AXSheet"):
            said.append(_str(v.get("AXTitle")))
            break
        for child in _seq(v.get("AXChildren"))[:30]:
            if wait.now() > stop:
                break
            cv = _attrs(child, ("AXRole", "AXValue", "AXTitle"))
            if _str(cv.get("AXRole")) == "AXStaticText":
                text = _str(cv.get("AXValue")) or _str(cv.get("AXTitle"))
                if text.strip():
                    said.append(" ".join(text.split())[:80])
        el = v.get("AXParent")
    if h.el is None and h.window is not None:
        said.append(_str(_attr(h.window, AS.kAXTitleAttribute)))
    return " ".join(w for w in said if w)[:600]


def _shown(el) -> bool:
    _x, _y, w, h = _point_size(el)
    return w > 0 and h > 0


def _sheet_of(win, kids: list | None = None, w: _Walk | None = None) -> Any:
    """The sheet open on a window (one with a size: a closed one lingering in the tree isn't).
    Its last _SHEET_KIDS children are looked at, from the last (where a sheet is added), within
    ``w``'s limits (default: a look's, _LOOK_S): an app that stopped answering ends the check."""
    w = w if w is not None else _Walk(_LOOK_S)
    if kids is None:
        got = _read(win, ("AXChildren",), w)
        kids = _seq(got.get("AXChildren")) if got else []
    for c in reversed(kids[-_SHEET_KIDS:]):
        if w.over():
            return None
        v = _read(c, ("AXRole", "AXSize"), w)
        if v and _str(v.get("AXRole")) == "AXSheet":
            size = _pair(v.get("AXSize"), False) or (0.0, 0.0)
            if size[0] > 0 and size[1] > 0:
                return c
    return None


def regular(pid: int) -> bool:
    """An app with windows and a menu bar (not the Dock, a menu-bar extra, a background agent)."""
    app = _running(pid)
    try:
        return app is not None and int(app.activationPolicy()) == 0
    except Exception:  # noqa: BLE001
        return False


def _inside(x: float, y: float, f: tuple[float, float, float, float]) -> bool:
    return f[0] <= x < f[0] + f[2] and f[1] <= y < f[1] + f[3]


def _blocked(pid: int, win, x: float, y: float, kids: list | None = None, w: _Walk | None = None) -> str:
    """What blocks a point on ``win`` (None: a point Accessibility can't see, on the app's front
    window): a sheet open on that window, or a modal window of the app's in front of it — its words;
    '' when nothing does. A point on the sheet or the modal window itself is never blocked by it.
    Within ``w``'s limits (default: a look's): an app that stopped answering says nothing more."""
    w = w if w is not None else _Walk(_LOOK_S)
    fv = (_read(_app_el(pid), ("AXFocusedWindow",), w) or {}) if pid else {}
    focused = fv.get("AXFocusedWindow")
    f = (0.0, 0.0, 0.0, 0.0)
    if focused is not None and (win is None or not _same(focused, win)) and not w.over():
        mv = _read(focused, ("AXModal", "AXPosition", "AXSize"), w) or {}
        f = _frame_of(mv)
        if mv.get("AXModal") is True and f[2] > 0 and f[3] > 0 and not _inside(x, y, f):
            return summary(focused)  # an app-modal alert blocks every other window of its app
    target = win if win is not None else focused
    if target is None:
        return ""
    if win is None and f[2] > 0 and f[3] > 0 and not _inside(x, y, f):
        return ""  # a point on another window of the app (a board beside the document): not its sheet's
    if w.over():
        return ""
    sheet = _sheet_of(target, kids, w)
    if sheet is None or w.over():
        return ""
    if not _inside(x, y, _frame_of(_read(sheet, ("AXPosition", "AXSize"), w))):
        return summary(sheet)
    return ""


_HIT_ATTRS = ("AXRole", "AXTitle", "AXDescription", "AXValue", "AXParent", "AXWindow")


@pooled
def hit(x: float, y: float) -> Hit:
    """Who owns a screen point and what's there — one Accessibility lookup for the gate, for
    bringing the right app forward, for keeping off NEO's own window and for the report. A point on
    a window that a sheet or an app-modal dialog blocks says so (``sheet``), even where the app
    shows Accessibility nothing (a board drawn by the app itself)."""
    el = _element_at(x, y)
    if el is None:  # an app that shows nothing to Accessibility there (a game drawn with Metal)
        name, pid = window_owner_at(x, y) or ("", 0)  # (can't say: unnamed, and the gates fail closed)
        if not pid:
            return Hit()
        h = Hit(pid=pid, app=name, regular=regular(pid))
        try:
            h.sheet = _blocked(pid, None, x, y)
        except Exception:  # noqa: BLE001 - can't tell: as before, nothing said
            h.sheet = ""
        return h
    pid = pid_of(el)
    app = _running(pid)
    w = _Walk(_LOOK_S)  # (one look's limits for all of it: an app that stopped answering is asked no more)
    v = _read_text(el, _HIT_ATTRS, w) or dict.fromkeys(_HIT_ATTRS)
    h = Hit(
        pid=pid,
        app=str(app.localizedName() or "") if app is not None else "",
        label=_label_of(el, v, w),
        role=_str(v.get("AXRole")),
        regular=app is not None and int(app.activationPolicy()) == 0,
        el=el,
    )
    win = v.get("AXWindow")
    h.window = win
    if win is not None and not w.over():
        wv = _read(win, ("AXRole", "AXChildren"), w) or {}
        if _str(wv.get("AXRole")) != "AXSheet" and not w.timed_out:
            h.sheet = _blocked(pid, win, x, y, _seq(wv.get("AXChildren")), w)
    return h


@pooled
def raise_window(h: Hit) -> bool:
    """Make the window under a point its app's key window when it isn't (a click on a background
    window of the front app is otherwise taken as "activate this window" by many views)."""
    if h.window is None or not h.pid:
        return False
    if _str(_attr(h.window, AS.kAXRoleAttribute)) != "AXWindow":
        return False
    focused = _attr(_app_el(h.pid), AS.kAXFocusedWindowAttribute)
    if focused is None or _same(focused, h.window) or _attr(focused, "AXModal") is True:
        return False  # (a modal window stays on top: never raise another over it)
    what = f"raise {h.app or 'an app'}'s window under the pointer"
    perform_action(h.window, "AXRaise", kind="focus", what=what)  # (a refusal at the door goes on up)
    set_attribute(h.window, "AXMain", True, kind="focus", what=what)
    return True


@dataclass
class Scene:
    """What the user would notice changing: which app is in front, and in the app acted on, its
    focused window's title, a sheet or dialog in it, and where the keyboard focus is."""

    front: str = ""
    front_pid: int = 0
    title: str = ""
    sheet: str = ""
    focus: str = ""
    known: bool = False


@pooled
def scene(pid: int, front: tuple[str, int] | None = None) -> Scene:
    """The scene now. ``front``: the front app when the caller already knows it (one question
    fewer)."""
    if front is None:
        from neo2.core.tools.computer.focus import front_app

        try:
            front = front_app()
        except Exception:  # noqa: BLE001
            return Scene()
    name, fpid = front
    s = Scene(front=name, front_pid=fpid, known=True)
    if not pid:
        return s
    # One look's limits for all of it (it is polled every few dozen ms while a press is watched): a
    # bounded number of elements, a deadline, and nothing more asked of an app that stopped answering.
    w = _Walk(_LOOK_S)
    rv = _read(_app_el(pid), ("AXFocusedWindow", "AXFocusedUIElement"), w) or {}
    win = rv.get("AXFocusedWindow")
    if win is not None and not w.timed_out:
        wv = _read(win, ("AXTitle", "AXSubrole", "AXChildren"), w) or {}
        s.title = _str(wv.get("AXTitle"))
        sheet = _sheet_of(win, _seq(wv.get("AXChildren")), w) if not w.timed_out else None
        if sheet is not None:
            s.sheet = "a sheet " + summary(sheet)
        elif _str(wv.get("AXSubrole")) in _DIALOG_SUBROLES:
            s.sheet = "a dialog " + summary(win)
    f = rv.get("AXFocusedUIElement")
    if f is not None and not w.timed_out:
        fv = _read(f, ("AXRole", "AXTitle", "AXDescription"), w) or {}
        role = _str(fv.get("AXRole")).removeprefix("AX")
        label = _str(fv.get("AXTitle")) or _str(fv.get("AXDescription"))
        s.focus = f"{role} “{label[:40]}”" if label else role
    return s


def window_owner_at(x: float, y: float) -> tuple[str, int] | None:
    """(name, pid) of the app owning the frontmost on-screen window under a screen point, from
    the window server — which knows every window, accessible or not. ("", 0): no window there;
    None: it couldn't say."""
    try:
        import Quartz as Q

        wins = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        )
    except Exception:  # noqa: BLE001
        return None
    if wins is None:
        return None
    for w in wins:  # front to back
        b = w.get("kCGWindowBounds") or {}
        wx, wy, ww, wh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if ww <= 0 or wh <= 0 or float(w.get("kCGWindowAlpha", 1) or 0) <= 0:
            continue
        if wx <= x < wx + ww and wy <= y < wy + wh:
            return str(w.get("kCGWindowOwnerName", "") or ""), int(w.get("kCGWindowOwnerPID", 0) or 0)
    return "", 0


def app_names(pid: int) -> set[str]:
    """Every name the app with this pid goes by: its own, its bundle id, its executable's."""
    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
        if app is None:
            return set()
        url = app.executableURL()
        names = (app.localizedName(), app.bundleIdentifier(), url.lastPathComponent() if url is not None else "")
    except Exception:  # noqa: BLE001
        return set()
    return {str(n) for n in names if n}


def names_of_app(name: str) -> set[str]:
    """Every name the running apps called `name` go by (their bundle ids and executables)."""
    want = (name or "").strip().lower()
    out: set[str] = set()
    try:
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if str(app.localizedName() or "").strip().lower() == want:
                out |= app_names(int(app.processIdentifier()))
    except Exception:  # noqa: BLE001
        pass
    return out


# ---- what is in the way -----------------------------------------------------------------------------


@dataclass
class Blocker:
    """Something to deal with before the app's window can be used: a sheet on it ("sheet"), a modal
    dialog of the app ("modal"), a dialog window in front ("dialog"), or another app's window over
    it ("covering"). ``buttons``: its controls, with ids (press them by id); ``text``: what it says
    (read off the screen)."""

    kind: str
    title: str
    buttons: list[Element] = field(default_factory=list)
    owner_pid: int = 0
    text: str = ""
    frame: tuple[float, float, float, float] | None = None
    ref: Any = None


def _as_blocker(b: _Within, pid: int) -> Blocker:
    ctl = [e for e in b.els if e.role not in _READ_ONLY_ROLES and e.role not in ("AXGroup", "AXScrollArea")]
    texts = [e.value or e.title for e in b.els if e.role == "AXStaticText" and (e.value or e.title)]
    title = b.title or (texts[0] if texts else "")
    return Blocker(b.kind, title, ctl, pid, " ".join(" ".join(t.split()) for t in texts)[:400], b.frame, b.ref)


def _layer_at(pid: int, x: float, y: float) -> int | None:
    """The window-server layer of the app's frontmost window at a point (None: can't tell)."""
    try:
        import Quartz as Q

        wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID) or []
    except Exception:  # noqa: BLE001
        return None
    for w in wins:  # front to back
        b = w.get("kCGWindowBounds") or {}
        wx, wy, ww, wh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if int(w.get("kCGWindowOwnerPID", 0) or 0) == pid and wx <= x < wx + ww and wy <= y < wy + wh:
            return int(w.get("kCGWindowLayer", 0) or 0)
    return None


def _covering(pid: int, frame: tuple[float, float, float, float] | None) -> list[Blocker]:
    """Another app's window floating over the middle of this app's window (a higher window-server
    layer than a normal window: a panel, an alert, an overlay), as the window server tells it:
    evidence, never names. An ordinary window of an app in front isn't one (bringing this app
    forward uncovers it), nor is any of NEO's own."""
    if not pid or frame is None or frame[2] <= 0 or frame[3] <= 0:
        return []
    mx, my = frame[0] + frame[2] / 2, frame[1] + frame[3] / 2
    got = window_owner_at(mx, my)
    if not got:
        return []
    name, owner = got
    if not owner or owner == pid or is_neo(owner, name):
        return []
    layer = _layer_at(owner, mx, my)
    if layer is not None and layer <= 0:
        return []
    return [Blocker("covering", name, [], owner)]


@pooled
def blockers(pid: int, *, deadline_s: float = _LOOK_S, cancel: threading.Event | None = None) -> list[Blocker]:
    """What must be answered or moved before the app's front window can be used, most urgent first:
    a sheet on it, a modal or dialog window of the app in front, another app's window over it."""
    if not pid:
        return []
    w = _Walk(deadline_s, cancel)
    rv = _read(_app_el(pid), ("AXFocusedWindow", "AXMainWindow"), w) or {}
    win = rv.get("AXFocusedWindow") or rv.get("AXMainWindow")
    if win is None:
        return []
    wv = _read(win, ("AXRole", "AXSubrole", "AXModal", "AXTitle", "AXPosition", "AXSize", "AXChildren"), w) or {}
    frame = _frame_of(wv)
    els: list[Element] = []
    if wv.get("AXModal") is True or _str(wv.get("AXSubrole")) in _DIALOG_SUBROLES:
        _walk(win, 1, els, _new_id, w=w)  # the dialog itself, and any sheet on it
    elif (sheet := _sheet_of(win, _seq(wv.get("AXChildren")), w)) is not None:
        _walk(sheet, 2, els, _new_id, w=w)
    _issue(els)  # (its controls can be pressed by id)
    found = sorted(w.blockers, key=lambda b: b.kind != "sheet")
    return [_as_blocker(b, pid) for b in found] + _covering(pid, frame)


# ---- menus -------------------------------------------------------------------------------------------------

_MENU_ATTRS = (
    "AXRole",
    "AXTitle",
    "AXEnabled",
    "AXMenuItemCmdChar",
    "AXMenuItemCmdModifiers",
    "AXMenuItemCmdVirtualKey",
    "AXMenuItemMarkChar",
    "AXChildren",
)
# AXMenuItemCmdModifiers: 0 is ⌘ alone; these bits add a modifier, the last takes ⌘ away.
_MENU_SHIFT, _MENU_OPTION, _MENU_CONTROL, _MENU_NO_COMMAND, _MENU_FN = 1, 2, 4, 8, 16
_MENU_MODS = ((_MENU_SHIFT, "shift"), (_MENU_OPTION, "alt"), (_MENU_CONTROL, "ctrl"), (_MENU_FN, "fn"))
# What a menu shows for a key with no letter: a control character (Carbon's and AppKit's)…
_MENU_CONTROL_KEYS = {
    "\x01": "home", "\x03": "return", "\x04": "end", "\x05": "help", "\x08": "delete", "\t": "tab",
    "\x0b": "pageup", "\x0c": "pagedown", "\r": "return", "\x19": "tab", "\x1b": "escape",
    "\x1c": "left", "\x1d": "right", "\x1e": "up", "\x1f": "down", "\x7f": "delete",
}  # fmt: skip
# …or one of NSEvent's function-key characters (U+F700–U+F8FF): the arrows, F1…, and the like.
_MENU_FUNCTION_KEYS = {
    0xF700: "up", 0xF701: "down", 0xF702: "left", 0xF703: "right", 0xF728: "forwarddelete",
    0xF729: "home", 0xF72B: "end", 0xF72C: "pageup", 0xF72D: "pagedown", 0xF746: "help",
    **{0xF704 + n: f"f{n + 1}" for n in range(35)},
}  # fmt: skip


def _menu_key(char: str, vkey: int) -> str:
    """A menu item's key as input names it: a printable character as itself (a space is "space"),
    a control or function-key character by the key it stands for, anything else (an emoji, a
    character with no key) by its virtual key; '' when none of those names a key NEO can press."""
    from neo2.core.tools.computer import input as inp

    char = char or ""
    if char == " " or (not char.strip() and char and " " in char):
        name = "space"
    else:
        c = char.strip()
        if len(c) == 1 and "!" <= c <= "~":
            name = c.lower()
        elif len(c) == 1 and (c in _MENU_CONTROL_KEYS or ord(c) in _MENU_FUNCTION_KEYS):
            name = _MENU_CONTROL_KEYS.get(c) or _MENU_FUNCTION_KEYS[ord(c)]
        else:  # (a virtual key names a letter by the US layout: only its named keys are taken)
            name = next((k for k, code in inp._KEYCODES.items() if vkey and code == vkey and len(k) > 1), "")
    return name if name in inp._KEYCODES or name in inp._SHIFTED else ""


@dataclass
class MenuItem:
    """One command in the app's menu bar: its title, where it is (``path``: the menu titles down to
    it), whether it can be chosen now, and its keyboard shortcut (``key`` with ``mods``, as
    AXMenuItemCmdModifiers gives them: 0 is ⌘ alone) — a menu item is gated like its shortcut."""

    title: str
    path: tuple[str, ...]
    enabled: bool
    key: str = ""
    mods: int = 0
    vkey: int = 0
    checked: bool = False
    ref: Any = None
    pid: int = 0

    @property
    def chord(self) -> str:
        """The shortcut as input.hotkey writes it ("cmd+shift+n", "cmd+delete", "ctrl+fn+left");
        '' when it has none, or none NEO could press."""
        key = _menu_key(self.key, self.vkey)
        if not key:
            return ""
        mods = [] if self.mods & _MENU_NO_COMMAND else ["cmd"]
        mods += [m for bit, m in _MENU_MODS if self.mods & bit]
        return "+".join([*mods, key])

    def where(self) -> str:
        return " › ".join(self.path)


def _menu_kids(menu, w: _Walk) -> list:
    """A menu's items — asked twice when it has none at first (AppKit fills some menus in only
    when first asked)."""
    got = _read(menu, ("AXChildren",), w)
    kids = _seq(got.get("AXChildren")) if got else []
    if kids or w.over():
        return kids

    def again() -> list:
        v = _read(menu, ("AXChildren",), w)
        return _seq(v.get("AXChildren")) if v else []

    return wait.until_sync(again, 0.05, every=0.01, stop=w.cancel) or []


def _menu_walk(menu, path: tuple[str, ...], out: list[MenuItem], pid: int, w: _Walk, depth: int, limit: int) -> None:
    for mi in _menu_kids(menu, w):
        if len(out) >= limit or w.over():
            return
        v = _read(mi, _MENU_ATTRS, w)
        if v is None or _str(v.get("AXRole")) != "AXMenuItem":
            continue
        title = _str(v.get("AXTitle")).strip()
        if not title:
            continue  # a separator
        en = v.get("AXEnabled")
        try:
            mods = int(v.get("AXMenuItemCmdModifiers") or 0)
            vkey = int(v.get("AXMenuItemCmdVirtualKey") or 0)
        except (TypeError, ValueError):
            mods, vkey = 0, 0
        item = MenuItem(
            title=title,
            path=(*path, title),
            enabled=True if en is None else bool(en),
            key=_str(v.get("AXMenuItemCmdChar")),
            mods=mods,
            vkey=vkey,
            checked=bool(_str(v.get("AXMenuItemMarkChar")).strip()),
            ref=mi,
            pid=pid,
        )
        out.append(item)
        if depth < 4:
            for sub in _seq(v.get("AXChildren")):
                sv = _read(sub, ("AXRole",), w)
                if sv and _str(sv.get("AXRole")) == "AXMenu":
                    _menu_walk(sub, item.path, out, pid, w, depth + 1, limit)


@pooled
def menu_items(
    pid: int, *, deadline_s: float = _LOOK_S, cancel: threading.Event | None = None, limit: int = 1500
) -> list[MenuItem]:
    """Every command in the app's menu bar, without opening a menu — never the system (Apple) menu,
    the bar's first item, which is in every app's menu bar (Restart…, Shut Down…), however many items
    the bar has."""
    if not pid:
        return []
    w = _Walk(deadline_s, cancel)
    rv = _read(_app_el(pid), ("AXMenuBar",), w) or {}
    bar = rv.get("AXMenuBar")
    if bar is None:
        return []
    bv = _read(bar, ("AXChildren",), w) or {}
    items = _seq(bv.get("AXChildren"))
    out: list[MenuItem] = []
    for i, it in enumerate(items):
        if i == 0 or w.over():
            continue  # (the system menu, however many items the bar has)
        iv = _read(it, ("AXRole", "AXTitle", "AXChildren"), w)
        if iv is None or _str(iv.get("AXRole")) != "AXMenuBarItem":
            continue
        top = _str(iv.get("AXTitle")).strip()
        for m in _seq(iv.get("AXChildren")):
            _menu_walk(m, (top,), out, pid, w, 1, limit)
    return out


def press_menu_item(item: MenuItem) -> str:
    """Choose a menu command (AXPress on its item) — never one in the system menu, never a disabled
    one, never NEO's own. The caller has gated it (like its keyboard shortcut) already."""
    where = item.where()
    if item.ref is None:
        return f"Error: no menu item “{where}” to press."
    if item.pid and is_neo(item.pid):
        return "Error: that's NEO's own menu — NEO never presses its own controls."
    if no := _system_menu_refusal(item.ref):
        return no
    if not item.enabled:
        return f"Error: “{where}” can't be chosen right now (it's greyed out)."
    err = perform_action(item.ref, "AXPress", what=f"choose the menu command {where}")
    if err == 0:
        return f"Chose {where}"
    if err == _CANNOT_COMPLETE:
        return f"Chose {where}, but the app didn't answer in time (busy, or it opened a dialog) — look before choosing again."
    return f"Error: couldn't choose {where} (AX error {err})"


# ---- what is on screen, for pictures and pickers ----------------------------------------------------------

_MARK_ORDER = (
    "AXButton", "AXLink", "AXMenuItem", "AXMenuBarItem", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
    "AXMenuButton", "AXComboBox", "AXTextField", "AXSearchField", "AXTextArea", "AXTab", "AXSlider",
    "AXIncrementor", "AXDisclosureTriangle", "AXRow", "AXCell",
)  # fmt: skip
_NOT_MARKS = {"AXWindow", "AXSheet", "AXDialog", "AXMenu", "AXWebArea", "AXToolbar", "AXTabGroup", "AXSecureTextField"}


@pooled
def visible_elements(
    pid: int | None = None,
    box: tuple[float, float, float, float] | None = None,
    *,
    interactive: bool = True,
    limit: int = 40,
    deadline_s: float = _LOOK_S,
    cancel: threading.Event | None = None,
) -> list[Element]:
    """What is on screen in the app's front window (within ``box``, when given), as elements with
    ids and their on-screen part (``visible``): what's in an open sheet or dialog first, then the
    most actionable, then the biggest. ``interactive``: controls only (plain text too otherwise)."""
    if pid is None:
        _name, pid = working_app()
    if not pid:
        return []
    w = _Walk(deadline_s, cancel)
    rv = _read(_app_el(pid), ("AXFocusedWindow", "AXMainWindow"), w) or {}
    win = rv.get("AXFocusedWindow") or rv.get("AXMainWindow")
    if win is None:
        return []
    els: list[Element] = []
    _walk(win, 1, els, _new_id, box=box, budget=[_MAX_NODES], w=w)
    _issue(els)
    inside = {id(e) for b in w.blockers for e in b.els}
    cands = [
        e for e in els
        if e.visible is not None and e.visible[2] >= 1 and e.visible[3] >= 1 and e.role not in _NOT_MARKS
        and (not interactive or e.role not in _READ_ONLY_ROLES or id(e) in inside)
    ]  # fmt: skip

    def rank(e: Element) -> tuple:
        order = _MARK_ORDER.index(e.role) if e.role in _MARK_ORDER else len(_MARK_ORDER)
        area = e.visible[2] * e.visible[3] if e.visible else 0.0
        return (id(e) not in inside, order, -area)

    return sorted(cands, key=rank)[:limit]


# ---- what a press changed -------------------------------------------------------------------------------


@dataclass
class PressWatch:
    """What a press may change, looked at before it: the scene, the element's own state, and the
    app's notification count (while NEO.app watches the app)."""

    pid: int
    scene: Scene
    state: tuple | None
    gen: int | None


@dataclass
class PressChange:
    changed: bool | None  # None: couldn't tell
    scene_changed: bool
    sheet: str  # a sheet or dialog open after the press (its words: read off the screen)
    said: str  # NEO's own words
    details: str  # words read from the screen (fenced by the caller)


_STATE_ATTRS = ("AXValue", "AXTitle", "AXEnabled", "AXSelected", "AXExpanded", "AXFocused", "AXDescription")


def _state(el) -> tuple | str | None:
    """An element's state now; "gone" when it no longer exists (a dialog's button after it closed)."""
    vals, err = _attrs_err(el, _STATE_ATTRS)
    if err == _INVALID:
        return "gone"
    if err:
        return None
    return tuple(_str(vals.get(a)) if not isinstance(vals.get(a), bool) else vals.get(a) for a in _STATE_ATTRS)


@pooled
def press_watch(e: Element, pid: int) -> PressWatch:
    """The look before a press (a few round trips)."""
    return PressWatch(pid, scene(pid), _state(e.ref) if e.ref is not None else None, _watch_gen(pid))


def _scene_diff(b: Scene, a: Scene) -> tuple[list[str], list[str]]:
    changes, details = [], []
    if not (b.known and a.known):
        return changes, details
    if a.front_pid != b.front_pid:
        changes.append(f"{a.front or 'another app'} is now in front")
    if a.sheet != b.sheet:
        changes.append("a sheet or dialog opened" if a.sheet else "the sheet or dialog closed")
        if a.sheet:
            details.append(f"now showing {a.sheet}")
    if a.title != b.title and (a.title or b.title):
        changes.append("the window changed")
        details.append(f"window: “{a.title[:80]}”")
    if a.focus != b.focus and a.focus:
        changes.append("the keyboard focus moved")
        details.append(f"focus: {a.focus}")
    return changes, details


@pooled
def press_change(
    before: PressWatch, e: Element, *, timeout: float = 0.5, every: float = 0.03, cancel: threading.Event | None = None
) -> PressChange:
    """What a press changed, looked at until the first change (at most ``timeout``): the element's
    own state (a checkbox ticked, a button gone with its dialog), the scene (the front app, the
    window, a sheet, the focus), or — while NEO.app watches the app — any notification it sent."""
    last: list[Any] = [None, None, None]

    def moved() -> bool:
        st = _state(e.ref) if e.ref is not None else None
        sc = scene(before.pid)
        gen = _watch_gen(before.pid)
        last[:] = [st, sc, gen]
        if st is not None and before.state is not None and st != before.state:
            return True
        if _scene_diff(before.scene, sc)[0]:
            return True
        return gen is not None and before.gen is not None and gen != before.gen

    changed = wait.until_sync(moved, timeout, every=every, stop=cancel)
    st, sc, _gen = last
    if sc is None:
        return PressChange(None, False, "", "I couldn't check whether anything changed.", "")
    changes, details = _scene_diff(before.scene, sc)
    if st == "gone":
        changes.insert(0, "the element is gone (its dialog or view closed)")
    elif st is not None and before.state is not None and st != before.state:
        changes.insert(0, "the element changed")
    if changed and not changes:
        changes.append("the app reported a change")
    if changes:
        return PressChange(True, bool(_scene_diff(before.scene, sc)[0]), sc.sheet, "Changed: " + "; ".join(changes) + ".", "; ".join(details))
    known = sc.known and before.scene.known and (st is not None or before.state is None)
    return PressChange(
        False if known else None,
        False,
        sc.sheet,
        f"Nothing visibly changed within {timeout:g}s." if known else "I couldn't tell whether anything changed.",
        "",
    )
