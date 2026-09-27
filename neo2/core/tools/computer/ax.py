"""macOS Accessibility (AX) tree — NEO's primary way of *seeing* the screen.

Fast, private, text-only: no screenshots, no vision model. Each interesting element gets
a short id (``e12``) that the model can click / type into / press via the input tools.

Ids are never reused: every walk hands out new ones, and an id the model holds from an earlier
look still means the element it was issued for (never whatever a newer walk numbered the same).
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


def perform_action(el: Any, action: str, *, kind: str = "accessibility", what: str = "") -> int:
    """AXUIElementPerformAction through the one door: the AX error code (0 = done)."""
    _through("perform", el, action, None)
    effects.guard(kind, what or f"press {action} in another app's UI", via=AS.AXUIElementPerformAction)
    try:
        return AS.AXUIElementPerformAction(el, action)
    except Exception:  # noqa: BLE001 - an element that can't be asked: a failure, not a refusal
        return _ILLEGAL


def set_attribute(el: Any, name: str, value: Any, *, kind: str = "accessibility", what: str = "") -> int:
    """AXUIElementSetAttributeValue through the one door: the AX error code (0 = done)."""
    _through("set", el, name, value)
    effects.guard(kind, what or f"set {name} in another app's UI", via=AS.AXUIElementSetAttributeValue)
    try:
        return AS.AXUIElementSetAttributeValue(el, name, value)
    except Exception:  # noqa: BLE001
        return _ILLEGAL


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
_AX_TIMEOUT_S = 2.0  # a hung app answers each AX call in this long, not the default ~6 s
_CANNOT_COMPLETE = -25204  # kAXErrorCannotComplete: the app didn't answer (busy, hung)


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

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def line(self) -> str:
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
        bits.append(f"@({int(self.x)},{int(self.y)} {int(self.w)}x{int(self.h)})")
        return " ".join(bits)


# The latest walk (what ax_find searches), and every id handed out lately (what an id resolves
# to). A walk that finishes after a newer one started — a slow app's, timed out long ago — adds
# its ids but never replaces the newer tree.
_last_tree: dict[str, Element] = {}
_last_tree_ts = 0.0
_last_tree_app = ""
_last_tree_pid = 0
_issued: dict[str, Element] = {}
_ISSUED_MAX = 4000
_ids = itertools.count(1)
_walks = itertools.count(1)
_latest_walk = 0
_lock = threading.Lock()


def _new_id() -> str:
    return f"e{next(_ids)}"


def is_trusted() -> bool:
    return bool(AS.AXIsProcessTrusted())


def _attr(el, name: str, default=None):
    try:
        err, val = AS.AXUIElementCopyAttributeValue(el, name, None)
    except Exception:  # noqa: BLE001 - an element that can't be asked has nothing to say
        return default
    return val if err == 0 else default


def _bounded(el):
    """The element, answering each AX call within _AX_TIMEOUT_S even when its app hangs."""
    try:
        AS.AXUIElementSetMessagingTimeout(el, _AX_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        pass
    return el


def _system_wide():
    """The system-wide AX element (whatever has the keyboard focus, whatever is under a point).
    macOS takes a timeout set on it as this process's default for every element: as NEO.app that
    is _AX_TIMEOUT_S, like every app element NEO makes. Embedded in another app's process
    (runtime.process_wide_ok) the default stays the host's: what NEO reads through this element
    gets its own timeout instead (_focused, _element_at)."""
    from neo2.core import runtime

    el = AS.AXUIElementCreateSystemWide()
    return _bounded(el) if runtime.process_wide_ok() else el


def _focused():
    """The element with the keyboard focus (None), answering within _AX_TIMEOUT_S."""
    el = _attr(_system_wide(), AS.kAXFocusedUIElementAttribute)
    return None if el is None else _bounded(el)


def _app_el(pid: int):
    """The app's AX element, answering each call within _AX_TIMEOUT_S even when the app hangs."""
    return _bounded(AS.AXUIElementCreateApplication(pid))


def _pair(v, point: bool) -> tuple[float, float] | None:
    if v is None:
        return None
    try:
        ok, p = AS.AXValueGetValue(v, AS.kAXValueCGPointType if point else AS.kAXValueCGSizeType, None)
    except Exception:  # noqa: BLE001 - not a geometry value (some Java/Qt builds)
        return None
    if not ok:
        return None
    return (float(p.x), float(p.y)) if point else (float(p.width), float(p.height))


def _point_size(el) -> tuple[float, float, float, float]:
    pos = _pair(_attr(el, AS.kAXPositionAttribute), True) or (0.0, 0.0)
    size = _pair(_attr(el, AS.kAXSizeAttribute), False) or (0.0, 0.0)
    return pos[0], pos[1], size[0], size[1]


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


# Transient system UI that can be "frontmost" for a moment without being what the user works in.
_TRANSIENT = {
    "UserNotificationCenter",
    "NotificationCenter",
    "Notification Center",
    "Control Center",
    "Spotlight",
    "loginwindow",
    "Dock",
}


@pooled
def frontmost_app() -> tuple[str, int]:
    """The app the user is working in. A passing notification that happens to be frontmost is
    skipped for the menu-bar app — but a system dialog the user is actually in (it has a
    focused window) is kept."""
    from neo2.core.tools.computer.focus import front_app

    name, pid = front_app()  # always current, unlike NSWorkspace.frontmostApplication
    if name in _TRANSIENT or not pid:
        has_window = bool(pid) and _attr(AS.AXUIElementCreateApplication(pid), AS.kAXFocusedWindowAttribute) is not None
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


@pooled
def pid_for_app(name: str) -> int | None:
    """The app's pid. An exact name wins over a substring match, and a real app (menu bar,
    windows) wins over a helper process that shares the name — "Notes" must not resolve to
    "LinkedNotesUIService", whose AX tree is empty. Asked fresh, so apps launched after NEO
    started are found too."""
    from AppKit import NSRunningApplication

    from neo2.core.tools.computer.focus import pid_running

    name_l = name.lower()
    best: tuple[int, int] | None = None  # (rank, pid) — lower rank is better
    fresh = [NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) for _, pid in pid_running(name_l)]
    for a in [x for x in fresh if x is not None] or NSWorkspace.sharedWorkspace().runningApplications():
        n = (a.localizedName() or "").lower()
        if not n or (n != name_l and name_l not in n):
            continue
        rank = (0 if n == name_l else 2) + (0 if a.activationPolicy() == 0 else 1)
        if best is None or rank < best[0]:
            best = (rank, int(a.processIdentifier()))
    return best[1] if best else None


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


def _walk(
    el, depth: int, out: list[Element], new_id, *, box=None, budget: list[int] | None = None, skip=None, clip: bool = True
) -> None:
    """List the interesting elements under ``el``, depth first. ``box``: the visible part of the
    screen here (the window, the scroll area); whatever lies wholly outside it is scrolled out of
    view and skipped with everything under it (``clip=False``: nothing is). One odd element never
    fails the walk."""
    if depth > _MAX_DEPTH or len(out) >= _MAX_ELEMENTS:
        return
    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            return
    if skip is not None and _same(el, skip):
        return
    try:
        role = _str(_attr(el, AS.kAXRoleAttribute))
        if role in _SKIP_ROLES:
            return
        if role in _FLOATING or not clip:
            box = None
        x, y, w, h = _point_size(el)
        if _outside((x, y, w, h), box):
            return
        if clip and role in _CLIPS and w > 0 and h > 0:
            box = _clip(box, (x, y, w, h))
        title = _str(_attr(el, AS.kAXTitleAttribute)) or _str(_attr(el, AS.kAXDescriptionAttribute))
        if not title:
            title = _str(_attr(el, "AXPlaceholderValue")) or _str(_attr(el, "AXLabel"))
        value = _str(_attr(el, AS.kAXValueAttribute))
        interesting = role in _INTERACTIVE and (title or value or role in _ALWAYS_SHOW) and w > 0 and h > 0
        if role in ("AXWindow", "AXSheet", "AXDialog"):
            interesting = True
        elif role == "AXMenu":
            interesting = w > 0 and h > 0  # an open menu; a closed one is noise
        if interesting:
            try:
                err, actions = AS.AXUIElementCopyActionNames(el, None)
            except Exception:  # noqa: BLE001
                err, actions = 1, None
            out.append(
                Element(
                    id=new_id(),
                    role=role,
                    title=title,
                    value=value,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    enabled=bool(_attr(el, AS.kAXEnabledAttribute, True)),
                    focused=bool(_attr(el, AS.kAXFocusedAttribute, False)),
                    actions=[str(a) for a in (actions or [])] if err == 0 else [],
                    ref=el,
                )
            )
        children = _attr(el, AS.kAXChildrenAttribute) or []
    except Exception:  # noqa: BLE001 - skip this one element, keep the tree
        return
    for c in children:
        _walk(c, depth + 1, out, new_id, box=box, budget=budget, skip=skip, clip=clip)


# Chromium and Electron apps show their web content (a page, Slack's messages, VS Code's editor)
# to Accessibility only once asked. NEO asks when it looks at one, and keeps the switch on while the
# ids it handed out may still be used — then puts it back (neo2.core.decide.axkit counts every flow).
_WEB_HOLD_S = 120.0
_WEB_SETTLE_S = 0.4
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


def _lines_budget() -> int:
    """Characters of tree that reach the model whole (the agent loop cuts longer results)."""
    try:
        from neo2.core.config import settings

        return max(1500, int(settings().max_tool_result_chars) - 400)
    except Exception:  # noqa: BLE001
        return 5600


@pooled
def snapshot(app: str | None = None, *, max_lines: int = 250) -> str:
    """Return a compact text tree of the frontmost (or named) app's UI: its focused window first,
    what's in view only, controls ahead of plain text when it's too big to show whole."""
    global _last_tree, _last_tree_ts, _last_tree_app, _last_tree_pid, _latest_walk
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
    err, _role = AS.AXUIElementCopyAttributeValue(root, AS.kAXRoleAttribute, None)
    if err == _CANNOT_COMPLETE:
        return f"Error: {name} isn't responding to Accessibility right now (busy or hung) — try again in a moment."
    if _expose_web(int(pid), root):
        time.sleep(_WEB_SETTLE_S)
    els: list[Element] = []
    budget = [_MAX_NODES]
    win = _attr(root, AS.kAXFocusedWindowAttribute)
    if win is not None:  # the window the user works in first: a big sidebar can't crowd it out
        _walk(win, 1, els, _new_id, budget=budget)
        if len(els) <= 1:
            # Nothing but the window itself in view: an app whose frames don't add up (some Java or
            # Qt builds) — look again without cutting anything to the window.
            els, budget = [], [_MAX_NODES]
            _walk(win, 1, els, _new_id, budget=budget, clip=False)
    _walk(root, 0, els, _new_id, budget=budget, skip=win)
    with _lock:
        for e in els:
            _issued[e.id] = e
        while len(_issued) > _ISSUED_MAX:
            _issued.pop(next(iter(_issued)))
        if walk > _latest_walk:
            _latest_walk = walk
            _last_tree = {e.id: e for e in els}
            _last_tree_ts = time.time()
            _last_tree_app = name.lower()
            _last_tree_pid = int(pid or 0)
    wtitle = _str(_attr(win, AS.kAXTitleAttribute)) if win is not None else ""
    head = f"App: {name}" + (f' — window "{wtitle}"' if wtitle else "")
    shown = els[:max_lines]
    left = _lines_budget() - len(head)
    if sum(len(e.line()) + 3 for e in shown) > left:
        # Too much for one look: what can be acted on first, in screen order within each group.
        order = sorted(range(len(shown)), key=lambda i: (shown[i].role in _READ_ONLY_ROLES, i))
        keep: set[int] = set()
        for i in order:
            cost = len(shown[i].line()) + 3
            if cost > left:
                continue
            keep.add(i)
            left -= cost
        shown = [e for i, e in enumerate(shown) if i in keep]
    lines = [head] + ["  " + e.line() for e in shown]
    if len(els) > len(shown):
        lines.append(f"  … {len(els) - len(shown)} more elements (use ax_find to search)")
    if budget[0] < 0:
        lines.append("  (the walk stopped early: this window is very large)")
    return "\n".join(lines)


@pooled
def find(query: str, app: str | None = None) -> str:
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
        out = snapshot(app)
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
    return looked + "\n".join(e.line() for e in hits[:40])


def get(eid: str) -> Element | None:
    """The element an id was issued for (from the latest look, or an earlier one)."""
    return _last_tree.get(eid) or _issued.get(eid)


# AXPress and its plain equivalents. Anything else an element offers (AXConfirm — a Return that
# sends —, AXIncrement, AXShowMenu, AXCancel…) is not a "press".
_PRESS_ACTIONS = ("AXPress", "AXPick", "AXOpen")
_UNSUPPORTED = (-25206, -25208, -25200)  # action unsupported, not implemented, plain failure


def press_action(e: Element) -> str:
    return next((a for a in _PRESS_ACTIONS if a in e.actions), "")


@pooled
def press_el(e: Element) -> str:
    """Press the element in hand."""
    action = press_action(e)
    what = f"{e.id} ({e.role.removeprefix('AX')} {e.title!r})"
    if not action:
        offers = ", ".join(a for a in e.actions) or "none"
        return f"Error: {e.id} can't be pressed (its actions: {offers}) — click it by id instead."
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
    return "AXSecureTextField" in (_str(_attr(el, AS.kAXRoleAttribute)), _str(_attr(el, AS.kAXSubroleAttribute)))


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
    return _str(_attr(el, AS.kAXRoleAttribute)), _str(_attr(el, AS.kAXSubroleAttribute))


def focused_is_secure() -> bool:
    """Is the caret in a password field? NEO never types there."""
    role, sub = focused_role()
    return "AXSecureTextField" in (role, sub)


def focused_editable() -> bool:
    """Is the keyboard focus in something you can type into (never a password field)?"""
    el = _focused()
    if el is None:
        return False
    role = _str(_attr(el, AS.kAXRoleAttribute))
    if "AXSecureTextField" in (role, _str(_attr(el, AS.kAXSubroleAttribute))):
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
    root = _app_el(pid)
    win = _attr(root, AS.kAXFocusedWindowAttribute) or _attr(root, AS.kAXMainWindowAttribute)
    return name, win


@pooled
def window_elements(app: str | None = None) -> tuple[str, tuple[float, float, float, float] | None, list[Element]]:
    """Elements of the focused window only (not every window of the app), plus its frame."""
    name, win = focused_window(app)
    if win is None:
        return name, None, []
    pid = pid_of(win)
    if pid and _expose_web(pid, _app_el(pid)):
        time.sleep(_WEB_SETTLE_S)
    els: list[Element] = []
    n = itertools.count(1)
    _walk(win, 0, els, lambda: f"w{next(n)}", budget=[_MAX_NODES])
    return name, _point_size(win), els


def _text_areas(el, depth: int, out: list[Element]) -> None:
    """Every text area under `el` — its own walk, so a busy sidebar can't hit an element cap."""
    if depth > _MAX_DEPTH or len(out) > 50:
        return
    role = _str(_attr(el, AS.kAXRoleAttribute))
    if role == "AXTextArea":
        x, y, w, h = _point_size(el)
        out.append(
            Element(
                id=f"t{len(out)}",
                role=role,
                title="",
                value=_str(_attr(el, AS.kAXValueAttribute)),
                x=x,
                y=y,
                w=w,
                h=h,
                enabled=bool(_attr(el, AS.kAXEnabledAttribute, True)),
                focused=bool(_attr(el, AS.kAXFocusedAttribute, False)),
                ref=el,
            )
        )
        return
    for c in _attr(el, AS.kAXChildrenAttribute) or []:
        _text_areas(c, depth + 1, out)


@pooled
def main_text_area(app: str | None = None) -> Element | None:
    """The document body of the focused window: the focused text area, else the biggest one.
    None when that window shows no editor (e.g. Notes with no note selected)."""
    _, win = focused_window(app)
    if win is None:
        return None
    areas: list[Element] = []
    _text_areas(win, 0, areas)
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
    root = _app_el(pid)
    win = _attr(root, AS.kAXFocusedWindowAttribute) or _attr(root, AS.kAXMainWindowAttribute)
    if win is None:
        return None
    f = _point_size(win)
    return f if f[2] > 0 and f[3] > 0 else None


def visible_frame(el) -> tuple[float, float, float, float] | None:
    """The part of an element that is on screen: its frame cut to the scroll areas and window
    around it (a text area or a web area reports its whole document). None when none of it is."""
    box = _point_size(el)
    if box[2] <= 0 or box[3] <= 0:
        return None
    p = el
    for _ in range(12):
        p = _attr(p, "AXParent")
        if p is None:
            break
        role = _str(_attr(p, AS.kAXRoleAttribute))
        if role in _FLOATING:
            break  # an open menu or popover floats over its window: the window doesn't cut it
        if role in _CLIPS:
            clip = _point_size(p)
            if clip[2] > 0 and clip[3] > 0:
                box = _clip(box, clip)
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
    err = perform_action(e.ref, action)
    return "Pressed" if err == 0 else f"Error: AX action failed ({err})"


def focused_element_text() -> str:
    el = _focused()
    if el is None:
        return ""
    return _str(_attr(el, AS.kAXValueAttribute)) or _str(_attr(el, "AXSelectedText"))


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
    if pid and _expose_web(pid, _app_el(pid)):
        time.sleep(_WEB_SETTLE_S)
    return is_terminal(_focused())


def is_terminal(el) -> bool:
    """Does this element, or the pane around it, say it is a terminal view?"""
    for _ in range(2):
        if el is None:
            return False
        said = " ".join(_str(_attr(el, a, "")) for a in ("AXDescription", "AXTitle", "AXHelp"))
        if _TERMINAL.search(said):
            return True
        el = _attr(el, "AXParent")
        el = None if el is None else _bounded(el)
    return False


def _element_at(x: float, y: float):
    try:
        err, el = AS.AXUIElementCopyElementAtPosition(_system_wide(), float(x), float(y), None)
    except Exception:  # noqa: BLE001
        return None
    if err or el is None:
        return None
    return _bounded(el)  # its app answers the questions that follow briefly, even when hung


def _label_of(el) -> str:
    parts: list[str] = []
    for _ in range(2):
        for attr in ("AXTitle", "AXDescription", "AXValue"):
            v = _str(_attr(el, attr, ""))
            if v:
                parts.append(v)
                break
        el = _attr(el, "AXParent")
        if el is None:
            break
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


def _words(el, limit: int = 3) -> tuple[list[str], list[str]]:
    """(texts, button titles) inside a sheet or dialog: what it says and what it offers."""
    texts: list[str] = []
    buttons: list[str] = []
    todo, seen = [(el, 0)], 0
    while todo and seen < 120:
        node, d = todo.pop(0)
        seen += 1
        role = _str(_attr(node, AS.kAXRoleAttribute))
        if role == "AXButton":
            t = _str(_attr(node, AS.kAXTitleAttribute)) or _str(_attr(node, AS.kAXDescriptionAttribute))
            if t and len(buttons) < 6:
                buttons.append(t[:30])
        elif role in ("AXStaticText", "AXTextArea") and len(texts) < limit:
            t = _str(_attr(node, AS.kAXValueAttribute)) or _str(_attr(node, AS.kAXTitleAttribute))
            if t.strip():
                texts.append(" ".join(t.split())[:80])
        if d < 6:
            todo.extend((c, d + 1) for c in (_attr(node, AS.kAXChildrenAttribute) or []))
    return texts, buttons


def summary(el) -> str:
    """A sheet or dialog in a few words: “Start a New Game?” (buttons: Start, Cancel)."""
    title = _str(_attr(el, AS.kAXTitleAttribute))
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
    stop = time.monotonic() + _WORDS_BUDGET_S  # a busy app answers each question slowly: enough is enough
    for _ in range(8):
        if el is None or len(said) >= 16 or time.monotonic() > stop:
            break
        said += [v[:80] for a in ("AXDescription", "AXHelp") if (v := _str(_attr(el, a, "")).strip())]
        if _str(_attr(el, AS.kAXRoleAttribute)) in ("AXWindow", "AXSheet"):
            said.append(_str(_attr(el, AS.kAXTitleAttribute)))
            break
        for child in (_attr(el, AS.kAXChildrenAttribute) or [])[:30]:
            if time.monotonic() > stop:
                break
            if _str(_attr(child, AS.kAXRoleAttribute)) == "AXStaticText":
                text = _str(_attr(child, AS.kAXValueAttribute)) or _str(_attr(child, AS.kAXTitleAttribute))
                if text.strip():
                    said.append(" ".join(text.split())[:80])
        el = _attr(el, "AXParent")
    if h.el is None and h.window is not None:
        said.append(_str(_attr(h.window, AS.kAXTitleAttribute)))
    return " ".join(w for w in said if w)[:600]


def _shown(el) -> bool:
    _x, _y, w, h = _point_size(el)
    return w > 0 and h > 0


def _sheet_of(win) -> Any:
    """The sheet open on a window (one with a size: a closed one lingering in the tree isn't)."""
    return next(
        (
            c
            for c in (_attr(win, AS.kAXChildrenAttribute) or [])
            if _str(_attr(c, AS.kAXRoleAttribute)) == "AXSheet" and _shown(c)
        ),
        None,
    )


def regular(pid: int) -> bool:
    """An app with windows and a menu bar (not the Dock, a menu-bar extra, a background agent)."""
    app = _running(pid)
    try:
        return app is not None and int(app.activationPolicy()) == 0
    except Exception:  # noqa: BLE001
        return False


@pooled
def hit(x: float, y: float) -> Hit:
    """Who owns a screen point and what's there — one Accessibility lookup for the gate, for
    bringing the right app forward, for keeping off NEO's own window and for the report."""
    el = _element_at(x, y)
    if el is None:  # an app that shows nothing to Accessibility there (a game drawn with Metal)
        name, pid = window_owner_at(x, y) or ("", 0)  # (can't say: unnamed, and the gates fail closed)
        return Hit(pid=pid, app=name, regular=regular(pid)) if pid else Hit()
    pid = pid_of(el)
    app = _running(pid)
    h = Hit(
        pid=pid,
        app=str(app.localizedName() or "") if app is not None else "",
        label=_label_of(el),
        role=_str(_attr(el, AS.kAXRoleAttribute)),
        regular=app is not None and int(app.activationPolicy()) == 0,
        el=el,
    )
    win = _attr(el, "AXWindow")
    h.window = win
    if win is not None and _str(_attr(win, AS.kAXRoleAttribute)) != "AXSheet":
        sheet = _sheet_of(win)
        if sheet is not None:
            sx, sy, sw, sh = _point_size(sheet)
            if not (sx <= x < sx + sw and sy <= y < sy + sh):
                h.sheet = summary(sheet)
        elif pid:
            # An app-modal dialog (an alert in its own window) blocks every other window of the app.
            focused = _attr(_app_el(pid), AS.kAXFocusedWindowAttribute)
            if focused is not None and not _same(focused, win) and _attr(focused, "AXModal") is True and _shown(focused):
                h.sheet = summary(focused)
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


_DIALOG_SUBROLES = {"AXDialog", "AXSystemDialog"}


@pooled
def scene(pid: int) -> Scene:
    from neo2.core.tools.computer.focus import front_app

    try:
        name, fpid = front_app()
    except Exception:  # noqa: BLE001
        return Scene()
    s = Scene(front=name, front_pid=fpid, known=True)
    if not pid:
        return s
    root = _app_el(pid)
    win = _attr(root, AS.kAXFocusedWindowAttribute)
    if win is not None:
        s.title = _str(_attr(win, AS.kAXTitleAttribute))
        sheet = _sheet_of(win)
        if sheet is not None:
            s.sheet = "a sheet " + summary(sheet)
        elif _str(_attr(win, AS.kAXSubroleAttribute)) in _DIALOG_SUBROLES:
            s.sheet = "a dialog " + summary(win)
    f = _attr(root, AS.kAXFocusedUIElementAttribute)
    if f is not None:
        role = _str(_attr(f, AS.kAXRoleAttribute)).removeprefix("AX")
        label = _str(_attr(f, AS.kAXTitleAttribute)) or _str(_attr(f, AS.kAXDescriptionAttribute))
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


@dataclass(frozen=True)
class Win:
    """A window on screen as the window server has it: the process that owns it, its layer (0: a
    normal app window — the menu bar, status items, the Dock, menus, Notification Center, Control
    Center and Spotlight are all on other layers) and its frame (points, from the top left)."""

    pid: int
    owner: str
    layer: int
    frame: tuple[float, float, float, float]


@pooled
def windows_at(x: float, y: float) -> list[Win] | None:
    """Every on-screen window under a screen point, front to back, from the window server (None:
    it couldn't say)."""
    try:
        import Quartz as Q

        wins = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        )
    except Exception:  # noqa: BLE001
        return None
    if wins is None:
        return None
    out = []
    for w in wins:
        b = w.get("kCGWindowBounds") or {}
        wx, wy, ww, wh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if ww <= 0 or wh <= 0 or float(w.get("kCGWindowAlpha", 1) or 0) <= 0:
            continue
        if wx <= x < wx + ww and wy <= y < wy + wh:
            layer = int(w.get("kCGWindowLayer", 1))  # (no layer given: not taken for a normal window)
            out.append(Win(int(w.get("kCGWindowOwnerPID", 0) or 0), str(w.get("kCGWindowOwnerName", "") or ""), layer, (wx, wy, ww, wh)))
    return out


@pooled
def key_window(pid: int) -> Any:
    """The app's key window: its focused window while it is the active app. None when it isn't
    active — its own word (AXFrontmost), whatever LaunchServices calls frontmost — or no window of
    it is key."""
    app = _app_el(pid)
    if _attr(app, "AXFrontmost") is False:
        return None
    return _attr(app, AS.kAXFocusedWindowAttribute)


def _on(win) -> Any:
    """A sheet counts as the window it is on."""
    return _attr(win, "AXParent") if win is not None and _str(_attr(win, AS.kAXRoleAttribute)) == "AXSheet" else win


@pooled
def is_key(h: Hit) -> bool:
    """Is the window under a point its app's key window, with the app active — so a click there
    reaches the view instead of only activating the window?"""
    if h.window is None or not h.pid:
        return False
    return _same(_on(key_window(h.pid)), _on(h.window))


@pooled
def same_window(a: Hit, b: Hit) -> bool | None:
    """Are two points in the same window (a sheet counts as the window it is on)? None: one of them
    has no window to tell by."""
    if a.window is None or b.window is None:
        return None
    return _same(_on(a.window), _on(b.window))


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
