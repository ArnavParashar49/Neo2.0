"""macOS Accessibility (AX) tree — NEO's primary way of *seeing* the screen.

Fast, private, text-only: no screenshots, no vision model. Each interesting element gets
a short id (``e12``) that the model can click / type into / press via the input tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import ApplicationServices as AS
from AppKit import NSWorkspace

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
_FIND_MAX_AGE_S = 30.0  # ax_find reuses the last tree (and its ids) unless it is older than this
_MAX_ELEMENTS = 400
_MAX_DEPTH = 40


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


_last_tree: dict[str, Element] = {}
_last_tree_ts = 0.0
_last_tree_app = ""


def is_trusted() -> bool:
    return bool(AS.AXIsProcessTrusted())


def _attr(el, name: str, default=None):
    err, val = AS.AXUIElementCopyAttributeValue(el, name, None)
    return val if err == 0 else default


def _point_size(el) -> tuple[float, float, float, float]:
    pos = _attr(el, AS.kAXPositionAttribute)
    size = _attr(el, AS.kAXSizeAttribute)
    x = y = w = h = 0.0
    if pos is not None:
        ok, p = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        if ok:
            x, y = p.x, p.y
    if size is not None:
        ok, s = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        if ok:
            w, h = s.width, s.height
    return x, y, w, h


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return ""


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


def frontmost_app() -> tuple[str, int]:
    """The app the user is working in. A passing notification that happens to be frontmost is
    skipped for the menu-bar app — but a system dialog the user is actually in (it has a
    focused window) is kept."""
    from neo.tools.computer.focus import front_app

    name, pid = front_app()  # always current, unlike NSWorkspace.frontmostApplication
    if name in _TRANSIENT or not pid:
        has_window = bool(pid) and _attr(AS.AXUIElementCreateApplication(pid), AS.kAXFocusedWindowAttribute) is not None
        if not has_window:
            app = NSWorkspace.sharedWorkspace().menuBarOwningApplication()
            if app is not None:
                return app.localizedName() or "", int(app.processIdentifier())
    return name, pid


def running_apps() -> list[str]:
    """Normal windowed apps, asked fresh from LaunchServices."""
    from neo.tools.computer.focus import running

    names = sorted({n for n, _, t in running() if t == "Foreground" and n})
    if names:
        return names
    out: list[str] = []
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if a.activationPolicy() == 0 and a.localizedName():  # regular (has a UI)
            out.append(a.localizedName())
    return sorted(set(out))


def pid_for_app(name: str) -> int | None:
    """The app's pid. An exact name wins over a substring match, and a real app (menu bar,
    windows) wins over a helper process that shares the name — "Notes" must not resolve to
    "LinkedNotesUIService", whose AX tree is empty. Asked fresh, so apps launched after NEO
    started are found too."""
    from AppKit import NSRunningApplication

    from neo.tools.computer.focus import pid_running

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


def _walk(el, depth: int, out: list[Element], counter: list[int]) -> None:
    if depth > _MAX_DEPTH or len(out) >= _MAX_ELEMENTS:
        return
    role = _str(_attr(el, AS.kAXRoleAttribute))
    if role in _SKIP_ROLES:
        return
    title = _str(_attr(el, AS.kAXTitleAttribute)) or _str(_attr(el, AS.kAXDescriptionAttribute))
    if not title:
        title = _str(_attr(el, "AXPlaceholderValue")) or _str(_attr(el, "AXLabel"))
    value = _str(_attr(el, AS.kAXValueAttribute))
    x, y, w, h = _point_size(el)
    interesting = role in _INTERACTIVE and (title or value or role in _ALWAYS_SHOW) and w > 0 and h > 0
    if role in ("AXWindow", "AXSheet", "AXDialog", "AXMenu"):
        interesting = True
    if interesting:
        counter[0] += 1
        eid = f"e{counter[0]}"
        err, actions = AS.AXUIElementCopyActionNames(el, None)
        out.append(
            Element(
                id=eid,
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
    for c in children:
        _walk(c, depth + 1, out, counter)


def snapshot(app: str | None = None, *, max_lines: int = 250) -> str:
    """Return a compact text tree of the frontmost (or named) app's UI."""
    global _last_tree, _last_tree_ts, _last_tree_app
    if not is_trusted():
        return "NEEDS_USER: Accessibility permission is off. Enable it for this app in System Settings → Privacy & Security → Accessibility, then try again."
    if app:
        pid = pid_for_app(app)
        if pid is None:
            return f"Error: app '{app}' is not running."
        name = app
    else:
        name, pid = frontmost_app()
    root = AS.AXUIElementCreateApplication(pid)
    els: list[Element] = []
    _walk(root, 0, els, [0])
    _last_tree = {e.id: e for e in els}
    _last_tree_ts = time.time()
    _last_tree_app = name.lower()
    win = _attr(root, AS.kAXFocusedWindowAttribute)
    wtitle = _str(_attr(win, AS.kAXTitleAttribute)) if win is not None else ""
    lines = [f"App: {name}" + (f' — window "{wtitle}"' if wtitle else "")]
    for e in els[:max_lines]:
        lines.append("  " + e.line())
    if len(els) > max_lines:
        lines.append(f"  … {len(els) - max_lines} more elements (use ax_find to search)")
    return "\n".join(lines)


def find(query: str, app: str | None = None) -> str:
    """Elements whose title/value/role contains `query` (case-insensitive).

    Searches the tree the model is already holding ids for; a fresh walk (new ids) only when
    that tree is stale or belongs to a different app than asked for."""
    wanted = (app or frontmost_app()[0]).lower()
    other_app = _last_tree_app not in (wanted, "")
    if not _last_tree or other_app or time.time() - _last_tree_ts > _FIND_MAX_AGE_S:
        snapshot(app)
    q = query.lower()
    hits = [
        e
        for e in _last_tree.values()
        if q in e.title.lower() or q in e.value.lower() or q in e.role.lower()
    ]
    if not hits:
        return f"No element matching '{query}' (tree from {int(time.time() - _last_tree_ts)}s ago)."
    return "\n".join(e.line() for e in hits[:40])


def get(eid: str) -> Element | None:
    return _last_tree.get(eid)


def press(eid: str) -> str:
    e = get(eid)
    if not e:
        return f"Error: unknown element {eid} — ids come from the latest ax_tree; call it again."
    action = "AXPress" if "AXPress" in e.actions else (e.actions[0] if e.actions else "")
    if not action:
        return f"Error: {eid} has no actions; try clicking its center {tuple(map(int, e.center))}."
    err = AS.AXUIElementPerformAction(e.ref, action)
    return (
        f"Pressed {eid} ({e.role.removeprefix('AX')} {e.title!r})"
        if err == 0
        else f"Error: AX action failed ({err})"
    )


def set_value(eid: str, value: str) -> str:
    e = get(eid)
    if not e:
        return f"Error: unknown element {eid} — ids come from the latest ax_tree; call it again."
    err = AS.AXUIElementSetAttributeValue(e.ref, AS.kAXValueAttribute, value)
    return (
        f"Set {eid} value."
        if err == 0
        else f"Error: could not set value ({err}); try focusing it and typing."
    )


_EDITABLE = {"AXTextArea", "AXTextField", "AXSearchField", "AXComboBox"}
_SINGLE_LINE = {"AXTextField", "AXSearchField", "AXComboBox"}


def focused_role() -> tuple[str, str]:
    """(role, subrole) of whatever has the keyboard focus, or ("", "")."""
    el = _attr(AS.AXUIElementCreateSystemWide(), AS.kAXFocusedUIElementAttribute)
    if el is None:
        return "", ""
    return _str(_attr(el, AS.kAXRoleAttribute)), _str(_attr(el, AS.kAXSubroleAttribute))


def focused_is_secure() -> bool:
    """Is the caret in a password field? NEO never types there."""
    role, sub = focused_role()
    return "AXSecureTextField" in (role, sub)


def focused_editable() -> bool:
    """Is the keyboard focus in something you can type into (never a password field)?"""
    el = _attr(AS.AXUIElementCreateSystemWide(), AS.kAXFocusedUIElementAttribute)
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


def focused_window(app: str | None = None):
    """(app name, AX window element) of the frontmost (or named) app's focused window."""
    if app:
        pid, name = pid_for_app(app), app
    else:
        name, pid = frontmost_app()
    if not pid:
        return name, None
    root = AS.AXUIElementCreateApplication(pid)
    win = _attr(root, AS.kAXFocusedWindowAttribute) or _attr(root, AS.kAXMainWindowAttribute)
    return name, win


def window_elements(app: str | None = None) -> tuple[str, tuple[float, float, float, float] | None, list[Element]]:
    """Elements of the focused window only (not every window of the app), plus its frame."""
    name, win = focused_window(app)
    if win is None:
        return name, None, []
    els: list[Element] = []
    _walk(win, 0, els, [0])
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


def press_element(e: Element) -> str:
    """Press the element in hand (not by id through the shared tree another call may replace)."""
    action = "AXPress" if "AXPress" in e.actions else ""
    if not action:
        return "Error: no press action"
    err = AS.AXUIElementPerformAction(e.ref, action)
    return "Pressed" if err == 0 else f"Error: AX action failed ({err})"


def focused_element_text() -> str:
    sys_el = AS.AXUIElementCreateSystemWide()
    el = _attr(sys_el, AS.kAXFocusedUIElementAttribute)
    if el is None:
        return ""
    return _str(_attr(el, AS.kAXValueAttribute)) or _str(_attr(el, "AXSelectedText"))
