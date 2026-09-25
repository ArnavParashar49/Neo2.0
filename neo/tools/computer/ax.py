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


def frontmost_app() -> tuple[str, int]:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return (app.localizedName() or "", int(app.processIdentifier())) if app else ("", 0)


def running_apps() -> list[str]:
    out = []
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if a.activationPolicy() == 0 and a.localizedName():  # regular (has a UI)
            out.append(a.localizedName())
    return sorted(set(out))


def pid_for_app(name: str) -> int | None:
    name_l = name.lower()
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        n = (a.localizedName() or "").lower()
        if n == name_l or name_l in n:
            return int(a.processIdentifier())
    return None


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
    interesting = role in _INTERACTIVE and (title or value) and w > 0 and h > 0
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
    global _last_tree, _last_tree_ts
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
    win = _attr(root, AS.kAXFocusedWindowAttribute)
    wtitle = _str(_attr(win, AS.kAXTitleAttribute)) if win is not None else ""
    lines = [f"App: {name}" + (f' — window "{wtitle}"' if wtitle else "")]
    for e in els[:max_lines]:
        lines.append("  " + e.line())
    if len(els) > max_lines:
        lines.append(f"  … {len(els) - max_lines} more elements (use ax_find to search)")
    return "\n".join(lines)


def find(query: str, app: str | None = None) -> str:
    """Elements whose title/value contains `query` (case-insensitive)."""
    if not _last_tree or time.time() - _last_tree_ts > 5:
        snapshot(app)
    q = query.lower()
    hits = [e for e in _last_tree.values() if q in e.title.lower() or q in e.value.lower()]
    if not hits:
        return f"No element matching '{query}'."
    return "\n".join(e.line() for e in hits[:40])


def get(eid: str) -> Element | None:
    return _last_tree.get(eid)


def press(eid: str) -> str:
    e = get(eid)
    if not e:
        return f"Error: unknown element {eid} — call ax_tree first."
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
        return f"Error: unknown element {eid} — call ax_tree first."
    err = AS.AXUIElementSetAttributeValue(e.ref, AS.kAXValueAttribute, value)
    return (
        f"Set {eid} value."
        if err == 0
        else f"Error: could not set value ({err}); try focusing it and typing."
    )


def focused_element_text() -> str:
    sys_el = AS.AXUIElementCreateSystemWide()
    el = _attr(sys_el, AS.kAXFocusedUIElementAttribute)
    if el is None:
        return ""
    return _str(_attr(el, AS.kAXValueAttribute)) or _str(_attr(el, "AXSelectedText"))
