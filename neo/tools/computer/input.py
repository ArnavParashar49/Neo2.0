"""Mouse + keyboard injection via Quartz CGEvents. Coordinates are global screen *points* (the
main display's top-left is 0,0; other displays sit at their own offsets, negative ones too) — the
space the AX tree uses. A screenshot's pixels map onto it by the transform its note gives."""

from __future__ import annotations

import math
import re
import time

import Quartz as Q

_KEYCODES = {
    "a": 0,
    "s": 1,
    "d": 2,
    "f": 3,
    "h": 4,
    "g": 5,
    "z": 6,
    "x": 7,
    "c": 8,
    "v": 9,
    "b": 11,
    "q": 12,
    "w": 13,
    "e": 14,
    "r": 15,
    "y": 16,
    "t": 17,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "6": 22,
    "5": 23,
    "=": 24,
    "9": 25,
    "7": 26,
    "-": 27,
    "8": 28,
    "0": 29,
    "]": 30,
    "o": 31,
    "u": 32,
    "[": 33,
    "i": 34,
    "p": 35,
    "l": 37,
    "j": 38,
    "'": 39,
    "k": 40,
    ";": 41,
    "\\": 42,
    ",": 43,
    "/": 44,
    "n": 45,
    "m": 46,
    ".": 47,
    "`": 50,
    "space": 49,
    "return": 36,
    "enter": 36,
    "tab": 48,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "forwarddelete": 117,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
    "capslock": 57,
}
_MODS = {
    "cmd": Q.kCGEventFlagMaskCommand,
    "shift": Q.kCGEventFlagMaskShift,
    "alt": Q.kCGEventFlagMaskAlternate,
    "ctrl": Q.kCGEventFlagMaskControl,
    "fn": Q.kCGEventFlagMaskSecondaryFn,
}
# Every way a modifier gets written ("opt+left", "ctl+c", "⌘⇧t", "super+space").
_MOD_ALIASES = {
    "cmd": "cmd", "command": "cmd", "meta": "cmd", "super": "cmd", "win": "cmd", "⌘": "cmd",
    "shift": "shift", "⇧": "shift",
    "alt": "alt", "option": "alt", "opt": "alt", "optn": "alt", "⌥": "alt",
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl", "⌃": "ctrl",
    "fn": "fn", "function": "fn",
}  # fmt: skip
_KEY_ALIASES = {"plus": "+", "minus": "-", "equals": "=", "esc": "escape", "enter": "return", "del": "delete"}
# A symbol typed with shift on the (US) key that carries it: ⌘⇧} is ⌘⇧], "cmd+?" is ⌘⇧/.
_SHIFTED = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}  # fmt: skip


def _post(ev) -> None:
    Q.CGEventPost(Q.kCGHIDEventTap, ev)


def can_post() -> bool:
    """May this process post input events? macOS checks that on its own (the "control your
    computer" part of Accessibility) and drops events silently when it's missing — so NEO asks
    first, and never reports a click that never arrived."""
    f = getattr(Q, "CGPreflightPostEventAccess", None)
    if f is None:
        return True
    try:
        return bool(f())
    except Exception:  # noqa: BLE001 - can't tell: let the events go
        return True


def _num(v, name: str = "coordinate") -> float:
    """A coordinate as a number ("612" too), or ValueError — checked before any event is posted."""
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {v!r}") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number")
    return f


def mods_flags(mods) -> int:
    """Event flags for modifier names (["cmd", "shift"], "cmd+shift"); ValueError on anything else."""
    if isinstance(mods, str):
        mods = [m for m in re.split(r"[+,\s]+", mods) if m]
    flags = 0
    for m in mods or ():
        name = _MOD_ALIASES.get(str(m).strip().lower())
        if name is None:
            raise ValueError(f"unknown modifier {m!r}")
        flags |= _MODS[name]
    return flags


def mouse_position() -> tuple[float, float]:
    loc = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return float(loc.x), float(loc.y)


def move(x: float, y: float) -> None:
    _post(Q.CGEventCreateMouseEvent(None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft))


def _buttons(button: str):
    if button == "right":
        return Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight
    if button == "middle":
        return Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGMouseButtonCenter
    return Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft


_COUNT_WORDS = {1: "", 2: "Double-", 3: "Triple-"}


def click(x: float, y: float, *, button: str = "left", count: int = 1, mods=()) -> str:
    """Click at a screen point. Everything is checked before the first event goes out: a bad
    coordinate, button, count or modifier is a ValueError, never half a click."""
    x, y = _num(x, "x"), _num(y, "y")
    if button not in ("left", "right", "middle"):
        raise ValueError(f"button must be left, right or middle, not {button!r}")
    if isinstance(count, bool) or not isinstance(count, (int, float, str)) or str(count).strip() not in ("1", "2", "3"):
        raise ValueError(f"count must be 1, 2 or 3, not {count!r}")
    count = int(str(count).strip())
    flags = mods_flags(mods)
    down, up, btn = _buttons(button)
    move(x, y)
    time.sleep(0.02)
    for i in range(1, count + 1):
        d = Q.CGEventCreateMouseEvent(None, down, (x, y), btn)
        u = Q.CGEventCreateMouseEvent(None, up, (x, y), btn)
        Q.CGEventSetIntegerValueField(d, Q.kCGMouseEventClickState, i)
        Q.CGEventSetIntegerValueField(u, Q.kCGMouseEventClickState, i)
        if flags:
            Q.CGEventSetFlags(d, flags)
            Q.CGEventSetFlags(u, flags)
        _post(d)
        _post(u)
        time.sleep(0.05)
    return f"{_COUNT_WORDS[count]}{button.capitalize()}-clicked at ({int(x)}, {int(y)})"


# A drag real apps take as one: the button held a moment before moving (some start a drag only
# after a press-and-hold), a small first step past the drag threshold, then the way there at a
# human pace, and a moment's rest before letting go (a drop target highlights first).
_DRAG_SETTLE_S = 0.05
_DRAG_HOLD_S = 0.12
_DRAG_FIRST_PT = 4.0
_DRAG_STEPS = 20
_DRAG_S = 0.4
_DRAG_REST_S = 0.08


def drag(x1: float, y1: float, x2: float, y2: float, steps: int = _DRAG_STEPS) -> str:
    x1, y1, x2, y2 = _num(x1, "x1"), _num(y1, "y1"), _num(x2, "x2"), _num(y2, "y2")
    steps = max(1, int(steps))
    move(x1, y1)
    time.sleep(_DRAG_SETTLE_S)
    at = (x1, y1)
    d = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, at, Q.kCGMouseButtonLeft)
    Q.CGEventSetIntegerValueField(d, Q.kCGMouseEventClickState, 1)
    _post(d)
    try:  # once the button is down, it always comes up again — whatever goes wrong on the way
        time.sleep(_DRAG_HOLD_S)
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        points = []
        if dist > _DRAG_FIRST_PT * 2:
            points.append((x1 + dx / dist * _DRAG_FIRST_PT, y1 + dy / dist * _DRAG_FIRST_PT))
        points += [(x1 + dx * i / steps, y1 + dy * i / steps) for i in range(1, steps + 1)]
        for p in points:
            ev = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDragged, p, Q.kCGMouseButtonLeft)
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, 1)
            _post(ev)
            at = p
            time.sleep(_DRAG_S / len(points))
        time.sleep(_DRAG_REST_S)
    finally:
        u = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp, at, Q.kCGMouseButtonLeft)
        Q.CGEventSetIntegerValueField(u, Q.kCGMouseEventClickState, 1)
        _post(u)
    return f"Dragged ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})"


def scroll(x: float, y: float, dy: int = -5, dx: int = 0) -> str:
    x, y = _num(x, "x"), _num(y, "y")
    dy, dx = int(_num(dy, "dy")), int(_num(dx, "dx"))
    move(x, y)
    _post(Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 2, dy, dx))
    return f"Scrolled {'down' if dy < 0 else 'up'} {abs(dy)} at ({int(x)},{int(y)})"


def type_text(text: str, *, chunk: int = 20, delay: float = 0.01) -> str:
    """Type Unicode text into the focused element (keyboard-layout independent)."""
    for i in range(0, len(text), chunk):
        piece = text[i : i + chunk]  # slicing by code point never splits a surrogate pair
        n16 = len(piece.encode("utf-16-le")) // 2  # CG wants UTF-16 unit count, not code points
        d = Q.CGEventCreateKeyboardEvent(None, 0, True)
        Q.CGEventKeyboardSetUnicodeString(d, n16, piece)
        _post(d)
        u = Q.CGEventCreateKeyboardEvent(None, 0, False)
        Q.CGEventKeyboardSetUnicodeString(u, n16, piece)
        _post(u)
        time.sleep(delay)
    return f"Typed {len(text)} characters"


def parse_chord(combo: str) -> tuple[int, str]:
    """(modifier flags, key name) of a chord: "cmd+shift+s", "⌘⇧T", "opt+left", "ctrl-c",
    "cmd+?" (shift is implied by the symbol). ValueError for an unknown modifier or key, or a
    second key."""
    s = str(combo).strip()
    for sym, name in (("⌘", "cmd"), ("⇧", "shift"), ("⌥", "alt"), ("⌃", "ctrl")):
        s = s.replace(sym, f"{name}+")
    s = s.replace(" ", "")
    if s.endswith("++"):
        s = s[:-2] + "+plus"
    parts = [p for p in s.split("+") if p]
    if len(parts) == 1 and "-" in parts[0].strip("-"):
        parts = [p for p in parts[0].split("-") if p]  # "ctrl-shift-tab"
    flags, key = 0, None
    for raw in parts:
        p = raw.lower()
        mod = _MOD_ALIASES.get(p)
        if mod is not None:
            flags |= _MODS[mod]
            continue
        p = _KEY_ALIASES.get(p, p)
        if p in _SHIFTED:
            flags |= _MODS["shift"]
            p = _SHIFTED[p]
        if p not in _KEYCODES:
            raise ValueError(f"unknown key or modifier {raw!r}")
        if key is not None:
            raise ValueError(f"two keys in {combo!r} ({key!r} and {raw!r}) — press one chord at a time")
        key = p
    if key is None:
        raise ValueError(f"no key in {combo!r}")
    return flags, key


def hotkey(combo: str) -> str:
    """Press a key chord like 'cmd+shift+s', 'return', 'cmd+space'. An unknown key or modifier is
    an error — never a bare key pressed without the modifiers that were asked for."""
    try:
        flags, key = parse_chord(combo)
    except ValueError as e:
        return f"Error: {e}"
    code = _KEYCODES[key]
    d = Q.CGEventCreateKeyboardEvent(None, code, True)
    u = Q.CGEventCreateKeyboardEvent(None, code, False)
    Q.CGEventSetFlags(d, flags)
    Q.CGEventSetFlags(u, flags)
    _post(d)
    time.sleep(0.02)
    _post(u)
    return f"Pressed {combo}"
