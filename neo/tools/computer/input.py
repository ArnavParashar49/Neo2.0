"""Mouse + keyboard injection via Quartz CGEvents. Coordinates are screen *points*
(top-left origin), the same space the AX tree and downscaled screenshots use."""

from __future__ import annotations

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
    "command": Q.kCGEventFlagMaskCommand,
    "meta": Q.kCGEventFlagMaskCommand,
    "shift": Q.kCGEventFlagMaskShift,
    "alt": Q.kCGEventFlagMaskAlternate,
    "option": Q.kCGEventFlagMaskAlternate,
    "ctrl": Q.kCGEventFlagMaskControl,
    "control": Q.kCGEventFlagMaskControl,
    "fn": Q.kCGEventFlagMaskSecondaryFn,
}


def _post(ev) -> None:
    Q.CGEventPost(Q.kCGHIDEventTap, ev)


def move(x: float, y: float) -> None:
    _post(Q.CGEventCreateMouseEvent(None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft))


def click(x: float, y: float, *, button: str = "left", count: int = 1) -> str:
    move(x, y)
    time.sleep(0.02)
    if button == "right":
        down, up, btn = Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight
    else:
        down, up, btn = Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft
    for i in range(1, count + 1):
        d = Q.CGEventCreateMouseEvent(None, down, (x, y), btn)
        u = Q.CGEventCreateMouseEvent(None, up, (x, y), btn)
        Q.CGEventSetIntegerValueField(d, Q.kCGMouseEventClickState, i)
        Q.CGEventSetIntegerValueField(u, Q.kCGMouseEventClickState, i)
        _post(d)
        _post(u)
        time.sleep(0.05)
    return f"{'Double-' if count == 2 else ''}{button.capitalize()}-clicked at ({int(x)}, {int(y)})"


def drag(x1: float, y1: float, x2: float, y2: float, steps: int = 12) -> str:
    move(x1, y1)
    _post(Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, (x1, y1), Q.kCGMouseButtonLeft))
    for i in range(1, steps + 1):
        t = i / steps
        _post(
            Q.CGEventCreateMouseEvent(
                None,
                Q.kCGEventLeftMouseDragged,
                (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t),
                Q.kCGMouseButtonLeft,
            )
        )
        time.sleep(0.015)
    _post(Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp, (x2, y2), Q.kCGMouseButtonLeft))
    return f"Dragged ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})"


def scroll(x: float, y: float, dy: int = -5, dx: int = 0) -> str:
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


def hotkey(combo: str) -> str:
    """Press a key chord like 'cmd+shift+s', 'return', 'cmd+space'."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
    flags = 0
    key = None
    for p in parts:
        if p in _MODS:
            flags |= _MODS[p]
        else:
            key = p
    if key is None:
        return f"Error: no key in '{combo}'"
    code = _KEYCODES.get(key)
    if code is None:
        if len(key) == 1:
            return type_text(key)
        return f"Error: unknown key '{key}'"
    d = Q.CGEventCreateKeyboardEvent(None, code, True)
    u = Q.CGEventCreateKeyboardEvent(None, code, False)
    Q.CGEventSetFlags(d, flags)
    Q.CGEventSetFlags(u, flags)
    _post(d)
    time.sleep(0.02)
    _post(u)
    return f"Pressed {combo}"
