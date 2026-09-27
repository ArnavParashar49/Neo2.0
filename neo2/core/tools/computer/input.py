"""Mouse + keyboard injection via Quartz CGEvents. Coordinates are global screen *points* (the
main display's top-left is 0,0; other displays sit at their own offsets, negative ones too) — the
space the AX tree uses. A screenshot's pixels map onto it by the transform its note gives."""

from __future__ import annotations

import ctypes
import math
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

import Quartz as Q

from neo2.core import effects
from neo2.core.tools.computer import wait

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
    "f13": 105,
    "f14": 107,
    "f15": 113,
    "f16": 106,
    "f17": 64,
    "f18": 79,
    "f19": 80,
    "f20": 90,
    "help": 114,
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
_KEY_ALIASES = {
    "plus": "+", "minus": "-", "equals": "=", "esc": "escape", "enter": "return", "del": "delete",
    # the names models write keys by
    "spacebar": "space", "comma": ",", "period": ".", "dot": ".", "slash": "/", "backslash": "\\",
    "semicolon": ";", "quote": "'", "apostrophe": "'", "backquote": "`", "backtick": "`", "grave": "`",
    "leftbracket": "[", "rightbracket": "]", "bracketleft": "[", "bracketright": "]",
    "pgup": "pageup", "pgdn": "pagedown", "arrowup": "up", "arrowdown": "down", "arrowleft": "left",
    "arrowright": "right", "uparrow": "up", "downarrow": "down", "leftarrow": "left", "rightarrow": "right",
    "fwddelete": "forwarddelete", "forwarddel": "forwarddelete", "bksp": "backspace", "ret": "return",
    "cr": "return", "caps": "capslock",
}  # fmt: skip
# A symbol typed with shift on the (US) key that carries it: ⌘⇧} is ⌘⇧], "cmd+?" is ⌘⇧/.
_SHIFTED = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}  # fmt: skip


# While the controls self-test runs (neo2.core.selftest), every event is shown to its watcher before it
# is posted: the watcher marks it as NEO's own and raises instead when it wouldn't reach the test
# window, or when the user has taken over the mouse or keyboard. None the rest of the time.
_watch: Callable[[Any], None] | None = None
_sent = 0  # events posted by this process (a look taken before one may be out of date after it)
_TIME = time  # (a test that skips the input pacing swaps this module's `time` for its own)


def _pause(s: float) -> None:
    """The input's own pacing: a precise sleep (no ~25% overshoot) on the wait clock."""
    if s <= 0:
        return
    if time is not _TIME:
        time.sleep(s)
        return
    wait.precise_sleep(s)


def _post(ev) -> None:
    global _sent
    _sent += 1
    if _watch is not None:
        _watch(ev)
    effects.guard("input", "post a keyboard or mouse event", via=Q.CGEventPost)
    Q.CGEventPost(Q.kCGHIDEventTap, ev)


def _post_to_pid(pid: int, ev) -> None:
    """An event for one app only (CGEventPostToPid: it reaches that app in front or not), through
    the same watcher and effects switch as _post — the watcher reads where it goes off the event
    (kCGEventTargetUnixProcessID) — and never into NEO's own processes or a protected app. Counted
    like _post's (a look kept from before it is out of date)."""
    global _sent
    _sent += 1
    try:
        Q.CGEventSetIntegerValueField(ev, Q.kCGEventTargetUnixProcessID, int(pid))
    except Exception:  # noqa: BLE001 - (a test's stand-in event)
        pass
    if _watch is not None:
        _watch(ev)
    from neo2.core.tools.computer import ax

    ax.refuse_off_limits(int(pid))
    effects.guard("input", "post a keyboard event to one app", via=Q.CGEventPostToPid)
    Q.CGEventPostToPid(pid, ev)


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
_MOVE_SETTLE_S = 0.02  # the pointer rests where it lands before the button goes down
_CLICK_GAP_S = 0.05  # between the clicks of a double or triple click
_SCROLL_SETTLE_S = 0.015  # the pointer rests where the wheel turns


def click_count(count) -> int:
    """1, 2 or 3 clicks ("2" and 2.0 too); ValueError for anything else — never a click that
    silently isn't one (0) or a different number of them."""
    try:
        f = float(count) if not isinstance(count, bool) else float("nan")
    except (TypeError, ValueError):
        f = float("nan")
    if f not in (1.0, 2.0, 3.0):
        raise ValueError(f"count must be 1, 2 or 3, not {count!r}")
    return int(f)


def click(x: float, y: float, *, button: str = "left", count: int = 1, mods=()) -> str:
    """Click at a screen point. Everything is checked before the first event goes out: a bad
    coordinate, button, count or modifier is a ValueError, never half a click."""
    x, y = _num(x, "x"), _num(y, "y")
    if button not in ("left", "right", "middle"):
        raise ValueError(f"button must be left, right or middle, not {button!r}")
    count = click_count(count)
    flags = mods_flags(mods)
    down, up, btn = _buttons(button)
    move(x, y)
    _pause(_MOVE_SETTLE_S)
    for i in range(1, count + 1):
        if i > 1:
            _pause(_CLICK_GAP_S)  # (between the clicks of a double click; none after the last)
        d = Q.CGEventCreateMouseEvent(None, down, (x, y), btn)
        u = Q.CGEventCreateMouseEvent(None, up, (x, y), btn)
        Q.CGEventSetIntegerValueField(d, Q.kCGMouseEventClickState, i)
        Q.CGEventSetIntegerValueField(u, Q.kCGMouseEventClickState, i)
        if flags:
            Q.CGEventSetFlags(d, flags)
            Q.CGEventSetFlags(u, flags)
        _post(d)
        _post(u)
    return f"{_COUNT_WORDS[count]}{button.capitalize()}-clicked at ({int(x)}, {int(y)})"


# A drag real apps take as one: the button held a moment before moving (some start a drag only
# after a press-and-hold), a small first step past the drag threshold, then the way there at a
# human pace, and a moment's rest before letting go (a drop target highlights first).
_DRAG_SETTLE_S = 0.05
_DRAG_HOLD_S = 0.12
_DRAG_FIRST_PT = 4.0
_DRAG_STEPS = 20
_DRAG_S = 0.4  # at most this long on the way…
_DRAG_MIN_S = 0.2  # …at least this long…
_DRAG_PT_PER_S = 1200.0  # …at about this speed in between
_DRAG_REST_S = 0.08


def drag(x1: float, y1: float, x2: float, y2: float, steps: int = _DRAG_STEPS) -> str:
    x1, y1, x2, y2 = _num(x1, "x1"), _num(y1, "y1"), _num(x2, "x2"), _num(y2, "y2")
    steps = max(1, int(steps))
    move(x1, y1)
    _pause(_DRAG_SETTLE_S)
    at = (x1, y1)
    d = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, at, Q.kCGMouseButtonLeft)
    Q.CGEventSetIntegerValueField(d, Q.kCGMouseEventClickState, 1)
    _post(d)
    try:  # once the button is down, it always comes up again — whatever goes wrong on the way
        _pause(_DRAG_HOLD_S)
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        points = []
        if dist > _DRAG_FIRST_PT * 2:
            points.append((x1 + dx / dist * _DRAG_FIRST_PT, y1 + dy / dist * _DRAG_FIRST_PT))
        points += [(x1 + dx * i / steps, y1 + dy * i / steps) for i in range(1, steps + 1)]
        # a human pace for the distance: a short drag isn't dragged out as long as a long one
        span = min(_DRAG_S, max(_DRAG_MIN_S, dist / _DRAG_PT_PER_S))
        for p in points:
            ev = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDragged, p, Q.kCGMouseButtonLeft)
            Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, 1)
            _post(ev)
            at = p
            _pause(span / len(points))
        _pause(_DRAG_REST_S)
    finally:
        u = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp, at, Q.kCGMouseButtonLeft)
        Q.CGEventSetIntegerValueField(u, Q.kCGMouseEventClickState, 1)
        _post(u)
    return f"Dragged ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})"


def scroll(x: float, y: float, dy: int = -5, dx: int = 0) -> str:
    """Turn the wheel at a screen point: the pointer moves there first, and the wheel event itself
    carries the point (an event without one is taken where the pointer was a moment before)."""
    x, y = _num(x, "x"), _num(y, "y")
    dy, dx = int(_num(dy, "dy")), int(_num(dx, "dx"))
    move(x, y)
    _pause(_SCROLL_SETTLE_S)
    ev = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 2, dy, dx)
    place = getattr(Q, "CGEventSetLocation", None)  # (a stand-in Quartz may not place events)
    if place is not None:
        place(ev, (x, y))
    _post(ev)
    return f"Scrolled {'down' if dy < 0 else 'up'} {abs(dy)} at ({int(x)},{int(y)})"


# ---- text: whole characters, at most 20 UTF-16 units an event ------------------------------------------
_CHUNK_UNITS = 20  # CGEventKeyboardSetUnicodeString takes at most this many UTF-16 units


def _units(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _extends(c: str) -> bool:
    """A character that belongs to the one before it: a combining mark (a vowel sign, a nukta, an
    accent), a variation selector, an emoji skin tone, a tag."""
    o = ord(c)
    return (
        unicodedata.category(c) in ("Mn", "Mc", "Me")
        or 0xFE00 <= o <= 0xFE0F
        or 0x1F3FB <= o <= 0x1F3FF
        or 0xE0020 <= o <= 0xE007F
        or 0xE0100 <= o <= 0xE01EF
    )


def _regional(c: str) -> bool:
    return 0x1F1E6 <= ord(c) <= 0x1F1FF


def clusters(text: str) -> list[str]:
    """The text as the characters a person sees (grapheme clusters, closely enough): a letter with
    its marks, a Devanagari conjunct (joined by a virama), an emoji with its skin tone or its ZWJ
    family, a flag's two letters."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        if _regional(text[i]) and j < n and _regional(text[j]):
            j += 1
        while j < n:
            c = text[j]
            if _extends(c):
                j += 1
            elif c == "\u200d" and j + 1 < n:  # a zero-width joiner joins the next one on
                j += 2
            elif unicodedata.combining(text[j - 1]) == 9 and unicodedata.category(c) == "Lo":
                j += 1  # after a virama: the next consonant is the same conjunct
            else:
                break
        out.append(text[i:j])
        i = j
    return out


def chunks(text: str, units: int = _CHUNK_UNITS) -> list[str]:
    """``text`` in pieces of at most ``units`` UTF-16 units each, never splitting a character a
    person sees (one longer than that alone is split between code points)."""
    out: list[str] = []
    cur, n = "", 0
    for g in clusters(text):
        u = _units(g)
        if u > units:  # a very long cluster: code points, never half a surrogate pair
            if cur:
                out.append(cur)
                cur, n = "", 0
            for ch in g:
                cu = _units(ch)
                if n + cu > units:
                    out.append(cur)
                    cur, n = "", 0
                cur, n = cur + ch, n + cu
            continue
        if n + u > units:
            out.append(cur)
            cur, n = "", 0
        cur, n = cur + g, n + u
    if cur:
        out.append(cur)
    return out


def type_text(text: str, *, chunk: int = _CHUNK_UNITS, delay: float = 0.01) -> str:
    """Type Unicode text into the focused element (keyboard-layout independent), in pieces of at
    most ``chunk`` UTF-16 units that never split a character."""
    pieces = chunks(text, max(1, min(int(chunk), _CHUNK_UNITS)))
    for k, piece in enumerate(pieces):
        n16 = _units(piece)
        d = Q.CGEventCreateKeyboardEvent(None, 0, True)
        Q.CGEventKeyboardSetUnicodeString(d, n16, piece)
        _post(d)
        u = Q.CGEventCreateKeyboardEvent(None, 0, False)
        Q.CGEventKeyboardSetUnicodeString(u, n16, piece)
        _post(u)
        if k < len(pieces) - 1:
            _pause(delay)
    return f"Typed {len(text)} characters"


# ---- the keyboard layout: where a letter's key is --------------------------------------------------------
# A shortcut names a character ("cmd+z"), and apps match it by the character the key types in the
# current layout: on AZERTY, ⌘Z is the key US keyboards call W. The letter keys are read from the
# current ASCII-capable layout (UCKeyTranslate); the Text Input Sources calls behind it must run on
# the main thread (macOS asserts it), so :func:`refresh_layout` reads them there and a hotkey on a
# worker thread uses what it read. Named keys (return, arrows, F-keys) are the same on every layout.
_layout: tuple[str, dict[str, int]] | None = None
_layout_lock = threading.Lock()


def _layout_id() -> str:
    """The current keyboard layout's id, from the preferences (safe on any thread)."""
    try:
        from CoreFoundation import CFPreferencesCopyAppValue

        return str(CFPreferencesCopyAppValue("AppleCurrentKeyboardLayoutInputSourceID", "com.apple.HIToolbox") or "")
    except Exception:  # noqa: BLE001
        return ""


def _tis_letters() -> tuple[str, dict[str, int]] | None:
    """(layout id, character → keycode) of the current ASCII-capable layout. Main thread only."""
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
    cf.CFDataGetBytePtr.restype, cf.CFDataGetBytePtr.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    carbon.TISCopyCurrentASCIICapableKeyboardLayoutInputSource.restype = ctypes.c_void_p
    carbon.TISGetInputSourceProperty.restype = ctypes.c_void_p
    carbon.TISGetInputSourceProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    carbon.LMGetKbdType.restype = ctypes.c_uint8
    carbon.UCKeyTranslate.restype = ctypes.c_int32
    carbon.UCKeyTranslate.argtypes = [
        ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
    ]  # fmt: skip
    src = carbon.TISCopyCurrentASCIICapableKeyboardLayoutInputSource()
    if not src:
        return None
    try:
        data = carbon.TISGetInputSourceProperty(src, ctypes.c_void_p.in_dll(carbon, "kTISPropertyUnicodeKeyLayoutData"))
        sid = carbon.TISGetInputSourceProperty(src, ctypes.c_void_p.in_dll(carbon, "kTISPropertyInputSourceID"))
        if not data:
            return None
        buf = ctypes.create_string_buffer(256)
        name = buf.value.decode() if sid and cf.CFStringGetCString(sid, buf, 256, 0x08000100) else ""
        layout, kbd = cf.CFDataGetBytePtr(data), carbon.LMGetKbdType()
        keys: dict[str, int] = {}
        for code in range(128):
            dead, n, out = ctypes.c_uint32(0), ctypes.c_ulong(0), (ctypes.c_uint16 * 4)()
            # kUCKeyActionDisplay, no modifiers, no dead keys
            if carbon.UCKeyTranslate(layout, code, 3, 0, kbd, 1, ctypes.byref(dead), 4, ctypes.byref(n), out) != 0:
                continue
            if n.value == 1 and (ch := chr(out[0]).lower()).isprintable() and not ch.isspace():
                keys.setdefault(ch, code)
        return name, keys
    finally:
        cf.CFRelease(src)


def refresh_layout() -> bool:
    """Read the current keyboard layout's character keys — on the main thread only (the event
    loop's), where the Text Input Sources calls may run; elsewhere nothing is read (False)."""
    global _layout
    if threading.current_thread() is not threading.main_thread():
        return False
    now = _layout_id()
    with _layout_lock:
        if _layout is not None and _layout[0] == now and now:
            return True
    try:
        got = _tis_letters()
    except Exception:  # noqa: BLE001 - can't read it: the US positions stay
        got = None
    if not got or len(got[1]) < 30:
        return False
    with _layout_lock:
        _layout = (now or got[0], got[1])
    return True


def keycode(key: str) -> int:
    """The virtual keycode that types ``key`` (a named key, or a character) on the current layout."""
    if len(key) == 1:
        with _layout_lock:
            got = _layout
        if got is not None and key in got[1] and (not got[0] or got[0] == _layout_id()):
            return got[1][key]
    return _KEYCODES[key]


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
    joined = parts[0].lower().replace("-", "").replace("_", "") if len(parts) == 1 else ""
    if len(parts) == 1 and "-" in parts[0].strip("-") and joined not in _KEYCODES and joined not in _KEY_ALIASES:
        parts = [p for p in parts[0].split("-") if p]  # "ctrl-shift-tab" (but "page-down" is one key)
    flags, key = 0, None
    for raw in parts:
        p = raw.lower()
        if len(p) > 1 and p not in _KEYCODES and p not in _KEY_ALIASES:
            p = p.replace("_", "").replace("-", "")  # "page_down", "arrow-down"
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


_KEY_HOLD_S = 0.02  # a key is held this long (some apps miss a press with no time between)


def hotkey(combo: str) -> str:
    """Press a key chord like 'cmd+shift+s', 'return', 'cmd+space'. An unknown key or modifier is
    an error — never a bare key pressed without the modifiers that were asked for."""
    try:
        flags, key = parse_chord(combo)
    except ValueError as e:
        return f"Error: {e}"
    code = keycode(key)
    d = Q.CGEventCreateKeyboardEvent(None, code, True)
    u = Q.CGEventCreateKeyboardEvent(None, code, False)
    Q.CGEventSetFlags(d, flags)
    Q.CGEventSetFlags(u, flags)
    _post(d)
    _pause(_KEY_HOLD_S)
    _post(u)
    return f"Pressed {combo}"
