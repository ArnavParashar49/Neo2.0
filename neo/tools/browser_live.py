"""NEO's browser, live in the chat (settings().browser_view == "chat").

The overlay's browser card shows NEO's own headless Chromium while it works: a CDP screencast of
the page NEO is on, streamed to the overlay as binary NBF1 frames (credits and a latest-frame slot,
so a slow view never queues pixels); what NEO is doing — the element it is about to click, the
field it types into, a caption — as `browser_action` events; its steps as a timeline with small
thumbnails; and the hand-offs (log in, a code, a CAPTCHA, an approval) as prompts on the card.

The user can take the page. Then their pointer and keys reach it through CDP — only theirs, only
while they have it, only for the session they are looking at — and NEO's next browser step waits
until they hand it back; it then re-reads the page instead of acting on what it saw before.

What the user types is theirs: input events, typed and pasted text, clipboard text, frames,
thumbnails and posters never go through the event bus, the transcript replay, the logs or the disk,
and no model is ever shown them. NEO takes no snapshot or screenshot while the user has the page,
and afterwards the fields the user filled read as "(filled by you)".

Every coordinate in the protocol is page viewport CSS px. The protocol itself is the overlay's
(ui/src/browser/types.ts); this module is its core side. Playwright objects come from
neo.tools.browser (imported lazily: that module imports this one).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import math
import re
import secrets
import struct
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

from neo.agent import confirm
from neo.agent.loop import add_pause_clock
from neo.config import settings
from neo.events import app_bus, bus

V = 1  # protocol version

# ---- frames (NBF1, §10.4.1) ---------------------------------------------------------------------

MAGIC = b"NBF1"
_HEADER = struct.Struct("<4sBBH16sIdHHHHHffHB7s")
HEADER_LEN = _HEADER.size  # 64
KIND_LIVE, KIND_STILL = 1, 2
FLAG_USER, FLAG_LOADING, FLAG_STILL = 1, 2, 4
_MIME_CODE = {"image/jpeg": 1, "image/webp": 2, "image/png": 3}
_MIME = {v: k for k, v in _MIME_CODE.items()}
MAX_IMAGE = 2 * 1024 * 1024


def _u16(v: float) -> int:
    return max(0, min(0xFFFF, int(round(v))))


def pack_frame(
    sid_bytes: bytes,
    fid: int,
    image: bytes,
    *,
    kind: int = KIND_LIVE,
    ts_ms: float = 0.0,
    vw: float,
    vh: float,
    iw: float,
    ih: float,
    dpr: float,
    sx: float = 0.0,
    sy: float = 0.0,
    flags: int = 0,
    mime: str = "image/jpeg",
) -> bytes:
    head = _HEADER.pack(
        MAGIC, 1, kind, HEADER_LEN, sid_bytes[:16].ljust(16, b"\0"), fid & 0xFFFFFFFF, float(ts_ms),
        _u16(vw), _u16(vh), _u16(iw), _u16(ih), _u16(dpr * 100), float(sx), float(sy), flags & 0xFFFF,
        _MIME_CODE.get(mime, 1), b"\0" * 7,
    )  # fmt: skip
    return head + image


# A site's keyboard hint baked into an element's name ("Search Wikipedia [ctrl-option-f]") and the
# role word some sites repeat ("Search Search Wikipedia"): not what the user would call it.
_KEY_HINT = re.compile(
    r"\s*[\[(](?:(?:ctrl|control|alt|option|opt|shift|cmd|command|meta|access\s*key)\b[^\])]*|[⌃⌥⇧⌘][^\])]*)[\])]",
    re.I,
)


def tidy_label(label: str | None) -> str:
    """An element's name the way a caption should say it."""
    words = _KEY_HINT.sub("", str(label or "")).split()
    out: list[str] = []
    for w in words:
        if not out or w.casefold() != out[-1].casefold():
            out.append(w)
    return " ".join(out)


class _Frame:
    """One page frame on its way to the views: decoded, packed (or turned into JSON) only when a
    view actually gets it, and then once for all of them."""

    __slots__ = ("sid", "sid_bytes", "fid", "src", "kind", "ts_ms", "vw", "vh", "sx", "sy", "flags", "_img", "_bin", "_txt")  # fmt: skip

    def __init__(self, sid: str, sid_bytes: bytes, fid: int, src: bytes | str, *, kind: int, ts_ms: float, vw: float,
                 vh: float, sx: float, sy: float, flags: int) -> None:  # fmt: skip
        self.sid, self.sid_bytes, self.fid, self.src = sid, sid_bytes, fid, src
        self.kind, self.ts_ms, self.flags = kind, ts_ms, flags
        self.vw, self.vh, self.sx, self.sy = vw, vh, sx, sy
        self._img: bytes | None = None
        self._bin: bytes | None = None
        self._txt: str | None = None

    def image(self) -> bytes | None:
        if self._img is None:
            src = self.src
            if isinstance(src, str):
                try:
                    src = base64.b64decode(src)
                except (ValueError, TypeError):
                    src = b""
            self._img = src if src and len(src) <= MAX_IMAGE else b""
        return self._img or None

    def payload(self, binary: bool) -> bytes | str | None:
        img = self.image()
        if img is None:
            return None
        iw, ih = jpeg_size(img) or (self.vw, self.vh)
        dpr = float(DSF)  # the page's device scale factor (the image itself may be scaled to the view)
        if binary:
            if self._bin is None:
                self._bin = pack_frame(
                    self.sid_bytes, self.fid, img, kind=self.kind, ts_ms=self.ts_ms, vw=self.vw, vh=self.vh, iw=iw,
                    ih=ih, dpr=dpr, sx=self.sx, sy=self.sy, flags=self.flags,
                )  # fmt: skip
            return self._bin
        if self._txt is None:
            b64 = self.src if isinstance(self.src, str) else base64.b64encode(img).decode()
            frame = {
                "sid": self.sid, "fid": self.fid, "kind": "still" if self.kind == KIND_STILL else "live",
                "ts": self.ts_ms, "vw": round(self.vw), "vh": round(self.vh), "iw": iw, "ih": ih, "dpr": dpr,
                "sx": self.sx, "sy": self.sy, "flags": self.flags, "mime": "image/jpeg", "b64": b64,
            }  # fmt: skip
            self._txt = json.dumps({"type": "browser_frame", "data": frame, "ts": time.time()})
        return self._txt


def unpack_frame(buf: bytes) -> tuple[dict, bytes] | None:
    """A frame's header and image (None for anything that isn't a v1 NBF1 frame)."""
    if len(buf) < HEADER_LEN:
        return None
    magic, ver, kind, hl, sid, fid, ts, vw, vh, iw, ih, d100, sx, sy, flags, mime, _ = _HEADER.unpack_from(
        buf
    )
    if magic != MAGIC or ver != 1 or hl < HEADER_LEN or hl > len(buf):
        return None
    meta = {
        "sid": _b64url(sid), "fid": fid, "kind": kind, "ts": ts, "vw": vw, "vh": vh, "iw": iw, "ih": ih,
        "dpr": d100 / 100, "sx": sx, "sy": sy, "flags": flags, "mime": _MIME.get(mime, "image/jpeg"),
    }  # fmt: skip
    return meta, bytes(buf[hl:])


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def new_sid() -> tuple[str, bytes]:
    """A session id: 16 random bytes, and their 22-character base64url form."""
    raw = secrets.token_bytes(16)
    return _b64url(raw), raw


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, without decoding it."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return (w, h) if w and h else None
        i += 2 + seg
    return None


# ---- sizes (§1.4: mirrors ui/src/browser/sizes.ts pageViewport) ----------------------------------------


def _jsround(v: float) -> int:
    return int(math.floor(v + 0.5))


def page_viewport(view_w: float, view_h: float) -> tuple[int, int]:
    """The page's CSS viewport for a view this size: desktop layouts at about 0.8 scale, always in
    the view's own aspect (no bars); at the height limits the width follows the height."""
    ar = max(1.0, view_w) / max(1.0, view_h)
    w = min(max(1440.0, view_w), max(1024.0, float(_jsround(view_w / 0.8))))
    h = _jsround(w / ar)
    if h < 560:
        h, w = 560, _jsround(560 * ar)
    elif h > 1600:
        h, w = 1600, _jsround(1600 * ar)
    return max(320, min(4096, int(w))), max(240, min(4096, int(h)))


DEFAULT_VIEWPORT = (1024, 590)  # the docked card on a 14" MacBook Pro, before any view reports
DSF = 2  # frames are Retina-crisp


# ---- words NEO's code builds (never page text, except quoted labels) ------------------------------------


def _host(url: str) -> str:
    try:
        return (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""


def _local_host(host: str) -> bool:
    if host in ("localhost",) or host.endswith((".localhost", ".local", ".test")):
        return True
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def secure_of(url: str) -> str:
    try:
        scheme = urlsplit(str(url)).scheme.lower()
    except ValueError:
        return "internal"
    if scheme == "chrome-error":
        return "error"
    if scheme in ("http", "https") and _local_host(_host(url)):
        return "local"
    return {"https": "secure", "http": "insecure"}.get(scheme, "internal")


def site_name(url: str) -> str:
    """ "amazon.co.uk" → "Amazon": the page's real host, never its text."""
    host = _host(url)
    if not host:
        return "this site"
    if _local_host(host):
        return host
    try:
        from neo.decide.sites import registrable

        reg = registrable(host) or host
    except Exception:  # noqa: BLE001 — no public-suffix data: the last two labels
        reg = ".".join(host.split(".")[-2:])
    first = reg.split(".")[0]
    return first[:1].upper() + first[1:] if first else host


def quoted(s: str, n: int = 32) -> str:
    s = " ".join(str(s or "").split())
    return f"“{s[:n].rstrip()}…”" if len(s) > n else f"“{s}”"


_NEED_TEXTS = {
    "login": ("Log in to {site} here, then hand back.", "NEO won't see what you type."),
    "captcha": ("Solve the check on {site}, then hand back.", "It's there to confirm a person is here."),
    "2fa": ("Enter your verification code here, then hand back.", "Codes stay between you and {site}."),
    "payment": ("Enter your payment details on {site} here, then hand back.", "NEO won't see what you type."),
    "other": ("NEO needs you to finish a step on this page, then hand back.", ""),
}
_USER_DONE = {"login": "You logged in", "2fa": "You entered the code", "captcha": "You solved the check",
              "payment": "You entered your payment details", "other": "You finished the step"}  # fmt: skip
_YOUR_TURN = {"login": "Your turn to log in", "2fa": "Your turn to enter the code", "captcha": "Your turn to solve the check",
              "payment": "Your turn to enter payment details", "other": "Your turn"}  # fmt: skip


def _mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


# ---- keys (CDP Input.dispatchKeyEvent) -------------------------------------------------------------------

_CODE_KEYCODE: dict[str, int] = {
    "Backspace": 8, "Tab": 9, "Enter": 13, "NumpadEnter": 13, "ShiftLeft": 16, "ShiftRight": 16,
    "ControlLeft": 17, "ControlRight": 17, "AltLeft": 18, "AltRight": 18, "Pause": 19, "CapsLock": 20,
    "Escape": 27, "Space": 32, "PageUp": 33, "PageDown": 34, "End": 35, "Home": 36, "ArrowLeft": 37,
    "ArrowUp": 38, "ArrowRight": 39, "ArrowDown": 40, "Insert": 45, "Delete": 46, "MetaLeft": 91,
    "MetaRight": 93, "ContextMenu": 93, "NumpadMultiply": 106, "NumpadAdd": 107, "NumpadSubtract": 109,
    "NumpadDecimal": 110, "NumpadDivide": 111, "Semicolon": 186, "Equal": 187, "Comma": 188, "Minus": 189,
    "Period": 190, "Slash": 191, "Backquote": 192, "BracketLeft": 219, "Backslash": 220, "BracketRight": 221,
    "Quote": 222, "IntlBackslash": 226,
}  # fmt: skip
for _i in range(10):
    _CODE_KEYCODE[f"Digit{_i}"] = 48 + _i
    _CODE_KEYCODE[f"Numpad{_i}"] = 96 + _i
for _i in range(26):
    _CODE_KEYCODE[f"Key{chr(65 + _i)}"] = 65 + _i
for _i in range(1, 25):
    _CODE_KEYCODE[f"F{_i}"] = 111 + _i
# A named key whose code didn't come along (a key released while the view lost focus).
_KEY_CODE = {
    "Backspace": "Backspace", "Tab": "Tab", "Enter": "Enter", "Shift": "ShiftLeft", "Control": "ControlLeft",
    "Alt": "AltLeft", "Meta": "MetaLeft", "Escape": "Escape", " ": "Space", "PageUp": "PageUp",
    "PageDown": "PageDown", "End": "End", "Home": "Home", "ArrowLeft": "ArrowLeft", "ArrowUp": "ArrowUp",
    "ArrowRight": "ArrowRight", "ArrowDown": "ArrowDown", "Delete": "Delete", "Insert": "Insert",
}  # fmt: skip
_MODIFIER_KEYS = {"Shift", "Control", "Alt", "Meta", "CapsLock"}
# macOS editing commands for keys (as Playwright maps them): headless Chromium on a Mac edits text
# only through these. Insert commands are left out, as Playwright does (the key's text inserts).
_MAC_COMMANDS: dict[str, tuple[str, ...]] = {
    "Backspace": ("deleteBackward",), "Delete": ("deleteForward",), "Escape": ("cancelOperation",),
    "ArrowUp": ("moveUp",), "ArrowDown": ("moveDown",), "ArrowLeft": ("moveLeft",), "ArrowRight": ("moveRight",),
    "Home": ("scrollToBeginningOfDocument",), "End": ("scrollToEndOfDocument",), "PageUp": ("scrollPageUp",),
    "PageDown": ("scrollPageDown",), "Shift+Backspace": ("deleteBackward",), "Shift+Delete": ("deleteForward",),
    "Shift+ArrowUp": ("moveUpAndModifySelection",), "Shift+ArrowDown": ("moveDownAndModifySelection",),
    "Shift+ArrowLeft": ("moveLeftAndModifySelection",), "Shift+ArrowRight": ("moveRightAndModifySelection",),
    "Shift+Home": ("moveToBeginningOfDocumentAndModifySelection",),
    "Shift+End": ("moveToEndOfDocumentAndModifySelection",), "Shift+PageUp": ("pageUpAndModifySelection",),
    "Shift+PageDown": ("pageDownAndModifySelection",), "Control+KeyA": ("moveToBeginningOfParagraph",),
    "Control+KeyB": ("moveBackward",), "Control+KeyD": ("deleteForward",), "Control+KeyE": ("moveToEndOfParagraph",),
    "Control+KeyF": ("moveForward",), "Control+KeyH": ("deleteBackward",), "Control+KeyK": ("deleteToEndOfParagraph",),
    "Control+KeyN": ("moveDown",), "Control+KeyP": ("moveUp",), "Control+KeyT": ("transpose",),
    "Control+Backspace": ("deleteBackwardByDecomposingPreviousCharacter",),
    "Control+ArrowLeft": ("moveToLeftEndOfLine",), "Control+ArrowRight": ("moveToRightEndOfLine",),
    "Shift+Control+ArrowLeft": ("moveToLeftEndOfLineAndModifySelection",),
    "Shift+Control+ArrowRight": ("moveToRightEndOfLineAndModifySelection",),
    "Alt+Backspace": ("deleteWordBackward",), "Alt+Delete": ("deleteWordForward",),
    "Alt+ArrowLeft": ("moveWordLeft",), "Alt+ArrowRight": ("moveWordRight",),
    "Alt+ArrowUp": ("moveBackward", "moveToBeginningOfParagraph"), "Alt+ArrowDown": ("moveForward", "moveToEndOfParagraph"),
    "Shift+Alt+Backspace": ("deleteWordBackward",), "Shift+Alt+ArrowLeft": ("moveWordLeftAndModifySelection",),
    "Shift+Alt+ArrowRight": ("moveWordRightAndModifySelection",),
    "Shift+Alt+ArrowUp": ("moveParagraphBackwardAndModifySelection",),
    "Shift+Alt+ArrowDown": ("moveParagraphForwardAndModifySelection",),
    "Meta+ArrowUp": ("moveToBeginningOfDocument",), "Meta+ArrowDown": ("moveToEndOfDocument",),
    "Meta+ArrowLeft": ("moveToLeftEndOfLine",), "Meta+ArrowRight": ("moveToRightEndOfLine",),
    "Meta+Backspace": ("deleteToBeginningOfLine",), "Meta+KeyA": ("selectAll",), "Meta+KeyZ": ("undo",),
    "Shift+Meta+KeyZ": ("redo",), "Shift+Meta+ArrowUp": ("moveToBeginningOfDocumentAndModifySelection",),
    "Shift+Meta+ArrowDown": ("moveToEndOfDocumentAndModifySelection",),
    "Shift+Meta+ArrowLeft": ("moveToLeftEndOfLineAndModifySelection",),
    "Shift+Meta+ArrowRight": ("moveToRightEndOfLineAndModifySelection",),
}  # fmt: skip
_NAME_OK = re.compile(r"^[A-Za-z0-9]{1,32}$")


def _key_ok(key: str) -> bool:
    """A DOM KeyboardEvent.key value: a named key ("ArrowLeft", "F5") or one character."""
    if _NAME_OK.fullmatch(key):
        return True
    return 0 < len(key) <= 4 and key.isprintable()


def key_event(
    kind: str, key: str, code: str, mods: int, *, text: str = "", repeat: bool = False, location: int = 0
) -> dict:
    """The CDP Input.dispatchKeyEvent params for one keydown / keyup from the overlay."""
    if not code and key in _KEY_CODE:
        code = _KEY_CODE[key]
    if not key and code:
        key = next((k for k, c in _KEY_CODE.items() if c == code), "")
    vk = _CODE_KEYCODE.get(code, 0)
    if not vk and len(key) == 1 and key.upper().isascii() and key.upper().isalnum():
        vk = ord(key.upper())
    p: dict[str, Any] = {
        "key": key,
        "code": code,
        "windowsVirtualKeyCode": vk,
        "nativeVirtualKeyCode": vk,
        "modifiers": mods,
    }
    if location:
        p["location"] = location
    if kind == "keyup":
        p["type"] = "keyUp"
        return p
    if key == "Enter" and not mods & (2 | 4):
        text = "\r"
    elif mods & (2 | 4):
        text = ""  # ⌃ / ⌘ combos never type
    p["type"] = "keyDown" if text else "rawKeyDown"
    if text:
        p["text"] = p["unmodifiedText"] = text
    if repeat:
        p["autoRepeat"] = True
    shortcut = "+".join(
        [m for bit, m in ((8, "Shift"), (2, "Control"), (1, "Alt"), (4, "Meta")) if mods & bit] + [code]
    )
    if cmds := _MAC_COMMANDS.get(shortcut):
        p["commands"] = list(cmds)
    return p


# ---- rate limits (§10.7) -------------------------------------------------------------------------------


class _Bucket:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst, self.tokens, self.t = rate, burst, burst, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_LIMITS = {"move": (60, 60), "wheel": (60, 60), "button": (20, 40), "key": (60, 120), "text": (20, 40),
           "paste": (2, 4), "control": (10, 20), "view": (8, 8)}  # fmt: skip
_INPUT_CLASS = {"mousemove": "move", "wheel": "wheel", "mousedown": "button", "mouseup": "button", "click": "button",
                "keydown": "key", "keyup": "key", "text": "text", "paste": "paste"}  # fmt: skip
_TEXT_MAX, _PASTE_MAX, _CLIP_MAX = 2000, 20000, 100_000
_INPUT_QUEUE = 256
_CMD_MAX, _PASTE_CMD_MAX = 64 * 1024, 128 * 1024

# ---- who watches (one per overlay / pop-out window connection) -------------------------------------------

_RANK = {"window": 4, "focus": 3, "docked": 2, "mini": 1, "hidden": 0}
# Frames a second: while NEO drives the view is only watched, so 24 does; while the user drives,
# every frame is their feedback.
_FPS = {"window": 24, "focus": 24, "docked": 24, "mini": 2}
_FPS_USER = {"window": 30, "focus": 30, "docked": 30, "mini": 2}
_JSON_FPS = 10


class Viewer:
    """One overlay or pop-out window connection. `conn` is the websocket server's client: it has
    `send_json(text) -> bool` (False: not reading) and `offer_frame(payload) -> bool` (True: it
    replaced a frame that hadn't gone out yet)."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.hello = False
        self.binary = False
        self.surface = "overlay"
        self.sid = ""  # the session its view is for
        self.view = "hidden"
        self.css_w = self.css_h = 0.0
        self.dpr = 1.0
        self.visible = False
        self.credits = 2
        self.slot: tuple[int, bytes | str] | None = None
        self.last_sent = 0.0
        self.last_ack = time.monotonic()
        self.outstanding: dict[int, float] = {}
        self.offered = 0  # the fid last handed to the connection
        self.timer: asyncio.TimerHandle | None = None
        self.seq: dict[str, int] = {}
        self.buckets: dict[str, _Bucket] = {}
        self.last_reject = 0.0
        self.carry = [0.0, 0.0]  # wheel deltas held back by the rate limit
        self.pending_view: dict | None = None

    def watching(self, sid: str) -> bool:
        return bool(sid) and self.hello and self.visible and self.view != "hidden" and self.sid == sid

    def fps(self, user: bool = False) -> float:
        cap = (_FPS_USER if user else _FPS).get(self.view, 2)
        return min(cap, _JSON_FPS) if not self.binary else cap

    def bucket(self, cls: str) -> _Bucket:
        if cls not in self.buckets:
            self.buckets[cls] = _Bucket(*_LIMITS[cls])
        return self.buckets[cls]

    def send(self, type_: str, data: dict) -> None:
        with contextlib.suppress(Exception):
            self.conn.send_json(json.dumps({"type": type_, "data": data, "ts": time.time()}, default=str))


# ---- the session ---------------------------------------------------------------------------------------


@dataclass
class Session:
    sid: str
    sid_bytes: bytes
    started_at: float
    job: str | None = None
    jobs: set[str] = field(default_factory=set)
    controller: str = "neo"
    status: str = "starting"
    seq: int = 0
    aid: int = 0
    step_id: int = 0
    fid: int = 0
    steps: list[dict] = field(default_factory=list)
    need_user: dict | None = None
    asked: set[str] = field(default_factory=set)  # the jobs a need_user stopped
    waiting: dict | None = None  # the action parked on a confirm card
    error: dict | None = None
    caption: str | None = None
    popped: bool = False
    popout: bool = False
    action: dict | None = None
    dialog: dict | None = None
    select: dict | None = None
    user_step: int | None = None
    user_since: float = 0.0
    touched: bool = False  # the user has had the page this session (their fields get masked)
    ended: bool = False
    alive: bool = True
    closed_at: float = 0.0
    url: str = ""
    title: str = ""
    favicon: str | None = None
    favicon_sent: bool = False
    secure: str = "internal"
    loading: bool = False
    can_back: bool = False
    can_forward: bool = False
    tabs: dict = field(default_factory=lambda: {"count": 1, "active": 0})
    last_ok: str = ""
    need_reason: str = ""  # what a need_user asked for (login, 2fa, captcha, payment, other)
    meta: dict[int, dict] = field(default_factory=dict)  # per step: NEO's own bookkeeping, never sent
    saw: set[str] = field(default_factory=set)  # jobs that had a browser step's result on this card


# ---- the page's own view of itself (run in the page, never shown to a model) ------------------------------

_JS_DEEP = "let a = document.activeElement; while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;"
_JS_HINT = (
    "([x, y]) => {"
    + _JS_DEEP
    + r"""
  let cursor = null;
  if (x !== null) {
    let e = document.elementFromPoint(x, y);
    while (e && e.shadowRoot) { const d = e.shadowRoot.elementFromPoint(x, y); if (!d || d === e) break; e = d; }
    if (e) {
      let c = getComputedStyle(e).cursor || 'auto';
      if (c === 'auto') {
        const f = e.closest('input,textarea,[contenteditable=""],[contenteditable=true]');
        const nontext = f && /^(?:button|submit|reset|checkbox|radio|range|color|file|image)$/i.test(f.type || '');
        c = f && !nontext ? 'text' : e.closest('a[href],button,[role=button],[role=link],summary,label,select') ? 'pointer' : 'default';
      }
      cursor = c.split(',').pop().trim();
    }
  }
  let focus = null;
  if (a && a !== document.body && a !== document.documentElement) {
    const tag = a.tagName, type = (a.type || '').toLowerCase();
    const kind = tag === 'TEXTAREA' ? 'textarea' : tag === 'SELECT' ? 'select' : a.isContentEditable ? 'contenteditable'
      : tag === 'INPUT' ? (type === 'password' ? 'password'
        : /^(?:button|submit|reset|checkbox|radio|range|color|file|image|hidden)$/.test(type) ? 'other' : 'text') : 'other';
    const r = a.getBoundingClientRect();
    focus = {rect: {x: r.x, y: r.y, w: r.width, h: r.height}, kind};
  }
  return {cursor, focus};
}"""
)
# Before handing a login / code page over: the field the user's first keys should land in.
_JS_FOCUS_FIELD = r"""(want) => {
  const vis = e => { const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return false;
    const s = getComputedStyle(e); return s.visibility !== 'hidden' && s.display !== 'none'; };
  const words = i => [i.name, i.id, i.placeholder, i.getAttribute('aria-label'), i.getAttribute('autocomplete')].join(' ');
  const all = [...document.querySelectorAll('input')].filter(i =>
    /^(?:email|text|tel|number|password|)$/.test((i.getAttribute('type') || '').toLowerCase()) && !i.disabled && !i.readOnly && vis(i) && !i.value);
  const code = i => i.getAttribute('autocomplete') === 'one-time-code' || /one[- ]?time|\botp\b|code|2fa|verif/i.test(words(i));
  const login = i => i.type === 'email' || /^(?:username|email)$/.test(i.getAttribute('autocomplete') || '')
    || (i.form && i.form.querySelector('input[type=password]'));
  const search = i => /search|query|\bq\b/i.test(words(i)) || i.closest('[role=search],search');
  const pick = (want === '2fa' && all.find(code)) || all.find(i => i.type !== 'password' && login(i))
    || all.find(i => i.type === 'password') || all.find(i => !search(i));
  if (!pick) return null;
  pick.focus();
  const r = pick.getBoundingClientRect();
  return {x: r.x, y: r.y, w: r.width, h: r.height, kind: pick.type === 'password' ? 'password' : 'text'};
}"""
# What the user edits while they have the page is theirs: the field it lands in is marked, and NEO's
# reads and pictures hide marked fields afterwards. The mark goes on the element the edit event
# itself reports (an <input>, <textarea>, <select>, or an editor's contenteditable host), so moving
# on at once (Tab, a click on the next field) or a page moving the focus by itself never marks the
# wrong one. The listener is installed in every frame when the user gets the page and switched off
# when NEO has it back (NEO's own typing is NEO's). A page script filling a field while the user
# drives counts too: masking more than needed is the safe side.
_JS_FILLABLE = (
    "const own = t => { if (!t || t.nodeType !== 1) return null;"
    " if (t.isContentEditable) { while (t.parentElement && t.parentElement.isContentEditable) t = t.parentElement; return t; }"
    " if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return t;"
    " if (t.tagName === 'INPUT' && !/^(?:checkbox|radio|button|submit|reset|image|file|hidden)$/i.test(t.type || '')) return t;"
    " return null; };"
)
_JS_ARM_FILLED = (
    "(on) => {"
    + _JS_FILLABLE
    + r""" const K = Symbol.for('neo.userFilled');
  let st = window[K];
  if (!st) {
    st = {on: false};
    Object.defineProperty(window, K, {value: st});
    const mark = e => { if (!st.on) return; const t = own((e.composedPath && e.composedPath()[0]) || e.target);
      if (t) t.setAttribute('data-neo-user-filled', ''); };
    for (const type of ['beforeinput', 'input', 'change']) addEventListener(type, mark, true);
  }
  st.on = !!on;
  return true;
}"""
)
# The same mark on the focused field, as a second line: right before anything that can move the
# focus (a press, Tab, Enter) after the user edited, and a moment after their last edit.
_JS_MARK_FILLED = (
    "() => {"
    + _JS_DEEP
    + _JS_FILLABLE
    + " const t = own(a); if (t) t.setAttribute('data-neo-user-filled', ''); }"
)
_JS_SELECTION = (
    "(cut) => {"
    + _JS_DEEP
    + r"""
  if (a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA')) {
    if ((a.type || '').toLowerCase() === 'password') return '';
    let t = ''; try { t = a.value.slice(a.selectionStart ?? 0, a.selectionEnd ?? 0); } catch (e) { t = ''; }
    if (cut && t && !a.readOnly && !a.disabled) document.execCommand('delete');
    return t;
  }
  const t = String(getSelection() || '');
  if (cut && t && a && a.isContentEditable) document.execCommand('delete');
  return t;
}"""
)
# A <select> under the user's press: headless Chromium draws no picker, the card does.
_JS_SELECT_AT = r"""([x, y, id]) => {
  let e = document.elementFromPoint(x, y);
  while (e && e.shadowRoot) { const d = e.shadowRoot.elementFromPoint(x, y); if (!d || d === e) break; e = d; }
  const s = e && e.closest ? e.closest('select') : null;
  if (!s || s.multiple || s.size > 1 || s.disabled) return null;
  s.setAttribute('data-neo-select', id);
  s.focus();
  const r = s.getBoundingClientRect();
  const grp = o => o.parentElement && o.parentElement.tagName === 'OPTGROUP' ? o.parentElement : null;
  const options = [...s.options].slice(0, 500).map(o => ({value: o.value, label: (o.label || o.text || '').trim().slice(0, 200),
    selected: o.selected, disabled: o.disabled || !!(grp(o) && grp(o).disabled), group: grp(o) ? (grp(o).label || '').slice(0, 100) : undefined}));
  return {rect: {x: r.x, y: r.y, w: r.width, h: r.height}, multiple: false, options};
}"""
_JS_FAVICON = r"""() => {
  const links = [...document.querySelectorAll('link[rel]')].filter(l => /(?:^|\s)icon(?:\s|$)/i.test(l.rel) && l.href);
  const small = links.find(l => /\b(?:16|32)x/.test(l.sizes && l.sizes.value || ''));
  return (small || links[0] || {}).href || (location.origin && location.origin !== 'null' ? location.origin + '/favicon.ico' : '');
}"""
# Where the user's own values sit in a frame (viewport CSS px): a picture of the page for the card
# (a step's thumbnail, the poster) paints over them once the user has had the page.
_JS_FILLED_RECTS = r"""(said) => {
  const out = [];
  const add = r => { if (r.width > 0 && r.height > 0 && out.length < 200) out.push({x: r.x, y: r.y, w: r.width, h: r.height}); };
  const walk = root => {
    for (const e of root.querySelectorAll('[data-neo-user-filled], input[type=password]')) add(e.getBoundingClientRect());
    for (const e of root.querySelectorAll('*')) if (e.shadowRoot) walk(e.shadowRoot);
  };
  walk(document);
  // Where the page repeats what the user typed: a field that holds it (the results page's own search
  // box), and its text (a suggestion list, a results heading).
  const words = (said || []).filter(v => v && v.length >= 3).map(v => v.toLowerCase());
  const holds = root => {
    for (const e of root.querySelectorAll('input, textarea')) {
      const v = String(e.value || '').toLowerCase();
      if (v && words.some(w => v.includes(w))) add(e.getBoundingClientRect());
    }
    for (const e of root.querySelectorAll('*')) if (e.shadowRoot) holds(e.shadowRoot);
  };
  if (words.length) holds(document);
  if (words.length && document.body) {
    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n && out.length < 200; n = tw.nextNode()) {
      const hay = n.data.toLowerCase();
      if (hay.length !== n.data.length) continue;
      for (const v of words) {
        let i = hay.indexOf(v);
        while (i >= 0 && out.length < 200) {
          const r = document.createRange();
          r.setStart(n, i); r.setEnd(n, i + v.length);
          for (const b of r.getClientRects()) add(b);
          i = hay.indexOf(v, i + v.length);
        }
      }
    }
  }
  return out;
}"""
_CURSORS = {
    "default", "pointer", "text", "move", "not-allowed", "grab", "grabbing", "crosshair", "help", "wait", "progress",
    "col-resize", "row-resize", "ew-resize", "ns-resize", "nesw-resize", "nwse-resize", "n-resize", "s-resize",
    "e-resize", "w-resize", "ne-resize", "nw-resize", "se-resize", "sw-resize", "zoom-in", "zoom-out", "copy",
    "alias", "context-menu", "cell", "vertical-text", "all-scroll", "no-drop", "none",
}  # fmt: skip
_FAVICON_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
    "image/svg+xml",
}
_WARM_S = 600.0  # a closed card's page stays open this long, for Reopen
_CLOSE_AFTER_S = 1.5  # a finished job's card folds away this long after it ends
_PRESENT_S = 0.24  # NEO's cursor gets this long to reach a target before it acts
_THUMB_AFTER_S = 0.35
_TAKE_S = 8.0
_USER_TURN_S = 60.0  # a word to NEO (voice) this recent is the user asking it to go on
_DIALOG_WAIT_S = 0.5  # Chromium's own report of a dialog (which frame opened it) comes this soon


def _rect(r: Any) -> dict | None:
    """{x, y, w, h} (page viewport CSS px) from a Playwright box or a JS rect."""
    if not isinstance(r, dict):
        return None
    try:
        x, y = float(r.get("x", 0)), float(r.get("y", 0))
        w, h = float(r.get("w", r.get("width", 0))), float(r.get("h", r.get("height", 0)))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)}


def _num(v: Any, lo: float, hi: float) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(min(hi, max(lo, f)), 1)


def _int(v: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, i))


class Hooks:
    """What the app lends the browser card: resume a job after a hand-back (a user turn), and
    decline a pending approval (Take control while NEO waits)."""

    resume: Any = None  # async (text) -> None
    decline: Any = None  # async (action_id) -> None


class Live:
    def __init__(self) -> None:
        self.hooks = Hooks()
        self.channel: Any = None  # the websocket server: broadcast(type, data, recent=…)
        self.viewers: dict[Any, Viewer] = {}
        self.s: Session | None = None
        self.last: Session | None = None  # the latest closed episode (Reopen, while warm)
        self.page: Any = None
        self.cdp: Any = None
        self.cast: dict | None = None
        self.vp: tuple[int, int] = DEFAULT_VIEWPORT
        self._vp_next: tuple[int, int] | None = None  # a size the page is changing to right now
        self.want_take = False
        self.inflight: dict | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task] = set()
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._holders = 0
        self._hold_t0 = 0.0
        self._held_total = 0.0
        self._inputs: deque[dict] = deque()
        self._hint: dict = {}
        self._hint_at: tuple[float, float] | None = None
        self._dialogs: dict[str, Any] = {}
        self._swallow_up = False
        self._favicons: dict[str, str | None] = {}
        self._main_frame = ""
        self._vp_asked = 0.0
        self._vp_last = 0.0
        self._bound_pages: set[int] = set()
        self._browser_up = False
        self.handed_back_at = 0.0  # when the user last handed the page back after a login, a code…
        # Pages the user has driven: NEO's reads and pictures of them hide what the user filled,
        # whichever card (episode) is open — the page outlives the card.
        self._user_pages: weakref.WeakSet = weakref.WeakSet()
        # What the user's marked fields held, per page, in memory only: a search they submit is
        # echoed by the page it leads to (its title, its links), where no field holds it any more.
        self._user_vals: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # When the user last spoke to NEO: per request (a typed request is a job), and without one
        # (a voice session's words). Only the user's own turn takes the page back from them.
        self._user_jobs: dict[str, float] = {}
        self._user_turn = 0.0
        self._bind_task: asyncio.Task | None = None
        self._bind_page: Any = None
        self._cast_on: Any = None  # the CDP session the screencast runs on
        self._stale_cdps: weakref.WeakSet = weakref.WeakSet()
        self._pinned = False  # NEO's action works at a point on the page: no resize until it's done
        self._arm_task: asyncio.Task | None = None
        self._arm_stale = False  # a frame navigated or appeared: its listener needs installing
        self._mark_pending = False
        self._dialog_seen: deque[tuple[float, str, str, str]] = deque(maxlen=8)  # (t, type, message, url)
        self._dialog_wake: asyncio.Event | None = None
        self._close_kept: dict | None = None  # the latest card's close, as the transcript replay keeps it
        self._close_sent: dict | None = None
        self.lock = asyncio.Lock()
        self._cast_lock = asyncio.Lock()
        self._released = asyncio.Event()
        self._released.set()
        self._input_wake = asyncio.Event()
        self._input_task: asyncio.Task | None = None

    # ---- plumbing ------------------------------------------------------------------------------------
    def on(self) -> bool:
        return settings().browser_view == "chat"

    def _ensure_loop(self) -> asyncio.AbstractEventLoop | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        if loop is not self._loop:
            self._loop = loop
            self.lock = asyncio.Lock()
            self._cast_lock = asyncio.Lock()
            self._released = asyncio.Event()
            self._released.set()
            self._input_wake = asyncio.Event()
            self._input_task = None
            self._tasks = set()
            self._timers = {}
            self._holders = 0
            self.want_take = False
            self._bind_task = self._arm_task = None
            self._dialog_wake = None
        return loop

    def _spawn(self, coro) -> asyncio.Task | None:
        if self._ensure_loop() is None:
            coro.close()
            return None
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._settled)
        return t

    def _settled(self, t: asyncio.Task) -> None:
        self._tasks.discard(t)
        if not t.cancelled() and (e := t.exception()) is not None:
            print(f"[browser] card: {type(e).__name__}")  # never the page's words or the user's input

    def _later(self, name: str, delay: float, fn) -> None:
        loop = self._ensure_loop()
        if loop is None:
            return
        self._cancel(name)
        self._timers[name] = loop.call_later(max(0.0, delay), self._fire, name, fn)

    def _fire(self, name: str, fn) -> None:
        self._timers.pop(name, None)
        try:
            r = fn()
            if asyncio.iscoroutine(r):
                self._spawn(r)
        except Exception as e:  # noqa: BLE001
            print(f"[browser] card timer: {type(e).__name__}")

    def _cancel(self, name: str) -> None:
        if (h := self._timers.pop(name, None)) is not None:
            h.cancel()

    def _user_has_it(self) -> None:
        """The page is the user's now: NEO's steps wait for the next hand-back. (The event is new;
        whoever waited on the old one wakes and waits again.) Every edit they make marks its field."""
        old, self._released = self._released, asyncio.Event()
        old.set()
        self._set_armed(True)

    def user_said(self, job: str = "") -> None:
        """The user spoke or typed to NEO (their own words, not a model's)."""
        now = time.time()
        if job:
            self._user_jobs[job] = now
            while len(self._user_jobs) > 64:
                self._user_jobs.pop(next(iter(self._user_jobs)))
        else:
            self._user_turn = now

    def job_for(self, child: str, parent: str) -> None:
        """`child` runs on behalf of `parent` (a browse run started by a request): the user's request
        is the user's in its run too."""
        if child and parent and (t := self._user_jobs.get(parent)) is not None:
            self._user_jobs[child] = t

    def _users_turn(self, job: str, s: Session) -> bool:
        """Is `job` the user asking NEO to go on since the page became theirs? A request they made
        after taking it (or a word to the voice session since), not a job that was already running,
        a scheduled one or a model's own call."""
        t = self._user_jobs.get(job)
        if t is not None:
            return t > s.user_since
        return self._user_turn > s.user_since and time.time() - self._user_turn < _USER_TURN_S

    def user_touched(self, page: Any = None) -> bool:
        """Has the user had this page (so NEO's reads and pictures hide what they filled)? Without a
        page: the page the card shows, or the card itself."""
        if page is None:
            s = self.s
            page = self.page
            if s is not None and s.touched:
                return True
        if page is None:
            return False
        try:
            return page in self._user_pages
        except TypeError:
            return False

    def remembered(self, page: Any) -> set[str]:
        """What the user typed on this page (kept in memory only, to hide it from NEO's reads)."""
        try:
            return set(self._user_vals.get(page, ()))
        except TypeError:
            return set()

    async def _keep_values(self, page: Any) -> None:
        from neo.tools import browser

        vals = await browser._user_values(page)
        if not vals:
            return
        with contextlib.suppress(TypeError):
            kept = self._user_vals.setdefault(page, set())
            kept.update(v[:500] for v in vals)
            while len(kept) > 64:
                kept.pop()

    def saw_page(self, job: str) -> bool:
        s = self.s
        return s is not None and job in s.saw

    def ran(self, job: str) -> None:
        """A browser step of `job` ran on this card (its model has seen the page)."""
        s = self.s
        if s is not None and job:
            s.saw.add(job)

    def held_seconds(self) -> float:
        """How long NEO's browser tools have waited for the user (not the runs' time)."""
        return self._held_total + (time.monotonic() - self._hold_t0 if self._holders else 0.0)

    def _broadcast(self, type_: str, data: dict, *, recent: bool = False, replay: dict | None = None) -> None:
        ch = self.channel
        if ch is not None:
            try:
                if replay is not None:
                    ch.broadcast(type_, data, recent=recent, replay=replay)
                else:
                    ch.broadcast(type_, data, recent=recent)
            except Exception as e:  # noqa: BLE001
                print(f"[browser] card broadcast: {type(e).__name__}")

    def _sync_url(self, page: Any = None) -> None:
        """The session's URL from the page itself (prompts and steps name the real host)."""
        s, page = self.s, page if page is not None else self.page
        if s is None or page is None or _closed(page):
            return
        with contextlib.suppress(Exception):
            url = str(page.url)
            if url != s.url:
                s.url, s.secure = url, secure_of(url)

    def watched(self) -> bool:
        s = self.s
        return s is not None and any(v.watching(s.sid) for v in self.viewers.values())

    # ---- messages -------------------------------------------------------------------------------------
    def _viewport(self) -> dict:
        return {"w": self.vp[0], "h": self.vp[1], "dpr": DSF}

    def _open_msg(self, s: Session, reason: str) -> dict:
        return {
            "sid": s.sid, "job": s.job or "", "reason": reason, "url": s.url[:2048], "title": s.title[:200],
            "viewport": self._viewport(), "controller": s.controller, "status": s.status, "started_at": s.started_at,
        }  # fmt: skip

    def _state_msg(self, s: Session, *, favicon: bool | None = None) -> dict:
        tabs = dict(s.tabs)
        msg = {
            "sid": s.sid, "seq": s.seq, "url": s.url[:2048], "title": s.title[:200], "secure": s.secure,
            "loading": s.loading, "progress": None, "can_back": s.can_back, "can_forward": s.can_forward,
            "controller": s.controller, "status": s.status, "viewport": self._viewport(), "tabs": tabs,
            "popped": s.popped, "job": s.job, "need_user": s.need_user, "error": s.error, "caption": s.caption,
            "held": self._holders > 0,  # a NEO step is waiting for the page to be handed back
        }  # fmt: skip
        if favicon if favicon is not None else not s.favicon_sent:
            msg["favicon"] = s.favicon
        return msg

    def _emit_state(self, now: bool = False) -> None:
        """browser_state, coalesced to ten a second."""
        s = self.s
        if s is None or s.ended:
            return
        if not now and "state" in self._timers:
            return

        def fire() -> None:
            s2 = self.s
            if s2 is None or s2.ended:
                return
            s2.seq += 1
            msg = self._state_msg(s2)
            s2.favicon_sent = True
            self._broadcast("browser_state", msg)

        if now:
            self._cancel("state")
            fire()
        else:
            self._later("state", 0.1, fire)

    def hello_snapshot(self) -> dict:
        """What a (re)connecting overlay needs to rebuild the card."""
        s = self.s
        if s is None or s.ended:
            return {
                "v": V,
                "live": None,
                "open": None,
                "steps": [],
                "action": None,
                "dialog": None,
                "select": None,
            }
        return {
            "v": V,
            "live": self._state_msg(s, favicon=True),
            "open": self._open_msg(s, "task"),
            "steps": s.steps[-60:],
            "action": s.action,
            "dialog": s.dialog,
            "select": s.select,
        }

    # ---- the session ------------------------------------------------------------------------------------
    def claim(self, job: str = "", *, url: str = "", starting: bool = False) -> Session | None:
        """A NEO browser tool is about to run for `job`: the card opens (or this job joins it)."""
        if not self.on():
            return None
        self._ensure_loop()
        self._cancel("close")
        s = self.s
        if s is None:
            self._cancel("warm")
            sid, raw = new_sid()
            s = Session(
                sid,
                raw,
                round(time.time(), 3),
                status="starting" if starting or not self._browser_up else "working",
            )
            if job:
                s.jobs.add(job)
                s.job = job
            s.url = url or (str(self.page.url) if self.page is not None and not _closed(self.page) else "")
            s.secure = secure_of(s.url)
            self.last = None  # a new card: the closed one can't be reopened any more
            self.s = s
            self._inputs.clear()
            for v in self.viewers.values():
                v.credits, v.slot, v.outstanding = 2, None, {}
            self._broadcast("browser_open", self._open_msg(s, "task"), recent=True)
            self._emit_state(now=True)
            if self.page is not None and not _closed(self.page):
                self._spawn(self._refresh())
            return s
        if s.controller == "user" and job not in s.jobs and not s.ended and self._users_turn(job, s):
            # The user asked NEO for something after taking the page (they said "continue" in the
            # chat instead of handing back on the card, or asked for something new after a Stop):
            # that's their hand-back. Any other job's step waits for the card's hand-back, and the
            # card says NEO is waiting.
            self._handed_back(resume=False, fold=False)
        if job:
            s.jobs.add(job)
            s.job = job
            self._emit_state()
        return s

    def launched(self) -> None:
        """NEO's browser is up (it was cold-starting, or it went away and a step started it again)."""
        self._browser_up = True
        s = self.s
        if s is None:
            return
        if s.status == "error" and (s.error or {}).get("code") == "browser_closed":
            s.error = None
            s.status = "starting"
        if s.status == "starting":
            s.status = "working" if s.controller == "neo" else "user"
            self._emit_state(now=True)

    def browser_down(self, *, unexpected: bool) -> None:
        """The browser closed (NEO closed it, or it went away under NEO)."""
        self._browser_up = False
        self.page = self.cdp = None
        self.cast = None
        self._bound_pages.clear()
        s = self.s
        if unexpected and s is not None and not s.ended:
            s.status, s.error = (
                "error",
                {"code": "browser_closed", "message": "NEO's browser closed unexpectedly"},
            )
            self._step("error", "NEO's browser closed", "fail", by="neo")
            self._emit_state(now=True)

    def _job_ended(self, job: str) -> None:
        s = self.s
        if s is None or s.ended or job not in s.jobs:
            return
        s.jobs.discard(job)
        s.job = next(iter(s.jobs), None) if s.job == job else s.job
        if not s.jobs:
            s.job = None
            self._later("close", _CLOSE_AFTER_S, self._close_if_done)
        self._emit_state()

    def _close_if_done(self):
        s = self.s
        if s is not None and self.lock.locked():
            self._later("close", _CLOSE_AFTER_S, self._close_if_done)  # a step is still on the page
            return None
        if (
            s is None or s.ended or s.jobs or s.controller != "neo" or s.need_user is not None
            or self._holders or s.status == "starting"
        ):  # fmt: skip
            return None
        if s.status == "error":
            # The page (or the whole browser) is gone and no job will bring it back.
            return self.end("error", alive=False)
        return self.end("done")

    async def end(self, reason: str, *, alive: bool = True, poster: bool = True) -> None:
        """The card's episode ends (the page may stay warm for Reopen)."""
        s = self.s
        if s is None or s.ended:
            return
        s.ended = True
        self._cancel("close")
        self.inflight = None
        self._pinned = False
        if s.user_step is not None:
            self._finish_user_step(s)
        for d in list(self._dialogs.values()):
            with contextlib.suppress(Exception):
                await d.dismiss()
        self._dialogs.clear()
        pic = None
        if poster and alive and reason != "erased":
            if s.controller == "user":
                # Never a picture of the user's turn: NEO's own last look at the page instead.
                pic = next(
                    (st["thumb"] for st in reversed(s.steps) if st.get("thumb") and st.get("by") == "neo"),
                    None,
                )
            else:
                pic = await self._capture(640, 60, 48 * 1024)
        if s.controller == "user":
            self._set_armed(False)
        await self._stop_cast()
        s.alive, s.closed_at = alive, time.time()
        steps = sum(1 for st in s.steps if st.get("by") == "neo" or st.get("kind") == "user")
        msg = {
            "sid": s.sid, "reason": reason, "alive": alive, "url": s.url[:2048] if reason != "erased" else "",
            "title": s.title[:200] if reason != "erased" else "", "favicon": s.favicon if reason != "erased" else None,
            "poster": pic, "steps": steps, "duration_ms": int((time.time() - s.started_at) * 1000),
            "summary": ("" if reason == "erased" else s.last_ok)[:60],
        }  # fmt: skip
        self.s = None
        self.last = s if alive else None
        self._released.set()
        self._inputs.clear()
        self._hint = {}
        for v in self.viewers.values():
            v.slot = None
        # A card the user had: its poster may show what they filled, so the replay for late-joining
        # overlays keeps the row without it (the live row has it, in memory).
        kept = {**msg, "poster": None} if s.touched and msg.get("poster") else None
        self._close_sent, self._close_kept = msg, kept if kept is not None else msg
        self._broadcast("browser_close", msg, recent=True, replay=kept)
        if alive:
            self._later("warm", _WARM_S, self._cool)
        else:
            self._cancel("warm")

    def _finish_user_step(self, s: Session) -> None:
        """The user's turn on the page ends: its step says what they did and for how long."""
        if s.user_step is None:
            return
        nu = s.need_user
        reason = (s.need_reason or "other") if nu is not None and nu.get("reason") != "confirm" else ""
        cur = next((x for x in s.steps if x["step_id"] == s.user_step), None)
        if reason:
            label = _USER_DONE.get(reason, "You took over")
        elif cur is not None and not cur["label"].startswith("Your turn"):
            label = cur["label"]  # "You took over", "You opened the page again", "You restarted the browser"
        else:
            label = "You took over"
        secs = time.time() - s.user_since
        self._step("user", f"{label} · {_mmss(secs)}", "ok", by="user", step_id=s.user_step,
                   duration_ms=int(secs * 1000))  # fmt: skip
        s.user_step = None

    async def _cool(self) -> None:
        """Ten minutes after a card closed with nothing reopening it: NEO's browser closes (the
        profile, and the user's logins, stay) — and the ended row stops offering Reopen."""
        if self.s is not None:
            return
        last, self.last = self.last, None
        sent, kept = self._close_sent, self._close_kept
        self._close_sent = self._close_kept = None
        if last is not None and sent is not None and sent.get("sid") == last.sid:
            if kept is not None:
                kept["alive"] = False
            self._broadcast(
                "browser_close", {k: v for k, v in {**sent, "alive": False}.items() if k != "poster"}
            )
        from neo.tools import browser

        await browser._b.close()

    # ---- the page -------------------------------------------------------------------------------------------
    def page_changed(self, page: Any) -> None:
        """NEO's browser switched tabs (it followed one, or the user picked one)."""
        if not self.on() or page is None or page is self.page:
            return
        self._spawn(self.bind(page))

    async def bind(self, page: Any) -> None:
        """Watch `page`: its screencast, loading and navigation feed the card. One bind per page at
        a time: a second caller (the tab switch and the step that follows it) waits for the first."""
        if not self.on() or page is None or _closed(page):
            return
        self._ensure_loop()
        if page is self.page and self.cdp is not None:
            return
        t = self._bind_task
        if t is None or t.done() or self._bind_page is not page:
            self._bind_page = page
            t = self._bind_task = self._spawn(self._bind(page))
        if t is not None:
            await asyncio.wait([t])

    async def _bind(self, page: Any) -> None:
        old = self.cdp
        self.page, self.cdp, self.cast = page, None, None
        self._sync_url(page)
        if old is not None:
            with contextlib.suppress(Exception):
                await old.send("Page.stopScreencast")
            with contextlib.suppress(Exception):
                await old.detach()
        if id(page) not in self._bound_pages:
            self._bound_pages.add(id(page))
            with contextlib.suppress(Exception):
                page.on("crash", lambda *_: self._crashed(page))
                page.on("close", lambda *_: self._page_closed(page))
                page.on("framenavigated", lambda *_: self._frames_changed(page))
                page.on("frameattached", lambda *_: self._frames_changed(page))
        try:
            cdp = await page.context.new_cdp_session(page)
        except Exception:  # noqa: BLE001 — a page that is going away (or a fake one)
            return
        if self.page is not page:
            with contextlib.suppress(Exception):
                await cdp.detach()
            return
        self.cdp = cdp
        cdp.on("Page.screencastFrame", lambda p: self._on_frame(cdp, p))
        cdp.on("Page.frameStartedLoading", lambda p: self._loading(p, True))
        cdp.on("Page.frameStoppedLoading", lambda p: self._loading(p, False))
        cdp.on("Page.frameNavigated", self._navigated)
        cdp.on("Page.navigatedWithinDocument", self._navigated)
        cdp.on("Page.javascriptDialogOpening", self._dialog_opening)
        with contextlib.suppress(Exception):
            await cdp.send("Page.enable")
            tree = await cdp.send("Page.getFrameTree")
            self._main_frame = str(((tree or {}).get("frameTree") or {}).get("frame", {}).get("id", ""))
        with contextlib.suppress(Exception):
            size = page.viewport_size or {}
            if (size.get("width"), size.get("height")) != self.vp:
                await page.set_viewport_size({"width": self.vp[0], "height": self.vp[1]})
        s = self.s
        if s is not None and s.controller == "user" and not s.ended:
            self._set_armed(True)  # a tab the user now has: their edits there are theirs too
        await self._retune(still=True)
        await self._refresh()

    def _frames_changed(self, page: Any) -> None:
        if page is self.page:
            self._arm_stale = True

    def _crashed(self, page: Any) -> None:
        s = self.s
        if s is None or page is not self.page:
            return
        s.status, s.error = "error", {"code": "crashed", "message": "This page crashed"}
        self._step("error", "Page crashed", "fail", by="neo")
        self._emit_state(now=True)

    def _page_closed(self, page: Any) -> None:
        if page is not self.page:
            return
        self.page = self.cdp = None
        self.cast = None
        self._spawn(self._follow_open_page())

    async def _follow_open_page(self) -> None:
        from neo.tools import browser

        pages = browser._b.pages()
        if pages:
            browser._b.current = pages[-1]
        elif self.s is not None:
            self._spawn(self._refresh())

    def new_page(self, page: Any) -> None:
        """A tab opened. While the user has the page, the card follows it (NEO follows its own)."""
        s = self.s
        if s is not None and s.controller == "user" and not s.ended:
            from neo.tools import browser

            browser._b.current = page

    def _loading(self, p: dict, on: bool) -> None:
        s = self.s
        if s is None or (self._main_frame and p.get("frameId") != self._main_frame):
            return
        if s.loading != on:
            s.loading = on
            self._emit_state()
        if not on:
            self._later("refresh", 0.15, self._refresh)

    def _navigated(self, p: dict) -> None:
        s = self.s
        if s is None:
            return
        f = p.get("frame") or {}
        if f.get("parentId"):
            return
        if f.get("unreachableUrl"):
            s.error = {
                "code": "navigation",
                "message": f"Couldn't reach {_host(f['unreachableUrl']) or 'the page'}",
            }
        elif f:
            s.error = None if (s.error or {}).get("code") == "navigation" else s.error
        self._later("refresh", 0.15, self._refresh)

    async def _refresh(self) -> None:
        """URL, title, history, tabs and favicon of the page, into the card's state."""
        s, page, cdp = self.s, self.page, self.cdp
        if s is None or page is None or _closed(page):
            return
        s.url = str(page.url)
        s.secure = secure_of(s.url)
        with contextlib.suppress(Exception):
            s.title = " ".join(str(await asyncio.wait_for(page.title(), 2)).split())[:200]
        if cdp is not None:
            with contextlib.suppress(Exception):
                h = await cdp.send("Page.getNavigationHistory")
                i, n = int(h.get("currentIndex", 0)), len(h.get("entries") or [])
                s.can_back, s.can_forward = i > 0, i < n - 1
        from neo.tools import browser

        pages = browser._b.pages()
        tabs: dict = {
            "count": max(1, len(pages)),
            "active": next((i for i, p in enumerate(pages) if p is page), 0),
        }
        if len(pages) > 1:
            listed = []
            for p in pages[:12]:
                title = ""
                with contextlib.suppress(Exception):
                    title = " ".join(str(await asyncio.wait_for(p.title(), 1)).split())[:120]
                listed.append({"title": title, "url": str(p.url)[:300]})
            tabs["list"] = listed
        s.tabs = tabs
        origin = _origin(s.url)
        if origin and origin in self._favicons:
            self._set_favicon(self._favicons[origin])
        elif origin:
            self._favicons[origin] = None
            self._spawn(self._fetch_favicon(page, origin))
        else:
            self._set_favicon(None)
        self._emit_state()

    def _set_favicon(self, icon: str | None) -> None:
        s = self.s
        if s is not None and s.favicon != icon:
            s.favicon, s.favicon_sent = icon, False
            self._emit_state()

    async def _fetch_favicon(self, page: Any, origin: str) -> None:
        """The site's icon as a small data URL (≤ 8 KB). The page names it, so the request is plain:
        no cookies or logins of NEO's profile go with it, and it never reaches this Mac or the local
        network from a page that isn't there itself (every redirect is checked the same way)."""
        from neo.security import policy
        from neo.tools import browser

        local_page = _local_host(_host(origin))
        try:
            href = str(await asyncio.wait_for(page.evaluate(_JS_FAVICON), 2) or "")
            client = None
            for _ in range(4):
                if not href.lower().startswith(("http://", "https://")):
                    return
                if not local_page and (
                    _local_host(_host(href)) or await asyncio.to_thread(policy.private_host, href)
                ):
                    return
                if client is None:
                    client = await browser._b.icon_client()
                    if client is None:
                        return
                resp = await client.get(href, timeout=3000, max_redirects=0, headers={"Accept": "image/*"})
                if 300 <= resp.status < 400 and (loc := str(resp.headers.get("location") or "")):
                    await _dispose(resp)
                    href = urljoin(href, loc)
                    continue
                break
            else:
                return
            try:
                if not resp.ok:
                    return
                kind = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
                body = await resp.body()
            finally:
                await _dispose(resp)
        except Exception:  # noqa: BLE001 — no icon then
            return
        if len(body) > 64 * 1024 or not body:
            return
        if kind not in _FAVICON_TYPES:
            kind = (
                "image/x-icon" if body[:4] == b"\0\0\1\0" else "image/png" if body[:4] == b"\x89PNG" else ""
            )
            if not kind:
                return
        if len(body) > 8 * 1024:
            if kind == "image/svg+xml":
                return
            body, kind = await asyncio.to_thread(_shrink_icon, body)
            if not body or len(body) > 8 * 1024:
                return
        url = f"data:{'image/x-icon' if kind == 'image/vnd.microsoft.icon' else kind};base64,{base64.b64encode(body).decode()}"
        self._favicons[origin] = url
        if len(self._favicons) > 64:
            self._favicons.pop(next(iter(self._favicons)))
        s = self.s
        if s is not None and _origin(s.url) == origin:
            self._set_favicon(url)

    # ---- frames ----------------------------------------------------------------------------------------------
    def _largest(self, views: tuple[str, ...] = ("window", "focus", "docked", "mini")) -> Viewer | None:
        """The largest surface showing the card: the highest-ranked kind (a window over the chat),
        then the most device pixels — the same one whichever overlay spoke last."""
        s = self.s
        if s is None:
            return None
        shown = [v for v in self.viewers.values() if v.watching(s.sid) and v.view in views]
        return max(shown, key=lambda v: (_RANK[v.view], v.css_w * v.css_h * v.dpr * v.dpr), default=None)

    def _cast_params(self) -> dict | None:
        s = self.s
        if s is None or s.ended or self.cdp is None:
            return None
        best = self._largest()
        if best is None:
            return None
        vw, vh = self.vp
        if best.view == "mini":
            mw, mh, q, nth = 480, 300, 55, 6
        else:
            # Frames as sharp as the sharpest view needs (never past the page's own pixels).
            big = [v for v in self.viewers.values() if v.watching(s.sid) and v.view != "mini"]
            d = [max(1.0, min(2.0, v.dpr)) for v in big]
            mw = int(min(2560, vw * DSF, max(160, *(v.css_w * k for v, k in zip(big, d, strict=True)))))
            mh = int(min(1600, vh * DSF, max(100, *(v.css_h * k for v, k in zip(big, d, strict=True)))))
            q, nth = (78, 2) if best.view in ("window", "focus") else (72, 2)
            if all(not v.binary for v in big):
                mw, q, nth = min(mw, 1280), 65, 3
        return {"format": "jpeg", "quality": q, "maxWidth": mw, "maxHeight": mh, "everyNthFrame": nth}

    async def _retune(self, *, still: bool = False) -> None:
        """Start, retune or stop the screencast for who is watching now (and push a still to a
        view that just started watching: a static page sends no frames of its own)."""
        if self._ensure_loop() is None:
            return
        async with self._cast_lock:
            cdp = self.cdp
            want = self._cast_params()
            if cdp is None:
                self.cast = self._cast_on = None
                return
            changed = want != self.cast or (want is not None and self._cast_on is not cdp)
            if changed:
                if self.cast is not None and self._cast_on is cdp:
                    with contextlib.suppress(Exception):
                        await cdp.send("Page.stopScreencast")
                self.cast, self._cast_on = want, (cdp if want is not None else None)
                if want is not None:
                    try:
                        await cdp.send("Page.startScreencast", want)
                    except Exception:  # noqa: BLE001
                        self.cast = self._cast_on = None
                        return
            if want is not None and (changed or still):
                await self._still()

    async def _stop_cast(self) -> None:
        cdp, self.cast, self._cast_on = self.cdp, None, None
        if cdp is not None:
            with contextlib.suppress(Exception):
                await cdp.send("Page.stopScreencast")

    def _on_frame(self, cdp: Any, p: dict) -> None:
        # Acked to Chromium at once, so it keeps producing fresh frames; the newest one replaces
        # whatever a view hasn't been sent yet (never a queue).
        self._spawn(_send(cdp, "Page.screencastFrameAck", {"sessionId": p.get("sessionId")}))
        if cdp is not self.cdp:
            # A session the card no longer listens to (a tab it left): its screencast stops.
            if cdp not in self._stale_cdps:
                with contextlib.suppress(TypeError):
                    self._stale_cdps.add(cdp)
                self._spawn(_send(cdp, "Page.stopScreencast", {}))
            return
        if self.cast is None:
            return
        data = p.get("data")
        if not isinstance(data, str) or not data:
            return
        md = p.get("metadata") or {}
        size = (float(md.get("deviceWidth") or self.vp[0]), float(md.get("deviceHeight") or self.vp[1]))
        if not any(
            abs(size[0] - vp[0]) <= 1.5 and abs(size[1] - vp[1]) <= 1.5
            for vp in (self.vp, self._vp_next)
            if vp
        ):
            return  # a frame from a moment the page was resized for a screenshot: not the page
        self._deliver(
            data, kind=KIND_LIVE, vw=float(md.get("deviceWidth") or self.vp[0]), vh=float(md.get("deviceHeight") or self.vp[1]),
            sx=float(md.get("scrollOffsetX") or 0), sy=float(md.get("scrollOffsetY") or 0),
            ts_ms=float(md.get("timestamp") or time.time()) * 1000,
        )  # fmt: skip

    async def _still(self) -> None:
        cast, cdp = self.cast, self.cdp
        if cast is None or cdp is None or not self.watched():
            return
        got = await self._shot(quality=cast["quality"])
        if got is None:
            return
        data, (x, y, w, h) = got
        size = jpeg_size(data)
        if size and (size[0] > cast["maxWidth"] * 1.05 or size[1] > cast["maxHeight"] * 1.05):
            data = await asyncio.to_thread(
                _resize_jpeg, data, cast["maxWidth"], cast["maxHeight"], cast["quality"]
            )
        if data:
            self._deliver(data, kind=KIND_STILL, vw=w, vh=h, sx=x, sy=y, ts_ms=time.time() * 1000)

    def _fps(self, v: Viewer) -> float:
        s = self.s
        return v.fps(user=s is not None and s.controller == "user")

    async def _shot(self, *, quality: int) -> tuple[bytes, tuple[float, float, float, float]] | None:
        """The visible viewport as a JPEG, as the screen shows it (device pixels). Never with a
        scaled clip: Chromium renders one by resizing the page for a moment, and the screencast
        would stream those frames."""
        cdp = self.cdp
        if cdp is None:
            return None
        try:
            lm = await cdp.send("Page.getLayoutMetrics")
            vv = lm.get("cssVisualViewport") or lm.get("visualViewport") or {}
            x, y = float(vv.get("pageX", 0)), float(vv.get("pageY", 0))
            w, h = float(vv.get("clientWidth") or self.vp[0]), float(vv.get("clientHeight") or self.vp[1])
            try:
                shot = await cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": quality})
            except Exception:  # noqa: BLE001 — right after a resize Chromium may not have a frame yet
                await asyncio.sleep(0.12)
                shot = await cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": quality})
            return base64.b64decode(shot.get("data") or ""), (x, y, w, h)
        except Exception as e:  # noqa: BLE001
            print(f"[browser] card picture failed: {type(e).__name__}")
            return None

    async def _capture(self, width: int, quality: int, limit: int) -> str | None:
        """A small picture of the page (a step's thumbnail, the card's poster) as a data URL of at
        most `limit` bytes. On a page the user has had, what they filled is painted over (no picture
        at all when the page can't say where that is)."""
        page = self.page
        rects: list[tuple[float, float, float, float]] = []
        if page is not None and self.user_touched(page):
            from neo.tools import browser

            said = (await browser._user_values(page) or set()) | self.remembered(page)
            got_rects = await _filled_rects(page, said)
            if got_rects is None:
                return None
            rects = got_rects
        got = await self._shot(quality=85)
        if got is None or not got[0]:
            return None
        data = await asyncio.to_thread(_small_jpeg, got[0], width, quality, limit * 3 // 4, rects, got[1][2])
        return "data:image/jpeg;base64," + base64.b64encode(data).decode() if data else None

    def _deliver(
        self, data: bytes | str, *, kind: int, vw: float, vh: float, sx: float, sy: float, ts_ms: float
    ) -> None:
        """A frame for everyone watching: it replaces whatever a view hasn't been sent yet. It is
        decoded and packed only if it actually goes out (most frames of a busy page never do)."""
        s = self.s
        if s is None or s.ended or not data:
            return
        if len(data) * (3 / 4 if isinstance(data, str) else 1) > MAX_IMAGE:
            return
        s.fid += 1
        flags = (
            (FLAG_USER if s.controller == "user" else 0)
            | (FLAG_LOADING if s.loading else 0)
            | (FLAG_STILL if kind == KIND_STILL else 0)
        )
        frame = _Frame(
            s.sid, s.sid_bytes, s.fid, data, kind=kind, ts_ms=ts_ms, vw=vw, vh=vh, sx=sx, sy=sy, flags=flags
        )
        for v in list(self.viewers.values()):
            if v.watching(s.sid):
                v.slot = (s.fid, frame)
                self._pump(v)

    def _pump(self, v: Viewer) -> None:
        """Send `v` its latest frame if it has credit and its frame rate allows (§10.4.4)."""
        s = self.s
        if v.slot is None:
            return
        if s is None or not v.watching(s.sid):
            v.slot = None
            return
        now = time.monotonic()
        if v.credits <= 0:
            if now - v.last_ack >= 2.0:
                # A lost ack or a reloaded view must not stall it — but a view that stopped acking
                # gets one frame per 2 s, not the full rate.
                v.credits, v.outstanding, v.last_ack = 1, {}, now
            else:
                self._pump_later(v, v.last_ack + 2.0 - now)
                return
        gap = 1.0 / max(1.0, self._fps(v)) - (now - v.last_sent)
        if gap > 0.001:
            self._pump_later(v, gap)
            return
        fid, frame = v.slot
        v.slot = None
        payload = frame.payload(v.binary) if isinstance(frame, _Frame) else frame
        if payload is None:
            return
        replaced = False
        with contextlib.suppress(Exception):
            replaced = bool(v.conn.offer_frame(payload))
        if replaced:
            v.outstanding.pop(v.offered, None)  # that one never went out: its credit is this one's
        else:
            v.credits -= 1
        v.offered = fid
        v.outstanding[fid] = now
        if len(v.outstanding) > 8:
            v.outstanding.pop(next(iter(v.outstanding)))
        v.last_sent = now

    def _pump_later(self, v: Viewer, delay: float) -> None:
        loop = self._ensure_loop()
        if loop is None or v.timer is not None:
            return

        def go() -> None:
            v.timer = None
            self._pump(v)

        v.timer = loop.call_later(max(0.001, delay), go)

    def _ack(self, v: Viewer, m: dict) -> None:
        s = self.s
        fid = m.get("fid")
        if s is None or m.get("sid") != s.sid or not isinstance(fid, int) or fid not in v.outstanding:
            return  # unknown fid: ignored
        v.outstanding.pop(fid, None)
        v.credits = min(2, v.credits + 1)
        v.last_ack = time.monotonic()
        self._pump(v)

    # ---- viewers ---------------------------------------------------------------------------------------------
    def viewer(self, conn: Any) -> Viewer:
        v = self.viewers.get(conn)
        if v is None:
            v = self.viewers[conn] = Viewer(conn)
        return v

    def viewer_gone(self, conn: Any) -> None:
        v = self.viewers.pop(conn, None)
        if v is None:
            return
        if v.timer is not None:
            v.timer.cancel()
        if self.s is not None:
            self._popped_now()
            self._spawn(self._retune())
            self._want_viewport()  # the page follows the largest view that remains

    def _view(self, v: Viewer, m: dict) -> None:
        s = self.s
        if s is None or m.get("sid") != s.sid:
            return
        if not v.bucket("view").take():
            v.pending_view = m  # over the limit: the latest one lands a moment later
            self._later(f"view:{id(v)}", 0.2, lambda: self._view_pending(v))
            return
        surface = str(m.get("surface") or "hidden")
        surface = surface if surface in _RANK else "hidden"
        was = v.watching(s.sid)
        v.sid = s.sid
        v.view = surface
        v.css_w = _num(m.get("css_w"), 0, 8192) or 0.0
        v.css_h = _num(m.get("css_h"), 0, 8192) or 0.0
        v.dpr = _num(m.get("dpr"), 0.5, 4) or 1.0
        v.visible = bool(m.get("visible")) and surface != "hidden" and v.css_w > 0 and v.css_h > 0
        now = v.watching(s.sid)
        if now and not was:
            v.credits, v.outstanding, v.last_ack = 2, {}, time.monotonic()
        if not now:
            v.slot = None
        self._popped_now()
        self._spawn(self._retune(still=now and not was))
        self._want_viewport()

    def _view_pending(self, v: Viewer) -> None:
        m, v.pending_view = v.pending_view, None
        if m is not None and v in self.viewers.values():
            self._view(v, m)

    def _popped_now(self) -> None:
        s = self.s
        if s is None:
            return
        popped = s.popout or any(v.watching(s.sid) and v.view == "window" for v in self.viewers.values())
        if popped != s.popped:
            s.popped = popped
            self._emit_state()

    # ---- the page's size follows the card (§1.4) -----------------------------------------------------------------
    def _target_viewport(self) -> tuple[int, int] | None:
        best = self._largest(("window", "focus", "docked"))  # a mini row never resizes the page
        return page_viewport(best.css_w, best.css_h) if best is not None else None

    def _want_viewport(self) -> None:
        want = self._target_viewport()
        if want is None or want == self.vp:
            self._vp_asked = 0.0
            self._cancel("viewport")
            return
        if not self._vp_asked:
            self._vp_asked = time.monotonic()
        self._later("viewport", 0.3, self._apply_viewport)

    async def _apply_viewport(self) -> None:
        s, page = self.s, self.page
        want = self._target_viewport()
        if s is None or want is None or want == self.vp:
            self._vp_asked = 0.0
            return
        if page is None or _closed(page):
            self.vp = want  # the browser isn't up yet: it starts at this size
            self._vp_asked = 0.0
            self._emit_state()
            return
        now = time.monotonic()
        waited = now - self._vp_asked if self._vp_asked else 0.0
        if s.status == "waiting":  # never under a pending approval: its target must stay put
            self._later("viewport", 0.25, self._apply_viewport)
            return
        if self.lock.locked() and (self._pinned or waited < 0.6):
            # Mid-action: after it, but never later than 600 ms — except under an action at a point
            # (a click at x, y), which must land on the layout it was checked on.
            self._later("viewport", 0.1, self._apply_viewport)
            return
        if now - self._vp_last < 1.0:
            self._later("viewport", self._vp_last + 1.0 - now, self._apply_viewport)
            return
        self._vp_asked, self._vp_last = 0.0, now
        self._vp_next = want
        try:
            await page.set_viewport_size({"width": want[0], "height": want[1]})
        except Exception:  # noqa: BLE001
            self._vp_next = None
            return
        self.vp, self._vp_next = want, None
        self._emit_state()
        a = self.inflight
        if a is not None and self.lock.locked() and a.get("remeasure") is not None and self.s is s:
            # The page changed size under NEO's action: its target moved, and so does the cursor.
            with contextlib.suppress(Exception):
                rect = _rect(await a["remeasure"]())
                if rect is not None and self.inflight is a:
                    a["base"].update(
                        rect=rect,
                        x=round(rect["x"] + rect["w"] / 2, 1),
                        y=round(rect["y"] + rect["h"] / 2, 1),
                    )
                    a["base"].update(vw=self.vp[0], vh=self.vp[1], sy=await self._scroll_y())
                    s.aid += 1
                    a["aid"] = s.aid
                    self._action(a, "intent")
        await asyncio.sleep(0.05)
        await self._retune(still=True)

    # ---- NEO's actions (called by the browser tools) --------------------------------------------------------
    async def enter(self) -> bool:
        """Before a NEO browser tool touches the page: wait while the user has it (no timeout
        while the card is open), then hold the page lock. True when it waited for the user."""
        self._ensure_loop()
        waited = False
        while True:
            waited = await self.hold() or waited
            if (t := self._arm_task) is not None and not t.done():
                await asyncio.wait([t])  # the user's last marks are in, the listener is off
            await self.lock.acquire()
            s = self.s
            if s is not None and (self.want_take or s.controller == "user"):
                self.lock.release()
                continue
            if s is not None and s.controller == "neo" and s.status in ("thinking", "user"):
                s.status = "working"
                self._emit_state()
            return waited

    def leave(self) -> None:
        s = self.s
        self._pinned = False
        if self.inflight is not None:
            self.fail("interrupted")
        if s is not None and s.controller == "neo" and s.status == "working":
            s.status = "thinking"
            self._emit_state()
        if self.lock.locked():
            self.lock.release()

    async def hold(self) -> bool:
        s = self.s
        if s is None or not (self.want_take or s.controller == "user"):
            return False
        self._holders += 1
        if self._holders == 1:
            self._hold_t0 = time.monotonic()
            self._emit_state()  # the card says NEO is waiting for the page
        try:
            while (s := self.s) is not None and (self.want_take or s.controller == "user"):
                if self.want_take and s.controller != "user":
                    await asyncio.sleep(0.05)  # "Take control" is landing
                    continue
                await self._released.wait()
        finally:
            self._holders -= 1
            if self._holders == 0:
                self._held_total += time.monotonic() - self._hold_t0
                self._emit_state()
        return True

    def pin(self) -> None:
        """NEO's action works at a point on the page (a click at x, y): the page keeps its size until
        the action is done, so the point it checked is the point it clicks."""
        self._pinned = True

    # ---- the user's marks on what they fill ---------------------------------------------------------------
    def _set_armed(self, on: bool) -> None:
        """Mark every field the user edits from now on (on), or stop (off: first whatever they edited
        last is marked). In order: each change waits for the one before it."""
        page = self.page
        if page is None or _closed(page):
            return
        if on:
            with contextlib.suppress(TypeError):
                self._user_pages.add(page)
        elif not self.user_touched(page):
            return
        prev = self._arm_task
        self._arm_task = self._spawn(self._arm(prev, page, on))

    async def _arm(self, prev: asyncio.Task | None, page: Any, on: bool) -> None:
        if prev is not None and not prev.done():
            await asyncio.wait([prev])

        async def go() -> None:
            if not on:
                await self._flush_mark()
                await self._keep_values(page)
            self._arm_stale = False
            await _arm_frames(page, on)

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(go(), 4)  # a page that doesn't answer never holds NEO (or the user) up

    def _action(self, a: dict, phase: str, **extra: Any) -> None:
        s = self.s
        if s is None:
            return
        msg = {k: v for k, v in {**a["base"], **extra}.items() if v is not None and v != ""}
        msg.update(sid=s.sid, aid=a["aid"], phase=phase)
        s.action = msg
        self._broadcast("browser_action", msg)

    async def _scroll_y(self) -> float | None:
        page = self.page
        if page is None:
            return None
        try:
            return round(float(await asyncio.wait_for(page.evaluate("() => scrollY"), 1)), 1)
        except Exception:  # noqa: BLE001
            return None

    async def intent(
        self,
        kind: str,
        step: str,
        *,
        rect: dict | None = None,
        x: float | None = None,
        y: float | None = None,
        label: str = "",
        role: str = "",
        fieldname: str = "",
        text: str | None = None,
        secret: bool = False,
        key: str = "",
        direction: str = "",
        amount: float | None = None,
        url: str = "",
        caption: str = "",
        remeasure: Any = None,
        present: bool = True,
    ) -> None:
        """NEO is about to act: the card shows where (cursor, highlight) and what (caption), and a
        running step. `step` is the step's past-tense label, built by NEO's code."""
        s = self.s
        if s is None or s.ended:
            return
        self._sync_url()
        rect = _rect(rect)
        if rect is not None and x is None:
            x, y = round(rect["x"] + rect["w"] / 2, 1), round(rect["y"] + rect["h"] / 2, 1)
        base: dict[str, Any] = {
            "kind": kind, "x": x, "y": y, "rect": rect, "label": tidy_label(label)[:60] or None,
            "role": role or None, "field": tidy_label(fieldname)[:60] or None,
            "text": None if secret or text is None else text[:200], "secret": True if secret else None,
            "key": key or None, "dir": direction or None, "amount": amount, "url": url[:2048] or None,
            "caption": caption or None, "vw": self.vp[0], "vh": self.vp[1],
        }  # fmt: skip
        if kind == "click":
            base.update(button="left", clicks=1)
        if kind in ("click", "type", "select", "key", "read") and (rect is not None or x is not None):
            base["sy"] = await self._scroll_y()
        w = s.waiting
        approved = bool(w) and confirm.approving() == w["action_id"]
        if approved:
            step_id = w["step_id"]
            s.waiting, s.need_user = None, None
            if s.status == "waiting":
                s.status = "working"
            self._emit_state(now=True)
            self._step(kind, f"Approved · {step}", "running", step_id=step_id)
        else:
            step_id = self._merge_step(kind, step, direction)
        s.aid += 1
        s.caption = None
        self.inflight = {"aid": s.aid, "base": base, "step_id": step_id, "step": step, "approved": approved,
                         "remeasure": remeasure, "t0": time.monotonic(), "kind": kind}  # fmt: skip
        self._action(self.inflight, "intent")
        if present and kind in ("click", "type", "select", "key") and self.watched():
            await asyncio.sleep(_PRESENT_S)  # the cursor gets there before the click does

    def done(self, step: str | None = None, *, thumb: bool = True) -> None:
        a, s = self.inflight, self.s
        self.inflight = None
        self._pinned = False
        if a is None or s is None:
            return
        label = step or a["step"]
        if a.get("approved"):
            label = f"Approved · {label}"
        self._action(a, "done")
        self._finish_step(a, label, "ok")
        s.last_ok = label
        if thumb:
            self._later(f"thumb:{a['step_id']}", _THUMB_AFTER_S, lambda: self._thumb(s, a["step_id"]))
        self._later("refresh", 0.2, self._refresh)

    def fail(self, reason: str, step: str | None = None) -> None:
        a, s = self.inflight, self.s
        self.inflight = None
        self._pinned = False
        if a is None or s is None:
            return
        self._action(a, "fail", reason=reason[:120])
        self._finish_step(a, step or a["step"], "fail", reason=reason[:160])

    def waiting(self, action_id: str, step: str) -> None:
        """NEO's action waits on the user's OK (the confirm card): the card spotlights its target."""
        a, s = self.inflight, self.s
        self.inflight = None
        if a is None or s is None:
            return
        s.aid += 1
        a["aid"] = s.aid
        a["base"]["gated"] = True
        self._action(a, "intent")  # the same target, now waiting
        s.waiting = {
            "action_id": action_id,
            "aid": a["aid"],
            "step_id": a["step_id"],
            "base": dict(a["base"]),
            "step": a["step"],
        }
        s.status = "waiting"
        s.need_user = {
            "reason": "confirm", "site": site_name(s.url), "title": "Waiting for your OK", "sub": "",
            "action_id": action_id, **({"rect": a["base"]["rect"]} if a["base"].get("rect") else {}),
        }  # fmt: skip
        self._step("confirm", step, "waiting", step_id=a["step_id"])
        self._broadcast("browser_need_user", {**s.need_user, "sid": s.sid})
        self._emit_state(now=True)

    def _answered(self, action_id: str, outcome: str) -> None:
        """The confirm card NEO's action waited on closed without a yes."""
        s = self.s
        w = s.waiting if s is not None else None
        if s is None or w is None or w["action_id"] != action_id or outcome == "confirmed":
            return
        s.waiting = None
        if (s.need_user or {}).get("action_id") == action_id:
            s.need_user = None
        declined = outcome not in ("superseded", "expired")
        s.action = {k: v for k, v in {**w["base"], "sid": s.sid, "aid": w["aid"], "phase": "fail",
                                      "reason": "declined" if declined else outcome}.items() if v is not None}  # fmt: skip
        self._broadcast("browser_action", s.action)
        what = w.get("base", {}).get("label") or ""
        self._step("confirm", f"Didn't click {quoted(what)}" if declined and what else "Declined" if declined else w["step"],
                   "declined" if declined else "skipped", step_id=w["step_id"])  # fmt: skip
        if s.status == "waiting":
            s.status = "user" if s.controller == "user" else "thinking"
        if not s.jobs and s.controller == "neo":
            self._later("close", _CLOSE_AFTER_S, self._close_if_done)
        self._emit_state(now=True)

    async def need_user(self, reason: str, *, page: Any = None, detail: str = "") -> None:
        """NEO hands the page to the user (log in, a code, a CAPTCHA, payment details…)."""
        s = self.s
        if s is None or s.ended:
            return
        self._sync_url(page)
        if reason not in _NEED_TEXTS:
            reason = "other"
        site = site_name(s.url)
        title, sub = (t.replace("{site}", site) for t in _NEED_TEXTS[reason])
        rect = None
        if reason in ("login", "2fa", "payment") and page is not None:
            with contextlib.suppress(Exception):
                got = await asyncio.wait_for(page.evaluate(_JS_FOCUS_FIELD, reason), 2)
                rect = _rect(got)
        nu: dict[str, Any] = {
            "reason": "other" if reason == "payment" else reason,
            "site": site,
            "title": title,
            "sub": sub,
        }
        if reason == "other" and detail:
            nu["detail"] = " ".join(str(detail).split())[:120]
        if rect is not None:
            nu["rect"] = rect
        if self.inflight is not None:
            self.fail("handed over")
        s.need_user, s.controller, s.status, s.need_reason = nu, "user", "need_user", reason
        s.user_since, s.touched = time.time(), True
        s.asked = set(s.jobs)
        self._user_has_it()
        s.user_step = self._step("user", _YOUR_TURN[reason], "waiting", by="user")
        self._broadcast("browser_need_user", {**nu, "sid": s.sid})
        self._emit_state(now=True)
        self._hint_soon(None)

    def _step(self, kind: str, label: str, status: str, *, by: str = "neo", step_id: int | None = None,
              reason: str = "", duration_ms: int | None = None) -> int:  # fmt: skip
        s = self.s
        if s is None:
            return 0
        if step_id is None:
            s.step_id += 1
            step_id = s.step_id
        cur = next((x for x in s.steps if x["step_id"] == step_id), None)
        st = {
            "sid": s.sid, "step_id": step_id, "ts": cur["ts"] if cur else round(time.time(), 3), "kind": kind,
            "label": " ".join(label.split())[:160], "status": status, "by": by, "url": s.url[:2048] or None,
        }  # fmt: skip
        if reason:
            st["reason"] = reason
        if duration_ms is not None:
            st["duration_ms"] = duration_ms
        if cur is not None:
            cur.update({k: v for k, v in st.items() if v is not None})
            st = cur
        else:
            s.steps.append({k: v for k, v in st.items() if v is not None})
            st = s.steps[-1]
            del s.steps[:-200]
        self._broadcast("browser_step", st)
        return step_id

    def _merge_step(self, kind: str, label: str, direction: str) -> int:
        """Consecutive scrolls are one step ("Scrolled down 3 times"); so is typing into a field
        and pressing Enter right after."""
        s = self.s
        last = s.steps[-1] if s is not None and s.steps else None
        meta = s.meta.get(last["step_id"], {}) if s is not None and last is not None else {}
        if last is not None and last.get("by") == "neo" and last.get("status") == "ok":
            if kind == "scroll" and last.get("kind") == "scroll" and meta.get("dir") == direction:
                meta["n"] = n = int(meta.get("n", 1)) + 1
                return self._step("scroll", f"{label} {n} times", "running", step_id=last["step_id"])
            if (
                kind == "key"
                and label == "Pressed Enter"
                and last.get("kind") == "type"
                and time.time() - meta.get("done_at", 0) < 1.0
            ):
                return self._step(
                    "type", f"{last['label']} and pressed Enter", "running", step_id=last["step_id"]
                )
        step_id = self._step(kind, label, "running")
        if s is not None:
            s.meta[step_id] = {"dir": direction}
            for old in list(s.meta)[:-200]:
                del s.meta[old]
        return step_id

    def _finish_step(self, a: dict, label: str, status: str, *, reason: str = "") -> None:
        s = self.s
        if s is None:
            return
        cur = next((x for x in s.steps if x["step_id"] == a["step_id"]), None)
        if cur is not None and cur.get("kind") in ("type", "scroll") and a["kind"] in ("key", "scroll"):
            label = cur["label"]  # a merged step keeps its merged words
        kind = cur.get("kind", a["kind"]) if cur else a["kind"]
        self._step(kind, label, status, step_id=a["step_id"], reason=reason,
                   duration_ms=int((time.monotonic() - a["t0"]) * 1000))  # fmt: skip
        s.meta.setdefault(a["step_id"], {})["done_at"] = time.time()

    async def _thumb(self, s: Session, step_id: int) -> None:
        # Never while the user has the page: what they do is theirs.
        if s is not self.s or s.controller != "neo" or self.cdp is None:
            return
        pic = await self._capture(400, 55, 16 * 1024)
        if pic is None or s is not self.s or s.controller != "neo":
            return
        cur = next((x for x in s.steps if x["step_id"] == step_id), None)
        if cur is not None and cur.get("by") == "neo":
            cur["thumb"] = pic
            self._broadcast("browser_step", cur)

    # ---- control ---------------------------------------------------------------------------------------------
    async def take(self) -> str:
        """The user takes the page: once NEO's current action finishes (≤ 8 s), NEO's next browser
        step waits. A pending approval is declined — taking over never approves."""
        s = self.s
        if s is None or s.ended:
            return "no_session"
        if s.controller == "user":
            return ""
        self.want_take = True
        got = False
        try:
            try:
                await asyncio.wait_for(self.lock.acquire(), _TAKE_S)
                got = True
            except TimeoutError:
                pass
            if s is not self.s or s.ended:
                return "no_session"
            w = s.waiting
            s.controller, s.status = "user", "user"
            s.user_since, s.touched = time.time(), True
            s.need_user = None if w is not None else s.need_user
            self._user_has_it()
            s.user_step = self._step("user", "You took over", "running", by="user")
            self._cancel("close")
            self._emit_state(now=True)
            if w is not None:
                self._answered(w["action_id"], "user cancelled")
                if self.hooks.decline is not None:
                    self._spawn(self.hooks.decline(w["action_id"]))
                else:
                    confirm.cancel("took over")
            self._hint_soon(None)
            return ""
        finally:
            self.want_take = False
            if got:
                self.lock.release()

    def _handed_back(self, *, resume: bool, fold: bool = True) -> None:
        """The user hands the page back to NEO."""
        s = self.s
        if s is None:
            return
        nu = s.need_user
        reason = (s.need_reason or "other") if nu is not None and nu.get("reason") != "confirm" else ""
        self._finish_user_step(s)
        for d in list(self._dialogs.values()):
            with contextlib.suppress(Exception):
                self._spawn(d.dismiss())
        self._dialogs.clear()
        s.dialog = s.select = None
        s.need_user = None if nu is None or nu.get("reason") != "confirm" else nu
        s.need_reason = ""
        s.controller, s.status = "neo", "thinking"
        self._hint = {}
        self._inputs.clear()
        self._set_armed(False)  # NEO's own typing from here on is NEO's (marks made so far stay)
        self._released.set()
        self._emit_state(now=True)
        if resume and reason and not self._holders and self.hooks.resume is not None:
            self.handed_back_at = time.time()
            what = {"login": "logging in", "2fa": "entering the code", "captcha": "the check",
                    "payment": "entering my payment details"}.get(reason, "that step")  # fmt: skip
            self._spawn(self.hooks.resume(f"I've finished {what} in the browser. Continue the task."))
            self._later("resume", 90.0, self._close_if_done)
        elif fold and not s.jobs and not self._holders:
            self._later("close", _CLOSE_AFTER_S, self._close_if_done)

    def stop(self) -> bool:
        """Stop the job that owns the card; the page stays, the user has it."""
        s = self.s
        if s is None or not s.jobs:
            return False
        jobs = set(s.jobs)
        self._stopped(s)
        for j in jobs:
            bus().cancel_job(j)
        return True

    def user_stopped(self) -> None:
        """The chat's own stop: whatever NEO was doing on the page ends; the page is the user's."""
        s = self.s
        if s is not None and not s.ended and (s.jobs or s.controller == "neo"):
            self._stopped(s)

    def _stopped(self, s: Session) -> None:
        if self.inflight is not None:
            self.fail("stopped")
        w = s.waiting
        s.jobs.clear()
        s.job = None
        if s.controller != "user":
            s.user_since = time.time()
            s.user_step = self._step("user", "You took over", "running", by="user")
        s.controller, s.touched = "user", True
        if s.status != "error":  # a dead page keeps its panel (Restart / Reload)
            s.status = "user"
        s.need_user, s.need_reason = None, ""
        self._user_has_it()
        self._cancel("close")
        self._emit_state(now=True)
        if w is not None:
            self._answered(w["action_id"], "user cancelled")
            if (p := confirm.peek()) is not None and p.action_id == w["action_id"]:
                confirm.cancel("stopped")

    async def _control(self, v: Viewer, m: dict) -> None:
        req = str(m.get("req") or "")[:40]
        action = str(m.get("action") or "")
        if not v.bucket("control").take():
            return self._result(v, req, False, "rate")
        code = await self._do_control(v, action, m, req)
        self._result(v, req, not code, code or None)

    def _result(self, v: Viewer, req: str, ok: bool, code: str | None = None) -> None:
        v.send("browser_result", {"req": req, "ok": ok, **({"code": code} if code else {})})

    async def _do_control(self, v: Viewer, action: str, m: dict, req: str) -> str:
        sid = m.get("sid")
        if action == "reopen":
            return await self._reopen(sid, str(m.get("url") or ""))
        s = self.s
        if s is None or s.ended:
            return "no_session"
        if sid != s.sid:
            return "bad_sid"
        page = self.page
        if action == "take":
            return await self.take()
        if action == "release":
            if s.controller != "user":
                return "not_user"
            if (
                not s.jobs
                and not self._holders
                and (s.need_user is None or s.need_user.get("reason") == "confirm")
            ):
                await self.end("closed")  # "Done": no job is waiting for the page
                return ""
            self._handed_back(resume=True)
            return ""
        if action == "stop":
            return "" if self.stop() else "no_session"
        if action == "close":
            busy = s.jobs and s.controller == "neo" and s.status in ("working", "thinking", "waiting")
            if busy and not m.get("force"):
                return "busy"
            if s.jobs:
                self.stop()
            await self.end("closed")
            return ""
        if action in ("popout", "popin"):
            s.popout = action == "popout"
            self._popped_now()
            self._emit_state()
            self._spawn(self._retune(still=True))
            self._want_viewport()
            return ""
        if action in ("expand", "collapse"):
            return ""
        if action == "restart":
            if s.status != "error":
                return "invalid"
            s.error = None
            if not s.jobs and s.controller != "user":
                # Nothing of NEO's is waiting for it: the user asked for the page back, so it's theirs.
                s.controller, s.touched, s.user_since = "user", True, time.time()
                s.user_step = self._step("user", "You restarted the browser", "running", by="user")
            s.status = "user" if s.controller == "user" else "working"
            self._emit_state(now=True)
            self._spawn(self._relaunch(s.url))
            return ""
        if action in ("back", "forward", "reload", "hard_reload", "stop_loading", "tab"):
            if s.controller != "user" and (code := await self.take()):
                return code
            if page is None or _closed(page):
                return "no_session"
            return await self._navigate(action, m)
        # Only while the user has the page:
        if s.controller != "user":
            return "not_user"
        if action == "navigate":
            url = str(m.get("url") or "").strip()
            if len(url) > 2048 or not re.match(r"^https?://[^\s]+$", url, re.I):
                return "blocked_url"
            if page is None or _closed(page):
                return "no_session"
            self._spawn(_nav(page.goto(url, wait_until="commit", timeout=20000)))
            return ""
        if action == "select":
            return await self._pick(m)
        if action == "dialog":
            return await self._answer_dialog(m)
        if action in ("copy", "cut"):
            text = ""
            if page is not None and not _closed(page):
                with contextlib.suppress(Exception):
                    text = str(await asyncio.wait_for(page.evaluate(_JS_SELECTION, action == "cut"), 2) or "")
            v.send("browser_clipboard", {"sid": s.sid, "req": req, "text": text[:_CLIP_MAX]})
            if action == "cut" and text:
                self._mark_filled_soon()
            return ""
        return "unsupported"

    async def _relaunch(self, url: str) -> None:
        """Start NEO's browser again at the page it was on — under the page lock, so a NEO step
        never has the browser closed under it."""
        from neo.tools import browser

        async with self.lock:
            await browser.relaunch(url)
        s = self.s
        if s is not None and s.controller == "user" and not s.ended:
            self._user_has_it()

    async def _navigate(self, action: str, m: dict) -> str:
        from neo.tools import browser

        page, cdp = self.page, self.cdp
        if action == "tab":
            pages = browser._b.pages()
            i = _int(m.get("index"), -1, 999, -1)
            if not 0 <= i < len(pages):
                return "invalid"
            browser._b.current = pages[i]
            with contextlib.suppress(Exception):
                await pages[i].bring_to_front()
            return ""
        if action == "back":
            self._spawn(_nav(page.go_back(wait_until="commit", timeout=20000)))
        elif action == "forward":
            self._spawn(_nav(page.go_forward(wait_until="commit", timeout=20000)))
        elif action == "reload":
            self._spawn(_nav(page.reload(wait_until="commit", timeout=20000)))
        elif action == "hard_reload" and cdp is not None:
            self._spawn(_send(cdp, "Page.reload", {"ignoreCache": True}))
        elif action == "stop_loading" and cdp is not None:
            self._spawn(_send(cdp, "Page.stopLoading", {}))
        return ""

    async def _reopen(self, sid: Any, url: str) -> str:
        from neo.tools import browser

        if self.s is not None and not self.s.ended:
            return "busy"
        last = self.last
        warm = last is not None and last.alive and sid == last.sid and time.time() - last.closed_at < _WARM_S
        if not warm and not url:
            return "no_session"
        if url and not re.match(r"^https?://[^\s]+$", url, re.I):
            return "blocked_url"
        self._cancel("warm")
        if warm and last is not None:
            s = last
            s.ended, s.alive = False, True
            s.jobs, s.job = set(), None
            s.need_user = s.waiting = None
        else:
            sid2, raw = new_sid()
            s = Session(sid2, raw, round(time.time(), 3))
            s.url = url
        s.controller, s.status, s.touched = "user", "user", True
        s.user_since = time.time()
        s.favicon_sent = False  # a card that opens starts without one: the next state carries it
        self.s, self.last = s, None
        self._user_has_it()
        self._broadcast("browser_open", self._open_msg(s, "reopen"), recent=True)
        s.user_step = self._step("user", "You opened the page again", "running", by="user")
        self._emit_state(now=True)
        if self.page is None or _closed(self.page) or (url and not warm):
            page = await browser._b.page()  # its browser closed meanwhile: open the page again
            if s.url.lower().startswith(("http://", "https://")):
                self._spawn(_nav(page.goto(s.url, wait_until="commit", timeout=20000)))
        else:
            await self._retune(still=True)
            await self._refresh()
        return ""

    # ---- the user's input (§6.1, §10.5, §11.2) -----------------------------------------------------------------
    def _reject(self, v: Viewer, code: str) -> None:
        now = time.monotonic()
        if now - v.last_reject >= 1.0:
            v.last_reject = now
            self._result(v, "input", False, code)

    def _input(self, v: Viewer, m: dict) -> None:
        """One `browser_input`: applied only for the live session's sid, while the user has the page
        and no NEO action holds it, in order, within the rate limits. Dropped silently otherwise."""
        s = self.s
        typ = m.get("type")
        cls = _INPUT_CLASS.get(typ) if isinstance(typ, str) else None
        if cls is None:
            return self._reject(v, "invalid")
        if s is None or s.ended or m.get("sid") != s.sid:
            return self._reject(v, "bad_sid")
        seq = m.get("seq")
        if not isinstance(seq, int) or seq <= v.seq.get(s.sid, -1):
            return None  # a replay or a reorder
        v.seq[s.sid] = seq
        if s.controller != "user" or s.status not in ("user", "need_user"):
            return self._reject(v, "not_user")
        if self.lock.locked() or self.want_take:
            return self._reject(v, "busy")  # a NEO action still holds the page (a take that timed out)
        ev = _valid(typ, m, self.vp)
        if ev is None:
            return self._reject(v, "invalid")
        ev["sid"] = s.sid
        if not v.bucket(cls).take():
            if typ == "wheel":
                v.carry[0] += ev["dx"]
                v.carry[1] += ev["dy"]
            elif typ == "mousemove" and self._inputs and self._inputs[-1]["type"] == "mousemove":
                self._inputs[-1] = ev
            else:
                self._reject(v, "rate")
            return None
        if typ == "wheel" and any(v.carry):
            ev["dx"], ev["dy"] = ev["dx"] + v.carry[0], ev["dy"] + v.carry[1]
            v.carry = [0.0, 0.0]
        self._enqueue(ev)
        return None

    def _enqueue(self, ev: dict) -> None:
        q = self._inputs
        if q and ev["type"] == "mousemove" and q[-1]["type"] == "mousemove":
            q[-1] = ev
        elif q and ev["type"] == "wheel" and q[-1]["type"] == "wheel" and q[-1]["mods"] == ev["mods"]:
            q[-1] = {**ev, "dx": q[-1]["dx"] + ev["dx"], "dy": q[-1]["dy"] + ev["dy"]}
        else:
            if len(q) >= _INPUT_QUEUE:
                for i, old in enumerate(q):
                    if old["type"] == "mousemove":
                        del q[i]
                        break
                else:
                    return  # still full: this one is dropped
            q.append(ev)
        if self._ensure_loop() is None:
            return
        self._input_wake.set()
        if self._input_task is None or self._input_task.done():
            self._input_task = self._spawn(self._drain_inputs())

    async def _drain_inputs(self) -> None:
        while True:
            if not self._inputs:
                self._input_wake.clear()
                try:
                    await asyncio.wait_for(self._input_wake.wait(), 30)
                except TimeoutError:
                    return
                continue
            ev = self._inputs.popleft()
            s = self.s
            if s is None or ev["sid"] != s.sid or s.controller != "user" or self.lock.locked():
                continue
            try:
                await self._dispatch(ev)
            except Exception as e:  # noqa: BLE001 — never what was typed: the kind of event and the failure
                print(f"[browser] browser input failed: {ev['type']} {type(e).__name__}")

    async def _dispatch(self, ev: dict) -> None:
        cdp, page = self.cdp, self.page
        if cdp is None or page is None:
            return
        t = ev["type"]
        mods = ev.get("mods", 0)
        if t not in ("mousemove", "wheel"):
            with contextlib.suppress(TypeError):
                self._user_pages.add(page)  # the user drives this page
            if (a := self._arm_task) is not None and not a.done():
                await asyncio.wait([a])
            if self._arm_stale:  # a new document (or frame) since the listener went in
                self._arm_stale = False
                await _arm_frames(page, True)
            printable = t == "keydown" and bool(ev.get("text")) and not mods & (2 | 4)
            if self._mark_pending and t in ("mousedown", "click", "keydown") and not printable:
                await self._flush_mark()  # the field they typed in, before this moves the focus on
        if t == "mousemove":
            await cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": ev["x"], "y": ev["y"], "modifiers": mods,
                                                        "buttons": ev["buttons"], "button": _held(ev["buttons"])})  # fmt: skip
            self._hint_soon((ev["x"], ev["y"]))
        elif t in ("mousedown", "click"):
            if ev["button"] == "left" and ev["clicks"] == 1 and await self._select_at(ev):
                self._swallow_up = t == "mousedown"
                return
            await cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": ev["x"], "y": ev["y"], "button": ev["button"],
                                                        "buttons": ev["buttons"] or _BUTTON_BIT[ev["button"]], "clickCount": ev["clicks"],
                                                        "modifiers": mods})  # fmt: skip
            if t == "click":
                await cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": ev["x"], "y": ev["y"], "button": ev["button"],
                                                            "buttons": 0, "clickCount": ev["clicks"], "modifiers": mods})  # fmt: skip
            self._hint_soon((ev["x"], ev["y"]))
        elif t == "mouseup":
            if self._swallow_up:
                self._swallow_up = False
                return
            await cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": ev["x"], "y": ev["y"], "button": ev["button"],
                                                        "buttons": ev["buttons"], "clickCount": ev["clicks"], "modifiers": mods})  # fmt: skip
            self._hint_soon((ev["x"], ev["y"]))
        elif t == "wheel":
            await cdp.send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": ev["x"], "y": ev["y"], "deltaX": ev["dx"],
                                                        "deltaY": ev["dy"], "modifiers": mods})  # fmt: skip
            self._hint_soon((ev["x"], ev["y"]))
        elif t in ("keydown", "keyup"):
            p = key_event(
                t,
                ev["key"],
                ev["code"],
                mods,
                text=ev.get("text", ""),
                repeat=ev.get("repeat", False),
                location=ev.get("location", 0),
            )
            await cdp.send("Input.dispatchKeyEvent", p)
            edits = any(re.match(r"delete|transpose|undo|redo|yank", c) for c in p.get("commands", ()))
            if t == "keydown" and (
                (p.get("text") and ev["key"] != "Enter") or ev["key"] in ("Backspace", "Delete") or edits
            ):
                self._mark_filled_soon()
            if t == "keydown":
                self._hint_soon(None)
        elif t in ("text", "paste"):
            await cdp.send("Input.insertText", {"text": ev["text"]})
            self._mark_filled_soon()

    def _mark_filled_soon(self) -> None:
        self._mark_pending = True
        self._later("filled", 0.15, self._flush_mark)

    async def _flush_mark(self) -> None:
        self._cancel("filled")
        if self._mark_pending:
            self._mark_pending = False
            await self._mark_filled()
            if (page := self.page) is not None and not _closed(page):
                await self._keep_values(page)  # before a key or a press that can submit (and navigate)

    async def _mark_filled(self) -> None:
        page = self.page
        if page is None or _closed(page):
            return
        for fr in _all_frames(page):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(fr.evaluate(_JS_MARK_FILLED), 1)

    async def _select_at(self, ev: dict) -> bool:
        s, page = self.s, self.page
        if s is None or page is None:
            return False
        sel_id = "s" + secrets.token_hex(4)
        try:
            got = await asyncio.wait_for(page.evaluate(_JS_SELECT_AT, [ev["x"], ev["y"], sel_id]), 1.5)
        except Exception:  # noqa: BLE001
            return False
        rect = _rect(got.get("rect")) if isinstance(got, dict) else None
        if rect is None:
            return False
        opts = [
            {
                k: v
                for k, v in {
                    "value": str(o.get("value", ""))[:500],
                    "label": str(o.get("label", ""))[:200],
                    "selected": bool(o.get("selected")),
                    "disabled": bool(o.get("disabled")),
                    "group": o.get("group") or None,
                }.items()
                if v is not None
            }  # fmt: skip
            for o in (got.get("options") or [])[:500]
            if isinstance(o, dict)
        ]
        s.select = {"sid": s.sid, "select_id": sel_id, "rect": rect, "multiple": False, "options": opts}
        self._broadcast("browser_select", s.select)
        return True

    async def _pick(self, m: dict) -> str:
        s, page = self.s, self.page
        sel = s.select if s is not None else None
        if sel is None or page is None or m.get("select_id") != sel["select_id"]:
            return "invalid"
        values = m.get("values") if isinstance(m.get("values"), list) else [m.get("value")]
        values = [str(x)[:500] for x in values if x is not None][:500]
        if not values or not all(any(o["value"] == x for o in sel["options"]) for x in values):
            return "invalid"
        s.select = None
        try:
            await page.locator(f'[data-neo-select="{sel["select_id"]}"]').first.select_option(
                value=values, timeout=3000
            )
        except Exception:  # noqa: BLE001
            return "timeout"
        self._mark_filled_soon()
        return ""

    def _dialog_opening(self, p: dict) -> None:
        """Chromium's own report of a dialog: which frame (URL) opened it."""
        self._dialog_seen.append(
            (time.monotonic(), str(p.get("type") or ""), str(p.get("message") or ""), str(p.get("url") or ""))
        )
        if self._dialog_wake is not None:
            self._dialog_wake.set()

    async def _dialog_url(self, kind: str, message: str) -> str | None:
        """The URL of the frame that opened this dialog (None: Chromium didn't say)."""
        deadline = time.monotonic() + _DIALOG_WAIT_S
        while True:
            for i, (t, k, m, url) in enumerate(self._dialog_seen):
                if k == kind and m == message and time.monotonic() - t < 5:
                    del self._dialog_seen[i]
                    return url
            left = deadline - time.monotonic()
            if left <= 0 or self.cdp is None:
                return None
            self._dialog_wake = asyncio.Event()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._dialog_wake.wait(), left)

    async def user_dialog(self, dialog: Any) -> bool:
        """A page's alert / confirm / prompt while the user has the page: the card asks them, naming
        the site that asked — an embedded page (an ad's frame) is named as one, never as the site."""
        s = self.s
        if not self.on() or s is None or s.ended or s.controller != "user":
            return False
        did = "d" + secrets.token_hex(4)
        kind = str(getattr(dialog, "type", "alert"))
        message = str(getattr(dialog, "message", ""))
        page = self.page
        top = str(page.url if page is not None else s.url)
        opener = await self._dialog_url(kind, message)
        if s is not self.s or s.ended:
            return False
        msg = {
            "sid": s.sid, "dialog_id": did, "kind": kind if kind in ("alert", "confirm", "prompt", "beforeunload") else "alert",
            "message": message[:2000], "origin": _host(opener or top),
        }  # fmt: skip
        if opener and _origin(opener) != _origin(top):
            msg["embedded"] = True
        if kind == "prompt":
            msg["default"] = str(getattr(dialog, "default_value", ""))[:2000]
        self._dialogs[did] = dialog
        s.dialog = msg
        self._broadcast("browser_dialog", msg)
        return True

    async def _answer_dialog(self, m: dict) -> str:
        s = self.s
        did = str(m.get("dialog_id") or "")
        d = self._dialogs.pop(did, None)
        if s is None or d is None:
            return "invalid"
        try:
            if m.get("accept"):
                text = m.get("text")
                await (
                    d.accept(str(text)[:2000])
                    if isinstance(text, str) and s.dialog and s.dialog.get("kind") == "prompt"
                    else d.accept()
                )
            else:
                await d.dismiss()
        except Exception:  # noqa: BLE001 — the page went away
            pass
        if s.dialog and s.dialog.get("dialog_id") == did:
            s.dialog = None
        self._broadcast("browser_dialog", {"sid": s.sid, "dialog_id": did, "closed": True})
        return ""

    def _hint_soon(self, at: tuple[float, float] | None) -> None:
        if at is not None:
            self._hint_at = at
        if "hint" not in self._timers:
            self._later("hint", 0.1, self._send_hint)

    async def _send_hint(self) -> None:
        """Where the user's pointer is (the page's cursor there) and which field has focus: ten a
        second at most, only on change, only while the user has the page."""
        s, page = self.s, self.page
        if s is None or page is None or s.controller != "user":
            return
        at = self._hint_at
        try:
            got = await asyncio.wait_for(page.evaluate(_JS_HINT, [at[0], at[1]] if at else [None, None]), 1)
        except Exception:  # noqa: BLE001
            return
        if not isinstance(got, dict) or s is not self.s or s.controller != "user":
            return
        msg: dict[str, Any] = {"sid": s.sid}
        cursor = got.get("cursor")
        if isinstance(cursor, str):
            cursor = cursor if cursor in _CURSORS else "default"
            if cursor != self._hint.get("cursor"):
                msg["cursor"] = self._hint["cursor"] = cursor
        f = got.get("focus")
        focus = None
        if isinstance(f, dict) and (r := _rect(f.get("rect"))) is not None:
            kind = (
                f.get("kind")
                if f.get("kind") in ("text", "password", "textarea", "contenteditable", "select", "other")
                else "other"
            )
            focus = {"rect": r, "kind": kind}
        if "focus" not in self._hint or focus != self._hint["focus"]:
            msg["focus"] = self._hint["focus"] = focus  # null said out loud: nothing has focus
        if len(msg) > 1:
            self._broadcast("browser_hint", msg)

    # ---- the websocket's browser_* commands ------------------------------------------------------------------------
    def command(self, conn: Any, m: dict, size: int = 0) -> None:
        """One browser_* command from an overlay (never through the event bus, never logged)."""
        self._ensure_loop()
        v = self.viewer(conn)
        cmd = m.get("cmd")
        paste = cmd == "browser_input" and m.get("type") == "paste"
        if size > (_PASTE_CMD_MAX if paste else _CMD_MAX):
            if cmd == "browser_control":
                self._result(v, str(m.get("req") or "")[:40], False, "invalid")
            return
        if cmd == "browser_hello":
            v.hello = True
            formats = m.get("formats") if isinstance(m.get("formats"), list) else []
            v.binary = bool(m.get("binary")) and (not formats or "jpeg" in formats)
            v.surface = "window" if m.get("surface") == "window" else "overlay"
            return
        if not v.hello and not (cmd == "browser_control" and m.get("action") in ("close", "stop")):
            return
        if cmd == "browser_view":
            self._view(v, m)
        elif cmd == "browser_ack":
            self._ack(v, m)
        elif cmd == "browser_input":
            self._input(v, m)
        elif cmd == "browser_control":
            self._spawn(self._control(v, m))

    # ---- erase / shutdown / tests -----------------------------------------------------------------------------
    async def shutdown(self, reason: str = "") -> None:
        if reason == "erased":
            self._favicons.clear()
            self._user_vals.clear()
            if self.s is not None:
                await self.end("erased", alive=False, poster=False)
            self.last = None
        elif self.s is not None:
            self.s = None
        self._cancel("warm")


def _closed(page: Any) -> bool:
    try:
        return bool(page.is_closed())
    except Exception:  # noqa: BLE001
        return True


def _all_frames(page: Any) -> list:
    main = getattr(page, "main_frame", page)
    rest = [f for f in (getattr(page, "frames", None) or []) if f is not main]
    return [main, *rest[:31]]


async def _filled_rects(
    page: Any, said: set[str] | None = None
) -> list[tuple[float, float, float, float]] | None:
    """Where the user's values are on screen — their fields, every password box, and the page's
    own text that repeats `said` — in viewport CSS px, frames included (None: the page couldn't say)."""
    out: list[tuple[float, float, float, float]] = []
    words = sorted(said or (), key=len, reverse=True)[:20]
    for i, fr in enumerate(_all_frames(page)):
        try:
            got = await asyncio.wait_for(fr.evaluate(_JS_FILLED_RECTS, words), 1)
            ox = oy = 0.0
            clip = None
            if i:
                box = await asyncio.wait_for((await fr.frame_element()).bounding_box(), 1)
                if not box:
                    continue
                ox, oy = float(box["x"]), float(box["y"])
                clip = (ox, oy, ox + float(box["width"]), oy + float(box["height"]))
        except Exception:  # noqa: BLE001
            if i == 0:
                return None
            continue
        for r in got or []:
            rr = _rect(r)
            if rr is None:
                continue
            x0, y0, x1, y1 = rr["x"] + ox, rr["y"] + oy, rr["x"] + ox + rr["w"], rr["y"] + oy + rr["h"]
            if clip is not None:
                x0, y0, x1, y1 = max(x0, clip[0]), max(y0, clip[1]), min(x1, clip[2]), min(y1, clip[3])
            if x1 > x0 and y1 > y0:
                out.append((x0, y0, x1 - x0, y1 - y0))
    return out


async def _arm_frames(page: Any, on: bool) -> None:
    for fr in _all_frames(page):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(fr.evaluate(_JS_ARM_FILLED, on), 1)


def _origin(url: str) -> str:
    try:
        p = urlsplit(url)
    except ValueError:
        return ""
    return f"{p.scheme}://{p.netloc}".lower() if p.scheme in ("http", "https") and p.netloc else ""


def _resize_jpeg(data: bytes, max_w: float, max_h: float, quality: int) -> bytes:
    import io

    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max(1, int(max_w)), max(1, int(max_h))), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality)
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return b""


def _small_jpeg(
    data: bytes, width: int, quality: int, limit: int, cover: list | tuple = (), css_w: float = 0
) -> bytes:
    """`data` at `width` px wide, as good a JPEG as fits in `limit` bytes (b"" if none does), with
    the `cover` rects (viewport CSS px, the page `css_w` wide) painted over."""
    import io

    from PIL import Image, ImageDraw

    try:
        src = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return b""
    if cover:
        k = src.width / css_w if css_w else 1.0
        draw = ImageDraw.Draw(src)
        for x, y, w, h in cover:
            draw.rectangle([x * k - 2, y * k - 2, (x + w) * k + 2, (y + h) * k + 2], fill=(42, 42, 48))
    for w, q in (
        (width, quality),
        (width, quality - 15),
        (width, quality - 25),
        (int(width * 0.8), quality - 25),
    ):
        im = src.copy()
        im.thumbnail((w, w * 4), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=max(20, q), optimize=True)
        if out.tell() <= limit:
            return out.getvalue()
    return b""


def _shrink_icon(body: bytes) -> tuple[bytes, str]:
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(body))
        im.thumbnail((32, 32))
        out = io.BytesIO()
        im.convert("RGBA").save(out, "PNG", optimize=True)
        return out.getvalue(), "image/png"
    except Exception:  # noqa: BLE001
        return b"", ""


async def _dispose(resp: Any) -> None:
    if (fn := getattr(resp, "dispose", None)) is not None:
        with contextlib.suppress(Exception):
            await fn()


async def _send(cdp: Any, method: str, params: dict) -> None:
    with contextlib.suppress(Exception):
        await cdp.send(method, params)


async def _nav(coro) -> None:
    with contextlib.suppress(Exception):
        await coro


_BUTTON_BIT = {"left": 1, "right": 2, "middle": 4}


def _held(buttons: int) -> str:
    return "left" if buttons & 1 else "right" if buttons & 2 else "middle" if buttons & 4 else "none"


def _valid(typ: str, m: dict, vp: tuple[int, int]) -> dict | None:
    """The event with every value checked and clamped (None: invalid)."""
    ev: dict[str, Any] = {"type": typ, "mods": _int(m.get("mods"), 0, 15)}
    if typ in ("mousemove", "mousedown", "mouseup", "click", "wheel"):
        x, y = _num(m.get("x"), 0, vp[0]), _num(m.get("y"), 0, vp[1])
        if x is None or y is None:
            return None
        ev.update(x=x, y=y)
        if typ == "wheel":
            dx, dy = _num(m.get("dx", 0), -10000, 10000), _num(m.get("dy", 0), -10000, 10000)
            if dx is None or dy is None:
                return None
            ev.update(dx=dx, dy=dy)
            return ev
        ev["buttons"] = _int(m.get("buttons"), 0, 7)
        if typ != "mousemove":
            button = m.get("button", "left")
            if button not in _BUTTON_BIT:
                return None
            ev.update(button=button, clicks=_int(m.get("clicks"), 1, 3, 1))
        return ev
    if typ in ("keydown", "keyup"):
        key, code = m.get("key", ""), m.get("code", "")
        if not isinstance(key, str) or not isinstance(code, str) or (not key and not code):
            return None
        if (key and not _key_ok(key)) or (code and not _NAME_OK.fullmatch(code)):
            return None
        ev.update(key=key, code=code, repeat=bool(m.get("repeat")), location=_int(m.get("location"), 0, 3))
        text = m.get("text")
        if typ == "keydown" and isinstance(text, str) and text and len(text) <= 4 and text.isprintable():
            ev["text"] = text
        return ev
    if typ in ("text", "paste"):
        text = m.get("text")
        if not isinstance(text, str) or not text:
            return None
        ev["text"] = text[: _TEXT_MAX if typ == "text" else _PASTE_MAX]
        return ev
    return None


live = Live()
add_pause_clock(live.held_seconds)
app_bus().on_job_end(live._job_ended)  # NEO.app's (an embedded agent has no card)


def _on_event(ev) -> None:
    if ev.type == "confirm_done":
        live._answered(str(ev.data.get("action_id") or ""), str(ev.data.get("outcome") or ""))
    elif ev.type == "transcript" and ev.data.get("role") == "user" and ev.data.get("final", True):
        live.user_said(str(ev.data.get("job") or ""))  # the user's own words (typed, or said)


app_bus().subscribe(_on_event)


def reset() -> None:
    """A fresh card (tests)."""
    global live
    for name in list(live._timers):
        live._cancel(name)
    fresh = Live()
    fresh.hooks = live.hooks
    fresh.channel = live.channel
    live.__dict__.update(fresh.__dict__)
