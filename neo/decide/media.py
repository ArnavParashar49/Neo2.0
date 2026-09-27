"""The driver for "play something": any app on this Mac, no app names or websites in the code.

:func:`play_best` asks :func:`neo.decide.engine.choose` which apps to try (Laya + what this user
uses + what worked before), then drives each one with the most direct generic method it exposes
and moves on when that doesn't work:

* ``learned_uri``        - replay the item id this app itself reported for the same request last
                           time (its dictionary's play-by-URI command, or its own URL scheme);
* ``script_search_play`` - the app's AppleScript dictionary: search + play the first hit. Built
                           from raw «event»/«class» codes with the bundle id and the query passed
                           as argv, so no app name is baked in and the query is never spliced into
                           script source;
* ``ax_search_play``     - the app's own search box and a matching play control through the
                           Accessibility tree (localized words from the OS's own string tables;
                           the web-content switch for Electron/Chromium apps is always restored);
* ``local_file``         - a matching audio/video file of the user's (Spotlight) opened in the app;
* ``web``                - a search-engine video search picks the page (only on the site the user
                           named, if they did; a host that worked before wins ties), opened in the
                           default browser or the browser they named.

Every method is confirmed by CoreAudio - sound actually started from the app's own processes
(helpers attributed through macOS's responsible-pid, NEO's own processes excluded) - never by "the
script said OK". When that can't be told (an app already playing, or macOS before 14.2) the attempt
is 'unconfirmed': nothing is learned and NEO asks rather than start a second source. Scripting an
app first gets macOS's Automation answer (its one-time question is asked and waited for there).
Every step has a time budget, and cleanup (AX switches, child processes) runs even when the attempt
is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import glob
import ipaddress
import json
import os
import plistlib
import re
import struct
import subprocess
import threading
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from neo.config import settings
from neo.decide._objc import pooled

if TYPE_CHECKING:
    from neo.decide.inventory import App
    from neo.decide.sdef import Dictionary

__all__ = ["Outcome", "methods_for", "play", "play_best", "AudioWatcher", "lexicon", "TIMEOUTS"]

KIND = "play_media"
WEB = "web"
METHODS = ("learned_uri", "script_search_play", "ax_search_play", "local_file")
# seconds per method: (launch / get ready, act, hear sound)
TIMEOUTS: dict[str, tuple[float, float, float]] = {
    "learned_uri": (8.0, 4.0, 4.0),
    "script_search_play": (8.0, 6.0, 4.0),
    "ax_search_play": (10.0, 10.0, 8.0),
    "local_file": (5.0, 2.0, 4.0),
    WEB: (12.0, 5.0, 10.0),  # finding the video may take up to three searches (see _find_videos)
}
OVERALL_S = 30.0  # the whole play_best loop
MAX_ATTEMPTS = 8
NOW_PLAYING_S = 3.0
_POLL_S = 0.1  # CoreAudio poll interval while confirming
_AX_STEP_S = 0.25  # AX re-scan interval while waiting for the UI
NO_BLAME = frozenset({"permission", "login", "offline", "not_found", "unconfirmed"})
ASK_S = 60.0  # how long NEO waits on macOS's one-time "allow NEO to control <app>?" question
_URI_FILE = "media_uris.json"
_URI_KEEP = 300  # learned items per app


@dataclass
class Outcome:
    ok: bool
    text: str
    blame: bool = True
    reason: str = ""  # '', permission, login, offline, not_found, timeout, unconfirmed, empty


def _fail(text: str, reason: str = "") -> Outcome:
    return Outcome(False, text, blame=reason not in NO_BLAME, reason=reason)


def _engine():
    from neo.decide import engine

    return engine


def _inventory():
    from neo.decide import inventory

    return inventory


_gen: ContextVar[int | None] = ContextVar("neo_media_generation", default=None)


def _erased() -> bool:
    """The user erased NEO's memory after this request started: learn nothing from it."""
    from neo import privacy

    started = _gen.get()
    return started is not None and started != privacy.generation()


def forget() -> int:
    """Erase the item ids NEO learned (the privacy erase). Returns how many there were."""
    with _uri_lock:
        n = sum(len(v) for v in _uri_load().values() if isinstance(v, dict))
        _uri_path().unlink(missing_ok=True)
        return n


def _record(key: str, method: str, ok: bool, reason: str = "") -> None:
    if _erased():
        return
    try:
        _engine().record(KIND, key, method, ok, reason)
    except Exception as e:  # noqa: BLE001 - learning must never break playback
        print(f"[media] couldn't record the outcome ({str(e)[:60]})")


def _prior(key: str, method: str) -> float:
    try:
        return float(_engine().prior(KIND, key, method))
    except Exception:  # noqa: BLE001
        return 0.5


def _one_line(s: str, n: int = 90) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


_STOP_WORDS = frozenset({"the", "a", "an", "by", "of", "on", "in", "and", "feat", "ft", "song", "track", "video", "official"})


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"\w+", (text or "").lower()))
    return (words - _STOP_WORDS) or words


def _overlap(query: str, text: str) -> float:
    q = _tokens(query)
    return len(q & _tokens(text)) / len(q) if q else 0.0


# ================================================================ the app's scripting dictionary
_TEXTY = frozenset({"text", "string", "unicode text", "styled text", "international text", "utf8 text", "utxt", "ctxt"})
_NOT_ITEM = _TEXTY | {
    "file", "alias", "url", "any", "list of file", "list of alias", "boolean", "integer", "real",
    "number", "date", "record", "type", "location specifier", "list of text",
}  # fmt: skip
_ITEM_RX = r"(track|item|song|episode|media|video|movie|stream|title|content)"
_URI_PROP = re.compile(r"^(?!.*\b(artwork|image|cover|icon|thumb\w*|album\s+art)\b).*\b(url|uri)$", re.I)
# A code goes into script source, so it must be exactly a code: printable ASCII, no quote/escape.
_CODE4 = re.compile(r'[ !#-\[\]-~]{4}')
_CODE8 = re.compile(r'[ !#-\[\]-~]{8}')


def _ok4(code: str | None) -> bool:
    return bool(code) and bool(_CODE4.fullmatch(code))


def _ok8(code: str | None) -> bool:
    return bool(code) and bool(_CODE8.fullmatch(code))


def _alts(t: str | None) -> list[str]:
    return [x.strip().lower() for x in (t or "").split("|") if x.strip()]


def _is_text(t: str | None) -> bool:
    return any(a in _TEXTY for a in _alts(t))


def _is_item(t: str | None) -> bool:
    a = _alts(t)
    return bool(a) and not any(x in _NOT_ITEM for x in a)


def _commands(d: Dictionary | None) -> list:
    return list((getattr(d, "commands", None) or {}).values()) if d is not None else []


def _name(c) -> str:
    return " ".join(str(getattr(c, "name", "") or "").lower().split())


def _eid(c) -> str:
    return str(getattr(c, "code", "") or "")[4:8].lower()


def _class_code(d: Dictionary, name: str) -> str:
    c = (getattr(d, "classes", None) or {}).get(name)
    return c[0] if c else ""


@dataclass(frozen=True)
class _SearchPlay:
    search: str  # 8-char event code
    term: str  # 4-char code of the text parameter ('' = the text is the direct parameter)
    container: tuple[str, ...]  # class codes, outermost first: what the search runs on
    play: str  # 8-char event code of a play command that takes the found item


@dataclass(frozen=True)
class _NowPlaying:
    current: str  # the application's current-item property code
    title: str
    artist: str
    uri: str


def _search_play_spec(d: Dictionary | None) -> _SearchPlay | None:
    cmds = _commands(d)
    play = next(
        (c for c in cmds if (_name(c) == "play" or _eid(c) == "play") and _is_item(c.direct) and _ok8(c.code)), None
    )
    if play is None:
        return None
    for s in cmds:
        if not (_eid(s) == "srch" or re.match(r"search\b", _name(s))) or not _ok8(s.code):
            continue
        where = (s.direct or "").split("|")[0].strip()
        if _is_text(where):
            return _SearchPlay(s.code, "", (), play.code)
        term = next((code for _n, code, typ in s.params if _is_text(typ) and _ok4(code)), "")
        if not term:
            continue
        if not where:
            return _SearchPlay(s.code, term, (), play.code)
        path: list[str] | None = None
        cp = getattr(d, "container_path", None)
        if callable(cp) and getattr(d, "elements", None):
            try:
                path = cp(where)
            except Exception:  # noqa: BLE001
                path = None
            if path is None:
                continue  # the dictionary knows its element tree and that class isn't in it
        if path is None:  # no element tree known: the first object of that class
            codes: tuple[str, ...] = (_class_code(d, where),)
        else:  # [] = the application itself is what's searched: no direct object
            codes = tuple(_class_code(d, c) for c in path)
        if all(_ok4(c) for c in codes):
            return _SearchPlay(s.code, term, codes, play.code)
    return None


def _play_uri_command(d: Dictionary | None):
    cmds = _commands(d)
    for c in cmds:  # "play track <uri>", "play <uri>"
        if (re.match(r"play\b", _name(c)) or _eid(c) == "play") and _is_text(c.direct) and _ok8(c.code):
            return c
    for c in cmds:  # "open location <url>"
        if (str(c.code).lower() == "gurlgurl" or _name(c) in ("open location", "open url", "openurl", "open uri")) and (
            _is_text(c.direct) and _ok8(c.code)
        ):
            return c
    return None


def _now_playing_spec(d: Dictionary | None) -> _NowPlaying | None:
    if d is None:
        return None
    props: dict[str, str] = getattr(d, "properties", None) or {}
    class_props: dict = getattr(d, "class_props", None) or {}
    app_props = class_props.get("application") or {}
    cur_name, cur_code, cur_type = "", "", ""
    for pname, v in (app_props or {p: (c, "") for p, c in props.items()}).items():
        code, typ = v if isinstance(v, tuple) else (v, "")
        if code == "pTrk" or re.fullmatch(rf"current\s+{_ITEM_RX}", pname.lower()):
            cur_name, cur_code, cur_type = pname, code, typ
            break
    if not cur_name or not _ok4(cur_code):
        return None
    # the item's own properties: its class (and superclasses) when the dictionary says, else any item-ish class
    item_classes: list[str] = []
    inherits: dict = getattr(d, "inherits", None) or {}
    t = cur_type
    while t and t not in item_classes and t in (getattr(d, "classes", None) or {}):
        item_classes.append(t)
        t = inherits.get(t, "")
    if not item_classes:
        item_classes = [n for n in (getattr(d, "classes", None) or {}) if re.fullmatch(_ITEM_RX, n.strip().lower())]

    def find(rx: re.Pattern | str, codes: set[str] = frozenset()) -> str:
        for k in item_classes:
            names = list((class_props.get(k) or {}).keys()) or list(d.classes[k][1])
            for p in names:
                code = (class_props.get(k) or {}).get(p, (props.get(p, ""), ""))[0]
                if (code in codes or re.search(rx, p)) and _ok4(code):
                    return code
        return ""

    title = find(re.compile(r"^name$", re.I), {"pnam"}) or (props.get("name", "") if _ok4(props.get("name")) else "")
    if not title:
        return None
    return _NowPlaying(cur_code, title, find(re.compile(r"^artist$", re.I), {"pArt"}), find(_URI_PROP))


def _local_tags(d: Dictionary | None) -> frozenset[str]:
    out = set()
    if _search_play_spec(d):
        out.add("media.search_play")
    if _play_uri_command(d):
        out.add("media.play_uri")
    np = _now_playing_spec(d)
    if np:
        out.add("media.now_playing")
        if np.uri:
            out.add("media.item_uri")
    return frozenset(out)


def _tags(d: Dictionary | None) -> frozenset[str]:
    if d is None:
        return frozenset()
    try:
        from neo.decide.sdef import caps

        return frozenset(caps(d))
    except Exception:  # noqa: BLE001 - the same inference, from this module's own finders
        return _local_tags(d)


_RUN = "on run argv\n\tset bid to item 1 of argv\n"


def _search_play_script(sp: _SearchPlay) -> str:
    direct = ""
    if sp.container:
        direct = "(" + " of ".join(f"«class {c}» 1" for c in reversed(sp.container)) + ") "
    call = f"«event {sp.search}» {direct}given «class {sp.term}»:q" if sp.term else f"«event {sp.search}» q"
    return (
        f"{_RUN}\tset q to item 2 of argv\n"
        f"\ttell application id bid\n\t\tset hits to {call}\n"
        '\t\tif hits is missing value or hits is {} then return "EMPTY"\n'
        "\t\tif class of hits is list then set hits to item 1 of hits\n"
        f"\t\t«event {sp.play}» hits\n\tend tell\n\treturn \"OK\"\nend run"
    )


def _play_uri_script(code: str) -> str:
    return f"{_RUN}\tset u to item 2 of argv\n\ttell application id bid to «event {code}» u\n\treturn \"OK\"\nend run"


def _now_playing_script(np: _NowPlaying) -> str:
    parts = []
    for code in (np.title, np.artist, np.uri):
        if code:
            parts.append(
                f"\t\ttry\n\t\t\tset end of r to («property {code}» of «property {np.current}») as text\n"
                '\t\ton error\n\t\t\tset end of r to ""\n\t\tend try\n'
            )
        else:
            parts.append('\t\tset end of r to ""\n')
    return (
        f"{_RUN}\tif application id bid is not running then return \"NOT_RUNNING\"\n"
        "\ttell application id bid\n\t\tset r to {}\n" + "".join(parts) + "\tend tell\n"
        "\tset AppleScript's text item delimiters to linefeed\n\treturn r as text\nend run"
    )


def _kill(p) -> None:
    with contextlib.suppress(ProcessLookupError, OSError):
        p.kill()


async def _osascript(script: str, args: list[str], timeout: float) -> tuple[int, str, str]:
    """Run AppleScript with argv (values are never part of the source). osascript is killed on
    timeout and on cancellation."""
    p = await asyncio.create_subprocess_exec(
        "osascript", "-e", script, "--", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        _kill(p)
        return 124, "", f"timed out after {timeout:.0f}s"
    except asyncio.CancelledError:
        _kill(p)
        raise
    return p.returncode or 0, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


async def _open(args: list[str], timeout: float) -> tuple[int, str, str]:
    from neo.tools.computer import apps

    return await apps._run(["open", *args], timeout)


# ================================================================ may NEO script this app? (Automation)
class _AEDesc(ctypes.Structure):
    _fields_ = [("descriptorType", ctypes.c_uint32), ("dataHandle", ctypes.c_void_p)]


AE_ALLOWED, AE_DENIED, AE_NOT_ASKED, AE_NOT_RUNNING = 0, -1743, -1744, -600
_ae_lib: Any = None


def _ae():
    global _ae_lib
    if _ae_lib is None:
        try:
            lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreServices.framework/CoreServices")
            lib.AECreateDesc.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_long, ctypes.POINTER(_AEDesc)]
            lib.AECreateDesc.restype = ctypes.c_int16
            lib.AEDisposeDesc.argtypes = [ctypes.POINTER(_AEDesc)]
            lib.AEDisposeDesc.restype = ctypes.c_int16
            fn = lib.AEDeterminePermissionToAutomateTarget
            fn.argtypes = [ctypes.POINTER(_AEDesc), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
            fn.restype = ctypes.c_int32
            _ae_lib = lib
        except (OSError, AttributeError):
            _ae_lib = False
    return _ae_lib or None


def automation(bundle_id: str, ask: bool = False) -> int | None:
    """macOS's Automation answer for NEO controlling this app: AE_ALLOWED, AE_DENIED, AE_NOT_ASKED
    or AE_NOT_RUNNING; None when it can't be checked. With ``ask``, an app NEO hasn't asked about
    gets macOS's own one-time question, and this blocks until the user answers it."""
    lib = _ae()
    if lib is None or not bundle_id:
        return None
    data = bundle_id.encode()
    desc = _AEDesc()
    if lib.AECreateDesc(_fcc("bund"), data, len(data), ctypes.byref(desc)) != 0:
        return None
    try:
        return int(lib.AEDeterminePermissionToAutomateTarget(ctypes.byref(desc), _fcc("****"), _fcc("****"), ask))
    finally:
        lib.AEDisposeDesc(ctypes.byref(desc))


def _scripts(app: App | None, method: str) -> bool:
    """Does this attempt send the app Apple Events?"""
    if app is None:
        return False
    return method == "script_search_play" or (
        method == "learned_uri" and _play_uri_command(getattr(app, "script", None)) is not None
    )


async def _control_gate(app: App) -> Outcome | None:
    """None when NEO may script the app. macOS asks the user once per app; that question is put
    here, where NEO waits for the answer, rather than by a script that would time out under it."""
    st = await asyncio.to_thread(automation, app.bundle_id, False)
    if st == AE_NOT_RUNNING:  # macOS only answers for a running app: start it in the background
        ready = TIMEOUTS["script_search_play"][0]
        code, _out, _err = await _open(["-g", "-b", app.bundle_id], ready)
        deadline = time.monotonic() + ready
        while code == 0 and time.monotonic() < deadline:
            st = await asyncio.to_thread(automation, app.bundle_id, False)
            if st != AE_NOT_RUNNING:
                break
            await asyncio.sleep(0.2)
    if st == AE_NOT_ASKED:
        try:
            st = await asyncio.wait_for(asyncio.to_thread(automation, app.bundle_id, True), ASK_S)
        except TimeoutError:
            return _fail(f"macOS is still asking whether NEO may control {app.name}", "permission")
    if st == AE_DENIED:
        return _fail(f"{app.name} needs your OK before NEO can control it", "permission")
    return None  # allowed, or macOS couldn't say: the script itself will tell


def _script_fail(app_name: str, code: int, err: str) -> Outcome:
    m = re.search(r"\((-?\d+)\)\s*$", err or "")
    n = int(m.group(1)) if m else 0
    if n in (-1743, -1744):  # errAEEventNotPermitted / would require user consent
        return _fail(f"{app_name} needs your OK before NEO can control it", "permission")
    if code == 124 or n == -1712:
        return _fail(f"{app_name} didn't answer in time", "timeout")
    return _fail(f"{app_name}'s scripting failed ({n or code})")


# ================================================================ did sound start? (CoreAudio)
class _Addr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]


def _fcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


_ca_libs: tuple | None = None
_ca_lock = threading.Lock()


def _coreaudio() -> tuple | None:
    """(CoreAudio, CoreFoundation, libc) via ctypes, or None where unavailable."""
    global _ca_libs
    with _ca_lock:
        if _ca_libs is None:
            try:
                ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio") or "CoreAudio")
                cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation") or "CoreFoundation")
                ca.AudioObjectGetPropertyDataSize.argtypes = [
                    ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
                ]  # fmt: skip
                ca.AudioObjectGetPropertyData.argtypes = [
                    ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                ]  # fmt: skip
                cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
                cf.CFRelease.argtypes = [ctypes.c_void_p]
                _ca_libs = (ca, cf)
            except (OSError, AttributeError, TypeError):
                _ca_libs = ()
        return _ca_libs or None


def _u32(ca, obj: int, sel: str, ctype=ctypes.c_uint32):
    a, v, n = _Addr(_fcc(sel), _fcc("glob"), 0), ctype(0), ctypes.c_uint32(4)
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(n), ctypes.byref(v)) == 0:
        return v.value
    return None


def _cfstr(ca, cf, obj: int, sel: str) -> str:
    a, ref, n = _Addr(_fcc(sel), _fcc("glob"), 0), ctypes.c_void_p(), ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(n), ctypes.byref(ref)) or not ref.value:
        return ""
    buf = ctypes.create_string_buffer(1024)
    cf.CFStringGetCString(ref, buf, 1024, 0x08000100)
    cf.CFRelease(ref)
    return buf.value.decode(errors="replace")


_ca_ok: bool | None = None


def audio_confirmable() -> bool:
    """Can NEO see which app's sound started? (CoreAudio's process list: macOS 14.2 and later.)"""
    global _ca_ok
    if _ca_ok is None:
        ok = False
        if libs := _coreaudio():
            ca, _cf = libs
            a, n = _Addr(_fcc("prs#"), _fcc("glob"), 0), ctypes.c_uint32(0)
            ok = ca.AudioObjectGetPropertyDataSize(1, ctypes.byref(a), 0, None, ctypes.byref(n)) == 0
        _ca_ok = ok
    return _ca_ok


_hal_meta: dict[int, tuple[int, str]] = {}  # HAL process object -> (pid, client bundle id)
_hal_lock = threading.Lock()


def _hal_clients() -> list[tuple[int, int, str, bool]]:
    """Every CoreAudio client process (macOS 14.2+, no permission): (object, pid, its bundle id,
    has a running output stream)."""
    libs = _coreaudio()
    if not libs:
        return []
    ca, cf = libs
    a, n = _Addr(_fcc("prs#"), _fcc("glob"), 0), ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(1, ctypes.byref(a), 0, None, ctypes.byref(n)) or not n.value:
        return []
    ids = (ctypes.c_uint32 * (n.value // 4))()
    if ca.AudioObjectGetPropertyData(1, ctypes.byref(a), 0, None, ctypes.byref(n), ids):
        return []
    live = list(ids)
    out = []
    with _hal_lock:
        for gone in set(_hal_meta) - set(live):
            _hal_meta.pop(gone, None)
        for obj in live:
            meta = _hal_meta.get(obj)
            if meta is None:
                meta = _hal_meta[obj] = (_u32(ca, obj, "ppid", ctypes.c_int32) or 0, _cfstr(ca, cf, obj, "pbid"))
            out.append((obj, meta[0], meta[1], _u32(ca, obj, "piro") == 1))
    return out


_resp_fn: Any = None


def responsible_pid(pid: int) -> int:
    """The app macOS holds responsible for a process (helpers, XPC services, GPU processes -> the
    app), as Activity Monitor and privacy prompts attribute them; the pid itself if unknown."""
    global _resp_fn
    if _resp_fn is None:
        try:
            fn = ctypes.CDLL(None).responsibility_get_pid_responsible_for_pid
            fn.argtypes, fn.restype = [ctypes.c_int], ctypes.c_int
            _resp_fn = fn
        except (AttributeError, OSError):
            _resp_fn = False
    r = _resp_fn(pid) if _resp_fn and pid > 0 else -1
    return r if r > 0 else pid


@pooled
def _bundle_id_of_pid(pid: int) -> str:
    try:
        from AppKit import NSRunningApplication

        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(a.bundleIdentifier() or "") if a is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _proc_path(pid: int) -> str:
    if pid <= 0:
        return ""
    buf = ctypes.create_string_buffer(4096)
    try:
        n = ctypes.CDLL(None).proc_pidpath(pid, buf, 4096)
    except (AttributeError, OSError):
        return ""
    return buf.value.decode(errors="replace") if n > 0 else ""


def _self_family() -> frozenset[int]:
    """NEO's own processes: this backend, its parent (NEO.app when it launched us) and whatever
    macOS holds responsible for them - so NEO's voice and its UI's audio never count."""
    me, parent = os.getpid(), os.getppid()
    fam = {me, responsible_pid(me)}
    if parent > 1:
        fam |= {parent, responsible_pid(parent)}
    return frozenset(p for p in fam if p > 1)


@dataclass(frozen=True)
class _Proc:
    pid: int
    client: str  # the audio client's own bundle id (a helper's, often)
    owner: str  # bundle id of the app responsible for it
    path: str  # executable path of the responsible process


class AudioWatcher:
    """Which apps have an output stream running right now (NEO excluded)."""

    def __init__(self) -> None:
        self._self = _self_family()
        self._meta: dict[tuple[int, int], _Proc | None] = {}

    def _describe(self, obj: int, pid: int, client: str) -> _Proc | None:
        key = (obj, pid)
        if key not in self._meta:
            rpid = responsible_pid(pid)
            if pid in self._self or rpid in self._self:
                self._meta[key] = None
            else:
                owner = _bundle_id_of_pid(rpid) or client
                self._meta[key] = _Proc(pid, client, owner, _proc_path(rpid) or _proc_path(pid))
        return self._meta[key]

    def playing(self) -> list[_Proc]:
        out = []
        for obj, pid, client, active in _hal_clients():
            if active and (p := self._describe(obj, pid, client)) is not None:
                out.append(p)
        return out

    def outputs(self) -> dict[str, set[int]]:
        """{owning app bundle id: pids with a running output stream}."""
        out: dict[str, set[int]] = {}
        for p in self.playing():
            out.setdefault(p.owner, set()).add(p.pid)
        return out


@lru_cache(maxsize=256)
def _nested_ids(app_path: str) -> frozenset[str]:
    """Bundle ids of the helpers an app ships (read from its bundle only)."""
    out = set()
    pats = (
        "Contents/Frameworks/*.app", "Contents/Frameworks/*.framework/Versions/*/Helpers/*.app",
        "Contents/Helpers/*.app", "Contents/XPCServices/*.xpc", "Contents/Library/LoginItems/*.app", "Wrapper/*.app",
    )  # fmt: skip
    for pat in pats:
        for b in glob.glob(os.path.join(glob.escape(app_path), pat)):
            for plist in (os.path.join(b, "Contents/Info.plist"), os.path.join(b, "Info.plist")):
                try:
                    with open(plist, "rb") as f:
                        bid = plistlib.load(f).get("CFBundleIdentifier")
                except (OSError, plistlib.InvalidFileException, ValueError, AttributeError):
                    continue
                if bid:
                    out.add(str(bid).lower())
                break
    return frozenset(out)


def _belongs(p: _Proc, target: Any) -> bool:
    bid = str(getattr(target, "bundle_id", "") or "").lower()
    path = str(getattr(target, "path", "") or "").rstrip("/")
    if bid:
        for b in (p.owner.lower(), p.client.lower()):
            if b and (b == bid or b.startswith(bid + ".")):
                return True
    if path:
        if p.path.startswith(path + "/"):
            return True
        if p.client and p.client.lower() in _nested_ids(path):
            return True
    return False


class _Watch:
    """Watch the targets' output streams from before an action until sound starts.

    Started = a process of the target outputs that wasn't before the action, or one that stopped
    and started again since (a new stream). 'unknown' = it was outputting before and still is,
    with no new stream seen (e.g. switching tracks in a player that keeps its stream open).
    'unavailable' = this Mac can't tell (CoreAudio's process list needs macOS 14.2)."""

    def __init__(self, targets: list[Any]) -> None:
        self.targets = [t for t in targets if t is not None]
        self.w = AudioWatcher()
        self.before: set[int] = set()
        self.seen: set[int] = set()
        self.off: set[int] = set()
        self.now: set[int] = set()
        self.hit = asyncio.Event()
        self.available = True
        self._task: asyncio.Task | None = None

    def _pids(self) -> set[int]:
        return {p.pid for p in self.w.playing() if any(_belongs(p, t) for t in self.targets)}

    def _update(self, now: set[int]) -> None:
        self.off |= (self.before | self.seen) - now
        self.seen |= now
        self.now = now
        if (now - self.before) or (now & self.off):
            self.hit.set()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(_POLL_S)
            try:
                now = await asyncio.to_thread(self._pids)
            except Exception:  # noqa: BLE001
                continue
            self._update(now)

    async def __aenter__(self) -> _Watch:
        self.available = await asyncio.to_thread(audio_confirmable)
        if not self.available:
            return self
        try:
            self.before = await asyncio.to_thread(self._pids)
        except Exception:  # noqa: BLE001
            self.before = set()
        self.now = set(self.before)
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def wait(self, budget: float) -> str:
        if not self.available:
            await asyncio.sleep(min(max(0.0, budget), 1.5))  # give the app a moment to start
            return "unavailable"
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.hit.wait(), max(0.05, budget))
        if self.hit.is_set():
            return "started"
        return "unknown" if (self.before and self.now) else "silent"


# ================================================================ the app's UI (Accessibility)
def _ax_mod():
    from neo.tools.computer import ax

    return ax


def _attr(el, name: str, default=None):
    if el is None:
        return default
    return _ax_mod()._attr(el, name, default)


def _set(el, name: str, value) -> bool:
    import ApplicationServices as AS

    try:
        return AS.AXUIElementSetAttributeValue(el, name, value) == 0
    except Exception:  # noqa: BLE001
        return False


def _perform(el, action: str) -> bool:
    import ApplicationServices as AS

    try:
        return AS.AXUIElementPerformAction(el, action) == 0
    except Exception:  # noqa: BLE001
        return False


def _actions(el) -> list[str]:
    import ApplicationServices as AS

    try:
        err, names = AS.AXUIElementCopyActionNames(el, None)
    except Exception:  # noqa: BLE001
        return []
    return [str(n) for n in (names or [])] if err == 0 else []


def _ax_app(pid: int):
    import ApplicationServices as AS

    el = AS.AXUIElementCreateApplication(pid)
    with contextlib.suppress(Exception):  # a hung app answers each AX call in <= 2 s, not the default ~6 s
        AS.AXUIElementSetMessagingTimeout(el, 2.0)
    return el


def _trusted() -> bool:
    try:
        return bool(_ax_mod().is_trusted())
    except Exception:  # noqa: BLE001
        return False


def _post_return(pid: int) -> None:
    """A Return keystroke to that app only (not to whatever is frontmost)."""
    import Quartz as Q

    for down in (True, False):
        Q.CGEventPostToPid(pid, Q.CGEventCreateKeyboardEvent(None, 36, down))


def _s(v) -> str:
    return str(v) if isinstance(v, str | int | float) and not isinstance(v, bool) else ""


# Localized UI words come from the OS's own string tables (looked up by their English text), so
# "Play"/"Search" match in the user's language as well as English.
_LOC_SOURCES = (
    "/System/Library/Frameworks/LinkPresentation.framework/Versions/A/Resources/Localizable.loctable",
    "/System/Library/Frameworks/Intents.framework/Versions/A/Resources/Localizable.loctable",
    "/System/Library/Frameworks/AppKit.framework/Versions/C/Resources/NSSearchBar.loctable",
    "/System/Library/Frameworks/AppKit.framework/Versions/C/Resources/FindPanel.loctable",
    "/System/Library/Frameworks/AVKit.framework/Versions/A/Resources/Localizable.loctable",
)
_BASE_WORDS = {"play": ("Play",), "search": ("Search",), "find": ("Find",), "pause": ("Pause",)}


@lru_cache(maxsize=1)
@pooled
def lexicon() -> dict[str, frozenset[str]]:
    """{'play': {'Play', 'Lecture', …}, 'search': …, 'find': …, 'pause': …} for English and the
    user's preferred languages."""
    words = {k: set(v) for k, v in _BASE_WORDS.items()}
    try:
        from Foundation import NSLocale

        prefs = [str(x) for x in (NSLocale.preferredLanguages() or [])]
    except Exception:  # noqa: BLE001
        prefs = []
    langs = {"en", *prefs, *(p.split("-")[0] for p in prefs)}
    for f in _LOC_SOURCES:
        try:
            with open(f, "rb") as fh:
                table = plistlib.load(fh)
        except (OSError, plistlib.InvalidFileException, ValueError):
            continue
        en = table.get("en") if isinstance(table, dict) else None
        if not isinstance(en, dict):
            continue
        for key, val in en.items():
            if not isinstance(val, str):
                continue
            w = val.strip().rstrip("…").strip().lower()
            if w not in words:
                continue
            for lang in langs:
                tr = table.get(lang) or table.get(lang.replace("-", "_"))
                tr = tr.get(key) if isinstance(tr, dict) else None
                if isinstance(tr, str) and 0 < len(tr.strip()) < 25:
                    words[w].add(tr.strip().rstrip("…").strip())
    return {k: frozenset(v) for k, v in words.items()}


@lru_cache(maxsize=8)
def _rx(kind: str) -> re.Pattern:
    alts = sorted({re.escape(w) for w in lexicon()[kind] if w}, key=len, reverse=True)
    return re.compile(r"(?<![\w])(" + "|".join(alts) + r")(?![\w])", re.I)


def _bfs(root, max_nodes: int = 4000, max_depth: int = 60):
    q, n = deque([(root, 0, None)]), 0
    while q and n < max_nodes:
        el, d, parent = q.popleft()
        n += 1
        yield el, d, parent
        if d < max_depth:
            q.extend((c, d + 1, el) for c in (_attr(el, "AXChildren") or []))


def _label(el) -> str:
    return " | ".join(
        _s(_attr(el, a)) for a in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp", "AXRoleDescription")
        if _s(_attr(el, a))
    )  # fmt: skip


def _subtree_text(el, max_nodes: int = 30) -> str:
    return " ".join(
        _s(_attr(c, "AXValue")) or _s(_attr(c, "AXTitle")) or _s(_attr(c, "AXDescription"))
        for c, _d, _p in _bfs(el, max_nodes, 4)
    )


@dataclass
class _Hit:
    score: float
    kind: str  # field | menu | button | row
    el: Any
    label: str


_TEXT_ROLES = ("AXTextField", "AXComboBox", "AXSearchField")


def find_search(app_el) -> list[_Hit]:
    """Ranked ways into the app's search: text fields that say they are one, then menu commands
    (only items with a keyboard shortcut are read - content menus like History never are)."""
    srx, frx, out = _rx("search"), _rx("find"), []
    for el, d, parent in _bfs(app_el):
        role, sub = _s(_attr(el, "AXRole")), _s(_attr(el, "AXSubrole"))
        if role in _TEXT_ROLES:
            if sub == "AXSecureTextField" or _attr(el, "AXEnabled", True) is False:
                continue
            lab = _label(el)
            s = 5 if "AXSearchField" in (role, sub) else 0
            s += 3 if srx.search(lab) else 0
            s += 1 if frx.search(lab) else 0
            if parent is not None and (
                _s(_attr(parent, "AXSubrole")) == "AXLandmarkSearch" or _s(_attr(parent, "AXRole")) == "AXToolbar"
            ):
                s += 1
            s += 1 if role == "AXComboBox" else 0  # ARIA combobox: the usual web-app search input
            if s >= 2:
                out.append(_Hit(s, "field", el, lab[:60]))
        elif role == "AXMenuItem" and d <= 4 and _attr(el, "AXMenuItemCmdChar"):
            t = _s(_attr(el, "AXTitle"))
            if t and (srx.search(t) or frx.search(t)):
                out.append(_Hit(2 + bool(srx.search(t)), "menu", el, t[:60]))
    return sorted(out, key=lambda h: -h.score)


def find_play_controls(root, query: str) -> list[_Hit]:
    """Controls that would play the thing asked for, best first: buttons labelled with the
    localized 'Play' naming it ('Play 505'), bare 'Play' buttons on a card/row that names it, and
    rows/cells whose text matches it. A bare transport 'Play' never qualifies."""
    prx, toks, out = _rx("play"), _tokens(query), []
    if not toks:
        return out
    for el, d, parent in _bfs(root):
        role = _s(_attr(el, "AXRole"))
        if role == "AXButton":
            lab = (_s(_attr(el, "AXDescription")) or _s(_attr(el, "AXTitle"))).strip()
            if not prx.match(lab):
                continue
            rest = _tokens(prx.sub("", lab, count=1))
            if rest:  # 'Play <name>': it must name what was asked for
                ov = len(toks & rest) / len(toks)
            elif parent is not None:  # a bare 'Play' on a card/row: the card must name it
                ov = 0.75 * len(toks & _tokens(_subtree_text(parent))) / len(toks)
                ov = ov if ov >= 0.375 else 0.0
            else:
                ov = 0.0
            if ov > 0:
                out.append(_Hit(1 + 2 * ov - 0.001 * d, "button", el, lab[:90]))
        elif role in ("AXRow", "AXCell"):
            ov = len(toks & _tokens(_subtree_text(el))) / len(toks)
            if ov >= 0.5:
                out.append(_Hit(ov - 0.001 * d, "row", el, ""))
    return sorted(out, key=lambda h: -h.score)


@lru_cache(maxsize=256)
def _runtime(app_path: str) -> str:
    """electron | cef | chromium | native - which switch exposes its web content to AX."""
    fw = os.path.join(app_path, "Contents", "Frameworks")
    try:
        names = set(os.listdir(fw))
    except OSError:
        return "native"
    if "Electron Framework.framework" in names:
        return "electron"
    if "Chromium Embedded Framework.framework" in names:
        return "cef"
    if any(n.endswith(" Framework.framework") and os.path.isdir(os.path.join(fw, n, "Versions/Current/Helpers")) for n in names):
        return "chromium"
    return "native"


def _enable_web_tree(app_el, runtime: str) -> list[tuple[str, bool]]:
    """Expose an Electron/Chromium app's web content to AX; returns what to restore."""
    if runtime == "native":
        return []
    order = ("AXManualAccessibility", "AXEnhancedUserInterface") if runtime == "electron" else ("AXEnhancedUserInterface",)
    for name in order:
        prev = _attr(app_el, name)
        if prev is True or prev == 1:
            return []  # already on (the user's own assistive setup): leave it alone
        if _set(app_el, name, True):
            return [(name, bool(prev))]
    return []


def _restore(app_el, changed: list[tuple[str, bool]]) -> None:
    for name, prev in reversed(changed):
        with contextlib.suppress(Exception):
            _set(app_el, name, prev)


def _window(app_el):
    wins = _attr(app_el, "AXWindows") or []
    return _attr(app_el, "AXFocusedWindow") or _attr(app_el, "AXMainWindow") or (wins[0] if wins else None)


def _needs_sign_in(win) -> bool:
    return any(
        "AXSecureTextField" in (_s(_attr(el, "AXRole")), _s(_attr(el, "AXSubrole"))) for el, _d, _p in _bfs(win, 1500, 40)
    )


def _set_query(field, text: str) -> bool:
    _set(field, "AXFocused", True)
    ok = _set(field, "AXValue", text)
    if ok:
        v = _attr(field, "AXValue")
        ok = v is None or _s(v) == text  # some web inputs ignore a value set
    _perform(field, "AXConfirm")
    return ok


def _focus_holds(app_el, roles: tuple[str, ...], value: str | None = None) -> bool:
    foc = _attr(app_el, "AXFocusedUIElement")
    if foc is None or _s(_attr(foc, "AXRole")) not in roles:
        return False
    return value is None or _s(_attr(foc, "AXValue")) == value


def _press(hit: _Hit, pid: int, app_el) -> bool:
    if "AXPress" in _actions(hit.el) and _perform(hit.el, "AXPress"):
        return True
    # A row: select it, then Return plays the selection in list-based players - only while the list
    # has the focus, so the keystroke can never land on a dialog's default button.
    lists = ("AXTable", "AXOutline", "AXList", "AXRow", "AXCell", "AXGrid")
    if hit.kind == "row" and _set(hit.el, "AXSelected", True) and _focus_holds(app_el, lists):
        _post_return(pid)
        return True
    return False


@pooled
def _ax_flow(pid: int, app_path: str, query: str, stop: threading.Event, ready_s: float, action_s: float) -> tuple[bool, str, str]:
    """Worker thread: type the query into the app's own search box and press the best matching
    play control. (pressed, label or why not, reason). The web-content switch is restored however
    this ends; `stop` ends it early (the async side was cancelled or ran out of time)."""
    app_el = _ax_app(pid)
    changed = _enable_web_tree(app_el, _runtime(app_path))
    t0 = time.monotonic()
    try:
        field, menu_tried = None, False
        while True:
            if stop.is_set():
                return False, "stopped", "timeout"
            win = _window(app_el)
            if win is not None:
                if _needs_sign_in(win):
                    return False, "it wants you to sign in", "login"
                hits = find_search(app_el)
                fields = [h for h in hits if h.kind == "field"]
                if fields:
                    field = fields[0].el
                    break
                menus = [h for h in hits if h.kind == "menu"]
                if menus and not menu_tried:
                    menu_tried = True
                    _perform(menus[0].el, "AXPress")
                    time.sleep(0.3)
                    if _focus_holds(app_el, _TEXT_ROLES):
                        field = _attr(app_el, "AXFocusedUIElement")
                        break
                    continue
            if time.monotonic() - t0 > ready_s:
                return False, "no search box found", ""
            time.sleep(_AX_STEP_S)
        if not _set_query(field, query):
            return False, "couldn't type into its search box", ""
        t1, returned = time.monotonic(), False
        while True:
            if stop.is_set():
                return False, "stopped", "timeout"
            hits = find_play_controls(_window(app_el) or app_el, query)
            if hits:
                break
            if not returned and time.monotonic() - t1 > 1.5 and _focus_holds(app_el, _TEXT_ROLES, query):
                _post_return(pid)  # web inputs that only search on a real Return
                returned = True
            if time.monotonic() - t1 > action_s:
                return False, "nothing in its results matched", ""
            time.sleep(_AX_STEP_S)
        for h in hits[:3]:
            if stop.is_set():
                return False, "stopped", "timeout"
            if _press(h, pid, app_el):
                return True, h.label, ""
        return False, "couldn't press play", ""
    finally:
        _restore(app_el, changed)


def _strip_play_word(label: str) -> str:
    return _rx("play").sub("", label or "", count=1).strip(" :,-–—") if label else ""


# ================================================================ helpers shared by the methods
@dataclass(frozen=True)
class _Target:
    bundle_id: str
    name: str
    path: str


@pooled
def _pid_of(bundle_id: str) -> int | None:
    """Fresh from LaunchServices (NSWorkspace's cached list misses apps launched after NEO)."""
    try:
        r = subprocess.run(["lsappinfo", "info", "-only", "pid", "-app", bundle_id], capture_output=True, text=True, timeout=2)
        m = re.search(r'"pid"\s*=\s*(\d+)', r.stdout)
        if m:
            return int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from AppKit import NSRunningApplication

        for a in NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id) or []:
            if not a.isTerminated():
                return int(a.processIdentifier())
    except Exception:  # noqa: BLE001
        pass
    return None


async def _launch(app: App, ready_s: float) -> int | None:
    """The app's pid, opening it (or its window) first when needed."""
    pid = await asyncio.to_thread(_pid_of, app.bundle_id)
    has_window = bool(pid) and await asyncio.to_thread(lambda: _window(_ax_app(pid)) is not None)
    if not has_window:
        from neo.tools.computer import apps

        out = await apps.open_app(app.path)
        if out.startswith("Error"):
            return None
    deadline = time.monotonic() + ready_s
    while True:
        pid = await asyncio.to_thread(_pid_of, app.bundle_id)
        if pid or time.monotonic() > deadline:
            return pid
        await asyncio.sleep(0.2)


@pooled
def _default_browser() -> Any | None:
    try:
        a = _inventory().default_for("https:")
        if a is not None:
            return a
    except Exception:  # noqa: BLE001
        pass
    try:
        from AppKit import NSWorkspace
        from Foundation import NSURL, NSBundle

        u = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(NSURL.URLWithString_("https:"))
        if u is None:
            return None
        path = str(u.path())
        b = NSBundle.bundleWithPath_(path)
        info = (b.infoDictionary() or {}) if b is not None else {}
        name = str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or Path(path).stem)
        return _Target(str(b.bundleIdentifier() or "") if b is not None else "", name, path)
    except Exception:  # noqa: BLE001
        return None


def _targets(app: App) -> list[Any]:
    """Whose sound counts for this app: the app, plus the default browser for a web-app wrapper."""
    out: list[Any] = [app]
    if getattr(app, "web_start_url", ""):
        out.append(_default_browser())
    return out


async def _now_playing(app: App) -> tuple[str, str, str] | None:
    """(title, artist, item uri) from the app's dictionary, without launching it."""
    np = _now_playing_spec(getattr(app, "script", None))
    if np is None:
        return None
    if await asyncio.to_thread(automation, app.bundle_id, False) not in (AE_ALLOWED, None):
        return None  # not running, or not allowed yet: never make macOS ask just to read a title
    try:
        code, out, _err = await _osascript(_now_playing_script(np), [app.bundle_id], NOW_PLAYING_S)
    except OSError:
        return None
    if code != 0 or not out or out == "NOT_RUNNING":
        return None
    parts = [*out.split("\n"), "", "", ""]
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _title(np: tuple[str, str, str] | None, fallback: str) -> str:
    if np and np[0]:
        return _one_line(f"{np[0]} by {np[1]}" if np[1] else np[0])
    return _one_line(fallback)


async def _confirmed(
    app: App, query: str, watch: _Watch, confirm_s: float, fallback_title: str,
    np_before: tuple[str, str, str] | None, *, learn: bool = True,
) -> Outcome:  # fmt: skip
    state = await watch.wait(confirm_s)
    if state == "silent":
        return _fail(f"{app.name} didn't start playing", "timeout")
    np = await _now_playing(app)
    if state == "unavailable":  # only the app's own now-playing can tell
        if not (np and _overlap(query, f"{np[0]} {np[1]}") >= 0.5):
            return _fail(f"asked {app.name} to play it but couldn't tell whether it started", "unconfirmed")
        state = "started"
    if state == "unknown" and np and np_before and np[0] == np_before[0] and _overlap(query, f"{np[0]} {np[1]}") < 0.5:
        return _fail(f"{app.name} kept playing what it was playing", "timeout")
    if learn and np and np[2] and _overlap(query, f"{np[0]} {np[1]}") >= 0.5 and _can_learn(app, np[2]):
        await asyncio.to_thread(_uri_store, app.bundle_id, query, np[2], _title(np, query))
    return Outcome(True, f"Playing {_title(np, fallback_title)} in {app.name}")


# ================================================================ learned item URIs
_uri_lock = threading.Lock()


def _uri_path() -> Path:
    return settings().data_dir / _URI_FILE


def _qkey(q: str) -> str:
    return " ".join(re.findall(r"\w+", (q or "").lower()))


def _uri_load() -> dict:
    try:
        d = json.loads(_uri_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _uri_save(d: dict) -> None:
    p = _uri_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(d, separators=(",", ":")))
        os.replace(tmp, p)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink()
        print(f"[media] couldn't save a learned item ({str(e)[:60]})")


def _uri_lookup(bundle_id: str, query: str) -> dict | None:
    with _uri_lock:
        hit = (_uri_load().get(bundle_id) or {}).get(_qkey(query))
    return hit if isinstance(hit, dict) and isinstance(hit.get("uri"), str) and hit["uri"] else None


def _uri_store(bundle_id: str, query: str, uri: str, title: str) -> None:
    if _erased():
        return
    if not _qkey(query) or not uri or len(uri) > 2000:
        return
    with _uri_lock:
        d = _uri_load()
        per = d.get(bundle_id) if isinstance(d.get(bundle_id), dict) else {}
        per[_qkey(query)] = {"uri": uri, "title": title, "ts": time.time()}
        if len(per) > _URI_KEEP:
            per = dict(sorted(per.items(), key=lambda kv: -float((kv[1] or {}).get("ts", 0)))[:_URI_KEEP])
        d[bundle_id] = per
        _uri_save(d)


def _uri_forget(bundle_id: str, query: str) -> None:
    with _uri_lock:
        d = _uri_load()
        if (d.get(bundle_id) or {}).pop(_qkey(query), None) is not None:
            _uri_save(d)


def _scheme(uri: str) -> str:
    m = re.match(r"([a-z][a-z0-9+.-]*):", uri or "", re.I)
    return m.group(1).lower() if m else ""


def _can_learn(app: App, uri: str = "") -> bool:
    """The app reports an item id for what it plays, and can be handed that id back (its
    play-by-URI command, or a URL scheme it declares - what :func:`_play_learned` uses)."""
    d = getattr(app, "script", None)
    np = _now_playing_spec(d)
    if not (np and np.uri):
        return False
    if _play_uri_command(d) is not None:
        return True
    if not getattr(app, "own_schemes", ()):
        return False
    return not uri or _scheme(uri) in {str(s).lower() for s in (getattr(app, "schemes", ()) or ())}


# ================================================================ which methods an app offers
_AV_EXTS = frozenset({
    "mp3", "m4a", "m4b", "aac", "wav", "aif", "aiff", "aifc", "flac", "alac", "ogg", "oga", "opus", "wma", "mka",
    "caf", "mp4", "m4v", "mov", "mkv", "avi", "webm", "wmv", "mpg", "mpeg", "ts", "m2ts", "3gp",
})  # fmt: skip


@lru_cache(maxsize=4096)
def _is_av(t: str) -> bool:
    s = (t or "").strip()
    low = s.lower().lstrip(".")
    if low in _AV_EXTS or low.startswith(("public.audio", "public.movie", "public.video", "public.audiovisual-content", "public.mp3", "public.mpeg")):
        return True
    try:
        import objc

        ut = objc.lookUpClass("UTType")
        av = ut.typeWithIdentifier_("public.audiovisual-content")
        ty = ut.typeWithIdentifier_(s) if "." in s else ut.typeWithFilenameExtension_(low)
        return bool(ty is not None and av is not None and ty.conformsToType_(av))
    except Exception:  # noqa: BLE001
        return False


def _plays_av_files(app: App) -> bool:
    return any(_is_av(str(t)) for t in getattr(app, "viewer_types", ()) or ())


@lru_cache(maxsize=4096)
def _opens_ext(viewer: tuple[str, ...], ext: str) -> bool:
    if not ext or not _is_av(ext):
        return False
    if ext in {str(v).lower().lstrip(".") for v in viewer}:
        return True
    try:
        import objc

        ut = objc.lookUpClass("UTType")
        ty = ut.typeWithFilenameExtension_(ext)
        if ty is None:
            return False
        for v in viewer:
            vt = ut.typeWithIdentifier_(str(v)) if "." in str(v) else None
            if vt is not None and ty.conformsToType_(vt):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def methods_for(app: App) -> tuple[str, ...]:
    """The generic ways this app can be driven to play something, most direct first."""
    kind = (tuple(getattr(app, "media_evidence", ()) or ()) or ("",))[0]
    if kind == "web browser":
        return (WEB,)  # named by the user ("... in <browser>"): the web method, in that browser
    if kind == "media file player":  # no catalog to search: only the user's own files
        return ("local_file",) if _plays_av_files(app) else ()
    d = getattr(app, "script", None)
    tags = _tags(d)
    out: list[str] = []
    np = _now_playing_spec(d)
    if "media.item_uri" in tags and np is not None and np.uri and (
        ("media.play_uri" in tags and _play_uri_command(d) is not None) or getattr(app, "own_schemes", ())
    ):
        out.append("learned_uri")
    if "media.search_play" in tags and _search_play_spec(d) is not None:
        out.append("script_search_play")
    files = _plays_av_files(app)
    catalog = bool(out) or bool(getattr(app, "own_schemes", ()))
    if files and not catalog:
        out.append("local_file")  # a file player: the user's own files first
    if getattr(app, "path", ""):
        out.append("ax_search_play")
    if files and catalog:
        out.append("local_file")
    return tuple(out)


# ================================================================ the methods
async def _play_script(q: str, app: App) -> Outcome:
    ready, action, confirm = TIMEOUTS["script_search_play"]
    spec = _search_play_spec(getattr(app, "script", None))
    if spec is None:
        return _fail(f"{app.name} has no scripting search", "not_found")
    np_before = await _now_playing(app)
    async with _Watch(_targets(app)) as watch:
        code, out, err = await _osascript(_search_play_script(spec), [app.bundle_id, q], ready + action)
        if code != 0:
            return _script_fail(app.name, code, err)
        if out == "EMPTY":  # its own library doesn't have it (blamed: the app learns to go later)
            return _fail(f"{app.name}'s search found nothing", "empty")
        if out != "OK":
            return _fail(f"{app.name} gave an unexpected answer")
        return await _confirmed(app, q, watch, confirm, q, np_before)


async def _play_learned(q: str, app: App) -> Outcome:
    ready, action, confirm = TIMEOUTS["learned_uri"]
    hit = await asyncio.to_thread(_uri_lookup, app.bundle_id, q)
    if hit is None:
        return _fail(f"nothing learned for this in {app.name} yet", "not_found")
    uri, cmd = hit["uri"], _play_uri_command(getattr(app, "script", None))
    if cmd is None and _scheme(uri) not in {s.lower() for s in (getattr(app, "schemes", ()) or ())}:
        return _fail(f"{app.name} can't take that item", "not_found")
    np_before = await _now_playing(app)
    async with _Watch(_targets(app)) as watch:
        if cmd is not None:
            code, _out, err = await _osascript(_play_uri_script(cmd.code), [app.bundle_id, uri], ready + action)
            res = _script_fail(app.name, code, err) if code != 0 else None
        else:
            code, _out, err = await _open(["-b", app.bundle_id, uri], ready)
            res = _fail(f"couldn't open it in {app.name}") if code != 0 else None
        if res is None:
            res = await _confirmed(app, q, watch, confirm, str(hit.get("title") or q), np_before, learn=False)
    if not res.ok and res.blame:
        await asyncio.to_thread(_uri_forget, app.bundle_id, q)  # a stale id: find it afresh next time
    return res


async def _play_ax(q: str, app: App) -> Outcome:
    ready, action, confirm = TIMEOUTS["ax_search_play"]
    if not await asyncio.to_thread(_trusted):
        return _fail("NEO's Accessibility permission is off", "permission")
    t0 = time.monotonic()
    pid = await _launch(app, ready)
    if not pid:
        return _fail(f"couldn't open {app.name}")
    np_before = await _now_playing(app)
    stop = threading.Event()
    async with _Watch(_targets(app)) as watch:
        try:
            ok, detail, reason = await asyncio.to_thread(
                _ax_flow, pid, app.path, q, stop, max(1.0, ready - (time.monotonic() - t0)), action
            )
        finally:
            stop.set()  # a cancelled attempt: the worker stops at its next step and restores the switches
        if not ok:
            return _fail(detail, reason)
        return await _confirmed(app, q, watch, confirm, _strip_play_word(detail) or q, np_before)


async def _local_media(q: str) -> list[str]:
    """The user's own audio/video files whose title, artist or name has every word of the query."""
    from neo.tools.computer import apps

    words = [w for w in re.findall(r"[\w'-]+", q) if w.strip("'-")][:6]
    if not words:
        return []
    cond = " && ".join(
        f'(kMDItemTitle == "*{w}*"cd || kMDItemAuthors == "*{w}*"cd || kMDItemDisplayName == "*{w}*"cd)' for w in words
    )
    query = f'(kMDItemContentTypeTree == "public.audio" || kMDItemContentTypeTree == "public.movie") && {cond}'
    code, out, _err = await apps._run(["mdfind", "-onlyin", str(Path.home()), query], 3.0)
    return [p for p in out.splitlines() if p][:50] if code == 0 else []


async def _play_local(q: str, app: App) -> Outcome:
    ready, _action, confirm = TIMEOUTS["local_file"]
    viewer = tuple(str(v) for v in (getattr(app, "viewer_types", ()) or ()))
    files = [f for f in await _local_media(q) if _opens_ext(viewer, Path(f).suffix.lstrip(".").lower())]
    if not files:
        return _fail("no matching file on this Mac", "not_found")
    path = files[0]
    async with _Watch(_targets(app)) as watch:
        code, _out, _err = await _open(["-b", app.bundle_id, path], ready)
        if code != 0:
            return _fail(f"couldn't open the file in {app.name}")
        return await _confirmed(app, q, watch, confirm, Path(path).stem, None, learn=False)


# ---- the web fallback --------------------------------------------------------------------------
_LONG = re.compile(
    r"\b(album|mix|playlist|full|movie|film|episode|podcast|live|concert|hours?|compilation|audiobook|lecture|"
    r"documentary|nonstop|jukebox|season|show)\b",
    re.I,
)
_last_open: tuple[float, str, str] | None = None  # (when, title, browser) of the last page the web method opened


def _video_search(query: str, n: int = 12) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.videos(query, max_results=n))


def _duration_s(s: str) -> int | None:
    s = (s or "").strip()
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", s)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.fullmatch(r"P(?:T)?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s, re.I)
    if m and any(m.groups()):
        return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)
    return None


def _host(url: str) -> str:
    try:
        h = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    for pre in ("www.", "m."):
        h = h.removeprefix(pre)
    return h


def _public_http(url: str) -> bool:
    try:
        u = urlsplit(url)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or not host or "." not in host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        return False
    with contextlib.suppress(ValueError):
        ip = ipaddress.ip_address(host.strip("[]"))
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)
    return True


def _pick_video(rows: list[dict], query: str) -> tuple[str, str, str] | None:
    """(url, title, host) of the result to open: the top results whose length fits the request
    (a track is not a 48-minute compilation), a host that worked here before breaking ties."""
    max_s = 4 * 3600 if _LONG.search(query) else 15 * 60
    cands = []
    for i, r in enumerate(rows or []):
        url = str(r.get("content") or r.get("url") or r.get("href") or "")
        if not _public_http(url):
            continue
        d = _duration_s(str(r.get("duration") or ""))
        cands.append((d is not None and 20 <= d <= max_s, i, url, _one_line(str(r.get("title") or ""), 80)))
    pool = [c for c in cands if c[0]] or cands
    if not pool:
        return None
    best = max(pool[:6], key=lambda c: _prior(f"{WEB}:{_host(c[2])}", WEB) - 0.08 * c[1] + 0.1 * _overlap(query, c[3]))
    return best[2], best[3] or query, _host(best[2])


def _on_site(host: str, site: str) -> bool:
    """Is this result on the site the user named ('youtube' ~ youtube.com, 'youtube music' ~
    music.youtube.com, 'vimeo.com' ~ vimeo.com)?"""
    host, site = (host or "").lower(), (site or "").lower().strip()
    if not host or not site:
        return False
    if "." in site:
        s = site.removeprefix("www.")
        return host == s or host.endswith("." + s)
    labels = host.split(".")[:-1] or [host]
    words = re.findall(r"[a-z0-9]+", site)
    return "".join(words) in labels or all(any(w in lab for lab in labels) for w in words)


def _without_site(q: str, site: str) -> str:
    """'505 on youtube' -> '505' (the words to search for on that site)."""
    if not site:
        return q
    pat = rf"\s*\b(?:on|from|via|using|at)\s+(?:the\s+|my\s+)?{re.escape(site)}\s*[.!?]*\s*$"
    return re.sub(pat, "", q, flags=re.I).strip() or q


def _text_search(query: str, n: int = 10) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.text(query, max_results=n) or [])


async def _resolve_site(word: str) -> str:
    """The host a site word names ('youtube' -> its real host), by evidence (neo.decide.sites);
    '' when it can't be told (offline): results are then matched by the word itself."""
    try:
        from neo.decide import sites

        res = await sites.resolve(word)
        return res.host if res else ""
    except Exception as e:  # noqa: BLE001
        print(f"[media] couldn't resolve the site {word!r} ({str(e)[:60]})")
        return ""


_RESULTS_BYTES = 2_000_000  # enough of a results page for its first items

# A link to one playable item (a video, a track), not a listing: the shapes item pages take.
_ITEM_PATH = re.compile(r"(?:[?&]v=|/watch\b|/video/|/videos/[\w-]|/v/|/embed/|/track/|/song/|/shorts/)", re.I)
_LINK = re.compile(r'(?:href=|"url":|"href":|"canonicalBaseUrl":)\s*"((?:https?:)?//[^"\s<>]+|/[^"\s<>]+)"', re.I)


def _item_links(html: str, host: str) -> list[str]:
    """Links to single items on `host`, in the order a results page lists them: its hrefs and the
    URLs inside its JSON ('/watch?v=…'). The one parameter that names the item (v=) is kept."""
    out: dict[str, None] = {}
    for m in _LINK.finditer(html or ""):
        u = m.group(1).replace("\\u0026", "&").replace("&amp;", "&")
        u = "https:" + u if u.startswith("//") else (f"https://{host}{u}" if u.startswith("/") else u)
        if not _public_http(u) or not _on_site(_host(u), host) or not _ITEM_PATH.search(u):
            continue
        parts = urlsplit(u)
        v = [p for p in parts.query.split("&") if p.startswith("v=")]
        out[f"https://{parts.netloc}{parts.path}" + (f"?{v[0]}" if v else "")] = None
    return list(out)


async def _find_videos(what: str, site: str, budget: float) -> tuple[list[dict], str, str]:
    """(result rows {content, title, duration?}, why nothing was found when it's the network's
    fault, the site's host). Three independent ways, first that finds something wins — one search
    service refusing a request (they rate-limit) doesn't fail the request:

    1. the site's own search results page (its published search: :mod:`neo.decide.sites`,
       learned after the first time), first item links on it — no third party to refuse;
    2. an ordinary web search limited to the site (``site:<host>``);
    3. a video search (retried once), which also knows each video's length.
    Without a site, only the video search can look everywhere."""
    deadline = time.monotonic() + budget
    host = await _resolve_site(site) if site else ""
    match = host or site
    errors: list[str] = []

    def left(cap: float) -> float:
        return max(0.5, min(cap, deadline - time.monotonic()))

    def on(rows: list[dict]) -> list[dict]:
        if not match:
            return rows
        return [r for r in rows if _on_site(_host(str(r.get("content") or r.get("url") or r.get("href") or "")), match)]

    async def site_search() -> list[dict]:
        try:
            rows = await asyncio.wait_for(asyncio.to_thread(_text_search, f"{what} site:{host}"), left(4.0))
        except Exception as e:  # noqa: BLE001 - timeouts, network errors, rate limits
            errors.append(f"web search: {str(e)[:60] or type(e).__name__}")
            print(f"[media] web search failed ({errors[-1]})")
            return []
        hits = [
            {"content": str(r.get("href") or ""), "title": str(r.get("title") or "")}
            for r in rows or []
            if _ITEM_PATH.search(str(r.get("href") or ""))
        ]
        return on(hits)

    async def video_search() -> list[dict]:
        for attempt in range(2):
            try:
                rows = await asyncio.wait_for(asyncio.to_thread(_video_search, f"{what} {site}".strip()), left(4.0))
                return on(rows or [])  # it answered: whatever it found on the site
            except Exception as e:  # noqa: BLE001
                errors.append(f"video search: {str(e)[:60] or type(e).__name__}")
                print(f"[media] video search failed ({errors[-1]})")
                if attempt == 0 and time.monotonic() < deadline - 1:
                    await asyncio.sleep(0.6)
        return []

    async def site_page() -> list[dict]:
        try:
            from neo.decide import sites

            t = sites.learned_template(host)
            if t is None:
                found = await asyncio.wait_for(sites.discover(host), left(4.0))
                t = found[0] if found else None
            if t is None:
                return []
            page = await asyncio.wait_for(sites._fetch(sites.fill(t.url, what), _RESULTS_BYTES, partial=True), left(5.0))
            links = await asyncio.to_thread(_item_links, page.text, host) if page is not None else []
            if links and not _erased():
                sites.learn_landing(site, host, t.url)  # this template works: no discovery next time
            return [{"content": u, "title": what} for u in links[:6]]
        except Exception as e:  # noqa: BLE001
            errors.append(f"site search: {str(e)[:60] or type(e).__name__}")
            print(f"[media] the site's own search failed ({errors[-1]})")
            return []

    ways = (site_page, site_search, video_search) if host else (video_search,)
    for way in ways:
        if time.monotonic() >= deadline - 0.25:
            break
        if rows := await way():
            return rows, "", host
    return [], ("couldn't reach a video search (" + "; ".join(errors) + ")") if errors else "", host


async def _play_web(q: str, site: str = "", browser: Any | None = None) -> Outcome:
    """Search for a video and open it: in the default browser, or in `browser` when the user named
    one; only on `site` when they named a website."""
    global _last_open
    ready, action, confirm = TIMEOUTS[WEB]
    named = str(getattr(browser, "bundle_id", "") or "") if browser is not None else ""
    if browser is None:
        browser = await asyncio.to_thread(_default_browser)
    if browser is None:
        return _fail("there's no default browser")
    what = _without_site(q, site)
    rows, why, site_host = await _find_videos(what, site, ready)
    if not rows:
        return _fail(why or (f"no video on {site}" if site else "no video found"), "offline" if why else "not_found")
    pick = _pick_video(rows, what)
    if pick is None:
        return _fail("no video found", "not_found")
    url, title, host = pick
    key, state, opened = f"{WEB}:{host}", "", False
    name = str(getattr(browser, "name", "") or "your browser")
    try:
        async with _Watch([browser]) as watch:
            if named:
                code, _out, err = await _open(["-b", named, url], action)
                out = f"Error: {_one_line(err, 60)}" if code != 0 else "opened"
            else:
                from neo.tools.computer import apps

                out = await asyncio.wait_for(apps.open_app(url), action)
            if out.startswith(("Error", "NEEDS_")):
                return _fail(f"couldn't open the page ({_one_line(out, 60)})")
            opened = True
            _last_open = (time.monotonic(), title, name)
            state = await watch.wait(confirm)
    finally:
        # this host's own record, also when the attempt is cut short; nothing is learned when NEO
        # couldn't tell (the browser was already playing something, or no CoreAudio process list)
        if opened and state not in ("unknown", "unavailable"):
            _record(key, WEB, state == "started", "" if state == "started" else "timeout")
    if state == "silent":
        return _fail(f"opened {title} in {name} but it didn't start playing", "timeout")
    if state != "started":
        return _fail(f"opened {title} in {name} but couldn't tell whether it started", "unconfirmed")
    if site and site_host and not _erased():
        with contextlib.suppress(Exception):  # it played there: next time the word needs no search
            from neo.decide import sites

            sites.learn_landing(site, site_host, None)
    return Outcome(True, f"Playing {title} in {name}")


# ================================================================ entry points
async def play(query: str, app: App | None, method: str, site: str = "") -> Outcome:
    """One attempt: play `query` with `app` by `method` (app None + 'web' = the web fallback),
    confirmed by sound actually starting."""
    q = " ".join(str(query or "").split()).strip(" .!?\"'")
    if not q:
        return _fail("play what?", "not_found")
    try:
        if app is None or method == WEB:
            return await _play_web(q, site, app)
        if method == "learned_uri":
            return await _play_learned(q, app)
        if method == "script_search_play":
            return await _play_script(q, app)
        if method == "ax_search_play":
            return await _play_ax(q, app)
        if method == "local_file":
            return await _play_local(q, app)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - one broken attempt must not end the loop
        return _fail(f"{getattr(app, 'name', 'the web')} failed ({str(e)[:80]})")
    return _fail(f"unknown method {method!r}", "not_found")


_LABELS = {
    "learned_uri": "saved item",
    "script_search_play": "scripting",
    "ax_search_play": "search box",
    "local_file": "your files",
    WEB: "web",
}


def _strip_named(query: str) -> str:
    """'505 on <app>' -> '505' when <app> is installed: the words to search inside that app."""
    try:
        app = _inventory().named_in(query)
    except Exception:  # noqa: BLE001
        return query
    if app is None:
        return query
    new = query
    names = {app.name, *(getattr(app, "aliases", ()) or ()), *(getattr(app, "own_schemes", ()) or ())}
    for n in sorted(names, key=len, reverse=True):
        if not n:
            continue
        pat = rf"\s*\b(?:on|in|with|using|via|through|from)\s+(?:the\s+|my\s+)?{re.escape(n)}(?:\s+app)?\b"
        new = re.sub(pat, " ", query, flags=re.I)
        if new != query:
            break
    new = " ".join(new.split()).strip(" ,.!?")
    return new or query


class _Fallback:
    app = None
    methods = (WEB,)
    key = WEB
    site = ""


def _late_start(tried: list[tuple[Any, str, str]], base: set[int]) -> tuple[Any, str, str] | None:
    """(app, key, method) of an attempt that gave up but whose app started playing afterwards -
    so NEO doesn't start a second source on top of it."""
    if not tried:
        return None
    try:
        playing = AudioWatcher().playing()
    except Exception:  # noqa: BLE001 - a check, never a reason to fail the request
        return None
    for p in playing:
        if p.pid not in base:
            for t in tried:
                if _belongs(p, t[0]):
                    return t
    return None


async def play_best(query: str, text: str = "") -> Any:
    """'Neo, play <query>': try the chosen apps in order (overall ~25 s), learn from each attempt,
    and say what happened. ``text`` is the user's own words when known (never the query or tool
    arguments): the decision reads them, and only they are reported for the reflex's learning."""
    from neo import privacy
    from neo.agent.confirm import ask_user

    _gen.set(privacy.generation())
    t0 = time.monotonic()
    q = await asyncio.to_thread(_strip_named, " ".join(str(query or "").split()))  # may scan the apps
    try:
        cands = list(await _engine().choose(KIND, text or query, said=text))
    except Exception as e:  # noqa: BLE001
        print(f"[media] choosing an app failed ({str(e)[:80]}); trying the web")
        cands = []
    if not any(getattr(c, "app", None) is None for c in cands):
        cands.append(_Fallback())
    named = [c for c in cands if getattr(c, "app", None) is not None and "named by the user" in str(getattr(c, "evidence", ""))]
    if settings().play_default == "web":
        # The user's choice: it plays in the browser unless they named an app ("… on Spotify").
        # No other app is opened, not even to see whether it has the song.
        cands = named + [c for c in cands if getattr(c, "app", None) is None][:1]
    web = next((c for c in cands if getattr(c, "app", None) is None), None)
    if web is not None and not getattr(web, "site", "") and settings().play_site:
        web.site = settings().play_site  # the site they prefer, unless they named another
    named_keys = {getattr(c, "key", "") for c in named}
    named_failed: list[str] = []
    site = next((str(getattr(c, "site", "") or "") for c in cands if getattr(c, "app", None) is None), "")
    q = _without_site(q, site)  # '505 on <site>': the apps search for '505'
    try:
        base = {p.pid for p in await asyncio.to_thread(AudioWatcher().playing)}
    except Exception:  # noqa: BLE001
        base = set()
    tried: list[str] = []
    tried_apps: list[tuple[Any, str, str]] = []  # (app, key, method) of attempts that gave up
    blocked: dict[str, list[str]] = {}
    attempts = 0
    out_of_time = False
    for c in cands:
        app = getattr(c, "app", None)
        methods = tuple(getattr(c, "methods", ()) or ()) or (methods_for(app) if app is not None else (WEB,))
        if app is None:
            methods = (WEB,)
        key = getattr(c, "key", None) or (app.bundle_id if app is not None else WEB)
        for m in methods:
            remaining = OVERALL_S - (time.monotonic() - t0)
            if remaining < 1.0 or attempts >= MAX_ATTEMPTS:
                out_of_time = True
                break
            if m == "learned_uri" and app is not None and await asyncio.to_thread(_uri_lookup, app.bundle_id, q) is None:
                continue  # nothing learned for this request: not an attempt
            if tried_apps and (late := await asyncio.to_thread(_late_start, tried_apps, base)) is not None:
                _record(late[1], late[2], True)  # it did work, just slower than its budget
                return f"Playing {_one_line(q)} in {late[0].name}"
            who = app.name if app is not None else "the web"
            if _scripts(app, m):
                asked_at = time.monotonic()
                gate = await _control_gate(app)
                t0 += time.monotonic() - asked_at  # time spent on macOS's question isn't playing time
                if gate is not None:
                    _record(key, m, False, gate.reason)
                    tried.append(f"{who} ({_LABELS.get(m, m)}: {_one_line(gate.text, 70)})")
                    blocked.setdefault(gate.reason, [])
                    if who not in blocked[gate.reason]:
                        blocked[gate.reason].append(who)
                    continue  # its search box may still work (no Apple Events needed)
                remaining = OVERALL_S - (time.monotonic() - t0)
            attempts += 1
            started = time.monotonic()
            on = str(getattr(c, "site", "") or "") if m == WEB else ""
            try:
                res = await asyncio.wait_for(
                    play(q, app, m, on), min(sum(TIMEOUTS.get(m, (5.0, 5.0, 5.0))) + 1.0, remaining)
                )
            except TimeoutError:
                res = _fail("took too long", "timeout")
            _record(key, m, res.ok, res.reason)
            if res.ok:
                if named_failed and key not in named_keys:
                    # They asked for an app and got another source: say so, and why.
                    from neo.agent.registry import ToolOutput

                    return ToolOutput(f"{res.text} ({'; '.join(named_failed)})", ok=True, say=True)
                return res.text
            if key in named_keys:
                named_failed.append(f"{who} couldn't: {_one_line(res.text, 70)}")
            tried.append(f"{who} ({_LABELS.get(m, m)}: {_one_line(res.text, 70)})")
            if app is not None:
                tried_apps.append((app, key, m))
            if m == WEB and _last_open and _last_open[0] >= started:
                _, title, browser = _last_open
                return ask_user(
                    f"I opened {title} in {browser} but couldn't hear it start - press play there if it hasn't"
                )
            if res.reason == "unconfirmed":  # it may be playing: never start a second source on top
                return ask_user(f"I asked {who} to play {_one_line(q, 60)} but couldn't tell whether it started")
            if res.reason == "empty":
                break  # the app itself says it doesn't have it: its search box won't either
            if on and res.reason == "not_found" and not any(
                getattr(x, "app", None) is None and not getattr(x, "site", "") for x in cands
            ):
                cands.append(_Fallback())  # nothing on that site: the web at large, after the apps
            if res.reason in ("permission", "login"):
                blocked.setdefault(res.reason, [])
                if who not in blocked[res.reason]:
                    blocked[res.reason].append(who)
        if out_of_time:
            break
    if tried_apps and (late := await asyncio.to_thread(_late_start, tried_apps, base)) is not None:
        _record(late[1], late[2], True)  # the last attempts failed fast while an earlier one got going
        return f"Playing {_one_line(q)} in {late[0].name}"
    if "permission" in blocked:
        who = " and ".join(blocked["permission"])
        return ask_user(
            f"{who} need{'s' if len(blocked['permission']) == 1 else ''} your OK first - allow NEO in System Settings > "
            "Privacy & Security (Automation and Accessibility), then ask me again"
        )
    if "login" in blocked:
        return ask_user(f"{' and '.join(blocked['login'])} wants you to sign in - sign in there, then ask me again")
    why = "; ".join(tried) or "nothing on this Mac can play it"
    return f"Error: couldn't play {_one_line(q, 60)} - tried {why}" + (" (ran out of time)" if out_of_time else "")
