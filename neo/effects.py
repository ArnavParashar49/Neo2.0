"""Real effects on the Mac, and the one switch that turns them all off.

With effects off, anything NEO would do outside its own process is refused with
:class:`EffectRefused` instead of done. They are off under pytest, and in any process started with
``NEO_NO_EFFECTS=1`` (an agent's scratch script). NEO.app sets neither: every check passes there,
the audit hook below is never installed, and nothing changes.

Each check sits at the lowest layer of its path, so a test that fakes anything above it (or the very
primitive) keeps working, while a path nobody faked is refused instead of acting:

* programs: every subprocess, ``os.system``/``posix_spawn``/``exec`` and ``webbrowser.open``, where
  Python starts it (an audit hook, PEP 578): ``open``, ``osascript``, ``shortcuts``, ``say``,
  ``afplay``, ``screencapture``, ``pmset``, ``killall``, ``defaults write``, a shell... Python, git,
  uv and the read-only probes (``lsappinfo``, ``mdfind``, ``defaults read``...) still run.
* the network: a connection to anywhere but this Mac's loopback (the model providers, Gemini Live,
  web pages, downloads), where the socket connects.
* the macOS calls with no program or socket under them: :func:`guard` right before each one
  (synthetic keyboard and mouse events, Accessibility actions and writes, bringing an app to the
  front, permission prompts, the Calendar/Reminders and Contacts stores, the speaker and the
  microphone, Playwright's Chromium, the local model server, a note file in the user's Documents).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from typing import Any


class EffectRefused(BaseException):
    """A real effect, refused because effects are off.

    A BaseException, like pytest's own outcomes: NEO survives almost any Exception by design, and a
    refusal swallowed there would let a test pass that never reached what it meant to test."""

    def __init__(self, kind: str, what: str, why: str) -> None:
        super().__init__(
            f"refused to {what}: real effects are off ({why}). Fake this step - or, when the real "
            f"thing is meant, neo.effects.allow({kind!r})."
        )
        self.kind, self.what = kind, what


def _under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def why_off() -> str:
    """Why real effects are off; "" when they are on (NEO.app)."""
    if os.environ.get("NEO_NO_EFFECTS", "").strip().lower() in ("1", "true", "yes", "on"):
        return "NEO_NO_EFFECTS=1"
    return "running under pytest" if _under_pytest() else ""


def on() -> bool:
    return not why_off()


_allowed: set[str] = set()


def allow(*kinds: str) -> None:
    """Let these kinds of effect through even while effects are off, for a run that opts into the
    real thing (NEO_TEST_RENDER=1: a headless Chromium and the public page it reads)."""
    _allowed.update(kinds)


def guard(kind: str, what: str, *, via: object = None) -> None:
    """Refuse to ``what`` (a ``kind`` of effect: "input", "audio"...) while effects are off.

    ``via``: the real API about to do it, where a test may have swapped that very API for a fake
    (a fake class in place of the Calendar store): a fake does nothing real, so it is let through."""
    why = why_off()
    if not why or kind in _allowed or (via is not None and _stand_in(via)):
        return
    raise EffectRefused(kind, what, why)


_REAL_APIS = ("objc", "ctypes", "sounddevice")  # where the APIs guarded with ``via`` come from


def _stand_in(api: object) -> bool:
    """A test's fake where the real API belongs. The real ones guarded with ``via`` are PyObjC's
    classes, functions and methods, C functions (ctypes) and sounddevice's streams."""
    objc = sys.modules.get("objc")
    if objc is not None and isinstance(api, objc.objc_class | objc.function | objc.selector):
        return False
    return str(getattr(api, "__module__", "") or "").partition(".")[0] not in _REAL_APIS


# ---- programs and the network: checked where Python starts or connects them ------------------------

# Programs that only read or build (Python's `platform` runs uname and file): they still run.
_READ_ONLY = frozenset(
    {"lsappinfo", "mdfind", "mdls", "ioreg", "sw_vers", "uname", "file", "sysctl", "osacompile", "git", "uv"}
)
_SPAWN = frozenset({"subprocess.Popen", "os.posix_spawn", "os.posix_spawnp", "os.exec", "os.spawn"})
_NET = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
_EVENTS = _SPAWN | _NET | {"os.system", "webbrowser.open"}
_IP = (socket.AF_INET, socket.AF_INET6)


def _program(path: Any, argv: Any) -> tuple[str, str] | None:
    """(kind, what) when starting this program is an effect; None when it may run."""
    exe = os.fsdecode(path) if path is not None else ""
    args = [os.fsdecode(argv)] if isinstance(argv, str | bytes) else [os.fsdecode(a) for a in argv or ()]
    name = os.path.basename(exe or (args[0] if args else "")).lower()
    if name.startswith("python") or name in _READ_ONLY or (name == "defaults" and args[1:2] == ["read"]):
        return None
    what = "run `" + " ".join(args or [exe])[:200] + "`"
    return ("browser" if "playwright" in exe else "process"), what


def _outside(sock: Any, address: Any) -> bool:
    """Does a connection to ``address`` leave this Mac? Unix sockets and loopback don't; a host
    name might resolve anywhere, so it does unless it's localhost."""
    if not isinstance(address, tuple) or getattr(sock, "family", None) not in _IP:
        return False
    host = str(address[0]).partition("%")[0]
    if host in ("", "localhost"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return not (ip.is_loopback or ip.is_unspecified)


def _refusal(event: str, args: tuple) -> tuple[str, str] | None:
    if event in _NET:
        address = args[1]
        if not _outside(args[0], address):
            return None
        host = f"[{address[0]}]" if ":" in str(address[0]) else address[0]
        return "network", f"connect to {host}:{address[1]}"
    if event == "os.system":
        return "process", f"run `{os.fsdecode(args[0])[:200]}` in a shell"
    if event == "webbrowser.open":
        return "process", f"open {args[0]} in the browser"
    if event == "os.spawn":  # (mode, path, args, env)
        return _program(args[1], args[2])
    return _program(args[0], args[1])  # subprocess.Popen, os.posix_spawn, os.exec: (path, argv, ...)


def _audit(event: str, args: tuple) -> None:
    if event not in _EVENTS:
        return
    why = why_off()
    if why and (refused := _refusal(event, args)) and refused[0] not in _allowed:
        raise EffectRefused(*refused, why)


_hooked = False


def _install() -> None:
    global _hooked
    if not _hooked:
        sys.addaudithook(_audit)
        _hooked = True


if why_off():  # decided when neo is first imported (neo/__init__.py imports this module)
    _install()
