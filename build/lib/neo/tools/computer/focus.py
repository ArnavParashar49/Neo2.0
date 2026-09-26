"""Bring another app's window to the front — reliably, from a background process.

Since macOS 14, "cooperative activation" lets the system ignore or defer a background process's
request to activate another app while the user is busy in a different one (NSRunningApplication,
AppleScript `activate` and `open -a` all fail or land seconds later). Window managers (yabai,
AltTab, Amethyst) switch focus with the WindowServer calls in SkyLight instead: make the process
frontmost as if the user did it, make its window key, raise it. These are private APIs — so every
call is optional, and `neo/tools/computer/apps.py::bring_front` falls back to the public ones.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess

_SKYLIGHT = "/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight"
_HISERVICES = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
_USER_GENERATED = 0x200  # kCPSUserGenerated: the switch counts as the user's own


class _PSN(ctypes.Structure):
    _fields_ = [("high", ctypes.c_uint32), ("low", ctypes.c_uint32)]


_libs: tuple | None = None


def _load():
    global _libs
    if _libs is None:
        try:
            sl = ctypes.cdll.LoadLibrary(_SKYLIGHT)
            hi = ctypes.cdll.LoadLibrary(_HISERVICES)
            sl._SLPSSetFrontProcessWithOptions.argtypes = [ctypes.POINTER(_PSN), ctypes.c_uint32, ctypes.c_uint32]
            sl._SLPSSetFrontProcessWithOptions.restype = ctypes.c_int32
            sl.SLPSPostEventRecordTo.argtypes = [ctypes.POINTER(_PSN), ctypes.c_char_p]
            sl.SLPSPostEventRecordTo.restype = ctypes.c_int32
            hi.GetProcessForPID.argtypes = [ctypes.c_int32, ctypes.POINTER(_PSN)]
            hi.GetProcessForPID.restype = ctypes.c_int32
            _libs = (sl, hi)
        except (OSError, AttributeError):
            _libs = ()
    return _libs or None


def front_window_id(pid: int) -> int:
    """The app's frontmost normal window (layer 0) on screen, or 0."""
    import Quartz as Q

    wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID) or []
    for w in wins:  # front-to-back order
        if int(w.get("kCGWindowOwnerPID", -1)) == pid and int(w.get("kCGWindowLayer", 1)) == 0:
            return int(w.get("kCGWindowNumber", 0))
    return 0


def force_front(pid: int) -> bool:
    """Ask the WindowServer to make `pid` frontmost with its front window key. Returns whether
    the calls went through (not whether the app is front yet — the caller verifies that)."""
    libs = _load()
    if not libs:
        return False
    sl, hi = libs
    psn = _PSN()
    if hi.GetProcessForPID(pid, ctypes.byref(psn)) != 0:
        return False
    wid = front_window_id(pid)
    if sl._SLPSSetFrontProcessWithOptions(ctypes.byref(psn), wid, _USER_GENERATED) != 0:
        return False
    if wid:
        # Make that window key (the two event records AltTab/yabai send).
        rec = bytearray(0xF8)
        rec[0x04] = 0xF8
        rec[0x3A] = 0x10
        rec[0x3C:0x40] = wid.to_bytes(4, "little")
        rec[0x20:0x30] = b"\xff" * 0x10
        for kind in (0x01, 0x02):
            rec[0x08] = kind
            sl.SLPSPostEventRecordTo(ctypes.byref(psn), bytes(rec))
    try:  # and raise it above the app's other windows
        import ApplicationServices as AS

        app = AS.AXUIElementCreateApplication(pid)
        err, win = AS.AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
        if err == 0 and win is not None:
            AS.AXUIElementPerformAction(win, "AXRaise")
    except Exception:  # noqa: BLE001
        pass
    return True


def front_app() -> tuple[str, int]:
    """(name, pid) of the app that is really in front right now.

    Asks LaunchServices (`lsappinfo`, ~9 ms), which is always current — unlike NSWorkspace's
    `frontmostApplication`, which lags until a Cocoa run loop delivers the change. Falls back to
    the owner of the frontmost on-screen window."""
    try:
        asn = subprocess.run(["lsappinfo", "front"], capture_output=True, text=True, timeout=1).stdout.strip()
        if asn:
            out = subprocess.run(
                ["lsappinfo", "info", "-only", "pid", "-only", "name", asn], capture_output=True, text=True, timeout=1
            ).stdout
            pid = name = None
            for line in out.splitlines():
                k, _, v = line.partition("=")
                k, v = k.strip().strip('"'), v.strip().strip('"')
                if k == "pid" and v.isdigit():
                    pid = int(v)
                elif k in ("LSDisplayName", "name"):
                    name = v
            if pid:
                return name or "", pid
    except (OSError, subprocess.SubprocessError):
        pass
    import Quartz as Q

    wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID) or []
    for w in wins:
        if int(w.get("kCGWindowLayer", 1)) == 0:
            return str(w.get("kCGWindowOwnerName", "")), int(w.get("kCGWindowOwnerPID", 0))
    return "", 0


def running() -> list[tuple[str, int, str]]:
    """Fresh (name, pid, type) of every app LaunchServices knows — type is Foreground (a normal
    windowed app), UIElement or BackgroundOnly. Not NSWorkspace's cached list."""
    import re

    try:
        out = subprocess.run(["lsappinfo", "list"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps: list[tuple[str, int, str]] = []
    name = None
    for line in out.splitlines():
        head = re.match(r'\s*\d+\)\s+"([^"]*)"', line)
        if head:
            name = head.group(1)
            continue
        m = re.match(r"\s*pid = (\d+).*?type=\"(\w+)\"", line)
        if m and name is not None:
            apps.append((name, int(m.group(1)), m.group(2)))
            name = None
    return apps


def pid_running(name: str) -> list[tuple[str, int]]:
    """Fresh (name, pid) of running apps matching a name: exact names first, then normal
    windowed apps whose name contains it (so "Notes" never picks LinkedNotesUIService first)."""
    want = name.lower().removesuffix(".app")
    hits = [(n, pid, t) for n, pid, t in running() if n.lower() == want or want in n.lower()]
    hits.sort(key=lambda x: (x[0].lower() != want, x[2] != "Foreground"))
    return [(n, pid) for n, pid, _ in hits]
