"""Bring another app's window to the front — reliably, from a background process.

Since macOS 14, "cooperative activation" lets the system ignore or defer a background process's
request to activate another app while the user is busy in a different one (NSRunningApplication,
AppleScript `activate` and `open -a` all fail or land seconds later). Window managers (yabai,
AltTab, Amethyst) switch focus with the WindowServer calls in SkyLight instead: make the process
frontmost as if the user did it, make its window key, raise it. These are private APIs — so every
call is optional, and `neo2/core/tools/computer/apps.py::bring_front` falls back to the public ones.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess

from neo2.core import effects

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
    effects.guard("focus", f"bring process {pid} to the front", via=sl._SLPSSetFrontProcessWithOptions)
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
            from neo2.core.tools.computer import ax

            ax.perform_action(win, "AXRaise", kind="focus", what=f"raise process {pid}'s window")
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


_DEFAULT_SHELLS = frozenset({"sh", "bash", "zsh", "csh", "tcsh", "ksh", "dash", "fish"})


def _shells() -> frozenset[str]:
    """The login shells this Mac lists (/etc/shells), by program name."""
    try:
        with open("/etc/shells", encoding="utf-8") as f:
            names = {line.strip().rsplit("/", 1)[-1] for line in f if line.strip().startswith("/")}
        return frozenset(names) or _DEFAULT_SHELLS
    except OSError:
        return _DEFAULT_SHELLS


class _BSDInfo(ctypes.Structure):
    """struct proc_bsdinfo (sys/proc_info.h): a process's name and its controlling terminal."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32), ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32), ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32), ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32), ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32), ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32), ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32), ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]  # fmt: skip


_PROC_PIDTBSDINFO = 3
_PROC_PIDLISTFDS = 1
_PROC_PIDFDVNODEPATHINFO = 2
_PROX_FDTYPE_VNODE = 1
# struct vnode_fdinfowithpath (sys/proc_info.h): a proc_fileinfo (24 bytes), a vnode_info (152), then
# the path (MAXPATHLEN).
_VNODE_PATH_AT = 24 + 152
_VNODE_INFO_SIZE = _VNODE_PATH_AT + 1024
_NO_TTY = 0xFFFFFFFF
_libproc: ctypes.CDLL | bool | None = None


class _FDInfo(ctypes.Structure):
    """struct proc_fdinfo: one open file of a process, and what kind it is."""

    _fields_ = [("proc_fd", ctypes.c_int32), ("proc_fdtype", ctypes.c_uint32)]


def _proclib():
    global _libproc
    if _libproc is None:
        try:
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            lib.proc_listchildpids.restype = ctypes.c_int
            lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            lib.proc_pidinfo.restype = ctypes.c_int
            lib.proc_pidfdinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            lib.proc_pidfdinfo.restype = ctypes.c_int
            _libproc = lib
        except (OSError, AttributeError):
            _libproc = False
    return _libproc or None


def _children(pid: int) -> list[int]:
    lib = _proclib()
    if lib is None:
        return []
    buf = (ctypes.c_int * 512)()
    n = lib.proc_listchildpids(int(pid), buf, ctypes.sizeof(buf))
    return [int(buf[i]) for i in range(max(0, min(n, 512))) if buf[i] > 0]


def _proc(pid: int) -> tuple[str, bool]:
    """(program name, whether it runs on a terminal device) of a process; ("", False) if unknown."""
    lib = _proclib()
    if lib is None:
        return "", False
    info = _BSDInfo()
    if lib.proc_pidinfo(int(pid), _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return "", False
    return info.pbi_comm.decode(errors="replace"), info.e_tdev != _NO_TTY


def _pty_master(path: str) -> bool:
    """A pseudo-terminal's master side, as an open file's path shows it."""
    return path == "/dev/ptmx" or (path.startswith("/dev/pty") and len(path) == len("/dev/ptyp0"))


def _holds_pty(pid: int) -> bool:
    """Does this process itself hold a pseudo-terminal's master side — the end a terminal draws
    from and types into? (Read from its open files; this user's own processes only.)"""
    lib = _proclib()
    if lib is None or not pid or pid <= 1:
        return False
    fds = (_FDInfo * 2048)()
    n = lib.proc_pidinfo(int(pid), _PROC_PIDLISTFDS, 0, fds, ctypes.sizeof(fds))
    for i in range(max(0, min(n // ctypes.sizeof(_FDInfo), 2048))):
        if fds[i].proc_fdtype != _PROX_FDTYPE_VNODE:
            continue
        info = ctypes.create_string_buffer(_VNODE_INFO_SIZE)
        if lib.proc_pidfdinfo(int(pid), fds[i].proc_fd, _PROC_PIDFDVNODEPATHINFO, info, _VNODE_INFO_SIZE) != _VNODE_INFO_SIZE:
            continue
        path = info.raw[_VNODE_PATH_AT:].split(b"\0", 1)[0].decode(errors="replace")
        if _pty_master(path):
            return True
    return False


def _shell_below(pid: int, depth: int) -> bool:
    """A shell this Mac lists in /etc/shells, on a terminal device, among the process's children
    (or theirs, ``depth`` levels down)."""
    shells = _shells()
    level = [pid]
    for _ in range(depth):
        nxt: list[int] = []
        for p in level:
            for child in _children(p):
                name, tty = _proc(child)
                if tty and name.lstrip("-") in shells:
                    return True
                nxt.append(child)
        level = nxt[:256]
    return False


def hosts_shell(pid: int, depth: int = 3) -> str:
    """Does this app run an interactive shell, where a Return runs whatever is on the command line?
    Told by its own processes, never by its name:

    "terminal" — the app itself holds a pseudo-terminal's master side: a terminal app, every view
    of it a command line (iTerm2 too, whose shells run under a server that isn't its child);
    "pane" — a shell on a terminal device runs under the app's helpers only: a terminal pane in an
    editor or a chat app, a command line only while the keyboard is in it (ax.focused_terminal);
    "" — neither."""
    if not pid or pid <= 1:
        return ""
    if _holds_pty(pid):
        return "terminal"
    return "pane" if _shell_below(pid, depth) else ""
