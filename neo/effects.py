"""Real effects on the Mac, and the one switch that turns them all off.

With effects off, anything NEO would do outside its own process is refused with
:class:`EffectRefused` instead of done. They are off under pytest, and in any process started with
``NEO_NO_EFFECTS=1`` (an agent's scratch script). NEO.app sets neither: every check passes there,
the audit hook below is never installed, and nothing changes.

Each check sits at the lowest layer of its path, so a test that fakes anything above it (or the very
primitive) keeps working, while a path nobody faked is refused instead of acting:

* programs: every subprocess, ``os.system``/``posix_spawn``/``exec``, ``webbrowser.open`` and a bare
  ``_posixsubprocess.fork_exec`` (multiprocessing's), where Python starts it (an audit hook, PEP 578):
  ``open``, ``osascript``, ``shortcuts``, ``say``, ``afplay``, ``screencapture``, ``pmset``,
  ``killall``, a shell, git, uv... Only read-only probes still run, checked with their arguments where
  the same program can also act (``lsappinfo info`` but not ``lsappinfo launch``, ``defaults read``,
  ``sysctl`` without ``-w``, ``osacompile`` of a script that runs nothing as it compiles), and a
  child pytest. Any other Python child is refused: it never imports neo, so it has no switch.
  C functions that start programs looked up through ctypes (``system``, ``posix_spawn``, ``exec*``...)
  are refused too, and a signal to anything but this process's own children.
* the network: a connection to anything but a server this process itself listens on, where the socket
  connects: the model providers, Gemini Live, web pages, downloads, but also this Mac's loopback where
  something else listens (NEO.app's local model server and ws port, a local proxy app). Proxies from
  the environment are switched off. An HTTP client written in Rust or C has no Python socket, so the
  hook never sees it: ddgs's (primp, under every web search) is refused at the one Python door above it.
* the macOS calls with no program or socket under them: :func:`guard` right before each one
  (synthetic keyboard and mouse events, Accessibility actions and writes, bringing an app to the
  front, permission prompts, the Calendar/Reminders and Contacts stores, the speaker and the
  microphone, Playwright's Chromium, the local model server, a note file in the user's Documents),
  and before loading a model into memory (Laya, the on-device voice, the memory's embedder).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import ipaddress
import os
import re
import socket
import sys
import threading
import types
import weakref
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit


class EffectRefused(BaseException):
    """A real effect, refused because effects are off.

    A BaseException, like pytest's own outcomes: NEO survives almost any Exception by design, and a
    refusal swallowed there would let a test pass that never reached what it meant to test."""

    def __init__(self, kind: str, what: str, why: str) -> None:
        super().__init__(
            f"refused to {what}: real effects are off ({why}). Fake this step - or, when the real "
            f"thing is meant, run it inside `with neo.effects.allow({kind!r}):`."
        )
        self.kind, self.what = kind, what


def _under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


# Tests only: NEO.app's mode for one test (tests/test_v2_effects.py's `neo_app`), in its own thread and
# what it starts from there (its asyncio tasks, asyncio.to_thread): never in any other thread.
_as_app: contextvars.ContextVar[bool] = contextvars.ContextVar("neo_effects_as_app", default=False)


def why_off() -> str:
    """Why real effects are off; "" when they are on (NEO.app)."""
    if os.environ.get("NEO_NO_EFFECTS", "").strip().lower() in ("1", "true", "yes", "on"):
        return "NEO_NO_EFFECTS=1"
    return "running under pytest" if _under_pytest() and not _as_app.get() else ""


def on() -> bool:
    return not why_off()


_allowed: frozenset[str] = frozenset()


@contextlib.contextmanager
def allow(*kinds: str) -> Iterator[None]:
    """Let these kinds of effect through while effects are off, until the block ends: for a test that
    opts into the real thing (NEO_TEST_RENDER=1: a headless Chromium and the public page it reads).
    Process-wide while it lasts; the kinds allowed before come back afterwards."""
    global _allowed
    before = _allowed
    _allowed = frozenset(before) | frozenset(kinds)
    try:
        yield
    finally:
        _allowed = before


def guard(kind: str, what: str, *, via: object = None) -> None:
    """Refuse to ``what`` (a ``kind`` of effect: "input", "audio"...) while effects are off.

    ``via``: the real API about to do it, where a test may have swapped that very API for a fake
    (a fake class in place of the Calendar store): a fake does nothing real, so it is let through -
    unless it calls through to the real one (a spy around it), which a fake must never do."""
    why = why_off()
    if not why or kind in _allowed or (via is not None and _stand_in(via)):
        return
    raise EffectRefused(kind, what, why)


# Where the real APIs guarded with ``via`` come from: PyObjC, C functions (ctypes), sounddevice's
# streams and the model loaders.
_REAL_APIS = ("objc", "ctypes", "sounddevice", "laya", "mlx_audio", "sentence_transformers")


def _stand_in(api: object) -> bool:
    """A test's fake where the real API belongs: neither the real one nor anything that reaches it."""
    return not _reaches_real(api, set(), 0)


def _real(api: object) -> bool:
    objc = sys.modules.get("objc")
    if objc is not None and isinstance(api, objc.objc_class | objc.function | objc.selector):
        return True
    kinds = api.__mro__[:-1] if isinstance(api, type) else (api,)  # a class, or one subclassing a real one
    return any(str(getattr(k, "__module__", "") or "").partition(".")[0] in _REAL_APIS for k in kinds)


def _reaches_real(api: object, seen: set[int], depth: int) -> bool:
    if depth > 6 or id(api) in seen:
        return False
    seen.add(id(api))
    try:
        return _real(api) or any(_reaches_real(held, seen, depth + 1) for held in _held(api))
    except Exception:  # noqa: BLE001 - an object that won't be looked into (a lazy proxy): not the real API
        return False


def _held(api: object) -> list[object]:
    """What a callable may call: what it wraps, the values its code names (closure and globals), a
    partial's function and arguments, a bound method's function and object, a class's methods, the
    object a mock wraps."""
    held: list[object] = []
    if isinstance(api, functools.partial):
        held += [api.func, *api.args, *api.keywords.values()]
    elif isinstance(api, types.MethodType):
        held += [api.__func__, api.__self__]
    elif isinstance(api, types.FunctionType):
        with contextlib.suppress(ValueError):  # a closure cell not filled in yet
            names = inspect.getclosurevars(api)
            held += [*names.nonlocals.values(), *names.globals.values()]
    elif isinstance(api, type):
        for v in vars(api).values():
            held.append(v.__func__ if isinstance(v, staticmethod | classmethod) else v)
    wraps = [getattr(api, "__wrapped__", None)]
    if "_mock_wraps" in getattr(api, "__dict__", {}):
        wraps.append(api.__dict__["_mock_wraps"])
    return held + [w for w in wraps if w is not None]


# ---- programs and the network: checked where Python starts or connects them ------------------------

# Programs that only read, whatever their arguments (Python's `platform` runs uname and file).
_READ_ONLY = frozenset({"mdfind", "mdls", "ioreg", "sw_vers", "uname", "file"})
# lsappinfo also launches, kills, brings to the front and writes an app's info: only these read.
_LSAPPINFO_READS = frozenset({"front", "info", "list", "processlist", "visibleprocesslist", "find", "metainfo"})
# osacompile's options that take a value, and the ones that don't; any other refuses (unseen input).
_OSA_VALUED = frozenset({"-l", "-e", "-o", "-a", "-r", "-t", "-c"})
_OSA_FLAGS = frozenset({"-x", "-s", "-u", "-d"})
# AppleScript that runs as it compiles: a property's first value and a script object's, a `use` (loads
# a library or a framework's code) and an app named in the text (loading its dictionary can launch it).
_OSA_RUNS = re.compile(
    r"^\s*(?:property|prop|script|use)\b|\b(?:application|app)\s+(?:id\s+)?[\"“]|«class capp»|\bterms\s+from\b",
    re.I | re.M,
)

_SPAWN = frozenset({"subprocess.Popen", "os.posix_spawn", "os.posix_spawnp", "os.exec", "os.spawn"})
_NET = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
_SIGNAL = frozenset({"os.kill", "os.killpg"})
_DLSYM = frozenset({"ctypes.dlsym", "ctypes.dlsym/handle"})
_EVENTS = _SPAWN | _NET | _SIGNAL | _DLSYM | {"os.system", "webbrowser.open", "socket.bind"}
_IP = (socket.AF_INET, socket.AF_INET6)
# C functions that start or signal programs, looked up through ctypes.
_STARTS = frozenset(
    {"system", "popen", "posix_spawn", "posix_spawnp", "fork", "vfork", "kill", "killpg"}
    | {"execv", "execve", "execvp", "execvP", "execvpe", "execl", "execle", "execlp", "fexecve"}
)


def _program(path: Any, argv: Any) -> tuple[str, str] | None:
    """(kind, what) when starting this program is an effect; None when it may run."""
    exe = os.fsdecode(path) if path is not None else ""
    args = [os.fsdecode(argv)] if isinstance(argv, str | bytes) else [os.fsdecode(a) for a in argv or ()]
    name = os.path.basename(exe or (args[0] if args else "")).lower()
    if _reads_only(name, args[1:]):
        return None
    what = "run `" + " ".join(args or [exe])[:200] + "`"
    if name.startswith("python"):
        what += " (a Python child never imports neo: it has no switch of its own)"
    return ("browser" if "playwright" in exe else "process"), what


def _reads_only(name: str, args: list[str]) -> bool:
    """Whether running program ``name`` with ``args`` only reads."""
    if name in _READ_ONLY:
        return True
    if name == "lsappinfo":
        words = [a for a in args if not a.startswith("-")]
        return bool(words) and words[0].lower() in _LSAPPINFO_READS
    if name == "defaults":
        return args[:1] == ["read"]
    if name == "sysctl":  # -w or name=value sets a value, -f sets them from a file
        return not any("=" in a or (a.startswith("-") and {"w", "f"} & set(a[1:])) for a in args)
    if name == "osacompile":
        return _compiles_only(args)
    if name.startswith("python"):
        return args[:2] == ["-m", "pytest"]  # a child test run: it is under this switch itself
    return False


def _compiles_only(args: list[str]) -> bool:
    """osacompile with AppleScript that runs nothing as it compiles (-e text, or source files read here)."""
    texts: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in _OSA_VALUED:
            if i + 1 == len(args) or (a == "-l" and args[i + 1].lower() != "applescript"):
                return False
            if a == "-e":
                texts.append(args[i + 1])
            i += 2
            continue
        if a.startswith("-") and a not in _OSA_FLAGS:
            return False
        if not a.startswith("-"):
            try:
                with open(a, encoding="utf-8") as f:
                    texts.append(f.read(1 << 20))
            except (OSError, UnicodeDecodeError):  # unreadable, or already compiled: unseen
                return False
        i += 1
    return bool(texts) and not any(_OSA_RUNS.search(t) for t in texts)


# Every IP socket this process bound: its own servers (a test's, or NEO's own proxy for Chromium).
_listening: weakref.WeakSet[socket.socket] = weakref.WeakSet()
_listening_mu = threading.Lock()


def _bound(sock: Any) -> None:
    if getattr(sock, "family", None) in _IP:
        with _listening_mu, contextlib.suppress(TypeError):  # a socket with no weak reference: not recorded
            _listening.add(sock)


def _ours(port: Any) -> bool:
    """Whether this process itself listens on ``port`` (checked now: one it has closed doesn't count)."""
    with _listening_mu:
        socks = list(_listening)
    for s in socks:
        try:
            if s.getsockname()[1] == port:
                return True
        except (OSError, IndexError, TypeError):
            continue
    return False


def _loopback(host: str) -> bool | None:
    """True for this Mac's loopback, False for anywhere else, None for a name that might resolve anywhere."""
    if host in ("", "localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_loopback or ip.is_unspecified


def _outside(sock: Any, address: Any) -> bool:
    """Does a connection to ``address`` reach past this process? A Unix socket doesn't; loopback does
    unless this process itself listens there (NEO.app's local model server and ws port, a proxy app,
    AirPlay all listen on this Mac's loopback); a host name might resolve anywhere."""
    if not isinstance(address, tuple) or getattr(sock, "family", None) not in _IP:
        return False
    return not _loopback(str(address[0]).partition("%")[0]) or not _ours(address[1])


_GONE = -1  # no such process (any more)


def _parent(pid: int) -> int | None:
    """The parent of process ``pid`` (launchd's is 0): _GONE when there's no such process, None when
    it can't be told. An exited child not yet reaped (a zombie) still has its parent."""
    if sys.platform != "darwin":
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as f:
                return int(f.read().rpartition(")")[2].split()[1])
        except FileNotFoundError:
            return _GONE
        except (OSError, ValueError, IndexError):
            return None
    import ctypes

    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID: its struct kinfo_proc
    info, size = ctypes.create_string_buffer(_KINFO_PROC), ctypes.c_size_t(_KINFO_PROC)
    if _sysctl()(mib, 4, info, ctypes.byref(size), None, 0) != 0:
        return None
    if size.value == 0:
        return _GONE
    return int.from_bytes(info.raw[_E_PPID : _E_PPID + 4], sys.byteorder) if size.value == _KINFO_PROC else None


_KINFO_PROC, _E_PPID = 648, 560  # sizeof(struct kinfo_proc) and offsetof(kp_eproc.e_ppid), 64-bit macOS


@functools.cache
def _sysctl() -> Any:
    import ctypes

    fn = ctypes.CDLL(None, use_errno=True).sysctl
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


def _signal(pid: int, sig: int, group: bool) -> tuple[str, str] | None:
    """A signal refused unless it's a probe (0), to this very process, or to one of its own children
    (or a group one leads): anything else could stop an app, or the shell running the suite."""
    if sig == 0 or (pid == os.getpid() and not group):
        return None
    if pid > 0 and _parent(pid) in (os.getpid(), _GONE):  # this process's child, or no process at all
        return None
    whom = f"process group {pid}" if group else ("every process" if pid == -1 else f"process {pid}")
    return "process", f"send signal {sig} to {whom} (not a child of this process)"


def _refusal(event: str, args: tuple) -> tuple[str, str] | None:
    if event in _NET:
        address = args[1]
        if not _outside(args[0], address):
            return None
        host = f"[{address[0]}]" if ":" in str(address[0]) else address[0]
        mine = " (a server there is not this process's own)" if _loopback(str(address[0]).partition("%")[0]) else ""
        return "network", f"connect to {host}:{address[1]}{mine}"
    if event in _SIGNAL:
        return _signal(int(args[0]), int(args[1]), event == "os.killpg")
    if event in _DLSYM:
        name = args[1]
        return ("process", f"look up the C function {name}() through ctypes") if name in _STARTS else None
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
    if event == "socket.bind":
        _bound(args[0])
        return
    why = why_off()
    if why and (refused := _refusal(event, args)) and refused[0] not in _allowed:
        raise EffectRefused(*refused, why)


def _check_fork_exec() -> None:
    """_posixsubprocess.fork_exec starts a program with no audit event of its own when called directly
    (multiprocessing does): it gets the same check as a subprocess."""
    try:
        import _posixsubprocess
    except ImportError:
        return
    real = _posixsubprocess.fork_exec
    if getattr(real, "neo_checked", False):
        return

    def fork_exec(args: Any, executables: Any, *rest: Any) -> Any:
        why = why_off()
        refused = why and _program(executables[0] if executables else None, args)
        if refused and refused[0] not in _allowed:
            raise EffectRefused(*refused, why)
        return real(args, executables, *rest)

    fork_exec.neo_checked = True  # type: ignore[attr-defined]
    _posixsubprocess.fork_exec = fork_exec


def _guard_web_search() -> None:
    """ddgs (every web search NEO makes) sends its requests through primp, an HTTP client written in
    Rust: no Python socket, so the audit hook never sees them. Its one Python door is guarded instead."""
    try:
        from ddgs import http_client
    except ImportError:
        return
    real = http_client.HttpClient.request
    if getattr(real, "neo_guarded", False):
        return

    @functools.wraps(real)
    def request(self: Any, *args: Any, **kwargs: Any) -> Any:
        method = args[0] if args else kwargs.get("method", "?")
        url = args[1] if len(args) > 1 else kwargs.get("url", "")
        guard("network", f"send a web search's {method} to {urlsplit(str(url)).netloc or url} (ddgs: Rust)")
        return real(self, *args, **kwargs)

    request.neo_guarded = True  # type: ignore[attr-defined]
    http_client.HttpClient.request = request


def _no_proxies() -> None:
    """A proxy named in the environment (or macOS's own settings, which urllib reads when the
    environment names none) would carry every HTTP client off the Mac through one loopback port."""
    for k in [k for k in os.environ if k.lower().endswith("_proxy") and k.lower() != "no_proxy"]:
        del os.environ[k]
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"


_hooked = False


def _install() -> None:
    global _hooked
    if not _hooked:
        sys.addaudithook(_audit)
        _check_fork_exec()
        _guard_web_search()
        _no_proxies()
        _hooked = True


if why_off():  # decided when neo is first imported (neo/__init__.py imports this module)
    _install()
