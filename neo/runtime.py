"""Per-agent state, for NEO embedded in another app as a library.

NEO.app is one agent per process and keeps its state in module globals: the settings, the tool
registry, the confirmation gate, the event bus, the memory store, the brains. A host app can run
several agents in one process, so each agent gets a :class:`Runtime` and each of its
conversations a :class:`Scope`, carried in a ContextVar. The accessors every module already calls
(``settings()``, ``registry()``, ``bus()``, ``store()``, the gate…) return the bound instance, and
fall back to the module globals when nothing is bound — NEO.app's path, unchanged.

Once an agent exists in the process (:func:`set_library_mode`), an accessor called with nothing
bound fails closed with :class:`NoRuntimeError` instead of falling back to ~/.neo and the app's
gate: a raw thread or a third-party callback that lost the context must never read the user's
own NEO data or stage an action where another agent could approve it.

asyncio carries the ContextVar into ``create_task``, ``gather``, ``to_thread`` and ``call_later``,
which is how the code an agent reaches spawns work. Plain ``threading.Thread``s don't: code that
runs there (the app's warm-ups and learning loops) fails closed in library mode.

Only TYPE_CHECKING imports at module level: config, registry and the rest import this module.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from neo.agent.confirm import Gate
    from neo.agent.early import EarlyActor
    from neo.agent.registry import Registry
    from neo.agent.session import Session
    from neo.agent.toolselect import ToolSelector
    from neo.config import Settings
    from neo.events import EventBus
    from neo.memory.store import Store
    from neo.providers import BrainPool

T = TypeVar("T")


class NoRuntimeError(RuntimeError):
    """A NEO accessor ran with no agent bound while NEO is embedded as a library."""


class UnsupportedPlatform(RuntimeError):
    """Tools that need macOS were asked for on another platform."""


@dataclass(eq=False)
class Runtime:
    """One embedded agent: everything NEO.app keeps in module globals, for this agent only."""

    name: str
    settings: Settings
    registry: Registry
    brains: BrainPool | None = None  # created on first use (neo.providers)
    early: EarlyActor | None = None  # created on first use (neo.agent.early)
    selector: ToolSelector | None = None  # created on first use, over `registry`
    store: Store | None = None  # created on first use, only with memory on
    browser: Any = None  # NEO's own browser for this agent (neo.tools.browser), on first use
    audit_sink: Callable[[dict[str, Any]], None] | None = None  # audit records, as dicts
    audit_file: bool = True  # also append them to settings.audit_log
    secret_paths: tuple[Path, ...] = ()  # never read or written by NEO's tools
    generation: int = 0  # bumped when this agent's memory is erased (neo.privacy)
    library: bool = True
    temporary: bool = False  # its data dir is a temp dir its owner removes when it closes
    closing: bool = False  # its owner is closing it: nothing of it is (re)built any more
    closed: bool = False  # no longer live: its folder is free, what it kept stays secret
    scopes: dict[str, Scope] = field(default_factory=dict, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.library:
            from neo.config import LibrarySettings

            if not isinstance(self.settings, LibrarySettings):
                raise TypeError(
                    "an embedded agent takes LibrarySettings (only what its host passes): NEO.app's "
                    "Settings read the environment, .env files and ~/.neo"
                )
            self.settings.check()
        with _live_lock:
            if self.library:
                _refuse_shared_data_dir(self)
            _live.add(self)
        if self.library:
            set_library_mode()  # from now on, code that lost its agent fails closed
            # Dropped without close(): what it kept stays secret all the same.
            self._retire = weakref.finalize(self, _retire, tuple(_kept(self)))

    def scope(self, conversation: str = "default") -> Scope:
        """The conversation's scope (its own gate, events and session), created on first use."""
        with self.lock:
            sc = self.scopes.get(conversation)
            if sc is None:
                sc = self.scopes[conversation] = Scope(self, conversation)
            return sc

    def get_or_create(self, attr: str, make: Callable[[], T]) -> T:
        """`getattr(self, attr)`, built by `make()` the first time (once, even across threads).
        Never after the agent closed: late work can't reopen its database or its browser."""
        v = getattr(self, attr)
        if v is None:
            with self.lock:
                v = getattr(self, attr)
                if v is None:
                    if self.closing or self.closed:
                        raise RuntimeError(f"the agent {self.name!r} is closed")
                    v = make()
                    setattr(self, attr, v)
        return v

    def close(self) -> None:
        """No longer live: its data dir is free for another agent. What it kept (its data dir,
        unless a temp dir its owner removes; its memory file; its secret paths) stays secret to
        every other agent for the rest of the process."""
        self.release()
        with _live_lock:
            _live.discard(self)

    def release(self) -> None:
        """close() without taking any lock — for a finalizer, which may run inside the GC while
        this very thread holds _live_lock: closed (a closed runtime counts as live nowhere) and
        what it kept retired, once."""
        if self.library and self._retire.detach() is not None:
            _retire(tuple(_kept(self)))
        self.closed = True


@dataclass(eq=False)
class Scope:
    """One conversation of one agent: its own confirmation gate, event bus and session."""

    runtime: Runtime
    conversation: str = "default"
    gate: Gate | None = None  # created on first use (neo.agent.confirm)
    bus: EventBus | None = None  # created on first use (neo.events)
    session: Session | None = None

    def get_or_create(self, attr: str, make: Callable[[], T]) -> T:
        v = getattr(self, attr)
        if v is None:
            with self.runtime.lock:
                v = getattr(self, attr)
                if v is None:
                    v = make()
                    setattr(self, attr, v)
        return v


_scope: ContextVar[Scope | None] = ContextVar("neo_scope", default=None)
_library = False  # set once an embedded agent exists; sticky for the process
_live: weakref.WeakSet[Runtime] = weakref.WeakSet()
_live_lock = threading.Lock()  # the policy reads it from worker threads
# What closed (or dropped) agents kept, secret for the rest of the process. A dict as an ordered
# set, lock-free: a finalizer adds to it whenever the GC runs, even inside _live_lock.
_retired: dict[Path, None] = {}


def current_scope() -> Scope | None:
    return _scope.get()


def current_runtime() -> Runtime | None:
    sc = _scope.get()
    return sc.runtime if sc is not None else None


@contextmanager
def bind(scope: Scope) -> Iterator[Scope]:
    """Everything run inside (and every task it spawns) sees this agent and conversation."""
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def set_library_mode(on: bool = True) -> None:
    """NEO is embedded in another app: from now on an accessor with nothing bound fails closed."""
    global _library
    _library = bool(on)


def library_mode() -> bool:
    return _library


def check_unbound(what: str) -> None:
    """Called by an accessor about to fall back to the app's globals: in library mode that is a
    context that escaped its agent, and it must fail rather than reach ~/.neo or the app's gate."""
    if _library and _scope.get() is None:
        raise NoRuntimeError(
            f"{what} was used with no NEO agent bound. NEO is embedded as a library in this process, "
            "so it never falls back to the app's data (~/.neo) or confirmation gate: call NEO "
            "through its agent (code in a raw thread doesn't carry the agent with it)."
        )


def library_bound() -> bool:
    """A library agent is bound right now (its requests, not NEO.app's)."""
    rt = current_runtime()
    return rt is not None and rt.library


def process_wide_ok() -> bool:
    """May NEO change something for the whole process or start work of its own in the background
    here — a process-wide Accessibility timeout, an AX observer thread, an idle scan of the OS's
    UI strings? Only as NEO.app: never inside another app's process (library mode, sticky once an
    embedded agent exists) or while an embedded agent's request runs. There, such things are left
    out (or asked for per call), and the host's process stays as the host set it."""
    return not (_library or library_bound())


def _refuse_shared_data_dir(rt: Runtime) -> None:
    """One live agent per data dir: two would share a memory, an audit log and a browser profile
    (its logins), and one nested in another's would be the other's secret. Under _live_lock."""
    if clash := _shared_with(rt):
        raise ValueError(clash)


def _shared_with(rt: Runtime) -> str:
    """Why `rt` can't live next to a live agent ("" = it can). Only a string leaves: the error's
    traceback must not keep the other agent alive (a notebook keeps the last one)."""
    from neo.config import overlaps

    mine = Path(rt.settings.data_dir)
    live = [o for o in _live if o.library and not o.closed]
    for other in live:
        if overlaps(mine, other.settings.data_dir):
            return (
                f"data_dir {mine} is another agent's ({other.name!r} uses {other.settings.data_dir}): "
                "give each agent its own folder, or close the other first"
            )
    # A memory database kept outside the data dir (memory_path) is one agent's too: never another
    # agent's file, in another's folder, or holding another's.
    own_db = _memory_db(rt.settings)
    for other in live:
        their_db = _memory_db(other.settings)
        pairs = [(mine, their_db)] if their_db is not None else []
        if own_db is not None:
            pairs += [(own_db, Path(other.settings.data_dir))] + ([(own_db, their_db)] if their_db is not None else [])
        if any(overlaps(a, b) for a, b in pairs):
            return f"that memory database is another agent's ({other.name!r}): give each agent its own"
    return ""


def _memory_db(s: Settings) -> Path | None:
    p = getattr(s, "memory_path", None)
    return Path(p) if p is not None else None


def live_runtimes() -> list[Runtime]:
    with _live_lock:
        return [r for r in _live if not r.closed]


def secret_roots() -> list[Path]:
    """Folders NEO's tools never read or write beyond the fixed list in the policy: every live
    agent's data dir and extra secret paths (and what closed agents kept: Runtime.close), and the
    app's own data dir once its settings have
    been built (read from the module global — this never builds settings). Embedded, also NEO.app's
    data wherever the environment's NEO_DATA_DIR / NEO_MEMORY_PATH moved it (its settings are
    never built in a host's process)."""
    roots: list[Path] = []
    for rt in live_runtimes():
        s = rt.settings
        roots.append(Path(s.data_dir))
        roots.extend(Path(p) for p in getattr(s, "extra_secret_paths", None) or ())
        roots.extend(_memory_files(s))
        roots.extend(Path(p) for p in rt.secret_paths)
    from neo import config

    app = config._settings
    if app is not None:
        roots.append(Path(app.data_dir))
        roots.extend(Path(p) for p in getattr(app, "extra_secret_paths", None) or ())
        roots.extend(_memory_files(app))
    if _library:
        for p in config.moved_app_data():  # a folder, or a memory file with its side files
            roots.extend([p, *(p.with_name(p.name + x) for x in _SIDE_FILES)])
    roots.extend(list(_retired))
    out: list[Path] = []
    for p in roots:
        p = Path(str(p)).expanduser()
        if p not in out:
            out.append(p)
    return out


_SIDE_FILES = ("-wal", "-shm", "-journal")  # what SQLite keeps next to a database


def _kept(rt: Runtime) -> list[Path]:
    """What a library agent keeps that no other agent may read: its data dir (unless a temp dir
    that goes with it), its memory file, its secret paths."""
    s = rt.settings
    out = [] if rt.temporary else [Path(s.data_dir)]
    out.extend(Path(p) for p in getattr(s, "extra_secret_paths", None) or ())
    out.extend(_memory_files(s))
    out.extend(Path(p) for p in rt.secret_paths)
    return out


def _retire(paths: tuple[Path, ...]) -> None:
    for p in paths:
        _retired[p] = None


def _memory_files(s: Settings) -> list[Path]:
    """A memory database kept outside the data dir (`memory_path`), with SQLite's side files."""
    p = _memory_db(s)
    return [] if p is None else [p, *(p.with_name(p.name + x) for x in _SIDE_FILES)]
