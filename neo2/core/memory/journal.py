"""The journal: every line the user said or typed and every line NEO actually said, written to
the memory database as it happens — so nothing is lost when a voice session closes, when NEO is
quit, or when it is killed.

Design:

* :func:`log_turn` never blocks its caller (the voice path, the event loop): it puts the line on
  a queue and returns. ONE writer thread per database drains the queue and commits in batches —
  a line is on disk within about a quarter of a second, a crash loses no more than that.
* The writer is the only thing that writes in the background, so what else must not write on a
  request's path goes through it too: the rule-based fact extraction on each of the user's lines
  (neo2.core.memory.extract — it runs here, right after the line has its id) and the recall counters.
* Privacy: a line is redacted before it is written (neo2.core.memory.redact, in the store). A line
  queued before "Erase NEO's Memory" is dropped, not written afterwards (the erase generation,
  neo2.core.privacy). What NEO read from a page, a mail or a file is never journaled as the user's
  words; a line of NEO's own that quotes such content is marked ``tainted``; tool outputs are
  never journaled — only a note of which tools ran (``source="tool"``).
* ``memory_journal = False`` (or memory off, as in an embedded agent by default) turns it off:
  :func:`log_turn` then does nothing.

``flush()`` waits until everything queued is on disk (tests, and shutdown).
"""

from __future__ import annotations

import contextvars
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Literal

from neo2.core import logger
from neo2.core.config import settings

log = logger(__name__)

Role = Literal["user", "assistant"]
Source = Literal["voice", "typed", "tool"]

_BATCH_S = 0.1  # how long the writer gathers lines before it commits (a line waits ≤ ~0.25 s)
_STOP = object()


@dataclass
class _Line:
    ts: float
    role: str
    text: str
    route: str
    source: str
    tainted: bool
    generation: int
    queued: float  # time.monotonic(): a line queued before an erase is never written


@dataclass
class _Flush:
    done: threading.Event


class Journal:
    """The queue and the one writer thread of one memory database."""

    def __init__(self, store: Any) -> None:
        self._store = weakref.ref(store)
        self._q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._start = threading.Lock()
        self._closed = False
        self.last_turn_at = 0.0  # when the latest line was logged (the idle work waits on it)
        self.written = 0
        # If the store goes away without close(), the thread ends instead of waiting forever.
        self._finalizer = weakref.finalize(store, self._q.put, _STOP)

    # ---- callers ---------------------------------------------------------------------------
    def log(
        self,
        role: str,
        text: str,
        *,
        route: str = "",
        source: str = "typed",
        tainted: bool = False,
        ts: float | None = None,
        generation: int | None = None,
    ) -> None:
        text = " ".join((text or "").split())
        if not text or self._closed:
            return
        from neo2.core import privacy

        gen = privacy.generation()
        if generation is not None and generation != gen:
            return  # the request began before the user erased NEO's memory: it stays erased
        now = time.time()
        self.last_turn_at = now
        self._q.put(_Line(ts or now, role, text, route, source, bool(tainted), gen, time.monotonic()))
        self._ensure_thread()

    def kick(self) -> None:
        """Wake the writer for its housekeeping (the recall counters)."""
        if not self._closed:
            self._q.put(None)
            self._ensure_thread()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far is written. False if it took longer than `timeout`."""
        if self._thread is None or not self._thread.is_alive():
            return True
        f = _Flush(threading.Event())
        self._q.put(f)
        return f.done.wait(timeout)

    def close(self, timeout: float = 2.0) -> None:
        """Write what is queued and stop the writer."""
        if self._closed:
            return
        self._closed = True
        t = self._thread
        self._q.put(_STOP)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    # ---- the writer ------------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start:
            if self._thread is not None and self._thread.is_alive():
                return
            # A copy of the caller's context: the thread sees the same agent (neo2.core.runtime) —
            # its settings and its erase generation — as the request that started it.
            ctx = contextvars.copy_context()
            self._thread = threading.Thread(target=ctx.run, args=(self._run,), daemon=True, name="neo-journal")
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            batch, flushes, stop = [], [], False
            deadline = time.monotonic() + _BATCH_S
            while True:
                if item is _STOP:
                    stop = True
                elif isinstance(item, _Flush):
                    flushes.append(item)
                elif isinstance(item, _Line):
                    batch.append(item)
                if stop or flushes:
                    # write now — but first take whatever else is already queued
                    try:
                        item = self._q.get_nowait()
                        continue
                    except queue.Empty:
                        break
                wait = deadline - time.monotonic()
                if wait <= 0:
                    break
                try:
                    item = self._q.get(timeout=wait)
                except queue.Empty:
                    break
            try:
                self._write(batch)
            except Exception as e:  # noqa: BLE001 — the journal never takes NEO down
                log.warning(f"[memory] journal write failed: {str(e)[:100]}")
            for f in flushes:
                f.done.set()
            if stop:
                return

    def _write(self, batch: list[_Line]) -> None:
        st = self._store()
        if st is None or st._closed:
            return
        from neo2.core import privacy

        if batch:
            # Decided with the database locked (Store.add_turns): a line queued before an erase is
            # dropped whether the erase came before the write or while it waited for the lock.
            def keep(i: int) -> bool:
                return batch[i].generation == privacy.generation() and batch[i].queued > st._erased_at

            rows = [(b.ts, b.role, b.text, b.route, b.source, b.tainted) for b in batch]
            turns = st.add_turns(rows, keep=keep)
            self.written += len(turns)
            if turns:
                self._extract(st, turns, min(b.queued for b in batch))
        st.flush_touches()

    @staticmethod
    def _extract(st: Any, turns: list[Any], since: float) -> None:
        """Facts the user stated in so many words ("my sister's name is Priya"), kept the moment
        the line is written — by rules, no model (neo2.core.memory.extract). `since`: when the
        oldest of these lines was queued (nothing of them is kept after a later erase)."""
        try:
            if settings().memory_extract == "off":
                return
            from neo2.core.memory import extract

            for t in turns:
                if t.role != "user" or t.source == "tool":
                    continue
                for fact in extract.rules(t.text):
                    extract.keep(st, fact, source=f"t{t.id}", since=since)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[memory] fact rules failed: {str(e)[:100]}")


_make = threading.Lock()


def journal_of(st: Any) -> Journal:
    """The journal of this store (made on first use)."""
    j = st._journal
    if j is None:
        with _make:
            j = st._journal
            if j is None:
                j = st._journal = Journal(st)
    return j


def enabled() -> bool:
    """Is the journal on for whoever is asking (NEO.app, or the bound agent)?"""
    try:
        s = settings()
        return bool(s.memory_enabled and s.memory_journal)
    except Exception:  # noqa: BLE001 — no settings here (an unbound thread in a host's process)
        return False


def journal() -> Journal | None:
    """The journal of the current memory store, or None when it is off."""
    if not enabled():
        return None
    try:
        from neo2.core.memory.store import store

        return journal_of(store())
    except Exception:  # noqa: BLE001 — memory is never a reason for a request to fail
        return None


def log_turn(
    role: Role,
    text: str,
    *,
    route: str = "",
    source: Source = "typed",
    tainted: bool = False,
    ts: float | None = None,
    generation: int | None = None,
) -> None:
    """Put one line of the conversation in the journal. Returns at once and never raises.

    `role`: "user" for the user's own words (typed, or their transcribed speech — never text a
    model wrote or NEO read somewhere), "assistant" for a line NEO actually said. `source`:
    "voice" or "typed"; "tool" for a note of which tools ran (never their output). `tainted`: a
    line of NEO's that quotes a page, a mail or a file. `generation`: the erase generation the
    request began in (a request older than the last erase writes nothing)."""
    try:
        j = journal()
        if j is not None:
            j.log(role, text, route=route, source=source, tainted=tainted, ts=ts, generation=generation)
    except Exception as e:  # noqa: BLE001
        log.warning(f"[memory] journal skipped a line: {str(e)[:80]}")


def log_tools(names: list[str] | tuple[str, ...], *, route: str = "", generation: int | None = None) -> None:
    """Note which tools NEO ran for the last request — their names only, so "what did we do
    yesterday" has something to go on. Never an argument, never an output."""
    names = [n for n in dict.fromkeys(names) if n]
    if names:
        log_turn("assistant", "(ran: " + ", ".join(names[:12]) + ")", route=route, source="tool", generation=generation)


def flush(timeout: float = 5.0) -> bool:
    """Wait until the current store's queued lines are written (True when they are)."""
    j = journal()
    return True if j is None else j.flush(timeout)


def close(timeout: float = 2.0) -> None:
    """Write what is queued and stop the current store's writer (shutdown)."""
    try:
        from neo2.core.memory import store as store_mod

        st = store_mod._store
        if st is not None and st._journal is not None:
            st._journal.close(timeout)
    except Exception:  # noqa: BLE001
        pass
