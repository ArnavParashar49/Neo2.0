"""Task-scoped approvals: "yes for this task".

A *task* is one request the agent works on, from the user's words until it finishes — across
every card the user answers on the way (the run after a "yes" is the same task). When the user
approves a card "for this task", later calls in that task with the same tool on the same app run
without a card, except:

* a call the tool can't show is safe to repeat: always asked. The tool decides by an allowlist,
  not a list of bad words — for AppleScript (policy.script_facts) only reading, clicking and
  filling in one named, ordinary app's window can be covered; a shell command, a delete, a send,
  a purchase, raw Apple event codes, System Events' own actions, several apps, an app picked at
  run time or named by its path, and apps that run commands (terminals, script runners) never;
* another app: asked;
* the request has read something new and untrusted since the grant (a page, a mail, a file, a
  screenshot — even from a tool it had used before): asked, since that text could be steering
  the model. Every untrusted result in a task is counted (`read_untrusted`) except what the
  granted calls themselves read back from the granted app — and that can be someone else's words
  too (a mail client's messages, a page in a browser). So a grant never rests on nothing new
  having been read: what it covers is checked call by call. It presses only elements named by
  their label, never one whose label sends, buys, deletes or grants, nor a bare "OK" (for
  AppleScript, policy.script_facts); pressing an element picked by its place (`button 3`) is
  asked about every time;
* another task: a grant never outlives its task, and never reaches another job running at the
  same moment (the task id travels in a context variable, like confirm's approvals);
* another conversation: the tasks, grants, notes and untrusted-read counts live in the
  confirmation gate's own :class:`Book` (neo.agent.confirm.Gate), so a task id carried into
  another gate's context is no task there, and two sessions never share a grant or a count.

Grants are NEO.app's: only its own gate gives them. An agent embedded in another app
(neo.runtime) never does — its cards offer no scope (``scopes: []``) and every yes stays one-off,
since its host approves each staged action through its own route; its notes and counts still
work, per conversation.

Every grant, every call it covers and its end are written to the audit log.

The module also keeps a few per-task notes (`first`, `note`, `count`): Accessibility is asked
about once per task, and a script that keeps failing on an app stops being retried.
"""

from __future__ import annotations

import secrets
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_TTL_S = 30 * 60.0  # no task runs this long: a grant older than this is gone whatever happened


@dataclass(frozen=True)
class Target:
    """What a gated call acts on, as its tool reads it.

    `app`: the one app ("" = none or unknown: never covered). `danger`: why no grant may cover
    it ("" = one may). `taint`: the untrusted sources the request had read when the call was
    made (ToolContext.taint)."""

    app: str = ""
    danger: str = ""
    taint: tuple[str, ...] = ()

    @property
    def grantable(self) -> bool:
        return bool(self.app.strip()) and not self.danger


@dataclass
class Grant:
    task: str
    tool: str
    app: str
    baseline: frozenset[str]  # what the request had read when the user said yes (plus the tool itself)
    action_id: str  # the card the user answered
    at: float = field(default_factory=time.time)
    reads: int = 0  # untrusted results the task had read when the user said yes


@dataclass
class _Call:
    """One tool call in flight (registry.invoke): the grant it ran under, if any."""

    grant: Grant | None = None


class Book:
    """One confirmation gate's tasks: their grants, notes and untrusted-read counts. `grants_on`:
    this gate may give a yes "for this task" (NEO.app's own gate only)."""

    def __init__(self, *, grants_on: bool = False) -> None:
        self.grants_on = grants_on
        self.grants: dict[str, list[Grant]] = {}
        self.notes: dict[str, Counter] = {}
        self.started: dict[str, float] = {}
        self.reads: Counter = Counter()  # per task: untrusted results read so far

    def clear(self) -> None:
        self.grants.clear()
        self.notes.clear()
        self.started.clear()
        self.reads.clear()


_task: ContextVar[str] = ContextVar("neo_task", default="")
_call: ContextVar[_Call | None] = ContextVar("neo_grant_call", default=None)


def _book() -> Book:
    """The current conversation's book: the gate's (neo.agent.confirm), NEO.app's when nothing is
    bound. In library mode with nothing bound this fails closed, like the gate."""
    from neo.agent.confirm import current_gate

    return current_gate().grants


def new_task() -> str:
    tid = f"t{secrets.token_hex(6)}"
    _book().started[tid] = time.time()
    return tid


def current() -> str:
    """The task this code runs in ("" = none: a direct command, a Live call — or a task of another
    conversation's, which is no task in this one)."""
    tid = _task.get()
    if tid:
        book = _book()
        if time.time() - book.started.get(tid, 0.0) > _TTL_S:
            _end(book, tid)
            return ""
    return tid


@contextmanager
def task(task_id: str) -> Iterator[str]:
    """Run the enclosed code as part of `task_id` ("" = outside any task), in this conversation."""
    if task_id:
        book = _book()
        if task_id not in book.started:
            book.started[task_id] = time.time()
    token = _task.set(task_id)
    try:
        yield task_id
    finally:
        _task.reset(token)


def end(task_id: str) -> None:
    """The task is over: its grants and notes go with it."""
    if task_id:
        _end(_book(), task_id)


def _end(book: Book, task_id: str) -> None:
    for g in book.grants.pop(task_id, []):
        _audit(g.action_id, g.tool, {"app": g.app, "task": task_id}, "task grant ended")
    book.notes.pop(task_id, None)
    book.started.pop(task_id, None)
    book.reads.pop(task_id, None)
    _prune(book)


def grant(task_id: str, tool: str, target: Target | None, action_id: str) -> Grant | None:
    """The user said yes for this task to the card `action_id` (tool on target.app). Nothing in a
    conversation whose gate gives no grants (an embedded agent's: its yes stays one-off)."""
    book = _book()
    if not book.grants_on or not task_id or target is None or not target.grantable:
        return None
    if time.time() - book.started.get(task_id, 0.0) > _TTL_S:
        return None
    g = Grant(task_id, tool, target.app.strip(), frozenset(target.taint) | {tool}, action_id, reads=book.reads[task_id])
    book.grants.setdefault(task_id, []).append(g)
    _audit(action_id, tool, {"app": g.app, "task": task_id}, "granted for this task")
    return g


def covering(tool: str, target: Target | None) -> Grant | None:
    """The grant of the current task that lets this call run without a card, if any."""
    tid = current()
    if not tid or target is None or not target.grantable:
        return None
    book = _book()
    if not book.grants_on:
        return None
    for g in book.grants.get(tid, ()):
        if g.tool != tool or g.app.casefold() != target.app.strip().casefold():
            continue
        if not set(target.taint) <= g.baseline or book.reads[tid] > g.reads:
            continue  # read something new and untrusted since that yes
        if time.time() - g.at > _TTL_S:
            continue
        return g
    return None


@contextmanager
def call() -> Iterator[_Call]:
    """One tool call (registry.invoke): whether a grant let it through is noted on it, so what
    it reads back doesn't count as new reading against that grant."""
    rec = _Call()
    token = _call.set(rec)
    try:
        yield rec
    finally:
        _call.reset(token)


def ran_under(g: Grant | None) -> None:
    """The call in flight runs under `g`: covered by it, or the very call whose card the user
    answered "for this task"."""
    rec = _call.get()
    if rec is not None and g is not None:
        rec.grant = g


def for_card(action_id: str) -> Grant | None:
    """The grant the user gave when answering the card `action_id`, in the current task."""
    tid = current()
    if not tid or not action_id:
        return None
    return next((g for g in _book().grants.get(tid, ()) if g.action_id == action_id), None)


def read_untrusted(tool: str) -> None:
    """A tool has handed the model untrusted content (a page, a mail, the screen) in the current
    task. Every such result after a yes "for this task" ends what that yes covers — except what a
    call the grant let through reads back from its own app, which may still be a third party's
    text: the grant's own limits (by-label presses only, see the module doc) are what hold then."""
    tid = current()
    if not tid:
        return
    book = _book()
    rec = _call.get()
    g = rec.grant if rec is not None else None
    if g is not None and g.tool == tool and g.task == tid and any(x is g for x in book.grants.get(tid, ())):
        return
    book.reads[tid] += 1


def offered(task_id: str, target: Target | None, *, book: Book | None = None) -> list[str]:
    """The answer scopes a card can offer: ["task"] when a yes for this task could cover it (never
    in a conversation whose gate gives no grants). `book`: that gate's (default: the current)."""
    if not task_id or target is None or not target.grantable:
        return []
    return ["task"] if (book or _book()).grants_on else []


def active(task_id: str = "") -> list[Grant]:
    return list(_book().grants.get(task_id or current(), ()))


def reads(task_id: str = "") -> int:
    """Untrusted results the task has read so far, in this conversation."""
    return _book().reads.get(task_id or current(), 0)


# ---- per-task notes ------------------------------------------------------------------------
def note(key: str) -> int:
    """Count `key` once more in the current task; returns the new count (0 outside a task)."""
    tid = current()
    if not tid:
        return 0
    c = _book().notes.setdefault(tid, Counter())
    c[key] += 1
    return c[key]


def count(key: str) -> int:
    tid = current()
    return _book().notes.get(tid, Counter())[key] if tid else 0


def first(key: str) -> bool:
    """True the first time `key` comes up in the current task — and always outside a task."""
    tid = current()
    return not tid or note(key) == 1


def _prune(book: Book) -> None:
    cutoff = time.time() - _TTL_S
    for tid in [t for t, at in book.started.items() if at < cutoff]:
        book.grants.pop(tid, None)
        book.notes.pop(tid, None)
        book.started.pop(tid, None)
        book.reads.pop(tid, None)


def _audit(action_id: str, tool: str, params: dict, status: str) -> None:
    from neo.agent.confirm import audit

    audit(action_id, tool, params, status)


def reset() -> None:
    """Forget every grant NEO.app's gate gave, with its tasks and notes (tests; an erase). An
    embedded agent's conversations keep theirs in their own gates, which go with them."""
    from neo.agent.confirm import app_gate

    app_gate().grants.clear()
