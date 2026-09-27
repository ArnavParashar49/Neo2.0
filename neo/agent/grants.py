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
  the model. Every untrusted result in a task is counted (`read_untrusted`); only what the granted
  calls themselves read back from the granted app doesn't count;
* another task: a grant never outlives its task, and never reaches another job running at the
  same moment (the task id travels in a context variable, like confirm's approvals).

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


_task: ContextVar[str] = ContextVar("neo_task", default="")
_call: ContextVar[_Call | None] = ContextVar("neo_grant_call", default=None)
_grants: dict[str, list[Grant]] = {}
_notes: dict[str, Counter] = {}
_started: dict[str, float] = {}
_reads: Counter = Counter()  # per task: untrusted results read so far


def new_task() -> str:
    tid = f"t{secrets.token_hex(6)}"
    _started[tid] = time.time()
    return tid


def current() -> str:
    """The task this code runs in ("" = none: a direct command, a Live call)."""
    tid = _task.get()
    if tid and time.time() - _started.get(tid, 0.0) > _TTL_S:
        end(tid)
        return ""
    return tid


@contextmanager
def task(task_id: str) -> Iterator[str]:
    """Run the enclosed code as part of `task_id` ("" = outside any task)."""
    if task_id and task_id not in _started:
        _started[task_id] = time.time()
    token = _task.set(task_id)
    try:
        yield task_id
    finally:
        _task.reset(token)


def end(task_id: str) -> None:
    """The task is over: its grants and notes go with it."""
    if not task_id:
        return
    for g in _grants.pop(task_id, []):
        _audit(g.action_id, g.tool, {"app": g.app, "task": task_id}, "task grant ended")
    _notes.pop(task_id, None)
    _started.pop(task_id, None)
    _reads.pop(task_id, None)
    _prune()


def grant(task_id: str, tool: str, target: Target | None, action_id: str) -> Grant | None:
    """The user said yes for this task to the card `action_id` (tool on target.app)."""
    if not task_id or target is None or not target.grantable or time.time() - _started.get(task_id, 0.0) > _TTL_S:
        return None
    g = Grant(task_id, tool, target.app.strip(), frozenset(target.taint) | {tool}, action_id, reads=_reads[task_id])
    _grants.setdefault(task_id, []).append(g)
    _audit(action_id, tool, {"app": g.app, "task": task_id}, "granted for this task")
    return g


def covering(tool: str, target: Target | None) -> Grant | None:
    """The grant of the current task that lets this call run without a card, if any."""
    tid = current()
    if not tid or target is None or not target.grantable:
        return None
    for g in _grants.get(tid, ()):
        if g.tool != tool or g.app.casefold() != target.app.strip().casefold():
            continue
        if not set(target.taint) <= g.baseline or _reads[tid] > g.reads:
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
    return next((g for g in _grants.get(tid, ()) if action_id and g.action_id == action_id), None) if tid else None


def read_untrusted(tool: str) -> None:
    """A tool has handed the model untrusted content (a page, a mail, the screen) in the current
    task. Every such result after a yes "for this task" ends what that yes covers — except what a
    call the grant let through reads back from its own app."""
    tid = current()
    if not tid:
        return
    rec = _call.get()
    if rec is not None and rec.grant is not None and rec.grant.tool == tool and rec.grant.task == tid:
        return
    _reads[tid] += 1


def offered(task_id: str, target: Target | None) -> list[str]:
    """The answer scopes a card can offer: ["task"] when a yes for this task could cover it."""
    return ["task"] if task_id and target is not None and target.grantable else []


def active(task_id: str = "") -> list[Grant]:
    return list(_grants.get(task_id or current(), ()))


# ---- per-task notes ------------------------------------------------------------------------
def note(key: str) -> int:
    """Count `key` once more in the current task; returns the new count (0 outside a task)."""
    tid = current()
    if not tid:
        return 0
    c = _notes.setdefault(tid, Counter())
    c[key] += 1
    return c[key]


def count(key: str) -> int:
    tid = current()
    return _notes.get(tid, Counter())[key] if tid else 0


def first(key: str) -> bool:
    """True the first time `key` comes up in the current task — and always outside a task."""
    tid = current()
    return not tid or note(key) == 1


def _prune() -> None:
    cutoff = time.time() - _TTL_S
    for tid in [t for t, at in _started.items() if at < cutoff]:
        _grants.pop(tid, None)
        _notes.pop(tid, None)
        _started.pop(tid, None)
        _reads.pop(tid, None)


def _audit(action_id: str, tool: str, params: dict, status: str) -> None:
    from neo.agent.confirm import audit

    audit(action_id, tool, params, status)


def reset() -> None:
    """Forget every grant (tests; an erase)."""
    _grants.clear()
    _notes.clear()
    _started.clear()
    _reads.clear()
