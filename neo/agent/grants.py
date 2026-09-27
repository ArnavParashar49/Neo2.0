"""Task-scoped approvals: "yes for this task".

A *task* is one request the agent works on, from the user's words until it finishes — across
every card the user answers on the way (the run after a "yes" is the same task). When the user
approves a card "for this task", later calls in that task with the same tool on the same app run
without a card, except:

* a call that is dangerous — the tool says why (a shell command, a delete, a send, a purchase,
  raw Apple event codes, several apps or an app picked at run time): always asked;
* another app: asked;
* the request has read something new and untrusted since the grant (a page, a mail, a file, a
  screenshot): asked, since that text could be steering the model;
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


_task: ContextVar[str] = ContextVar("neo_task", default="")
_grants: dict[str, list[Grant]] = {}
_notes: dict[str, Counter] = {}
_started: dict[str, float] = {}


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
    _prune()


def grant(task_id: str, tool: str, target: Target | None, action_id: str) -> Grant | None:
    """The user said yes for this task to the card `action_id` (tool on target.app)."""
    if not task_id or target is None or not target.grantable or time.time() - _started.get(task_id, 0.0) > _TTL_S:
        return None
    g = Grant(task_id, tool, target.app.strip(), frozenset(target.taint) | {tool}, action_id)
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
        if not set(target.taint) <= g.baseline:
            continue  # read something new and untrusted since that yes
        if time.time() - g.at > _TTL_S:
            continue
        return g
    return None


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


def _audit(action_id: str, tool: str, params: dict, status: str) -> None:
    from neo.agent.confirm import audit

    audit(action_id, tool, params, status)


def reset() -> None:
    """Forget every grant (tests; an erase)."""
    _grants.clear()
    _notes.clear()
    _started.clear()
