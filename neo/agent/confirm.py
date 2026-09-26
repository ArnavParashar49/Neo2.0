"""Confirm-before-acting gate with an append-only audit log.

A consequential tool calls :func:`gated` with its exact parameters. The first time, the operation
is staged and the tool returns ``NEEDS_CONFIRM``; the run halts and the user is asked, with the
full command / script / path on the overlay's card. Only the user can approve:

* the session re-invokes the staged tool inside :func:`approved` after the user's own "yes" —
  words they typed or said, or the overlay's button for that action id;
* a ``confirm`` argument from a model is ignored, so no model (and no web page or email steering
  one) can approve anything;
* the approval covers exactly the staged parameters. The action id is a hash of all of them, so a
  call that differs in any argument — a longer command, another mail body — asks again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from neo.config import settings

NEEDS_CONFIRM = "NEEDS_CONFIRM"
NEEDS_USER = "NEEDS_USER"
_CONTROL_ARGS = ("confirm", "cancel")  # never part of an operation: models used to pass these


@dataclass
class Pending:
    action_id: str
    tool: str
    params: dict[str, Any]
    summary: str
    detail: str = ""  # exactly what will run (command, script, path + content…), for the card
    staged_at: float = field(default_factory=time.time)


@dataclass
class _Approval:
    action_id: str
    released: dict[str, Any] | None = None


_pending: Pending | None = None
_approved: ContextVar[_Approval | None] = ContextVar("neo_approved_action", default=None)
_SECRET_KEYS = ("key", "token", "password", "secret")
_TTL_S = 120.0  # a question nobody answered: forget it rather than hijack the next request


def clean(params: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if k not in _CONTROL_ARGS}


def action_id(tool: str, params: dict[str, Any]) -> str:
    blob = json.dumps(clean(params), sort_keys=True, default=str, ensure_ascii=False)
    return f"{tool}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in clean(params).items():
        if any(s in k.lower() for s in _SECRET_KEYS):
            v = "***"
        s = str(v)
        out[k] = s if len(s) <= 120 else s[:117] + "..."
    return out


def audit(action_id: str, tool: str, params: dict[str, Any], status: str) -> None:
    """Best-effort append; never raises."""
    try:
        line = json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": status,
                "action": action_id,
                "tool": tool,
                "params": _redact(params),
            }
        )
        with open(settings().audit_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y", "confirm", "ok")


def _announce(kind: str, **data: Any) -> None:
    """Tell the overlay (fire-and-forget; stage() is sync and may run outside the loop)."""
    try:
        from neo.events import bus

        loop = asyncio.get_running_loop()
        loop.create_task(bus().publish(kind, **data))
    except RuntimeError:
        pass  # no running loop (CLI tools, tests): nothing to show


def card(p: Pending | None) -> dict[str, Any] | None:
    """What the overlay shows for a pending question."""
    if not p:
        return None
    return {"action_id": p.action_id, "tool": p.tool, "summary": p.summary, "detail": p.detail}


def stage(tool: str, params: dict[str, Any], summary: str, detail: str = "") -> str:
    """Stage an operation and return the halt sentinel + the question for the user."""
    global _pending
    params = clean(params)
    aid = action_id(tool, params)
    if _pending and _pending.action_id != aid:
        audit(_pending.action_id, _pending.tool, _pending.params, "superseded")
    _pending = Pending(action_id=aid, tool=tool, params=dict(params), summary=summary, detail=detail)
    audit(aid, tool, params, "staged")
    _announce("confirm", **card(_pending))  # type: ignore[arg-type]
    try:  # close the card when nobody answers, even if nothing else asks in the meantime
        asyncio.get_running_loop().call_later(_TTL_S + 1, peek)
    except RuntimeError:
        pass
    return f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?"


def peek() -> Pending | None:
    if _pending and time.time() - _pending.staged_at > _TTL_S:
        cancel("expired")
    return _pending


@contextmanager
def approved(aid: str) -> Iterator[None]:
    """The user said yes to exactly this action: the call made inside may go ahead.

    A context variable, so it covers only the task that re-invokes the tool — another job running
    at the same moment can't ride on the approval. Every gate that call passes through (the
    registry's and the handler's) sees the same approval; when it's over the question is closed
    whether or not a gate consumed it."""
    approval = _Approval(aid)
    token = _approved.set(approval)
    try:
        yield
    finally:
        _approved.reset(token)
        if approval.released is None and _pending and _pending.action_id == aid:
            consume_now(aid)


def consume_now(aid: str) -> dict[str, Any] | None:
    global _pending
    if not _pending or _pending.action_id != aid:
        return None
    p, _pending = _pending, None
    audit(aid, p.tool, p.params, "confirmed")
    _announce("confirm_done", action_id=aid, outcome="confirmed")
    return p.params


def consume(aid: str) -> dict[str, Any] | None:
    """Release the staged params — only inside :func:`approved` for this very action."""
    approval = _approved.get()
    if not aid or approval is None or approval.action_id != aid:
        return None
    if approval.released is not None:
        return approval.released  # a second gate on the same approved call
    released = consume_now(aid)
    approval.released = released
    return released


def cancel(reason: str = "user cancelled") -> str:
    global _pending
    if _pending:
        audit(_pending.action_id, _pending.tool, _pending.params, f"cancelled: {reason}")
        _announce("confirm_done", action_id=_pending.action_id, outcome=reason)
    _pending = None
    return "CANCELLED"


def gated(tool: str, params: dict[str, Any], summary: str, detail: str = "") -> dict[str, Any] | str:
    """One-liner for tool handlers.

    Returns the released params when the user approved this exact operation (or when
    confirmation is turned off), otherwise the ``NEEDS_CONFIRM`` string for the model.
    """
    params = clean(params)
    if not settings().confirm_destructive:
        return params
    released = consume(action_id(tool, params))
    if released is not None:
        return released
    return stage(tool, params, summary, detail)


_own_question = ""


def ask_user(msg: str) -> str:
    """A NEEDS_USER question from NEO's own code (not from a page or a file it read)."""
    global _own_question
    _own_question = f"{NEEDS_USER}: {msg}"
    return _own_question


def genuine_halt(tool: str, text: str, since: float) -> bool:
    """Did NEO's own code produce this NEEDS_* result during the call that started at `since`?
    Content read from the web, mail or a file can begin with the same words; it must never be
    able to halt a run or put a question to the user."""
    t = text.lstrip()
    if t.startswith(NEEDS_CONFIRM):
        return _pending is not None and _pending.tool == tool and _pending.staged_at >= since
    if t.startswith(NEEDS_USER):
        return t == _own_question
    return True
