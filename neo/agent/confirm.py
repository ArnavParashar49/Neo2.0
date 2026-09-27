"""Confirm-before-acting gate with an append-only audit log.

A consequential tool calls :func:`gated` with its exact parameters. The first time, the operation
is staged and the tool returns ``NEEDS_CONFIRM``; the run halts and the user is asked, with the
full command / script / path on the overlay's card. Only the user can approve:

* the session re-invokes the staged tool inside :func:`approved` after the user's own "yes" —
  words they typed or said, or the overlay's button for that action id;
* a ``confirm`` argument from a model is ignored, so no model (and no web page or email steering
  one) can approve anything;
* the approval covers exactly what the card showed: the staged parameters *and* the summary and
  detail built from the screen at that moment. A call that differs in any of them — a longer
  command, another mail body, an element id that now points at a different button — asks again;
* every staging gets a fresh random action id, so a stale card can't approve a later one;
* a yes "for this task" (neo.agent.grants) lets later calls of the same tool on the same app in
  the same task through without a card — never a dangerous one, and never once the request has
  read something new and untrusted. The card says when it can be answered that way (`scopes`).

Only NEO's own code can halt a run with a question: :func:`stage` and :func:`ask_user` return a
:class:`Halt` string, which a web page or an email that happens to start with "NEEDS_CONFIRM"
never is.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from neo.agent import grants
from neo.config import settings

NEEDS_CONFIRM = "NEEDS_CONFIRM"
NEEDS_USER = "NEEDS_USER"
_CONTROL_ARGS = ("confirm", "cancel")  # never part of an operation: models used to pass these


class Halt(str):
    """A NEEDS_CONFIRM / NEEDS_USER result produced by NEO's own code (see the module doc).

    `action_id`: the card a NEEDS_CONFIRM stands for, so whoever asks the user adopts that card
    and never one another job staged at the same moment."""

    action_id: str

    def __new__(cls, text: str, action_id: str = "") -> Halt:
        h = super().__new__(cls, text)
        h.action_id = action_id
        return h


@dataclass
class Pending:
    action_id: str  # random per staging: what the card's buttons send back
    tool: str
    params: dict[str, Any]
    summary: str
    detail: str = ""  # exactly what will run (command, script, path + content…), for the card
    staged_at: float = field(default_factory=time.time)
    key: str = ""  # hash of tool + params + summary + detail: what an approval covers
    task: str = ""  # the task that asked (grants.current()); "" = a direct command
    target: grants.Target | None = None  # what it acts on, when a yes for this task could cover it


@dataclass
class _Approval:
    action_id: str
    released: dict[str, Any] | None = None
    key: str = ""


_pending: Pending | None = None
_approved: ContextVar[_Approval | None] = ContextVar("neo_approved_action", default=None)
_SECRET_KEYS = ("key", "token", "password", "secret")
_TTL_S = 120.0  # a question nobody answered: forget it rather than hijack the next request


def clean(params: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if k not in _CONTROL_ARGS}


def op_key(tool: str, params: dict[str, Any], summary: str = "", detail: str = "") -> str:
    blob = json.dumps([tool, clean(params), summary, detail], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


_CONTENT_KEYS = {
    "body", "content", "text", "new_string", "old_string", "brief", "value", "subject", "notes", "message",
}


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    """The audit trail records what was done (the command, the recipient, the file) — not the
    words of a message or a file."""
    out: dict[str, Any] = {}
    for k, v in clean(params).items():
        if any(s in k.lower() for s in _SECRET_KEYS):
            v = "***"
        elif k in _CONTENT_KEYS:
            v = f"<{len(str(v))} chars>"
        s = str(v)
        out[k] = s if len(s) <= 120 else s[:117] + "..."
    return out


_AUDIT_MAX_BYTES = 1_000_000
_AUDIT_KEEP_S = 180 * 86400


def audit(action_id: str, tool: str, params: dict[str, Any], status: str) -> None:
    """Best-effort append; never raises. Kept to ~1 MB plus one older file, at most 180 days."""
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
        path = settings().audit_log
        old = path.with_name(path.name + ".1")
        try:
            if path.stat().st_size > _AUDIT_MAX_BYTES:
                path.replace(old)
        except FileNotFoundError:
            pass
        _prune_audit(path, old)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


_last_prune = 0.0


def _prune_audit(*paths) -> None:
    """Drop lines older than 180 days (checked at most once a day): a light user's log never
    reaches the size limit, so size-based rotation alone would keep everything forever."""
    global _last_prune
    if time.time() - _last_prune < 86400:
        return
    _last_prune = time.time()
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - _AUDIT_KEEP_S))
    for p in paths:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        keep = [ln for ln in lines if _ts(ln) >= cutoff]
        if len(keep) == len(lines):
            continue
        if not keep:
            p.unlink(missing_ok=True)
            continue
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(p)


def _ts(line: str) -> str:
    try:
        return str(json.loads(line).get("ts", ""))
    except ValueError:
        return ""


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
    """What the overlay shows for a pending question. `scopes`: the ways beyond "just this once"
    it may be answered — ["task"] when a yes for this task could cover the calls that follow."""
    if not p:
        return None
    return {
        "action_id": p.action_id,
        "tool": p.tool,
        "summary": p.summary,
        "detail": p.detail,
        "scopes": grants.offered(p.task, p.target),
        "app": p.target.app if p.target is not None and grants.offered(p.task, p.target) else "",
    }


def stage(
    tool: str, params: dict[str, Any], summary: str, detail: str = "", *, target: grants.Target | None = None
) -> Halt:
    """Stage an operation and return the halt sentinel + the question for the user."""
    global _pending
    params = clean(params)
    key = op_key(tool, params, summary, detail)
    if peek() is not None and _pending is not None and _pending.key == key:
        # asked again: same card, same id
        return Halt(f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?", _pending.action_id)
    if _pending:
        audit(_pending.action_id, _pending.tool, _pending.params, "superseded")
        _announce("confirm_done", action_id=_pending.action_id, outcome="superseded")
    aid = f"{tool}:{secrets.token_hex(8)}"
    _pending = Pending(
        action_id=aid, tool=tool, params=dict(params), summary=summary, detail=detail, key=key,
        task=grants.current(), target=target,
    )
    audit(aid, tool, params, "staged")
    _announce("confirm", **card(_pending))  # type: ignore[arg-type]
    try:  # close the card when nobody answers, even if nothing else asks in the meantime
        asyncio.get_running_loop().call_later(_TTL_S + 1, peek)
    except RuntimeError:
        pass
    return Halt(f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?", aid)


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


def consume(key: str) -> dict[str, Any] | None:
    """Release the staged params — only inside :func:`approved` for the pending action, and only
    when the call rebuilt exactly the operation the card showed (same `op_key`)."""
    approval = _approved.get()
    if approval is None:
        return None
    if approval.released is not None:
        return approval.released if approval.key == key else None  # a second gate on the same call
    if not _pending or _pending.action_id != approval.action_id or _pending.key != key:
        return None
    approval.key = key
    released = consume_now(approval.action_id)
    approval.released = released
    return released


def cancel(reason: str = "user cancelled") -> str:
    global _pending
    if _pending:
        audit(_pending.action_id, _pending.tool, _pending.params, f"cancelled: {reason}")
        _announce("confirm_done", action_id=_pending.action_id, outcome=reason)
    _pending = None
    return "CANCELLED"


def approval_active() -> bool:
    """True only inside :func:`approved`: the session re-invoking a staged action after the
    user's yes."""
    return _approved.get() is not None


def approving() -> str:
    """The action id the user said yes to, inside :func:`approved` ("" elsewhere)."""
    a = _approved.get()
    return a.action_id if a is not None else ""


def gated(
    tool: str,
    params: dict[str, Any],
    summary: str,
    detail: str = "",
    *,
    always: bool = False,
    target: grants.Target | None = None,
) -> dict[str, Any] | str:
    """One-liner for tool handlers.

    Returns the released params when the user approved this exact operation (or when
    confirmation is turned off), otherwise the ``NEEDS_CONFIRM`` string for the model.
    `always`: ask even with confirmation turned off — for what reaches another person (a send).
    `target`: what the call acts on, from a tool whose calls a yes "for this task" may cover
    (neo.agent.grants); without one, every call asks.
    """
    params = clean(params)
    if not settings().confirm_destructive and not always:
        return params
    released = consume(op_key(tool, params, summary, detail))
    if released is not None:
        return released
    if target is not None and not always and (g := grants.covering(tool, target)) is not None:
        audit(f"{tool}:{secrets.token_hex(8)}", tool, params, f"ran under the task grant of {g.action_id} ({g.app})")
        return params
    return stage(tool, params, summary, detail, target=target)


def ask_user(msg: str) -> Halt:
    """A NEEDS_USER question from NEO's own code (not from a page or a file it read)."""
    return Halt(f"{NEEDS_USER}: {msg}")
