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
* every staging gets a fresh random action id, so a stale card can't approve a later one.

Only NEO's own code can halt a run with a question: :func:`stage` and :func:`ask_user` return a
:class:`Halt` string, which a web page or an email that happens to start with "NEEDS_CONFIRM"
never is.

The pending question lives in a :class:`Gate`. NEO.app has one; an agent embedded in another app
(neo.runtime) has one per conversation, so one user's "yes" can never approve another's card
and two threads of one user never supersede each other's. An approval belongs to the gate that
issued it: inside another conversation it releases nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo import runtime
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


class Gate:
    """Where one conversation's pending question lives.

    NEO.app's own gate (`app=True`) audits to settings().audit_log, read at call time. An
    embedded agent's gate audits to `audit_path` (None: no file) and hands each record to
    `audit_sink`; its lines carry the agent's name. `ttl_s` None: the module's _TTL_S, read at
    call time."""

    def __init__(
        self,
        *,
        ttl_s: float | None = None,
        audit_path: Path | None = None,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        agent: str = "",
        app: bool = False,
    ) -> None:
        self.pending: Pending | None = None
        self.ttl_s = ttl_s
        self.audit_path = audit_path
        self.audit_sink = audit_sink
        self.agent = agent
        self.app = app

    def ttl(self) -> float:
        return _TTL_S if self.ttl_s is None else self.ttl_s

    def log_path(self) -> Path | None:
        return settings().audit_log if self.app else self.audit_path


@dataclass
class _Approval:
    action_id: str
    released: dict[str, Any] | None = None
    key: str = ""
    gate: Gate | None = None  # the gate whose card this approves: no other gate honours it
    via: dict[str, Any] | None = None  # who approved, through which channel (audited)


_default_gate = Gate(app=True)  # NEO.app's
_approved: ContextVar[_Approval | None] = ContextVar("neo_approved_action", default=None)
# The action ids staged while handling one request (neo.runtime's hosts ask "did *this* request
# stage the pending card?"). A list shared by reference, so ids staged in asyncio.gather children
# (read-only tools run in parallel) reach the request that created it.
_stage_sink: ContextVar[list[str] | None] = ContextVar("neo_stage_sink", default=None)
_SECRET_KEYS = ("key", "token", "password", "secret")
_TTL_S = 120.0  # a question nobody answered: forget it rather than hijack the next request


def _gate() -> Gate:
    """The bound conversation's gate (neo.runtime), else NEO.app's. In library mode with nothing
    bound this fails closed: never a shared slot another agent could approve."""
    sc = runtime.current_scope()
    if sc is not None:
        return sc.get_or_create("gate", lambda: _scope_gate(sc))
    runtime.check_unbound("the confirmation gate")
    return _default_gate


def _scope_gate(sc: runtime.Scope) -> Gate:
    rt = sc.runtime
    return Gate(
        audit_path=rt.settings.audit_log if rt.audit_file else None, audit_sink=rt.audit_sink, agent=rt.name
    )


def current_gate() -> Gate:
    return _gate()


def ttl_s() -> float:
    """How long the current gate's question stays open."""
    return _gate().ttl()


@contextmanager
def collecting() -> Iterator[list[str]]:
    """Collect the action ids staged (or asked again) inside, including by parallel tool calls."""
    sink: list[str] = []
    token = _stage_sink.set(sink)
    try:
        yield sink
    finally:
        _stage_sink.reset(token)


def _collect(aid: str) -> None:
    sink = _stage_sink.get()
    if sink is not None and aid not in sink:
        sink.append(aid)


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


def audit(
    action_id: str,
    tool: str,
    params: dict[str, Any],
    status: str,
    *,
    via: dict[str, Any] | None = None,
    gate: Gate | None = None,
) -> None:
    """Best-effort append; never raises. Kept to ~1 MB plus one older file, at most 180 days.

    `via`: who approved or declined, through which channel. It and the gate's agent name are
    added only when set, so NEO.app's lines stay exactly as they were."""
    g = gate or _gate()
    try:
        rec: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "action": action_id,
            "tool": tool,
            "params": _redact(params),
        }
        if g.agent:
            rec["agent"] = g.agent
        if via is not None:
            rec["via"] = via
        line = json.dumps(rec, default=str)  # a host's odd `via` value never costs the line
        if g.audit_sink is not None:
            try:
                g.audit_sink(json.loads(line))  # a copy: the host can't change what was logged
            except Exception:  # noqa: BLE001 — the host's callback must not block the gate
                pass
        path = g.log_path()
        if path is None:
            logging.getLogger("neo.audit").info(line)
            return
        old = path.with_name(path.name + ".1")
        try:
            if path.stat().st_size > _AUDIT_MAX_BYTES:
                path.replace(old)
        except FileNotFoundError:
            pass
        _prune_audit(path, old, gate=g)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


_last_prune = 0.0  # NEO.app's audit log
_last_prune_at: dict[str, float] = {}  # an embedded agent's, by file


def _prune_audit(*paths, gate: Gate | None = None) -> None:
    """Drop lines older than 180 days (checked at most once a day): a light user's log never
    reaches the size limit, so size-based rotation alone would keep everything forever."""
    global _last_prune
    if gate is None or gate.app:
        if time.time() - _last_prune < 86400:
            return
        _last_prune = time.time()
    else:
        k = str(paths[0])
        if time.time() - _last_prune_at.get(k, 0.0) < 86400:
            return
        _last_prune_at[k] = time.time()
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
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")
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
    """What the overlay shows for a pending question."""
    if not p:
        return None
    return {"action_id": p.action_id, "tool": p.tool, "summary": p.summary, "detail": p.detail}


def stage(tool: str, params: dict[str, Any], summary: str, detail: str = "") -> Halt:
    """Stage an operation and return the halt sentinel + the question for the user."""
    g = _gate()
    params = clean(params)
    key = op_key(tool, params, summary, detail)
    if _peek(g) is not None and g.pending is not None and g.pending.key == key:
        # asked again: same card, same id
        _collect(g.pending.action_id)
        return Halt(f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?", g.pending.action_id)
    if g.pending:
        audit(g.pending.action_id, g.pending.tool, g.pending.params, "superseded", gate=g)
        _announce("confirm_done", action_id=g.pending.action_id, outcome="superseded")
    aid = f"{tool}:{secrets.token_hex(8)}"
    g.pending = Pending(action_id=aid, tool=tool, params=dict(params), summary=summary, detail=detail, key=key)
    audit(aid, tool, params, "staged", gate=g)
    _collect(aid)
    _announce("confirm", **card(g.pending))  # type: ignore[arg-type]
    try:  # close the card when nobody answers, even if nothing else asks in the meantime
        asyncio.get_running_loop().call_later(g.ttl() + 1, peek)  # carries this gate's context
    except RuntimeError:
        pass
    return Halt(f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?", aid)


def peek() -> Pending | None:
    return _peek(_gate())


def _peek(g: Gate) -> Pending | None:
    if g.pending and time.time() - g.pending.staged_at > g.ttl():
        _cancel(g, "expired")
    return g.pending


@contextmanager
def approved(aid: str, *, via: dict[str, Any] | None = None) -> Iterator[None]:
    """The user said yes to exactly this action: the call made inside may go ahead.

    A context variable, so it covers only the task that re-invokes the tool — another job running
    at the same moment can't ride on the approval. Every gate that call passes through (the
    registry's and the handler's) sees the same approval; when it's over the question is closed
    whether or not a gate consumed it. It approves only this conversation's card (the gate it
    was given in). `via`: who said yes, through which channel — audited."""
    g = _gate()
    approval = _Approval(aid, gate=g, via=via)
    token = _approved.set(approval)
    try:
        yield
    finally:
        _approved.reset(token)
        if approval.released is None and g.pending and g.pending.action_id == aid:
            _consume_now(g, aid, via=via)


def consume_now(aid: str) -> dict[str, Any] | None:
    return _consume_now(_gate(), aid)


def _consume_now(g: Gate, aid: str, *, via: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not g.pending or g.pending.action_id != aid:
        return None
    p, g.pending = g.pending, None
    audit(aid, p.tool, p.params, "confirmed", via=via, gate=g)
    _announce("confirm_done", action_id=aid, outcome="confirmed")
    return p.params


def consume(key: str) -> dict[str, Any] | None:
    """Release the staged params — only inside :func:`approved` for the pending action, and only
    when the call rebuilt exactly the operation the card showed (same `op_key`)."""
    approval = _approved.get()
    if approval is None:
        return None
    g = _gate()
    if approval.gate is not g:
        return None  # another conversation's yes: it approves nothing here
    if approval.released is not None:
        return approval.released if approval.key == key else None  # a second gate on the same call
    if not g.pending or g.pending.action_id != approval.action_id or g.pending.key != key:
        return None
    approval.key = key
    released = _consume_now(g, approval.action_id, via=approval.via)
    approval.released = released
    return released


def cancel(reason: str = "user cancelled", *, via: dict[str, Any] | None = None) -> str:
    return _cancel(_gate(), reason, via=via)


def _cancel(g: Gate, reason: str, *, via: dict[str, Any] | None = None) -> str:
    if g.pending:
        audit(g.pending.action_id, g.pending.tool, g.pending.params, f"cancelled: {reason}", via=via, gate=g)
        _announce("confirm_done", action_id=g.pending.action_id, outcome=reason)
    g.pending = None
    return "CANCELLED"


def approval_active() -> bool:
    """True only inside :func:`approved` for this conversation's gate: the session re-invoking a
    staged action after the user's yes."""
    a = _approved.get()
    return a is not None and a.gate is _gate()


def gated(
    tool: str, params: dict[str, Any], summary: str, detail: str = "", *, always: bool = False
) -> dict[str, Any] | str:
    """One-liner for tool handlers.

    Returns the released params when the user approved this exact operation (or when
    confirmation is turned off), otherwise the ``NEEDS_CONFIRM`` string for the model.
    `always`: ask even with confirmation turned off — for what reaches another person (a send).
    """
    params = clean(params)
    # An embedded agent always asks: its confirmation is pinned on (config.LibrarySettings).
    if not settings().confirm_destructive and not always and not runtime.library_bound():
        return params
    released = consume(op_key(tool, params, summary, detail))
    if released is not None:
        return released
    return stage(tool, params, summary, detail)


def ask_user(msg: str) -> Halt:
    """A NEEDS_USER question from NEO's own code (not from a page or a file it read)."""
    return Halt(f"{NEEDS_USER}: {msg}")
