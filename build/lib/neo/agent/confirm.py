"""Confirm-before-destructive gate with an append-only audit log.

A destructive tool stages its operation via :func:`stage` and returns ``NEEDS_CONFIRM``.
The agent loop halts and asks the user. On "yes", the loop re-invokes the *same* tool
with ``confirm=True``; :func:`consume` releases the staged params only when the action id
matches, so a stale confirmation can never trigger a different staged operation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from neo.config import settings

NEEDS_CONFIRM = "NEEDS_CONFIRM"
NEEDS_USER = "NEEDS_USER"


@dataclass
class Pending:
    action_id: str
    tool: str
    params: dict[str, Any]
    summary: str
    staged_at: float = field(default_factory=time.time)


_pending: Pending | None = None
_SECRET_KEYS = ("key", "token", "password", "secret")


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k in ("confirm", "cancel"):
            continue
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


def stage(action_id: str, tool: str, params: dict[str, Any], summary: str) -> str:
    """Stage a destructive operation and return the halt sentinel + human question."""
    global _pending
    _pending = Pending(action_id=action_id, tool=tool, params=dict(params), summary=summary)
    audit(action_id, tool, params, "staged")
    return f"{NEEDS_CONFIRM}: {summary} — Should I go ahead?"


_TTL_S = 120.0  # a question nobody answered: forget it rather than hijack the next request


def peek() -> Pending | None:
    if _pending and time.time() - _pending.staged_at > _TTL_S:
        cancel("expired")
    return _pending


def consume(action_id: str) -> dict[str, Any] | None:
    """Release staged params if the id matches; otherwise leave untouched and return None."""
    global _pending
    if not _pending or _pending.action_id != action_id:
        return None
    p, _pending = _pending, None
    audit(action_id, p.tool, p.params, "confirmed")
    return p.params


def cancel(reason: str = "user cancelled") -> str:
    global _pending
    if _pending:
        audit(_pending.action_id, _pending.tool, _pending.params, f"cancelled: {reason}")
    _pending = None
    return "CANCELLED"


def gated(action_id: str, tool: str, params: dict[str, Any], summary: str) -> dict[str, Any] | str:
    """One-liner for tool handlers.

    Returns the released params when confirmed (or when confirmation is globally off),
    otherwise the ``NEEDS_CONFIRM`` string to return to the model.
    """
    if not settings().confirm_destructive:
        return params
    if as_bool(params.get("cancel")):
        return cancel()
    if as_bool(params.get("confirm")):
        released = consume(action_id)
        if released is not None:
            return released
        # Model said confirm without a matching stage — treat as a fresh request.
    return stage(action_id, tool, params, summary)
