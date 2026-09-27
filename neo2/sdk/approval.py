"""Approvals: who may say yes to an action NEO staged, and what they are shown.

Only host code approves — :meth:`AsyncAgent.approve` with an :class:`Approval`, or an
``approver`` callback returning one. ``isinstance(x, Approval)`` is the only yes: None, True,
"yes", 1, an exception or a timeout all decline. Text passed to ``ask()`` never approves (NEO
treats it as a model's words), nor does any tool, tool argument or exported tool spec.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from neo.agent.session import answer_of as parse_yes_no

__all__ = [
    "Approval",
    "ApprovalReentryError",
    "ApprovalRequest",
    "Approver",
    "parse_yes_no",
]


@dataclass(frozen=True)
class Approval:
    """Which human channel approved: recorded in the audit log with the decision."""

    channel: str  # e.g. "web-ui", "slack-dm", "cli-tty", "mcp-dialog"
    approver: str  # e.g. "user:42"
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("Approval.channel must name the channel the human approved through")
        if not isinstance(self.approver, str):
            raise TypeError("Approval.approver must be a string (who approved)")

    def audit(self) -> dict[str, Any]:
        """What the audit line records about this approval (its `via`)."""
        out: dict[str, Any] = {"channel": self.channel, "approver": self.approver}
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class ApprovalRequest:
    """One staged action, waiting for a human. Show `summary` and `detail` verbatim."""

    action_id: str  # host-side only: never put it into text a model reads
    tool: str
    summary: str
    detail: str  # exactly what will run (command, script, path + content, mail body)
    params: Mapping[str, Any]  # a read-only copy of the staged parameters
    after_reading: tuple[str, ...]  # untrusted sources read before this was staged
    conversation: str
    request_id: str
    staged_at: float
    expires_at: float  # staged_at + the gate's lifetime (120 s): after that, approving does nothing

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool": self.tool,
            "summary": self.summary,
            "detail": self.detail,
            "params": _plain(self.params),
            "after_reading": list(self.after_reading),
            "conversation": self.conversation,
            "request_id": self.request_id,
            "staged_at": self.staged_at,
            "expires_at": self.expires_at,
        }


Approver = Callable[[ApprovalRequest], "Awaitable[Approval | None] | Approval | None"]


class ApprovalReentryError(RuntimeError):
    """ask() or stream() was called while an approver was deciding: an approver must be a human's
    answer, never another model run (and never this agent asking itself)."""


# Set while an approver runs (and in everything it starts): ask() and stream() refuse.
_approving: ContextVar[bool] = ContextVar("neo2_approving", default=False)


def guard_reentry() -> None:
    if _approving.get():
        raise ApprovalReentryError(
            "an approver called ask()/stream(): an approval must come from a human, not from another "
            "agent request"
        )


def _plain(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v
