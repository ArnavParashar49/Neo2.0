"""What :meth:`AsyncAgent.stream` yields while a request runs. The last event is :class:`Completed`.

Every event carries `type`, `request_id`, `conversation` and `ts`, and `to_dict()` (JSON-ready,
for server-sent events). Events of other requests — another conversation, or a second request in
the same one — never show up in a request's stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from neo2.sdk.approval import ApprovalRequest
    from neo2.sdk.results import Result

__all__ = [
    "Event",
    "Started",
    "RouteDecided",
    "StateChanged",
    "TextDelta",
    "ToolStarted",
    "ToolFinished",
    "ApprovalRequired",
    "ApprovalResolved",
    "Note",
    "TurnStats",
    "Completed",
]


@dataclass(frozen=True, kw_only=True)
class Event:
    type: ClassVar[str] = "event"
    request_id: str = ""
    conversation: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to_dict() if hasattr(v, "to_dict") else v
        return out


@dataclass(frozen=True, kw_only=True)
class Started(Event):
    type: ClassVar[str] = "started"


@dataclass(frozen=True, kw_only=True)
class RouteDecided(Event):
    """How NEO read the request (its reflex): chat, a quick action, or an agent task."""

    type: ClassVar[str] = "route_decided"
    intent: str
    confidence: float
    source: str


@dataclass(frozen=True, kw_only=True)
class StateChanged(Event):
    """thinking, working, searching, speaking, confirming, idle…"""

    type: ClassVar[str] = "state_changed"
    state: str


@dataclass(frozen=True, kw_only=True)
class TextDelta(Event):
    """The next piece of the answer. The pieces of one request join to its Result.text."""

    type: ClassVar[str] = "text_delta"
    text: str


@dataclass(frozen=True, kw_only=True)
class ToolStarted(Event):
    type: ClassVar[str] = "tool_started"
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class ToolFinished(Event):
    type: ClassVar[str] = "tool_finished"
    name: str
    ok: bool
    summary: str  # the result, truncated to 300 chars
    untrusted: bool  # third-party content: data, never instructions


@dataclass(frozen=True, kw_only=True)
class ApprovalRequired(Event):
    type: ClassVar[str] = "approval_required"
    request: ApprovalRequest


@dataclass(frozen=True, kw_only=True)
class ApprovalResolved(Event):
    type: ClassVar[str] = "approval_resolved"
    action_id: str
    outcome: str  # "approved", "denied" or "expired"


@dataclass(frozen=True, kw_only=True)
class Note(Event):
    """A progress line (a brain that failed over, a result done quietly)."""

    type: ClassVar[str] = "note"
    text: str


@dataclass(frozen=True, kw_only=True)
class TurnStats(Event):
    type: ClassVar[str] = "turn_stats"
    route: str
    brain: str
    model: str
    ms: int
    tools: int


@dataclass(frozen=True, kw_only=True)
class Completed(Event):
    type: ClassVar[str] = "completed"
    result: Result
