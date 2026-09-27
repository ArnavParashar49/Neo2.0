"""What a request, an approval or a single tool call returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from neo.providers.base import ImagePart
from neo2.sdk.approval import ApprovalRequest, _plain

Status = Literal[
    "done", "needs_approval", "needs_user", "denied", "expired", "stopped", "max_steps", "thrash", "error"
]
Route = Literal["chat", "quick", "agent", "confirm", "stop", "tool"]


@dataclass(frozen=True)
class StepRecord:
    """One tool call a request made."""

    tool: str
    args: Mapping[str, Any]
    ok: bool
    summary: str  # the result, truncated to 300 chars

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": _plain(self.args), "ok": self.ok, "summary": self.summary}


@dataclass(frozen=True)
class Result:
    """How a request (ask), or an approval of one (approve / deny), ended.

    `status`: done; needs_approval (show `approval` to a human, then approve() or deny());
    needs_user (NEO needs something only the user can say: `text` is the question); denied;
    expired (the approval was unknown, superseded or too old — nothing ran); stopped; max_steps;
    thrash (the model kept repeating itself); error."""

    status: Status
    text: str
    route: Route
    approval: ApprovalRequest | None
    steps: tuple[StepRecord, ...]
    brain: str
    model: str
    elapsed_ms: int
    request_id: str
    conversation: str
    read_untrusted: tuple[str, ...]  # untrusted sources read while handling it ("caller": source="model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text": self.text,
            "route": self.route,
            "approval": self.approval.to_dict() if self.approval else None,
            "steps": [s.to_dict() for s in self.steps],
            "brain": self.brain,
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
            "request_id": self.request_id,
            "conversation": self.conversation,
            "read_untrusted": list(self.read_untrusted),
        }


@dataclass(frozen=True)
class ToolOutcome:
    """One NEO tool called by the host's own model loop (AsyncAgent.call_tool)."""

    status: Literal["ok", "error", "needs_approval", "needs_user", "denied"]
    text: str  # already wrapped in <untrusted source=...> for untrusted tools
    images: tuple[ImagePart, ...]
    approval: ApprovalRequest | None
    untrusted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text": self.text,
            "images": len(self.images),
            "approval": self.approval.to_dict() if self.approval else None,
            "untrusted": self.untrusted,
        }


@dataclass(frozen=True)
class ToolInfo:
    """One of an agent's tools, as the host sees it."""

    name: str
    description: str
    parameters: Mapping[str, Any]  # JSON schema (object)
    risk: Literal["safe", "reversible", "destructive"]
    read_only: bool
    untrusted: bool  # its output is third-party content: fenced, and it taints the request
    gated: bool  # always asks before it runs (a host tool with confirmation, NEO's destructive tools)
    requires: tuple[str, ...]  # "macos", "playwright"
    origin: Literal["builtin", "host"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _plain(self.parameters),
            "risk": self.risk,
            "read_only": self.read_only,
            "untrusted": self.untrusted,
            "gated": self.gated,
            "requires": list(self.requires),
            "origin": self.origin,
        }
