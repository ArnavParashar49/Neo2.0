"""NEO as a library: embed NEO's agent — its tools, safety rules and confirmation gate — in
another app.

    import neo2
    from neo2.providers import OpenAICompatible

    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"),
                               provider=OpenAICompatible(client=my_openai, model="gpt-5.1")) as neo:
        r = await neo.ask("Summarise https://example.com/post in 3 bullets")
        if r.status == "needs_approval":            # show r.approval.summary / .detail to the human
            r = await neo.approve(r.approval.action_id, via=neo2.Approval("web-ui", "user:42"))

Only host code can approve an action NEO staged: :meth:`AsyncAgent.approve` with an
:class:`Approval`, or an ``approver`` callback that returns one. Text passed to ``ask()``, a tool
argument or a model's output never can.

Other AI frameworks: ``neo2.adapters.openai``, ``.anthropic``, ``.gemini`` and ``.langchain`` give
the whole agent as one tool of their loop; ``python -m neo2.mcp`` (the ``neo2-mcp`` script) serves it
to MCP clients.

Everything here is loaded on first use (PEP 562): ``import neo2`` loads only the standard
library, and ``from neo2 import AsyncAgent`` loads NEO's core without any model, vendor SDK,
macOS framework or browser.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names below, for type checkers and IDEs (nothing is imported at runtime)
    from neo.agent.prompt import SAFETY_RULES as SAFETY_RULES
    from neo.agent.registry import ToolContext as ToolContext
    from neo.agent.registry import ToolOutput as ToolOutput
    from neo.agent.session import answer_of as parse_yes_no  # noqa: F401
    from neo.runtime import NoRuntimeError as NoRuntimeError
    from neo.runtime import UnsupportedPlatform as UnsupportedPlatform
    from neo2.sdk.agent import AsyncAgent as AsyncAgent
    from neo2.sdk.approval import Approval as Approval
    from neo2.sdk.approval import ApprovalReentryError as ApprovalReentryError
    from neo2.sdk.approval import ApprovalRequest as ApprovalRequest
    from neo2.sdk.approval import Approver as Approver
    from neo2.sdk.config import AgentConfig as AgentConfig
    from neo2.sdk.config import MemoryConfig as MemoryConfig
    from neo2.sdk.hosttools import HostTool as HostTool
    from neo2.sdk.hosttools import host_tool as host_tool
    from neo2.sdk.results import Result as Result
    from neo2.sdk.results import StepRecord as StepRecord
    from neo2.sdk.results import ToolInfo as ToolInfo
    from neo2.sdk.results import ToolOutcome as ToolOutcome
    from neo2.sdk.sync import Agent as Agent
    from neo2.sdk.tools_api import ToolTurn as ToolTurn

__version__ = "2.1.0"

# name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    "AsyncAgent": ("neo2.sdk.agent", "AsyncAgent"),
    "Agent": ("neo2.sdk.sync", "Agent"),
    "AgentConfig": ("neo2.sdk.config", "AgentConfig"),
    "MemoryConfig": ("neo2.sdk.config", "MemoryConfig"),
    "Result": ("neo2.sdk.results", "Result"),
    "StepRecord": ("neo2.sdk.results", "StepRecord"),
    "ToolInfo": ("neo2.sdk.results", "ToolInfo"),
    "ToolOutcome": ("neo2.sdk.results", "ToolOutcome"),
    "ApprovalRequest": ("neo2.sdk.approval", "ApprovalRequest"),
    "Approval": ("neo2.sdk.approval", "Approval"),
    "Approver": ("neo2.sdk.approval", "Approver"),
    "ApprovalReentryError": ("neo2.sdk.approval", "ApprovalReentryError"),
    "ToolTurn": ("neo2.sdk.tools_api", "ToolTurn"),
    "ToolContext": ("neo.agent.registry", "ToolContext"),
    "ToolOutput": ("neo.agent.registry", "ToolOutput"),
    "host_tool": ("neo2.sdk.hosttools", "host_tool"),
    "HostTool": ("neo2.sdk.hosttools", "HostTool"),
    "SAFETY_RULES": ("neo.agent.prompt", "SAFETY_RULES"),
    "parse_yes_no": ("neo.agent.session", "answer_of"),
    "NoRuntimeError": ("neo.runtime", "NoRuntimeError"),
    "UnsupportedPlatform": ("neo.runtime", "UnsupportedPlatform"),
}
_SUBMODULES = ("events", "providers", "testing", "macos", "sdk", "adapters", "mcp")

__all__ = [*_EXPORTS, *_SUBMODULES]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod, attr = _EXPORTS[name]
        value = getattr(importlib.import_module(mod), attr)
        globals()[name] = value  # once: the next lookup is a plain attribute
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
