"""NEO inside other AI frameworks: OpenAI, Anthropic, Gemini and LangChain tool loops.

Every adapter sits on one :class:`neo2.AsyncAgent` (or :class:`neo2.Agent`), so NEO's gate,
fencing, taint and tool restriction are the same everywhere. Two ways in:

- **The whole agent as one tool** (recommended): ``agent_tool(agent)`` in
  :mod:`neo2.adapters.openai`, :mod:`~neo2.adapters.anthropic`, :mod:`~neo2.adapters.gemini` gives
  ``(spec, handler)``; :mod:`~neo2.adapters.langchain` gives a ``StructuredTool``. The model passes
  ``{task, context?}``; the handler runs ``agent.ask(task, context=context, source="model")`` and
  returns the text the model reads (:func:`result_text`).
- **Individual NEO tools** in the host's own loop: ``agent.export_tools(fmt)`` and
  ``agent.call_tool(name, args, turn=t)``, answered with each module's helper
  (``openai.tool_message`` / ``function_call_output``, ``anthropic.tool_result``,
  ``gemini.function_response``, ``langchain.tool_message``; the text is :func:`outcome_text`).

No adapter exposes an approve, confirm or cancel tool, argument or route, and no text a model reads
carries an action id. A staged action reads as ``NEEDS_CONFIRM: <summary> — waiting for the user's
decision in the app``; only host code approves it: ``agent.approve(action_id, via=neo2.Approval(...))``
from the host's own UI, or an inline ``approver`` that returns an ``Approval``. An outer framework's
own approval step (OpenAI Agents ``needs_approval``, a LangGraph interrupt, an MCP client's tool
confirmation) approves handing the task to NEO, never NEO's staged action.

Taint carries through: what NEO wrote after reading third-party content (a page, a host tool's
output) — its answer, a card's summary, a question — reaches the outer model inside
``<untrusted source="...">``, and an individual untrusted tool's output stays fenced as
``call_tool`` returned it. Every request an adapter makes starts as a model's words
(``source="model"``), so memory writes and data-carrying links ask the human first.

Nothing here imports a vendor SDK; :mod:`~neo2.adapters.langchain` imports langchain-core when
called.
"""

from __future__ import annotations

import importlib
from typing import Any

from neo2.adapters._common import outcome_text, result_text

__all__ = ["anthropic", "gemini", "langchain", "openai", "outcome_text", "result_text"]

_SUBMODULES = ("anthropic", "gemini", "langchain", "openai")


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
