"""NEO as a LangChain tool (needs langchain-core: the ``[langchain]`` extra). langchain-core is
imported only when a function here is called.

    neo_tool = neo2.adapters.langchain.agent_tool(agent)      # a StructuredTool
    llm_with_tools = llm.bind_tools([neo_tool])               # or create_agent(model, tools=[neo_tool])
    msg = await neo_tool.ainvoke(tool_call)                   # a ToolMessage for the model

The tool runs ``agent.ask(task, context=context, source="model")``: NEO keeps its prompt, safety
rules and gate. A card waiting for the human reads to the model as ``NEEDS_CONFIRM: … waiting for
the user's decision``, never with its id; the host shows ``agent.pending(conversation)`` (or what
`on_result` got) to its human and calls ``agent.approve(id, via=neo2.Approval(...))``. A LangGraph
interrupt or a framework's own tool approval approves handing the task to NEO — never NEO's own
staged action.

For individual NEO tools run in the host's own loop (``agent.call_tool``), :func:`tool_message`
turns an outcome into the ToolMessage answering the call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neo2.adapters import _common as c

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import StructuredTool

    from neo2.sdk.agent import AsyncAgent
    from neo2.sdk.results import Result, ToolOutcome
    from neo2.sdk.sync import Agent

__all__ = ["NeoTask", "agent_tool", "tool_message"]

_NEEDS = "neo2.adapters.langchain needs langchain-core (the [langchain] extra): pip install langchain-core"


class NeoTask(BaseModel):
    """The whole-agent tool's arguments (anything else the model passes is ignored)."""

    model_config = ConfigDict(extra="ignore")

    task: str = Field(description=c.TASK_DESCRIPTION)
    context: str = Field(default="", description=c.CONTEXT_DESCRIPTION)


def agent_tool(
    agent: AsyncAgent | Agent,
    *,
    name: str = c.DEFAULT_NAME,
    description: str | None = None,
    conversation: str = "default",
    tool_context: Mapping[str, Any] | None = None,
    source: Literal["user", "model"] = "model",
    on_result: Callable[[Result], Any] | None = None,
) -> StructuredTool:
    """The whole agent as a LangChain ``StructuredTool`` (a coroutine tool: use ``ainvoke``; a
    sync :class:`neo2.Agent` also gets a blocking ``invoke``).

    It returns the text for the model — the answer, or NEEDS_CONFIRM / NEEDS_USER / a refusal,
    never an action id. With an inline approver the human's decision is made before it returns.
    `conversation`: the NEO thread these calls share; `tool_context`: ToolContext.extras for NEO's
    tools; `on_result`: gets each full :class:`neo2.Result` (host side: the approval card)."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(_NEEDS) from e
    from neo2.sdk.sync import Agent

    c.check_agent(agent)
    c.check_name(name)
    desc = c.description_for(agent, name, description)
    handler = c.make_handler(
        agent,
        name=name,
        conversation=conversation,
        tool_context=tool_context,
        source=source,
        on_result=on_result,
    )

    async def arun(task: str, context: str = "") -> str:
        return await handler({"task": task, "context": context})

    run: Callable[..., str] | None = None
    if isinstance(agent, Agent):
        blocking = c.make_sync_handler(
            agent,
            name=name,
            conversation=conversation,
            tool_context=tool_context,
            source=source,
            on_result=on_result,
        )

        def run(task: str, context: str = "") -> str:
            return blocking({"task": task, "context": context})

    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name=name,
        description=desc,
        args_schema=NeoTask,
        infer_schema=False,
        handle_validation_error=c.usage_text(name),  # the model reads how to call it
    )


def tool_message(outcome: ToolOutcome, tool_call_id: str, *, name: str | None = None) -> ToolMessage:
    """The ToolMessage answering one NEO tool call: its text (with screenshots as standard image
    blocks), status "error" only when the call failed."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(_NEEDS) from e
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("tool_call_id is the id of the model's tool call (a non-empty string)")
    shots, skipped = c.images(outcome, c.COMMON_IMAGE_TYPES)
    text = c.with_skipped(c.outcome_text(outcome), skipped)
    content: str | list[str | dict[str, Any]] = text
    if shots:
        content = [{"type": "text", "text": text}]
        content += [
            {"type": "image", "base64": c.b64(im.data), "mime_type": c.mime_of(im.mime)} for im in shots
        ]
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=name,
        status="error" if c.is_error(outcome) else "success",
    )
