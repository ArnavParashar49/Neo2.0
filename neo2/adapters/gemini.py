"""NEO in a Gemini (google-genai) function-calling loop. Plain dicts: google-genai isn't imported.

The whole agent as one function (recommended — NEO keeps its prompt, safety rules and gate):

    decl, run_neo = neo2.adapters.gemini.agent_tool(agent)
    config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[decl])])
    resp = await client.aio.models.generate_content(model=..., contents=contents, config=config)
    for fc in resp.function_calls or []:
        if fc.name == decl["name"]:
            text = await run_neo(fc.args)
            part = {"function_response": {"id": fc.id, "name": fc.name, "response": {"output": text}}}

Individual NEO tools in the host's own loop: ``agent.export_tools("gemini")`` (function
declarations), run each call with ``agent.call_tool(fc.name, fc.args, turn=t)`` inside ``async
with agent.turn() as t``, and answer it with :func:`function_response`. Put ``neo2.SAFETY_RULES``
in the system instruction.

A card waiting for the human (status needs_approval) reads to the model as ``NEEDS_CONFIRM: …
waiting for the user's decision``, never with its id: the host shows ``agent.pending(conversation)``
(or what `on_result` got) to its human and calls ``agent.approve(id, via=neo2.Approval(...))``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from neo2.adapters import _common as c

if TYPE_CHECKING:
    from neo2.sdk.agent import AsyncAgent
    from neo2.sdk.results import Result, ToolOutcome
    from neo2.sdk.sync import Agent

__all__ = ["agent_tool", "function_response"]


def agent_tool(
    agent: AsyncAgent | Agent,
    *,
    name: str = c.DEFAULT_NAME,
    description: str | None = None,
    conversation: str = "default",
    tool_context: Mapping[str, Any] | None = None,
    source: Literal["user", "model"] = "model",
    on_result: Callable[[Result], Any] | None = None,
) -> tuple[dict[str, Any], c.Handler]:
    """The whole agent as one Gemini function: ``(declaration, handler)``.

    `declaration` is a FunctionDeclaration dict ``{name, description, parameters_json_schema}``
    (for ``types.Tool(function_declarations=[...])``). `handler(args)` takes the function call's
    args (a dict, or their JSON), runs ``agent.ask(task, context=context, source="model")`` and
    returns the text for the model — the answer, or NEEDS_CONFIRM / NEEDS_USER / a refusal, never
    an action id. With an inline approver the human's decision is made before it returns.
    `conversation`: the NEO thread these calls share; `tool_context`: ToolContext.extras for NEO's
    tools; `on_result`: gets each full :class:`neo2.Result` (host side: the approval card)."""
    c.check_agent(agent)
    c.check_name(name)
    decl = {
        "name": name,
        "description": c.description_for(agent, name, description),
        "parameters_json_schema": c.agent_parameters(),
    }
    handler = c.make_handler(
        agent,
        name=name,
        conversation=conversation,
        tool_context=tool_context,
        source=source,
        on_result=on_result,
    )
    return decl, handler


def function_response(outcome: ToolOutcome, name: str, *, call_id: str | None = None) -> list[dict[str, Any]]:
    """The parts answering one NEO function call: a ``function_response`` part (``{"output":
    text}``, or ``{"error": text}`` when the call failed), then any screenshots as ``inline_data``
    parts — as NEO's own Gemini brain sends them. Put them in the user turn that answers the call.
    `name` is the called function's name; `call_id` its ``FunctionCall.id`` when it had one."""
    if not isinstance(name, str) or not name:
        raise ValueError("name is the called function's name (a non-empty string)")
    if call_id is not None and not isinstance(call_id, str):
        raise TypeError("call_id must be a string (the FunctionCall's id) or None")
    shots, skipped = c.images(outcome)
    text = c.with_skipped(c.outcome_text(outcome), skipped)
    fr: dict[str, Any] = {"name": name, "response": {"error" if c.is_error(outcome) else "output": text}}
    if call_id:
        fr["id"] = call_id
    parts: list[dict[str, Any]] = [{"function_response": fr}]
    parts += [{"inline_data": {"mime_type": c.mime_of(im.mime), "data": bytes(im.data)}} for im in shots]
    return parts
