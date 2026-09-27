"""NEO in an Anthropic Messages tool loop. Plain dicts: the anthropic package isn't imported.

The whole agent as one tool (recommended — NEO keeps its prompt, safety rules and gate):

    spec, run_neo = neo2.adapters.anthropic.agent_tool(agent)
    msg = await client.messages.create(model=..., max_tokens=..., messages=msgs, tools=[spec])
    results = [{"type": "tool_result", "tool_use_id": b.id, "content": await run_neo(b.input)}
               for b in msg.content if b.type == "tool_use" and b.name == spec["name"]]

Individual NEO tools in the host's own loop: ``agent.export_tools("anthropic")``, run each
``tool_use`` with ``agent.call_tool(b.name, b.input, turn=t)`` inside ``async with agent.turn() as
t``, and answer it with :func:`tool_result`. Put ``neo2.SAFETY_RULES`` in the system prompt.

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

__all__ = ["agent_tool", "tool_result"]


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
    """The whole agent as one Anthropic tool: ``(spec, handler)``.

    `spec` is ``{name, description, input_schema}``. `handler(input)` takes the ``tool_use``
    block's input (a dict, or its JSON), runs ``agent.ask(task, context=context, source="model")``
    and returns the text for the model — the answer, or NEEDS_CONFIRM / NEEDS_USER / a refusal,
    never an action id. With an inline approver the human's decision is made before it returns.
    `conversation`: the NEO thread these calls share; `tool_context`: ToolContext.extras for NEO's
    tools; `on_result`: gets each full :class:`neo2.Result` (host side: the approval card)."""
    c.check_agent(agent)
    c.check_name(name)
    spec = {
        "name": name,
        "description": c.description_for(agent, name, description),
        "input_schema": c.agent_parameters(),
    }
    handler = c.make_handler(
        agent,
        name=name,
        conversation=conversation,
        tool_context=tool_context,
        source=source,
        on_result=on_result,
    )
    return spec, handler


# The image types a Messages API image block takes.
IMAGE_TYPES = c.COMMON_IMAGE_TYPES


def tool_result(outcome: ToolOutcome, tool_use_id: str) -> dict[str, Any]:
    """A ``tool_result`` block answering one NEO tool call: a text block plus any screenshots as
    base64 image blocks. ``is_error`` is set only when the call failed — not for a card waiting
    on the human, a refusal or a question."""
    if not isinstance(tool_use_id, str) or not tool_use_id:
        raise ValueError("tool_use_id is the id of the model's tool_use block (a non-empty string)")
    shots, skipped = c.images(outcome, IMAGE_TYPES)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": c.with_skipped(c.outcome_text(outcome), skipped)}
    ]
    content += [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": c.mime_of(im.mime), "data": c.b64(im.data)},
        }
        for im in shots
    ]
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if c.is_error(outcome):
        block["is_error"] = True
    return block
