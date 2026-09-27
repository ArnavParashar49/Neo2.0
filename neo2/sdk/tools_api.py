"""NEO's tools inside a host's own model loop: turns, and the tool specs in each API's format.

    specs = agent.export_tools("openai_chat")
    async with agent.turn(tool_context={"tenant": "acme"}) as t:     # one user request
        out = await agent.call_tool(call.name, call.args, turn=t)      # each call the host's model makes

A turn shares one ToolContext across the host's separate calls, so what one call read (a web
page) counts for the next (a memory write then asks first) — as inside NEO's own loop. Put
``neo2.SAFETY_RULES`` in the host model's system prompt.
"""

from __future__ import annotations

import copy
import re
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from neo.agent.registry import RegisteredTool, ToolContext

if TYPE_CHECKING:
    from neo2.sdk.agent import AsyncAgent

__all__ = ["FORMATS", "ToolTurn", "export"]

Format = Literal["openai_chat", "openai_responses", "anthropic", "gemini", "mcp", "jsonschema"]
FORMATS: tuple[str, ...] = ("openai_chat", "openai_responses", "anthropic", "gemini", "mcp", "jsonschema")
_PREFIX = re.compile(r"^[a-zA-Z0-9_-]*$")
_OPEN_WORLD = {"web", "browser"}  # categories whose tools reach the internet


@dataclass(eq=False)
class ToolTurn:
    """One request of the host's own model loop: the context its NEO tool calls share."""

    conversation: str
    source: Literal["user", "model"]
    context: ToolContext = field(repr=False)
    _agent: weakref.ReferenceType[AsyncAgent] = field(repr=False, default=None)  # type: ignore[assignment]
    closed: bool = False

    @property
    def taint(self) -> tuple[str, ...]:
        """The untrusted sources this turn has read so far ("caller": a model wrote the task)."""
        return tuple(self.context.taint)

    @property
    def extras(self) -> Mapping[str, Any]:
        return MappingProxyType(self.context.extras)

    def belongs_to(self, agent: AsyncAgent) -> bool:
        return self._agent is not None and self._agent() is agent


def check_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not _PREFIX.match(prefix) or len(prefix) > 30:
        raise ValueError("prefix must be letters, digits, '_' or '-' (at most 30)")
    return prefix


def export(tools: Iterable[RegisteredTool], fmt: str, prefix: str = "") -> list[dict[str, Any]]:
    """The tools (never hidden ones, never NEO's more_tools) as `fmt` wants them."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r} (one of {', '.join(FORMATS)})")
    check_prefix(prefix)
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.hidden or t.name == "more_tools":
            continue
        name = prefix + t.name
        if len(name) > 64:
            raise ValueError(f"{name!r} is longer than 64 characters: use a shorter prefix")
        params = copy.deepcopy(t.parameters) or {"type": "object", "properties": {}}
        desc = t.description
        if fmt == "openai_chat":
            out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
        elif fmt == "openai_responses":
            # strict stays off: NEO's schemas have optional properties
            out.append({"type": "function", "name": name, "description": desc, "parameters": params, "strict": False})
        elif fmt == "anthropic":
            out.append({"name": name, "description": desc, "input_schema": params})
        elif fmt == "gemini":
            out.append({"name": name, "description": desc, "parameters_json_schema": params})
        elif fmt == "mcp":
            read_only = t.read_only if t.read_only is not None else False
            out.append(
                {
                    "name": name,
                    "description": desc,
                    "inputSchema": params,
                    "annotations": {
                        "readOnlyHint": bool(read_only),
                        "destructiveHint": t.risk == "destructive",
                        "openWorldHint": bool(t.untrusted or t.category in _OPEN_WORLD),
                    },
                }
            )
        else:
            out.append({"name": name, "description": desc, "parameters": params})
    return out
