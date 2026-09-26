"""Tool registry — the single catalog every brain, the reflex layer and the UI read from.

Register with the :func:`tool` decorator::

    @tool(
        "open_app",
        "Open an application by name.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        risk="safe", parallel_safe=True, category="apps",
    )
    async def open_app(args: dict, ctx: ToolContext) -> str: ...

Handlers may be sync or async and return a string (or a :class:`ToolOutput` when they
have images to attach, e.g. screenshots).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from neo.providers.base import ImagePart, ToolSpec

Risk = Literal["safe", "reversible", "destructive"]


@dataclass
class ToolOutput:
    text: str
    images: list[ImagePart] = field(default_factory=list)
    ok: bool = True


@dataclass
class ToolContext:
    """Runtime context handed to handlers."""

    user_text: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[str | ToolOutput] | str | ToolOutput]


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    risk: Risk = "safe"
    parallel_safe: bool = False
    category: str = "general"
    slow: bool = False  # UI speaks a filler while it runs
    fast_path: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (regex, args)
    early: bool = False  # fast path may fire mid-sentence (opt in: cheap, idempotent tools only)
    chain: bool = True  # may run as a later clause ("open notes and type hi"); False = first step only
    payload: bool = False  # the fast path captures free text that runs to the end of the sentence
    quiet: bool = False  # an action: when it succeeds, NEO just does it — no spoken reply
    hidden: bool = False  # not exposed to the model (internal/UI-only)
    timeout: float = 120.0  # seconds; a hung tool returns an error instead of stalling the loop

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class Registry:
    def __init__(self) -> None:
        self.last: tuple[str, float, bool] | None = None  # (tool, when, ok) of the latest invoke
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, t: RegisteredTool) -> RegisteredTool:
        if t.name in self._tools:
            print(f"[registry] replacing tool {t.name}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def all(self, *, category: str | None = None) -> list[RegisteredTool]:
        ts = list(self._tools.values())
        return [t for t in ts if category is None or t.category == category]

    def specs(self, *, include_hidden: bool = False) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values() if include_hidden or not t.hidden]

    def catalog(self, max_desc: int = 100) -> str:
        """Compact list for prompts / reflex descriptions."""
        return "\n".join(
            f"- {t.name} [{t.risk}]: {t.description[:max_desc]}" for t in self._tools.values() if not t.hidden
        )

    async def invoke(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        out = await self._invoke(name, args, ctx)
        self.last = (name, time.time(), out.ok)
        return out

    async def _invoke(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        t = self.get(name)
        if not t:
            return ToolOutput(f"Unknown tool: {name}", ok=False)
        try:
            r = t.handler(args or {}, ctx)
            if inspect.isawaitable(r):
                r = await asyncio.wait_for(r, t.timeout)
            if isinstance(r, ToolOutput):
                return r
            text = (r or "Done.").strip() if isinstance(r, str) else str(r)
            ok = not text.lower().startswith(("error", "failed"))
            return ToolOutput(text, ok=ok)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return ToolOutput(f"Error: {name} took longer than {int(t.timeout)}s and was stopped", ok=False)
        except Exception as e:  # noqa: BLE001 — tool failures are data for the model
            return ToolOutput(f"Error: {name} failed: {e}", ok=False)


_registry: Registry | None = None


def registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    *,
    risk: Risk = "safe",
    parallel_safe: bool = False,
    category: str = "general",
    slow: bool = False,
    fast_path: list[tuple[str, dict[str, Any]]] | None = None,
    early: bool = False,
    chain: bool = True,
    payload: bool = False,
    quiet: bool = False,
    hidden: bool = False,
    timeout: float = 120.0,
) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        registry().register(
            RegisteredTool(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                handler=fn,
                risk=risk,
                parallel_safe=parallel_safe,
                category=category,
                slow=slow,
                fast_path=fast_path or [],
                early=early,
                chain=chain,
                payload=payload,
                quiet=quiet,
                hidden=hidden,
                timeout=timeout,
            )
        )
        return fn

    return deco
