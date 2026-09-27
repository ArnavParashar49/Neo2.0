"""Tool registry — the catalog every brain, the reflex layer and the UI read from.

Every tool module registers into one process-wide *catalog* (:func:`catalog`). NEO.app uses it
as is: with no agent bound, :func:`registry` *is* the catalog. An agent embedded in another app
(neo.runtime) gets its own strict registry — a copy of just the catalog tools it was given, plus
its host's tools — and :func:`registry` returns that one while the agent is bound. So a tool
module imported in the middle of a run (the web tools import the policy lazily) grows the
catalog, never an agent's tool set.

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
import dataclasses
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from neo import runtime
from neo.providers.base import ImagePart, ToolSpec

log = logging.getLogger(__name__)

Risk = Literal["safe", "reversible", "destructive"]


@dataclass
class ToolOutput:
    text: str
    images: list[ImagePart] = field(default_factory=list)
    ok: bool = True
    halt: bool = False  # a NEEDS_CONFIRM / NEEDS_USER that NEO's own code produced (confirm.Halt)
    staged: str = ""  # a NEEDS_CONFIRM halt: the action id of the card it asks about
    # Overrides the tool's `quiet` for a success: True makes a quiet action speak (a first use,
    # a guessed time, a result NEO couldn't check), False keeps an answer tool's success silent.
    say: bool | None = None


def voiced(out: ToolOutput, quiet: bool) -> bool:
    """Is this result said aloud? A failure or a question for the user always is; a success
    follows the tool's own `say`, else only a non-quiet tool's success is."""
    if not out.ok or out.text.startswith("NEEDS_"):
        return True
    return out.say if out.say is not None else not quiet


# Who chose this call: the user's own words through a fast path ("user"), Gemini Live calling
# the tool directly ("live"), a model inside the agent loop ("agent"), or the session running a
# staged action after the user's yes ("approval").
Origin = Literal["user", "live", "agent", "approval"]


@dataclass
class ToolContext:
    """Runtime context handed to handlers."""

    user_text: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Untrusted sources (web pages, mail, files, the screen) read while handling this request.
    # Shared by reference across one request's tool calls; tools that could leak data or plant
    # instructions ask the user first once it is non-empty.
    taint: list[str] = field(default_factory=list)
    origin: Origin = "agent"

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    def tainted(self) -> str:
        """The first untrusted source this request has read ("" = none)."""
        return self.taint[0] if self.taint else ""


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[str | ToolOutput] | str | ToolOutput]
# (regex, args) or (regex, args, check): `check(filled_args)` runs on cached data only and says
# whether the pattern really claims the utterance; False hands it to the next pattern.
FastPath = tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], Callable[[dict[str, Any]], bool]]


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
    fast_path: list[FastPath] = field(default_factory=list)
    early: bool = False  # fast path may fire mid-sentence (opt in: cheap, idempotent tools only)
    chain: bool = True  # may run as a later clause ("open notes and type hi"); False = first step only
    payload: bool = False  # the fast path captures free text that runs to the end of the sentence
    quiet: bool = False  # an action: when it succeeds, NEO just does it — no spoken reply
    hidden: bool = False  # not exposed to the model (internal/UI-only)
    timeout: float = 120.0  # seconds; a hung tool returns an error instead of stalling the loop
    untrusted: bool = False  # output is third-party content (web, mail, files, screen): data, never instructions
    # doing it twice in a row gives nothing more (opening the same page or app): a repeat of the
    # same call moments later is a model retry, not the user's intent — unlike a keystroke
    idempotent: bool = False
    requires: frozenset[str] = frozenset()  # what it needs to run: "macos", "playwright", "web"
    origin: Literal["builtin", "host"] = "builtin"  # NEO's own tool, or an embedding host's
    read_only: bool | None = None  # changes nothing (None: not declared; parallel_safe hints it)

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class Registry:
    def __init__(self, *, strict: bool = False) -> None:
        self.last: tuple[str, float, bool] | None = None  # (tool, when, ok) of the latest invoke
        self._tools: dict[str, RegisteredTool] = {}
        # An agent's registry: a second tool under a name never silently replaces the first
        # (a host tool called "shell" must not stand in for NEO's gated shell).
        self.strict = strict

    def register(self, t: RegisteredTool, *, replace: bool = False) -> RegisteredTool:
        if t.name in self._tools:
            if self.strict and not replace:
                raise ValueError(f"a tool named {t.name!r} is already registered")
            log.info(f"[registry] replacing tool {t.name}")
        self._tools[t.name] = t
        return t

    def remove(self, name: str) -> RegisteredTool | None:
        return self._tools.pop(name, None)

    def subset(self, names: Iterable[str]) -> Registry:
        """A strict registry holding copies of just these tools (in this registry's order; a name
        it doesn't have is skipped). A copy, so later changes to either side never reach the other."""
        want = set(names)
        out = Registry(strict=True)
        for t in self._tools.values():
            if t.name in want:
                # as registered: never the surface NEO.app's task modes gave the catalog entry
                out._tools[t.name] = dataclasses.replace(t, fast_path=list(t.fast_path), **AS_REGISTERED.get(t.name, {}))
        return out

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
        from neo.agent import confirm

        # `confirm` / `cancel` belong to the user, not to a tool call: whatever a model passes is
        # dropped here (approval travels through confirm.approved()). So are NEO's own `_` args
        # (a staged send's `_plan`): they only exist in params NEO staged itself, so outside an
        # approval they come from a model, a page or an email and must never steer a tool.
        approving = confirm.approval_active()
        args = {
            k: v
            for k, v in (args or {}).items()
            if k not in ("confirm", "cancel") and (approving or not k.startswith("_"))
        }
        out = await self._invoke(name, args, ctx)
        self.last = (name, time.time(), out.ok)
        t = self.get(name)
        # Only NEO's own code can stop a run to ask the user something (confirm.Halt). Anything
        # else that merely starts with those words — a page, a file printed by `cat`, an email
        # read by AppleScript — is content.
        if not out.halt and out.text.lstrip().startswith(("NEEDS_CONFIRM", "NEEDS_USER")):
            out.text = "(content) " + out.text
        if t and t.untrusted and out.ok and not out.halt and name not in ctx.taint:
            ctx.taint.append(name)
        return out

    async def _invoke(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        t = self.get(name)
        if not t:
            return ToolOutput(f"Unknown tool: {name}", ok=False)
        try:
            r = t.handler(args or {}, ctx)
            if inspect.isawaitable(r):
                r = await asyncio.wait_for(r, t.timeout)
            from neo.agent.confirm import Halt

            if isinstance(r, ToolOutput):
                if isinstance(r.text, Halt):
                    r.halt, r.staged = True, r.staged or r.text.action_id
                return r
            halt = isinstance(r, Halt)
            text = (r or "Done.").strip() if isinstance(r, str) else str(r)
            ok = not text.lower().startswith(("error", "failed"))
            return ToolOutput(text, ok=ok, halt=halt, staged=r.action_id if halt else "")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return ToolOutput(f"Error: {name} took longer than {int(t.timeout)}s and was stopped", ok=False)
        except Exception as e:  # noqa: BLE001 — tool failures are data for the model
            return ToolOutput(f"Error: {name} failed: {e}", ok=False)


_registry: Registry | None = None
# The fields NEO.app's task modes rewrote on a catalog entry (neo.tools.tasks.apply_modes), as the
# tool registered them: an agent's copy always gets these (its task kinds run their legacy scripts).
AS_REGISTERED: dict[str, dict[str, Any]] = {}


def catalog() -> Registry:
    """Every tool any module registered, process-wide. NEO.app's registry."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def registry() -> Registry:
    """The tools of whoever is asking: the bound agent's own registry, else the catalog."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.registry
    runtime.check_unbound("registry()")
    return catalog()


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    *,
    risk: Risk = "safe",
    parallel_safe: bool = False,
    category: str = "general",
    slow: bool = False,
    fast_path: list[FastPath] | None = None,
    early: bool = False,
    chain: bool = True,
    payload: bool = False,
    quiet: bool = False,
    hidden: bool = False,
    timeout: float = 120.0,
    untrusted: bool = False,
    idempotent: bool = False,
    requires: Iterable[str] = (),
    read_only: bool | None = None,
) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        # Always the process catalog — even when a module is imported while an agent is bound.
        catalog().register(
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
                untrusted=untrusted,
                idempotent=idempotent,
                requires=frozenset(requires),
                read_only=read_only,
            )
        )
        return fn

    return deco


def fence(name: str, text: str) -> str:
    """Wrap third-party content so the model reads it as data, never as instructions."""
    body = text.replace("</untrusted", "</ untrusted").replace("<untrusted", "< untrusted")
    return f'<untrusted source="{name}">\n{body}\n</untrusted>'
