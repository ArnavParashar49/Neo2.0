"""NEO's MCP server: one :class:`~neo2.AsyncAgent` behind MCP's ``tools/list`` and ``tools/call``,
served over stdio (:func:`run_stdio`) on the low-level ``mcp.server.lowlevel.Server``.

Everything goes through the agent, so the gate, fencing, taint, secret rules and tool restriction
are the SDK's. What this module adds is who may say yes (``--approve``, the operator's choice):

- ``deny``: the agent's approver is "deny" — nothing gated runs.
- ``dialog``: the agent's approver is :class:`~neo2.mcp.approvers.DialogApprover`.
- ``elicit``: the agent defers, and the server asks the MCP client inside the same tool call — a
  server-to-client ``elicitation/create`` on handshake-era connections, an ``InputRequiredResult``
  on 2026-07-28 ones (the client answers by retrying the call; the ``request_state`` it echoes is
  a random one-use key into this process, bound to the call's tool and client, not a
  capability) — and only when the client's name is on the ``--trust-client`` list.

A gated call always ends inside the call it started in: done, denied or expired. The client (and
its model) gets text plus ``structuredContent {"status", "summary"}``, never an action id, and no
tool, argument or result is a way to approve.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import io
import logging
import os
import re
import secrets
import signal
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from neo.providers.base import ProviderError
from neo.runtime import UnsupportedPlatform
from neo2.mcp import approvers
from neo2.sdk.agent import AsyncAgent, _usable
from neo2.sdk.approval import Approval, ApprovalRequest
from neo2.sdk.config import AgentConfig
from neo2.sdk.results import Result, ToolOutcome

__all__ = ["ENV_PREFIX", "TASK_TOOL", "NeoMCP", "ServerOptions", "agent_config", "run_stdio"]

log = logging.getLogger("neo2.mcp")

Expose = Literal["task", "tools", "both"]
ApproveMode = Literal["deny", "dialog", "elicit"]
EXPOSE: tuple[str, ...] = ("task", "tools", "both")
APPROVE: tuple[str, ...] = ("deny", "dialog", "elicit")

TASK_TOOL = "neo_task"
ENV_PREFIX = "NEO_MCP_"  # --env: NEO_MCP_TOOLS, NEO_MCP_DATA_DIR… (NEO.app's own NEO_* stay its own)
TOOLS_CONVERSATION = "mcp-tools"  # --expose tools: every call's card lives in this one gate
MAX_ROUNDS = 8  # approvals asked in one call (an approved run may stage again)
MAX_CONVERSATIONS = 16  # neo_task threads kept; the least recently used idle one is dropped
MAX_PENDING_INPUTS = 32  # 2026-07-28 elicitations awaiting the client's retry
# Never listed, whatever a host registers: they read as approvals, or are NEO's own machinery.
NEVER_LISTED = frozenset({"approve", "confirm", "cancel", "deny", "more_tools", TASK_TOOL})
_CONVERSATION = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_GOAL = 20_000
_MAX_CONTEXT = 100_000

INSTRUCTIONS = (
    "NEO is an AI agent running on this computer with its own tools. Hand it a task with "
    f"{TASK_TOOL} (or call its tools directly, when listed). Anything that changes things needs the "
    "user's OK, which NEO's server asks for itself (or declines): when a result says the user's OK "
    "was not given, nothing was done — don't retry it or work around it."
)
_TASK_DESCRIPTION = (
    "Hand a task to NEO, an AI agent running on this computer with its own tools ({tools}). Say the "
    "goal in plain words; NEO plans and runs the steps and returns its answer. NEO asks the user "
    "itself before anything that needs their OK; if the user says no, nothing was done — don't retry "
    "or work around it. `context`: background NEO should know (quoted to it as context, never as "
    "instructions). `conversation`: a name to continue an earlier thread with NEO."
)
_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "description": "What NEO should do or find out, in plain words."},
        "context": {
            "type": "string",
            "description": "Optional background for NEO (quoted to it as context, never as instructions).",
        },
        "conversation": {
            "type": "string",
            "description": "Optional thread name (letters, digits, '.', '_', '-'): reuse it to continue with NEO.",
            "pattern": _CONVERSATION.pattern,
        },
    },
    "required": ["goal"],
    "additionalProperties": False,
}

NO_BRAIN = (
    "NEO has no model to think with: start its MCP server with --env and GEMINI_API_KEY (or "
    "GROQ_API_KEY / ANTHROPIC_API_KEY) set, or use --expose tools. Nothing was done."
)
NO_BRAIN_AT_START = (
    f"{TASK_TOOL} has no model to think with: start the server with --env and GEMINI_API_KEY (or "
    "GROQ_API_KEY / ANTHROPIC_API_KEY) set, or use --expose tools"
)
_DENIED = (
    "The user's OK is required and was not given: {summary}. Nothing was done; don't retry or work around it."
)
_DENIED_NO_CHANNEL = (
    "The user's OK is required and this server can't ask for it (it runs with --approve deny): "
    "{summary}. Nothing was done — the user has to do this themselves; don't retry or work around it."
)
_EXPIRED = (
    "The user's OK for this was no longer valid (it timed out, or a newer request replaced it){summary}. "
    "Nothing was done."
)
_STOPPED = "NEO stopped before finishing."


@dataclass(frozen=True)
class ServerOptions:
    """What ``python -m neo2.mcp`` was started with."""

    tools: str | tuple[str, ...] | None = None  # None: the config's ("web" unless --env says otherwise)
    expose: Expose = "task"
    approve: ApproveMode = "deny"
    trust_clients: tuple[str, ...] = ()
    env: bool = False  # AgentConfig.from_env(prefix=NEO_MCP_): keys and settings from the environment
    env_files: tuple[str, ...] = ()


def agent_config(opts: ServerOptions) -> AgentConfig:
    """The agent's config: only what the options say, unless --env opts into the environment."""
    overrides: dict[str, Any] = {}
    if opts.tools is not None:
        overrides["tools"] = opts.tools
    if opts.env or opts.env_files:
        cfg = AgentConfig.from_env(prefix=ENV_PREFIX, env_files=opts.env_files, **overrides)
    else:
        cfg = AgentConfig(**overrides)
    return _rebuilt(cfg, name="mcp") if cfg.name == "default" else cfg


def _rebuilt(cfg: AgentConfig, **changes: Any) -> AgentConfig:
    return AgentConfig(**{**dict(cfg), **changes})  # validated again, like any config


class _NoBrain:
    """The agent's brain when the operator gave none: every call says so."""

    name = "none"
    model = ""
    supports_vision = False
    supports_tools = True

    async def complete(self, messages: Any, **kw: Any) -> Any:
        raise ProviderError(NO_BRAIN)

    async def stream(self, messages: Any, **kw: Any) -> Any:
        raise ProviderError(NO_BRAIN)
        yield ""  # pragma: no cover — an async generator


@dataclass
class _Waiting:
    """A 2026-07-28 elicitation sent back in an InputRequiredResult: what the retry resumes."""

    tool: str
    client: str | None
    card: ApprovalRequest
    rounds: int


_UNASKED: Any = object()


class NeoMCP:
    """NEO's MCP server: an agent built from `config` plus the low-level MCP ``server``.

    `expose`: "task" (one ``neo_task`` tool), "tools" (NEO's tools one by one) or "both".
    `approve`: "deny", "dialog" or "elicit" (with `trust_clients`, the client names whose
    elicitation answers count). `provider`, `fast_provider`, `host_tools`, `on_audit`: as for
    :class:`~neo2.AsyncAgent`. Without a provider or a key in `config`, ``neo_task`` answers that
    NEO has no model (the tools still work)."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        expose: Expose = "task",
        approve: ApproveMode = "deny",
        trust_clients: Sequence[str] = (),
        provider: Any = None,
        fast_provider: Any = None,
        host_tools: Sequence[Any] = (),
        on_audit: Any = None,
        dialog: approvers.DialogApprover | None = None,
    ) -> None:
        if expose not in EXPOSE:
            raise ValueError(f"expose is one of {', '.join(EXPOSE)}")
        if approve not in APPROVE:
            raise ValueError(f"approve is one of {', '.join(APPROVE)}")
        trust = tuple(trust_clients)
        if any(not isinstance(c, str) or not c.strip() for c in trust):
            raise ValueError("a trusted client is named by a non-empty string (its clientInfo.name)")
        if trust and approve != "elicit":
            raise ValueError("--trust-client only means something with --approve elicit")
        cfg = config if config is not None else AgentConfig()
        if not isinstance(cfg, AgentConfig):
            raise TypeError("config must be a neo2.AgentConfig")
        approver: Any
        self.dialog: approvers.DialogApprover | None = None
        if approve == "dialog":
            if sys.platform != "darwin":
                raise UnsupportedPlatform("--approve dialog shows a macOS dialog; use deny or elicit here")
            if "osascript" not in cfg.protected_apps:  # NEO's own clicks never press "Allow once"
                cfg = _rebuilt(cfg, protected_apps=(*cfg.protected_apps, "osascript"))
            self.dialog = dialog or approvers.DialogApprover()
            approver = self.dialog.ask
        elif approve == "deny":
            approver = "deny"
        else:
            approver = "defer"  # the server asks the client, inside the call (see _settle)
        if approve == "elicit" and not trust:
            log.warning("--approve elicit with no --trust-client: every gated action will be declined")
        self.expose: Expose = expose
        self.approve: ApproveMode = approve
        self.trust_clients = frozenset(trust)
        self.has_brain = (
            _given(provider)
            or _given(fast_provider)
            or _usable(cfg, cfg.brain)
            or _usable(cfg, cfg.fast_brain)
        )
        self.agent = AsyncAgent(
            cfg,
            provider=provider if self.has_brain else _NoBrain(),
            fast_provider=fast_provider,
            approver=approver,
            host_tools=host_tools,
            on_audit=on_audit,
        )
        self._waiting: OrderedDict[str, _Waiting] = OrderedDict()
        self._conversations: OrderedDict[str, None] = OrderedDict()
        self.server: Server[Any] = Server(
            "neo",
            version=_version(),
            title="NEO",
            instructions=INSTRUCTIONS,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    @classmethod
    def from_options(cls, opts: ServerOptions, **kw: Any) -> NeoMCP:
        return cls(
            agent_config(opts),
            expose=opts.expose,
            approve=opts.approve,
            trust_clients=opts.trust_clients,
            **kw,
        )

    async def aclose(self) -> None:
        self._waiting.clear()
        await self.agent.aclose()

    async def __aenter__(self) -> NeoMCP:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---- tools/list -------------------------------------------------------------------------------
    def tools(self) -> list[types.Tool]:
        """What ``tools/list`` shows."""
        out: list[types.Tool] = []
        specs = [s for s in self.agent.export_tools("mcp") if s["name"] not in NEVER_LISTED]
        if self.expose in ("task", "both"):
            out.append(self._task_tool(specs))
        if self.expose in ("tools", "both"):
            for s in specs:
                a = s["annotations"]
                out.append(
                    types.Tool(
                        name=s["name"],
                        description=s["description"],
                        input_schema=s["inputSchema"],
                        annotations=types.ToolAnnotations(
                            read_only_hint=bool(a["readOnlyHint"]),
                            destructive_hint=bool(a["destructiveHint"]),
                            open_world_hint=bool(a["openWorldHint"]),
                        ),
                    )
                )
        leaked = [t.name for t in out if t.name in NEVER_LISTED - {TASK_TOOL}]
        if leaked:  # never: they're filtered above, and the SDK refuses such names
            raise RuntimeError(f"refusing to list {leaked}")
        return out

    def _task_tool(self, specs: list[dict[str, Any]]) -> types.Tool:
        names = [s["name"] for s in specs]
        gated = any(t.gated for t in self.agent.tools)
        shown = ", ".join(names[:24]) + (f" and {len(names) - 24} more" if len(names) > 24 else "")
        return types.Tool(
            name=TASK_TOOL,
            description=_TASK_DESCRIPTION.format(tools=shown or "none: it answers from what it knows"),
            input_schema=copy.deepcopy(_TASK_SCHEMA),
            annotations=types.ToolAnnotations(
                read_only_hint=False,
                destructive_hint=gated or any(s["annotations"]["destructiveHint"] for s in specs),
                open_world_hint=any(s["annotations"]["openWorldHint"] for s in specs),
            ),
        )

    def _callable_tools(self) -> set[str]:
        return {t.name for t in self.agent.tools} - NEVER_LISTED

    async def _list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools())

    # ---- tools/call -------------------------------------------------------------------------------
    async def _call_tool(
        self, ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult | types.InputRequiredResult:
        client = _client_name(ctx)
        token = approvers.current_client.set(client)
        try:
            if params.request_state is not None or params.input_responses:
                return await self._resume(ctx, params, client)
            name, args = params.name, dict(params.arguments or {})
            if name == TASK_TOOL and self.expose in ("task", "both"):
                return await self._task(ctx, args, client)
            if self.expose in ("tools", "both") and name in self._callable_tools():
                return await self._tool(ctx, name, args, client)
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {name}")
        finally:
            approvers.current_client.reset(token)

    async def _task(
        self, ctx: Any, args: dict[str, Any], client: str | None
    ) -> types.CallToolResult | types.InputRequiredResult:
        goal, context, conv = args.get("goal"), args.get("context", ""), args.get("conversation") or "default"
        if not isinstance(goal, str) or not goal.strip() or len(goal) > _MAX_GOAL:
            return _text(
                "`goal` must be a non-empty string (at most 20,000 characters).", "error", error=True
            )
        if not isinstance(context, str) or len(context) > _MAX_CONTEXT:
            return _text("`context` must be a string (at most 100,000 characters).", "error", error=True)
        if not isinstance(conv, str) or not _CONVERSATION.match(conv):
            return _text("`conversation` must be 1-64 letters, digits, '.', '_' or '-'.", "error", error=True)
        if not self.has_brain:
            return _text(NO_BRAIN, "error", error=True)
        r = await self.agent.ask(
            goal, context=context, conversation=self._conversation("task:" + conv), source="model"
        )
        if r.status == "needs_approval" and r.approval is not None:
            return await self._settle(ctx, TASK_TOOL, r.approval, client, rounds=0)
        return self._from_result(r, r.approval)

    async def _tool(
        self, ctx: Any, name: str, args: dict[str, Any], client: str | None
    ) -> types.CallToolResult | types.InputRequiredResult:
        # Each call is its own turn, started as a model's request: it counts as having read
        # something untrusted ("caller"), so memory writes and data-carrying links ask first.
        async with self.agent.turn(conversation=TOOLS_CONVERSATION, source="model") as turn:
            out = await self.agent.call_tool(name, args, turn=turn)
        if out.status == "needs_approval" and out.approval is not None:
            return await self._settle(ctx, name, out.approval, client, rounds=0)
        return self._from_outcome(out)

    # ---- approvals (elicit mode; the agent's own approver settled deny and dialog) -----------------
    async def _settle(
        self,
        ctx: Any,
        tool: str,
        card: ApprovalRequest,
        client: str | None,
        *,
        rounds: int,
        answer: Any = _UNASKED,
    ) -> types.CallToolResult | types.InputRequiredResult:
        """Ask about `card` (or take the `answer` a retry brought), then approve or deny it — as
        often as the approved run stages again, at most MAX_ROUNDS times. Never leaves a card
        pending except while a 2026-07-28 client is asked (its retry resumes here)."""
        while True:
            refusal = self._refusal(ctx, client)
            if refusal is None and answer is _UNASKED and rounds >= MAX_ROUNDS:
                refusal = f"more than {MAX_ROUNDS} approvals in one call"
            if refusal is not None:
                log.warning("declined %s without asking: %s", card.tool, refusal)
                r = await self.agent.deny(
                    card.action_id,
                    via=Approval(
                        approvers.ELICIT_CHANNEL if self.approve == "elicit" else "mcp",
                        f"client:{client or 'unnamed'}",
                        note=refusal,
                    ),
                )
                return self._from_result(r, card)
            if answer is _UNASKED:
                if ctx.protocol_version in MODERN_PROTOCOL_VERSIONS:
                    return self._ask_later(tool, card, client, rounds)
                answer = await self._elicit(ctx, card, client)
            rounds += 1
            decision = approvers.elicit_decision(answer, client)
            answer = _UNASKED
            if decision is None:
                r = await self.agent.deny(
                    card.action_id,
                    via=Approval(
                        approvers.ELICIT_CHANNEL, f"client:{client or 'unnamed'}", note="not approved"
                    ),
                )
                return self._from_result(r, card)
            r = await self.agent.approve(card.action_id, via=decision)
            if r.status == "needs_approval" and r.approval is not None:
                card = r.approval  # the approved run staged again: ask about that one
                continue
            return self._from_result(r, card)

    def _refusal(self, ctx: Any, client: str | None) -> str | None:
        """Why this call's card can't be put to anyone (None: ask the client)."""
        if self.approve != "elicit":
            return "this server has no one to ask"
        if client is None or client not in self.trust_clients:
            return f"client {client!r} isn't on the --trust-client list"
        caps = ctx.session.client_capabilities
        if caps is None or caps.elicitation is None:
            return f"client {client!r} can't ask its user (no elicitation support)"
        return None

    async def _elicit(self, ctx: Any, card: ApprovalRequest, client: str | None) -> Any:
        """Handshake-era connections: ask the client now, within the card's lifetime."""
        wait = min(self.agent.config.approval_timeout_s, card.expires_at - time.time() - 0.5)
        if wait <= 0:
            return None
        try:
            async with asyncio.timeout(wait):
                return await ctx.session.elicit_form(
                    approvers.approval_text(card, client=client),
                    approvers.ELICIT_SCHEMA,
                    related_request_id=ctx.request_id,
                )
        except TimeoutError:
            log.warning("the client didn't answer in time: declined")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — an elicitation that fails declines
            log.warning("elicitation failed (%s): declined", e)
        return None

    def _ask_later(
        self, tool: str, card: ApprovalRequest, client: str | None, rounds: int
    ) -> types.InputRequiredResult:
        """2026-07-28 connections: the question goes back in the result, and the client's retry
        (carrying its answer and this one-use key) resumes the call."""
        now = time.time()
        for k in [k for k, w in self._waiting.items() if w.card.expires_at < now]:
            del self._waiting[k]
        while len(self._waiting) >= MAX_PENDING_INPUTS:
            self._waiting.popitem(last=False)  # its card lapses on its own
        key = secrets.token_urlsafe(32)
        self._waiting[key] = _Waiting(tool, client, card, rounds)
        return types.InputRequiredResult(
            input_requests={approvers.ELICIT_KEY: approvers.elicit_request(card, client=client)},
            request_state=key,
        )

    async def _resume(
        self, ctx: Any, params: types.CallToolRequestParams, client: str | None
    ) -> types.CallToolResult | types.InputRequiredResult:
        w = self._waiting.pop(params.request_state or "", None)
        if w is None or w.tool != params.name:
            return _text(
                "That approval question is unknown or no longer open, so nothing was done. Ask again if it's still wanted.",
                "expired",
                error=True,
            )
        if client != w.client:
            r = await self.agent.deny(
                w.card.action_id,
                via=Approval(
                    approvers.ELICIT_CHANNEL,
                    f"client:{client or 'unnamed'}",
                    note="answered by another client",
                ),
            )
            return self._from_result(r, w.card)
        answer = (params.input_responses or {}).get(approvers.ELICIT_KEY)
        return await self._settle(ctx, w.tool, w.card, client, rounds=w.rounds, answer=answer)

    # ---- results ------------------------------------------------------------------------------------
    def _from_result(self, r: Result, card: ApprovalRequest | None) -> types.CallToolResult:
        card = r.approval or card
        summary = card.summary if card is not None else ""
        status = r.status
        if status == "done":
            text = r.text.strip() or "Done."
        elif status in ("denied", "needs_approval"):  # never left pending: it was declined
            status = "denied"
            text = (_DENIED_NO_CHANNEL if self.approve == "deny" else _DENIED).format(
                summary=summary or "that action"
            )
        elif status == "needs_user":
            text = f"NEEDS_USER: {r.text}"
        elif status == "expired":
            text = _EXPIRED.format(summary=f": {summary}" if summary else "")
        elif status == "stopped":
            text = _STOPPED
        else:  # max_steps, thrash, error
            text = r.text.strip() or f"NEO couldn't finish ({status})."
        return _text(_scrub(text, card), status, summary=summary, error=status == "error")

    def _from_outcome(self, out: ToolOutcome) -> types.CallToolResult:
        summary = out.approval.summary if out.approval is not None else ""
        if out.status == "denied":
            text = (_DENIED_NO_CHANNEL if self.approve == "deny" else _DENIED).format(
                summary=summary or "that action"
            )
        elif out.status == "needs_user":
            text = f"NEEDS_USER: {out.text}"
        elif out.status == "stopped":
            text = _STOPPED
        else:
            text = out.text
        images = [
            types.ImageContent(type="image", data=base64.b64encode(i.data).decode("ascii"), mime_type=i.mime)
            for i in out.images
        ]
        return _text(
            _scrub(text, out.approval),
            out.status,
            summary=summary,
            error=out.status == "error",
            images=images,
        )

    def _conversation(self, name: str) -> str:
        """A neo_task thread; beyond MAX_CONVERSATIONS the least recently used idle one goes."""
        self._conversations[name] = None
        self._conversations.move_to_end(name)
        while len(self._conversations) > MAX_CONVERSATIONS:
            oldest = next(iter(self._conversations))
            if oldest == name:
                break
            try:
                self.agent.reset(oldest)
            except RuntimeError:  # still running a request: kept for now
                break
            del self._conversations[oldest]
        return name


def _given(p: Any) -> bool:
    return bool(p) if isinstance(p, (list, tuple)) else p is not None


def _client_name(ctx: Any) -> str | None:
    params = getattr(ctx.session, "client_params", None)
    info = getattr(params, "client_info", None)
    name = getattr(info, "name", None)
    return name if isinstance(name, str) and name else None


def _scrub(text: str, card: ApprovalRequest | None) -> str:
    """Belt and braces: an action id never reaches the client, whatever a text said."""
    if card is not None and card.action_id:
        text = text.replace(card.action_id, "[action]")
    return text


def _text(
    text: str,
    status: str,
    *,
    summary: str = "",
    error: bool = False,
    images: Sequence[types.ImageContent] = (),
) -> types.CallToolResult:
    structured: dict[str, Any] = {"status": status}
    if summary:
        structured["summary"] = summary
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text), *images],
        structured_content=structured,
        is_error=error,
    )


def _version() -> str:
    try:
        import neo2

        return str(neo2.__version__)
    except Exception:  # noqa: BLE001
        return ""


# ---- stdio ---------------------------------------------------------------------------------------


async def run_stdio(
    options: ServerOptions | None = None,
    *,
    wire_fd: int | None = None,
    handle_signals: bool = False,
    **neo_kw: Any,
) -> None:
    """Serve NEO over stdin/stdout until the client closes stdin. `wire_fd`: the real stdout, as
    :func:`neo2.mcp.claim_stdout` saved it (``main()`` claims it before importing anything);
    without it this claims fd 1 itself. `handle_signals` (``main()``): SIGTERM and SIGHUP close
    the agent — its cards, browser, memory and temp folder — before the process ends, instead of
    killing it with its temp folder left behind."""
    import anyio
    from mcp.server.stdio import stdio_server

    from neo2.mcp import claim_stdout, release_stdout

    opts = options or ServerOptions()
    own = wire_fd is None
    fd = claim_stdout() if wire_fd is None else wire_fd
    unhook = None
    try:
        neo = NeoMCP.from_options(opts, **neo_kw)
        if handle_signals:
            unhook = _stop_on_signals(neo)
        try:
            if not neo.has_brain and neo.expose != "tools":
                log.warning(NO_BRAIN_AT_START)
            names = sorted(neo._callable_tools())
            log.info(
                "serving %d NEO tool(s) (%s), expose=%s, approve=%s",
                len(names),
                ", ".join(names) or "none",
                neo.expose,
                neo.approve,
            )
            raw = os.fdopen(fd, "wb", closefd=False)
            wire = anyio.wrap_file(io.TextIOWrapper(raw, encoding="utf-8", newline="\n"))
            async with stdio_server(stdout=wire) as (read, write):
                await neo.server.run(read, write, neo.server.create_initialization_options())
        finally:
            with contextlib.suppress(Exception):
                await neo.aclose()
    finally:
        if unhook is not None:
            unhook()
        if own:
            release_stdout(fd)


def _stop_on_signals(neo: NeoMCP) -> Any:
    """SIGTERM/SIGHUP: close the agent (its cards, browser, memory and temp folder), then end the
    process — the transport's stdin reader blocks in a thread nothing can interrupt. Returns the
    undo."""
    loop = asyncio.get_running_loop()
    stopping: list[asyncio.Task[None]] = []

    def stop() -> None:
        if not stopping:
            log.info("stopping on a signal")
            stopping.append(loop.create_task(_close_and_exit(neo)))

    hooked = []
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, stop)
            hooked.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):  # not the main thread, or Windows
            pass

    def unhook() -> None:
        for sig in hooked:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)

    return unhook


async def _close_and_exit(neo: NeoMCP) -> None:
    try:
        async with asyncio.timeout(10):
            await neo.aclose()
    except BaseException:  # noqa: BLE001 — exiting regardless
        pass
    finally:
        with contextlib.suppress(Exception):  # (run once only: a no-op after a whole aclose)
            neo.agent._cleanup()  # its cards closed and its temp folder removed, whatever failed
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        os._exit(0)
