"""AsyncAgent: one embedded NEO agent — its own settings, tools, brains, memory and, per
conversation, its own confirmation gate, event bus and session (neo.runtime).

Requests always reach NEO as a model's words (``Session.handle(from_user=False)``): a "yes" in
``ask()`` never answers a pending card. Only :meth:`AsyncAgent.approve` (host code, with an
:class:`Approval`) or the host's ``approver`` callback can say yes, and only to a card this agent
staged in that conversation, within its 120 s.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import importlib.util
import inspect
import re
import secrets
import shutil
import sys
import tempfile
import time
import weakref
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from neo import runtime
from neo.agent import confirm
from neo.agent.registry import ToolContext, ToolOutput, catalog, fence
from neo.agent.session import Reply, Session
from neo.events import Event as CoreEvent
from neo.events import EventBus
from neo.providers import BrainPool
from neo.providers.base import ImagePart
from neo.reflex import Reflex
from neo.runtime import UnsupportedPlatform
from neo2 import events as ev
from neo2.sdk.approval import Approval, ApprovalRequest, _approving, guard_reentry
from neo2.sdk.config import AgentConfig
from neo2.sdk.hosttools import HostTool, check_name, host_tool
from neo2.sdk.results import Result, StepRecord, ToolInfo, ToolOutcome
from neo2.sdk.tools_api import ToolTurn, export

__all__ = ["AsyncAgent"]

_MAX_APPROVALS = 8  # approver rounds in one request (a continuation may stage again)
_EXPIRED = "That approval is no longer pending (it expired, was superseded, or isn't this agent's). Nothing was done."
_STOPPED_AS = {
    "done": "done",
    "max_steps": "max_steps",
    "thrash": "thrash",
    "error": "error",
    "needs_user": "needs_user",
    "needs_confirm": "expired",  # its card was gone by the time the run asked
}


def _platform() -> str:
    return sys.platform


class _Req:
    """One request (ask, approve, deny or call_tool) while it runs: what it did, for its Result,
    and — when streamed — its events."""

    def __init__(self, agent: AsyncAgent, conversation: str, source: str, queue: asyncio.Queue | None) -> None:
        # Never the agent itself: a task the request spawned (a card's timer) carries it in its
        # context, and must not keep an agent its host let go of alive.
        self.owner = weakref.ref(agent)
        self.rt = agent._rt
        self.conversation = conversation
        self.id = "r_" + secrets.token_hex(6)
        self.queue = queue
        self.t0 = time.monotonic()
        self.steps: list[StepRecord] = []
        self.untrusted: list[str] = ["caller"] if source == "model" else []
        self.said = ""  # the answer text streamed so far
        self.turn: dict[str, Any] = {}
        self.outcome: ToolOutcome | None = None  # call_tool: the last tool call's outcome
        self._open: list[tuple[str, dict[str, Any]]] = []
        self.done = False

    def emit(self, cls: type[ev.Event], **fields: Any) -> None:
        if self.queue is not None and not self.done:
            self.queue.put_nowait(cls(request_id=self.id, conversation=self.conversation, **fields))

    def step(self, name: str, args: Mapping[str, Any], ok: bool, summary: str) -> StepRecord:
        halted = summary.lstrip().startswith(("NEEDS_CONFIRM", "NEEDS_USER"))
        t = self.rt.registry.get(name)
        untrusted = bool(t and t.untrusted)
        rec = StepRecord(name, MappingProxyType(dict(args)), ok and not halted, summary[:300])
        self.steps.append(rec)
        # read: what it returned, or what it said failing when that was fenced (a host tool's)
        read = rec.ok or (not halted and f'<untrusted source="{name}">' in summary[:300])
        if untrusted and read and name not in self.untrusted:
            self.untrusted.append(name)
        return rec

    def say(self, text: str, scope: runtime.Scope | None = None) -> None:
        """The answer so far (the bus says it whole each time): stream what's new. NEO's protocol
        lines and the question of a card this request is asking about aren't the answer."""
        t = text.strip()
        if not t or t.startswith(("NEEDS_CONFIRM", "NEEDS_USER")):
            return
        g = scope.gate if scope is not None else None
        if g is not None and g.pending is not None and t == f"{g.pending.summary} — Should I go ahead?":
            return
        if t.startswith(self.said) and len(t) > len(self.said):
            delta, self.said = t[len(self.said) :], t
            self.emit(ev.TextDelta, text=delta)

    def feed(self, e: CoreEvent, scope: runtime.Scope) -> None:
        """An event of the conversation's bus, published while this request ran."""
        d = e.data
        if e.type == "reflex":
            self.emit(
                ev.RouteDecided,
                intent=str(d.get("intent", "")),
                confidence=float(d.get("intent_confidence") or 0.0),
                source=str(d.get("source", "")),
            )
        elif e.type == "state":
            self.emit(ev.StateChanged, state=str(d.get("state", "")))
        elif e.type == "transcript" and d.get("role", "assistant") == "assistant":
            self.say(str(d.get("text", "")), scope)
        elif e.type == "tool_start":
            args = dict(d.get("args") or {})
            self._open.append((str(d.get("name", "")), args))
            self.emit(ev.ToolStarted, name=str(d.get("name", "")), args=args)
        elif e.type == "tool_end":
            name = str(d.get("name", ""))
            i = next((i for i, (n, _) in enumerate(self._open) if n == name), None)
            args = self._open.pop(i)[1] if i is not None else {}
            rec = self.step(name, args, bool(d.get("ok")), str(d.get("summary", "")))
            t = self.rt.registry.get(name)
            self.emit(ev.ToolFinished, name=name, ok=rec.ok, summary=rec.summary, untrusted=bool(t and t.untrusted))
        elif e.type == "note":
            self.emit(ev.Note, text=str(d.get("text", "")))
        elif e.type == "turn":
            self.turn = dict(d)
            self.emit(
                ev.TurnStats,
                route=str(d.get("route", "")),
                brain=str(d.get("brain", "")),
                model=str(d.get("model", "")),
                ms=int(d.get("ms") or 0),
                tools=int(d.get("tools") or 0),
            )


_req_var: ContextVar[_Req | None] = ContextVar("neo2_request", default=None)


@dataclass
class _Card:
    """A card this agent showed its host: where it lives, and what was shown."""

    conversation: str
    request: ApprovalRequest
    tool_ctx: ToolContext | None = None  # staged by call_tool: re-run with this turn's context


class AsyncAgent:
    """One NEO agent embedded in the host's process. Use it from one event loop.

    `provider`: the host's brain(s) — a list is a failover order — for agent runs, vision and
    single actions; `fast_provider` for chat-only turns and summaries (default: `provider`).
    Without either, the brains named in the config are built from its API keys.
    `approver`: "defer" (default: ask() returns needs_approval and the host calls approve() or
    deny()), "deny" (every gated action is declined at once), or a callable that must return an
    :class:`Approval` to approve (anything else declines). `host_tools`: the host's own functions
    as tools (:func:`neo2.host_tool`). `on_audit`: gets every audit record (a dict)."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        /,
        *,
        provider: Any = None,
        fast_provider: Any = None,
        approver: Callable[[ApprovalRequest], Any] | Literal["defer", "deny"] = "defer",
        host_tools: Sequence[HostTool | Callable[..., Any]] = (),
        on_audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        cfg = config if config is not None else AgentConfig()
        if not isinstance(cfg, AgentConfig):
            raise TypeError("config must be a neo2.AgentConfig")
        if not (approver in ("defer", "deny") or callable(approver)):
            raise ValueError('approver is "defer", "deny" or a callable returning neo2.Approval')
        if on_audit is not None and not callable(on_audit):
            raise TypeError("on_audit must be callable")
        brains = _providers(provider, "provider")
        fast = _providers(fast_provider, "fast_provider")
        if not brains and not fast:
            _require_keys(cfg)
        self._cfg = cfg
        self._approver = approver
        self._closed = False
        self._cards: dict[str, _Card] = {}
        self._stopping: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()
        # The agent's own request tasks (ask, stream, approve, deny, call_tool), by conversation:
        # what cancel() and aclose() stop — never a task of the host's.
        self._tasks: dict[asyncio.Task[Any], str] = {}
        self._prefixes: set[str] = set()

        runtime.set_library_mode()  # from here on, NEO code that lost its agent fails closed
        names = _tools_for(cfg)
        own_tmp = cfg.data_dir is None
        data_dir = Path(tempfile.mkdtemp(prefix="neo-agent-")) if own_tmp else Path(cfg.data_dir)  # type: ignore[arg-type]
        rt: runtime.Runtime | None = None
        try:
            settings = cfg.library_settings(data_dir)
            overrides: dict[str, list[Any]] = {}
            if brains:
                overrides.update(dict.fromkeys(("agent", "vision", "light", "offline"), brains))
            if fast or brains:
                overrides["fast"] = fast or brains
            # Checked (pins, ~/.neo, another live agent's folder or memory) before anything is made.
            rt = runtime.Runtime(
                cfg.name,
                settings,
                catalog().subset(names),
                brains=BrainPool(overrides),
                audit_sink=on_audit,
                audit_file=not own_tmp,  # a temp dir's log would vanish with it: on_audit and the logger
                temporary=own_tmp,
            )
            settings.ensure_dirs()
            if settings.memory_path is not None:
                from neo.config import _mkdirs_private

                _mkdirs_private(Path(settings.memory_path).parent)
        except BaseException:
            if rt is not None:
                rt.close()
            if own_tmp:
                shutil.rmtree(data_dir, ignore_errors=True)
            raise
        self._rt = rt
        self._data_dir = data_dir
        self._own_tmp = own_tmp
        # Dropped without aclose(): its cards are closed (audited), its folder given back and its
        # temp dir removed all the same. Only a weak reference to the runtime: a host tool that
        # closes over the agent must not keep both alive forever.
        self._cleanup = weakref.finalize(self, _finalize, weakref.ref(rt), data_dir if own_tmp else None)
        try:
            for t in host_tools:
                self.add_tool(t)
        except BaseException:  # a bad host tool: this agent never existed
            self._closed = True
            self._rt.close()
            self._cleanup()
            raise

    # ---- properties ----------------------------------------------------------------------------
    @property
    def config(self) -> AgentConfig:
        return self._cfg

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def data_dir(self) -> Path:
        """The agent's folder (its temp dir when the config gave none)."""
        return self._data_dir

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        return f"<AsyncAgent {self._cfg.name!r} tools={len(self._rt.registry.all())}{' closed' if self._closed else ''}>"

    # ---- requests ------------------------------------------------------------------------------
    async def ask(
        self,
        text: str,
        *,
        images: Sequence[bytes | tuple[bytes, str]] = (),
        context: str = "",
        tool_context: Mapping[str, Any] | None = None,
        conversation: str = "default",
        source: Literal["user", "model"] = "user",
    ) -> Result:
        """Handle one request. `context`: the conversation it came from, quoted to the model as
        context, never as instructions (it can't close its quote). It can carry text nobody vetted
        (what a model said, a page it read), so a request with context counts as having read
        something untrusted ("context"): memory writes and data-carrying links ask first.
        `tool_context`: handed to every tool this request calls (ToolContext.extras).
        `conversation`: a thread with its own history and its own confirmation gate. `source`:
        "model" when another model wrote `text` — the request starts as having read something
        untrusted ("caller"). Neither ever grants anything."""
        guard_reentry()
        self._check_open()
        return await self._own(
            self._ask(
                text, images=images, context=context, tool_context=tool_context, conversation=conversation, source=source
            ),
            conversation,
        )

    def stream(self, text: str, **ask_kwargs: Any) -> AsyncIterator[ev.Event]:
        """ask(), as the events of that request while it runs; the last one is Completed."""
        guard_reentry()
        self._check_open()
        return self._stream(text, ask_kwargs)

    async def _stream(self, text: str, kw: dict[str, Any]) -> AsyncIterator[ev.Event]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        end = object()
        task = self._start(self._ask(text, queue=queue, **kw), kw.get("conversation", "default"))
        task.add_done_callback(lambda _t: queue.put_nowait(end))
        try:
            while True:
                item = await queue.get()
                if item is end:
                    task.result()  # the request's own error, if it failed
                    return
                yield item
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _ask(
        self,
        text: str,
        *,
        images: Sequence[bytes | tuple[bytes, str]] = (),
        context: str = "",
        tool_context: Mapping[str, Any] | None = None,
        conversation: str = "default",
        source: Literal["user", "model"] = "user",
        queue: asyncio.Queue | None = None,
    ) -> Result:
        guard_reentry()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if source not in ("user", "model"):
            raise ValueError('source is "user" or "model"')
        if tool_context is not None and not isinstance(tool_context, Mapping):
            raise TypeError("tool_context must be a mapping")
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        parts = [_image(i) for i in images]
        sc = self._scope(conversation)
        req = _Req(self, conversation, source, queue)
        if context.strip():
            req.untrusted.append("context")
        token = _req_var.set(req)
        try:
            with runtime.bind(sc):
                req.emit(ev.Started)
                try:
                    session = sc.session
                    assert session is not None
                    if self._rt.settings.memory_enabled:
                        session.memory_context = await asyncio.to_thread(_memory_context, text)
                    with confirm.collecting() as sink:
                        reply = await session.handle(
                            text,
                            images=parts or None,
                            conversation=_quoted(context) if context.strip() else "",
                            from_user=False,  # never the user's own words: can't answer a card
                            origin="user" if source == "user" else "live",
                            taint=tuple(req.untrusted),
                            extras=dict(tool_context or {}),
                        )
                        result = self._from_reply(req, sc, reply, sink)
                    result = await self._settle(req, result)
                except asyncio.CancelledError:
                    if not self._stopped_by_us():
                        raise
                    result = self._result(req, "stopped", "Stopped.", "stop")
                return self._complete(req, result)
        finally:
            req.done = True
            _req_var.reset(token)

    def _complete(self, req: _Req, result: Result) -> Result:
        if result.text.startswith(req.said) and len(result.text) > len(req.said):
            req.emit(ev.TextDelta, text=result.text[len(req.said) :])
            req.said = result.text
        req.emit(ev.Completed, result=result)
        return result

    # ---- approvals -----------------------------------------------------------------------------
    async def approve(self, action_id: str, *, via: Approval) -> Result:
        """The host's human said yes to this card (`via`: who, through which channel — audited).
        Runs exactly what the card showed. A card that's unknown, another agent's, superseded or
        past its 120 s gives status "expired" and runs nothing."""
        if not isinstance(via, Approval):
            raise TypeError("approve() needs via=neo2.Approval(channel, approver): who approved, and how")
        return await self._answer(action_id, yes=True, via=via.audit())

    async def deny(self, action_id: str, *, via: Approval | None = None) -> Result:
        """The host's human said no (or nobody answered): the card is closed, nothing runs."""
        if via is not None and not isinstance(via, Approval):
            raise TypeError("via must be a neo2.Approval (or None)")
        return await self._answer(action_id, yes=False, via=via.audit() if via else None)

    async def _answer(self, action_id: str, *, yes: bool, via: dict[str, Any] | None) -> Result:
        self._check_open()
        card = self._cards.get(action_id) if isinstance(action_id, str) else None
        return await self._own(self._answering(card, action_id, yes=yes, via=via), card.conversation if card else "")

    async def _answering(self, card: _Card | None, action_id: str, *, yes: bool, via: dict[str, Any] | None) -> Result:
        req = _Req(self, card.conversation if card else "", "user", None)
        if card is not None:  # the request it continues had read these: so has its result
            req.untrusted.extend(dict.fromkeys(card.request.after_reading))
        token = _req_var.set(req)
        try:
            try:
                result = await self._resolve(req, action_id, yes=yes, via=via)
                if yes:
                    result = await self._settle(req, result)
            except asyncio.CancelledError:
                if not self._stopped_by_us():
                    raise
                result = self._result(req, "stopped", "Stopped.", "stop")
            return result
        finally:
            req.done = True
            _req_var.reset(token)

    def pending(self, conversation: str = "default") -> ApprovalRequest | None:
        """The card waiting in this conversation, if any."""
        sc = self._rt.scopes.get(conversation)
        if sc is None or sc.gate is None:
            return None
        with runtime.bind(sc):
            p = confirm.peek()
        if p is None:
            return None
        card = self._cards.get(p.action_id)
        if card is not None:
            return card.request
        sess = sc.session
        after = tuple(sess._pending_taint) if sess is not None and sess._pending_action == p.action_id else ()
        req = self._request_for(p, sc, request_id="", after=after)
        self._remember_card(p.action_id, _Card(conversation, req))
        return req

    async def cancel(self, conversation: str | None = None) -> int:
        """Stop the running requests (of one conversation, or all) — asks, streams, approvals and
        the runs they continue, call_tool calls, approver waits — and close their cards. Returns
        how many were stopped; each ends with status "stopped". (A host function already running
        on a thread finishes there; its result is dropped.)"""
        n = self._stop(conversation)
        if conversation is None:
            scopes = list(self._rt.scopes.values())
        else:
            scopes = [s] if (s := self._rt.scopes.get(conversation)) is not None else []
        for sc in scopes:
            sess = sc.session
            if sess is None:
                continue
            with runtime.bind(sc):
                await sess.cancel_all()  # its agent runs are among ours; this closes the card
            self._forget_cards(sc.conversation)
        return n

    def _stop(self, conversation: str | None) -> int:
        """Cancel this agent's own request tasks (of one conversation, or all): each ends as
        "stopped". Never the task calling this."""
        me = asyncio.current_task()
        n = 0
        for t, conv in list(self._tasks.items()):
            if t is not me and not t.done() and (conversation is None or conv == conversation):
                self._stopping.add(t)
                t.cancel()
                n += 1
        return n

    def _stopped_by_us(self) -> bool:
        """In a request task's CancelledError: was it our cancel() or aclose()? Then the request
        ends as stopped (the cancellation is consumed); otherwise it's the host's, and propagates."""
        task = asyncio.current_task()
        if task is None or task not in self._stopping:
            return False
        self._stopping.discard(task)
        task.uncancel()
        return True

    def _start(self, coro: Coroutine[Any, Any, Any], conversation: str) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks[task] = conversation if isinstance(conversation, str) else ""
        task.add_done_callback(self._tasks_done)
        return task

    def _tasks_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.pop(task, None)

    async def _own(self, coro: Coroutine[Any, Any, Any], conversation: str) -> Any:
        """Run one request as a task of the agent's own, awaited by the host's: cancel() and
        aclose() stop it (it returns "stopped"); the host cancelling its task cancels it too."""
        return await self._start(coro, conversation)

    async def _settle(self, req: _Req, result: Result) -> Result:
        """Hand a needs_approval to the approver (unless deferred), as often as the continuation
        asks again — at most 8 rounds."""
        rounds = 0
        while result.status == "needs_approval" and result.approval is not None and self._approver != "defer":
            aid = result.approval.action_id
            if self._approver == "deny":
                return await self._resolve(req, aid, yes=False, via={"channel": "none"})
            if rounds >= _MAX_APPROVALS:
                return await self._resolve(req, aid, yes=False, via={"channel": "none", "reason": "approval limit"})
            rounds += 1
            decision, why = await self._ask_approver(result.approval)
            if isinstance(decision, Approval):
                result = await self._resolve(req, aid, yes=True, via=decision.audit())
            else:
                result = await self._resolve(req, aid, yes=False, via={"channel": "approver", "outcome": why})
        return result

    async def _ask_approver(self, request: ApprovalRequest) -> tuple[Approval | None, str]:
        """The host's approver, bounded by approval_timeout_s. Only an Approval instance is a yes."""
        fn = self._approver
        assert callable(fn)
        token = _approving.set(True)  # ask()/stream() inside it (on any agent) raise

        async def call() -> Any:
            if inspect.iscoroutinefunction(fn):
                r = await fn(request)
            else:
                r = await asyncio.to_thread(fn, request)
            if inspect.isawaitable(r):
                r = await r
            return r

        try:
            left = request.expires_at - time.time() - 0.5
            r = await asyncio.wait_for(call(), timeout=max(0.01, min(self._cfg.approval_timeout_s, left)))
        except TimeoutError:
            return None, "timeout"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — an approver that fails declines
            return None, "error"
        finally:
            _approving.reset(token)
        if isinstance(r, Approval):
            return r, "approved"
        return None, "declined"

    async def _resolve(self, req: _Req, aid: str, *, yes: bool, via: dict[str, Any] | None) -> Result:
        """Answer card `aid` (yes or no) in the conversation that staged it."""
        card = self._cards.get(aid) if isinstance(aid, str) else None
        sc = self._rt.scopes.get(card.conversation) if card is not None else None
        if card is None or sc is None or sc.session is None:
            req.emit(ev.ApprovalResolved, action_id=str(aid), outcome="expired")
            return self._result(req, "expired", _EXPIRED, "confirm")
        with runtime.bind(sc), confirm.collecting() as sink:
            p = confirm.peek()
            if p is None or p.action_id != aid:
                self._cards.pop(aid, None)
                req.emit(ev.ApprovalResolved, action_id=aid, outcome="expired")
                return self._result(req, "expired", _EXPIRED, "confirm", approval=card.request)
            self._cards.pop(aid, None)
            req.emit(ev.ApprovalResolved, action_id=aid, outcome="approved" if yes else "denied")
            if card.tool_ctx is not None:
                return await self._resolve_tool(req, sc, card, p, yes=yes, via=via, sink=sink)
            reply = await sc.session.answer(yes, action_id=aid, via=via)
            if reply is None:
                return self._result(req, "expired", _EXPIRED, "confirm", approval=card.request)
            if not yes:
                return self._result(req, "denied", reply.text or "Cancelled.", "confirm", approval=card.request)
            ran = [s for s in req.steps if s.tool == card.request.tool]
            return self._from_reply(req, sc, reply, sink, approved_ok=ran[-1].ok if ran else None)

    async def _resolve_tool(
        self,
        req: _Req,
        sc: runtime.Scope,
        card: _Card,
        p: confirm.Pending,
        *,
        yes: bool,
        via: dict[str, Any] | None,
        sink: list[str],
    ) -> Result:
        """A card call_tool staged: re-run that tool with the staged parameters, inside the
        approval and with its turn's context (the same taint and extras)."""
        t = self._rt.registry.get(p.tool)
        untrusted = bool(t and t.untrusted)
        if not yes:
            confirm.cancel(via=via)
            text = f"The user's OK is required and was not given: {p.summary}. Nothing was done."
            req.outcome = ToolOutcome("denied", text, (), card.request, untrusted)
            return self._result(req, "denied", text, "tool", approval=card.request)
        assert card.tool_ctx is not None
        ctx = dataclasses.replace(card.tool_ctx, origin="approval")  # taint and extras stay shared
        with confirm.approved(p.action_id, via=via):
            out = await self._rt.registry.invoke(p.tool, dict(p.params), ctx)
        outcome = self._tool_outcome(req, sc, p.tool, dict(p.params), out, card.tool_ctx, sink)
        status = {"ok": "done", "error": "error", "needs_approval": "needs_approval", "needs_user": "needs_user"}
        return self._result(req, status.get(outcome.status, "error"), outcome.text, "tool", approval=outcome.approval)  # type: ignore[arg-type]

    def _from_reply(
        self, req: _Req, sc: runtime.Scope, reply: Reply, sink: list[str], *, approved_ok: bool | None = None
    ) -> Result:
        """Session's reply as a Result. needs_approval iff this request staged the card now pending."""
        p = confirm.peek()
        if p is not None and p.action_id in sink:
            route = "agent" if (reply.route == "confirm" and reply.result is not None) else reply.route
            sess = sc.session
            after = (
                tuple(dict.fromkeys([*sess._pending_taint, *req.untrusted]))
                if sess is not None and sess._pending_action == p.action_id
                else tuple(req.untrusted)
            )
            appr = self._surface(req, sc, p, after=after)
            return self._result(
                req, "needs_approval", f"{p.summary} — Should I go ahead?", route, approval=appr,
                brain=reply.brain, model=reply.model,
            )  # fmt: skip
        text, res = reply.text, reply.result
        if reply.route == "stop":
            status = "stopped"
        elif text.lstrip().startswith("NEEDS_USER") or (res is not None and res.stopped == "needs_user"):
            status, text = "needs_user", _unprefixed(text, "NEEDS_USER")
        elif res is not None:
            status = _STOPPED_AS.get(res.stopped, "error")
        elif text.lstrip().startswith("NEEDS_CONFIRM"):
            status = "expired"  # the card it asked about was gone
        elif approved_ok is False:
            status = "error"
        else:
            status = "done"
        return self._result(req, status, text, reply.route, brain=reply.brain, model=reply.model)  # type: ignore[arg-type]

    def _surface(self, req: _Req, sc: runtime.Scope, p: confirm.Pending, *, after: tuple[str, ...], tool_ctx: ToolContext | None = None) -> ApprovalRequest:
        """A card this request staged, as the host will see it (and remembered, for approve())."""
        appr = self._request_for(p, sc, request_id=req.id, after=after)
        self._remember_card(p.action_id, _Card(sc.conversation, appr, tool_ctx))
        req.emit(ev.ApprovalRequired, request=appr)
        return appr

    def _request_for(self, p: confirm.Pending, sc: runtime.Scope, *, request_id: str, after: tuple[str, ...]) -> ApprovalRequest:
        ttl = sc.gate.ttl() if sc.gate is not None else confirm._TTL_S
        return ApprovalRequest(
            action_id=p.action_id,
            tool=p.tool,
            summary=p.summary,
            detail=p.detail,
            params=_frozen(p.params),
            after_reading=after,
            conversation=sc.conversation,
            request_id=request_id,
            staged_at=p.staged_at,
            expires_at=p.staged_at + ttl,
        )

    def _remember_card(self, aid: str, card: _Card) -> None:
        # One card at a time per conversation: an older one there was superseded.
        self._forget_cards(card.conversation)
        self._cards[aid] = card

    def _forget_cards(self, conversation: str) -> None:
        for k in [k for k, c in self._cards.items() if c.conversation == conversation]:
            del self._cards[k]

    def _result(
        self,
        req: _Req,
        status: str,
        text: str,
        route: str,
        *,
        approval: ApprovalRequest | None = None,
        brain: str = "",
        model: str = "",
    ) -> Result:
        return Result(
            status=status,  # type: ignore[arg-type]
            text=text,
            route=route,  # type: ignore[arg-type]
            approval=approval,
            steps=tuple(req.steps),
            brain=brain or str(req.turn.get("brain", "")),
            model=model or str(req.turn.get("model", "")),
            elapsed_ms=int((time.monotonic() - req.t0) * 1000),
            request_id=req.id,
            conversation=req.conversation,
            read_untrusted=tuple(req.untrusted),
        )

    # ---- host tools ----------------------------------------------------------------------------
    def tool(self, fn: Callable[..., Any] | None = None, /, **host_tool_kwargs: Any) -> Any:
        """Decorator: the function becomes one of this agent's tools (see neo2.host_tool)."""

        def deco(f: Callable[..., Any]) -> HostTool:
            t = host_tool(f, **host_tool_kwargs)
            self.add_tool(t)
            return t

        return deco(fn) if fn is not None else deco

    def add_tool(self, t: HostTool | Callable[..., Any]) -> None:
        """Give this agent a host tool. A name another tool of this agent (or one of NEO's own)
        already has raises ValueError."""
        self._check_open()
        ht = t if isinstance(t, HostTool) else host_tool(t)
        check_name(ht.name)
        if ht.name not in self._rt.registry:  # (a duplicate: the registry's own error below)
            self._check_exported_names(self._prefixes, new=ht.name)
        self._rt.registry.register(ht.registered())  # strict: a duplicate raises

    def remove_tool(self, name: str) -> None:
        self._check_open()
        if self._rt.registry.remove(name) is None:
            raise KeyError(f"this agent has no tool named {name!r}")

    @property
    def tools(self) -> list[ToolInfo]:
        """This agent's tools (NEO's and the host's), as the model sees them."""
        out: list[ToolInfo] = []
        for t in self._rt.registry.all():
            if t.hidden:
                continue
            gated = t.risk == "destructive"
            if t.origin == "host":
                handler = getattr(t.handler, "__self__", None)
                gated = bool(getattr(handler, "gated", gated))
            out.append(
                ToolInfo(
                    name=t.name,
                    description=t.description,
                    parameters=MappingProxyType(copy.deepcopy(t.parameters)),
                    risk=t.risk,
                    read_only=bool(t.read_only if t.read_only is not None else t.parallel_safe),
                    untrusted=t.untrusted,
                    gated=gated,
                    requires=tuple(sorted(t.requires)),
                    origin=t.origin,
                )
            )
        return out

    # ---- NEO's tools in the host's own model loop ---------------------------------------------
    def export_tools(self, fmt: str, *, prefix: str = "") -> list[dict[str, Any]]:
        """This agent's tools as `fmt` declares them: openai_chat, openai_responses, anthropic,
        gemini, mcp or jsonschema. Run the calls with call_tool() inside a turn()."""
        specs = export(self._rt.registry.all(), fmt, prefix)
        if prefix and prefix not in self._prefixes:
            self._check_exported_names({*self._prefixes, prefix})
            self._prefixes.add(prefix)
        return specs

    def _check_exported_names(self, prefixes: set[str], new: str | None = None) -> None:
        """Every name call_tool() takes — a tool's own, or an exported prefix + it — must mean one
        tool: never another tool's name, nor the same name reached two ways."""
        names = [t.name for t in self._rt.registry.all()] + ([new] if new else [])
        meant: dict[str, str] = {}
        for p in ("", *sorted(prefixes)):
            for n in names:
                other = meant.setdefault(p + n, n)
                if other != n:
                    raise ValueError(
                        f"{p + n!r} would name two tools ({other!r} and {n!r}): pick another prefix or tool name"
                    )

    @contextlib.asynccontextmanager
    async def turn(
        self,
        *,
        tool_context: Mapping[str, Any] | None = None,
        conversation: str = "default",
        source: Literal["user", "model"] = "model",
    ) -> AsyncIterator[ToolTurn]:
        """One request of the host's own model loop: its call_tool() calls share one context
        (what one read counts for the next). `source`="model" (default): the task came from a
        model, so the turn starts as having read something untrusted."""
        self._check_open()
        yield_turn = self._new_turn(tool_context, conversation, source)
        try:
            yield yield_turn
        finally:
            yield_turn.closed = True

    def _new_turn(self, tool_context: Mapping[str, Any] | None, conversation: str, source: str) -> ToolTurn:
        if source not in ("user", "model"):
            raise ValueError('source is "user" or "model"')
        if tool_context is not None and not isinstance(tool_context, Mapping):
            raise TypeError("tool_context must be a mapping")
        _check_conversation(conversation)
        ctx = ToolContext(
            user_text="",
            extras=dict(tool_context or {}),
            taint=["caller"] if source == "model" else [],
            origin="agent",
        )
        return ToolTurn(conversation, source, ctx, weakref.ref(self))  # type: ignore[arg-type]

    async def call_tool(self, name: str, args: Mapping[str, Any] | None = None, *, turn: ToolTurn | None = None) -> ToolOutcome:
        """Run one of this agent's tools, as the host's model asked. A gated action comes back as
        needs_approval (the host asks its human, then approve()), unless an approver decides.
        Untrusted output comes back fenced (<untrusted source=…>)."""
        self._check_open()
        if turn is None:
            async with self.turn() as t:
                return await self.call_tool(name, args, turn=t)
        if not isinstance(turn, ToolTurn) or not turn.belongs_to(self):
            raise ValueError("that turn isn't this agent's: start one with agent.turn()")
        if turn.closed:
            raise RuntimeError("that turn is over: start a new one with agent.turn()")
        if args is not None and not isinstance(args, Mapping):
            raise TypeError("args must be a mapping")
        real = self._tool_name(str(name))
        t = self._rt.registry.get(real) if real else None
        if t is None or t.hidden:
            return ToolOutcome("error", f"Unknown tool: {name}", (), None, False)
        return await self._own(self._calling(real, dict(args or {}), turn, bool(t.untrusted)), turn.conversation)

    async def _calling(self, real: str, args: dict[str, Any], turn: ToolTurn, untrusted: bool) -> ToolOutcome:
        sc = self._scope(turn.conversation)
        req = _Req(self, turn.conversation, turn.source, None)
        token = _req_var.set(req)
        try:
            with runtime.bind(sc):
                with confirm.collecting() as sink:
                    out = await self._rt.registry.invoke(real, args, turn.context)
                    outcome = self._tool_outcome(req, sc, real, args, out, turn.context, sink)
                if outcome.status == "needs_approval" and self._approver != "defer":
                    res = await self._settle(
                        req, self._result(req, "needs_approval", outcome.text, "tool", approval=outcome.approval)
                    )
                    if res.status == "expired":  # the card went before it was answered: nothing ran
                        outcome = ToolOutcome("error", res.text, (), None, outcome.untrusted)
                    else:
                        outcome = req.outcome or outcome
            return outcome
        except asyncio.CancelledError:
            if not self._stopped_by_us():
                raise
            return ToolOutcome("stopped", "Stopped.", (), None, untrusted)
        finally:
            req.done = True
            _req_var.reset(token)

    def _tool_outcome(
        self,
        req: _Req,
        sc: runtime.Scope,
        name: str,
        args: dict[str, Any],
        out: ToolOutput,
        ctx: ToolContext,
        sink: list[str],
    ) -> ToolOutcome:
        t = self._rt.registry.get(name)
        untrusted = bool(t and t.untrusted)
        req.step(name, args, out.ok, out.text)
        text = out.text
        p = confirm.peek()
        if out.halt and text.startswith(confirm.NEEDS_CONFIRM):
            if p is not None and (p.action_id == out.staged or p.action_id in sink):
                appr = self._surface(req, sc, p, after=tuple(ctx.taint), tool_ctx=ctx)
                req.outcome = ToolOutcome("needs_approval", text, (), appr, untrusted)
            else:
                req.outcome = ToolOutcome("error", text, (), None, untrusted)
        elif out.halt:
            req.outcome = ToolOutcome("needs_user", _unprefixed(text, "NEEDS_USER"), (), None, untrusted)
        elif not out.ok:
            req.outcome = ToolOutcome("error", text, tuple(out.images), None, untrusted)
        else:
            from neo.agent.loop import _truncate

            body = _truncate(text, self._rt.settings.max_tool_result_chars)
            req.outcome = ToolOutcome("ok", fence(name, body) if untrusted else body, tuple(out.images), None, untrusted)
        return req.outcome

    def _tool_name(self, name: str) -> str:
        reg = self._rt.registry
        if name in reg:
            return name
        for p in sorted(self._prefixes, key=len, reverse=True):
            if name.startswith(p) and name[len(p) :] in reg:
                return name[len(p) :]
        return name

    # ---- conversations and data ----------------------------------------------------------------
    def history(self, conversation: str = "default") -> list[dict[str, Any]]:
        """The conversation's messages (text, tool calls and results; no images)."""
        sc = self._rt.scopes.get(conversation)
        if sc is None or sc.session is None:
            return []
        out: list[dict[str, Any]] = []
        for m in sc.session.history:
            d: dict[str, Any] = {"role": m.role, "text": m.text}
            if m.tool_calls:
                d["tool_calls"] = [{"id": c.id, "name": c.name, "args": copy.deepcopy(c.args)} for c in m.tool_calls]
            if m.tool_results:
                d["tool_results"] = [
                    {"call_id": r.call_id, "name": r.name, "content": r.content, "ok": r.ok} for r in m.tool_results
                ]
            out.append(d)
        return out

    def reset(self, conversation: str = "default") -> None:
        """Start the conversation over: its history, its card (closed) and its session go."""
        sc = self._rt.scopes.get(conversation)
        if sc is None:
            return
        sess = sc.session
        if any(c == conversation and not t.done() for t, c in self._tasks.items()) or (
            sess is not None and any(not t.done() for t in sess.jobs.values())
        ):
            raise RuntimeError("a request is still running in this conversation: cancel() it first")
        if sc.gate is not None and sc.gate.pending is not None:
            with runtime.bind(sc):
                confirm.cancel("reset")
        if sess is not None:
            for t in (*sess._expiries, *sess._bg):
                t.cancel()
        self._forget_cards(conversation)
        with self._rt.lock:
            self._rt.scopes.pop(conversation, None)

    async def forget_all(self) -> dict[str, int]:
        """Erase what this agent kept: its memory, audit log, browser profile (cookies, logins) and
        conversations. Only NEO's own files in its folder go (never anything else there). Host code
        only: never a tool. Other agents, and NEO.app, are untouched."""
        self._check_open()
        rt = self._rt
        rt.generation += 1  # a request already running doesn't write its conversation back
        counts: dict[str, int] = {}
        with runtime.bind(runtime.Scope(rt, "")):
            if rt.browser is not None:
                with contextlib.suppress(Exception):
                    await rt.browser.close()
                rt.browser = None
            if rt.store is None and rt.settings.memory_enabled and Path(rt.settings.memory_db).exists():
                from neo.memory.store import store

                with contextlib.suppress(Exception):
                    store()  # a database from before: its rows go even when its file must stay
            if rt.store is not None:
                with contextlib.suppress(Exception):
                    counts.update(rt.store.forget_all())
                with contextlib.suppress(Exception):
                    rt.store.close()
                rt.store = None
        n_conv = 0
        for sc in list(rt.scopes.values()):
            if sc.session is not None and sc.session.history:
                n_conv += 1
                sc.session.history.clear()
                sc.session._summarized = 0
                sc.session.memory_context = ""
        counts["conversations"] = n_conv
        counts["files"] = _erase_files(rt.settings)
        return counts

    async def aclose(self) -> None:
        """Stop every request of this agent still running (each ends as "stopped" for whoever
        awaits it: never a task of the host's), close the cards (audited), the browser and the
        memory database; remove the temp folder when the config gave no data_dir. Final: nothing
        of this agent runs or reopens afterwards. Idempotent."""
        if self._closed:
            return
        self._closed = True
        rt = self._rt
        rt.closing = True  # from now on nothing reopens its memory or browser
        me = asyncio.current_task()
        mine = [t for t in self._tasks if t is not me and not t.done()]
        self._stop(None)
        timers: list[asyncio.Task[Any]] = []
        for sc in list(rt.scopes.values()):
            sess = sc.session
            if sess is not None:  # a card's timer, a summary: the session's own, never the host's
                timers += [t for t in (*sess._expiries, *sess._bg) if t is not me and not t.done()]
            if sc.gate is not None and sc.gate.pending is not None:
                with runtime.bind(sc):
                    confirm.cancel("closed")
        for t in timers:
            t.cancel()
        if mine or timers:
            await asyncio.gather(*mine, *timers, return_exceptions=True)
        self._cards.clear()
        with runtime.bind(runtime.Scope(rt, "")):
            if rt.browser is not None:
                with contextlib.suppress(Exception):
                    await rt.browser.close()
                rt.browser = None
            if rt.store is not None:
                with contextlib.suppress(Exception):
                    rt.store.close()
                rt.store = None
        rt.close()
        self._cleanup()

    async def __aenter__(self) -> AsyncAgent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---- helpers -------------------------------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this agent is closed")

    def _scope(self, conversation: str) -> runtime.Scope:
        """The conversation's scope, with its session and its bus feeding this agent's requests."""
        _check_conversation(conversation)
        rt = self._rt
        sc = rt.scope(conversation)
        if sc.session is None:
            with rt.lock:
                if sc.session is None:
                    bus = sc.get_or_create("bus", EventBus)
                    me = weakref.ref(self)  # the bus lives in the runtime: it never keeps the agent alive
                    bus.subscribe(lambda e, sc=sc, me=me: _deliver(me, e, sc))
                    sc.session = Session(reflex=Reflex(), max_agent_jobs=self._cfg.max_concurrent_tasks)
        return sc

    def _on_event(self, e: CoreEvent, sc: runtime.Scope) -> None:
        req = _req_var.get()
        if req is not None and req.owner() is self and not req.done:
            req.feed(e, sc)


def _deliver(agent: weakref.ReferenceType[AsyncAgent], e: CoreEvent, sc: runtime.Scope) -> None:
    a = agent()
    if a is not None:
        a._on_event(e, sc)


# ---- construction helpers -----------------------------------------------------------------------


def _providers(p: Any, what: str) -> list[Any]:
    if p is None:
        return []
    items = list(p) if isinstance(p, (list, tuple)) else [p]
    for x in items:
        if not (callable(getattr(x, "complete", None)) and callable(getattr(x, "stream", None))):
            raise TypeError(f"{what}: {x!r} isn't a provider (it needs complete() and stream(); see neo2.providers)")
    return items


def _usable(cfg: AgentConfig, name: str) -> bool:
    if name in ("gemini", "gemini_lite"):
        return bool(cfg.gemini_api_keys)
    if name == "groq":
        return cfg.groq_api_key is not None
    if name == "claude":
        return cfg.anthropic_api_key is not None
    return name == "local"


def _require_keys(cfg: AgentConfig) -> None:
    if not (_usable(cfg, cfg.brain) or _usable(cfg, cfg.fast_brain)):
        raise ValueError(
            "no brain: pass provider= (e.g. neo2.providers.OpenAICompatible(...)), or the API key "
            f"of AgentConfig.brain ({cfg.brain}) in the config"
        )


def _tools_for(cfg: AgentConfig) -> list[str]:
    """The catalog tools this agent gets: its selection (the memory tool only with memory on),
    their modules loaded, and only what can run here."""
    from neo.tools import load, resolve

    plat = _platform()
    items = [cfg.tools] if isinstance(cfg.tools, str) else list(cfg.tools)
    names = resolve(items, cfg.exclude_tools, platform=plat)
    explicit = set(resolve([i for i in items if str(i).strip() != "all"], cfg.exclude_tools, platform=plat))
    if cfg.memory_config is None:
        if "memory" in [str(i).strip() for i in items]:
            raise ValueError("the memory tool needs memory on: AgentConfig(memory=True or MemoryConfig(...))")
        names = [n for n in names if n != "memory"]
    elif "memory" not in names and resolve("memory", cfg.exclude_tools, platform=plat):
        names.append("memory")  # memory on: its tool comes with it (unless excluded)
    errors = load(names, platform=plat)
    cat = catalog()
    missing = [n for n in names if cat.get(n) is None]
    if missing:
        why = "; ".join(f"{m}: {e}" for m, e in errors.items()) or "not registered"
        raise ImportError(f"these tools couldn't be loaded: {', '.join(missing)} ({why})")
    has_playwright = importlib.util.find_spec("playwright") is not None
    keep: list[str] = []
    for n in names:
        req = cat.get(n).requires  # type: ignore[union-attr]
        if "macos" in req and plat != "darwin":
            if n in explicit:
                raise UnsupportedPlatform(f"{n} needs macOS; this is {plat}")
            continue
        if "playwright" in req and not has_playwright:
            if n in explicit:
                raise UnsupportedPlatform(
                    f"{n} needs NEO's browser: pip install playwright && playwright install chromium"
                )
            continue
        keep.append(n)
    return keep


def _check_conversation(conversation: str) -> None:
    if not isinstance(conversation, str) or not conversation or len(conversation) > 200:
        raise ValueError("conversation is a non-empty string of at most 200 characters")


def _image(i: bytes | tuple[bytes, str]) -> ImagePart:
    if isinstance(i, (bytes, bytearray)):
        return ImagePart(bytes(i))
    if isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], (bytes, bytearray)) and isinstance(i[1], str):
        return ImagePart(bytes(i[0]), i[1])
    raise TypeError("an image is bytes (PNG) or (bytes, mime type)")


def _memory_context(text: str) -> str:
    from neo.memory.store import store

    try:
        return store().prompt_context(text)
    except Exception:  # noqa: BLE001 — memory is context, never a failure
        return ""


def _frozen(params: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        d = copy.deepcopy(dict(params))
    except Exception:  # noqa: BLE001
        d = dict(params)
    return MappingProxyType(d)


_QUOTE_TAG = re.compile(r"<\s*(/?)\s*(recent_conversation)", re.IGNORECASE)


def _quoted(context: str) -> str:
    """The host's context, never able to close (or reopen) the quote NEO puts it in
    (neo.agent.session._with_conversation), however the tag is spelled — as fence() does."""
    return _QUOTE_TAG.sub(lambda m: f"<{m.group(1)} {m.group(2)}", context)


def _unprefixed(text: str, prefix: str) -> str:
    t = text.lstrip()
    if t.startswith(prefix):
        t = t[len(prefix) :].lstrip(":").strip()
    return t


def _remove_tmp(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _finalize(rt_ref: weakref.ReferenceType[runtime.Runtime], tmp: Path | None) -> None:
    """An agent its host let go of without aclose() (or its constructor failing): close its cards
    (audited), stop their timers, give its folder back, remove its temp dir. Runs wherever the
    last reference went — maybe another thread: the timers are cancelled through their loop."""
    rt = rt_ref()
    if rt is not None and not rt.closed:
        rt.closing = True
        for sc in list(rt.scopes.values()):
            with contextlib.suppress(Exception), runtime.bind(sc):
                if sc.gate is not None and sc.gate.pending is not None:
                    confirm.cancel("closed")
            sess = sc.session
            for t in (*sess._expiries, *sess._bg) if sess is not None else ():
                with contextlib.suppress(Exception):
                    t.get_loop().call_soon_threadsafe(t.cancel)
        rt.release()  # never a lock here: the GC may run this while this thread holds one
    _remove_tmp(tmp)


# What NEO itself writes into an agent's folder (never another file there): the memory database,
# the audit log, NEO's browser profile, brain cooldowns, and what NEO.app's task kinds and learning
# would keep (an embedded agent never writes those; erased all the same if present).
_OWN_FILES = (
    "memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm", "memory.sqlite3-journal",
    "audit.log", "audit.log.1", "audit.log.tmp", "brain_cooldowns.json",
    "decide.json", "decide_shadow.log", "media_uris.json", "reflex_extra.jsonl", "reflex_learn.json",
)  # fmt: skip
_OWN_DIRS = ("browser", "decide_learned", "tmp")


def _erase_files(s: Any) -> int:
    from neo.config import AGENT_MARKER, agent_memory_db

    d = Path(s.data_dir)
    n = 0
    if (d / AGENT_MARKER).is_file():  # NEO's own folder: never erase in one it didn't make its own
        for name in _OWN_FILES:
            p = d / name
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
                n += 1
        for name in _OWN_DIRS:
            p = d / name
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
                n += 1
    mp = getattr(s, "memory_path", None)
    # Memory kept elsewhere (MemoryConfig.path): that file and its side files, only while it's the
    # database NEO made for the agent (a table the host added since keeps it: its rows are gone).
    if mp is not None and agent_memory_db(mp):
        for p in (Path(mp), *(Path(str(mp) + x) for x in ("-wal", "-shm", "-journal"))):
            if p.is_file():
                p.unlink(missing_ok=True)
                n += 1
    return n

