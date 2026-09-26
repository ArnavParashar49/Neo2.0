"""Conversation session: reflex → route → (stream chat | quick tool | agent loop).

Requests are *jobs* and several may run at once — you can ask for the time while NEO is still
reading your mail. Each job reports its own orb state and tool activity under a job id; history
is a sequence of complete blocks appended under a lock when a job finishes, so concurrent jobs
never interleave inside each other's transcript. Long agent jobs are limited to a few at a time.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from neo.agent import confirm, fastpath
from neo.agent.loop import AgentResult, run_agent
from neo.agent.prompt import build as build_prompt
from neo.agent.registry import ToolContext, registry
from neo.config import settings
from neo.events import NeoState, bus
from neo.providers import brain
from neo.providers.base import ImagePart, Message, ToolResult
from neo.reflex import Reflex, is_bare_stop
from neo.reflex.schema import Decision

# A confirmation must be a short, pure affirmative; any retraction anywhere wins over a leading "ok".
_YES = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go ahead|do it|confirm(?:ed)?|please do|proceed|go for it|"
    r"yes please|sure thing|absolutely|correct)(?:[\s,]+(?:please|go ahead|do it|thanks))*[\s.!]*$",
    re.I,
)
_NO = re.compile(
    r"\b(?:no|nope|don'?t|do not|cancel|stop|never ?mind|leave it|not yet|wait|hold on|abort)\b", re.I
)
_MAX_HISTORY = 40  # messages kept verbatim; older ones are dropped (memory keeps the gist)
_WAKE_PREFIX = re.compile(r"^\s*(?:hey|hi|ok|okay)?[\s,]*neo[\s,:!.-]*", re.I)
_MAX_AGENT_JOBS = 2
# "what time is it in Tokyo", "what's on my calendar tomorrow": a qualifier the no-argument tool
# can't take — those go to a model that sees the tool (and can ask for more).
_QUALIFIER = re.compile(
    r"\b(?:in|at|on|for|from|since|until|by|about|with|tomorrow|yesterday|next|last|ago|week|month|"
    r"year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\d",
    re.I,
)
_job_ids = itertools.count(1)


@dataclass
class Reply:
    text: str
    route: Literal["chat", "quick", "agent", "confirm", "stop"]
    decision: Decision | None = None
    result: AgentResult | None = None
    job: str = ""
    silent: bool = False  # the task was done quietly: nothing to speak, no chat bubble


@dataclass
class Session:
    reflex: Reflex = field(default_factory=Reflex)
    history: list[Message] = field(default_factory=list)
    memory_context: str = ""
    voice_owned: bool = False  # kept for callers; per-job states made it unnecessary
    _pending_tool: str = ""
    _pending_args: dict = field(default_factory=dict)
    _pending_action: str = ""
    _hist_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _agent_slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(_MAX_AGENT_JOBS))
    jobs: dict[str, asyncio.Task] = field(default_factory=dict)

    # ---- public --------------------------------------------------------------------
    async def handle(self, text: str, *, images: list[ImagePart] | None = None) -> Reply:
        text = _WAKE_PREFIX.sub("", text.strip(), count=1).strip() or text.strip()
        if not text:
            return Reply("", "chat")
        job = f"j{next(_job_ids)}"
        await bus().say(text, role="user", job=job)

        if confirm.peek():
            return await self._handle_confirmation(text, job)
        if self._pending_action:  # the staged question expired: drop it and the "confirming" orb
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            await self._clear_confirming()

        d = await self.reflex.decide(text)
        await bus().publish("reflex", job=job, **d.__dict__)

        # A bare "stop" / "never mind" cancels everything; "stop the timer" is a command.
        if is_bare_stop(text) or (d.intent == "stop" and len(text.split()) <= 2 and d.intent_confidence >= 0.9):
            await self.cancel_all()
            await bus().end_job(job)
            return Reply("Okay.", "stop", d, job=job)

        # Regex fast paths are precise; trust them unless the reflex says this is conversation.
        # ("type Laptops in the heading" reads as an agent task to the reflex, but the words are
        # the whole job — no model needed.) A plan covers the whole utterance or nothing.
        if d.intent != "chat" and (steps := fastpath.plan(text)):
            return await self._quick(text, d, steps, job=job)

        if d.intent == "chat" and not images and not d.needs_screen:
            return await self._chat(text, d, job)

        # Laya named the tool itself: skip the agent loop. A read-only tool without arguments
        # runs straight away (unless the request carries a qualifier it can't take); anything
        # else — and every action — gets one small model call that only sees that tool.
        if d.intent == "quick_action" and d.tool and d.tool_p >= settings().reflex_tool_floor:
            if (t := registry().get(d.tool)) is not None and not t.hidden:
                print(f"  ~ laya picked {d.tool} ({d.tool_p:.2f})")
                bare = not t.parameters.get("required") and not t.parameters.get("properties")
                if bare and not t.quiet and not _QUALIFIER.search(text):
                    return await self._quick(text, d, [(d.tool, {})], job=job)
                return await self._agent(text, d, images, job, only=[d.tool])

        return await self._agent(text, d, images, job)

    async def cancel_all(self) -> int:
        """Stop every running job (the user said stop)."""
        n = 0
        for t in list(self.jobs.values()):
            if not t.done():
                t.cancel()
                n += 1
        return n

    # ---- routes ----------------------------------------------------------------------
    async def _chat(self, text: str, d: Decision, job: str) -> Reply:
        await bus().set_state(NeoState.THINKING, job=job)
        t0 = time.time()
        msgs = self._trimmed() + [Message.user(text)]
        out: list[str] = []
        chain = brain("fast")
        try:
            first = True
            async for chunk in chain.stream(msgs, system=self._system()):
                if first:
                    await bus().set_state(NeoState.SPEAKING, job=job)
                    first = False
                out.append(chunk)
                await bus().say("".join(out), final=False, job=job)
        except Exception:  # noqa: BLE001 — fall back to the agent loop, which has its own failover
            return await self._agent(text, d, None, job)
        reply = "".join(out).strip()
        await self._remember(Message.user(text), Message.assistant(reply))
        await bus().say(reply, final=True, job=job)
        await bus().publish(
            "turn",
            job=job,
            route="chat",
            brain=getattr(chain, "last_used", ""),
            model=getattr(chain, "last_model", ""),
            ms=int((time.time() - t0) * 1000),
            tools=0,
        )
        await bus().end_job(job)
        return Reply(reply, "chat", d, job=job)

    async def _quick(self, text: str, d: Decision, steps: list[tuple[str, dict]], *, job: str) -> Reply:
        """Run fast-path steps in order with no model. Actions that worked are silent; answers,
        and anything that needs the user, are spoken. An action that fails hands the whole
        request to the agent, which can look at the screen and find another way."""
        await bus().set_state(NeoState.WORKING, job=job)
        t0 = time.time()
        from neo.agent.early import _APP_TOOLS, _SETTLE_AFTER_APP_S, early

        spoken: list[str] = []
        texts: list[str] = []
        prev = ""
        for tool, args in steps:
            t = registry().get(tool)
            done = early().claim(tool, args)
            if done is not None:
                out_text, ok = done, True  # the early actor already did this while the user was speaking
            else:
                if prev in _APP_TOOLS and tool not in _APP_TOOLS:
                    await asyncio.sleep(_SETTLE_AFTER_APP_S)
                await bus().tool_start(tool, args, job=job)
                out = await registry().invoke(tool, args, ToolContext(user_text=text))
                await bus().tool_end(tool, out.ok, out.text, job=job)
                out_text, ok = out.text, out.ok
                if not ok and t is not None and t.quiet:
                    print(f"  ~ {tool} failed ({out_text[:60]}); handing to the agent")
                    return await self._agent(text, d, None, job)
            prev = tool
            texts.append(out_text)
            needs_user = out_text.startswith(("NEEDS_", "Error"))
            if not (t and t.quiet and ok) or needs_user:
                spoken.append(out_text)
        reply_text = " ".join(spoken) if spoken else "; ".join(texts)
        await self._remember(Message.user(text), Message.assistant(reply_text))
        silent = not spoken  # every step was an action that worked: just done, nothing to say
        if silent:
            await bus().note(reply_text[:120])
        else:
            await bus().say(reply_text, final=True, job=job)
        await bus().publish(
            "turn", job=job, route="quick", brain="", model="", ms=int((time.time() - t0) * 1000), tools=len(steps)
        )
        await bus().end_job(job)
        return Reply(reply_text, "quick", d, job=job, silent=silent)

    async def _agent(
        self,
        text: str,
        d: Decision | None,
        images: list[ImagePart] | None,
        job: str,
        *,
        only: list[str] | None = None,
    ) -> Reply:
        needs_screen = bool(images or (d and d.needs_screen))
        if needs_screen:
            purpose = "vision"
        elif only or (d and d.intent == "quick_action"):
            purpose = "light"  # one simple action the regex fast path didn't cover
        else:
            purpose = "agent"
        effort = "high" if (d and d.intent == "agent_task") else "medium"
        t0 = time.time()
        chain = brain(purpose)
        self.jobs[job] = asyncio.current_task()  # type: ignore[assignment]
        try:
            async with self._agent_slots:
                playbook_ctx = await asyncio.to_thread(self._playbooks_for, text)
                if only:  # the reflex chose: that tool's schema and nothing else to get lost in
                    from neo.agent.toolselect import MORE_TOOLS

                    tools = [registry().get(n).spec() for n in only if registry().get(n)] + [MORE_TOOLS]
                else:
                    tools = await asyncio.to_thread(self._tools_for, text, needs_screen, playbook_ctx)
                snapshot = self._trimmed()
                res = await run_agent(
                    text,
                    provider=chain,
                    system=self._system(playbook_ctx + _already_done_note()),
                    tools=tools,
                    history=snapshot,
                    ctx=ToolContext(user_text=text),
                    effort=effort,
                    images=images,
                    job=job,
                )
        except asyncio.CancelledError:
            await bus().say("Stopped.", final=True, job=job)
            await bus().end_job(job)
            raise
        finally:
            self.jobs.pop(job, None)
        await self._remember(*res.messages[len(snapshot) :])  # only this job's new turns
        if res.stopped == "needs_confirm" and self._stage_pending(res):
            await bus().say(res.question, final=True, job=job)
            return Reply(res.question, "confirm", d, res, job=job)
        if res.stopped == "done":
            await asyncio.to_thread(self._save_playbook, text, res)
        silent = _quietly(d, res)
        if silent:
            await bus().note(res.text[:160])
        else:
            await bus().say(res.text, final=True, job=job)
        await bus().publish(
            "turn",
            job=job,
            route="agent",
            brain=getattr(chain, "last_used", ""),
            model=getattr(chain, "last_model", ""),
            ms=int((time.time() - t0) * 1000),
            tools=len(res.steps),
        )
        await bus().end_job(job)
        return Reply(res.text, "agent", d, res, job=job, silent=silent)

    # ---- tool selection ---------------------------------------------------------------
    def _tools_for(self, text: str, needs_screen: bool, playbook_ctx: str):
        """Only the tools this request plausibly needs (see neo.agent.toolselect)."""
        from neo.agent.toolselect import selector

        recent = [c.name for m in self.history[-8:] if m.role == "assistant" for c in m.tool_calls]
        pb_tools = re.findall(r"\b([a-z_]+)\(", playbook_ctx) if playbook_ctx else []
        sel = selector().select(
            text, needs_screen=needs_screen, playbook_tools=pb_tools, history_tools=recent
        )
        return sel.specs(registry())

    # ---- playbooks -------------------------------------------------------------------
    @staticmethod
    def _playbooks_for(goal: str) -> str:
        try:
            from neo.memory.store import store

            return store().playbook_context(goal)
        except Exception as e:  # noqa: BLE001
            print(f"[playbook] recall failed: {str(e)[:80]}")
            return ""

    @staticmethod
    def _save_playbook(goal: str, res: AgentResult) -> None:
        """Keep the successful tool path of a multi-step task with at least one real action."""
        ok_steps = [s for s in res.steps if s.ok]
        if len(ok_steps) < 2:
            return
        try:
            from neo.agent.registry import registry as _reg
            from neo.memory.store import store

            if not any((t := _reg().get(s.tool)) and not t.parallel_safe for s in ok_steps):
                return
            store().save_playbook(goal, [{"tool": s.tool, "args": s.args} for s in ok_steps], res.text)
        except Exception as e:  # noqa: BLE001
            print(f"[playbook] save failed: {str(e)[:80]}")

    # ---- confirmation ------------------------------------------------------------------
    def _stage_pending(self, res: AgentResult) -> bool:
        """Remember exactly which staged action the question refers to."""
        p = confirm.peek()
        if not p or (res.pending_action and p.action_id != res.pending_action):
            confirm.cancel("stale")
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            return False
        self._pending_tool, self._pending_args, self._pending_action = p.tool, dict(p.params), p.action_id
        asyncio.create_task(self._expire_confirm(p.action_id))
        return True

    async def _handle_confirmation(self, text: str, job: str) -> Reply:
        if _NO.search(text) or not _YES.match(text):
            if _NO.search(text):
                confirm.cancel()
                self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
                await bus().say("Cancelled.", final=True, job=job)
                await bus().end_job(job)
                await bus().set_state(
                    NeoState.IDLE
                ) if not bus().active_jobs and bus().state == NeoState.CONFIRMING else None
                return Reply("Cancelled.", "confirm", job=job)
            # Not a yes/no — treat as a new request; drop the stale confirmation.
            confirm.cancel("superseded")
            self._pending_tool, self._pending_action = "", ""
            await self._clear_confirming()
            return await self.handle(text)

        p = confirm.peek()
        if not p or p.action_id != self._pending_action:
            confirm.cancel("mismatch")
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            msg = "That request changed underneath me, so I didn't do anything. Ask again and I'll confirm first."
            await bus().say(msg, final=True, job=job)
            await self._clear_confirming()
            await bus().end_job(job)
            return Reply(msg, "confirm", job=job)
        tool, args = self._pending_tool, dict(self._pending_args)
        self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
        args["confirm"] = True
        await self._clear_confirming()
        await bus().set_state(NeoState.WORKING, job=job)
        await bus().tool_start(tool, args, job=job)
        out = await registry().invoke(tool, args, ToolContext(user_text=text))
        await bus().tool_end(tool, out.ok, out.text, job=job)
        # Let the agent see the outcome and finish whatever it was doing.
        follow = f"[User confirmed. {tool} result: {out.text[:800]}] Continue and finish the task, or report the result."
        snapshot = self._trimmed()
        res = await run_agent(
            follow, provider=brain("agent"), system=self._system(), history=snapshot, effort="medium", job=job
        )
        await self._remember(*res.messages[len(snapshot) :])
        if res.stopped == "needs_confirm" and self._stage_pending(res):
            await bus().say(res.question, final=True, job=job)
            return Reply(res.question, "confirm", None, res, job=job)
        # The user just said yes to something consequential: always tell them how it went.
        await bus().say(res.text, final=True, job=job)
        await bus().end_job(job)
        return Reply(res.text, "agent", None, res, job=job)

    async def _expire_confirm(self, action_id: str) -> None:
        """If nobody answers, the question lapses and the orb stops asking."""
        await asyncio.sleep(confirm._TTL_S)
        if self._pending_action == action_id:
            confirm.cancel("expired")
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            await bus().note("No answer — I didn't do it.")
            await self._clear_confirming()

    async def _clear_confirming(self) -> None:
        """The confirming state belongs to the job that asked; that job is over now."""
        for j in list(bus().active_jobs):
            if bus()._jobs.get(j) == NeoState.CONFIRMING:
                await bus().end_job(j)

    # ---- helpers ---------------------------------------------------------------------
    def _system(self, extra: str = "") -> str:
        return build_prompt(self.memory_context, extra)

    async def _remember(self, *msgs: Message) -> None:
        async with self._hist_lock:
            self.history.extend(msgs)

    def _trimmed(self) -> list[Message]:
        """Recent history for the model. Tool results older than the last two turns are stubbed —
        they were consumed when fresh and are the bulk of the tokens otherwise."""
        h = self.history[-_MAX_HISTORY:]
        while h and h[0].role == "tool":  # never start a transcript with a tool-result turn
            h = h[1:]
        keep_from = _last_user_index(h, 2)
        out: list[Message] = []
        for i, m in enumerate(h):
            if m.role == "tool" and i < keep_from:
                stubbed = [ToolResult(r.call_id, r.name, _stub(r.content), r.ok) for r in m.tool_results]
                out.append(Message.tool(stubbed))
            else:
                out.append(m)
        return out


def _last_user_index(h: list[Message], n: int) -> int:
    """Index of the n-th most recent user message (0 if fewer)."""
    seen = 0
    for i in range(len(h) - 1, -1, -1):
        if h[i].role == "user":
            seen += 1
            if seen == n:
                return i
    return 0


def _quietly(d: Decision | None, res: AgentResult) -> bool:
    """Was this a command that simply got done? Then there's nothing to say.

    Questions, information requests, failures, budget stops and anything the model asks back
    are always spoken."""
    if d is None or d.reply or res.stopped != "done" or res.question:
        return False
    if not res.steps or not all(st.ok for st in res.steps):
        return False
    low = res.text.lower()
    return not any(w in low for w in ("?", "couldn't", "could not", "can't", "cannot", "unable", "error", "failed"))


_strip_quotes = fastpath.strip_quotes
_match_fast_path = fastpath.match


def _already_done_note() -> str:
    """What the early actor did while the user was still talking and nobody has accounted for
    yet — i.e. for *this* request — so the model doesn't redo it. Consumes those records."""
    from neo.agent.early import early

    done = [r for r in early().unclaimed() if r.ok]
    if not done:
        return ""
    lines = "; ".join(f"{r.tool}({', '.join(f'{k}={v!r}' for k, v in r.args.items())})" for r in done)
    return f"\n\nAlready done a moment ago for this request (do NOT repeat): {lines}"


def _stub(text: str, keep: int = 160) -> str:
    return (
        text
        if len(text) <= keep
        else text[:keep].rstrip() + f" …[{len(text) - keep} chars trimmed from an earlier turn]"
    )


_ = settings  # settings are read by callers; keep the import for the module's public surface
