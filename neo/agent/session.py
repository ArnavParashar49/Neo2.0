"""Conversation session: reflex → route → (stream chat | quick tool | agent loop).

One session per running NEO. Holds transcript history, the pending confirmation (if any),
and decides which brain handles each utterance.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from neo.agent import confirm
from neo.agent.loop import AgentResult, run_agent
from neo.agent.prompt import build as build_prompt
from neo.agent.registry import ToolContext, registry
from neo.events import NeoState, bus
from neo.providers import brain
from neo.providers.base import ImagePart, Message
from neo.reflex import Reflex
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


@dataclass
class Reply:
    text: str
    route: Literal["chat", "quick", "agent", "confirm", "stop"]
    decision: Decision | None = None
    result: AgentResult | None = None


@dataclass
class Session:
    reflex: Reflex = field(default_factory=Reflex)
    history: list[Message] = field(default_factory=list)
    memory_context: str = ""
    voice_owned: bool = False  # a Live voice session is driving the orb; don't drop to idle after turns
    _pending_tool: str = ""
    _pending_args: dict = field(default_factory=dict)
    _pending_action: str = ""

    # ---- public --------------------------------------------------------------------
    async def handle(self, text: str, *, images: list[ImagePart] | None = None) -> Reply:
        text = _WAKE_PREFIX.sub("", text.strip(), count=1).strip() or text.strip()
        if not text:
            return Reply("", "chat")
        await bus().say(text, role="user")

        if confirm.peek():
            return await self._handle_confirmation(text)

        d = await self.reflex.decide(text)
        await bus().publish("reflex", **d.__dict__)

        if d.intent == "stop":
            await self._settle()
            return Reply("Okay.", "stop", d)

        # Regex fast paths are precise; trust them when the reflex agrees or isn't available.
        if (d.intent == "quick_action" or d.source == "rules") and (q := _match_fast_path(text)):
            return await self._quick(text, d, *q)

        if d.intent == "chat" and not images and not d.needs_screen:
            return await self._chat(text, d)

        return await self._agent(text, d, images)

    # ---- routes ----------------------------------------------------------------------
    async def _chat(self, text: str, d: Decision) -> Reply:
        await bus().set_state(NeoState.THINKING)
        t0 = time.time()
        msgs = self._trimmed() + [Message.user(text)]
        out: list[str] = []
        chain = brain("fast")
        try:
            first = True
            async for chunk in chain.stream(msgs, system=self._system()):
                if first:
                    await bus().set_state(NeoState.SPEAKING)
                    first = False
                out.append(chunk)
                await bus().say("".join(out), final=False)
        except Exception:  # noqa: BLE001 — fall back to the agent loop, which has its own failover
            return await self._agent(text, d, None)
        reply = "".join(out).strip()
        self._remember(Message.user(text), Message.assistant(reply))
        await bus().say(reply, final=True)
        await bus().publish(
            "turn",
            route="chat",
            brain=getattr(chain, "last_used", ""),
            model=getattr(chain, "last_model", ""),
            ms=int((time.time() - t0) * 1000),
            tools=0,
        )
        await self._settle()
        return Reply(reply, "chat", d)

    async def _quick(self, text: str, d: Decision, tool: str, args: dict) -> Reply:
        await bus().set_state(NeoState.WORKING)
        t0 = time.time()
        await bus().tool_start(tool, args)
        out = await registry().invoke(tool, args, ToolContext(user_text=text))
        await bus().tool_end(tool, out.ok, out.text)
        self._remember(Message.user(text), Message.assistant(out.text))
        await bus().say(out.text, final=True)
        await bus().publish(
            "turn", route="quick", brain="", model="", ms=int((time.time() - t0) * 1000), tools=1
        )
        await self._settle()
        return Reply(out.text, "quick", d)

    async def _agent(self, text: str, d: Decision | None, images: list[ImagePart] | None) -> Reply:
        purpose = "vision" if (images or (d and d.needs_screen)) else "agent"
        effort = "high" if (d and d.intent == "agent_task") else "medium"
        t0 = time.time()
        chain = brain(purpose)
        res = await run_agent(
            text,
            provider=chain,
            system=self._system(await asyncio.to_thread(self._playbooks_for, text)),
            history=self._trimmed(),
            ctx=ToolContext(user_text=text),
            effort=effort,
            images=images,
            idle_on_finish=not self.voice_owned,
        )
        self.history = res.messages
        if res.stopped == "needs_confirm" and self._stage_pending(res):
            await bus().say(res.question, final=True)
            return Reply(res.question, "confirm", d, res)
        if res.stopped == "done":
            await asyncio.to_thread(self._save_playbook, text, res)
        await bus().say(res.text, final=True)
        await bus().publish(
            "turn",
            route="agent",
            brain=getattr(chain, "last_used", ""),
            model=getattr(chain, "last_model", ""),
            ms=int((time.time() - t0) * 1000),
            tools=len(res.steps),
        )
        return Reply(res.text, "agent", d, res)

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

    def _stage_pending(self, res: AgentResult) -> bool:
        """Remember exactly which staged action the question refers to."""
        p = confirm.peek()
        if not p or (res.pending_action and p.action_id != res.pending_action):
            confirm.cancel("stale")
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            return False
        self._pending_tool, self._pending_args, self._pending_action = p.tool, dict(p.params), p.action_id
        return True

    async def _handle_confirmation(self, text: str) -> Reply:
        if _NO.search(text) or not _YES.match(text):
            if _NO.search(text):
                confirm.cancel()
                self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
                await bus().say("Cancelled.", final=True)
                await bus().set_state(NeoState.IDLE)
                return Reply("Cancelled.", "confirm")
            # Not a yes/no — treat as a new request; drop the stale confirmation.
            confirm.cancel("superseded")
            self._pending_tool, self._pending_action = "", ""
            return await self.handle(text)

        p = confirm.peek()
        if not p or p.action_id != self._pending_action:
            confirm.cancel("mismatch")
            self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
            msg = "That request changed underneath me, so I didn't do anything. Ask again and I'll confirm first."
            await bus().say(msg, final=True)
            await bus().set_state(NeoState.IDLE)
            return Reply(msg, "confirm")
        tool, args = self._pending_tool, dict(self._pending_args)
        self._pending_tool, self._pending_args, self._pending_action = "", {}, ""
        args["confirm"] = True
        await bus().set_state(NeoState.WORKING)
        out = await registry().invoke(tool, args, ToolContext(user_text=text))
        # Let the agent see the outcome and finish whatever it was doing.
        follow = f"[User confirmed. {tool} result: {out.text[:800]}] Continue and finish the task, or report the result."
        res = await run_agent(
            follow, provider=brain("agent"), system=self._system(), history=self.history, effort="medium"
        )
        self.history = res.messages
        if res.stopped == "needs_confirm" and self._stage_pending(res):
            await bus().say(res.question, final=True)
            return Reply(res.question, "confirm", None, res)
        await bus().say(res.text, final=True)
        return Reply(res.text, "agent", None, res)

    # ---- helpers ---------------------------------------------------------------------
    async def _settle(self) -> None:
        if not self.voice_owned:
            await bus().set_state(NeoState.IDLE)

    def _system(self, extra: str = "") -> str:
        return build_prompt(self.memory_context, extra)

    def _remember(self, *msgs: Message) -> None:
        self.history.extend(msgs)

    def _trimmed(self) -> list[Message]:
        h = self.history[-_MAX_HISTORY:]
        # Never start a transcript with a tool-result turn.
        while h and h[0].role == "tool":
            h = h[1:]
        return h


def _match_fast_path(text: str) -> tuple[str, dict] | None:
    """Zero-LLM dispatch for tools that declare regex fast paths (volume, brightness…).

    System tools (volume, brightness, timer, clock) are tried before app tools so a broad
    pattern like open_app can never shadow a specific one."""
    tools = sorted(registry().all(), key=lambda t: 0 if t.category == "system" else 1)
    for t in tools:
        for pattern, args in t.fast_path:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            # "<name>" values are filled from the named regex group of the same name.
            filled: dict = {}
            for k, v in args.items():
                if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                    filled[k] = m.groupdict().get(v[1:-1])
                else:
                    filled[k] = v
            return t.name, filled
    return None
