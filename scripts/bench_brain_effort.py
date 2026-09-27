"""Hermetic before/after numbers for the brain-effort work: thinking level per step, the vision
chain under overload, the run budget, the wait after opening an app, and picture token costs.

Nothing leaves the Mac and nothing on it is touched: every brain is a scripted fake, the clock is
fake, the tools are fakes in a private registry, NEO_NO_EFFECTS=1 and a throwaway data dir.
Model latencies are *modeled* from published time-to-first-token medians (Gemini Flash: high
9.8 s, low ~1.2 s; medium ~4 s is an estimate) — informational, not a done-gate: the thinking
default changes only after the user's live A/B (DECISIONS 47).

Run it against this tree, and against the base tree for "before":
    NEO_NO_EFFECTS=1 .venv/bin/python scripts/bench_brain_effort.py
    NEO_NO_EFFECTS=1 PYTHONPATH=<base worktree> .venv/bin/python scripts/bench_brain_effort.py
It prints one JSON object; the new code's features are detected, so one script serves both.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("NEO_NO_EFFECTS", "1")
os.environ["NEO_REFLEX"] = "off"
os.environ["NEO_VOICE"] = "off"
os.environ["NEO_DATA_DIR"] = tempfile.mkdtemp(prefix="neo-bench-")
for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)

import neo.providers as prov  # noqa: E402
from neo.agent.registry import RegisteredTool, Registry, ToolOutput  # noqa: E402
from neo.events import EventBus  # noqa: E402
from neo.providers.base import ImagePart, Overloaded, ProviderError, RateLimited, ToolCall, Turn  # noqa: E402

NEW = hasattr(prov, "EYES")
LATENCY = {"minimal": 0.9, "low": 1.2, "medium": 4.0, "high": 9.8}  # s per reply (modeled)
LITE_LATENCY = {"minimal": 1.5, "low": 2.0, "medium": 5.0, "high": 8.4}  # Flash-Lite (8.4 s: published, default)
GROQ_STEP_S = 30.0  # a ~4K-token step on 8K tokens a minute: the wait it pays back (modeled)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t


CLOCK = Clock()
_REAL_MONOTONIC = time.monotonic
_REAL_SLEEP = asyncio.sleep


async def _advance(s: float) -> None:
    CLOCK.t += s
    await _REAL_SLEEP(0)


class _Loop(asyncio.SelectorEventLoop):
    def time(self) -> float:  # asyncio's own timers keep real time; the agent loop sees the fake one
        return _REAL_MONOTONIC()


def run(coro, *, fake_waits: bool = False):
    """Run `coro` with time.monotonic on the fake clock (the agent loop's deadline) and, with
    `fake_waits`, the chain's rate-limit waits spent on it too (never in real time)."""

    async def fake_sleep(s, *a, **k):
        CLOCK.t += float(s)
        await _REAL_SLEEP(0)

    loop = _Loop()
    time.monotonic = CLOCK.now
    if fake_waits:
        asyncio.sleep = fake_sleep
    try:
        return loop.run_until_complete(coro)
    finally:
        time.monotonic = _REAL_MONOTONIC
        asyncio.sleep = _REAL_SLEEP
        loop.close()


async def _no_note(name: str) -> None:
    return None


if NEW:
    prov._patience = _no_note  # (the 20 s "still waiting" note: nothing to say in a bench)


# ---- fakes ---------------------------------------------------------------------------------------


def _png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1512, 982), (40, 90, 40)).save(buf, format="PNG")
    return buf.getvalue()


PNG = ImagePart(_png())


def registry(miss_once: bool) -> Registry:
    reg = Registry()
    state = {"clicks": 0}

    async def ax_tree(a, ctx):
        return ToolOutput("window “Game”: the board area has no accessible elements")

    async def screenshot(a, ctx):
        return ToolOutput("Screenshot s1 attached (1512×982 px).", images=[PNG], fence=False)

    async def click(a, ctx):
        if ctx.get("blind") and ("x" in a or "y" in a):  # neo.tools.computer.pointer.points' rule
            return ToolOutput("Error: you can't see the screen — this model has no vision", ok=False)
        state["clicks"] += 1
        if miss_once and state["clicks"] == 2:
            return ToolOutput("Clicked. Nothing visibly changed there within 1s — it may have missed", ok=False)
        return ToolOutput("Clicked. Changed: the screen changed around the point.")

    reg.register(RegisteredTool("ax_tree", "look", {"type": "object"}, ax_tree, parallel_safe=True))
    reg.register(RegisteredTool("screenshot", "look", {"type": "object"}, screenshot))
    reg.register(RegisteredTool("click", "click", {"type": "object"}, click))
    return reg


def seeing_script(miss_once: bool) -> list[Turn]:
    c = [
        ("ax_tree", {}),
        ("screenshot", {}),
        ("click", {"shot": "s1", "x": 600, "y": 800}),  # pick up e2
        ("screenshot", {}),
        ("click", {"shot": "s1", "x": 600, "y": 600}),  # drop on e4
        ("screenshot", {}),
    ]
    if miss_once:
        c += [("click", {"shot": "s1", "x": 602, "y": 598}), ("screenshot", {})]
    return [Turn(tool_calls=[ToolCall(f"t{i}", n, a)]) for i, (n, a) in enumerate(c)] + [Turn(text="Moved e2 to e4.")]


class Seeing:
    """A brain that sees: plays the script, taking the modeled latency of the level it's asked at."""

    supports_vision, supports_tools = True, True

    def __init__(self, name: str, turns: list[Turn], latency: dict, *, down: bool = False) -> None:
        self.name = self.model = name
        self.turns, self.latency, self.down = list(turns), latency, down
        self.levels: list[str] = []
        self.efforts = frozenset({"minimal", "low", "medium", "high"})
        self.probed = False

    def cooling_until(self) -> float:
        return 0.0

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        if self.down:
            # The logged run: 3–8 s lost on an overloaded model; then, like GeminiProvider, it's
            # remembered for 90 s and fails at once.
            await _advance(0.0 if self.probed else 4.0)
            self.probed = True
            raise Overloaded(f"{self.name}: overloaded")
        self.levels.append(str(effort))
        await _advance(self.latency.get(str(effort), 9.8))
        n = sum(1 for m in messages if m.role == "assistant")
        return self.turns[min(n, len(self.turns) - 1)]

    async def stream(self, messages, **kw):
        yield ""


class BlindGroq:
    """Groq's text model in the logged run: can't see, clicks at coordinates it made up."""

    name = model = "groq"
    supports_vision, supports_tools = False, True

    def __init__(self) -> None:
        self.asked = 0

    def cooling_until(self) -> float:
        return 0.0

    async def complete(self, messages, **kw):
        self.asked += 1
        await _advance(GROQ_STEP_S)
        return Turn(tool_calls=[ToolCall(f"g{self.asked}", "click", {"x": 612, "y": 540})], sees=False)

    async def stream(self, messages, **kw):
        yield ""


def _make(brains: dict):
    def make(n):
        if n in brains:
            return brains[n]
        raise ProviderError(f"{n} is not set up")

    return make


async def _run(chain, reg, *, effort, max_seconds=180.0, bind_budget=False):
    from neo.agent.loop import run_agent

    provider = chain
    if bind_budget:  # the wiring the integrator adds in run_agent (neo.providers.budget)
        from neo.providers import budget

        class Bound:
            def __getattr__(self, k):
                return getattr(chain, k)

            async def complete(self, *a, **kw):
                with budget.seconds(max_seconds - (CLOCK.now() - t0), clock=CLOCK.now):
                    return await chain.complete(*a, **kw)

        provider = Bound()
    t0 = CLOCK.now()
    res = await run_agent("move the pawn from e2 to e4", provider=provider, system="", reg=reg, ev=EventBus(),
                          effort=effort, tools=[t.spec() for t in reg._tools.values()], max_seconds=max_seconds)  # fmt: skip
    return res, CLOCK.now() - t0


def _chain(purpose: str, level: str | None):
    if NEW and level is not None:
        from neo.providers.effort import Effort

        return prov.brain(purpose, effort=Effort(level))
    return prov.brain(purpose)


# ---- scenarios ------------------------------------------------------------------------------------


def thinking(miss_once: bool) -> dict:
    """A move on a board with no accessible squares, Gemini healthy: modeled model time per arm."""
    out = {}
    arms = [("today (high)", None)] + ([("screen_effort=high (default)", "high"), ("screen_effort=low (A/B arm)", "low")] if NEW else [])
    for label, level in arms:
        g = Seeing("gemini", seeing_script(miss_once), LATENCY)
        prov.make = _make({"gemini": g})
        prov._demoted.clear()
        CLOCK.t = 1000.0
        res, took = run(_run(_chain("vision", level), registry(miss_once), effort="high"))
        out[label] = {"round_trips": len(g.levels), "levels": "".join(x[0] for x in g.levels),
                      "model_s": round(took, 1), "outcome": res.stopped}  # fmt: skip
    return out


def overload(all_eyes_busy: bool) -> dict:
    """The logged run: every Flash model overloaded mid-task. Flash-Lite up (or not)."""
    g = Seeing("gemini", [], LATENCY, down=True)
    lite = Seeing("gemini_lite", seeing_script(False), LITE_LATENCY, down=all_eyes_busy)
    gv = Seeing("groq_vision", seeing_script(False), {k: GROQ_STEP_S for k in LATENCY}, down=all_eyes_busy)
    groq = BlindGroq()
    prov.make = _make({"gemini": g, "gemini_lite": lite, "groq": groq, "groq_vision": gv})
    prov._demoted.clear()
    CLOCK.t = 1000.0
    res, took = run(_run(_chain("vision", "high"), registry(False), effort="high"))
    return {"blind_round_trips": groq.asked, "seeing_round_trips": len(lite.levels) + len(gv.levels),
            "wall_s": round(took, 1), "outcome": res.stopped, "said": res.text[:140]}  # fmt: skip


class Throttled(Seeing):
    """A seeing brain on a per-minute budget: every step first gets a 429 asking for `wait` s."""

    def __init__(self, wait: float) -> None:
        super().__init__("groq_vision", seeing_script(False), {k: 5.0 for k in LATENCY})
        self.wait, self.fresh = wait, True

    async def complete(self, messages, **kw):
        if self.fresh:
            self.fresh = False
            await _advance(0.3)
            raise RateLimited("429 tokens per minute", retry_after=self.wait)
        self.fresh = True
        return await super().complete(messages, **kw)


def budget_overrun(wait: float) -> dict:
    """Only a rate-limited seeing brain left: does the run end within its 180 s?"""
    out = {}
    arms = [("chain waits regardless (today)", False)] + ([("budget bound (new)", True)] if NEW else [])
    for label, bind in arms:
        t = Throttled(wait)
        prov.make = _make({"groq_vision": t})
        prov._demoted.clear()
        CLOCK.t = 1000.0
        chain = prov.Chain(["groq_vision"], sticky=False)
        res, took = run(_run(chain, registry(False), effort="high", bind_budget=bind), fake_waits=True)
        out[label] = {"wall_s": round(took, 1), "over_limit_s": round(max(0.0, took - 180.0), 1),
                      "steps_done": len(res.steps), "outcome": res.stopped, "said": res.text[:110]}  # fmt: skip
    return out


class _FakeMac:
    """An app that is in front from `up_at` s, with a text field focused from `field_at` s."""

    def __init__(self, up_at: float, field_at: float) -> None:
        self.t, self.up_at, self.field_at = 0.0, up_at, field_at

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += s

    def probe(self, app, typing):
        from neo.agent.settle import Ready

        self.t += 0.002  # a look: ~2 ms of Accessibility reads
        return Ready(self.t >= self.up_at, self.t >= self.up_at, self.t >= self.field_at)


def settle() -> dict:
    """The wait before the next step after opening an app (fixed 0.8 s before)."""
    if not NEW:
        return {s: {"waited_s": 0.8, "keys_lost": s == "cold launch 1.3 s"} for s in
                ("already running", "warm 0.15 s", "cold launch 1.3 s")}  # fmt: skip
    from neo.agent import settle as st

    out = {}
    for label, up_at, field_at in (("already running", 0.0, 0.0), ("warm 0.15 s", 0.15, 0.2),
                                   ("cold launch 1.3 s", 1.3, 1.35)):  # fmt: skip
        mac = _FakeMac(up_at, field_at)
        st.probe = mac.probe
        waited = asyncio.run(st.app_ready(typing=True, clock=mac.now, sleep=mac.sleep))
        out[label] = {"waited_s": round(waited, 3), "keys_lost": waited < field_at}
    return out


def picture_tokens() -> dict:
    """What a 1512×982 screenshot is counted as, per brain, when fitting a request."""
    from neo.providers import fit

    out = {}
    for name in ("gemini", "gemini_lite", "claude", "groq_vision"):
        if NEW:
            out[name] = {d: fit.image_tokens(ImagePart(PNG.data, detail=d), name) for d in ("auto", "medium")}
        else:
            out[name] = {"auto": fit._IMAGE_TOKENS}
    return out


def main() -> None:
    t0 = time.perf_counter()
    report = {
        "code": "new" if NEW else "base",
        "thinking_move": thinking(False),
        "thinking_move_with_a_miss": thinking(True),
        "overload_flash_lite_up": overload(False),
        "overload_every_eye_busy": overload(True),
        "budget_429_wait_50s": budget_overrun(50.0),
        "settle_after_open": settle(),
        "picture_tokens_1512x982": picture_tokens(),
    }
    if NEW:
        from neo.providers import ImageLimit, limit_of

        report["image_limit_examples"] = {
            "gemini+groq_vision": str(ImageLimit().tighter(limit_of(Seeing("groq_vision", [], {})))),
        }
    report["bench_s"] = round(time.perf_counter() - t0, 2)
    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
