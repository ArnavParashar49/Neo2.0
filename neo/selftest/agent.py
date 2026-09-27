"""The self-test's user-run agent section and A/B harness: real requests, through the real brain chain,
against the test window — the only real end-to-end model latencies in the controls plan.

    python -m neo.selftest agent              # 3 scripted requests, once each
    python -m neo.selftest ab --runs 10       # low vs high thinking, batching on vs off, ≥10 runs per arm

It uses the user's own model quota, so only the user starts it, from their own Terminal (the same
rules as ``python -m neo.selftest controls``). Each request runs in the controls test window with
every guard of the controls check: events only to the test window and Accessibility actions only on
its elements (neo.selftest.watch), stop the moment the user touches the mouse or keyboard, NEO's own
window never, and a question for the user (NEEDS_CONFIRM) ends that run unanswered. The model gets
only the screen tools, and ``hotkey`` refuses chords the whole Mac acts on (⌘Tab, ⌘Space, ⌘Q…).

For each run: did the goal happen (from the test window's own log), model calls and their seconds,
tool calls, total seconds, and which brain answered (a failover to another brain — an overloaded
Gemini — is reported apart, with its seconds: that, not the thinking level, dominated the Chess log).

The A/B result is the evidence the defaults change on (DECISIONS 47): it says which arm wins, and
by how much, but it changes no setting itself. An arm whose switch this build doesn't have (no
batching setting yet) is skipped and says so.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from neo.selftest import controls
from neo.selftest import geometry as G
from neo.selftest import layout as L

AGENT_TOOLS = (
    "ax_tree", "ax_find", "screenshot", "click", "drag", "scroll", "hotkey", "type_text", "ax_press",
    "ax_set_value", "click_text", "menu_command",
)  # fmt: skip
RUNS_PER_ARM = 10
RUN_S = 90.0  # one request's limit
# Batching's switch, by the names the brain workstream may give it (the first that exists is used).
BATCH_SETTINGS = ("batch_actions", "action_batching", "batching", "batch")
EFFORT_SETTINGS = ("screen_effort",)


@dataclass
class Arm:
    name: str
    effort: str = "high"
    batch: bool | None = None  # None: as the build has it


@dataclass
class Request:
    key: str
    words: Callable[[controls._Run], str]
    goal: Callable[[controls._Run, int], bool]


def _bravo(r: controls._Run, mark: int) -> bool:
    return bool(r.since(mark, "button", name=L.BUTTON_B))


def _move_words(r: controls._Run) -> str:
    piece = r.state().get("piece") or r.ready.get("piece") or list(L.PIECE)
    col, row = piece
    to = [col, row - 2] if row >= 4 else [col, row + 2]
    r.scratch_move = (list(piece), to)  # type: ignore[attr-defined]
    return f"move the piece on {G.square(col, row)} to {G.square(*to)} on the test window's flat board"


def _moved(r: controls._Run, mark: int) -> bool:
    want = getattr(r, "scratch_move", None)
    return any([e.get("start"), e.get("end")] == list(want) for e in r.since(mark, "piece_moved")) if want else False


def _round(r: controls._Run, mark: int) -> bool:
    return bool(r.since(mark, "round"))


REQUESTS = (
    Request("bravo", lambda r: "click the Bravo button in the NEO Self-Test window", _bravo),
    Request("move", _move_words, _moved),
    Request("round", lambda r: "start a new round in the NEO Self-Test window", _round),
)

ARMS = {
    "effort": (Arm("low", effort="low"), Arm("high", effort="high")),
    "batch": (Arm("batch off", batch=False), Arm("batch on", batch=True)),
}

# Chords the whole Mac acts on, whatever app is in front: never sent in the agent section.
_GLOBAL = re.compile(r"(?:cmd|command|⌘).*(?:tab|space|\bq\b|\bh\b|\bm\b|`)|ctrl.*(?:up|down|left|right|space)|"
                     r"(?:cmd|command|⌘).*(?:alt|option|opt).*esc|shift.*[345]\b|\bfn\b", re.I)  # fmt: skip


def _safe_hotkey(handler: Callable) -> Callable:
    async def hotkey(a: dict, c: Any) -> Any:
        keys = str(a.get("keys") or "")
        if _GLOBAL.search(keys) or a.get("app"):
            return f"Error: {keys!r} isn't available in the self-test (it would act outside the test window)."
        return await handler(a, c)

    return hotkey


class _Counting:
    """The brain chain, with every call timed and whoever answered noted."""

    def __init__(self, chain: Any) -> None:
        self.chain = chain
        self.calls: list[dict] = []
        self.name = getattr(chain, "name", "chain")
        self.supports_vision = getattr(chain, "supports_vision", True)
        self.supports_tools = getattr(chain, "supports_tools", True)

    @property
    def last_used(self) -> str:
        return getattr(self.chain, "last_used", "")

    async def complete(self, messages: list, **kw: Any) -> Any:
        t = time.monotonic()
        err = ""
        try:
            return await self.chain.complete(messages, **kw)
        except Exception as e:
            err = type(e).__name__
            raise
        finally:
            self.calls.append({"s": round(time.monotonic() - t, 2), "effort": kw.get("effort", ""), "brain": self.last_used, "error": err})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.chain, name)


@contextlib.contextmanager
def _settings_for(arm: Arm) -> Iterator[dict]:
    """The arm's switches set on NEO's settings for the run, and put back after. Says which exist."""
    from neo.config import settings

    s = settings()
    old: list[tuple[str, Any]] = []
    have: dict[str, bool] = {}
    for names, value in ((EFFORT_SETTINGS, arm.effort), (BATCH_SETTINGS, arm.batch)):
        name = next((n for n in names if hasattr(s, n)), None)
        have[names[0]] = name is not None
        if name is None or value is None:
            continue
        old.append((name, getattr(s, name)))
        object.__setattr__(s, name, value)
    try:
        yield have
    finally:
        for name, v in old:
            object.__setattr__(s, name, v)


def _check(req: Request, arm: Arm, results: list[dict]) -> Callable:
    async def check(r: controls._Run) -> controls.Check:
        from neo.agent import confirm
        from neo.agent.loop import run_agent
        from neo.agent.prompt import build
        from neo.agent.registry import ToolContext, registry
        from neo.providers import brain

        c = controls.Check(f"agent_{req.key}", f"{req.key} ({arm.name})")
        with _settings_for(arm) as have:
            if arm.batch is not None and not have[BATCH_SETTINGS[0]]:
                return c.skip("this build has no batching switch")
            reg = registry().subset([n for n in AGENT_TOOLS if registry().get(n) is not None])
            if (hk := reg.get("hotkey")) is not None:
                reg._tools["hotkey"] = dataclasses.replace(hk, handler=_safe_hotkey(hk.handler))
            words = req.words(r)
            chain = _Counting(brain("vision"))
            m = r.mark()
            t0 = time.monotonic()
            try:
                res = await run_agent(
                    words, provider=chain, system=build(""), tools=reg.specs(), reg=reg, strict=True, effort=arm.effort,
                    max_seconds=RUN_S, ctx=ToolContext(user_text=words, taint=r.taint, origin="user"), verify=True,
                )  # fmt: skip
                stopped, steps = res.stopped, len(res.steps)
            except Exception as e:  # noqa: BLE001 - a broken run is a failed run, not a broken test
                stopped, steps = f"error: {type(e).__name__}", 0
            secs = time.monotonic() - t0
        if confirm.peek() is not None:
            confirm.cancel("the self-test never answers a question")
        await r.wait(m, lambda e: False, 0.5)  # (the window's log catches up)
        ok = req.goal(r, m)
        first = (getattr(chain.chain, "names", None) or [""])[0]
        first = getattr(first, "name", first)
        failover = [x for x in chain.calls if x["brain"] and first and x["brain"] != first]
        row = {
            "arm": arm.name, "request": req.key, "words": words, "success": ok, "stopped": stopped,
            "model_calls": len(chain.calls), "model_s": round(sum(x["s"] for x in chain.calls), 2),
            "tool_calls": steps, "seconds": round(secs, 2), "failover_calls": len(failover),
            "failover_s": round(sum(x["s"] for x in failover), 2), "brains": sorted({x["brain"] for x in chain.calls if x["brain"]}),
            "errors": [x["error"] for x in chain.calls if x["error"]],
        }  # fmt: skip
        results.append(row)
        c.sent = words
        c.said = f"{row['model_calls']} model calls in {row['model_s']} s, {steps} tool calls, {secs:.1f} s in all ({stopped})"
        return c.ok("done") if ok else c.fail("the goal didn't happen in the test window")

    check.__name__ = f"_agent_{req.key}"
    return check


def summarize(results: list[dict]) -> dict[str, Any]:
    by: dict[str, list[dict]] = {}
    for row in results:
        by.setdefault(row["arm"], []).append(row)
    out: dict[str, Any] = {}
    for arm, rows in by.items():
        calls = [x["model_calls"] for x in rows]
        secs = [x["seconds"] for x in rows]
        out[arm] = {
            "runs": len(rows), "success": round(sum(x["success"] for x in rows) / len(rows), 3),
            "model_calls_p50": statistics.median(calls), "seconds_p50": round(statistics.median(secs), 2),
            "seconds_p90": round(sorted(secs)[min(len(secs) - 1, int(0.9 * (len(secs) - 1) + 0.5))], 2),
            "failover_share": round(sum(x["failover_calls"] for x in rows) / max(1, sum(calls)), 3),
            "failover_s": round(sum(x["failover_s"] for x in rows), 2),
        }  # fmt: skip
    return out


def verdict(summary: dict[str, Any], a: str, b: str, *, min_runs: int = RUNS_PER_ARM, slack: float = 0.05) -> str:
    """Whether arm ``a`` may become the default over ``b``: at least ``min_runs`` each, success no
    more than ``slack`` lower, and faster (fewer seconds at p50). Words, for the user."""
    sa, sb = summary.get(a), summary.get(b)
    if not sa or not sb:
        return f"{a} vs {b}: not measured"
    if min(sa["runs"], sb["runs"]) < min_runs:
        return f"{a} vs {b}: too few runs ({sa['runs']} and {sb['runs']}; {min_runs} each needed)"
    if sa["success"] + slack < sb["success"]:
        return f"{a} vs {b}: keep {b} — {a} succeeded less ({sa['success']:.0%} vs {sb['success']:.0%})"
    if sa["seconds_p50"] >= sb["seconds_p50"]:
        return f"{a} vs {b}: keep {b} — {a} wasn't faster ({sa['seconds_p50']} s vs {sb['seconds_p50']} s at p50)"
    return (
        f"{a} vs {b}: {a} can be the default — as successful ({sa['success']:.0%} vs {sb['success']:.0%}) and faster "
        f"({sa['seconds_p50']} s vs {sb['seconds_p50']} s at p50, {sa['model_calls_p50']} vs {sb['model_calls_p50']} model calls)"
    )


@dataclass
class Result:
    report: controls.Report
    runs: list[dict] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        s = summarize(self.runs)
        return {"runs": self.runs, "arms": s, "verdicts": verdicts(s), "controls": self.report.to_json()}

    def text(self) -> str:
        s = summarize(self.runs)
        lines = [self.report.summary(), ""]
        for arm, v in s.items():
            lines.append(
                f"{arm}: {v['runs']} runs, success {v['success']:.0%}, {v['model_calls_p50']} model calls and "
                f"{v['seconds_p50']} s at p50 (p90 {v['seconds_p90']} s); failover {v['failover_share']:.0%} of calls, {v['failover_s']} s"
            )
        lines += ["", *verdicts(s)]
        skipped = [c for c in self.report.checks if c.status == "skip"]
        if skipped:
            lines.append("skipped: " + "; ".join(f"{c.title} ({c.arrived})" for c in skipped[:4]))
        return "\n".join(lines) + "\n"


def verdicts(s: dict[str, Any]) -> list[str]:
    out = []
    if "low" in s or "high" in s:
        out.append(verdict(s, "low", "high"))
    if "batch on" in s or "batch off" in s:
        out.append(verdict(s, "batch on", "batch off"))
    return out


async def run(*, arms: list[Arm] | None = None, runs: int = 1, mac: Any = None, notify: Callable[[str], Any] | None = None) -> Result:
    """The requests, ``runs`` times per arm (cycling through them), in one test window session."""
    arms = arms or [Arm("as built")]
    results: list[dict] = []
    checks = []
    for arm in arms:
        for i in range(runs):
            checks.append(_check(REQUESTS[i % len(REQUESTS)], arm, results))
    report = await controls.run(
        mac=mac, notify=notify, checks=tuple(checks), check_s=RUN_S + 10, budget_s=max(60.0, len(checks) * (RUN_S + 10)),
    )  # fmt: skip
    return Result(report, results)


def save(result: Result, kind: str) -> Any:
    try:
        from neo.config import settings

        folder = settings().data_dir / "selftest"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / time.strftime(f"{kind}-%Y%m%d-%H%M%S.json")
        path.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        path.chmod(0o600)
        return path
    except OSError:
        return None
