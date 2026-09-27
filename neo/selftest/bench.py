"""The controls self-test's numbers: how long each of NEO's tools took, against the plan's targets,
and the two scores — plus an opt-in, read-only timing walk of one app's Accessibility tree.

* Timings are tool timings only: the time inside NEO's tool, from the call to its answer (not the
  check's own waiting for the test window's log). ``--bench N`` repeats the checks of each
  primitive N times (default 5) within a 90 s budget; one run gives one sample each.
* Success score = passed / (passed + failed) over checks × repetitions; skipped checks are listed,
  never counted. Latency score = share of the primitives with a target whose p50 meets it (P0).
* ``ax_walk`` times Accessibility calls on one app the user names (or the one in front): how long
  a call really takes on this Mac, so the hermetic bench's per-call costs aren't guesses. It reads
  only (roles, positions, sizes, titles, children — timings are kept, never what they say), never
  NEO's own windows or a protected app, and runs only when the user asks for it.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

# Per-primitive targets, ms of tool time (P0: the plan's P0 as the critique relaxed it; P1: with
# ScreenCaptureKit). type_text is per 40 characters. See .controls_design.json "targets".
TARGETS_MS: dict[str, dict[str, float]] = {
    "screenshot": {"p0": 250, "p1": 120},
    "click": {"p0": 550, "p1": 250},
    "drag": {"p0": 1200, "p1": 900},
    "scroll": {"p0": 450, "p1": 200},
    "hotkey": {"p0": 450, "p1": 200},
    "ax_press": {"p0": 450, "p1": 200},
    "type_text": {"p0": 100, "p1": 100},
    "ax_tree": {"p0": 300, "p1": 150},
    "click_text": {"p0": 300, "p1": 150},
}
BENCH_REPS = 5
BENCH_BUDGET_S = 90.0


def primitive(tool: str, args: dict) -> tuple[str, float]:
    """(the primitive a tool call measures, the factor its time is scaled by) — type_text per 40
    characters; tools without a target are their own name."""
    if tool == "type_text":
        n = len(str(args.get("text") or ""))
        return tool, (40.0 / n) if n else 1.0
    return tool, 1.0


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def summarize(timings: list[tuple[str, float, dict]]) -> dict[str, dict[str, Any]]:
    """Per primitive: samples, p50 and p90 ms, and the targets met."""
    by: dict[str, list[float]] = {}
    for tool, ms, args in timings:
        name, k = primitive(tool, args)
        by.setdefault(name, []).append(ms * k)
    out: dict[str, dict[str, Any]] = {}
    for name, xs in sorted(by.items()):
        t = TARGETS_MS.get(name, {})
        p50, p90 = statistics.median(xs), _pct(xs, 0.9)
        out[name] = {
            "n": len(xs), "p50_ms": round(p50, 1), "p90_ms": round(p90, 1), "target_ms": t,
            "met_p0": (p50 <= t["p0"]) if "p0" in t else None, "met_p1": (p50 <= t["p1"]) if "p1" in t else None,
        }  # fmt: skip
    return out


def scores(statuses: list[str], summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Success score over checks × repetitions (skips listed apart), latency score over the
    primitives that have a target."""
    passed, failed = statuses.count("pass"), statuses.count("fail")
    timed = [r for r in summary.values() if r["met_p0"] is not None]
    return {
        "success": round(passed / (passed + failed), 3) if passed + failed else None,
        "passed": passed,
        "failed": failed,
        "skipped": statuses.count("skip"),
        "latency": round(sum(1 for r in timed if r["met_p0"]) / len(timed), 3) if timed else None,
        "latency_p1": round(sum(1 for r in timed if r["met_p1"]) / len(timed), 3) if timed else None,
        "timed": len(timed),
    }


def words(sc: dict[str, Any]) -> str:
    """The scores as NEO says them."""
    if sc.get("success") is None:
        return ""
    out = f"Success {sc['success']:.0%} ({sc['passed']} of {sc['passed'] + sc['failed']})"
    if sc.get("latency") is not None:
        out += f"; speed targets met by {sc['latency']:.0%} of {sc['timed']} tools"
    return out + "."


# ---- a read-only timing walk of one app's Accessibility tree ----------------------------------------

_WALK_ATTRS = ("AXRole", "AXPosition", "AXSize", "AXTitle", "AXChildren")


def ax_walk(app: str = "", *, max_nodes: int = 1500, budget_s: float = 3.0) -> dict[str, Any]:
    """Time every Accessibility read of a walk of ``app`` (the one in front when ""). Returns only
    numbers: calls, per-call p50/p90/max ms, batched reads' ms, errors by code, nodes, seconds."""
    from neo import effects

    effects.guard("selftest", "read an app's Accessibility tree for timings")
    import ApplicationServices as AS

    from neo.tools import computer
    from neo.tools.computer import ax

    if app:
        pid, name = ax.pid_for_app(app), app
        if pid is None:
            return {"error": f"{app} isn't running"}
    else:
        name, pid = ax.working_app()
    if not pid or ax.is_neo(pid, name):
        return {"error": "that is NEO's own window"}
    if computer._protected(name, pid):
        return {"error": f"{name} is an app NEO never touches"}
    root = AS.AXUIElementCreateApplication(pid)
    single: list[float] = []
    multi: list[float] = []
    errors: dict[int, int] = {}
    todo, nodes = [root], 0
    end = time.monotonic() + budget_s
    batched = hasattr(AS, "AXUIElementCopyMultipleAttributeValues")
    while todo and nodes < max_nodes and time.monotonic() < end:
        el = todo.pop(0)
        nodes += 1
        kids = None
        try:  # a hung app answers each read within half a second, never the default six
            AS.AXUIElementSetMessagingTimeout(el, 0.5)
        except Exception:  # noqa: BLE001
            pass
        for attr in _WALK_ATTRS:
            t = time.perf_counter()
            try:
                err, val = AS.AXUIElementCopyAttributeValue(el, attr, None)
            except Exception:  # noqa: BLE001
                err, val = -1, None
            single.append((time.perf_counter() - t) * 1000)
            if err:
                errors[int(err)] = errors.get(int(err), 0) + 1
            elif attr == "AXChildren":
                kids = val
        if batched:
            t = time.perf_counter()
            try:
                AS.AXUIElementCopyMultipleAttributeValues(el, list(_WALK_ATTRS), 0, None)
            except Exception:  # noqa: BLE001
                pass
            multi.append((time.perf_counter() - t) * 1000)
        todo.extend(list(kids or [])[:200])

    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        return {"p50": round(statistics.median(xs), 3), "p90": round(_pct(xs, 0.9), 3), "max": round(max(xs), 3),
                "mean": round(statistics.fmean(xs), 3)}  # fmt: skip

    return {
        "app": name,
        "nodes": nodes,
        "calls": len(single),
        "per_call_ms": stats(single),
        "batched_ms": stats(multi),
        "errors": errors,
        "seconds": round(sum(single + multi) / 1000, 3),
        "complete": not todo,
    }
