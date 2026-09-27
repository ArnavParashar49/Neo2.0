"""The controls bench: per-primitive cost and per-task round trips, hermetic, in one report.

    python -m bench.controls.run                       # table + bench/results/<sha>.json
    python -m bench.controls.run --diff bench/baselines/step0.json
    python -m bench.controls.run --only primitives --reps 3
    python -m bench.controls.run --save-baseline       # bench/baselines/step0.json (the Step-0 tree only)

(a) Primitives. NEO-side time = measured Python ms (time in fakes subtracted) + AX calls × per-call
latency (0.05 / 0.15 / 0.5 / 0.9 ms) + captures × (screencapture CLI 0.2 s | ScreenCaptureKit 0.1 s)
+ the sleeps it counted. Each primitive runs through NEO's real tool handler (origin "agent": the
looks before and after a model's input included) on a fresh fake world.

(b) Tasks. Each scenario (bench/controls/scenarios.py) goes through the real Session → fast paths →
agent loop → tool selection → prompt, with the scripted brain; it reports round trips, prompt and
image tokens and effort per call, and a MODELED wall time: NEO-side time + Σ per-call model latency
by effort (low 1.5 s, medium 5 s, high 12 s — published first-token figures, not measurements).

Everything runs with NEO_NO_EFFECTS=1 in a scratch data folder: no event reaches the Mac, no model
is called, nothing is read from or written to ~/.neo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
RESOLVER_DATA = ROOT / "tests" / "data" / "ground_eval.jsonl"


def prepare_process() -> Path:
    """A process that runs the bench on its own: effects off before NEO is imported, a scratch data
    folder, no model, no voice, no keys. (Under pytest the suite's conftest has done all this.)"""
    os.environ["NEO_NO_EFFECTS"] = "1"
    os.environ.setdefault("NEO_REFLEX", "off")
    os.environ.setdefault("NEO_VOICE", "off")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY"):
        os.environ.pop(k, None)
    data = Path(tempfile.mkdtemp(prefix="neo-bench-"))
    os.environ["NEO_DATA_DIR"] = str(data / "neo_data")
    os.environ["NEO_NOTES_DIR"] = str(data / "notes")
    return data


# ---- targets -------------------------------------------------------------------------------------------
# NEO-side ms at 0.15 ms per AX call and the screencapture CLI. "p0": the plan's P0 targets as the
# critique relaxed them (a pixel-checked click can't beat two CLI captures); "p1": the plan's P1
# (ScreenCaptureKit) figures. "ax": the AX-call budgets. See .controls_design.json "targets".
TARGETS: dict[str, dict[str, float]] = {
    "screenshot_full": {"p0": 250, "p1": 120},
    "screenshot_window": {"p0": 250, "p1": 120},
    "screenshot_zoom": {"p0": 250, "p1": 120},
    "click_front": {"p0": 550, "p1": 250},
    "click_bring": {"p0": 670, "p1": 370},
    "drag": {"p0": 1200, "p1": 900},
    "scroll": {"p0": 450, "p1": 200},
    "hotkey": {"p0": 450, "p1": 200},
    "hotkey_quiet": {"p0": 700, "p1": 450},
    "ax_press": {"p0": 450, "p1": 200},
    "ax_press_quiet": {"p0": 700, "p1": 450},
    "type_text": {"p0": 100, "p1": 100},
    "ax_tree": {"p0": 200, "p1": 150, "ax": 1000},
    "ax_tree_hung": {"p0": 1500, "p1": 1500},
    "click_text": {"p0": 200, "p1": 120, "ax": 600},
    "ax_hit": {"p0": 5, "p1": 5, "ax": 4},
}


# ---- primitives ----------------------------------------------------------------------------------------


@dataclass
class Prim:
    name: str
    build: Callable[[], Any]  # → World
    act: Callable[[Any, dict], Awaitable[Any]]  # (env, scratch) → the tool's output
    setup: Callable[[Any, dict], Awaitable[None]] | None = None
    worked: Callable[[Any, Any], bool] | None = None  # (world, output) → did it do its job
    note: str = ""


def _ctx(text: str = "") -> Any:
    from neo.agent.registry import ToolContext

    return ToolContext(user_text=text, origin="agent")


async def _tool(name: str, **args: Any) -> Any:
    from neo.agent.registry import registry

    return await registry().invoke(name, args, _ctx())


def _id_of(tree: str, label: str) -> str:
    from bench.controls.scenarios import find_id

    eid = find_id(tree, label)
    if eid is None:
        raise RuntimeError(f"no {label!r} in the tree:\n{tree[:400]}")
    return eid


def _prims() -> list[Prim]:
    from bench.controls import scenarios as S
    from bench.controls.fakes import World

    def board() -> World:
        return S.board_world()

    def form() -> World:
        return S.world(lambda w: S.form_app(), front="Accounts")

    def big() -> World:
        return S.world(lambda w: S.big_app(), front="Safari")

    def notes() -> World:
        return S.world(lambda w: S.notes_app(), front="Notes")

    def form_and_notes() -> World:
        return S.world(lambda w: S.notes_app(), lambda w: S.form_app(), front="Accounts")

    def hung() -> World:
        w = big()
        for el in w.named("Safari").root.walk():
            el.hung = True
        return w

    async def tree_first(env: Any, s: dict) -> None:
        s["tree"] = (await _tool("ax_tree")).text

    def shot_params() -> set[str]:
        from neo.agent.registry import registry

        return set(registry().get("screenshot").parameters.get("properties", {}))

    async def zoom(env: Any, s: dict) -> Any:
        args: dict[str, Any] = {"x": 400, "y": 130, "w": 240, "h": 160}
        if "zoom" in shot_params():
            from neo.agent.registry import registry

            kind = registry().get("screenshot").parameters["properties"]["zoom"].get("type")
            args["zoom"] = True if kind == "boolean" else 2
        return await _tool("screenshot", **args)

    def checkbox_on(w: World, out: Any) -> bool:
        el = w.named("Accounts").root.find(lambda e: e.role == "AXCheckBox")
        return bool(el and el.attrs.get("AXValue"))

    def sheet_open(app_name: str) -> Callable[[World, Any], bool]:
        return lambda w, out: w.named(app_name).root.find(lambda e: e.role == "AXSheet") is not None

    def hit(env: Any, s: dict) -> Awaitable[Any]:
        from neo.tools.computer import ax

        return asyncio.to_thread(ax.hit, 130 + 40 * 3 + 16, 94)

    return [
        Prim("screenshot_full", board, lambda e, s: _tool("screenshot"), worked=lambda w, o: bool(o.images)),
        Prim("screenshot_window", board, lambda e, s: _tool("screenshot", x=360, y=90, w=900, h=800), worked=lambda w, o: bool(o.images)),
        Prim("screenshot_zoom", board, zoom, worked=lambda w, o: bool(o.images), note="a 240×160 pt region (zoom= when offered)"),
        Prim("click_front", form, lambda e, s: _tool("click", x=430, y=341), worked=checkbox_on,
             note="a checkbox in the app in front; its picture changes"),  # fmt: skip
        Prim("click_bring", form_and_notes, lambda e, s: _tool("click", x=1350, y=145),
             worked=lambda w, o: w.front_pid == S.NOTES and w.pressed("Bold"), note="a button of an app behind the front one"),  # fmt: skip
        Prim("drag", board, lambda e, s: _tool("drag", x1=762, y1=644, x2=762, y2=492),
             worked=lambda w, o: bool(w.board and "e4" in w.board.pieces), note="a piece e2→e4 on the board"),  # fmt: skip
        Prim("scroll", big, lambda e, s: _tool("scroll", x=785, y=496, dy=-5), worked=lambda w, o: any(v > 0 for v in w.scrolled.values())),
        Prim("hotkey", board, lambda e, s: _tool("hotkey", keys="cmd+n"), worked=sheet_open("Chess"), note="⌘N opens a sheet"),
        Prim("hotkey_quiet", board, lambda e, s: _tool("hotkey", keys="cmd+j"), worked=lambda w, o: w.pressed("Show Coordinates"),
             note="a menu shortcut nothing on screen shows"),  # fmt: skip
        Prim("ax_press", form, lambda e, s: _tool("ax_press", id=_id_of(s["tree"], "Options…")), setup=tree_first,
             worked=sheet_open("Accounts"), note="a button that opens a sheet"),  # fmt: skip
        Prim("ax_press_quiet", form, lambda e, s: _tool("ax_press", id=_id_of(s["tree"], "Remember me")), setup=tree_first,
             worked=checkbox_on, note="a checkbox: nothing in the scene changes"),  # fmt: skip
        Prim("type_text", notes, lambda e, s: _tool("type_text", text="The quick brown fox jumps over a lazy do"),
             worked=lambda w, o: "lazy do" in str(w.named("Notes").focused.attrs.get("AXValue") or ""), note="40 characters"),  # fmt: skip
        Prim("ax_tree", big, lambda e, s: _tool("ax_tree"), worked=lambda w, o: "Tool 3" in o.text, note="the 2,522-element app"),
        Prim("ax_tree_hung", hung, lambda e, s: _tool("ax_tree"), worked=lambda w, o: True, note="an app that answers nothing"),
        Prim("click_text", big, lambda e, s: _tool("click_text", label="Action 20"), worked=lambda w, o: w.pressed("Action 20"),
             note="a button in the 2,522-element app (the gate's walk of its window)"),  # fmt: skip
        Prim("ax_hit", big, hit, worked=lambda w, o: getattr(o, "label", "") == "Tool 3", note="what's under a toolbar button"),
    ]


def _modeled(m: dict, lat_ms: float, capture_s: float) -> float:
    return m["py_ms"] + m["ax_calls"] * lat_ms + m["captures"] * capture_s * 1000 + (m["sleep_s"] + m["ax_block_s"]) * 1000


async def _one_prim(p: Prim) -> dict:
    from bench.controls.fakes import install
    from neo.tools.computer import apps

    w = p.build()
    with install(w) as env:
        env.patches.set(apps, "_target", None)
        scratch: dict = {}
        if p.setup is not None:
            await p.setup(env, scratch)
        m = env.mark()
        out = await p.act(env, scratch)
        got = env.since(m)
    text = getattr(out, "text", None)
    got["ok"] = bool(getattr(out, "ok", True)) and not str(text or "").startswith(("Error", "NEEDS_"))
    got["worked"] = bool(p.worked(w, out)) if p.worked else got["ok"]
    got["said"] = " ".join(str(text if text is not None else out).split())[:160]
    return got


def measure_primitives(reps: int = 5, only: list[str] | None = None) -> dict:
    from neo.tools import load_all

    load_all()
    out: dict[str, Any] = {}
    for p in _prims():
        if only and p.name not in only:
            continue
        runs = [asyncio.run(_one_prim(p)) for _ in range(max(1, reps))]
        last = runs[-1]
        py = sorted(r["py_ms"] for r in runs)
        row = {k: last[k] for k in ("ax_calls", "captures", "sleeps", "sleep_s", "ax_block_s", "posted", "brings", "ok", "worked", "said", "slept_in")}
        row["py_ms_p50"] = round(statistics.median(py), 2)
        row["py_ms_p90"] = round(py[min(len(py) - 1, int(round(0.9 * (len(py) - 1))))], 2)
        row["steady"] = len({(r["ax_calls"], r["captures"], r["sleeps"]) for r in runs}) == 1
        base = dict(last, py_ms=row["py_ms_p50"])
        row["modeled_ms"] = {
            kind: {f"{lat:g}": round(_modeled(base, lat, cap), 1) for lat in _latencies()}
            for kind, cap in _captures().items()
        }
        row["neo_ms"] = row["modeled_ms"]["cli"][f"{_headline():g}"]
        row["target"] = TARGETS.get(p.name, {})
        row["note"] = p.note
        out[p.name] = row
    return out


def _latencies() -> tuple[float, ...]:
    from bench.controls.fakes import AX_LATENCY_MS

    return AX_LATENCY_MS


def _captures() -> dict[str, float]:
    from bench.controls.fakes import CAPTURE_S

    return CAPTURE_S


def _headline() -> float:
    from bench.controls.fakes import HEADLINE_AX_MS

    return HEADLINE_AX_MS


# ---- scenarios ---------------------------------------------------------------------------------------------

# Tools that act for real on the fake Mac. The rest (a program, a file, the web, AppleScript, opening
# things) get a fake handler that records the call; opening or switching apps moves the fake world.
_REAL_TOOLS = frozenset({
    "ax_tree", "ax_find", "screenshot", "apps_running", "ax_press", "ax_set_value", "click", "drag", "scroll",
    "type_text", "hotkey", "click_text", "activate_app",
})  # fmt: skip
_NEVER_REAL = frozenset({
    "shell", "applescript", "open_url", "page_read", "safari_read", "finder_reveal", "read_file", "write_file",
    "edit_file", "trash", "quit_app", "close", "dictate", "open_app",
})  # fmt: skip
_COORDS = frozenset({"x", "y", "x1", "y1", "x2", "y2", "box_2d", "nx", "ny", "point", "points", "at"})


def _real(t: Any) -> bool:
    return t.name in _REAL_TOOLS or (t.category == "computer" and t.name not in _NEVER_REAL)


def _fake_handler(name: str, world: Any) -> Callable:
    async def handler(a: dict, c: Any) -> str:
        if name in ("open_app", "activate_app"):
            target = str(a.get("target") or a.get("name") or "").strip().lower()
            got = next((x for x in world.apps.values() if x.name.lower() == target), None)
            if got is None:
                return f"Error: {a.get('target') or a.get('name')} isn't on this Mac."
            world.activate(got.pid)
            return f"Opened {got.name}."
        return f"Error: {name} isn't available in the bench."

    return handler


class _Reflex:
    def __init__(self, decision: Any, real: Any) -> None:
        self._d, self._real = decision, real

    async def decide(self, text: str) -> Any:
        return self._d

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


async def run_scenario(sc: Any) -> dict:
    from bench.controls.fakes import MODEL_S, FakeBrain, install
    from neo.agent import confirm, fastpath, grants
    from neo.agent import session as sess
    from neo.agent.registry import Registry, registry
    from neo.memory import embed
    from neo.reflex.schema import Decision
    from neo.tools import load_all
    from neo.tools.computer import apps

    load_all()
    w = sc.build()
    brain = FakeBrain(sc.policy(w), sees=sc.sees)
    rec: list[dict] = []
    decision = Decision(sc.intent, 0.95, False, 0.0, sc.needs_screen, 0.9 if sc.needs_screen else 0.1, source="bench",
                        reply=False, reply_p=0.1)  # fmt: skip
    with install(w) as env:
        p = env.patches
        p.set(embed, "embed_texts", lambda texts: None)
        p.set(apps, "_target", None)

        def brain_for(purpose: str = "agent") -> FakeBrain:
            brain.purposes.append(purpose)
            return brain

        p.set(sess, "brain", brain_for)
        p.set(sess, "_game_to_play", lambda text: None, only_if_present=True)
        real_plan = fastpath.plan

        def plan(text: str, *a: Any, **k: Any) -> Any:
            steps = real_plan(text, *a, **k)
            ok = steps and all((t := registry().get(n)) is not None and (_real(t) or n == "open_app") for n, _ in steps)
            return steps if ok else None

        p.set(fastpath, "plan", plan)
        for t in registry().all():
            if not _real(t):
                p.set(t, "handler", _fake_handler(t.name, w))
        real_invoke = Registry.invoke

        async def invoke(self: Any, name: str, args: dict, ctx: Any) -> Any:
            m = env.mark()
            out = await real_invoke(self, name, args, ctx)
            got = env.since(m)
            rec.append({"tool": name, "args": {k: v for k, v in args.items() if k != "text"}, "origin": ctx.origin,
                        "blind": bool(ctx.get("blind")), "ok": out.ok, "text": out.text[:200], "images": len(out.images),
                        **{k: got[k] for k in ("ax_calls", "captures", "sleep_s", "posted", "brings")}})  # fmt: skip
            return out

        p.set(Registry, "invoke", invoke)
        s = sess.Session()
        s.reflex = _Reflex(decision, s.reflex)
        m = env.mark()
        try:
            reply = await asyncio.wait_for(s.handle(sc.utterance, from_user=True), 120)
            err = ""
        except Exception as e:  # noqa: BLE001 - a broken run is a failed scenario, not a broken bench
            reply, err = None, f"{type(e).__name__}: {e}"[:300]
        neo = env.since(m)
        asked = any(r["text"].startswith("NEEDS_CONFIRM") for r in rec) or bool(reply and reply.route == "confirm")
        if confirm.peek() is not None:
            confirm.cancel("the bench never answers a question")
        grants.reset()
    info = {"asked": asked}
    blind_acted = [r for r in rec if r["blind"] and _COORDS & set(r["args"]) and (r["posted"] or r["brings"])]
    checks = {
        "goal": bool(sc.goal(w, info)) and not err,
        "no_unconfirmed_commit": not w.commits_pressed(),
        "no_blind_coords": not blind_acted,
        "not_neo_window": not w.on_neo(),
    }
    calls = brain.calls
    model_s = sum(MODEL_S.get(c.effort, MODEL_S["medium"]) for c in calls)
    neo_ms = _modeled(neo, _headline(), _captures()["cli"])
    return {
        "utterance": sc.utterance,
        "pass": all(checks.values()),
        "checks": checks,
        "round_trips": len(calls),
        "max_calls_per_turn": max(brain.per_turn, default=0),
        "efforts": [c.effort for c in calls],
        "purposes": brain.purposes,
        "prompt_tokens": [round(c.prompt_tokens) for c in calls],
        "image_tokens": [round(c.image_tokens) for c in calls],
        "tools_offered": [c.n_tools for c in calls],
        "total_prompt_tokens": round(sum(c.prompt_tokens for c in calls)),
        "tools": [r["tool"] for r in rec],
        "route": getattr(reply, "route", "") if reply else "",
        "stopped": (reply.result.stopped if reply is not None and reply.result is not None else ""),
        "asked": asked,
        "error": err,
        "neo": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in neo.items() if k != "slept_in"},
        "neo_s": round(neo_ms / 1000, 3),
        "model_s": round(model_s, 2),
        "modeled_s": round(neo_ms / 1000 + model_s, 2),
        "answer": (reply.text if reply else "")[:120],
        "targets": sc.targets,
        "note": sc.note,
    }


def run_scenarios(only: list[str] | None = None) -> dict:
    from bench.controls.scenarios import SCENARIOS

    out = {}
    for sc in SCENARIOS:
        if only and sc.name not in only:
            continue
        out[sc.name] = asyncio.run(run_scenario(sc))
    return out


# ---- features (what the code offers, so thresholds switch on as work lands) ---------------------------------


def features() -> dict[str, Any]:
    """What NEO's code offers now, found by looking (a tool's parameters, a dataclass field, a
    setting, a module) — never by a flag a workstream sets for the bench."""
    import dataclasses
    import importlib
    import inspect

    from neo.agent.registry import ToolOutput, registry, tool
    from neo.config import settings
    from neo.providers.base import ImagePart
    from neo.tools import load_all

    load_all()

    def props(name: str) -> set[str]:
        t = registry().get(name)
        return set(t.parameters.get("properties", {})) if t else set()

    def has_module(name: str) -> bool:
        try:
            importlib.import_module(name)
            return True
        except ImportError:
            return False

    s = settings()
    return {
        "cell_click": "cell" in props("click"),
        "marks": "mark" in props("click"),
        "zoom": "zoom" in props("screenshot"),
        "menu_command": registry().get("menu_command") is not None,
        "effect": "effect" in {f.name for f in dataclasses.fields(ToolOutput)},
        "image_detail": "detail" in {f.name for f in dataclasses.fields(ImagePart)},
        "wait_until": has_module("neo.tools.computer.wait"),
        "gates_module": has_module("neo.tools.computer.gates"),
        "by_utterance": "by_utterance" in inspect.signature(tool).parameters,
        "screen_effort": getattr(s, "screen_effort", None),
        "prefetch_look": getattr(s, "prefetch_look", None),
        "shot_format": getattr(s, "shot_format", None),
        "resolver": getattr(s, "resolver", None),
        "sck_capture": getattr(s, "sck_capture", None),
    }


def resolver_eval(path: Path = RESOLVER_DATA) -> dict | None:
    """The resolver eval's rows, when the commands workstream has written them (else None). The
    eval itself loads models, so it runs as its own opt-in script (bench/controls/resolver_eval.py)."""
    if not path.exists():
        return None
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"rows": n, "path": str(path.relative_to(ROOT)), "run": "python -m bench.controls.resolver_eval"}


# ---- the report -----------------------------------------------------------------------------------------------


def git_sha(root: Path = ROOT) -> str:
    """HEAD's commit, read from .git (no git process: effects refuse programs)."""
    try:
        g = root / ".git"
        if g.is_file():
            g = Path(g.read_text().split("gitdir:", 1)[1].strip())
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1]
        common = Path((g / "commondir").read_text().strip()) if (g / "commondir").exists() else g
        common = common if common.is_absolute() else (g / common).resolve()
        for base in (g, common):
            if (base / ref).exists():
                return (base / ref).read_text().strip()[:12]
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:12]
    except (OSError, IndexError):
        pass
    return "unknown"


def build_report(prims: dict | None, scens: dict | None, *, reps: int) -> dict:
    from bench.controls.fakes import AX_LATENCY_MS, CAPTURE_S, HEADLINE_AX_MS, IMG_TOKENS, MODEL_S

    return {
        "schema": SCHEMA,
        "sha": git_sha(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "params": {
            "ax_latency_ms": list(AX_LATENCY_MS), "headline_ax_ms": HEADLINE_AX_MS, "capture_s": CAPTURE_S,
            "model_s": MODEL_S, "image_tokens": IMG_TOKENS, "reps": reps,
            "modeled": "model latency is modeled from published first-token figures, not measured",
        },  # fmt: skip
        "features": features(),
        "primitives": prims or {},
        "scenarios": scens or {},
        "resolver_eval": resolver_eval(),
        "scores": scores(prims or {}, scens or {}),
    }


def scores(prims: dict, scens: dict) -> dict:
    met = [n for n, r in prims.items() if r.get("target", {}).get("p0") is not None and r["neo_ms"] <= r["target"]["p0"]]
    met1 = [n for n, r in prims.items() if r.get("target", {}).get("p1") is not None and r["neo_ms"] <= r["target"]["p1"]]
    timed = [n for n, r in prims.items() if r.get("target", {}).get("p0") is not None]
    passed = [n for n, r in scens.items() if r["pass"]]
    return {
        "latency_p0": round(len(met) / len(timed), 3) if timed else None,
        "latency_p1": round(len(met1) / len(timed), 3) if timed else None,
        "tasks_passed": f"{len(passed)}/{len(scens)}",
        "round_trips": sum(r["round_trips"] for r in scens.values()),
    }


def _delta(now: float, was: float | None, fmt: str = "{:+.0f}") -> str:
    if was is None:
        return ""
    return fmt.format(now - was)


def table(rep: dict, base: dict | None = None) -> str:
    bp = (base or {}).get("primitives", {})
    bs = (base or {}).get("scenarios", {})
    lines = [f"## Controls bench — {rep['sha']} ({rep['when']})", ""]
    if rep["primitives"]:
        lines += [
            f"Primitives (NEO-side ms at {rep['params']['headline_ax_ms']} ms/AX call, screencapture CLI; p50 Python of "
            f"{rep['params']['reps']} runs)",
            "",
            "| primitive | AX calls | captures | sleeps s | Python ms | NEO ms | target P0/P1 | worked | Δ NEO ms | Δ AX |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
        for n, r in rep["primitives"].items():
            b = bp.get(n)
            t = r.get("target", {})
            tgt = f"{t.get('p0', '')}/{t.get('p1', '')}" + (f" (≤{int(t['ax'])} AX)" if "ax" in t else "")
            lines.append(
                f"| {n} | {r['ax_calls']} | {r['captures']} | {r['sleep_s'] + r['ax_block_s']:.2f} | {r['py_ms_p50']:.1f} | "
                f"{r['neo_ms']:.0f} | {tgt} | {'yes' if r['worked'] else 'NO'} | "
                f"{_delta(r['neo_ms'], b and b['neo_ms'])} | {_delta(r['ax_calls'], b and b['ax_calls'])} |"
            )
        lines.append("")
    if rep["scenarios"]:
        lines += [
            "Tasks (model time MODELED: low 1.5 s, medium 5 s, high 12 s per call)",
            "",
            "| task | pass | round trips | effort | prompt tokens | NEO s | modeled s | Δ trips | Δ modeled s | failed checks |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for n, r in rep["scenarios"].items():
            b = bs.get(n)
            eff = ",".join(sorted(set(r["efforts"]))) or "-"
            bad = ", ".join(k for k, v in r["checks"].items() if not v) + (f" ({r['error']})" if r["error"] else "")
            lines.append(
                f"| {n} | {'PASS' if r['pass'] else 'FAIL'} | {r['round_trips']} | {eff} | {r['total_prompt_tokens']} | "
                f"{r['neo_s']:.2f} | {r['modeled_s']:.1f} | {_delta(r['round_trips'], b and b['round_trips'])} | "
                f"{_delta(r['modeled_s'], b and b['modeled_s'], '{:+.1f}')} | {bad} |"
            )
        lines.append("")
    sc = rep["scores"]
    lines.append(
        f"Latency score: {sc['latency_p0']} of primitives meet P0 ({sc['latency_p1']} meet P1); tasks passed "
        f"{sc['tasks_passed']}; {sc['round_trips']} round trips in all."
    )
    feats = {k: v for k, v in rep["features"].items() if v not in (None, False)}
    lines.append(f"Features found: {', '.join(f'{k}={v}' if v is not True else k for k, v in feats.items()) or 'none'}")
    if rep.get("resolver_eval"):
        lines.append(f"Resolver eval data: {rep['resolver_eval']['rows']} rows — {rep['resolver_eval']['run']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controls.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", choices=["primitives", "scenarios"], default=None)
    ap.add_argument("--names", nargs="*", default=None, help="only these primitives / scenarios")
    ap.add_argument("--out", type=Path, default=None, help="default: bench/results/<sha>.json")
    ap.add_argument("--diff", type=Path, default=None, help="a baseline report to compare with")
    ap.add_argument("--save-baseline", action="store_true", help="write bench/baselines/step0.json too")
    args = ap.parse_args(argv)
    prepare_process()
    sys.path.insert(0, str(ROOT))
    prims = None if args.only == "scenarios" else measure_primitives(args.reps, args.names)
    scens = None if args.only == "primitives" else run_scenarios(args.names)
    rep = build_report(prims, scens, reps=args.reps)
    base = json.loads(args.diff.read_text()) if args.diff else None
    print(table(rep, base))
    out = args.out or ROOT / "bench" / "results" / f"{rep['sha']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"(written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out})")
    if args.save_baseline:
        bl = ROOT / "bench" / "baselines" / "step0.json"
        bl.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"(baseline written to {bl.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
