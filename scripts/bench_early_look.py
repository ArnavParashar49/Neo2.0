"""Hermetic benchmark for the early first look and the resolver prewarm (neo.agent.prefetch).

    NEO_NO_EFFECTS=1 .venv/bin/python scripts/bench_early_look.py

Nothing real is touched: a temporary data dir, a made-up Mac (front app, installed apps, a window of
40 elements, a synthetic 1600×1000 window picture), fake brains that follow today's prompt ("look
first, act, answer"), and a fake zero-shot checkpoint whose load and pass take the times measured on
an M3 Pro (neo/reflex/zeroshot.py: 0.40-0.70 s cold, 33-35 ms warm). It measures, through the real
agent loop and the real zeroshot state machine:

1. model round trips and prompt tokens per screen task, without and with the first look;
2. how much the look adds before the first model call when it runs beside routing (fake AX walks
   of 50-700 ms, a picture taken in parallel);
3. the resolver's first question after the user speaks, cold and prewarmed on the wake word.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ["NEO_NO_EFFECTS"] = "1"
os.environ["NEO_DATA_DIR"] = tempfile.mkdtemp(prefix="neo-bench-")
os.environ["NEO_NOTES_DIR"] = tempfile.mkdtemp(prefix="neo-bench-notes-")
os.environ["NEO_REFLEX"] = "off"
os.environ["NEO_VOICE"] = "off"
os.environ["NEO_PREFETCH_LOOK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from neo.agent import prefetch  # noqa: E402
from neo.agent.loop import run_agent  # noqa: E402
from neo.agent.registry import ToolContext, ToolOutput, registry  # noqa: E402
from neo.config import Settings  # noqa: E402
from neo.decide import inventory as inv  # noqa: E402
from neo.events import EventBus  # noqa: E402
from neo.memory import embed  # noqa: E402
from neo.providers.base import ImagePart, ToolCall, Turn  # noqa: E402
from neo.reflex.schema import Decision  # noqa: E402
from neo.tools import load_all  # noqa: E402
from neo.tools.computer import apps, ax, focus, pointer, screen  # noqa: E402

Settings.model_config["env_file"] = None
embed.embed_texts = lambda texts: None
load_all()

BYTES_PER_TOKEN = 3.4  # neo.providers.fit's rule
IMG_TOKENS = 1120  # a Gemini 3 picture at the default (high) media resolution


def tok(s: str) -> float:
    return len(s.encode("utf-8", "ignore")) / BYTES_PER_TOKEN


# ---- the made-up Mac ----------------------------------------------------------------------------

W = types.SimpleNamespace(walk_s=0.15, shot_s=0.25, front=("Tablero", 104))
_PIDS = {"Quillpad": 101, "Tablero": 104}


def _window_png() -> bytes:
    img = Image.new("RGB", (1600, 1000), (236, 236, 236))
    d = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            d.rectangle((400 + c * 100, 100 + r * 100, 500 + c * 100, 200 + r * 100),
                        fill=(238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86))  # fmt: skip
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


PNG = _window_png()
LABELS = ["New Page", "Share", "Title", "Body", "Bold", "Italic", "Search", "Sidebar", "Settings", "Help"]


def _snapshot(app=None, *, max_lines=250):
    time.sleep(W.walk_s)
    els = [ax.Element(f"e{i}", "AXButton" if i % 3 else "AXStaticText", f"{LABELS[i % 10]} {i}", "", 100 + i, 60 + i, 80, 22)
           for i in range(1, 41)]  # fmt: skip
    els[0] = ax.Element("e1", "AXButton", "New Page", "", 120, 60, 80, 22)
    for e in els:
        ax._issued[e.id] = e
    return "\n".join([f'App: {app} — window "Draft"'] + ["  " + e.line() for e in els])


def _capture(wid, region=None):
    time.sleep(W.shot_s)
    return ImagePart(PNG, "image/png")


INSTALLED = [inv.App("test.quillpad", "Quillpad", "/A/Quillpad.app"), inv.App("test.tablero", "Tablero", "/A/Tablero.app")]
ax.is_trusted = lambda: True
ax.working_app = lambda: W.front
ax.app_names = lambda pid: {n for n, p in _PIDS.items() if p == pid}
ax.is_neo = lambda pid, name="": False
ax.pid_for_app = lambda name: _PIDS.get(name)
ax.snapshot = _snapshot
inv.snapshot = lambda: list(INSTALLED)
inv.app_for_pid = lambda pid: next((a for a in INSTALLED if _PIDS[a.name] == pid), None)
screen.can_capture = lambda: True
screen.capture_window = _capture
focus.front_window_id = lambda pid: 7000 + pid
prefetch._window_bounds = lambda wid: (100.0, 50.0, 800.0, 500.0)
apps.set_target("")


# ---- 1. round trips and tokens -------------------------------------------------------------------


class Brain:
    """Today's prompt, followed faithfully: look (ax_tree; a screenshot too when the task needs the
    picture) before acting, act on what it saw, answer. Every request is accounted."""

    supports_vision, supports_tools, name = True, True, "fake"

    def __init__(self, needs_picture: bool) -> None:
        self.needs_picture = needs_picture
        self.calls: list[dict] = []

    async def complete(self, messages, *, system="", tools=None, effort="medium", **kw):
        text = sum(tok(m.text) + sum(tok(json.dumps(c.args)) + tok(c.name) for c in m.tool_calls)
                   + sum(tok(r.content) for r in m.tool_results) + 6 for m in messages)  # fmt: skip
        images = sum(len(m.images) + sum(len(r.images) for r in m.tool_results) for m in messages)
        schemas = sum(tok(t.name) + tok(t.description) + tok(json.dumps(t.parameters)) + 12 for t in tools or [])
        self.calls.append({"text": text + tok(system) + schemas, "images": images})
        seen = " ".join([m.text for m in messages] + [r.content for m in messages for r in m.tool_results])
        has_pic = any(m.images for m in messages) or any(r.images for m in messages for r in m.tool_results)
        n = len(self.calls)
        if any(c.name in ("ax_press", "click") for m in messages for c in m.tool_calls):
            return Turn(text="Done.")
        shot = re.search(r'shot="?(s\d+-\w+)', seen)
        if self.needs_picture and has_pic and shot:
            return Turn(tool_calls=[ToolCall(f"c{n}", "click", {"x": 450, "y": 650, "shot": shot.group(1)})])
        m = re.search(r'(e\d+) Button "New Page"', seen)
        if not self.needs_picture and m:
            return Turn(tool_calls=[ToolCall(f"c{n}", "ax_press", {"id": m.group(1)})])
        if "App:" not in seen:
            return Turn(tool_calls=[ToolCall(f"c{n}", "ax_tree", {})])
        return Turn(tool_calls=[ToolCall(f"c{n}", "screenshot", {})])


async def _h_tree(a, c):
    return ax.snapshot(a.get("app") or W.front[0])


async def _h_shot(a, c):
    s = pointer.remember(0, 0, 1.0, 1600, 1000)
    return ToolOutput(pointer.note(s, "all of display 1", []), images=[ImagePart(PNG)], fence=False)


async def _h_act(a, c):
    return ToolOutput("Done it. Changed: the screen changed around the point.")


registry().get("ax_tree").handler = _h_tree
registry().get("screenshot").handler = _h_shot
registry().get("ax_press").handler = _h_act
registry().get("click").handler = _h_act
TOOLS = [registry().get(n).spec() for n in ("ax_tree", "screenshot", "ax_press", "click")]


async def scenario(name: str, text: str, d: Decision, front: tuple[str, int], game=None) -> dict:
    W.front = front
    W.walk_s, W.shot_s = 0.0, 0.0
    needs_picture = bool(d.needs_screen)
    out = {}
    for arm in ("today", "first look"):
        brain = Brain(needs_picture)
        goal, images, ctx = text, None, ToolContext(user_text=text)
        if arm == "first look":
            p = prefetch.start(text, d, game=game, sees=True)
            look = await prefetch.collect(p, TOOLS)
            assert look is not None, "the first look should have been taken"
            goal, images = look.attach(goal, None, ctx)
        res = await run_agent(goal, provider=brain, system="s" * 3500, tools=TOOLS, ev=EventBus(), ctx=ctx, images=images)
        assert res.stopped == "done", res.text
        tokens = [c["text"] + min(c["images"], 1) * IMG_TOKENS for c in brain.calls]  # fit keeps the newest picture
        out[arm] = {"round_trips": len(brain.calls), "tools": [s.tool for s in res.steps], "prompt_tokens": round(sum(tokens)),
                    "first_call_tokens": round(tokens[0])}  # fmt: skip
    return {"task": name, **out}


# ---- 2. what the look adds before the first call --------------------------------------------------


async def added_latency(walk_s: float, shot_s: float, route_s: float, image: bool) -> dict:
    W.front, W.walk_s, W.shot_s = ("Tablero", 104), walk_s, shot_s
    d = Decision("agent_task", 0.95, False, 0.0, image, 0.9 if image else 0.1, tool="" if image else "click_text", tool_p=0.6)
    runs, attached = [], 0
    prefetch._slow_until.clear()
    for _ in range(5):
        t0 = time.perf_counter()
        p = prefetch.start("make the first move in tablero", d, sees=True)
        await asyncio.sleep(route_s)  # playbook recall and tool selection, which run anyway
        look = await prefetch.collect(p, TOOLS)
        runs.append((time.perf_counter() - t0 - route_s) * 1000)
        attached += look is not None
    return {"walk_ms": walk_s * 1000, "picture_ms": shot_s * 1000 if image else 0, "routing_ms": route_s * 1000,
            "added_ms_first": round(runs[0]), "added_ms_p50_next4": round(statistics.median(runs[1:])),
            "attached": f"{attached}/5"}  # fmt: skip


# ---- 3. prewarm -----------------------------------------------------------------------------------


def prewarm_bench(load_s: float, pass_s: float, speak_s: float) -> dict:
    import threading

    from neo import config
    from neo.reflex import zeroshot

    os.environ["NEO_REFLEX"] = "laya"
    config._settings = None

    class Fake:
        def system_one(self, state, questions):
            time.sleep(pass_s)
            return {"answers": {q: {"type": "choice", "choice": "A", "probabilities": {"A": 1.0}} for q in questions}}

        def __exit__(self, *e):
            pass

    zeroshot._find = lambda: "/nowhere"
    zeroshot._open = lambda path: (time.sleep(load_s - pass_s), Fake())[1]  # + its warm-up pass = load_s
    q = {"app": {"type": "choice", "instructions": "Which?", "criteria": {"A": "a", "B": "b"}}}
    out = {}
    for arm in ("cold", "prewarmed on wake"):
        zeroshot._agent, zeroshot._loading, zeroshot._used = None, None, 0.0
        prefetch._last_warm = float("-inf")
        if arm != "cold":
            prefetch.prewarm("wake")
        time.sleep(speak_s)  # the user speaks; speech to text
        t0 = time.perf_counter()
        zeroshot.ask({"request": "x"}, q, wait_s=2.0)
        out[arm] = round((time.perf_counter() - t0) * 1000)
        if zeroshot._timer is not None:
            zeroshot._timer.cancel()
    threading.Event().wait(0.01)
    os.environ["NEO_REFLEX"] = "off"
    config._settings = None
    return {"load_ms": load_s * 1000, "speech_ms": speak_s * 1000, **{f"resolver_first_question_ms[{k}]": v for k, v in out.items()}}


async def main() -> None:
    rows = [
        await scenario("AX task: 'in quillpad, press New Page'", "in quillpad, press new page",
                       Decision("agent_task", 0.9, False, 0.0, False, 0.1, tool="click_text", tool_p=0.6), ("Quillpad", 101)),
        await scenario("picture task: a board move in tablero", "make the first move in tablero",
                       Decision("agent_task", 0.9, False, 0.0, True, 0.9), ("Tablero", 104)),
    ]  # fmt: skip
    print("\n1. model round trips and prompt tokens (real agent loop, fake brain following today's prompt)")
    for r in rows:
        a, b = r["today"], r["first look"]
        print(f"   {r['task']}\n     today:      {a['round_trips']} round trips {a['tools']}, {a['prompt_tokens']} prompt tokens"
              f"\n     first look: {b['round_trips']} round trips {b['tools']}, {b['prompt_tokens']} prompt tokens "
              f"(first call {a['first_call_tokens']} → {b['first_call_tokens']})")  # fmt: skip
    print("\n2. added before the first model call (look beside routing; deadline 500 ms)")
    lat = []
    for walk, image in ((0.05, False), (0.15, False), (0.40, False), (0.70, False), (0.15, True), (0.40, True)):
        r = await added_latency(walk, 0.25, 0.03, image)
        lat.append(r)
        print(f"   AX walk {r['walk_ms']:>4.0f} ms, picture {r['picture_ms']:>3.0f} ms, routing {r['routing_ms']:.0f} ms → "
              f"+{r['added_ms_first']} ms first request, +{r['added_ms_p50_next4']} ms next ones, attached {r['attached']}")  # fmt: skip
    print("\n3. the resolver's first question after the user speaks (fake checkpoint at measured timings)")
    pw = [prewarm_bench(0.45, 0.035, s) for s in (1.2, 0.3)] + [prewarm_bench(0.70, 0.035, 1.2)]
    for r in pw:
        print(f"   load {r['load_ms']:.0f} ms, speech {r['speech_ms']:.0f} ms → cold {r['resolver_first_question_ms[cold]']} ms, "
              f"prewarmed {r['resolver_first_question_ms[prewarmed on wake]']} ms")  # fmt: skip
    print("\n" + json.dumps({"round_trips": rows, "added_latency": lat, "prewarm": pw}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
