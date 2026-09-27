"""Hermetic before/after numbers for the brain-batch work: model round trips per screen task (the
check turn, several actions per reply, finishing on verified actions, the replay lane) and the
fixed tokens every call pays (the system prompt and the tool list of a screen run).

Nothing leaves the Mac and nothing on it is touched: the brain is a scripted fake that follows
what it is offered (it sends several actions per reply only when the prompt invites it, looks
again after a move only when the result didn't show it, resends what was skipped, calls `finish`
with its last actions only when that is offered), every tool is a fake, NEO_NO_EFFECTS=1 and a
throwaway data dir. Seconds are MODELED: round trips × a published first-token median per reply
(low 1.5 s, high 12 s) — informational only.

    NEO_NO_EFFECTS=1 .venv/bin/python scripts/bench_brain_batch.py                      # this tree
    NEO_NO_EFFECTS=1 NEO_BATCH_ACTIONS=1 .venv/bin/python scripts/bench_brain_batch.py  # batching on
    NEO_NO_EFFECTS=1 PYTHONPATH=<base tree> .venv/bin/python scripts/bench_brain_batch.py  # before

It prints a table and one JSON line; the new code is detected, so one script serves every tree.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("NEO_NO_EFFECTS", "1")
os.environ["NEO_REFLEX"] = "off"
os.environ["NEO_VOICE"] = "off"
os.environ["NEO_DATA_DIR"] = tempfile.mkdtemp(prefix="neo-bench-")
os.environ["NEO_NOTES_DIR"] = os.path.join(os.environ["NEO_DATA_DIR"], "notes")
os.environ.setdefault("NEO_REPLAY_FLOWS", "1")
for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY"):
    os.environ.pop(_k, None)

from neo2.core.config import Settings  # noqa: E402

Settings.model_config["env_file"] = None  # never the user's ~/.neo/.env (their keys): hermetic, like the suite

from neo2.core.agent.registry import ToolOutput, registry  # noqa: E402
from neo2.core.providers.base import ImagePart, ToolCall, Turn  # noqa: E402

try:
    from neo2.core.agent import batch as _batch  # noqa: E402

    Effect: Any = _batch.Effect
except ImportError:  # the base tree: tools may still carry an Effect the loop doesn't read

    @dataclass
    class Effect:  # type: ignore[no-redef]
        changed: bool | None = None
        scene_changed: bool = False
        sheet: str = ""
        looked: bool = False


BYTES_PER_TOKEN = 3.4  # neo2.core.providers.fit's rule
IMG_TOKENS = 1120  # Gemini 3's default per picture
MODEL_S = {"low": 1.5, "high": 12.0}
PNG = ImagePart(b"\x89PNG bench", "image/png")
_BATCH_WORDS = re.compile(r"\b(?:several|multiple|a batch of)\s+(?:actions|tool calls|calls|steps)\s+in\s+one\s+(?:turn|reply|step)", re.I)
_PIXEL = {"x", "y", "shot", "cell"}


def tok(s: str) -> float:
    return len(s.encode("utf-8", "ignore")) / BYTES_PER_TOKEN


def _out(text: str, *, ok: bool = True, effect: Any = None, images: list | None = None) -> ToolOutput:
    o = ToolOutput(text, images=list(images or []), ok=ok, fence=False)
    if effect is not None:
        o.effect = effect  # (ToolOutput.effect once Step 0's seam lands)
    return o


# ---- the scripted brain --------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    utterance: str
    phases: list[list[tuple[str, dict]]]  # actions whose targets are visible together
    outputs: dict[str, Any]  # tool → ToolOutput (or fn(args) → ToolOutput)
    pixel: bool = False  # the moves are at pixels on a picture
    intent: str = "agent_task"
    needs_screen: bool = True
    note: str = ""
    repeat: int = 1  # the same request asked this many times (the replay lane)
    observe: bool = False  # the eyes' one-call look is there (a stand-in for neo2.core.tools.computer.observe)


@dataclass
class Brain:
    sc: Scenario
    calls: int = 0
    tokens: list[float] = field(default_factory=list)
    images: list[int] = field(default_factory=list)
    tools_offered: list[int] = field(default_factory=list)
    schema_tokens: list[float] = field(default_factory=list)
    system_tokens: list[float] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    request: int = -1

    name = "gemini"
    supports_vision = True
    supports_tools = True
    last_used = "gemini"
    last_model = "bench"

    async def complete(self, messages, *, system="", tools=None, effort="medium", **kw) -> Turn:
        self.calls += 1
        specs = list(tools or [])
        sch = sum(tok(t.name) + tok(t.description) + tok(json.dumps(t.parameters, separators=(",", ":"))) + 12 for t in specs)
        hist = sum(6 + tok(m.text) + sum(tok(json.dumps(c.args, default=str)) + tok(c.name) for c in m.tool_calls)
                   + sum(tok(r.content) + 6 for r in m.tool_results) for m in messages)  # fmt: skip
        pics = sum(len(m.images) + sum(len(r.images) for r in m.tool_results) for m in messages)
        self.system_tokens.append(tok(system))
        self.schema_tokens.append(sch)
        self.tools_offered.append(len(specs))
        self.images.append(pics)
        self.tokens.append(tok(system) + sch + hist + min(pics, 1) * IMG_TOKENS)
        return self.policy(messages, system, specs)

    def policy(self, messages, system, specs) -> Turn:
        start = max(i for i, m in enumerate(messages) if m.role == "user" and not m.text.startswith("["))
        if start != self.request:
            self.request = start
            self.state = {"looked": [], "phase": 0, "pending": [], "need_look": False, "checked": False}
        st = self.state
        batch = bool(_BATCH_WORDS.search(system))
        final_ok = batch and any(t.name == "finish" for t in specs)  # offered: end with the last actions
        last = messages[-1]
        if last.role == "user" and last.text.startswith("[verify]"):
            if "NEO looked at the screen" in last.text or st["checked"]:
                return Turn(text="Done.")
            st["checked"] = True
            return self._call("screenshot" if self.sc.pixel else "ax_tree")
        if last.role == "tool":
            for r in last.tool_results:
                if r.content.startswith("skipped:"):
                    st["pending"].append(next(c for m in messages for c in m.tool_calls if c.id == r.call_id))
                elif r.name in ("click", "drag") and not r.images:
                    st["need_look"] = True  # a move whose result didn't show it: look again
                elif r.images:
                    st["need_look"] = False
                if st["checked"] and r.name in ("screenshot", "ax_tree"):
                    return Turn(text="Done.")
        looks = ["ax_tree"] + (["screenshot"] if self.sc.pixel else [])
        for lk in looks:
            if lk not in st["looked"]:
                st["looked"].append(lk)
                st["need_look"] = False
                return self._call(lk)
        if st["need_look"]:
            st["need_look"] = False
            return self._call("screenshot")
        todo: list[ToolCall] = []
        if st["pending"]:
            todo, st["pending"] = st["pending"], []
        elif st["phase"] < len(self.sc.phases):
            todo = [self._tc(n, a) for n, a in self.sc.phases[st["phase"]]]
            st["phase"] += 1
        if not todo:
            return Turn(text="Done.")
        if not batch:
            st["pending"] = todo[1:] + st["pending"]
            todo = todo[:1]
        if final_ok and st["phase"] >= len(self.sc.phases) and not st["pending"]:
            todo.append(self._tc("finish", {"answer": "Done."}))
        return Turn(tool_calls=todo)

    def _tc(self, name, args) -> ToolCall:
        self.state["n"] = self.state.get("n", 0) + 1
        return ToolCall(f"c{self.calls}-{self.state['n']}", name, dict(args))

    def _call(self, name, **args) -> Turn:
        return Turn(tool_calls=[self._tc(name, args)])

    async def stream(self, messages, **kw):
        yield "Done."


# ---- the fake apps ------------------------------------------------------------------------------------

TREE = 'App: Boardy — window "Game 1"\n  e3 TextField "Name"\n  e7 Button "Start"\n  e9 Button "Continue"\n'


def _shot(a):
    return _out('Screenshot s1 attached; give pixel positions with shot="s1".', images=[PNG])


def _click(pictures: bool):
    def f(a):
        eff = Effect(changed=True, looked=pictures)
        return _out("Clicked. Changed: the screen changed around the point.", effect=eff, images=[PNG] if pictures else [])

    return f


SCENARIOS = [
    Scenario(
        "fill_field", "fill the name field with Arnav", [[("ax_set_value", {"id": "e3", "value": "Arnav"})]],
        {"ax_set_value": _out("Set e3 to “Arnav”.")}, note="the value is read back by the tool",
    ),
    Scenario(
        "press_button", "press start in the dialog", [[("ax_press", {"id": "e7"})]],
        {"ax_press": _out("Pressed e7 (Button “Start”). Changed: the element is gone.", effect=Effect(changed=True, scene_changed=False))},
        intent="quick_action", needs_screen=False, note="ax-core's press Effect: the button's dialog closed",
    ),
    Scenario(
        "fill_and_continue", "fill the name with Arnav and continue",
        [[("ax_set_value", {"id": "e3", "value": "Arnav"}), ("ax_press", {"id": "e9"})]],
        {"ax_set_value": _out("Set e3 to “Arnav”."),
         "ax_press": _out("Pressed e9. Changed: the window changed.", effect=Effect(changed=True, scene_changed=False))},
        note="both targets visible in one look",
    ),  # fmt: skip
    Scenario(
        "board_move_after_looks", "move the pawn from e2 to e4", [[("click", {"x": 506, "y": 770, "shot": "s1"}),
                                                                   ("click", {"x": 506, "y": 570, "shot": "s1"})]],
        {"screenshot": _shot, "click": _click(True)}, pixel=True, note="each click's result carries the picture after it (eyes)",
    ),  # fmt: skip
    Scenario(
        "board_move_today", "move the pawn from e2 to e4", [[("click", {"x": 506, "y": 770, "shot": "s1"}),
                                                             ("click", {"x": 506, "y": 570, "shot": "s1"})]],
        {"screenshot": _shot, "click": _click(False)}, pixel=True, note="click results with no picture (today's tools)",
    ),  # fmt: skip
    Scenario(
        "press_label_check_look", "press new game", [[("click_text", {"label": "New Game"})]],
        {"click_text": _out("Pressed “New Game”. Changed: the window changed.", effect=Effect(changed=True))},
        observe=True, note="a press NEO couldn't verify: the check note carries the eyes' look",
    ),
    Scenario(
        "new_game_sheet", "start a new game", [[("hotkey", {"keys": "cmd+n"})], [("ax_press", {"id": "e7"})]],
        {"hotkey": _out("Pressed cmd+n. Changed: a sheet or dialog opened.", effect=Effect(changed=True, scene_changed=True, sheet="a sheet")),
         "ax_press": _out("Pressed e7. Changed: the sheet or dialog closed.", effect=Effect(changed=True, scene_changed=True))},
        note="the sheet must open before its button can be seen",
    ),  # fmt: skip
    Scenario(
        "repeat_command_x3", "start a new game in Boardy",
        [[("activate_app", {"name": "Boardy"})], [("click_text", {"label": "New Game"})]],
        {"activate_app": _out("Boardy is in front.", effect=Effect(changed=True, scene_changed=True)),
         "click_text": _out("Pressed “New Game”. Changed: the window changed.", effect=Effect(changed=True))},
        intent="quick_action", needs_screen=False, repeat=3, note="the same request three times (the replay lane)",
    ),  # fmt: skip
]


async def run_one(sc: Scenario) -> dict:
    from neo2.core.agent import fastpath
    from neo2.core.agent import session as sess
    from neo2.core.memory import embed
    from neo2.core.reflex.schema import Decision
    from neo2.core.tools import load_all

    load_all()
    embed.embed_texts = lambda texts: None  # keyword tool selection (no model loads)
    fastpath.plan = lambda text, *a, **k: None  # the model's route, not a fast path
    sess._game_to_play = lambda text: None
    try:
        from neo2.core.agent import replay

        replay.forget()
        replay.working_app = lambda: "boardy"
    except ImportError:
        pass
    ran: list[str] = []

    def handler(name):
        async def h(a, ctx):
            ran.append(name)
            got = sc.outputs.get(name)
            if callable(got):
                return got(a)
            if got is not None:
                return got
            if name == "ax_tree":
                return TREE
            return _out(f"{name}: done.", effect=Effect(changed=True))

        return h

    for name in ("ax_tree", "screenshot", "click", "ax_press", "ax_set_value", "hotkey", "click_text", "activate_app"):
        registry().get(name).handler = handler(name)
    brain = Brain(sc)
    sess.brain = lambda purpose="agent": brain
    import types

    fake = types.ModuleType("neo2.core.tools.computer.observe")

    async def observe(app=None, *, image=False, deadline_s=1.5, ctx=None):
        ran.append("look(by NEO)")
        return _out("Look (read-only):\n" + TREE, images=[PNG] if image else [])

    fake.observe = observe
    real = sys.modules.get("neo2.core.tools.computer.observe")
    if sc.observe and real is None:
        sys.modules["neo2.core.tools.computer.observe"] = fake
    elif not sc.observe and real is fake:
        sys.modules.pop("neo2.core.tools.computer.observe", None)

    class Reflex:
        async def decide(self, t):
            return Decision(sc.intent, 0.95, False, 0.0, sc.needs_screen, 0.9 if sc.needs_screen else 0.1,
                            "laya+head", reply=False, reply_p=0.1)  # fmt: skip

    s = sess.Session(reflex=Reflex())
    per: list[int] = []
    routes: list[str] = []
    try:
        for _ in range(sc.repeat):
            before = brain.calls
            r = await s.handle(sc.utterance, from_user=True)
            per.append(brain.calls - before)
            routes.append(r.route)
    finally:
        if sys.modules.get("neo2.core.tools.computer.observe") is fake:
            sys.modules.pop("neo2.core.tools.computer.observe", None)
    return {
        "round_trips": brain.calls,
        "per_request": per,
        "routes": routes,
        "tools_run": ran,
        "prompt_tokens": round(sum(brain.tokens)),
        "system_tokens": round(brain.system_tokens[0]) if brain.system_tokens else 0,
        "tool_schema_tokens": round(brain.schema_tokens[0]) if brain.schema_tokens else 0,
        "tools_offered": brain.tools_offered[0] if brain.tools_offered else 0,
        "modeled_s_low": round(brain.calls * MODEL_S["low"], 1),
        "modeled_s_high": round(brain.calls * MODEL_S["high"], 1),
        "note": sc.note,
    }


def main() -> int:
    try:
        from neo2.core.agent import batch

        tree = "brain-batch" + (" (batching on)" if batch.enabled() else " (batching off)")
    except ImportError:
        tree = "base"
    out = {sc.name: asyncio.run(run_one(sc)) for sc in SCENARIOS}
    print(f"## brain-batch bench — {tree}\n")
    print("| task | round trips | per request | tools run | system tok | schema tok (tools) | prompt tok total | modeled s (low/high) |")
    print("|---|---:|---|---|---:|---:|---:|---|")
    for n, r in out.items():
        print(f"| {n} | {r['round_trips']} | {r['per_request']} | {','.join(r['tools_run'])} | {r['system_tokens']} | "
              f"{r['tool_schema_tokens']} ({r['tools_offered']}) | {r['prompt_tokens']} | {r['modeled_s_low']}/{r['modeled_s_high']} |")  # fmt: skip
    print()
    print(json.dumps({"tree": tree, "scenarios": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
