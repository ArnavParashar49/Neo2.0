"""The per-app flow memory (neo.reflex.uimemory) and the outcome ledger's hooks that feed it: a flow is
kept only when its request settles verified, keyed by bundle id + script + the centred reflex
embedding, replayed step by step with an effect check, and forgotten by the privacy erase. Hermetic:
fake clock, fake steps, fake embeddings."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import stat

import numpy as np
import pytest

from neo.reflex import learn, outcomes, uimemory
from neo.reflex.schema import Decision

_ids = itertools.count(1)
BUNDLE = "org.example.board"  # a made-up bundle id: evidence from the element's process
NEW_GAME = [
    {"kind": "menu", "role": "AXMenuItem", "label": "New Game", "path": ["Game", "New Game"], "bundle": BUNDLE,
     "expect": {"sheet": ["Start", "Cancel"]}, "changed": True},
    {"kind": "press", "role": "AXButton", "label": "Start", "bundle": BUNDLE, "expect": {"sheet_closed": True}, "changed": True},
]  # fmt: skip


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s
        outcomes.tick()


@pytest.fixture
def clock(monkeypatch):
    from neo.tools import load_all

    load_all()
    c = Clock()
    monkeypatch.setattr(outcomes, "_ledger", outcomes.Ledger(clock=c))
    mem = uimemory.UIMemory(clock=c)
    monkeypatch.setattr(uimemory, "_memory", mem)
    return c


def run(text: str, sigs: list[dict], *, ok: bool = True, tool: str = "menu_command") -> str:
    k = f"j{next(_ids)}"
    outcomes.begin(k, text, decision=Decision("quick_action", 0.9, False, 0.0, False, 0.0, "laya+head", tool=tool, tool_p=0.9))
    outcomes.route(k, "quick")
    for s in sigs:
        outcomes.signature(k, s)
    outcomes.step(k, tool, ok)
    outcomes.end(k)
    return k


def test_a_verified_flow_becomes_a_recipe_for_that_app(clock):
    run("start a new game please", NEW_GAME)
    clock.advance(30)
    r = uimemory.recall("start a new game please", BUNDLE)
    assert r is not None and [s.describe() for s in r.steps] == ["Game › New Game", "“Start”"]
    assert r.steps[0].expect == {"sheet": ["Start", "Cancel"]} and r.script == "latin"
    assert uimemory.recall("start a new game please", "org.example.other") is None  # another app


def test_a_failed_corrected_or_stopped_request_teaches_no_flow(clock):
    run("start a new game please", NEW_GAME, ok=False)
    clock.advance(30)
    assert uimemory.recall("start a new game please", BUNDLE) is None
    k = run("start a new game now", NEW_GAME)
    outcomes.stop(k, early=True)
    clock.advance(30)
    assert uimemory.recall("start a new game now", BUNDLE) is None


def test_a_step_that_changed_nothing_or_two_apps_keep_nothing(clock):
    run("start a new game please", [NEW_GAME[0], {**NEW_GAME[1], "changed": False}])
    run("open the game menu and start", [NEW_GAME[0], {**NEW_GAME[1], "bundle": "org.example.other"}])
    clock.advance(30)
    assert len(uimemory.memory()) == 0


def test_only_ui_chrome_labels_are_kept(clock):
    row = {"kind": "press", "role": "AXRow", "label": "Priya Sharma — invoice", "bundle": BUNDLE, "changed": True}
    run("open priya's invoice mail", [row])
    clock.advance(30)
    assert len(uimemory.memory()) == 0
    assert uimemory.Step.of({"kind": "press", "role": "AXStaticText", "label": "x"}) is None
    assert uimemory.Step.of({"kind": "type", "label": "hello"}) is None


def test_the_file_holds_skeleton_text_and_chrome_only_with_owner_only_access(clock):
    run("start a new game please", NEW_GAME)
    clock.advance(30)
    p = learn._path(uimemory.FILE)
    data = json.loads(p.read_text())
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    r = data["recipes"][0]
    assert r["skeleton"] == "start a new game please" and r["bundle"] == BUNDLE
    assert "emb" not in json.dumps(data) and "vec" not in json.dumps(data)  # embeddings never on disk


def test_recall_by_the_centred_reflex_embedding(clock):
    center = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    emb = np.array([1.0, 1.0, 3.0, 1.0], dtype=np.float32)
    uimemory.memory().seen("start a new game please", emb)
    run("start a new game please", NEW_GAME)
    clock.advance(30)
    near = emb + np.array([0.0, 0.05, 0.0, 0.0], dtype=np.float32)
    far = np.array([3.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert uimemory.recall("begin a fresh game", BUNDLE, near, center) is not None
    assert uimemory.recall("resign this game", BUNDLE, far, center) is None
    assert uimemory.recall("नया गेम शुरू करो", BUNDLE, near, center) is None  # another script


def test_learning_off_and_library_mode_keep_nothing(clock, monkeypatch):
    from neo import runtime

    monkeypatch.setattr(uimemory.learn, "enabled", lambda: False)
    assert uimemory.remember("start a new game please", BUNDLE, NEW_GAME) is None
    monkeypatch.setattr(uimemory.learn, "enabled", lambda: True)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    assert uimemory.remember("start a new game please", BUNDLE, NEW_GAME) is None
    assert uimemory.alias("click begin", BUNDLE) is None


def test_old_recipes_go_and_the_cap_holds(clock, monkeypatch):
    monkeypatch.setattr(uimemory, "CAP", 3)
    for i in range(5):
        uimemory.remember(f"start game number {i}", BUNDLE, NEW_GAME)
        clock.t += 1
    assert len(uimemory.memory()) == 3
    clock.t += uimemory.KEEP_S + 10
    fresh = uimemory.UIMemory(clock=clock)
    assert len(fresh) == 0


def test_replay_checks_every_step_and_hands_over_at_the_first_mismatch(clock):
    r = uimemory.remember("start a new game please", BUNDLE, NEW_GAME)
    ran: list[str] = []

    async def run_step(step):
        ran.append(step.describe())
        return {"sheet": ["Start", "Cancel"]} if step.kind == "menu" else {"sheet_closed": False}

    def check(step, got):
        if step.expect.get("sheet"):
            return got.get("sheet") == step.expect["sheet"], f"sheet {got.get('sheet')}"
        return bool(got.get("sheet_closed")), "the sheet still open"

    out = asyncio.run(uimemory.replay(r, run_step, check))
    assert not out.ok and out.done == 1 and ran == ["Game › New Game", "“Start”"]
    assert "Game › New Game" in out.handoff and "sheet_closed=True" in out.handoff and "still open" in out.handoff
    assert asyncio.run(uimemory.replay(r, run_step, check)).ok is False
    assert uimemory.recall("start a new game please", BUNDLE) is None  # two misses in a row: retired


def test_a_replay_that_works_counts(clock):
    r = uimemory.remember("start a new game please", BUNDLE, NEW_GAME)
    out = asyncio.run(uimemory.replay(r, lambda s: "ok", lambda s, g: (True, "")))
    assert out.ok and out.done == 2
    assert uimemory.recall("start a new game please", BUNDLE).ok == 2


def test_a_step_that_raises_hands_over(clock):
    r = uimemory.remember("start a new game please", BUNDLE, NEW_GAME)

    def boom(step):
        raise RuntimeError("the app went away")

    out = asyncio.run(uimemory.replay(r, boom, lambda s, g: (True, "")))
    assert not out.ok and out.done == 0 and "RuntimeError" in out.handoff


def test_a_shortlist_pick_that_was_verified_becomes_an_alias_the_resolver_uses(clock):
    pick = {**NEW_GAME[1], "label": "Begin", "alias": True, "kind": "press"}
    run("kick it off", [pick], tool="click_text")
    clock.advance(30)
    assert uimemory.alias("kick it off", BUNDLE) == "Begin"
    assert uimemory.alias("click kick it off", BUNDLE) == "Begin"  # the frame doesn't matter

    from neo.tools.computer import resolve as rs

    els = [rs.Candidate("Begin", "AXButton", box=(0, 0, 80, 20)), rs.Candidate("Cancel", "AXButton", box=(0, 40, 80, 20))]
    res = rs.resolve("kick it off", els, app=BUNDLE, stages=("lexical", "alias"), index=False, embed=False, log_it=False)
    assert isinstance(res, rs.Resolved) and res.stage == "alias" and res.label == "Begin"
    shadow = rs.resolve("kick it off", els, app=BUNDLE, index=False, embed=False, log_it=False)
    assert isinstance(shadow, rs.Ambiguous) and shadow.shadow["picks"] == {"alias": "Begin"}  # shadow by default


def test_the_privacy_erase_forgets_recipes_and_aliases(clock):
    uimemory.remember("start a new game please", BUNDLE, NEW_GAME)
    uimemory.learn_alias("kick it off", BUNDLE, "Begin")
    assert uimemory.forget() == 2
    assert not learn._path(uimemory.FILE).exists()
    assert uimemory.recall("start a new game please", BUNDLE) is None and uimemory.alias("kick it off", BUNDLE) is None


def test_rebuild_re_embeds_skeletons_in_ram(clock):
    uimemory.remember("start a new game please", BUNDLE, NEW_GAME)
    fresh = uimemory.UIMemory(clock=clock)
    assert len(fresh.stale()) == 1
    assert fresh.rebuild(lambda text: np.ones(4, dtype=np.float32)) == 1 and fresh.stale() == []


def test_signatures_are_capped_and_ignored_without_an_episode(clock):
    outcomes.signature("nobody", NEW_GAME[0])  # no episode: nothing, no error
    k = f"j{next(_ids)}"
    outcomes.begin(k, "start a new game please")
    for _ in range(20):
        outcomes.signature(k, NEW_GAME[0])
    assert len(outcomes.ledger()._eps[k].sigs) == 12


def test_rebuild_soon_waits_for_a_loaded_laya(clock, monkeypatch):
    from neo import reflex

    monkeypatch.setattr(reflex, "_laya", None)
    assert uimemory.rebuild_soon(lambda: True) is False  # never loads Laya for this
