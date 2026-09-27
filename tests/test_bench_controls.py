"""The hermetic controls bench (bench/controls): its fakes count exactly, its clock accounts for every
sleep, its goal checks catch a tool that lies, its report keeps one shape — and, marked ``bench``, the
plan's targets as regression thresholds that switch on as each feature lands (found by looking at
what the code offers, never by a flag). Nothing here posts a real event or calls a model."""

from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from pathlib import Path

import pytest

from bench.controls import fakes as F  # (imported at collection: fakes keeps the real code conftest swaps)
from bench.controls import run as R
from bench.controls import scenarios as S

with warnings.catch_warnings():  # ("bench" is registered in pyproject by the integrator)
    warnings.simplefilter("ignore")
    bench = pytest.mark.bench

BASELINE = Path(__file__).resolve().parents[1] / "bench" / "baselines" / "step0.json"


@pytest.fixture(autouse=True)
def _nothing_real(monkeypatch):
    """Whatever the fakes miss is refused here, before NEO_NO_EFFECTS would be asked."""
    import ApplicationServices as AS
    import Quartz

    def refuse(*a, **k):
        raise AssertionError("a real side effect was attempted")

    for n in ("CGEventPost", "CGEventPostToPid", "CGEventTapCreate", "CGWarpMouseCursorPosition"):
        monkeypatch.setattr(Quartz, n, refuse)
    for n in ("AXUIElementPerformAction", "AXUIElementSetAttributeValue", "AXUIElementCopyElementAtPosition"):
        monkeypatch.setattr(AS, n, refuse)
    yield


def _prim(name: str) -> dict:
    p = next(p for p in R._prims() if p.name == name)
    return asyncio.run(R._one_prim(p))


def _scenario(name: str) -> dict:
    return asyncio.run(R.run_scenario(S.by_name(name)))


# ---- the fakes --------------------------------------------------------------------------------------------


def test_fake_as_counts_every_call_exactly():
    w = F.World()
    b = F.button("OK", (10, 40, 60, 20))
    a = w.add(F.app("Tiny", 7, F.window("W", (0, 30, 200, 100), b)))
    AS = w.AS
    AS.AXUIElementCopyAttributeValue(a.root, "AXRole", None)
    AS.AXUIElementCopyAttributeValue(b, "AXTitle", None)
    AS.AXUIElementCopyActionNames(b, None)
    AS.AXUIElementCopyMultipleAttributeValues(b, ["AXRole", "AXTitle", "AXNope"], 0, None)
    AS.AXUIElementPerformAction(b, "AXPress")
    assert AS.count() == 5
    assert AS.kinds() == {"get AXRole": 1, "get AXTitle": 1, "actions": 1, "multi": 1, "perform": 1}
    err, vals = AS.AXUIElementCopyMultipleAttributeValues(b, ["AXTitle", "AXNope"], 0, None)
    assert err == 0 and vals[0] == "OK" and isinstance(vals[1], F.FakeAXError)
    assert w.pressed("OK")


def test_a_real_ax_tree_through_the_fakes_counts_the_same_every_time():
    runs = [_prim("ax_tree") for _ in range(2)]
    assert runs[0]["ax_calls"] == runs[1]["ax_calls"] > 1000
    assert runs[0]["worked"] and runs[0]["captures"] == 0


def test_the_fake_clock_counts_every_sleep_and_moves_monotonic_on():
    w = S.world(lambda w: S.notes_app())
    with F.install(w) as env:
        t0 = time.monotonic()

        async def go():
            await asyncio.sleep(0.5)
            await asyncio.sleep(0)  # (a yield, not a wait)
            time.sleep(0.25)

        wall = time.perf_counter()
        asyncio.run(go())
        assert time.perf_counter() - wall < 0.3  # nothing really waited
        assert env.clock.slept == pytest.approx(0.75)
        assert [s.kind for s in env.clock.sleeps] == ["async", "sync"]
        assert time.monotonic() - t0 >= 0.75
        assert env.clock.sleeps[0].where.endswith(":go")
    assert asyncio.sleep is F._REAL_ASLEEP and time.sleep is F._REAL_SLEEP  # all put back


def test_wait_until_is_accounted_when_neo_has_it():
    wait = pytest.importorskip("neo.tools.computer.wait", reason="Step 0's wait.until isn't in this tree yet")
    w = S.world(lambda w: S.notes_app())
    with F.install(w) as env:
        got = asyncio.run(wait.until(lambda: False, timeout=0.3, every=0.1))
        assert not got
        assert env.clock.waits == 1
        assert env.clock.slept == pytest.approx(0.3, abs=0.11)


def test_a_primitive_measures_neo_work_only():
    r = _prim("click_front")
    assert r["worked"] and r["ok"]
    assert r["captures"] == 2  # the picture before and after a model's click
    assert r["sleep_s"] > 0 and r["posted"] >= 3
    assert r["py_ms"] < 500  # fakes' own time (rendering the frames) is not NEO's


def test_screenshots_are_synthetic_retina_frames():
    w = S.board_world()
    with F.install(w) as env:
        out = asyncio.run(R._tool("screenshot"))
        assert out.ok and out.images
        assert env.screen.captures == [{"region": None, "fmt": "png", "window": False}]
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(out.images[0].data)).size == (1512, 982)  # NEO's own downscale of 3024×1964


# ---- the scenarios ---------------------------------------------------------------------------------------


def test_the_chess_move_takes_seven_round_trips_on_the_step0_policy():
    r = _scenario("chess_move")
    if R.features()["cell_click"] or R.features()["prefetch_look"]:
        pytest.skip("the board helper or the early look is in: fewer trips (the thresholds below)")
    assert r["pass"], r
    assert r["round_trips"] == 7 and r["tools"] == ["ax_tree", "screenshot", "click", "screenshot", "click", "screenshot"]
    assert set(r["efforts"]) == {"high"} and r["purposes"] == ["vision"]
    assert r["modeled_s"] > r["model_s"] > 0


def test_a_goal_check_fails_when_a_tool_lies(monkeypatch):
    from neo.agent.registry import ToolOutput, registry
    from neo.tools import load_all

    load_all()

    async def liar(a, c):
        return ToolOutput("Clicked. Changed: the board changed.")

    monkeypatch.setattr(registry().get("click"), "handler", liar)
    r = _scenario("chess_move")
    assert not r["pass"] and not r["checks"]["goal"]
    assert r["checks"]["no_unconfirmed_commit"] and r["checks"]["not_neo_window"]


def test_a_blind_brain_never_gets_coordinates_acted_on():
    r = _scenario("blind_chess_move")
    assert r["pass"], r
    assert r["checks"]["no_blind_coords"]
    assert r["neo"]["posted"] == 0


def test_open_and_type_needs_no_model():
    r = _scenario("open_and_type")
    assert r["pass"] and r["round_trips"] == 0 and r["route"] == "quick"


def test_the_scroll_goal_is_where_the_scroll_landed():
    w = S.world(lambda w: S.big_app(), lambda w: S.notes_app(), front="Notes")
    w.acted.append({"kind": "scroll", "app": "Notes"})
    assert S._scrolled("Notes")(w, {}) and not S._scrolled("Safari")(w, {})


# ---- the report ----------------------------------------------------------------------------------------------

_ROW_PRIM = {"ax_calls", "captures", "sleeps", "sleep_s", "ax_block_s", "posted", "brings", "ok", "worked", "said",
             "slept_in", "py_ms_p50", "py_ms_p90", "steady", "modeled_ms", "neo_ms", "target", "note"}  # fmt: skip
_ROW_SCEN = {"utterance", "pass", "checks", "round_trips", "max_calls_per_turn", "efforts", "purposes", "prompt_tokens", "image_tokens",
             "tools_offered", "total_prompt_tokens", "tools", "route", "stopped", "asked", "error", "neo", "neo_s",
             "model_s", "modeled_s", "answer", "targets", "note"}  # fmt: skip
_TOP = {"schema", "sha", "when", "python", "params", "features", "primitives", "scenarios", "resolver_eval", "scores"}


def test_the_report_keeps_one_shape():
    rep = R.build_report(R.measure_primitives(1, ["ax_hit"]), R.run_scenarios(["open_and_type"]), reps=1)
    assert set(rep) == _TOP and rep["schema"] == R.SCHEMA
    assert set(rep["primitives"]["ax_hit"]) == _ROW_PRIM
    assert set(rep["scenarios"]["open_and_type"]) == _ROW_SCEN
    assert set(rep["primitives"]["ax_hit"]["modeled_ms"]) == {"cli", "sck"}
    json.dumps(rep)  # all of it serializable
    text = R.table(rep, rep)
    assert "| ax_hit |" in text and "| open_and_type | PASS |" in text


def test_the_step0_baseline_is_committed_with_the_same_shape():
    rep = json.loads(BASELINE.read_text())
    assert set(rep) == _TOP and rep["schema"] == R.SCHEMA
    assert all(set(r) == _ROW_PRIM for r in rep["primitives"].values())
    assert all(set(r) == _ROW_SCEN for r in rep["scenarios"].values())
    assert set(rep["primitives"]) == {p.name for p in R._prims()}
    assert set(rep["scenarios"]) == {s.name for s in S.SCENARIOS}
    # the numbers design-speed and design-loop-latency measured, reproduced within 1% (recorded as measured)
    assert abs(rep["primitives"]["ax_tree"]["ax_calls"] - 7879) <= 79
    assert abs(rep["primitives"]["click_text"]["ax_calls"] - 4072) <= 41
    assert rep["scenarios"]["chess_move"]["round_trips"] == 7
    assert rep["scenarios"]["chess_full"]["round_trips"] == 11


def test_the_runner_never_reads_the_real_data_folder():
    assert Path(os.environ["NEO_DATA_DIR"]).resolve() != (Path.home() / ".neo").resolve()
    assert R.git_sha() != ""


# ---- thresholds: the plan's targets, on once the feature exists ------------------------------------------------


def _batched_ax(r_name: str) -> dict:
    """Run a primitive and say whether the code read attributes in batches (CopyMultiple)."""
    p = next(p for p in R._prims() if p.name == r_name)
    w = p.build()
    with F.install(w) as env:
        from neo.tools.computer import apps

        env.patches.set(apps, "_target", None)
        asyncio.run(p.act(env, {}))
        kinds = env.AS.kinds()
    return {"calls": env.AS.count(), "batched": any(k.startswith("multi") for k in kinds)}


@bench
@pytest.mark.parametrize("name,limit", [("ax_tree", 1000), ("click_text", 600), ("ax_hit", 4)])
def test_ax_call_budgets(name, limit):
    got = _batched_ax(name)
    if not got["batched"]:
        pytest.skip("Accessibility is still read one attribute at a time (ax-core's batched reads aren't in)")
    assert got["calls"] <= limit


@bench
def test_ax_calls_never_regress_from_step0():
    base = json.loads(BASELINE.read_text())["primitives"]
    for name in ("ax_tree", "click_text", "ax_hit"):
        assert _prim(name)["ax_calls"] <= base[name]["ax_calls"], name


@bench
def test_a_hung_app_costs_at_most_1_5_s():
    from neo.tools.computer import ax

    if getattr(ax, "_AX_TIMEOUT_S", 2.0) > 1.5:
        pytest.skip("the 1.5 s Accessibility cap isn't in yet")
    r = _prim("ax_tree_hung")
    assert r["ax_block_s"] + r["sleep_s"] <= 1.5


@bench
def test_chess_move_round_trips():
    feats = R.features()
    if not feats["cell_click"]:
        pytest.skip("click takes no cell= yet (the board helper)")
    r = _scenario("chess_move")
    assert r["pass"], r
    batched = r["max_calls_per_turn"] > 1  # (only when the prompt allows it: scenarios.may_batch)
    assert r["round_trips"] <= (2 if batched else 3)


@bench
def test_menu_command_and_its_sheet_take_no_model_call():
    if not R.features()["menu_command"]:
        pytest.skip("there's no menu_command tool yet")
    r = _scenario("menu_command")
    assert r["pass"] and r["round_trips"] == 0


@bench
def test_hindi_commit_labels_are_gated():
    from neo.security import policy

    r = _scenario("hindi_commit_gate")
    if not r["pass"] and not policy.commits("भेजें"):
        pytest.xfail("Devanagari commit labels aren't gated yet (0/27 at c769cb0)")
    assert r["pass"], r


def _passed_at_step0() -> set[str]:
    try:
        return {n for n, r in json.loads(BASELINE.read_text())["scenarios"].items() if r["pass"]}
    except (OSError, ValueError, KeyError):
        return set()


@bench
@pytest.mark.parametrize("name", [s.name for s in S.SCENARIOS])
def test_scenario_invariants(name):
    """Never NEO's own window, never coordinates from a brain that can't see, never a commit pressed
    unasked (the Devanagari one aside: above) — and what passed at Step 0 still passes."""
    r = _scenario(name)
    assert r["checks"]["not_neo_window"] and r["checks"]["no_blind_coords"], r
    if name != "hindi_commit_gate":
        assert r["checks"]["no_unconfirmed_commit"], r
    if name in _passed_at_step0():
        assert r["pass"], r


@bench
@pytest.mark.skipif(os.environ.get("NEO_BENCH_TIMING") != "1", reason="wall-clock targets: NEO_BENCH_TIMING=1 on a quiet Mac")
@pytest.mark.parametrize("name", sorted(R.TARGETS))
def test_primitive_meets_its_p0_target(name):
    r = R.measure_primitives(3, [name])[name]
    assert r["neo_ms"] <= R.TARGETS[name]["p0"]
