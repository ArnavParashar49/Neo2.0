"""The hermetic controls bench (bench/controls): its fakes count exactly, its clock accounts for every
sleep, its goal checks catch a tool that lies, its report keeps one shape — and, marked ``bench``, the
plan's targets as regression thresholds that switch on as each feature lands (found by looking at
what the code offers, never by a flag). Nothing here posts a real event or calls a model."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import textwrap
import time
import types
import warnings
from pathlib import Path

import pytest
from PIL import Image

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
    # The bench measures NEO.app's defaults, as `python -m bench.controls.run` does: the suite's
    # conftest turns the early first look off (it would read this Mac's screen); here the Mac is fake.
    from neo2.core import config

    monkeypatch.delenv("NEO_PREFETCH_LOOK", raising=False)
    monkeypatch.setattr(config, "_settings", None)
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
    # (the AX core reads in batches and only what shows: ~290 calls on the 2,522-element app, 7,879 at Step 0)
    assert runs[0]["ax_calls"] == runs[1]["ax_calls"] > 100
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
    wait = pytest.importorskip("neo2.core.tools.computer.wait", reason="Step 0's wait.until isn't in this tree yet")
    w = S.world(lambda w: S.notes_app())
    with F.install(w) as env:
        got = asyncio.run(wait.until(lambda: False, timeout=0.3, every=0.1))
        assert not got
        assert env.clock.waits == 1
        assert env.clock.slept == pytest.approx(0.3, abs=0.11)


def test_a_primitive_measures_neo_work_only(monkeypatch):
    r = _prim("click_front")
    assert r["worked"] and r["ok"]
    assert r["captures"] == 2  # the picture before and after a model's click
    assert r["sleep_s"] > 0 and r["posted"] >= 3
    # The fakes' own time (rendering two Retina frames, hit tests) is measured by the Meter and taken
    # out of NEO's: what is left is the wall time less exactly that.
    assert r["fake_ms"] > 0 and r["py_ms"] < r["wall_ms"]
    assert r["py_ms"] == pytest.approx(r["wall_ms"] - r["fake_ms"], abs=0.01)

    @contextlib.contextmanager
    def nothing():
        yield

    monkeypatch.setattr(F.METER, "fake", nothing)  # the Meter off: nothing is subtracted
    off = _prim("click_front")
    assert off["fake_ms"] == 0 and off["py_ms"] == pytest.approx(off["wall_ms"], abs=0.01)


def test_screenshots_are_synthetic_retina_frames():
    w = S.board_world()
    with F.install(w) as env:
        out = asyncio.run(R._tool("screenshot"))
        assert out.ok and out.images
        # (screencapture writes a TIFF: the quickest to write and read back — eyes-capture measured it)
        assert env.screen.captures == [{"region": None, "fmt": "tiff", "window": False, "in_process": False}]
    assert Image.open(io.BytesIO(out.images[0].data)).size == (1512, 982)  # NEO's own downscale of 3024×1964


def test_an_in_process_capture_is_counted_and_charged_as_one(monkeypatch, tmp_path):
    """A capture path a workstream registers in CAPTURE_HOOKS (ScreenCaptureKit, in-process) is answered
    by the fake screen, counted like a screencapture run, and charged ScreenCaptureKit's cost — never
    free. An adapter turns the fake's picture into what the real function returns."""
    (tmp_path / "bench_sck_standin.py").write_text(
        "def capture(region=None):\n    raise AssertionError('a real capture')\n\n"
        "def capture_png(region=None):\n    raise AssertionError('a real capture')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import bench_sck_standin as sck

    def png(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(F, "CAPTURE_HOOKS", [("bench_sck_standin", "capture"), ("bench_sck_standin", "capture_png", png)])
    w = S.board_world()
    with F.install(w) as env:
        m = env.mark()
        img = sck.capture(region=(0, 0, 100, 100))
        data = sck.capture_png((10, 10, 50, 20))
        got = env.since(m)
        assert img.size == (200, 200)  # the region, at 2 px per point
        assert data.startswith(b"\x89PNG") and Image.open(io.BytesIO(data)).size == (100, 40)
        assert got["captures"] == 2 and got["captures_in_process"] == 2
        assert env.screen.captures[-1] == {"region": (10, 10, 50, 20), "fmt": "sck", "window": False, "in_process": True}
    assert sck.capture.__name__ == "capture"  # (put back)
    none = {"py_ms": 0.0, "ax_calls": 0, "sleep_s": 0.0, "ax_block_s": 0.0}
    sck_ms = F.CAPTURE_S["sck"] * 1000
    cli_ms = F.CAPTURE_S["cli"] * 1000
    assert R._modeled({**none, "captures": 2, "captures_in_process": 2}, 0.15) == pytest.approx(2 * sck_ms)
    assert R._modeled({**none, "captures": 3, "captures_in_process": 1}, 0.15) == pytest.approx(2 * cli_ms + sck_ms)
    assert R._modeled({**none, "captures": 3, "captures_in_process": 1}, 0.15, every=F.CAPTURE_S["sck"]) == pytest.approx(3 * sck_ms)


# ---- the scenarios ---------------------------------------------------------------------------------------


def test_the_chess_move_takes_seven_round_trips_on_the_step0_policy():
    base = json.loads(BASELINE.read_text())
    if R.features() != base["features"]:
        pytest.skip("the code offers something Step 0 didn't (features differ from the baseline's): the thresholds below")
    r = _scenario("chess_move")
    assert r["pass"], r
    assert r["round_trips"] == 7 and r["tools"] == ["ax_tree", "screenshot", "click", "screenshot", "click", "screenshot"]
    assert r["tools"] == base["scenarios"]["chess_move"]["tools"]
    assert r["modeled_s"] > r["model_s"] > 0


def test_a_goal_check_fails_when_a_tool_lies(monkeypatch):
    from neo2.core.agent.registry import ToolOutput, registry
    from neo2.core.tools import load_all

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
    assert r["checks"]["no_blind_coords"] and r["checks"]["blind_given_no_coords"]
    assert r["neo"]["posted"] == 0


def test_blind_coordinates_that_act_through_accessibility_are_caught(monkeypatch):
    """A blind brain's x/y that acts with no event posted (a click that snaps the point to the element
    there and presses it by Accessibility) fails the blind check; so do the frames it was handed."""
    from neo2.core.agent.registry import registry
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import ax

    load_all()

    async def snapping_click(a, c):
        h = await asyncio.to_thread(ax.hit, float(a["x"]), float(a["y"]))
        el = ax.Element("e0", h.role, h.label, "", 0, 0, 0, 0, actions=["AXPress"], ref=h.el)
        return await asyncio.to_thread(ax.press_el, el)

    monkeypatch.setattr(registry().get("click"), "handler", snapping_click)

    def policy(w):
        def p(b, msgs, system, tools):
            if not b.state.get("tree"):
                b.state["tree"] = True
                return S.turn(S.call("ax_tree"))
            if not b.state.get("clicked"):
                b.state["clicked"] = True
                # (NEO hands a blind brain no frames: these numbers come from somewhere else — made up)
                start = w.named("Chess").root.find(lambda e: e.attrs.get("AXTitle") == "Start")
                x, y, ww, hh = start.frame
                return S.turn(S.call("click", x=x + ww / 2, y=y + hh / 2))
            return S.say("done")

        return p

    # (words that name no control: NEO's own by-utterance lane would press "Start" for "press start")
    sc = S.Scenario("blind_snap", "carry on with the dialog", lambda: S.board_world(sheet=True), policy,
                    lambda w, run: not w.pressed("Start"), sees=False, needs_screen=True)  # fmt: skip
    r = asyncio.run(R.run_scenario(sc))
    assert r["neo"]["posted"] == 0  # (nothing posted: the old check saw nothing)
    assert not r["checks"]["no_blind_coords"] and not r["pass"]
    assert r["checks"]["blind_given_no_coords"], r["blind_given"]  # (the tree it got had ids, no frames)


def test_frames_handed_to_a_blind_brain_are_caught():
    given = R.coords_in('<untrusted source="ax_tree">\n  e10 Button "Start" @(930,238 100x28)\n</untrusted>')
    assert given and "@(" in given[0]


@pytest.mark.parametrize("text", [
    'e8 Button "Start" @(930,238 100x28)', "box_2d: [120, 40, 180, 90]", "clicked x=506, y=770", "at (506, 770)",
    "nx=0.42 ny=0.61", "moved it at 506, 770", "Error: nothing at (12.5, -3)",
])  # fmt: skip
def test_numeric_geometry_is_found_in_what_a_blind_brain_is_handed(text):
    assert R.coords_in(text)


@pytest.mark.parametrize("text", [
    "Error: you can't see the screen — this model has no vision, so no screenshot ever reached you — and I "
    "won't click at coordinates nobody saw. Use element ids from ax_tree (click id=…, ax_press or click_text).",
    'e8 Button "Start"', "Pressed e12.", "cell e4", "step 3 of 5", "a 1512×982 display", "Typed 2 line(s).",
    "Moved e2 to e4.", "mark 7 (Start)", "a tree of 2,522 elements", "(1 of 3)",
])  # fmt: skip
def test_ids_and_words_are_not_coordinates(text):
    assert not R.coords_in(text)


def test_open_and_type_needs_no_model():
    r = _scenario("open_and_type")
    assert r["pass"] and r["round_trips"] == 0 and r["route"] == "quick"


def test_the_scroll_goal_is_where_the_scroll_landed():
    w = S.world(lambda w: S.big_app(), lambda w: S.notes_app(), front="Notes")
    w.acted.append({"kind": "scroll", "app": "Notes"})
    assert S._scrolled("Notes")(w, {}) and not S._scrolled("Safari")(w, {})


def test_the_covered_scroll_passes_only_by_scrolling_safari_or_saying_what_covers_it():
    goal = S.by_name("scroll_covered_app").goal
    w = S.world(lambda w: S.big_app(), lambda w: S.notes_app(), front="Notes")
    covered = "Error: Notes covers that part of Safari's page."
    assert not goal(w, {"answer": "Scrolled down in Safari.", "results": []})  # nothing scrolled, nothing said
    assert not goal(w, {"answer": "Done.", "results": [("scroll", "Error: that spot isn't scrollable.")]})  # never says why
    assert not goal(w, {"answer": "Done.", "results": [("ax_tree", "Notes"), ("scroll", "Scrolled.")]})  # (a tree isn't a reason)
    assert goal(w, {"answer": "Done.", "results": [("scroll", covered)]})
    assert goal(w, {"answer": covered, "results": []})
    w.acted.append({"kind": "scroll", "app": "Safari"})
    assert goal(w, {"answer": "Scrolled down in Safari.", "results": []})
    w.acted.append({"kind": "scroll", "app": "Notes"})
    assert not goal(w, {"answer": covered, "results": [("scroll", covered)]})  # never Notes scrolled


def test_the_scripted_menu_command_names_its_command_argument():
    def spec(*names):
        props = dict.fromkeys(names, {"type": "string"})
        return types.SimpleNamespace(name="menu_command", description="", parameters={"type": "object", "properties": props})

    assert S.menu_arg({"app", "confirm", "item"}) == "item"  # not whichever sorts first
    assert S.menu_arg({"app", "command", "item"}) == "command"
    assert S.menu_arg({"app", "zzz"}) == "zzz"  # (none of the known names: the one there is)
    b = F.FakeBrain(lambda *a: None)
    policy = S.menu_policy()
    policy(b, [], "", [spec("app", "confirm", "item")])  # (its first look)
    got = policy(b, [], "", [spec("app", "confirm", "item")])
    assert got.tool_calls[0].name == "menu_command" and got.tool_calls[0].args == {"item": "New Game", "app": "Chess"}


def test_a_models_click_on_another_app_never_brings_it_forward(monkeypatch):
    """The self-test's watch (neo2/core/selftest/watch.py), end to end on the bench's pretend Mac: the
    self-test app in front, a model's click at a point on the user's other app's window. The watch
    refuses before that app is activated (it used to refuse only at the window raise or the mouse-down,
    after the activation)."""
    from neo2.core.agent.registry import ToolContext, registry
    from neo2.core.selftest import watch as watch_mod
    from neo2.core.tools.computer import apps
    from neo2.core.tools.computer import input as inp

    class Mac:
        def __init__(self, w):
            self.w = w

        def front(self):
            a = self.w.apps.get(self.w.front_pid)
            return (a.name, a.pid) if a else ("", 0)

        def pointer(self):
            return self.w.pointer

        def windows(self):
            out = []
            for d in self.w.window_list():
                b = d["kCGWindowBounds"]
                out.append({"pid": d["kCGWindowOwnerPID"], "name": d["kCGWindowOwnerName"], "number": d["kCGWindowNumber"],
                            "alpha": 1.0, "bounds": (b["X"], b["Y"], b["Width"], b["Height"])})  # fmt: skip
            return out

        def window(self, n):
            return next((x for x in self.windows() if x["number"] == n), None)

        def hit_pid(self, x, y):
            el = self.w.element_at(x, y)
            return el.pid if el is not None else 0

        def buttons_down(self):
            return False

        def element_pid(self, ref):
            return getattr(ref, "pid", 0)

        def post(self, ev):
            self.w.post(ev)

    w = S.world(lambda w: S.big_app(), lambda w: S.board_app(w, name="SelfTest"), front="SelfTest")
    test = w.named("SelfTest")
    win = test.windows()[0]
    with F.install(w) as env:
        env.patches.set(apps, "_target", None)
        watch = watch_mod.Watch(Mac(w), test.pid, {w.window_numbers[id(win)]: win.frame})
        watch.mode = "pointer"
        monkeypatch.setattr(inp, "_watch", watch)
        watch._install_ax()
        try:
            asyncio.run(registry().invoke("click", {"x": 200, "y": 400}, ToolContext(user_text="click at 200, 400", origin="agent")))
        except watch_mod.Stop:
            pass
        finally:
            watch._uninstall_ax()
    assert w.brings == [] and w.front_pid == test.pid  # the user's app stayed where it was
    assert "brought another app forward" in watch.refused and not w.posted


# ---- the report ----------------------------------------------------------------------------------------------

_ROW_PRIM = {"ax_calls", "captures", "captures_in_process", "sleeps", "sleep_s", "ax_block_s", "posted", "brings", "ok", "worked", "said",
             "slept_in", "py_ms_p50", "py_ms_p90", "steady", "modeled_ms", "neo_ms", "target", "note"}  # fmt: skip
_ROW_SCEN = {"utterance", "pass", "checks", "round_trips", "max_calls_per_turn", "efforts", "purposes", "prompt_tokens", "image_tokens",
             "tools_offered", "total_prompt_tokens", "tools", "route", "stopped", "asked", "error", "neo", "neo_s",
             "model_s", "modeled_s", "answer", "targets", "note", "blind_given"}  # fmt: skip
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


def _row(passed: bool, trips: int, *, asked: bool = False) -> dict:
    """A task's report row, as run_scenario writes it (what table() reads)."""
    return {"pass": passed, "asked": asked, "round_trips": trips, "efforts": ["low"] * trips, "checks": {"goal": passed},
            "error": "", "total_prompt_tokens": 0, "neo_s": 0.1, "modeled_s": 0.1 + 1.5 * trips}  # fmt: skip


def test_a_task_that_stops_to_ask_is_not_counted_as_done():
    """A scenario that passes by asking the user (NEEDS_CONFIRM; the bench never answers) is a user
    turn, not a finished task: counted apart from the tasks done, and its trips are never compared
    with a run that ended another way (did the task, pressed without asking, failed)."""
    base = {"did": _row(True, 5), "asks": _row(True, 5), "gate": _row(False, 5), "same": _row(True, 3), "fixed": _row(False, 3)}
    now = {"did": _row(True, 0), "asks": _row(True, 0, asked=True), "gate": _row(True, 0, asked=True), "same": _row(True, 1),
           "fixed": _row(True, 5)}  # fmt: skip
    sc = R.scores({}, now)
    assert sc["tasks_passed"] == "5/5" and sc["tasks_done"] == "3/5" and sc["tasks_asked"] == 2
    assert sc["round_trips"] == 6  # (every task's, as before)
    rep = {"sha": "x", "when": "now", "params": {}, "primitives": {}, "scenarios": now, "scores": sc, "features": {}}
    text = R.table(rep, {"scenarios": base})
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines() if line.startswith("| ")}
    assert rows["did"][1] == "PASS" and rows["did"][7] == "-5"
    assert rows["asks"][1] == "ASKED" and rows["asks"][7] == "" and rows["asks"][8] == ""  # (asked: no Δ)
    assert rows["gate"][1] == "ASKED" and rows["gate"][7] == ""
    assert rows["fixed"][1] == "PASS" and rows["fixed"][7] == ""  # (failed before: no Δ either)
    assert rows["same"][7] == "-2"
    assert "3/5 done, 2 stopped to ask the user" in text
    # the total that compares: the tasks that ended the same way in both runs (did, same)
    assert "2 tasks that ended the same way in both runs: 1 round trips vs 8" in text
    alone = R.table(rep)  # (no base: nothing to compare)
    assert "ended the same way" not in alone and "3/5 done, 2 stopped to ask the user" in alone


def test_the_step0_baseline_is_committed_with_the_same_shape():
    rep = json.loads(BASELINE.read_text())
    assert set(rep) == _TOP and rep["schema"] == R.SCHEMA
    assert all(set(r) == _ROW_PRIM for r in rep["primitives"].values())
    assert all(set(r) == _ROW_SCEN for r in rep["scenarios"].values())
    assert set(rep["primitives"]) == {p.name for p in R._prims()}
    assert set(rep["scenarios"]) == {s.name for s in S.SCENARIOS}
    assert set(rep["features"]) == set(R.features())  # (the Step-0 pin compares them)
    # the numbers design-speed and design-loop-latency measured, reproduced within 1% (recorded as measured)
    assert abs(rep["primitives"]["ax_tree"]["ax_calls"] - 7879) <= 79
    assert abs(rep["primitives"]["click_text"]["ax_calls"] - 4072) <= 41
    assert rep["scenarios"]["chess_move"]["round_trips"] == 7
    assert rep["scenarios"]["chess_full"]["round_trips"] == 11


def test_the_runner_never_opens_anything_in_the_users_neo_folder(tmp_path):
    """``python -m bench.controls.run`` and the resolver eval, as a developer runs them (no pytest
    conftest): not a file under ~/.neo is opened — not even the .env the settings loader would read and
    throw away. A fake HOME with a canary key, and an audit hook that lists every open under it."""
    home = tmp_path / "home"
    (home / ".neo").mkdir(parents=True)
    (home / ".neo" / ".env").write_text("CEREBRAS_API_KEY=canary\nGEMINI_API_KEY=canary\n")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"text": "click Start", "elements": [["AXButton", "Start", ""]], "gold": 0}) + "\n")
    code = textwrap.dedent("""
        import os, sys
        neo = os.path.realpath(os.path.expanduser("~/.neo"))
        assert neo.startswith(sys.argv[4]), neo  # (the fake one)
        opened = []

        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                p = os.path.realpath(os.fsdecode(args[0]))
                if p == neo or p.startswith(neo + os.sep):
                    opened.append(p)

        sys.addaudithook(hook)
        sys.path.insert(0, sys.argv[1])
        from bench.controls import resolver_eval, run
        run.main(["--only", "primitives", "--names", "ax_hit", "--reps", "1", "--out", sys.argv[2]])
        resolver_eval.main(["--data", sys.argv[3]])
        from neo2.core.config import settings
        print("KEY", repr(settings().gemini_api_key))
        print("OPENED", sorted(set(opened)))
    """)
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "HOME": str(home), "NEO_NO_EFFECTS": "1"}
    for k in ("NEO_DATA_DIR", "NEO_NOTES_DIR", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
        env.pop(k, None)
    from neo2.core import effects

    with effects.allow("process"):  # a Python child, meant here: it runs under NEO_NO_EFFECTS=1 itself
        out = subprocess.run([sys.executable, "-c", code, root, str(tmp_path / "rep.json"), str(rows), str(home.resolve())],
                             cwd=root, env=env, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OPENED []" in out.stdout, out.stdout.splitlines()[-2:]
    assert "KEY ''" in out.stdout  # (and the canary never reached the settings)
    assert R.git_sha() != ""


# ---- thresholds: the plan's targets, on once the feature exists ------------------------------------------------


def _batched_ax(r_name: str) -> dict:
    """Run a primitive and say whether the code read attributes in batches (CopyMultiple)."""
    p = next(p for p in R._prims() if p.name == r_name)
    w = p.build()
    with F.install(w) as env:
        from neo2.core.tools.computer import apps

        env.patches.set(apps, "_target", None)
        asyncio.run(p.act(env, {}))
        kinds = env.AS.kinds()
    return {"calls": env.AS.count(), "batched": any(k.startswith("multi") for k in kinds)}


@bench
@pytest.mark.parametrize("name,limit", [
    ("ax_tree", 1000), ("click_text", 600),
    pytest.param("ax_hit", 4, marks=pytest.mark.xfail(strict=True, reason=(
        "ax.hit takes 8 AX calls (the element, its attributes, its parent's label, its window, the app's "
        "focused window and a sheet check) against the plan's 4: open (ax-core's deferred item)"))),
])  # fmt: skip
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
    from neo2.core.tools.computer import ax

    if getattr(ax, "_AX_TIMEOUT_S", 2.0) > 1.5:
        pytest.skip("the 1.5 s Accessibility cap isn't in yet")
    r = _prim("ax_tree_hung")
    assert r["ax_block_s"] + r["sleep_s"] <= 1.5


@bench
def test_chess_move_round_trips():
    limit = R.chess_trips_limit(R.features())
    if limit is None:
        pytest.skip("the board helper (click cell=) and the after-look picture aren't both in yet")
    r = _scenario("chess_move")
    assert r["pass"], r
    assert r["round_trips"] <= limit


def _words_only(monkeypatch):
    """A click whose result is words only: no after-look picture (as at Step 0)."""
    import dataclasses

    from neo2.core.agent.registry import ToolOutput, registry
    from neo2.core.tools import load_all

    load_all()
    click = registry().get("click")
    real = click.handler

    async def words(a, c):
        out = await real(a, c)
        out = out if isinstance(out, ToolOutput) else ToolOutput(str(out))
        out.images = []
        if out.effect is not None:
            out.effect = dataclasses.replace(out.effect, looked=False)
        return out

    monkeypatch.setitem(registry()._tools, "click", dataclasses.replace(click, handler=words))


def test_the_chess_threshold_waits_for_the_after_look_too(monkeypatch):
    """Squares by name alone (click cell=), with the board in the request's first look, leave the
    scripted brain a checking screenshot after its moves: four round trips (two clicks, a
    screenshot, the answer). The ≤3 threshold switches on only once click results carry the
    after-look picture as well; ≤2 once the prompt also allows several actions in one turn; one more
    without the early first look."""
    _words_only(monkeypatch)
    feats = R.features()
    assert feats["cell_click"] and not feats["after_look"] and feats["prefetch_look"]
    assert R.chess_trips_limit(feats) is None  # (so test_chess_move_round_trips would skip)
    r = _scenario("chess_move")
    assert r["pass"] and r["round_trips"] == 4, r
    assert r["tools"] == ["click", "click", "screenshot"]
    assert R.chess_trips_limit({**feats, "after_look": True}) == 3
    assert R.chess_trips_limit({**feats, "after_look": True, "batching": True}) == 2
    assert R.chess_trips_limit({**feats, "after_look": True, "prefetch_look": False}) == 4


def test_the_chess_move_starts_from_the_boards_squares_in_its_first_look():
    """The move is in the game NEO last worked in: the request comes with NEO's first look at it —
    the board's squares by name — so the first model call names the square it moves (no screenshot,
    no ax_tree first)."""
    r = _scenario("chess_move")
    assert r["pass"] and r["tools"][0] == "click" and "screenshot" not in r["tools"], r


def test_after_look_is_found_by_what_a_models_click_returns(monkeypatch):
    assert R.features()["after_look"]  # (a model's click at a picture's pixel comes back with the picture after it)
    _words_only(monkeypatch)
    assert not R.features()["after_look"]


@bench
def test_menu_command_and_its_sheet_take_no_model_call():
    feats = R.features()
    if not R.menu_zero_ready(feats):
        pytest.skip("menu_command isn't a tool NEO runs from the user's words alone yet (by_utterance)")
    r = _scenario("menu_command")
    assert r["pass"] and r["round_trips"] == 0, r
    assert r["route"] == "quick" and r["tools"] == ["menu_command"], r


def test_the_menu_threshold_needs_the_by_utterance_lane():
    feats = {"menu_command": True, "menu_by_utterance": False, "resolver": "shadow"}
    assert not R.menu_zero_ready(feats)  # a menu_command the model must call is still a round trip
    # "New Game" is in "start a new game": the live lexical stage grounds it, with the meaning stage
    # in shadow and nothing promoted (DECISIONS 40)
    assert R.menu_zero_ready({**feats, "menu_by_utterance": True})
    assert R.menu_zero_ready({**feats, "menu_by_utterance": True, "resolver_promoted": ""})
    assert not R.menu_zero_ready({**feats, "menu_by_utterance": True, "resolver": "off"})  # (the lane shut)
    assert not R.menu_zero_ready({**feats, "menu_command": False, "menu_by_utterance": True})


def test_the_menu_scenario_models_the_reflexs_pick_of_menu_command():
    """Laya's tool head routes a menu command to menu_command (the tool lane the plan built for it);
    the bench's scripted reflex says so for this one scenario. Without that pick the words point at
    no control ("click X"), so a model is asked: the round trips come from the bench's reflex, not
    from the resolver's shadow policy."""
    import dataclasses

    from neo2.core.config import settings

    sc = S.by_name("menu_command")
    assert sc.tool == "menu_command" and sc.tool_p >= settings().reflex_tool_floor
    assert all(not s.tool for s in S.SCENARIOS if s.name != "menu_command")  # (the others: as before)
    r = asyncio.run(R.run_scenario(dataclasses.replace(sc, tool="", tool_p=0.0)))
    assert r["pass"] and r["round_trips"] > 0 and r["route"] != "quick", r


def _blind_brains_are_handed_no_coordinates() -> None:
    for sc in [s for s in S.SCENARIOS if not s.sees]:
        r = _scenario(sc.name)
        assert r["checks"]["blind_given_no_coords"], (sc.name, r["blind_given"])


@bench
def test_a_blind_brain_is_handed_no_coordinates():
    """DECISIONS 43: a brain that can't see never gets coordinates in any form — no request, look or
    tool result it is handed carries any (an invariant: no escape while a feature is missing)."""
    _blind_brains_are_handed_no_coordinates()


def test_a_tree_with_frames_handed_to_a_blind_brain_fails_the_guard(monkeypatch):
    """The regression the guard is for: an ax_tree that keeps its frames for a blind brain. The
    guard fails on it — it never xfails or skips."""
    from neo2.core.tools.computer import ax

    real = ax.snapshot

    def leaky(*a, **k):
        k["blind"] = False
        return real(*a, **k)

    monkeypatch.setattr(ax, "snapshot", leaky)
    try:
        _blind_brains_are_handed_no_coordinates()
    except AssertionError as e:
        assert "@(" in str(e), e  # (the tree's frames, found in what the blind brain was handed)
    except (pytest.xfail.Exception, pytest.skip.Exception) as e:
        pytest.fail(f"the guard let a leak through as {type(e).__name__}: {e}")
    else:
        pytest.fail("a blind brain was handed a tree with frames and the guard passed")


def test_marks_are_found_by_what_a_screenshot_lists():
    """eyes-marks landed as a list of the picture's controls by id (click id=…), not a click
    parameter: the feature is found in what a screenshot returns."""
    assert R.features()["marks"]


@bench
def test_hindi_commit_labels_are_gated():
    from neo2.core.security import policy

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


# ---- the resolver eval (opt-in script; its scoring is tested here) ---------------------------------------------


def test_the_resolver_eval_scores_coverage_precision_and_commit_presses(tmp_path):
    from bench.controls import resolver_eval as E

    rows = [
        {"text": "click Start", "elements": [["AXButton", "Start", "sheet"], ["AXButton", "Cancel", "sheet"]], "gold": 0},
        {"text": "start wala button dabao", "elements": [["AXButton", "Start", ""], ["AXButton", "Stop", ""]], "gold": "Start"},
        {"q": "गुलाबी बटन दबाओ", "screen": [{"role": "AXButton", "label": "गुलाबी"}, {"role": "AXButton", "label": "Alpha"}], "target": 0},
        {"text": "send it", "elements": [{"role": "AXButton", "label": "Send", "commit": True}], "gold": 0, "n": 50},
        {"text": "press OK", "elements": [["AXButton", "OK", ""], ["AXButton", "OK", ""]], "gold": [0, 1]},
    ]  # fmt: skip
    data = tmp_path / "ground_eval.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    got = E.load_rows(data)
    assert [r["script"] for r in got] == ["latin", "latin", "devanagari", "latin", "latin"]
    assert got[1]["gold"] == {0} and got[3]["n"] == 50
    base = E.evaluate(got, E.lexical)
    # exact labels (verb and "button" stripped, Hinglish and Hindi frames too); never an ambiguous one
    assert base["overall"]["acted"] == 3 and base["overall"]["precision"] == 1.0 and base["overall"]["coverage"] == 0.6
    assert base["by_script"]["devanagari"]["acted"] == 1 and base["commit_presses"] == 0

    def reckless(text, els):
        return {"index": 0, "stage": "minilm+laya", "p": 0.9}

    bad = E.evaluate(got, reckless)
    assert bad["commit_presses"] == 1 and not bad["gates"]["no_commit_presses"]
    assert bad["by_stage"]["minilm+laya"]["acted"] == 5 and bad["by_window"]["50"]["acted"] == 1
    assert bad["ms"]["cold_first"] is not None and bad["ms"]["p90"] is not None
    assert E.find_resolver("")[0] in (*E.RESOLVERS, "lexical (built-in)")


def test_the_resolver_eval_script_does_nothing_without_data(tmp_path, capsys):
    from bench.controls import resolver_eval as E

    assert E.main(["--data", str(tmp_path / "missing.jsonl")]) == 0
    assert "nothing to do" in capsys.readouterr().err


def test_the_resolver_under_test_never_sees_the_commit_answer_key(tmp_path):
    """The rows' gold ``commit`` flag scores commit presses; the resolver has to tell a commit label
    itself (a real one has no such flag), so it is never handed it."""
    from bench.controls import resolver_eval as E

    rows = [{"text": "press the first one", "elements": [{"role": "AXButton", "label": "Send", "commit": True},
                                                          {"role": "AXButton", "label": "Save draft"}], "gold": 1}]  # fmt: skip
    data = tmp_path / "g.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    seen = []

    def cheat(text, els):
        seen.append([sorted(e) for e in els])
        return next((i for i, e in enumerate(els) if e.get("commit") is False), 0)

    got = E.load_rows(data)
    out = E.evaluate(got, cheat)
    assert seen == [[["context", "label", "role"], ["context", "label", "role"]]]
    assert out["commit_presses"] == 1 and not out["gates"]["no_commit_presses"]  # scored from the key it never saw
    assert got[0]["elements"][0]["commit"]  # (the rows keep it, for scoring)


def test_a_move_on_a_board_takes_one_picture_before_and_one_after():
    """A model's click on a board's square: the look before it (with the board's check before the
    move) is one capture, and the look after it — the change check, the after-look picture and the
    board's check after the move — is one more."""
    from neo2.core.tools.computer import apps

    w = S.board_world()
    with F.install(w) as env:
        env.patches.set(apps, "_target", None)
        assert asyncio.run(R._tool("screenshot")).ok  # (NEO finds the board)
        m = env.mark()
        out = asyncio.run(R._tool("click", cell="e2"))
        got = env.since(m)
    assert out.ok and out.images and "square e2" in out.text, out.text
    assert got["captures"] == 2, got
