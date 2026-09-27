"""The controls self-test's parts: who may start it, how it's reached, the CLI's guard, the test
window's own helpers, the screenshot marker, the input hook and the protection of NEO's windows.

Hermetic: no window is opened, no event is posted, no app is activated, no subprocess is run for
real, no model is called (the session tests use a fake reflex path and fake handlers).
"""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import types

import pytest
import Quartz as _RealQ
from PIL import Image, ImageDraw

from neo.agent.registry import ToolContext, ToolOutput, registry
from neo.events import EventBus
from neo.selftest import controls, phrases, testwin
from neo.selftest import layout as L
from neo.selftest import watch as watch_mod


@pytest.fixture(autouse=True)
def _nothing_real(monkeypatch, tmp_path):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()

    def refuse(*a, **k):
        raise AssertionError("a real side effect was attempted")

    for n in ("CGEventPost", "CGEventPostToPid", "CGEventTapCreate"):
        monkeypatch.setattr(_RealQ, n, refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)

    async def no_run(**kw):
        raise AssertionError("the real controls self-test must not run here")

    monkeypatch.setattr(controls, "run", no_run)
    yield


# ---- the words ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "check your computer controls",
        "Neo, check your computer controls",
        "Hey Neo check your computer controls.",
        "can you check your computer controls?",
        "please test your computer controls now",
        "check if your computer controls are working",
        "check whether your computer controls work properly",
        "can you check neo's computer controls",
        "verify your controls",
        "run a controls self-test",
        "run your controls check",
        "self-test your controls",
        "self-test the computer controls",  # "self-test" says whose
        "diagnose your mouse and keyboard controls",
        "apne computer controls check karo",
        "neo ke computer controls test kar lo",
        "अपने कंप्यूटर कंट्रोल्स चेक करो",
    ],
)
def test_the_request_is_recognised(text):
    assert phrases.is_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "are your computer controls working?",  # a question: answered, nothing moves
        "check the controls",
        "check the controls of the car",
        "check your controls in the game settings",
        "check your computer controls and open notes",
        "check my email",
        "check your computer",
        "type check your computer controls",
        "what are computer controls",
        # nobody's controls in particular (a game's, an app the user is building): never NEO's own test
        "test the input controls",
        "test input controls",
        "check the mouse and keyboard controls",
        "check my computer controls",
        "check computer controls",
        "check whether the computer controls work properly",
        "run the controls test",
        "start a controls check",
        "computer controls check kar lo",
        "कंप्यूटर कंट्रोल्स चेक करो",
    ],
)
def test_other_sentences_are_not_a_request(text):
    assert not phrases.is_request(text)


def test_the_fast_path_runs_it_and_only_it():
    from neo.agent import fastpath

    assert fastpath.match("check your computer controls") == ("selftest_controls", {})
    assert fastpath.plan("Neo, check your computer controls") == [("selftest_controls", {})]
    assert not any(t == "selftest_controls" for t, _ in fastpath.plan("are your computer controls working?") or [])


def test_the_reflex_never_takes_it_for_conversation():
    from neo.reflex import Reflex, _rules

    d = _rules("check your computer controls")
    assert d is not None and d.intent == "quick_action" and d.source == "rules" and d.intent_confidence == 1.0
    assert _rules("are your computer controls working?") is None
    assert asyncio.run(Reflex().decide("Neo, test your computer controls")).intent == "quick_action"


# ---- never a model's choice -------------------------------------------------------------------


def test_no_model_is_ever_offered_the_tool():
    from neo.agent.toolselect import ToolSelector
    from neo.voice import live

    t = registry().get("selftest_controls")
    assert t is not None and t.hidden and not t.chain and not t.early and not t.quiet
    assert "selftest_controls" not in {s.name for s in registry().specs()}
    assert "selftest_controls" not in registry().catalog()
    sel = ToolSelector(registry())
    picked = sel.select("check your computer controls self test", needs_screen=True)
    assert "selftest_controls" not in {s.name for s in picked.specs(registry())}
    assert "selftest_controls" not in {t.name for t in sel.search("check computer controls self-test", k=50)}
    assert not live._shown("selftest_controls")


def _ctx(origin, text="check your computer controls"):
    return ToolContext(user_text=text, origin=origin)


def test_a_model_calling_it_anyway_is_refused():
    out = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("agent")))
    assert not out.ok and "only the user can start it" in out.text


@pytest.mark.parametrize(
    ("origin", "text"),
    [
        ("live", "check your computer controls"),  # Gemini Live's own paraphrase
        ("approval", "check your computer controls"),
        ("user", "check your computer controls and open notes"),  # not on its own
        ("user", "please look at the screen"),
    ],
)
def test_anything_but_the_users_own_request_is_answered_not_run(origin, text):
    out = asyncio.run(registry().invoke("selftest_controls", {}, _ctx(origin, text)))
    # ok: a refusal the user hears, never a failure a model would then try to "fix"
    assert out.ok and out.say is True and "say “check your computer controls”" in out.text


def _fake_report(ok=True):
    r = controls.Report(info={"in front before": "Safari"})
    r.checks = [controls.Check("click_point", "Click at a point").ok("it landed on cell 2,2")]
    if not ok:
        r.checks.append(
            controls.Check("first_click", "First click on a window that isn't in front", said="Left-clicked at (1, 2)").fail(
                "the app got the click, but it only brought the window to the front"
            )
        )
    return r


def test_the_users_request_runs_the_test_and_speaks_the_report(monkeypatch, tmp_path):
    from neo.events import bus

    notes = []

    async def note(text):
        notes.append(text)

    async def fake_run(*, notify=None, **kw):
        await notify("starting")
        return _fake_report(ok=False)

    monkeypatch.setattr(bus(), "note", note)
    monkeypatch.setattr(controls, "run", fake_run)
    monkeypatch.setattr(controls, "save", lambda report: tmp_path / "controls.txt")
    out = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("user", "Neo, check your computer controls")))
    assert out.ok and out.say is True
    assert out.text.startswith("Controls check: 1 of 2 passed.")
    assert "First click on a window that isn't in front: the app got the click" in out.text
    assert "Full report:" in out.text
    assert notes == ["starting"]


def test_it_asks_for_accessibility_first(monkeypatch):
    from neo.tools.computer import ax

    monkeypatch.setattr(ax, "is_trusted", lambda: False)
    out = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("user")))
    assert out.halt and "Accessibility" in out.text


def test_one_run_at_a_time(monkeypatch, tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_run(**kw):
        started.set()
        await release.wait()
        return _fake_report()

    monkeypatch.setattr(controls, "run", slow_run)
    monkeypatch.setattr(controls, "save", lambda report: None)

    async def both():
        first = asyncio.create_task(registry().invoke("selftest_controls", {}, _ctx("user")))
        await started.wait()
        second = await registry().invoke("selftest_controls", {}, _ctx("user"))
        release.set()
        return await first, second

    first, second = asyncio.run(both())
    assert second.text == "The controls check is already running."
    assert first.text.startswith("Controls check: 1 of 1 passed.")


# ---- the session: the user's words go straight to it -------------------------------------------


def test_the_session_runs_it_from_the_users_words_and_never_hands_it_to_a_model(monkeypatch):
    from neo.agent import fastpath
    from neo.agent import session as sess_mod
    from neo.reflex import Reflex

    seen = []

    async def handler(a, c):
        seen.append((c.origin, c.user_text))
        return ToolOutput("Controls check: 19 of 19 passed.", say=True)

    async def no_agent(*a, **k):
        raise AssertionError("handed to a model")

    monkeypatch.setattr(fastpath, "plan", lambda text, **k: [("selftest_controls", {})] if phrases.is_request(text) else None)
    monkeypatch.setattr(registry().get("selftest_controls"), "handler", handler)
    monkeypatch.setattr(sess_mod.Session, "_agent", no_agent)
    s = sess_mod.Session(reflex=Reflex())  # the reflex's rule: no model loads for these words
    r = asyncio.run(s.handle("Neo, check your computer controls", from_user=True))
    assert r.route == "quick" and r.tools == ("selftest_controls",)
    assert r.text == "Controls check: 19 of 19 passed." and not r.silent
    assert seen == [("user", "check your computer controls")]
    # Gemini Live handing over its own paraphrase: the tool sees who chose the words
    asyncio.run(s.handle("check your computer controls", origin="live"))
    assert seen[-1][0] == "live"


def test_a_laya_pick_of_the_hidden_tool_is_ignored(monkeypatch):
    from neo.agent import session as sess_mod
    from neo.reflex.schema import Decision

    class Picks:
        async def decide(self, text):
            return Decision("quick_action", 0.99, False, 0.0, False, 0.0, "laya+head", tool="selftest_controls", tool_p=0.99)

    called = []

    async def fake_agent(self, text, d, images, job, *, only=None, done_note=""):
        called.append(only)
        return sess_mod.Reply("agent", "agent")

    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text, **k: None)
    monkeypatch.setattr(sess_mod.Session, "_agent", fake_agent)
    s = sess_mod.Session(reflex=Picks())
    asyncio.run(s.handle("please test all the things", from_user=True))
    assert called == [None]  # the general agent (which can't see it), never the single-tool lane


# ---- the CLI -----------------------------------------------------------------------------------

_TREE = {
    500: (400, "/Users/me/Library/Application Support/NEO/.venv/bin/python"),  # this command
    400: (300, "/bin/zsh"),  # NEO's shell tool
    300: (200, "/Users/me/Library/Application Support/NEO/.venv/bin/python"),  # NEO's core
    200: (1, "/Applications/NEO.app/Contents/MacOS/neo-ui"),
}


def test_the_cli_knows_whether_it_runs_inside_neo_app():
    from neo.selftest.__main__ import neo_app_ancestor

    assert neo_app_ancestor(500, _TREE) == "/Applications/NEO.app/Contents/MacOS/neo-ui"
    terminal = {500: (400, "/usr/bin/python3"), 400: (300, "/bin/zsh"), 300: (1, "/System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal")}
    assert neo_app_ancestor(500, terminal) == ""
    assert neo_app_ancestor(500, {500: (400, "/bin/zsh"), 400: (500, "/bin/zsh")}) == ""  # a loop ends
    assert neo_app_ancestor(500, {}) == ""


def test_the_cli_reads_the_process_table(monkeypatch):
    from neo.selftest import __main__ as cli

    out = "  200     1 /Applications/NEO.app/Contents/MacOS/neo-ui\n  300   200 /Users/me/NEO/.venv/bin/python\nbad line\n"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout=out))
    assert cli.process_table() == {200: (1, "/Applications/NEO.app/Contents/MacOS/neo-ui"), 300: (200, "/Users/me/NEO/.venv/bin/python")}


def test_the_cli_outside_neo_app_does_nothing(monkeypatch, capsys):
    from neo.selftest import __main__ as cli

    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: "")
    assert cli.main(["controls"]) == 2
    err = capsys.readouterr().err
    assert "only inside NEO.app" in err and "Neo, check your computer controls" in err


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_the_cli_inside_neo_app_runs_and_prints_the_report(monkeypatch, capsys, tmp_path, ok, code):
    from neo.selftest import __main__ as cli

    async def fake_run(**kw):
        return _fake_report(ok=ok)

    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: "/Applications/NEO.app/Contents/MacOS/neo-ui")
    monkeypatch.setattr(controls, "run", fake_run)
    monkeypatch.setattr(controls, "save", lambda report: tmp_path / "r.txt")
    assert cli.main(["controls"]) == code
    out = capsys.readouterr().out
    assert "[PASS] Click at a point" in out and "saved to" in out
    assert cli.main(["controls", "--json"]) == code
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is ok and data["checks"][0]["name"] == "click_point"


# ---- the test window's own helpers --------------------------------------------------------------


def test_window_geometry_helpers():
    assert testwin.rect_tl(10, 20, 30, 40, height=100) == (10, 40, 30, 40)
    assert testwin.to_global(100, 700, 50, 20, 982) == [100, 262, 50, 20]
    assert testwin.point_to_global(10, 900, 982) == [10, 82]
    assert testwin.cell_at(0, 0) == [0, 0]
    assert testwin.cell_at(239.9, 239.9) == [7, 7]
    assert testwin.cell_at(240, 10) is None and testwin.cell_at(-1, 5) is None
    # top-left of the visible area, MARGIN in; always on the screen
    assert testwin.place((0, 0, 1512, 957), 720, 358) == (40, 957 - 358 - 40)
    assert testwin.place((0, 0, 700, 300), 720, 358) == (0, 0)


def test_window_commands_and_log(tmp_path):
    assert testwin.command('{"cmd": "deactivate"}') == "deactivate"
    assert testwin.command('{"cmd": "quit"}\n') == "quit"
    assert testwin.command('{"cmd": "rm -rf"}') == "" and testwin.command("garbage") == "" and testwin.command("[]") == ""
    log = testwin.Log(str(tmp_path / "events.jsonl"))
    log.write("text", target="field", value=L.TYPE_TEXT)
    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["value"] == L.TYPE_TEXT and "नमस्ते" in line  # written as typed, flushed at once
    args = testwin.parse_args(["--log", "x", "--tag", "AB12", "--parent", "7"])
    assert (args.log, args.tag, args.parent, args.lifetime) == ("x", "AB12", 7, 120.0)


def test_window_classes_have_the_signatures_appkit_calls():
    """Defining the classes opens nothing; a wrong signature would only show on the real screen."""
    k = testwin.classes()
    assert testwin.classes() is k  # defined once per process
    sig = {
        (k["Tile"], "draggingSession_sourceOperationMaskForDraggingContext_"): b"Q@:@q",
        (k["Tile"], "draggingSession_endedAtPoint_operation_"): b"v@:@{CGPoint=dd}Q",
        (k["Well"], "draggingEntered_"): b"Q@:@",
        (k["Well"], "performDragOperation_"): b"B@:@",
        (k["Board"], "acceptsFirstMouse_"): b"B@:@",
        (k["Board"], "isFlipped"): b"B@:",
        (k["Controller"], "tick_"): b"v@:@",
        (k["Controller"], "controlTextDidChange_"): b"v@:@",
    }
    for (cls, sel), want in sig.items():
        assert getattr(cls, sel).signature == want, sel


def test_no_label_in_the_window_reads_as_a_commitment():
    """NEO's click gate would ask before pressing a "Send" or an "OK": the test window has none."""
    from neo.tools.computer import policy

    for label in (L.TITLE, L.BUTTON_A, L.BUTTON_B, L.FIELD_HINT, L.MENU_TITLE, "Double-click me", "Drag me", "Drop here"):
        assert not policy.commits(label, L.TITLE), label


# ---- screenshots → coordinates ------------------------------------------------------------------


def _png(size, draw):
    img = Image.new("RGB", size, (120, 120, 120))
    draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_the_marker_is_found_by_its_colour_and_shape():
    png = _png((400, 300), lambda d: d.rectangle([100, 50, 135, 85], fill=L.MARKER_RGB))
    assert controls.find_marker(png) == (118.0, 68.0, 36.0, 36.0)
    # a magenta bar elsewhere (not a square) and a smaller square don't fool it
    busy = _png(
        (400, 300),
        lambda d: (
            d.rectangle([10, 200, 300, 215], fill=L.MARKER_RGB),
            d.rectangle([300, 20, 311, 31], fill=L.MARKER_RGB),
            d.rectangle([100, 50, 135, 85], fill=L.MARKER_RGB),
        ),
    )
    assert controls.find_marker(busy, 36) == (118.0, 68.0, 36.0, 36.0)
    assert controls.find_marker(_png((200, 100), lambda d: None)) is None
    assert controls.find_marker(_png((600, 600), lambda d: d.rectangle([0, 0, 599, 599], fill=L.MARKER_RGB))) is None


def test_what_a_model_makes_of_the_screenshot_note():
    region = "Region screenshot attached, origin at screen (40,64). screen_x = 40 + image_x×1.00; screen_y = 64 + image_y×1.00."
    assert controls.model_point(region, 298, 290) == (338, 354)
    assert controls.note_scale(region) == 1.0
    full = "Screenshot attached (full screen). 1 image px = 2.00 screen pt; image coords = click coords."
    assert controls.model_point(full, 169, 177) == (169, 177)  # what the note says to do
    assert controls.note_scale(full) == 2.0
    scaled = "Screenshot attached (full screen). 1 image px = 2.00 screen pt."
    assert controls.model_point(scaled, 169, 177) == (338, 354)
    assert controls.model_point("an image", 1, 2) is None


# ---- the report ---------------------------------------------------------------------------------


def test_the_report_reads_well():
    r = _fake_report(ok=False)
    r.checks.append(controls.Check("second_display", "Click on a second display").skip("there's one display"))
    reply = r.reply()
    assert reply.splitlines()[0] == "Controls check: 1 of 2 passed."
    assert "Didn't work:\n- First click on a window that isn't in front: the app got the click" in reply
    assert "NEO's tool said: “Left-clicked at (1, 2)”" in reply
    assert "Worked: click at a point." in reply and "Skipped: click on a second display (there's one display)." in reply
    text = r.text()
    assert "[FAIL] First click" in text and "[skip] Click on a second display" in text and "in front before: Safari" in text
    assert controls.Report(refused="the test window didn't open").summary() == (
        "I didn't run the controls check: the test window didn't open"
    )
    stopped = controls.Report(stopped="you moved the mouse", checks=[controls.Check("a", "A").ok("x")])
    assert stopped.summary() == "Controls check: 1 of 1 passed before I stopped (you moved the mouse)."
    assert not stopped.ok


# ---- the input hook and the protection of NEO's windows -----------------------------------------


def _quartz_recorder(posted):
    Q = types.SimpleNamespace(**{n: getattr(_RealQ, n) for n in dir(_RealQ) if n.startswith("kCG")})
    Q.CGEventCreateMouseEvent = lambda src, typ, pos, btn: {"type": typ, "pos": pos, "fields": {}}
    Q.CGEventSetIntegerValueField = lambda ev, f, v: ev["fields"].__setitem__(f, v)
    Q.CGEventPost = lambda tap, ev: posted.append(ev)
    return Q


def test_every_event_is_shown_to_the_watch_before_it_is_posted(monkeypatch):
    from neo.tools.computer import input as inp

    posted, shown = [], []
    monkeypatch.setattr(inp, "Q", _quartz_recorder(posted))
    monkeypatch.setattr(inp, "time", types.SimpleNamespace(sleep=lambda s: None))
    assert inp._watch is None  # nothing watches outside a self-test

    def watch(ev):
        shown.append(ev["type"])
        if ev["type"] == _RealQ.kCGEventLeftMouseDown:
            raise watch_mod.Stop("not the test window")

    monkeypatch.setattr(inp, "_watch", watch)
    with pytest.raises(watch_mod.Stop):
        inp.click(10, 20)
    assert shown == [_RealQ.kCGEventMouseMoved, _RealQ.kCGEventLeftMouseDown]
    assert [e["type"] for e in posted] == [_RealQ.kCGEventMouseMoved]  # the refused press never went out


def test_a_drag_that_would_leave_the_test_window_is_cut_and_let_go(monkeypatch):
    posted = []
    Q = _quartz_recorder(posted)
    Q.CGEventGetType = lambda ev: ev["type"]
    Q.CGEventGetLocation = lambda ev: types.SimpleNamespace(x=ev["pos"][0], y=ev["pos"][1])
    Q.CGEventGetIntegerValueField = lambda ev, f: ev["fields"].get(f, 0)
    monkeypatch.setattr(watch_mod, "Q", Q)

    class Mac:
        def windows(self):
            return [{"pid": 4242, "name": "NEO Self-Test", "number": 777, "alpha": 1.0, "bounds": (0.0, 0.0, 100.0, 100.0)}]

        def hit_pid(self, x, y):
            return 4242

        def post(self, ev):
            posted.append(ev)

    w = watch_mod.Watch(Mac(), 4242, {777: (0.0, 0.0, 100.0, 100.0)})
    w.mode = "tap"
    ev = lambda typ, x, y: {"type": typ, "pos": (x, y), "fields": {}}  # noqa: E731
    w(ev(_RealQ.kCGEventLeftMouseDown, 50, 50))
    w(ev(_RealQ.kCGEventLeftMouseDragged, 70, 50))
    with pytest.raises(watch_mod.Stop, match="would have left the test window"):
        w(ev(_RealQ.kCGEventLeftMouseDragged, 150, 50))
    assert [(e["type"], e["pos"]) for e in posted] == [(_RealQ.kCGEventLeftMouseUp, (70, 50))]  # let go where it was
    assert posted[0]["fields"][_RealQ.kCGEventSourceUserData] == w.tag
    with pytest.raises(watch_mod.Stop):
        w(ev(_RealQ.kCGEventMouseMoved, 60, 60))  # and nothing more after that


@pytest.fixture
def front(monkeypatch):
    from neo.tools.computer import apps, focus

    now = {"app": ("TextEdit", 1)}
    monkeypatch.setattr(apps, "_target", None)
    monkeypatch.setattr(apps, "front_name", lambda: now["app"][0])
    monkeypatch.setattr(focus, "front_app", lambda: now["app"])
    return now


def test_neos_own_windows_stay_protected_during_a_self_test(front):
    from neo.tools import computer

    async def guard():
        return await computer._guard_front()

    with computer.allow_test_window(4242):
        front["app"] = ("NEO", 4242)  # the test window, even if something took it for NEO's own
        assert asyncio.run(guard()) is None
        for app in (("NEO", 999), ("neo-ui", 1), ("NEO", 0)):  # NEO's overlay, or anything else by that name
            front["app"] = app
            assert "click where you want me to type first" in asyncio.run(guard())
    front["app"] = ("NEO", 4242)
    assert "click where you want me to type first" in asyncio.run(guard())  # only while the test runs
    with pytest.raises(RuntimeError), computer.allow_test_window(4242):
        raise RuntimeError("the test broke")
    assert computer._test_window_pid == 0
    front["app"] = ("NEO Self-Test AB12", 4242)
    assert asyncio.run(guard()) is None  # its own name was never NEO's


def test_the_self_test_never_touches_the_real_mac_under_tests(tmp_path):
    from neo.effects import EffectRefused
    from neo.selftest.mac import Mac

    m = Mac()
    for call in (
        lambda: m.launch(tmp_path / "log", "AB12"),
        m.windows,
        lambda: m.window(1),
        m.front,
        m.pointer,
        lambda: m.hit_pid(1, 2),
        lambda: m.pids_named("x"),
        lambda: m.activate(1),
        lambda: m.post({}),
        lambda: m.tap(lambda t, e: None),
        m.can_capture,
    ):
        with pytest.raises(EffectRefused):
            call()
