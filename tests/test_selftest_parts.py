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


def test_one_run_per_request_of_the_users(monkeypatch):
    """The same words handed over again with nothing new from the user (a model retrying after the
    report, Gemini Live mapping its own call onto the user's last sentence) never start a second run.
    The user asking again does."""
    from neo.events import bus

    ran = []

    async def fake_run(**kw):
        ran.append(1)
        return _fake_report()

    monkeypatch.setattr(controls, "run", fake_run)
    monkeypatch.setattr(controls, "save", lambda report: None)
    bus().new_user_turn()  # the user said it
    first = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("user")))
    again = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("user")))
    assert ran == [1] and first.text.startswith("Controls check:")
    assert again.ok and again.say is True and "already answered that request" in again.text
    bus().new_user_turn()  # "check your computer controls again"
    third = asyncio.run(registry().invoke("selftest_controls", {}, _ctx("user")))
    assert ran == [1, 1] and third.text.startswith("Controls check:")


def test_it_does_not_start_while_another_neo_job_is_running(monkeypatch):
    """Another job (an agent task, a Live tool call) may be clicking or typing: the test would take the
    mouse and keyboard from under it. It says so instead, and runs once that job is done."""
    from neo.events import NeoState, bus

    ran = []

    async def fake_run(**kw):
        ran.append(kw.get("busy"))
        return _fake_report()

    monkeypatch.setattr(controls, "run", fake_run)
    monkeypatch.setattr(controls, "save", lambda report: None)

    async def scenario():
        other_started, other_done = asyncio.Event(), asyncio.Event()

        async def other_job():
            await bus().set_state(NeoState.WORKING, job="j1")
            other_started.set()
            await other_done.wait()
            await bus().end_job("j1")

        async def ask(job):  # the way the session runs the fast path: its own job, announced
            bus().new_user_turn()
            await bus().set_state(NeoState.WORKING, job=job)
            await bus().tool_start("selftest_controls", {}, job=job)
            out = await registry().invoke("selftest_controls", {}, _ctx("user"))
            await bus().end_job(job)
            return out

        t = asyncio.create_task(other_job())
        await other_started.wait()
        busy = await ask("j2")
        other_done.set()
        await t
        # that request is answered: its words handed over again (a model's retry) don't run it later
        stale = await registry().invoke("selftest_controls", {}, _ctx("user"))
        free = await ask("j3")
        return busy, stale, free

    busy, stale, free = asyncio.run(scenario())
    assert busy.ok and busy.say is True and "in the middle of something else (another NEO job" in busy.text
    assert "already answered" in stale.text
    assert len(ran) == 1 and free.text.startswith("Controls check:")
    assert callable(ran[0]) and ran[0]() == ""  # checked again right before the mouse is taken


def _live_voice(said, *, utt=3):
    """A Gemini Live voice layer between sentences: the user's last one is over and answered (its
    words are only in the label now), and the session is a fake that records what it is handed."""
    import neo.voice.live as live_mod
    from neo.agent.session import Reply

    calls = []

    class S:
        async def handle(self, text, **kw):
            calls.append((text, kw.get("origin")))
            return Reply("done", "quick")

    class Socket:  # Gemini Live's end: the results NEO sends back
        async def send_tool_response(self, **kw):
            pass

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = S(), {}, {}, set()
    v._live, v._ui_lock, v._dialog, v._call_labels = Socket(), asyncio.Lock(), [], {}
    v._taint, v._utt_started, v._last_answer = [], 0.0, (0.0, "")
    v._lookups, v._seen_calls = {}, {}
    v._last_activity, v._utt_id, v._calls_this_turn = 0.0, utt, 0
    v._gate, v._held, v._result_pending, v._result_speech = None, [], False, False
    v._last_response_at, v._last_heard_at = 0.0, 0.0
    v._utterance = ""
    v._label = {"text": said, "tools": [], "spoke": True, "open": False, "id": ""}

    async def _noop(*a, **k):
        return None

    v._refresh_state = v._think = _noop
    return v, calls


def _model_calls(v, goal, n=1):
    """Gemini Live makes an agent_task call of its own (no new words from the user)."""
    fc = types.SimpleNamespace(id=f"c{n}", name="agent_task", args={"goal": goal})
    msg = types.SimpleNamespace(
        server_content=None, tool_call=types.SimpleNamespace(function_calls=[fc]), tool_call_cancellation=None, go_away=None
    )

    async def drive():
        await v._on_message(msg, [], [], v._live)
        await asyncio.gather(*list(v._tool_tasks.values()), return_exceptions=True)

    asyncio.run(drive())


def test_a_live_model_call_after_the_lane_ran_the_sentence_is_the_models_own(monkeypatch):
    """The user said "check your computer controls"; NEO's command lane ran it (well past the lane's
    30 s claim: the test takes about that long). The report goes to the model, which then calls
    agent_task itself (a retry, or a goal it read somewhere): that is the model's goal, never the
    user's old sentence run again as the user's own request."""
    from neo.agent import fastpath

    said = "check your computer controls"
    monkeypatch.setattr(fastpath, "plan", lambda text, **k: [("selftest_controls", {})] if phrases.is_request(text) else None)
    v, calls = _live_voice(said)
    v._lane_utt, v._lane_at = 3, 0.0  # the lane did sentence 3, long ago
    _model_calls(v, "open notes")
    assert calls == [("open notes", "live")]


def test_a_live_model_maps_onto_the_users_words_once_per_sentence(monkeypatch):
    """The model wrapping the user's command in agent_task runs the user's own words (fast) — once.
    A second call of its own for the same sentence is its goal; a new sentence maps again."""
    from neo.agent import fastpath

    said = "check your computer controls"
    monkeypatch.setattr(fastpath, "plan", lambda text, **k: [("selftest_controls", {})] if phrases.is_request(text) else None)
    v, calls = _live_voice(said)
    v._lane_utt, v._lane_at = -1, 0.0  # the model got there first: the lane stayed out
    _model_calls(v, said, 1)
    assert calls == [(said, "user")]
    _model_calls(v, "open notes", 2)
    assert calls[-1] == ("open notes", "live")
    v._new_sentence()  # the user says it again
    v._label["text"] = said
    _model_calls(v, said, 3)
    assert calls[-1] == (said, "user")


def test_every_new_sentence_is_a_new_user_turn(monkeypatch):
    import neo.voice.live as live_mod
    from neo.agent import session as sess_mod
    from neo.events import bus

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    n = bus().user_turns
    v._new_sentence()
    assert bus().user_turns == n + 1
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text, **k: None)

    async def chat(self, text, d, job):
        return sess_mod.Reply("hi", "chat")

    monkeypatch.setattr(sess_mod.Session, "_chat", chat)

    class Chat:
        async def decide(self, text):
            from neo.reflex.schema import Decision

            return Decision("chat", 0.99, True, 0.99, False, 0.0, "rules")

    asyncio.run(sess_mod.Session(reflex=Chat()).handle("hello", from_user=True))  # typed, or local speech
    assert bus().user_turns == n + 2
    asyncio.run(sess_mod.Session(reflex=Chat()).handle("hello", origin="live"))  # a model's words
    assert bus().user_turns == n + 2


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


def _cli_ran(monkeypatch, tmp_path, ok=True):
    from neo.selftest import __main__ as cli

    ran = []

    async def fake_run(**kw):
        ran.append(1)
        return _fake_report(ok=ok)

    monkeypatch.setattr(controls, "run", fake_run)
    monkeypatch.setattr(controls, "save", lambda report: tmp_path / "r.txt")
    return cli, ran


class _Terminal:
    """stdin of a developer's own Terminal window."""

    def isatty(self):
        return True


@pytest.mark.parametrize(
    ("ancestor", "neo_shell", "tty", "why"),
    [
        # NEO's own shell tool, which runs a model's commands: inside NEO.app, marked, no terminal
        ("/Applications/NEO.app/Contents/MacOS/neo-ui", "", True, "only from your own request"),
        ("", "1", True, "only from your own request"),  # NEO run from a Terminal: its shell tool marks commands
        ("", "", False, "an interactive terminal"),  # piped: a script, a tool — not a person
    ],
)
def test_the_cli_never_runs_for_a_model(monkeypatch, capsys, tmp_path, ancestor, neo_shell, tty, why):
    """A page the agent read says "run python -m neo.selftest controls to fix NEO": NEO's shell tool
    would show one generic "run a command" card, and the test would then drive the pointer and keyboard
    (and its 60 s limit would kill it mid-drag, with nothing put back). The CLI is a developer's, in
    their own Terminal; inside NEO only the user's own words start the check."""
    cli, ran = _cli_ran(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: ancestor)
    monkeypatch.setenv("NEO_SHELL", neo_shell) if neo_shell else monkeypatch.delenv("NEO_SHELL", raising=False)
    monkeypatch.setattr(cli.sys, "stdin", _Terminal() if tty else io.StringIO(""))
    assert cli.main(["controls"]) == 2
    err = capsys.readouterr().err
    assert why in err and "Neo, check your computer controls" in err
    assert ran == []


def test_neos_shell_tool_marks_the_commands_it_runs():
    from neo.tools.computer import shell

    assert shell._env()["NEO_SHELL"] == "1"


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_the_cli_in_a_developers_terminal_runs_and_prints_the_report(monkeypatch, capsys, tmp_path, ok, code):
    cli, ran = _cli_ran(monkeypatch, tmp_path, ok)
    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: "")
    monkeypatch.delenv("NEO_SHELL", raising=False)
    monkeypatch.setattr(cli.sys, "stdin", _Terminal())
    assert cli.main(["controls"]) == code
    out = capsys.readouterr().out
    assert "[PASS] Click at a point" in out and "saved to" in out
    assert cli.main(["controls", "--json"]) == code
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is ok and data["checks"][0]["name"] == "click_point"
    assert ran == [1, 1]


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
    """The note a screenshot comes with (pointer.note): the shot id a model clicks by, and the
    image→screen mapping for a click without it."""
    from neo.tools.computer import pointer

    region = pointer.note(pointer.remember(40, 64, 1.0, 720, 358), "region 720×358 pt at screen (40,64)", [])
    sid = controls.note_shot(region)
    assert sid and pointer.get_shot(sid) is not None
    assert controls.model_point(region, 298, 290) == (338, 354)
    assert controls.note_scale(region) == 1.0
    full = pointer.note(pointer.remember(0, 0, 1.0, 1512, 982), "all of display 1", [])
    assert controls.model_point(full, 169, 177) == (169, 177)  # image pixels are screen points here
    assert controls.note_scale(full) == 1.0
    scaled = pointer.note(pointer.remember(0, 0, 2.0, 1600, 900), "all of display 1", [])
    assert controls.model_point(scaled, 169, 177) == (338, 354)
    assert controls.note_scale(scaled) == 2.0
    assert controls.note_shot(scaled) != sid
    assert controls.model_point("an image", 1, 2) is None and controls.note_shot("an image") is None


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

        def window(self, number):
            return next((w for w in self.windows() if w["number"] == number), None)

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
        m.buttons_down,
        lambda: m.tap(lambda t, e: None),
        m.can_capture,
    ):
        with pytest.raises(EffectRefused):
            call()


# ---- the boards' geometry, the new window parts ---------------------------------------------------


def test_square_names_and_the_tilted_boards_perspective():
    from neo.selftest import geometry as G

    assert G.square(0, 0) == "a8" and G.square(4, 6) == "e2" and G.cell_of("e2") == [4, 6] and G.cell_of("H1") == [7, 7]
    for bad in ("i2", "e9", "e", "e0"):
        with pytest.raises(ValueError):
            G.cell_of(bad)
    quad = G.tilt_quad(240, 240)
    assert quad[0][0] > quad[3][0] and quad[1][0] < quad[2][0]  # the far edge is narrower
    m = G.board_map(quad)
    for (gx, gy), corner in zip(G.GRID, quad, strict=True):
        assert G.apply(m, gx, gy) == pytest.approx(corner)
    inv = G.invert(m)
    assert G.apply(inv, *G.apply(m, 3.3, 5.7)) == pytest.approx((3.3, 5.7))
    for c in range(8):
        for r in range(8):
            assert G.cell_at(quad, *G.cell_center(quad, c, r)) == [c, r]
    assert G.cell_at(quad, 1, 1) is None  # (above the far edge: off the board)
    near, far = G.cell_polygon(quad, 0, 7), G.cell_polygon(quad, 0, 0)
    assert abs(near[2][1] - near[0][1]) > abs(far[2][1] - far[0][1])  # nearer squares look bigger


def test_the_new_labels_never_read_as_commitments():
    from neo.tools.computer import policy

    for label in (L.HINDI_BUTTON, L.ROUND_MENU, L.SHEET_TITLE, L.SHEET_START, L.SHEET_CANCEL, *L.HINDI_PHRASES.values()):
        assert not policy.commits(label, L.SHEET_TITLE), label


def test_the_boards_pieces_and_moves():
    pieces = testwin.start_pieces()
    assert len(pieces) == 32 and pieces[(1, 6)] == "P" and pieces[(4, 0)] == "k" and tuple(L.PIECE) in pieces
    assert testwin.move_piece(pieces, [2, 6], [2, 4]) and pieces[(2, 4)] == "P" and (2, 6) not in pieces
    assert not testwin.move_piece(pieces, [2, 6], [2, 5])  # nothing there any more
    assert not testwin.move_piece(pieces, [0, 0], [0, 0]) and not testwin.move_piece(pieces, None, [1, 1])


def test_the_new_views_have_the_signatures_appkit_calls():
    k = testwin.classes()
    assert k["Tilt"].acceptsFirstMouse_.signature == b"B@:@" and k["Tilt"].isFlipped.signature == b"B@:"
    for sel in ("newRound_", "buttonHindi_"):
        assert getattr(k["Controller"], sel).signature == b"v@:@", sel


# ---- timings and scores ------------------------------------------------------------------------------


def test_timings_are_summarized_against_the_targets():
    from neo.selftest import bench

    t = [("click", 300.0, {}), ("click", 700.0, {}), ("click", 500.0, {}), ("type_text", 50.0, {"text": "x" * 20}),
         ("ax_set_value", 5.0, {})]  # fmt: skip
    s = bench.summarize(t)
    assert s["click"]["n"] == 3 and s["click"]["p50_ms"] == 500.0 and s["click"]["p90_ms"] == 700.0
    assert s["click"]["met_p0"] is True and s["click"]["met_p1"] is False
    assert s["type_text"]["p50_ms"] == 100.0  # per 40 characters
    assert s["ax_set_value"]["met_p0"] is None  # no target: not in the latency score
    sc = bench.scores(["pass", "pass", "fail", "skip"], s)
    assert sc["success"] == pytest.approx(0.667, abs=1e-3) and sc["skipped"] == 1
    assert sc["latency"] == 1.0 and sc["timed"] == 2
    assert bench.words(sc).startswith("Success 67% (2 of 3); speed targets met by 100% of 2 tools")
    assert bench.scores([], {})["success"] is None and bench.words(bench.scores([], {})) == ""


def test_the_report_carries_scores_and_timings():
    r = _fake_report(ok=True)
    r.timings = [("click", 200.0, {}), ("click", 400.0, {})]
    r.repeats = [controls.Check("click_point", "Click at a point", "pass", rep=1)]
    j = r.to_json()
    assert j["scores"]["success"] == 1.0 and j["scores"]["passed"] == 2 and j["timings"]["click"]["n"] == 2
    assert j["repeats"][0]["rep"] == 1
    assert "scores: success 1.0" in r.text() and "tool timings" in r.text() and "click: 300.0 / 400.0 of 2" in r.text()
    assert "Success 100% (2 of 2)" in r.reply()


def test_the_ax_timing_walk_is_refused_under_tests():
    from neo.effects import EffectRefused
    from neo.selftest import bench

    with pytest.raises(EffectRefused):
        bench.ax_walk()


def test_the_ax_timing_walk_keeps_numbers_only(monkeypatch):
    import ApplicationServices as AS

    from neo import effects
    from neo.selftest import bench
    from neo.tools import computer
    from neo.tools.computer import ax

    kids = {"root": ["a", "b"], "a": ["c"], "b": [], "c": []}

    def get(el, attr, _):
        if attr == "AXChildren":
            return (0, kids[el]) if kids[el] else (-25212, None)
        return (0, f"secret {attr} of {el}")

    monkeypatch.setattr(AS, "AXUIElementCreateApplication", lambda pid: "root")
    monkeypatch.setattr(AS, "AXUIElementCopyAttributeValue", get)
    monkeypatch.setattr(AS, "AXUIElementCopyMultipleAttributeValues", lambda el, names, o, _: (0, []))
    monkeypatch.setattr(AS, "AXUIElementSetMessagingTimeout", lambda el, t: 0)
    monkeypatch.setattr(ax, "working_app", lambda: ("Notes", 77))
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(computer, "_protected", lambda name, pid=0: False)
    with effects.allow("selftest"):
        out = bench.ax_walk()
        assert out["nodes"] == 4 and out["calls"] == 20 and out["complete"] and out["errors"] == {-25212: 2}
        assert "secret" not in json.dumps(out)  # timings, never what the app said
        monkeypatch.setattr(computer, "_protected", lambda name, pid=0: True)
        assert "never touches" in bench.ax_walk()["error"]
        monkeypatch.setattr(ax, "is_neo", lambda pid, name="": True)
        assert "NEO's own" in bench.ax_walk()["error"]


# ---- Accessibility actions go through the watch ----------------------------------------------------------


class _AxMac:
    def __init__(self, owners):
        self.owners = owners
        self.posted = []

    def element_pid(self, ref):
        return self.owners.get(ref, 0)

    def post(self, ev):
        self.posted.append(ev)


def test_accessibility_actions_are_shown_to_the_watch(monkeypatch):
    from neo.tools.computer import ax
    from neo.tools.computer import input as inp

    w = watch_mod.Watch(_AxMac({"mine": 4242, "theirs": 111}), 4242, {1: (0, 0, 100, 100)})
    pressed = []
    monkeypatch.setattr(ax, "press_el", lambda e: pressed.append(e.ref) or "Pressed")
    monkeypatch.setattr(ax, "_ax_watch", None, raising=False)
    monkeypatch.setattr(inp, "_watch", w)
    w._install_ax()
    try:
        assert ax._ax_watch == w.ax_hook  # NEO's AX choke point, when it has one, calls the watch
        assert ax.press_el(types.SimpleNamespace(ref="mine")) == "Pressed"
        with pytest.raises(watch_mod.Stop, match="another app"):
            ax.press_el(types.SimpleNamespace(ref="theirs"))
        assert pressed == ["mine"] and w.refused
        assert [e["pid"] for e in w.ax_seen] == [4242]
    finally:
        w._uninstall_ax()
    assert ax._ax_watch is None and not hasattr(ax.press_el, "__wrapped__")


def test_a_watch_that_isnt_armed_checks_nothing(monkeypatch):
    from neo.tools.computer import input as inp

    w = watch_mod.Watch(_AxMac({}), 4242, {})
    monkeypatch.setattr(inp, "_watch", None)
    w.ax_hook("anything", "press")  # (a watch left over from a run that ended: no one's)
    assert w.ax_seen == [] and not w.refused


# ---- a local grounder, scored on a saved picture ------------------------------------------------------------


def test_a_grounder_is_scored_on_the_windows_own_targets(tmp_path):
    from neo.selftest import geometry as G
    from neo.selftest import grounding

    png = _png((500, 400), lambda d: None)
    quad = [[300, 100], [400, 100], [420, 200], [280, 200]]
    meta = {"region": [100, 50, 500, 400], "scale": 1.0,
            "targets": {"button_a": [120, 60, 50, 20], "marker": [300, 300, 36, 36]}, "tilt_quad": quad}  # fmt: skip

    def good(img, ask):
        if ask.startswith("square"):
            x, y = G.cell_center([tuple(p) for p in quad], *G.cell_of(ask.split()[1]))
        else:
            f = meta["targets"]["button_a" if "Alpha" in ask else "marker"]
            x, y = f[0] + f[2] / 2, f[1] + f[3] / 2
        return x - 100, y - 50

    out = grounding.evaluate(good, png, meta)
    assert out["targets"] == 5 and out["hit_rate"] == 1.0 and out["p50_s"] is not None
    bad = grounding.evaluate(lambda img, ask: (0, 0), png, meta)
    assert bad["hit_rate"] == 0.0
    broken = grounding.evaluate(lambda img, ask: 1 / 0, png, meta)
    assert broken["hit_rate"] == 0.0 and {r["error"] for r in broken["rows"]} == {"ZeroDivisionError"}
    shot = tmp_path / "window-1.png"
    shot.write_bytes(png)
    shot.with_suffix(".json").write_text(json.dumps(meta))
    assert grounding.latest_shot(tmp_path) == shot
    with pytest.raises(ValueError):
        grounding.load("no-function-here")


# ---- the agent section and the A/B verdict -------------------------------------------------------------------


def test_the_agent_section_never_presses_chords_the_whole_mac_acts_on():
    from neo.selftest import agent

    sent = []

    async def real(a, c):
        sent.append(a["keys"])
        return "Pressed"

    hk = agent._safe_hotkey(real)
    for keys in ("cmd+tab", "cmd+space", "cmd+q", "cmd+h", "cmd+m", "ctrl+left", "cmd+alt+esc", "cmd+shift+3", "fn+f"):
        assert asyncio.run(hk({"keys": keys}, None)).startswith("Error"), keys
    assert asyncio.run(hk({"keys": "cmd+k", "app": "Safari"}, None)).startswith("Error")  # never another app
    for keys in ("cmd+k", "return", "escape", "cmd+j", "tab"):
        assert asyncio.run(hk({"keys": keys}, None)) == "Pressed", keys
    assert sent == ["cmd+k", "return", "escape", "cmd+j", "tab"]
    chosen = []

    async def menu(a, c):
        chosen.append(a)
        return "Chose it."

    mc = agent._safe_menu(menu)
    for words in ("Restart…", "shut down the mac", "Log Out Arnav…", "Force Quit…", "System Settings…"):
        assert asyncio.run(mc({"command": words}, None)).startswith("Error"), words
    assert asyncio.run(mc({"command": "New Round", "app": "Finder"}, None)).startswith("Error")
    assert asyncio.run(mc({"command": "start a new round"}, None)) == "Chose it." and chosen == [{"command": "start a new round"}]


def test_the_ab_verdict_needs_enough_runs_and_success_before_speed():
    from neo.selftest import agent

    def arm(runs, success, secs, calls=3):
        return {"runs": runs, "success": success, "seconds_p50": secs, "model_calls_p50": calls}

    s = {"low": arm(10, 0.9, 8.0, 3), "high": arm(10, 0.9, 30.0, 5)}
    assert "low can be the default" in agent.verdict(s, "low", "high")
    assert "keep high" in agent.verdict({**s, "low": arm(10, 0.7, 8.0)}, "low", "high")
    assert "wasn't faster" in agent.verdict({**s, "low": arm(10, 0.9, 40.0)}, "low", "high")
    assert "too few runs" in agent.verdict({**s, "low": arm(4, 0.9, 8.0)}, "low", "high")
    assert "not measured" in agent.verdict(s, "batch on", "batch off")
    rows = [{"arm": "low", "success": True, "model_calls": 2, "seconds": 5.0, "failover_calls": 1, "failover_s": 1.5},
            {"arm": "low", "success": False, "model_calls": 4, "seconds": 9.0, "failover_calls": 0, "failover_s": 0.0}]  # fmt: skip
    got = agent.summarize(rows)["low"]
    assert got["runs"] == 2 and got["success"] == 0.5 and got["model_calls_p50"] == 3 and got["failover_share"] == pytest.approx(1 / 6, abs=1e-3)


def test_an_arm_sets_only_switches_the_build_has_and_puts_them_back(monkeypatch):
    from neo import config
    from neo.selftest import agent

    s = types.SimpleNamespace(screen_effort="medium")
    monkeypatch.setattr(config, "settings", lambda: s)
    with agent._settings_for(agent.Arm("low", effort="low", batch=True)) as have:
        assert s.screen_effort == "low" and have == {"screen_effort": True, "batch_actions": False}
        assert not hasattr(s, "batch_actions")
    assert s.screen_effort == "medium"


def test_the_cli_runs_grounding_only_with_a_grounder(monkeypatch, capsys):
    from neo.selftest import __main__ as cli

    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: "")
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["grounding"]) == 2 and "--grounder" in capsys.readouterr().err
    monkeypatch.setattr(cli, "neo_app_ancestor", lambda *a, **k: "/Applications/NEO.app/Contents/MacOS/neo")
    for test in ("ab", "agent", "grounding"):
        assert cli.main([test]) == 2  # inside NEO: never, whatever the test
