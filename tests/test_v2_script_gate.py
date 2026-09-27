"""The AppleScript gate after the Chess run ("play chess as white": six read-only System Events
queries, each a card, then the brain gave up).

* policy.script_facts: a statement-by-statement reader, not a regex — what only reads runs
  without a card (unless the request is tainted), everything else asks;
* a yes "for this task" (neo.agent.grants): later calls of the same tool on the same app in the
  same task run without a card, never a dangerous one, never after new untrusted reading, never
  in another task;
* Accessibility off: the first tool that needs it stops the run with how to turn it on;
* a script that keeps failing on an app stops being retried; the prompt steers canvas apps to
  screenshot + click.

No script ever reaches osascript here: apps.osascript is replaced in every test that runs one.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import neo.agent.session as sess_mod
from neo.agent import confirm, grants
from neo.agent.registry import RegisteredTool, ToolContext, registry
from neo.agent.session import Session, answer_of, scope_of
from neo.events import EventBus, bus
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision
from neo.tools.computer import policy
from neo.tools.computer.policy import script_facts, script_is_read_only

_OBJ = {"type": "object", "properties": {}}

# The six scripts of the Chess run (as the log showed them), and the shapes models write.
CHESS_READS = [
    'tell application "System Events" to tell process "Chess" to get name of windows',
    'tell application "System Events" to tell process "Chess" to get class of UI elements of window 1',
    'tell application "System Events"\n  tell process "Chess"\n    tell sheet 1 of window 1\n'
    "      get name of pop up buttons\n    end tell\n  end tell\nend tell",
    'tell application "System Events"\n\ttell process "Chess"\n\t\tset winNames to name of windows\n'
    "\tend tell\nend tell\nreturn winNames",
    'tell application "System Events" to tell process "Chess" to get name of menu items of menu 1 of '
    "menu bar item 2 of menu bar 1",
    'tell application "System Events" to tell process "Chess" to tell sheet 1 of window 1 to get name of pop up buttons',
]


# ---- the reader: what only reads -----------------------------------------------------------------


@pytest.mark.parametrize("script", CHESS_READS)
def test_the_chess_runs_queries_only_read(script):
    f = script_facts(script)
    assert f.read_only and f.app == "Chess" and f.ui and f.danger == ""


@pytest.mark.parametrize(
    "script",
    [
        'tell application "Music" to get name of current track',
        'tell application "Safari" to return URL of current tab of window 1',
        'tell application "Finder" to count (every item of desktop)',
        'tell application "System Events" to get value of attribute "AXRole" of window 1 of process "Chess"',
        'tell application "System Events" to tell process "Chess" to get {name, role description, position, size} '
        "of every UI element of window 1",
        'tell application "Finder" to get POSIX path of (target of front window as alias)',
        'tell application "Calendar" to get summary of every event of calendar "Work" whose start date > (current date)',
        'tell application "Reminders" to get name of every reminder whose completed is false',
        'set output to ""\ntell application "System Events"\n  repeat with w in windows of process "Chess"\n'
        "    set output to output & name of w & linefeed\n  end repeat\nend tell\nreturn output",
        "set AppleScript's text item delimiters to \", \"\ntell application \"Notes\" to set theNames to name of every note\n"
        "return theNames as text",
        'tell application "System Events"\n  if exists (process "Chess") then\n    tell process "Chess" to return count of windows\n'
        "  else\n    return 0\n  end if\nend tell",
        'try\n  tell application "System Events" to tell process "Chess" to get name of sheet 1 of window 1\n'
        "on error errMsg number errNum\n  return errMsg\nend try",
        "repeat with i from 1 to 3\n  log i\nend repeat",
        'use AppleScript version "2.4"\nuse scripting additions\ntell application "Music" to get player state',
        'tell front window of application "Safari" to get name of current tab',
        '(* the board *) tell application "Music" to get sound volume -- how loud',
        'tell application "System Events" to get name of ¬\n  every window of process "Chess"',
        "GET NAME OF WINDOW 1",
        "with timeout of 5 seconds\n  get name of window 1\nend timeout",
        'tell application "Finder"\n  set w to name of window 1\nend tell\nget w',
        'if exists window 1 then return name of window 1',
        'tell application "Mail" to get subject of every message of inbox whose read status is false',
    ],
)
def test_reads_run_without_asking(script):
    assert script_is_read_only(script), script_facts(script)


def test_a_verb_inside_a_string_is_only_text():
    assert script_is_read_only('get "delete everything, then do shell script \\"rm -rf ~\\" and click OK"')
    assert script_is_read_only('tell application "System Events" to tell process "Chess" to get name of button "Delete"')
    assert script_is_read_only('get "(* not a comment *) -- nor this"')


# ---- the reader: what acts --------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        # commands, in every position
        'tell application "System Events" to tell process "Chess" to click menu item "New Game" of menu 1 of '
        'menu bar item "Game" of menu bar 1',
        'tell application "System Events" to tell process "Chess" to perform action "AXPress" of button 1 of window 1',
        'tell application "System Events" to key code 36',
        'tell application "System Events" to tell process "Chess" to select row 2 of table 1 of window 1',
        "get (do shell script \"id\")",
        'get name of window 1 & (do shell script "id")',
        'return (click button 1 of window 1)',
        'set x to (make new document)',
        'tell application "Finder" to get name of (delete file 1)',
        'tell application "Chess" to activate',
        'tell application "Music" to playpause',
        'tell application "Finder" to open file "x.app"',
        'tell application "Chess" to quit',
        'tell application "Finder" to duplicate file 1 to desktop',
        'tell application "Finder" to move file 1 to trash',
        'display dialog "hi"',
        "copy 1 to x",
        'load script file "x.scpt"',
        'store script s in "x"',
        # a verb in a nested tell of an otherwise reading script
        'tell application "System Events"\n  get name of windows of process "Chess"\n'
        '  tell process "Chess" to perform action "AXPress" of button 1 of window 1\nend tell',
        'tell application "System Events"\n  tell process "Chess"\n    get name of windows\n'
        "    tell window 1\n      click button 1\n    end tell\n  end tell\nend tell",
        # setting a property is acting, however it's spelt
        'tell application "System Events"\n  tell process "Chess"\n    set value of text field 1 of window 1 to "x"\n'
        "  end tell\nend tell",
        'tell application "System Events" to set frontmost of process "Chess" to true',
        'tell application "System Events" to tell process "Chess" to set frontmost to true',
        'tell application "Music" to set mute to true',
        'tell application "Music" to set sound volume to 0',
        'tell application "Finder" to set selection to file 1',
        'tell application "Safari" to set URL of document 1 to "https://example.com"',
        "tell window 1 to set position to {0, 0}",
        "set the clipboard to \"x\"",
        "get the clipboard",
        # the bridge, raw codes, handlers, scripts
        "current application's NSTask's alloc()",
        'use framework "Foundation"',
        '«event sysoexec» "id"',
        'tell application "Finder" to «event aevtquit»',
        "foo()",
        "set foo to 1\nget foo(2)",
        'set x to "a"\nreturn x "b"',
        "on run\n  get name of window 1\nend run",
        "to go()\nend go",
        "script s\nend script",
        "property p : 1",
        "global g",
        'using terms from application "Mail"\nend using terms from',
        "considering case\n  get name of window 1\nend considering",
        # the lexer won't vouch for these
        'get |name|',
        "ɡet name of window 1",  # a lookalike g
        "get name of window 1; delete file 1",
        "get name of window 1\\\ndelete file 1",
        '(* "*) do shell script "x" (* " *)',
        '(* unterminated',
        'get "unterminated',
        "get name -- a comment\ndelete file 1",
        "get x delete file 1",
        # an app picked by an expression
        'tell application ("Fin" & "der") to get name',
        # a variable from outside the tell that could be the app's own command
        "set x to 1\ntell application \"Music\" to get x",
        # a block left open
        'tell application "System Events" to tell process "Chess"\nget name of windows',
        'tell application "Finder"\n  get name of window 1',
        "list folder \"/Users\"",
        "info for file \"x\"",
        "",
        "   ",
    ],
)
def test_scripts_that_act_ask(script):
    assert not script_is_read_only(script), script


def test_the_old_tests_intent_holds():
    """test_v2_security's and test_v2_review_fixes' cases, in one place."""
    acting = [
        'tell application "Terminal" to do script "rm -rf ~"',
        'tell application "Safari" to do JavaScript "document.forms[0].submit()" in document 1',
        'run script "do sh" & "ell script \\"id\\""',
        'tell application "System Events" to keystroke "a"',
        'open location "https://example.com"',
        "set volume output volume 0",
        'tell application "Mail" to enable rule 1',
        'tell application "Music" to next track',
        'tell application "X" to toggle y',
        'tell application "Finder" to delete file "x"',
        'do shell script "rm -rf ~"',
        'tell application "Mail" to send m',
    ]
    assert not any(script_is_read_only(s) for s in acting)


# ---- facts for the gate: the app, the UI, the danger ---------------------------------------------


@pytest.mark.parametrize(
    "script, app, danger",
    [
        ('tell application "System Events" to tell process "Chess" to click button 1 of window 1', "Chess", ""),
        ('tell application "Chess" to activate', "Chess", ""),
        ('tell application "Terminal" to do script "ls"', "Terminal", "runs code"),
        ('tell application "Safari" to do JavaScript "1" in document 1', "Safari", "runs code"),
        ('«event sysoexec» "id"', "", "runs code"),
        # keystrokes are System Events' own (they go to whichever app is in front): not Chess's
        ('tell application "System Events" to tell process "Chess" to keystroke "e2e4"', "", "keystrokes"),
        ('tell application "System Events" to tell process "Mail" to click button "Send" of window 1', "Mail", "send"),
        ('tell application "Finder" to delete file "x"', "Finder", "send, buy, delete"),
        ('tell application "Music" to buy track 1', "Music", "send, buy, delete"),
        ('tell application "Finder" to open file "x"', "Finder", "reaches outside"),
        ('tell application "Finder" to activate\ntell application "Safari" to activate', "", "more than one app"),
        ('tell application "System Events" to tell (first process whose frontmost is true) to click button 1', "", "picks"),
        ('tell application "System Events" to sleep', "System Events", "reaches outside"),
        ('tell application "System Events" to set dark mode of appearance preferences to true', "System Events", "whole Mac"),
    ],
)
def test_script_facts_for_task_grants(script, app, danger):
    f = script_facts(script)
    assert f.app == app
    assert (danger in f.danger) if danger else f.danger == ""


def test_ui_scripting_is_told_apart_from_other_system_events_scripts():
    assert script_facts('tell application "System Events" to tell process "Chess" to get name of windows').ui
    assert script_facts('tell application "System Events" to keystroke "a"').ui
    assert not script_facts('tell application "System Events" to get name of every process').ui
    assert not script_facts('tell application "Music" to get name of current track').ui


# ---- the applescript tool -----------------------------------------------------------------------


@pytest.fixture
def ran(monkeypatch):
    """apps.osascript never runs: it records the script and answers `ran.reply`."""
    from neo.tools.computer import apps

    class Ran(list):
        reply = "ok"

    rec = Ran()

    async def fake(script, timeout=30):
        rec.append(script)
        return rec.reply

    monkeypatch.setattr(apps, "osascript", fake)
    confirm.cancel("test")
    yield rec
    confirm.cancel("test")


def _script(script: str, ctx: ToolContext | None = None):
    from neo.tools import load_all

    load_all()
    return asyncio.run(registry().invoke("applescript", {"script": script}, ctx or ToolContext(user_text="x")))


def test_a_read_only_script_runs_without_a_card_when_nothing_untrusted_was_read(ran):
    out = _script(CHESS_READS[0])
    assert out.ok and ran == [CHESS_READS[0]] and confirm.peek() is None


def test_a_read_only_script_after_untrusted_reading_asks(ran):
    out = _script(CHESS_READS[0], ToolContext(user_text="x", taint=["web_fetch"]))
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and ran == []
    p = confirm.peek()
    assert p.summary == "read Chess with an AppleScript" and p.detail == CHESS_READS[0]


def test_a_script_that_acts_asks_and_names_its_app(ran):
    s = 'tell application "System Events" to tell process "Chess" to click button 1 of window 1'
    out = _script(s)
    assert out.halt and ran == [] and confirm.peek().summary == "run an AppleScript in Chess"


def test_a_script_that_keeps_failing_on_an_app_stops_being_retried(ran):
    ran.reply = "Error: System Events got an error: Can't get sheet 1 of window 1."

    async def go():
        with grants.task(grants.new_task()):
            outs = [await registry().invoke("applescript", {"script": s}, ToolContext()) for s in CHESS_READS[:3]]
            other = await registry().invoke("applescript", {"script": 'tell application "Music" to get sound volume'}, ToolContext())
        return outs, other

    from neo.tools import load_all

    load_all()
    outs, other = asyncio.run(go())
    assert len(ran) == 3  # two failures on Chess, then the third never ran; Music's did
    assert "failed 2 times on Chess" in outs[2].text and "screenshot" in outs[2].text
    assert other.text.startswith("Error: System Events")  # Music is another app: tried


def test_outside_a_task_failures_are_not_counted(ran):
    ran.reply = "Error: nope"
    for s in CHESS_READS[:3]:
        _script(s)
    assert len(ran) == 3


# ---- Accessibility off -------------------------------------------------------------------------


@pytest.fixture
def ax_off(monkeypatch):
    from neo.tools.computer import ax

    monkeypatch.setattr(ax, "is_trusted", lambda: False)


@pytest.mark.parametrize(
    "name, args",
    [
        ("ax_tree", {"app": "Chess"}),
        ("ax_find", {"query": "New Game"}),
        ("ax_press", {"id": "e1"}),
        ("ax_set_value", {"id": "e1", "value": "x"}),
        ("click", {"x": 10, "y": 10}),
        ("drag", {"x1": 1, "y1": 1, "x2": 2, "y2": 2}),
        ("scroll", {"dy": -5}),
        ("type_text", {"text": "hi"}),
        ("hotkey", {"keys": "cmd+n"}),
        ("click_text", {"label": "New Game"}),
        ("applescript", {"script": CHESS_READS[0]}),
    ],
)
def test_without_accessibility_the_tools_that_need_it_ask_the_user(ax_off, ran, name, args, monkeypatch):
    from neo.tools import load_all
    from neo.tools.computer import inp

    load_all()
    for f in ("click", "drag", "scroll", "hotkey", "type_text"):
        monkeypatch.setattr(inp, f, lambda *a, **k: pytest.fail("an event was posted"))
    out = asyncio.run(registry().invoke(name, args, ToolContext(user_text="x")))
    assert out.halt and out.text.startswith(confirm.NEEDS_USER)
    assert "Accessibility" in out.text and "Privacy & Security" in out.text
    assert ran == []


def test_without_accessibility_scripts_that_dont_touch_the_ui_still_run(ax_off, ran):
    out = _script('tell application "Music" to get name of current track')
    assert out.ok and len(ran) == 1


def test_accessibility_is_asked_about_once_per_task(ax_off):
    from neo.tools import load_all

    load_all()

    async def go():
        with grants.task(grants.new_task()):
            return [await registry().invoke("ax_tree", {}, ToolContext()) for _ in range(2)]

    first, second = asyncio.run(go())
    assert first.halt and first.text.startswith(confirm.NEEDS_USER)
    assert not second.halt and second.text.startswith("Error:") and "already been told" in second.text


def test_osascript_refusing_assistive_access_is_the_same_question(ran):
    ran.reply = "Error: System Events got an error: osascript is not allowed assistive access. (-1719)"
    out = _script(CHESS_READS[0])
    assert out.halt and out.text.startswith(confirm.NEEDS_USER) and "Accessibility" in out.text


# ---- grants: the module and the gate -----------------------------------------------------------


def _audit_lines() -> list[dict]:
    from neo.config import settings

    try:
        return [json.loads(ln) for ln in settings().audit_log.read_text().splitlines()]
    except FileNotFoundError:
        return []


def test_a_grant_covers_the_same_tool_on_the_same_app_in_the_same_task_only():
    confirm.cancel("test")
    t = grants.new_task()
    chess = grants.Target("Chess", "", ("ax_tree",))
    with grants.task(t):
        assert isinstance(confirm.gated("applescript", {"script": "a"}, "run an AppleScript in Chess", target=chess), str)
        p = confirm.peek()
        assert p.task == t and confirm.card(p)["scopes"] == ["task"] and confirm.card(p)["app"] == "Chess"
        assert grants.grant(t, "applescript", p.target, p.action_id) is not None
        confirm.cancel("test")
        # the next call on Chess: no card
        assert confirm.gated("applescript", {"script": "b"}, "run an AppleScript in Chess", target=chess) == {"script": "b"}
        assert confirm.peek() is None
        # another app, another tool, a dangerous call, new untrusted reading: a card each
        for tool, target in [
            ("applescript", grants.Target("Finder", "", ("ax_tree",))),
            ("zz_other", chess),
            ("applescript", grants.Target("Chess", "it runs code or raw Apple events", ("ax_tree",))),
            ("applescript", grants.Target("Chess", "", ("ax_tree", "web_fetch"))),
            ("applescript", grants.Target("", "", ("ax_tree",))),
        ]:
            assert isinstance(confirm.gated(tool, {"script": "c"}, f"x {tool} {target}", target=target), str)
            confirm.cancel("test")
        # a gate that always asks (a send) is never covered
        assert isinstance(confirm.gated("applescript", {"script": "d"}, "run", always=True, target=chess), str)
        confirm.cancel("test")
        # the tool's own reading (applescript is untrusted) doesn't void the grant
        mine = grants.Target("Chess", "", ("ax_tree", "applescript"))
        assert confirm.gated("applescript", {"script": "e"}, "run", target=mine) == {"script": "e"}
    # another task: nothing covered
    with grants.task(grants.new_task()):
        assert isinstance(confirm.gated("applescript", {"script": "f"}, "run", target=chess), str)
        confirm.cancel("test")
    # no task at all (a direct command, a Live call): nothing covered, nothing offered
    assert isinstance(confirm.gated("applescript", {"script": "g"}, "run", target=chess), str)
    assert confirm.card(confirm.peek())["scopes"] == []
    confirm.cancel("test")
    grants.end(t)
    with grants.task(t):  # the grant ended with its task
        assert isinstance(confirm.gated("applescript", {"script": "h"}, "run", target=chess), str)
    confirm.cancel("test")
    statuses = [ln["status"] for ln in _audit_lines()]
    assert "granted for this task" in statuses and "task grant ended" in statuses
    assert sum(s.startswith("ran under the task grant") for s in statuses) == 2


def test_cards_that_no_grant_could_cover_offer_no_task_scope():
    confirm.cancel("test")
    with grants.task(grants.new_task()):
        confirm.gated("shell", {"command": "rm x"}, "run rm x")  # no target: never grantable
        assert confirm.card(confirm.peek())["scopes"] == []
        confirm.gated("applescript", {"script": "x"}, "run", target=grants.Target("Chess", "keystrokes go…"))
        assert confirm.card(confirm.peek())["scopes"] == []
    confirm.cancel("test")


def test_a_grant_never_reaches_another_job_running_at_the_same_time():
    confirm.cancel("test")
    t = grants.new_task()
    chess = grants.Target("Chess")
    grants.grant(t, "applescript", chess, "applescript:x")

    async def job(task_id):
        with grants.task(task_id):
            await asyncio.sleep(0)
            r = confirm.gated("applescript", {"script": task_id}, "run", target=chess)
            confirm.cancel("test")
            return isinstance(r, dict)

    async def both():
        return await asyncio.gather(job(t), job(grants.new_task()))

    assert asyncio.run(both()) == [True, False]


# ---- the session: "yes for this task" -------------------------------------------------------------


def _reflex():
    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)

    return R()


class Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return self.turns.pop(0)

    async def stream(self, messages, **kw):
        yield ""


@pytest.fixture
def session_env(monkeypatch, tmp_path, ran):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()

    def look(a, c):
        return "App: Chess — window “Game”\n  e1 Window"

    monkeypatch.setitem(registry()._tools, "zz_look", RegisteredTool("zz_look", "t", _OBJ, look, untrusted=True))
    yield ran


def _call(i, script):
    return ToolCall(str(i), "applescript", {"script": script})


NEW_GAME = 'tell application "System Events" to tell process "Chess" to click menu item "New Game" of menu 1 of menu bar item "Game" of menu bar 1'
PICK_WHITE = 'tell application "System Events" to tell process "Chess" to click pop up button 1 of sheet 1 of window 1'
START = 'tell application "System Events" to tell process "Chess" to click button "Start" of sheet 1 of window 1'


def test_yes_for_this_task_lets_the_rest_of_the_chess_task_through(session_env, monkeypatch):
    ran = session_env
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("0", "zz_look", {})]),  # reads the screen: the request is tainted
            Turn(tool_calls=[_call(1, CHESS_READS[0])]),  # a read after untrusted reading: a card
            # after the yes for this task:
            Turn(tool_calls=[_call(2, NEW_GAME)]),
            Turn(tool_calls=[_call(3, CHESS_READS[2])]),
            Turn(tool_calls=[_call(4, PICK_WHITE)]),
            Turn(tool_calls=[_call(5, START)]),
            Turn(tool_calls=[_call(6, 'tell application "Finder" to get name of window 1')]),  # another app
        ]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)
    cards: list = []

    async def go():
        bus().subscribe(lambda e: cards.append(e.data) if e.type == "confirm" else None)
        s = Session(reflex=_reflex())
        r = await s.handle("zz chess as white", from_user=True)
        await asyncio.sleep(0)
        assert r.route == "confirm" and "read Chess" in r.text
        assert cards[-1]["scopes"] == ["task"] and cards[-1]["app"] == "Chess"
        task = confirm.peek().task
        r = await s.handle("yes for this task", from_user=True)
        return s, r, task

    s, r, task = asyncio.run(go())
    # Every Chess script ran; Finder's (another app) is the next question.
    assert ran == [CHESS_READS[0], NEW_GAME, CHESS_READS[2], PICK_WHITE, START]
    assert r.route == "confirm" and "Finder" in r.text
    assert confirm.peek().task == task  # still the same task
    assert [g.app for g in grants.active(task)] == ["Chess"]
    statuses = [ln["status"] for ln in _audit_lines()]
    assert statuses.count("granted for this task") == 1
    assert sum(st.startswith("ran under the task grant") for st in statuses) == 4


def test_a_plain_yes_covers_only_the_card(session_env, monkeypatch):
    ran = session_env
    brain = Brain([Turn(tool_calls=[_call(1, NEW_GAME)]), Turn(tool_calls=[_call(2, PICK_WHITE)])])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        return await s.handle("yes", from_user=True)

    r = asyncio.run(go())
    assert ran == [NEW_GAME] and r.route == "confirm" and confirm.peek().params["script"] == PICK_WHITE
    # The second card of the same kind says how to stop the questions.
    assert "yes for this task" in r.text


def test_the_cards_button_can_answer_for_the_task(session_env, monkeypatch):
    ran = session_env
    brain = Brain([Turn(tool_calls=[_call(1, NEW_GAME)]), Turn(tool_calls=[_call(2, PICK_WHITE)]), Turn(text="Done.")])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        task = confirm.peek().task
        r = await s.answer(True, action_id=confirm.peek().action_id, scope="task")
        return r, task

    r, task = asyncio.run(go())
    assert ran == [NEW_GAME, PICK_WHITE] and r.route == "agent" and confirm.peek() is None
    assert grants.active(task) == []  # the task is over: so is its grant


def test_a_dangerous_call_in_a_granted_task_still_asks(session_env, monkeypatch):
    ran = session_env
    send = 'tell application "System Events" to tell process "Chess" to click button "Send" of window 1'
    shell = 'tell application "Chess" to do shell script "id"'
    brain = Brain([Turn(tool_calls=[_call(1, NEW_GAME)]), Turn(tool_calls=[_call(2, send)])])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        return await s.handle("haan sab allow karo", from_user=True)

    r = asyncio.run(go())
    assert ran == [NEW_GAME] and r.route == "confirm" and confirm.peek().params["script"] == send
    assert script_facts(shell).danger and script_facts(send).danger


def test_new_untrusted_reading_after_the_grant_asks_again(session_env, monkeypatch):
    ran = session_env
    brain = Brain(
        [
            Turn(tool_calls=[_call(1, NEW_GAME)]),
            Turn(tool_calls=[ToolCall("2", "zz_look", {})]),  # reads something untrusted after the yes
            Turn(tool_calls=[_call(3, PICK_WHITE)]),
        ]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        return await s.handle("is task ke liye haan", from_user=True)

    r = asyncio.run(go())
    assert ran == [NEW_GAME] and r.route == "confirm" and confirm.peek().params["script"] == PICK_WHITE


def test_a_grant_ends_with_its_task_and_never_reaches_the_next_request(session_env, monkeypatch):
    ran = session_env
    brain = Brain(
        [
            Turn(tool_calls=[_call(1, NEW_GAME)]),
            Turn(text="Started a new game."),
            Turn(tool_calls=[_call(2, PICK_WHITE)]),  # the next request
        ]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz a new chess game", from_user=True)
        r1 = await s.handle("yes to all", from_user=True)
        r2 = await s.handle("zz now white in chess", from_user=True)
        return r1, r2

    r1, r2 = asyncio.run(go())
    assert r1.route == "agent" and ran == [NEW_GAME]
    assert r2.route == "confirm" and confirm.peek().params["script"] == PICK_WHITE


def test_no_ends_the_task_and_its_grants(session_env, monkeypatch):
    brain = Brain([Turn(tool_calls=[_call(1, NEW_GAME)])])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        task = confirm.peek().task
        grants.grant(task, "applescript", grants.Target("Chess"), "x")
        await s.handle("no", from_user=True)
        return task

    task = asyncio.run(go())
    assert grants.active(task) == [] and confirm.peek() is None


def test_without_accessibility_the_chess_run_stops_at_the_first_look(session_env, monkeypatch, ax_off):
    ran = session_env
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("1", "ax_tree", {"app": "Chess"})]),
            Turn(tool_calls=[_call(2, CHESS_READS[0])]),  # never reached
        ]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)
    r = asyncio.run(Session(reflex=_reflex()).handle("zz chess as white", from_user=True))
    assert brain.calls == 1 and ran == [] and confirm.peek() is None
    assert r.result.stopped == "needs_user" and "Accessibility" in r.text


# ---- words -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "yes for this task", "Yes, for this task", "yes for the whole task", "ok for the rest of this task",
        "yes to all", "yes to everything", "allow all", "for this task", "haan sab allow karo",
        "sab allow kar do", "is task ke liye haan", "haan is task ke liye", "iss task ke liye haan",
        "इस टास्क के लिए हाँ", "सब के लिए हाँ",
    ],
)
def test_yes_for_this_task(text):
    assert answer_of(text) is True and scope_of(text) == "task"


@pytest.mark.parametrize("text", ["yes", "haan", "ok do it", "yes please", "हाँ"])
def test_a_plain_yes_is_for_this_card(text):
    assert answer_of(text) is True and scope_of(text) == "once"


@pytest.mark.parametrize("text", ["yes for this task, no wait", "nahi, sab allow mat karo", "no to all"])
def test_a_no_still_wins(text):
    assert answer_of(text) is False and scope_of(text) == "once"


# ---- the prompt -------------------------------------------------------------------------------


def test_the_prompt_steers_canvas_apps_to_screenshot_and_click():
    from neo.agent.prompt import persona

    p = persona()
    assert "Chess" in p and "screenshot" in p and "coordinates" in p
    assert "more than twice" in p and "Accessibility is off" in p
    assert "System Events" in p  # never drive an app's window through its scripts
    d = registry().get("applescript").description if registry().get("applescript") else ""
    assert not d or "ax_tree" in d


@pytest.mark.parametrize(
    "verb",
    [
        "click", "perform", "keystroke", "key", "select", "do", "run", "load", "store", "open", "close", "quit",
        "delete", "make", "move", "duplicate", "save", "send", "activate", "launch", "display", "say", "beep",
        "copy", "put", "set", "empty", "erase", "restart", "sleep", "play", "pause", "next", "previous", "reveal",
        "print", "choose", "reply", "forward", "mount", "eject",
    ],
)
def test_no_command_passes_for_a_word_of_reading(verb):
    assert verb not in policy._AS_READ and verb not in policy._AS_PHRASE_WORDS and verb in policy._AS_NEVER_LOCAL


# ---- more of the reader's edges -----------------------------------------------------------------


@pytest.mark.parametrize(
    "script, reads",
    [
        ("set {x, y} to {1, 2}\nreturn x & y", True),
        ("set {a, b} to {1, 2}", False),  # "a" is the article, not a variable
        ('set "{" x "}" to 1', False),
        # a variable from outside an app's tell could be one of that app's commands (VLC's volumeUp)
        ('set theList to {}\ntell application "Finder"\n  set end of theList to name of window 1\nend tell\nreturn theList', False),
        ('tell application "Finder"\n  set theList to {}\n  set end of theList to name of window 1\nend tell\nreturn theList', True),
        # a variable spelt like a command System Events has: never a variable
        ('set click to 1\ntell application "System Events" to tell process "Chess" to get click', False),
        ('set confirm to 1\ntell application "System Events" to get confirm', False),
        # a variable named like a property: `set` it inside a tell and it's the app's property
        ('tell application "Music" to set shuffling to true', False),
        ('tell application "System Events" to tell process "Chess" to set visible to false', False),
        # tell <variable>: whatever it holds
        ('set p to process "Chess"\ntell application "System Events" to tell p to get name of windows', False),
        ('set appName to "Chess"\ntell application "System Events" to tell process appName to get name of windows', False),
        # a block closed by the wrong end
        ('tell application "Finder"\n  get name of window 1\nend repeat', False),
        ('repeat 3 times\n  get name of window 1\nend', True),
        ("exit repeat", True),
        ('get name of window 1 of application "Finder"', True),
        ("get x", False),  # a variable nobody set: a command, as far as anyone can tell
    ],
)
def test_the_readers_edges(script, reads):
    assert script_is_read_only(script) is reads, script_facts(script)


def test_a_process_picked_by_a_variable_is_neither_read_only_nor_grantable():
    f = script_facts('set appName to "Chess"\ntell application "System Events" to tell process appName to get name of windows')
    assert not f.read_only and f.app == "" and "picks" in f.danger


def test_the_overlay_can_answer_for_the_task(monkeypatch):
    from neo import app as app_mod

    a = app_mod.App()
    calls = []

    async def answer(yes, **kw):
        calls.append(kw)

    monkeypatch.setattr(a.session, "answer", answer)
    asyncio.run(a.on_command({"cmd": "confirm", "yes": True, "action_id": "applescript:1", "scope": "task"}))
    asyncio.run(a.on_command({"cmd": "confirm", "yes": True, "action_id": "applescript:2"}))
    asyncio.run(a.on_command({"cmd": "confirm", "yes": True, "action_id": "applescript:3", "scope": "forever"}))
    assert [c["scope"] for c in calls] == ["task", None, None]


# ---- review: what a task grant may cover --------------------------------------------------------
# A grant's reach comes from the script's own text, which a steered model writes. So the app a
# script acts on is worked out statement by statement, and a grant only ever covers reading and
# clicking in one named app's window (an allowlist), never System Events' own actions, an app picked
# while the script runs, or an app that runs commands.

LOGIN_ITEM = (
    'tell application "System Events"\n  get name of process "Chess"\n'
    '  make new login item at end of login items with properties {path:"/Users/me/Downloads/Thing.app", hidden:true}\n'
    "end tell"
)


@pytest.mark.parametrize(
    "script",
    [
        LOGIN_ITEM,
        'tell application "System Events"\n  get name of process "Chess"\n  set dark mode of appearance preferences to true\nend tell',
        'tell application "System Events" to tell process "Chess" to make new login item at end of login items',
        'tell application "System Events"\n  get name of every disk\n  get name of windows of process "Chess"\nend tell',
        'tell application "System Events"\n  tell process "Chess" to get name of windows\n  get name of every disk\nend tell',
        'tell application "System Events"\n  tell process "Chess" to click button 1 of window 1\n  sleep\nend tell',
        'tell application "System Events" to tell process "Chess" to keystroke "e"',
        'tell application "System Events" to tell process "Chess" to click at {10, 10}',
    ],
)
def test_a_process_name_never_relabels_system_events_own_actions(script):
    f = script_facts(script)
    assert f.app != "Chess" and f.danger, f


@pytest.mark.parametrize(
    "script",
    [
        'tell application "System Events"\n  get name of process "Chess"\n'
        "  click button 1 of window 1 of item 1 of (application processes whose frontmost is true)\nend tell",
        'tell application "Chess" to get name\n'
        'tell application "System Events" to click button 1 of window 1 of item 1 of (processes whose frontmost is true)',
        'tell application "System Events" to get name of every process whose background only is false',
        'tell application "System Events" to get name of windows of (processes whose frontmost is true)',
        'tell application "System Events" to get name of every application process',
        'tell application "System Events" to repeat with p in processes\n  get name of p\nend repeat',
        "get name of every application",
        "get path to frontmost application",
        'tell application "System Events" to get name of window 1 of (first process whose frontmost is true)',
        'set appName to "Chess"\ntell application "System Events" to tell process appName to get name of windows',
    ],
)
def test_an_app_chosen_while_the_script_runs_is_never_read_only_nor_grantable(script):
    f = script_facts(script)
    assert not f.read_only and f.app == "" and f.danger, f


@pytest.mark.parametrize(
    "script",
    [
        'tell application "iTerm" to create window with default profile command "rm -rf ~/Documents"',
        'tell application "iTerm2" to tell current session of current window to split vertically with default profile '
        'command "rm -rf ~"',
        'tell application "Terminal" to set custom title of tab 1 of window 1 to "x"',
        'tell application "Terminal" to activate',
        'tell application id "com.googlecode.iterm2" to activate',
        'tell application "System Events" to tell process "Terminal" to click button 1 of window 1',
        'tell application "System Events" to tell process "iTerm2" to get name of windows',
        'tell application "Script Editor" to activate',
        'tell application "Shortcuts Events" to get name of every shortcut',
        'tell application "System Events" to tell process "NEO" to click button 1 of window 1',
        # beyond reading and clicking in a window: not on the list a grant may cover
        'tell application "Chess" to frobnicate board 1',
        'tell application "Music" to back track',
        'tell application "Notes" to make new note with properties {name:"x"}',
    ],
)
def test_a_grant_covers_only_reading_and_clicking_in_an_ordinary_app(script):
    assert script_facts(script).danger, script_facts(script)


@pytest.mark.parametrize(
    "script",
    [
        'tell application "System Events" to tell process "Chess" to click menu item "New Game" of menu 1 of menu bar 1',
        'tell application "System Events" to click button 1 of window 1 of process "Chess"',
        'tell application "System Events" to tell process "Chess" to perform action "AXPress" of button 1 of window 1',
        'tell application "System Events" to tell process "Chess" to select row 2 of table 1 of window 1',
        'tell application "System Events" to set frontmost of process "Chess" to true',
        'tell application "System Events" to tell process "Chess" to set value of text field 1 of window 1 to "x"',
        'tell application "Chess" to activate',
        *CHESS_READS,
    ],
)
def test_reading_and_clicking_in_one_apps_window_can_be_granted(script):
    f = script_facts(script)
    assert f.app == "Chess" and f.danger == "", f


@pytest.mark.parametrize(
    "script",
    [
        'tell application "Macintosh HD:Users:me:Downloads:Thing.app" to get name of every window',
        'tell application "/Users/me/Downloads/Thing.app" to get name of every window',
        'tell application "Thing.app" to get name',
        'get name of window 1 of application "~/Downloads/Thing.app"',
        'tell application "Music" to back track',
        # a variable from outside the tell may be one of the app's own commands (VLC's volumeUp)
        'set volumeUp to 0\ntell application "VLC"\n  get volumeUp\nend tell',
        'set list2 to 0\ntell application "Music" to get list2',
    ],
)
def test_more_scripts_that_are_not_read_only(script):
    assert not script_is_read_only(script), script_facts(script)


def test_an_app_named_by_its_path_is_never_granted():
    f = script_facts('tell application "/Users/me/Downloads/Thing.app" to activate')
    assert f.danger and f.app == ""


@pytest.mark.parametrize(
    "script, ui",
    [
        ('tell application "System Events" to get name of every folder action', False),
        ('tell application "Music" to get name of current track -- not a System Events window', False),
        ('tell application "Music" to get "System Events window"', False),
        ('(* tell application "System Events" to click button 1 *)\ntell application "Music" to get sound volume', False),
        ('tell application "System Events" to get name of every disk', False),
        ('tell application "System Events" to tell process "Chess" to get name of windows', True),
        ('tell application "System Events" to click button 1 of window 1 of process "Chess"', True),
        ('tell application "System Events" to key code 36', True),
    ],
)
def test_ui_scripting_is_read_from_the_code_not_its_comments_or_strings(script, ui):
    assert script_facts(script).ui is ui


def test_without_accessibility_a_system_events_script_that_needs_none_still_runs(ax_off, ran):
    out = _script('tell application "System Events" to get name of every folder action')
    assert out.ok and len(ran) == 1


def _granted_chess_task(taint: tuple[str, ...] = ()) -> str:
    t = grants.new_task()
    grants.grant(t, "applescript", grants.Target("Chess", "", taint), "applescript:granted")
    return t


@pytest.mark.parametrize(
    "script",
    [
        LOGIN_ITEM,
        'tell application "System Events"\n  get name of process "Chess"\n'
        "  click button 1 of window 1 of item 1 of (application processes whose frontmost is true)\nend tell",
    ],
)
def test_a_chess_grant_never_covers_system_events_or_the_frontmost_app(ran, script):
    from neo.tools import load_all

    load_all()
    t = _granted_chess_task()

    async def go():
        with grants.task(t):
            return await registry().invoke("applescript", {"script": script}, ToolContext(user_text="x"))

    out = asyncio.run(go())
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and ran == []
    assert confirm.card(confirm.peek())["scopes"] == []


def test_a_grant_ends_at_new_untrusted_reading_even_by_a_tool_already_read(ran):
    """Taint is a set of names; the grant counts every untrusted result after the yes."""
    from neo.tools import load_all

    load_all()
    t = _granted_chess_task()
    registry()._tools["zz_look"] = RegisteredTool("zz_look", "t", _OBJ, lambda a, c: "mail from Mallory", untrusted=True)
    try:

        async def go():
            ctx = ToolContext(user_text="x", taint=["zz_look"])
            with grants.task(t):
                grants.grant(t, "applescript", grants.Target("Chess", "", ("zz_look",)), "applescript:yes")
                first = await registry().invoke("applescript", {"script": PICK_WHITE}, ctx)
                await registry().invoke("zz_look", {}, ctx)  # the same tool reads something new
                second = await registry().invoke("applescript", {"script": START}, ctx)
            return first, second

        first, second = asyncio.run(go())
    finally:
        registry()._tools.pop("zz_look", None)
    assert first.ok and not first.halt  # covered: nothing new read since the yes
    assert second.halt and ran == [PICK_WHITE]  # a card again


def test_a_granted_calls_own_reading_keeps_the_grant(ran):
    from neo.tools import load_all

    load_all()
    t = _granted_chess_task(("ax_tree",))

    async def go():
        ctx = ToolContext(user_text="x", taint=["ax_tree"])
        with grants.task(t):
            return [await registry().invoke("applescript", {"script": s}, ctx) for s in (NEW_GAME, CHESS_READS[2], START)]

    outs = asyncio.run(go())
    assert all(o.ok and not o.halt for o in outs) and len(ran) == 3


def test_new_untrusted_reading_by_a_tool_in_the_baseline_asks_again_in_the_session(session_env, monkeypatch):
    ran = session_env
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("0", "zz_look", {})]),  # the request is tainted
            Turn(tool_calls=[_call(1, NEW_GAME)]),  # a card
            Turn(tool_calls=[_call(2, PICK_WHITE)]),  # covered
            Turn(tool_calls=[ToolCall("3", "zz_look", {})]),  # reads the screen again: new content
            Turn(tool_calls=[_call(4, START)]),  # a card again
        ]
    )
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        await s.handle("zz chess as white", from_user=True)
        return await s.handle("yes for this task", from_user=True)

    r = asyncio.run(go())
    assert ran == [NEW_GAME, PICK_WHITE] and r.route == "confirm" and confirm.peek().params["script"] == START


@pytest.mark.parametrize(
    "text",
    [
        "to all", "for all", "for everything", "to everything", "sab ke liye", "sabhi ke liye", "is task ke liye",
        "poore task ke liye", "सब के लिए", "इस टास्क के लिए",
    ],
)
def test_scope_words_alone_are_no_yes(text):
    assert answer_of(text) is None and scope_of(text) == "once"


@pytest.mark.parametrize(
    "text", ["yes to all", "haan sab ke liye", "sab ke liye haan", "ok for everything", "haan poore task ke liye"]
)
def test_scope_words_with_a_yes_are_a_yes_for_the_task(text):
    assert answer_of(text) is True and scope_of(text) == "task"


def test_erasing_everything_ends_every_grant(monkeypatch):
    from neo import app as app_mod

    t = _granted_chess_task()
    assert grants.active(t)
    a = app_mod.App()

    async def nothing():
        return {}

    monkeypatch.setattr("neo.privacy.forget_everything", nothing)
    asyncio.run(a.erase())
    assert grants.active(t) == []


@pytest.mark.parametrize(
    "script",
    [
        # the Apple menu is the Mac's, whichever app's menu bar it is reached through
        'tell application "System Events" to tell process "Chess" to click menu item "Restart…" of menu 1 of '
        'menu bar item "Apple" of menu bar 1',
        'tell application "System Events" to tell process "Chess" to click menu item 9 of menu 1 of menu bar item 1 of menu bar 1',
        'tell application "System Events" to tell process "Chess" to click menu item "Lock Screen" of menu 1 of menu bar item 2',
        # Services run other apps' workflows (a Terminal at a folder, a shell workflow)
        'tell application "System Events" to tell process "Chess" to click menu item "New Terminal at Folder" of menu 1 '
        'of menu item "Services" of menu 1 of menu bar item 2 of menu bar 1',
    ],
)
def test_the_macs_own_menus_are_never_granted(script):
    assert script_facts(script).danger, script_facts(script)


def test_system_events_by_its_bundle_id_is_system_events():
    f = script_facts('tell application id "com.apple.systemevents" to tell process "Chess" to get name of windows')
    assert f.ui and f.app == "Chess" and f.read_only
    f = script_facts('tell application id "com.apple.systemevents" to set dark mode of appearance preferences to true')
    assert f.danger and f.app == "com.apple.systemevents"


def test_an_app_named_in_a_tell_counts_even_when_nothing_inside_acts_on_it():
    f = script_facts('tell application "Chess"\n  tell application "Finder" to activate\nend tell')
    assert f.app == "" and "more than one app" in f.danger
