"""in_front (neo2.core.tools.context_now): "this" — what the user selected, copied or has in front of
them, read only when their own words ask for it, fenced as untrusted content, and never kept.

Accessibility, the pasteboard, the window server and Finder are all fakes here: nothing reads this
Mac's screen or clipboard (the real readers are refused while effects are off: neo2/core/effects.py).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo2.core.agent.registry import USERS_WORDS, ToolContext, catalog, registry
from neo2.core.reflex.schema import Decision

ASKED = "summarize what I copied"


# ---- fakes ----------------------------------------------------------------------------------------


class Pasteboard:
    """NSPasteboard as in_front asks it: its types first, its contents only then."""

    def __init__(self, types=(), text=None, behavior=0):
        self._types, self.text, self.behavior = list(types), text, behavior
        self.read: list[str] = []

    def types(self):
        return self._types

    def accessBehavior(self):
        return self.behavior

    def stringForType_(self, t):
        self.read.append(t)
        return self.text


class World:
    """The apps, the focused element and the pasteboard the tool sees."""

    def __init__(self):
        self.front = ("Notes", 600)
        self.windows = [("Notes", 600)]
        self.regular = {600, 700}
        self.neo = {400}
        self.protected: set[str] = set()
        self.focused: object | None = SimpleNamespace(
            role="AXTextArea", sub="", sel="", marker="", parent=None
        )
        self.window = ("Groceries", "file:///Users/someone/Documents/Lists/Groceries%20list.txt")
        self.web = False
        self.pb = Pasteboard(["public.utf8-plain-text"], "Meeting moved to 4 pm on Thursday.")
        self.files: list[str] = []
        self.image: tuple[int, int] | None = None
        self.finder = ""


@pytest.fixture
def world(monkeypatch):
    from neo2.core.tools import context_now as cn
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import apps, ax, gates

    load_all()
    w = World()
    monkeypatch.setattr(cn, "_front", lambda: w.front)
    monkeypatch.setattr(cn, "_window_owners", lambda: list(w.windows))
    monkeypatch.setattr(ax, "regular", lambda pid: pid in w.regular)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid in w.neo)
    monkeypatch.setattr(gates, "_protected", lambda owner, pid=0: owner in w.protected)
    monkeypatch.setattr(cn, "_focused_in", lambda pid: w.focused)
    monkeypatch.setattr(cn, "_roles", lambda el: (el.role, el.sub))
    monkeypatch.setattr(cn, "_selected_text", lambda el: el.sel)
    monkeypatch.setattr(cn, "_window_facts", lambda pid: w.window)
    monkeypatch.setattr(cn, "_shows_web", lambda pid: w.web)
    monkeypatch.setattr(cn, "_pasteboard", lambda: w.pb)
    monkeypatch.setattr(cn, "_file_names", lambda pb: list(w.files))
    monkeypatch.setattr(cn, "_image_size", lambda pb: w.image)
    w.scripts = []

    async def osa(script, timeout=30):
        w.scripts.append(script)
        return w.finder

    monkeypatch.setattr(apps, "osascript", osa)
    return w


def _call(what="auto", words=ASKED, origin="agent", user_text=None, extras=None):
    ctx = ToolContext(
        user_text=words if user_text is None else user_text,
        origin=origin,
        extras={USERS_WORDS: words} if extras is None and origin in ("agent", "user") else (extras or {}),
    )
    out = asyncio.run(registry().invoke("in_front", {"what": what}, ctx))
    return out, ctx


# ---- the origin rule: the user's own words, or nothing ---------------------------------------------


def test_the_users_words_read_the_clipboard_fenced_and_tainting(world):
    out, ctx = _call()
    assert out.ok and "Meeting moved to 4 pm on Thursday." in out.text
    assert ctx.taint == ["in_front"]  # what was read counts: a send or a link next asks first
    t = registry().get("in_front")
    assert t.untrusted and t.read_only and t.parallel_safe and "macos" in t.requires


@pytest.mark.parametrize("asked", ["what's the weather", "open notes", "what is this?", "ye kya hai"])
def test_a_live_call_without_the_users_words_is_refused(world, asked):
    out, ctx = _call(origin="live", words=asked)
    assert not out.ok and "only when you ask" in out.text
    assert world.pb.read == [] and ctx.taint == []


def test_a_live_call_with_the_users_words_runs(world):
    out, _ = _call(origin="live", words="Neo, translate what I just copied into Hindi")
    assert out.ok and "Meeting moved" in out.text


def test_a_live_call_whose_only_words_are_its_own_arguments_is_refused(world):
    # live.py hands a direct tool `asked or str(args)`: with no sentence, the call's own arguments.
    out, _ = _call(
        origin="live", words="", user_text=str({"what": "clipboard", "why": "summarize what I copied"})
    )
    assert not out.ok and world.pb.read == []


def test_an_agents_call_needs_the_users_own_words_marked_by_the_session(world):
    out, _ = _call(origin="agent", words=ASKED, extras={})  # a model's goal, a Live relay: no USERS_WORDS
    assert not out.ok and "only when you ask" in out.text
    out, _ = _call(origin="agent", words=ASKED, extras={USERS_WORDS: "book a table for two"})
    assert not out.ok  # the user's words don't ask for it
    assert world.pb.read == []


def test_an_approval_replay_never_reads(world):
    out, _ = _call(origin="approval", words=ASKED, extras={USERS_WORDS: ASKED})
    assert not out.ok and world.pb.read == []


def test_switched_off_it_refuses(world, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    out, _ = _call()
    assert not out.ok and "off in NEO's settings" in out.text and world.pb.read == []


def test_an_embedded_agent_never_reads(world, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_mode", lambda: True)
    out, _ = _call()
    assert not out.ok and world.pb.read == []


# ---- the clipboard -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        "org.nspasteboard.ConcealedType",
        "org.nspasteboard.TransientType",
        "org.nspasteboard.AutoGeneratedType",
    ],
)
def test_a_private_copy_is_refused_unread(world, marker):
    world.pb = Pasteboard(["public.utf8-plain-text", marker], "hunter2")
    out, ctx = _call()
    assert not out.ok and "hunter2" not in out.text and world.pb.read == [] and ctx.taint == []


def test_a_secret_looking_copy_is_refused(world):
    world.pb = Pasteboard(
        ["public.utf8-plain-text"], "my api key is sk-proj-abcdefghijklmnopqrstuvwxyz012345"
    )
    out, ctx = _call()
    assert not out.ok and "sk-proj" not in out.text and ctx.taint == []


def test_a_mac_that_never_lets_apps_read_the_clipboard_is_honoured(world):
    from neo2.core.tools import context_now as cn

    world.pb = Pasteboard(["public.utf8-plain-text"], "a note", behavior=cn._always_deny())
    out, _ = _call()
    assert not out.ok and "never" in out.text and world.pb.read == []


def test_copied_files_are_their_names_only(world):
    world.pb = Pasteboard(["public.file-url", "public.utf8-plain-text"], "/Users/someone/Secret Plans/a.pdf")
    world.files = ["a.pdf", "b.txt"]
    out, _ = _call()
    assert out.ok and "a.pdf" in out.text and "b.txt" in out.text and "Secret Plans" not in out.text


def test_a_copied_picture_is_its_size(world):
    world.pb = Pasteboard(["public.png"])
    world.image = (1200, 800)
    out, _ = _call()
    assert out.ok and "image" in out.text and "1200×800" in out.text


def test_an_empty_clipboard(world):
    world.pb = Pasteboard([])
    out, ctx = _call()
    assert not out.ok and "empty" in out.text and ctx.taint == []


def test_the_clipboard_only_on_the_users_word_for_it(world):
    # The user said "this"; a model asking for the clipboard gets told, and nothing is read.
    world.focused.sel = "Bonjour tout le monde"
    out, _ = _call(what="clipboard", words="translate this")
    assert not out.ok and "clipboard" in out.text and world.pb.read == []


def test_the_text_is_capped(world):
    world.pb = Pasteboard(["public.utf8-plain-text"], "word " * 3000)
    out, _ = _call()
    assert out.ok and len(out.text) <= 4000


# ---- the selection ------------------------------------------------------------------------------------


def test_the_selection_in_the_users_app(world):
    world.focused.sel = "The quick brown fox."
    out, _ = _call(words="translate the selected text")
    assert out.ok and "The quick brown fox." in out.text and "Notes" in out.text
    assert world.pb.read == []  # the selection was asked for, not the clipboard


@pytest.mark.parametrize("role, sub", [("AXSecureTextField", ""), ("AXTextField", "AXSecureTextField")])
def test_a_password_field_is_never_read(world, role, sub):
    world.focused = SimpleNamespace(role=role, sub=sub, sel="hunter2", marker="", parent=None)
    out, ctx = _call(words="explain the selected text")
    assert not out.ok and "hunter2" not in out.text and ctx.taint == []


def test_a_protected_app_is_never_read(world):
    world.protected = {"Notes"}
    world.focused.sel = "private words"
    out, _ = _call(words="translate the selected text")
    assert not out.ok and "private words" not in out.text


@pytest.mark.parametrize("front", [("NEO", 400), ("NotchHub", 500)])
def test_with_neo_or_an_accessory_panel_in_front_the_users_app_is_read(world, front):
    """Typed in NEO's window, or in a notch app's panel (an accessory app, not a regular one): the
    selection is the one in the app the user works in behind it."""
    world.front = front
    world.windows = [front, ("Pages", 700), ("Notes", 600)]
    world.focused.sel = "Paragraph two."
    out, _ = _call(words="proofread the selected text")
    assert (
        out.ok and "Pages" in out.text and "NotchHub" not in out.text and "NEO" not in out.text.split(":")[0]
    )


def test_no_app_of_the_users_is_said(world):
    world.front, world.windows = ("NEO", 400), [("NEO", 400)]
    out, _ = _call(words="translate the selected text")
    assert not out.ok and "which app" in out.text


def test_a_select_all_is_capped_and_a_secret_at_the_cut_is_still_left_out(world):
    # The secret's value lands just before the 4,000-character cut.
    world.focused.sel = "word " * 740 + "the password is Tr0ub4dor&3 " + "more " * 4000
    out, _ = _call(words="summarize the selected text")
    assert out.ok and len(out.text) <= 4000 and "Tr0ub4dor" not in out.text and "[secret removed]" in out.text


def test_a_secret_in_the_selection_is_left_out(world):
    world.focused.sel = "Login for the wiki: the password is Tr0ub4dor&3, ask Sam."
    out, _ = _call(words="summarize the selected text")
    assert out.ok and "Tr0ub4dor" not in out.text and "ask Sam" in out.text


# ---- auto: the user's word first, then the selection, then the window ------------------------------


def test_auto_an_explicit_word_wins(world):
    world.focused.sel = "selected words"
    out, _ = _call(what="auto", words="summarize what I copied")
    assert "Meeting moved" in out.text and "selected words" not in out.text
    # The model's guess says the selection: the user's own word for the clipboard wins.
    out, _ = _call(what="selection", words="summarize what I copied")
    assert "Meeting moved" in out.text


def test_auto_this_is_the_selection_when_there_is_one(world):
    world.focused.sel = "selected words"
    out, _ = _call(what="auto", words="translate this")
    assert out.ok and "selected words" in out.text and world.pb.read == []


def test_auto_with_nothing_selected_is_the_window(world):
    out, _ = _call(what="auto", words="explain this")
    assert out.ok and "Groceries" in out.text and "Groceries list.txt" in out.text
    assert "Documents" not in out.text and "/Users" not in out.text  # the document's name, never its folders
    assert world.pb.read == []


def test_nothing_selected_when_the_selection_was_asked_for(world):
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected" in out.text and ctx.taint == []


def test_the_window_of_a_browser_points_at_page_read(world):
    world.web = True
    world.window = ("Deals — Today", "https://shop.example/deals?token=abc")
    out, _ = _call(what="window", words="explain this")
    assert out.ok and "page_read" in out.text and "token" not in out.text


# ---- Finder's selection --------------------------------------------------------------------------------


def test_the_finder_selection_is_names(world):
    world.finder = "report.pdf\nbudget.xlsx"
    out, _ = _call(what="auto", words="summarize these files")
    assert out.ok and "report.pdf" in out.text and "budget.xlsx" in out.text
    assert world.pb.read == []


def test_the_finder_script_only_reads():
    from neo2.core.security.policy import script_facts, script_is_read_only
    from neo2.core.tools import context_now as cn

    assert script_is_read_only(cn.FINDER_SCRIPT)
    assert script_facts(cn.FINDER_SCRIPT).app == "Finder"


def test_finder_errors_are_said_plainly(world):
    world.finder = "Error: Not authorized to send Apple events to Finder. (-1743)"
    out, _ = _call(what="files", words="what are the selected files")
    assert not out.ok and "Finder" in out.text and "-1743" not in out.text
    world.finder = ""
    out, _ = _call(what="files", words="what are the selected files")
    assert not out.ok and "Nothing is selected in Finder" in out.text


# ---- the real readers: refused in tests, never reading this Mac ------------------------------------------


def test_the_real_readers_are_effects():
    from neo2.core import effects
    from neo2.core.tools import context_now as cn

    for reader in (
        cn._front,
        cn._window_owners,
        lambda: cn._focused_in(1),
        lambda: cn._window_facts(1),
        cn._pasteboard,
    ):
        with pytest.raises(effects.EffectRefused):
            reader()


def test_the_selected_text_falls_back_to_a_web_pages_text_markers(monkeypatch):
    """A page's selection (Safari, Chromium, Electron) has no AXSelectedText on the focused element:
    its text-marker range, on it or the web area it is in, has the words."""
    from neo2.core.tools import context_now as cn
    from neo2.core.tools.computer import ax

    area = SimpleNamespace(role="AXWebArea", sel="", marker="Selected on the page.", parent=None)
    leaf = SimpleNamespace(role="AXStaticText", sel="", marker="", parent=area)
    monkeypatch.setattr(ax, "_attrs", lambda el, names: {"AXSelectedText": el.sel, "AXRole": el.role})
    monkeypatch.setattr(
        ax,
        "_attr",
        lambda el, name, default=None: {"AXRole": el.role, "AXParent": el.parent}.get(name, default),
    )
    monkeypatch.setattr(cn, "_marker_text", lambda el: el.marker)
    assert cn._selected_text(leaf) == "Selected on the page."
    plain = SimpleNamespace(role="AXTextArea", sel="Plain selection.", marker="", parent=None)
    assert cn._selected_text(plain) == "Plain selection."


# ---- the catalog --------------------------------------------------------------------------------------


def test_the_catalog_has_it_for_neo_app_only():
    from neo2.core import tools
    from neo2.core.tools import REFUSED, load_all, resolve

    load_all()
    assert catalog().get("in_front") is not None
    assert "neo2.core.tools.context_now" in tools._MAC_ONLY
    assert "in_front" in REFUSED and "in_front" not in resolve("*", platform="darwin")
    with pytest.raises(ValueError, match="NEO.app's user"):
        resolve(("web", "in_front"))


# ---- the Session: the user's own words reach a model that has the tool -------------------------------------


class _Reflex:
    def __init__(self, intent="chat"):
        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.fixture
def session_env(world, monkeypatch):
    from neo2.core import events
    from neo2.core.agent import session as sess_mod
    from neo2.core.providers.base import ToolCall, Turn

    monkeypatch.setattr(events, "_bus", events.EventBus())

    class Chain:
        name = "fake"
        supports_vision = False
        supports_tools = True
        streamed = 0
        calls: list = []

        async def stream(self, messages, *, system="", max_tokens=2048):
            Chain.streamed += 1
            yield "a chat reply"

        async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
            Chain.calls.append((list(messages), [t.name for t in tools or []]))
            if len(Chain.calls) == 1 and "in_front" in Chain.calls[0][1]:
                return Turn(tool_calls=[ToolCall("c1", "in_front", {"what": "auto"})])
            return Turn(text="Your meeting moved to Thursday at 4.")

    Chain.calls = []
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Chain())
    return Chain


def test_the_session_hands_the_users_words_to_a_model_with_in_front_even_when_laya_says_chat(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
    assert r.route == "agent" and session_env.streamed == 0
    first_tools = session_env.calls[0][1]
    assert "in_front" in first_tools and "page_read" in first_tools
    # The tool ran for the user's words; the model got what it read fenced, and the request is tainted.
    second = session_env.calls[1][0]
    read = [res for m in second for res in m.tool_results if res.name == "in_front"]
    assert read and read[0].content.startswith('<untrusted source="in_front">')
    assert "Meeting moved" in read[0].content and r.taint == ("in_front",)


def test_the_session_hook_also_takes_laya_s_one_tool_lane(session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("quick_action"))
    s.reflex.d = Decision(
        "quick_action", 0.95, False, 0.0, False, 0.0, "fake", tool="calendar_add", tool_p=0.99
    )
    r = asyncio.run(s.handle("add this to my calendar", from_user=True))
    assert r.route == "agent" and "in_front" in session_env.calls[0][1]


@pytest.mark.parametrize("text", ["tell me a joke", "explain quantum physics"])
def test_other_words_take_their_old_route(session_env, text):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(text, from_user=True))
    assert r.route == "chat" and session_env.streamed == 1 and session_env.calls == []


def test_a_models_goal_never_takes_the_hook(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, origin="live"))
    assert r.route == "chat" and session_env.calls == []


def test_switched_off_the_hook_is_skipped(session_env, monkeypatch):
    from neo2.core import config
    from neo2.core.agent.session import Session

    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "false")
    monkeypatch.setattr(config, "_settings", None)
    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
    assert r.route == "chat" and session_env.calls == []


def test_a_routine_definition_keeps_its_lane(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(
        Session(reflex=_Reflex("chat")).handle(
            "when I say good night, summarize what I copied", from_user=True
        )
    )
    assert r.route == "chat" and session_env.calls == []


def test_without_the_tool_the_hook_is_skipped(session_env):
    from neo2.core.agent.session import Session

    prev = catalog().remove("in_front")
    try:
        r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
        assert r.route == "chat"
    finally:
        catalog().register(prev)


# ---- Gemini Live ------------------------------------------------------------------------------------------


def test_live_declares_and_explains_in_front_only_while_it_is_there(world, monkeypatch):
    from neo2.core import config
    from neo2.core.voice import live as live_mod

    assert "in_front" in live_mod._direct_tools()
    assert "in_front" in [d.name for d in live_mod._declarations()]
    assert "in_front" in live_mod._live_extra()
    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    assert "in_front" not in live_mod._live_extra()
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.delenv("NEO_CONTEXT_CAPTURE")
    prev = catalog().remove("in_front")
    try:
        assert "in_front" not in live_mod._live_extra()
        assert "in_front" not in [d.name for d in live_mod._declarations()]
    finally:
        catalog().register(prev)


def _live_voice(session=None):
    """A Gemini Live voice with no connection: tool responses land in `sent`."""
    import threading

    import neo2.core.voice.live as live_mod

    sent: list = []

    class FakeLive:
        async def send_tool_response(self, function_responses):
            sent.extend(function_responses)

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = session, {}, {}, set()
    v._live, v._ui_lock, v._call_labels, v._taint, v._dialog = FakeLive(), asyncio.Lock(), {}, [], []
    v._utt_started, v._last_answer, v._lookups, v._seen_calls, v._recent_idem = 0.0, (0.0, ""), {}, {}, {}
    v._speaking, v.spk = False, SimpleNamespace(playing=threading.Event())
    v._result_pending = v._result_speech = v._turn_open = False
    v._asked = v._voiced_action = None
    return v, sent


def test_live_direct_call_reads_for_the_users_sentence_fenced(world, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    v, sent = _live_voice()
    fc = SimpleNamespace(id="c1", name="in_front", args={"what": "auto"})
    asyncio.run(v._run_tool(v._live, fc, "summarize what I copied"))
    result = sent[0].response["result"]
    assert result.startswith('<untrusted source="in_front">') and "Meeting moved" in result
    assert v._taint == ["in_front"]


def test_live_direct_call_on_the_models_own_idea_is_refused(world, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    v, sent = _live_voice()
    fc = SimpleNamespace(id="c1", name="in_front", args={"what": "clipboard"})
    asyncio.run(v._run_tool(v._live, fc, "what's the weather tomorrow"))
    result = sent[0].response["result"]
    assert "Meeting moved" not in result and not result.startswith("<untrusted") and v._taint == []
    assert world.pb.read == []


@pytest.mark.parametrize(
    "said, handed",
    [
        ("summarize what I copied", ("summarize what I copied", "user")),  # the user's own words, as theirs
        ("tell me something nice", ("Summarize the clipboard", "live")),  # the model's goal, as before
    ],
)
def test_live_agent_task_hands_over_the_users_own_words_for_this(world, monkeypatch, said, handed):
    from neo2.core import events
    from neo2.core.agent.session import Reply

    monkeypatch.setattr(events, "_bus", events.EventBus())
    seen: list = []

    class S:
        async def handle(self, text, **kw):
            seen.append((text, kw.get("origin")))
            return Reply("Done.", "agent")

    v, _ = _live_voice(S())
    fc = SimpleNamespace(id="c1", name="agent_task", args={"goal": "Summarize the clipboard"})
    asyncio.run(v._run_tool(v._live, fc, said))
    assert seen == [handed]


def test_live_agent_task_switched_off_keeps_the_models_goal(world, monkeypatch):
    from neo2.core import config, events
    from neo2.core.agent.session import Reply

    monkeypatch.setattr(events, "_bus", events.EventBus())
    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    seen: list = []

    class S:
        async def handle(self, text, **kw):
            seen.append((text, kw.get("origin")))
            return Reply("Done.", "agent")

    v, _ = _live_voice(S())
    fc = SimpleNamespace(id="c1", name="agent_task", args={"goal": "Summarize the clipboard"})
    asyncio.run(v._run_tool(v._live, fc, ASKED))
    assert seen == [("Summarize the clipboard", "live")]
