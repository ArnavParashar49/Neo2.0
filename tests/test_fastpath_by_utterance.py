"""The by-utterance lane (neo2.core.agent.fastpath.by_utterance): a tool registered with by_utterance=True
gets the user's whole request, with no model call to fill in a label — only for the user's own
words, never a model's or Gemini Live's. Fake registry, fake decisions; nothing runs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neo2.core.agent import fastpath
from neo2.core.agent.registry import RegisteredTool, Registry


async def _noop(a, c):
    return "ok"


def _tool(name: str, by, props=("label",)) -> RegisteredTool:
    t = RegisteredTool(name, name, {"type": "object", "properties": {p: {"type": "string"} for p in props}}, _noop, category="computer")
    t.by_utterance = by  # the Step-0 registry field; a plain attribute until then
    return t


@pytest.fixture
def reg(monkeypatch):
    r = Registry()
    r.register(_tool("click_text", True))
    r.register(_tool("menu_command", "menu", props=("item",)))
    r.register(_tool("hotkey", False, props=("keys",)))
    monkeypatch.setattr(fastpath, "registry", lambda: r)
    return r


@dataclass
class D:
    intent: str = "agent_task"
    tool: str = ""
    tool_p: float = 0.0


@pytest.mark.parametrize(
    "text, label",
    [
        ("new game pe click karo", "new game"),
        ("start wala button dabao", "start"),
        ("नया गेम दबाओ", "नया गेम"),
        ("press start", "start"),
        ("hit play", "play"),
        ("click the reply all button", "reply all"),
    ],
)
def test_a_pointing_request_from_the_user_goes_to_the_click_tool(reg, text, label):
    assert fastpath.by_utterance(text, origin="user", decision=D()) == ("click_text", {"utterance": text, "label": label})


@pytest.mark.parametrize("origin", ["live", "agent", "approval", ""])
def test_never_for_words_that_arent_the_users(reg, origin):
    assert fastpath.by_utterance("new game pe click karo", origin=origin, decision=D("quick_action", "click_text", 0.99)) is None


@pytest.mark.parametrize(
    "text",
    ["press enter", "hit cmd+s", "what does the share button do?", "click reply and send", "save karo", "click it", ""],
)
def test_keys_questions_two_things_and_bare_verbs_stay_out(reg, text):
    assert fastpath.by_utterance(text, origin="user", decision=D()) is None


def test_the_reflex_calling_it_conversation_keeps_it_out(reg):
    assert fastpath.by_utterance("click start", origin="user", decision=D("chat")) is None


def test_the_reflexs_pick_of_a_by_utterance_tool_takes_even_unframed_words(reg, monkeypatch):
    got = fastpath.by_utterance("start a new game", origin="user", decision=D("quick_action", "menu_command", 0.95))
    assert got == ("menu_command", {"utterance": "start a new game", "label": "start a new game"})
    assert fastpath.by_utterance("start a new game", origin="user", decision=D("quick_action", "menu_command", 0.5)) is None
    assert fastpath.by_utterance("start a new game", origin="user", decision=D("quick_action", "hotkey", 0.99)) is None
    assert fastpath.by_utterance("reply and send it", origin="user", decision=D("quick_action", "click_text", 0.99)) is None


def test_no_by_utterance_tools_no_lane(monkeypatch):
    r = Registry()
    r.register(_tool("click_text", False))
    monkeypatch.setattr(fastpath, "registry", lambda: r)
    assert fastpath.by_utterance("click start", origin="user", decision=D()) is None


def test_the_regex_fast_paths_are_unchanged(reg):
    # the lane is beside plan(), never inside it: plan() knows nothing of by_utterance
    assert fastpath.plan("new game pe click karo") is None


# ---- review fixes: the lane decides on the raw words, and refuses what pointing() refuses ---------------
@pytest.mark.parametrize("text", ["click send?", "save pe click karna?", "start button dabana?", "press delete account?"])
def test_a_question_mark_is_read_before_anything_is_stripped(reg, text):
    assert fastpath.by_utterance(text, origin="user", decision=D()) is None
    assert fastpath.by_utterance(text, origin="user", decision=D("quick_action", "click_text", 0.95)) is None


@pytest.mark.parametrize(
    "text",
    [
        "what does the share button do?",
        "what does the share button do",
        "which button deletes my account?",
        "kya ye button safe hai?",
        "kya ye button safe hai",
        "ये बटन क्या करता है",
        "press enter",
        "hit cmd+s",
        "click it",
        "is button pe click karo",
    ],
)
def test_the_reflexs_pick_never_turns_a_question_a_key_or_a_vague_word_into_a_press(reg, text):
    assert fastpath.by_utterance(text, origin="user", decision=D("quick_action", "click_text", 0.95)) is None


def test_the_reflexs_pick_still_takes_a_plain_request(reg):
    got = fastpath.by_utterance("start a new game", origin="user", decision=D("quick_action", "click_text", 0.95))
    assert got == ("click_text", {"utterance": "start a new game", "label": "start a new game"})
    got = fastpath.by_utterance("click new tab.", origin="user", decision=D())
    assert got == ("click_text", {"utterance": "click new tab", "label": "new"})  # the handler resolves the utterance


def test_resolver_off_means_nothing_is_pressed_from_words(reg, monkeypatch):
    from neo2.core.tools.computer import resolve

    monkeypatch.setattr(resolve, "mode", lambda: "off")
    assert fastpath.by_utterance("new game pe click karo", origin="user", decision=D()) is None
    assert fastpath.by_utterance("start a new game", origin="user", decision=D("quick_action", "click_text", 0.99)) is None


def test_an_embedding_hosts_requests_keep_the_regex_paths_only(reg, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    assert fastpath.by_utterance("new game pe click karo", origin="user", decision=D()) is None


# ---- the reflex's confident pick of another tool: its one-tool lane, not a press -------------------------
@pytest.mark.parametrize(
    "text, tool",
    [
        ("press cmd space", "hotkey"),
        ("press pause", "play_music"),
        ("press next track", "play_music"),
        ("select dark mode", "applescript"),
    ],
)
def test_words_the_reflex_gives_another_tool_are_left_to_that_tool(reg, text, tool):
    """"press pause" reads as a press, but the reflex is sure it's play_music's: the session's one-tool
    lane (one small model call that sees only that tool) takes it — never an AX walk, a menu-bar scan
    and a read of the screen for a control named "pause", then a full agent run."""
    reg.register(_tool("play_music", False, props=("query",)))
    reg.register(_tool("applescript", False, props=("script",)))
    assert fastpath.by_utterance(text, origin="user", decision=D("quick_action", tool, 0.97)) is None
    assert fastpath.by_utterance(text, origin="user", decision=D("agent_task", tool, 0.97)) is None


def test_an_unsure_or_unknown_pick_leaves_the_pointing_words_to_the_click_tool(reg):
    reg.register(_tool("play_music", False, props=("query",)))
    hidden = _tool("secret", False, props=("x",))
    hidden.hidden = True
    reg.register(hidden)
    want = ("click_text", {"utterance": "press start", "label": "start"})
    assert fastpath.by_utterance("press start", origin="user", decision=D("quick_action", "play_music", 0.5)) == want
    assert fastpath.by_utterance("press start", origin="user", decision=D("quick_action", "no_such_tool", 0.99)) == want
    assert fastpath.by_utterance("press start", origin="user", decision=D("quick_action", "secret", 0.99)) == want


@pytest.mark.parametrize(
    "keys", ["cmd space", "command space", "option space", "ctrl shift tab", "cmd + space", "command space bar", "control option delete"]
)
def test_a_chord_said_with_spaces_is_a_key_not_a_control(reg, keys):
    from neo2.core.tools.computer import phrases

    assert phrases.is_key_name(keys)
    assert fastpath.by_utterance(f"press {keys}", origin="user", decision=D()) is None


@pytest.mark.parametrize("label", ["space invaders", "command center", "control panel", "option menu", "shift report"])
def test_a_label_that_starts_like_a_key_is_still_a_label(reg, label):
    from neo2.core.tools.computer import phrases

    assert not phrases.is_key_name(label)
    assert fastpath.by_utterance(f"click {label}", origin="user", decision=D()) == (
        "click_text", {"utterance": f"click {label}", "label": label},
    )  # fmt: skip
