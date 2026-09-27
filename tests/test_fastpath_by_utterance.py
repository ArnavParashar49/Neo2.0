"""The by-utterance lane (neo.agent.fastpath.by_utterance): a tool registered with by_utterance=True
gets the user's whole request, with no model call to fill in a label — only for the user's own
words, never a model's or Gemini Live's. Fake registry, fake decisions; nothing runs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from neo.agent import fastpath
from neo.agent.registry import RegisteredTool, Registry


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
