"""page_read, first-class: the page the user is looking at in their own browser. Its fast paths
(English, Hinglish, Devanagari), the session answering the user's own words from the page, and
Live's direct tools and prompt lines (browse only while a tool of that name is there). The page
and the brains are fakes: no app is read, no screen is captured, no model is called."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent import fastpath
from neo.agent.registry import RegisteredTool, registry
from neo.events import EventBus
from neo.reflex.schema import Decision


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()


@pytest.mark.parametrize(
    "text",
    [
        "summarize this page",
        "Summarise the article please",
        "can you summarize this page for me?",
        "and now sum up this tab",
        "tl;dr this post",
        "give me a quick summary of this page",
        "what's the gist of this article",
        "summarize the comments",
        "what's on this page?",
        "what is on the page",
        "what's this page about?",
        "what is this article all about",
        "what's the article about?",
        "what does this page say",
        "tell me about this page",
        "read me this page",
        "please read this article out loud",
        "read me the comments",
        "read the comments on this video",
        "read the top comments",
        "read out the reviews on this page",
        "what are people saying in the comments?",
        "what are the comments saying",
        "what do the reviews on this page say",
        "what's the price here?",
        "what is the price on this page",
        "is page pe kya hai",
        "is page mein kya likha hai",
        "ye page summarize karo",
        "is page ka summary do",
        "ye article padh ke sunao",
        "comments padh ke sunao",
        "comments mein kya likha hai",
        "log comments mein kya keh rahe hain",
        "yahan price kya hai",
        "इस पेज पर क्या है",
        "ये पेज समराइज़ करो",
    ],
)
def test_asking_about_the_page_goes_to_page_read(text):
    assert fastpath.plan(text) == [("page_read", {})]


@pytest.mark.parametrize(
    "text, tool",
    [
        ("find on this page", "hotkey"),
        ("close this tab", "hotkey"),
        ("reload this page", "hotkey"),
        ("bookmark this page", "hotkey"),
        ("read my email", "mail_unread"),
        ("what's on my calendar", "calendar_today"),
        # a referent only the conversation has, or no page at all: a model decides
        ("what is the story", None),
        ("what's the article", None),
        ("tell me about the article", None),
        ("what do the reviews say", None),
        ("read me a story", None),
        ("read me the thread", None),
        ("summarize this", None),
        ("summarize my emails", None),
        ("summarize the document", None),
        ("what's the price of bitcoin", None),
        ("what's on my screen", None),
        # "open X and summarize it" would read the page before it loads
        ("open notes and summarize this page", None),
        ("summarize this page and close it", None),
    ],
)
def test_what_is_not_a_question_about_the_page(text, tool):
    steps = fastpath.plan(text)
    assert (steps[0][0] if steps else None) == tool


class _Reflex:
    async def decide(self, text):
        return Decision("agent_task", 0.9, False, 0.0, True, 0.0, "fake")


def test_the_users_own_words_are_answered_from_the_page(monkeypatch):
    """Typed or spoken "summarize this page": the fast path reads the page and NEO answers from it,
    aloud; no model call picks the tool."""
    from neo.agent.session import Session
    from neo.decide import run
    from neo.decide.kinds import site
    from neo.tools import computer

    asked: list[str] = []

    async def page_read(limit=site.PAGE_CHARS, ask=""):
        asked.append(ask)
        return run.Result("It's a flood report: the old bridge closed at dawn.", True)

    monkeypatch.setattr(registry().get("page_read"), "handler", computer.page_read)
    monkeypatch.setattr(site, "page_read", page_read)
    s = Session(reflex=_Reflex())
    r = asyncio.run(s.handle("summarize this page", from_user=True))
    assert r.route == "quick" and r.tools == ("page_read",) and not r.silent
    assert r.text == "It's a flood report: the old bridge closed at dawn." and asked == ["summarize this page"]


def _declared() -> set[str]:
    from neo.voice import live

    return {d.name for d in live._declarations()}


def test_live_calls_page_read_itself_and_is_told_when(monkeypatch):
    from neo.voice import live

    assert "page_read" in live._DIRECT_TOOLS and "page_read" in _declared()
    assert "safari_read" not in _declared()
    extra = live._live_extra()
    assert "call page_read and answer from it" in extra and "read me the comments" in extra


def test_browse_is_offered_to_live_only_while_it_is_there(monkeypatch):
    from neo.voice import live

    assert "browse" in live._DIRECT_TOOLS
    if registry().get("browse") is None:  # until NEO's own browser registers it
        assert "browse" not in _declared() and "call browse" not in live._live_extra()

    async def browse(a, c):
        return "done"

    t = RegisteredTool("browse", "Do something on a website in NEO's own browser.", {"type": "object", "properties": {}}, browse)
    monkeypatch.setitem(registry()._tools, "browse", t)
    assert "browse" in _declared()
    assert "call browse with a self-contained goal" in live._live_extra()
    t.hidden = True  # a tool the models don't see is neither declared nor mentioned
    assert "browse" not in _declared() and "call browse" not in live._live_extra()


def test_a_window_is_captured_by_its_id_and_scaled_down(monkeypatch):
    """screencapture's own window mode (no shadow, no sound), never the whole screen; faked here."""
    import io
    from types import SimpleNamespace

    from PIL import Image

    from neo.tools.computer import screen

    ran: list[list[str]] = []

    def run(cmd, check, timeout):
        ran.append(cmd)
        Image.new("RGB", (3200, 1000), "white").save(cmd[-1], format="PNG")

    asked: list[int] = []
    monkeypatch.setattr(screen, "Q", SimpleNamespace(CGRequestScreenCaptureAccess=lambda: asked.append(1)))
    monkeypatch.setattr(screen.subprocess, "run", run)
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    img = screen.capture_window(77)
    assert ran[0][:-1] == ["screencapture", "-x", "-o", "-t", "png", "-l", "77"]
    assert Image.open(io.BytesIO(img.data)).size == (1600, 500) and img.mime == "image/png"
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    with pytest.raises(PermissionError):
        screen.capture_window(77)
    assert asked == [1] and len(ran) == 1
