"""Tool selection, more_tools, the light lane, and history stubbing."""

from __future__ import annotations

import asyncio

from neo.agent.loop import run_agent
from neo.agent.registry import RegisteredTool, Registry
from neo.agent.session import Session, _stub
from neo.agent.toolselect import MORE_TOOLS, ToolSelector
from neo.events import EventBus
from neo.providers.base import Message, ToolCall, ToolResult, Turn


def _reg() -> Registry:
    r = Registry()

    def t(name, desc, cat, safe=True):
        r.register(
            RegisteredTool(
                name,
                desc,
                {"type": "object", "properties": {}},
                lambda a, c: name,
                parallel_safe=safe,
                category=cat,
            )
        )

    t("memory", "Long-term memory: remember and recall facts.", "memory")
    t("browser_open", "Open a URL in NEO's browser.", "browser")
    t("browser_read", "Read the current browser page.", "browser")
    t("browser_click", "Click an element on the current page.", "browser")
    t("web_search", "Search the web. Returns titles, URLs and snippets.", "web")
    t("mail_unread", "List unread inbox messages.", "apps")
    t("mail_send", "Compose an email in Mail.", "apps", False)
    t("volume", "Set system volume or mute.", "system")
    t("ax_tree", "Read the frontmost app's UI as a text tree.", "computer")
    t("click", "Click at screen coordinates.", "computer", False)
    t("screenshot", "Take a screenshot.", "computer")
    t("read_file", "Read a text file or list a directory.", "files")
    return r


def test_keyword_selection_picks_relevant_tools_and_families():
    sel = ToolSelector(_reg())  # tests disable the embedder → keyword path
    s = sel.select("search the web for the tallest building and open the page in the browser")
    assert "web_search" in s.names and "browser_open" in s.names
    assert {"browser_read", "browser_click"} <= set(s.names)  # family expansion
    assert "memory" in s.names  # always
    assert "volume" not in s.names and "mail_send" not in s.names
    specs = s.specs(_reg())
    assert specs[-1].name == "more_tools"


def test_screen_rule_and_playbook_tools():
    sel = ToolSelector(_reg())
    s = sel.select("what is that", needs_screen=True, playbook_tools=["mail_unread"])
    assert {"ax_tree", "click", "screenshot"} <= set(s.names)
    assert "mail_unread" in s.names and s.reason["mail_unread"] == "playbook"


def test_search_excludes_enabled():
    sel = ToolSelector(_reg())
    found = sel.search("send an email", exclude={"mail_unread"})
    assert found and found[0].name == "mail_send"


class Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self._turns = list(turns)
        self.tools_seen = []

    async def complete(self, messages, *, tools=None, **kw):
        self.tools_seen.append([t.name for t in tools or []])
        return self._turns.pop(0)

    async def stream(self, messages, **kw):
        yield "x"


def test_more_tools_grows_the_run_tool_list():
    reg = _reg()
    sel = ToolSelector(reg)
    import neo.agent.toolselect as ts

    ts._selector = sel
    initial = sel.select("what's the volume").specs(reg)
    assert "mail_send" not in [t.name for t in initial]
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("1", "more_tools", {"query": "send an email"})]),
            Turn(tool_calls=[ToolCall("2", "mail_send", {})]),
            Turn(text="sent"),
        ]
    )
    res = asyncio.run(
        run_agent(
            "email sam", provider=brain, system="s", reg=reg, ev=EventBus(), tools=initial, verify=False
        )
    )
    assert res.stopped == "done"
    assert "mail_send" not in brain.tools_seen[0] and "mail_send" in brain.tools_seen[1]
    assert (
        res.messages[2].tool_results[0].name == "more_tools"
        and "mail_send" in res.messages[2].tool_results[0].content
    )
    ts._selector = None


def test_history_stubs_old_tool_results():
    s = Session(reflex=None)  # type: ignore[arg-type]
    big = "x" * 5000
    s.history = [
        Message.user("first"),
        Message.assistant("", [ToolCall("1", "read_file", {})]),
        Message.tool([ToolResult("1", "read_file", big)]),
        Message.assistant("done"),
        Message.user("second"),
        Message.assistant("", [ToolCall("2", "read_file", {})]),
        Message.tool([ToolResult("2", "read_file", big)]),
        Message.assistant("done again"),
        Message.user("third"),
    ]
    h = s._trimmed()
    # oldest turn's result is stubbed; the two most recent user turns keep theirs verbatim
    assert len(h[2].tool_results[0].content) < 300 and "trimmed" in h[2].tool_results[0].content
    assert len(h[6].tool_results[0].content) == 5000
    assert _stub("short") == "short"


def test_quick_action_uses_light_lane(monkeypatch):
    import neo.agent.session as sess_mod
    from neo.reflex.schema import Decision
    from neo.tools import load_all

    load_all()
    purposes = []

    def fake_brain(purpose="agent"):
        purposes.append(purpose)
        return Brain([Turn(text="ok")])

    monkeypatch.setattr(sess_mod, "brain", fake_brain)

    class R:
        async def decide(self, t):
            return Decision("quick_action", 0.9, False, 0.0, False, 0.0, "fake")

    r = asyncio.run(Session(reflex=R()).handle("open netflix please now"))  # no regex fast path → light lane
    assert r.route == "agent" and purposes == ["light"]


def test_more_tools_spec_shape():
    assert MORE_TOOLS.parameters["required"] == ["query"]
