"""Hybrid memory, playbooks, session summaries, the verification turn, and the browser helpers."""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from neo.agent.loop import run_agent
from neo.agent.registry import RegisteredTool, Registry
from neo.events import EventBus
from neo.memory import embed as emb
from neo.memory.store import Store
from neo.providers.base import ToolCall, Turn


def _fake_embed(texts):
    """Deterministic bag-of-words vectors: shared words → high cosine. Good enough to test fusion."""
    out = np.zeros((len(texts), emb.DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % emb.DIM] += 1.0
        n = np.linalg.norm(out[i]) or 1.0
        out[i] /= n
    return out


@pytest.fixture
def st(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    return Store(tmp_path / "m.sqlite3")


def test_semantic_recall_without_keyword_overlap_is_possible(st):
    st.remember("favourite editor is Zed", "profile", "editor")
    st.remember("the dentist appointment is on Tuesday", "note")
    # exact keyword hit still first
    assert st.recall("editor")[0].key == "editor"
    # shared tokens 'dentist appointment' rank the note via the vector path even with FTS present
    hits = st.recall("dentist appointment")
    assert hits and "dentist" in hits[0].text


def test_recall_returns_nothing_for_junk(st):
    st.remember("x y z", "note")
    assert st.recall("???") == []


def test_playbook_roundtrip_and_context(st):
    steps = [
        {"tool": "open_app", "args": {"target": "Mail"}},
        {"tool": "mail_send", "args": {"to": "sam@x.com"}},
    ]
    pid = st.save_playbook("reply to sam's email saying I'll be late", steps, "Sent.")
    again = st.save_playbook("reply to sam's email saying I'll be late", steps, "Sent again.")
    assert again == pid  # near-duplicate goals update, not duplicate
    ctx = st.playbook_context("reply to sam's email that I'm late")
    assert "PLAYBOOKS" in ctx and "open_app" in ctx and "mail_send" in ctx
    assert st.playbook_context("what's the weather") == ""


def test_session_summary_in_prompt_context(st):
    st.add_session_summary("Set up the Node project and emailed Priya the notes.", started=1_700_000_000.0)
    assert "Recent sessions" in st.prompt_context() and "Priya" in st.prompt_context()


class Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self._turns = list(turns)
        self.seen = []

    async def complete(self, messages, **kw):
        self.seen.append([m.text for m in messages if m.role == "user"])
        return self._turns.pop(0)

    async def stream(self, messages, **kw):
        yield "x"


def _reg():
    reg = Registry()
    reg.register(
        RegisteredTool(
            "look", "l", {"type": "object", "properties": {}}, lambda a, c: "seen", parallel_safe=True
        )
    )
    reg.register(RegisteredTool("act", "a", {"type": "object", "properties": {}}, lambda a, c: "did it"))
    return reg


def test_verification_turn_after_side_effect():
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("1", "act", {})]),
            Turn(text="Done!"),  # claims done without looking → verify note injected
            Turn(tool_calls=[ToolCall("2", "look", {})]),
            Turn(text="Confirmed done."),
        ]
    )
    res = asyncio.run(run_agent("do the thing", provider=brain, system="s", reg=_reg(), ev=EventBus()))
    assert res.stopped == "done" and res.text == "Confirmed done."
    assert any("[verify]" in u for u in brain.seen[2])  # the note reached the model
    assert [s.tool for s in res.steps] == ["act", "look"]


def test_no_verification_when_last_step_was_observation_or_readonly_only():
    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("1", "act", {})]),
            Turn(tool_calls=[ToolCall("2", "look", {})]),
            Turn(text="ok"),
        ]
    )
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=_reg(), ev=EventBus()))
    assert res.stopped == "done" and len(brain.seen) == 3  # no extra verify round
    brain = Brain([Turn(tool_calls=[ToolCall("1", "look", {})]), Turn(text="ok")])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=_reg(), ev=EventBus()))
    assert len(brain.seen) == 2


def test_verification_happens_at_most_once():
    brain = Brain([Turn(tool_calls=[ToolCall("1", "act", {})]), Turn(text="done"), Turn(text="still done")])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=_reg(), ev=EventBus()))
    assert res.text == "still done" and len(brain.seen) == 3


def test_verify_can_be_disabled():
    brain = Brain([Turn(tool_calls=[ToolCall("1", "act", {})]), Turn(text="done")])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=_reg(), ev=EventBus(), verify=False))
    assert res.text == "done" and len(brain.seen) == 2


def test_playbook_saved_after_successful_multistep_run(tmp_path, monkeypatch):
    from neo import config
    from neo.agent import session as sess_mod
    from neo.agent.session import Session
    from neo.memory import store as store_mod
    from neo.reflex.schema import Decision
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store_mod, "_store", None)
    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    load_all()

    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.9, False, 0.0, False, 0.0, "fake")

    brain = Brain(
        [
            Turn(tool_calls=[ToolCall("1", "open_app", {"target": "Notes"})]),
            Turn(tool_calls=[ToolCall("2", "notes_create", {"title": "t", "body": "b"})]),
            Turn(text="Created the note."),
            Turn(tool_calls=[ToolCall("3", "apps_running", {})]),
            Turn(text="Created the note."),
        ]
    )
    import neo.tools.computer as comp

    async def ok(a, c):
        return "ok"

    from neo.agent.registry import registry

    for name in ("open_app", "notes_create", "apps_running"):
        registry().get(name).handler = ok  # type: ignore[union-attr]
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)
    r = asyncio.run(Session(reflex=R()).handle("make a note called t saying b"))
    assert r.route == "agent"
    pbs = store_mod.store().similar_playbooks("make a note called t saying b", min_sim=0.3)
    assert pbs and [s["tool"] for s in pbs[0].steps][:2] == ["open_app", "notes_create"]
    _ = comp


def test_browser_url_and_selector_heuristics():
    from neo.tools.browser import _norm_url

    assert _norm_url("github.com") == "https://github.com"
    assert _norm_url("http://x.y") == "http://x.y"
    assert _norm_url(" https://a.b ") == "https://a.b"
