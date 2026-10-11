"""The reliability report (neo2.core.tools.quality): an explicit ask reads back the local counters; a
greeting ("how are you doing?") is conversation and never gets statistics. No model, no network."""

from __future__ import annotations

import asyncio

import pytest

from neo2.core.agent import fastpath


@pytest.fixture(autouse=True)
def _tools():
    from neo2.core.tools import load_all

    load_all()


ASKS = [
    "how's your reliability?",
    "how is your reliability",
    "how's neo doing?",
    "show me your quality report",
    "show neo's recent reliability",
]

GREETINGS = [
    "how are you doing?",
    "How are you doing, Neo?",
    "how have you been doing?",
    "how is neo",
    "how's neo?",
    "how are you",
    "how have you been",
]


@pytest.mark.parametrize("text", ASKS)
def test_an_explicit_ask_is_the_report(text):
    assert fastpath.plan(text) == [("quality_report", {"days": 7})]


@pytest.mark.parametrize("text", GREETINGS)
def test_a_greeting_is_not(text):
    steps = fastpath.plan(text) or []
    assert all(t != "quality_report" for t, _ in steps)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


def test_the_session_chats_back_to_a_greeting(monkeypatch):
    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    chats: list[str] = []

    async def chat(self, text, d, job):
        chats.append(text)
        await events.bus().end_job(job)
        return sess_mod.Reply("I'm good, thanks!", "chat", d, job=job)

    monkeypatch.setattr(sess_mod.Session, "_chat", chat)
    s = sess_mod.Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("Neo, how are you doing?", from_user=True))
    assert r.route == "chat" and chats == ["how are you doing?"]
    r = asyncio.run(s.handle("how's your reliability?", from_user=True))
    assert r.route == "quick" and r.tools == ("quality_report",) and chats == ["how are you doing?"]


# ---- the report's text: local counters only, content-free ----------------------------------------
def _counted(**counts):
    from neo2.core.reflex import learn

    for name, n in counts.items():
        learn.count(name, n)
    return learn.quality_report(7)


_BASE = dict(
    requests=10, decisions=8, unsure=2, corrections=1, barge_ins=1, verified=6, recovered=1, unresolved=2
)
_BEFORE = (
    "NEO reliability, last 7 days: 10 settled requests; 6 outcomes NEO could verify; 1 user corrections (10.0%); "
    "1 replies stopped within 3 seconds; 1 corrected requests later completed; 2 outcomes remained unclear. "
    "Laya was unsure on 25.0% of 8 routing decisions. These are interaction signals, not a measure of factual accuracy."
)


def test_with_none_of_the_new_counters_the_report_is_word_for_word_what_it_was():
    assert _counted(**_BASE) == _BEFORE


def test_no_data_is_still_no_data():
    assert _counted(mic_stall=3) == "I don't have reliability data for the last 7 days yet."


def test_the_wake_line_reads_the_gate_and_its_false_drops():
    text = _counted(
        **_BASE,
        wake_voice=40,
        wake_gate_pass=28,
        wake_gate_drop=12,
        wake_gate_drop_words=1,
        wake_gate_pass_nowords=3,
    )
    assert (
        "Wake word: 40 spoken wakes; the wake check flagged 12 to drop (1 had words after all) and passed 28 (3 with no words after)."
        in text
    )
    assert text.endswith("These are interaction signals, not a measure of factual accuracy.")


def test_without_the_gates_counters_the_wake_check_is_n_a():
    text = _counted(**_BASE, wake_voice=7)
    assert "Wake word: 7 spoken wakes; the wake check n/a." in text


def test_the_microphone_line():
    text = _counted(**_BASE, mic_stall=4, mic_reopen_silent=6, mic_recovered=2)
    assert (
        "Microphone: 4 stalls; 6 reopens that brought no sound; 2 recoveries; 0 notices that NEO couldn't hear."
        in text
    )


def test_the_computer_control_line_names_the_commonest_kind_of_failure():
    text = _counted(
        **_BASE,
        tool_fail=9,
        tool_fail_not_found=5,
        tool_fail_timeout=3,
        tool_fail_other=1,
        tool_skip_repeat=2,
    )
    assert "Computer control: 9 tool failures, most often not found (5); 2 repeated steps skipped." in text


def test_a_tie_between_kinds_is_settled_the_same_way_every_time():
    text = _counted(**_BASE, tool_fail=4, tool_fail_timeout=2, tool_fail_offscreen=2)
    assert "most often off screen (2); 0 repeated steps skipped." in text


def test_agent_runs_by_how_they_ended():
    text = _counted(**_BASE, agent_end_done=15, agent_end_max_steps=3, agent_end_needs_user=2)
    assert "Agent runs: 20 (15 finished, 3 ran out of steps, 2 needed you)." in text


def test_only_counter_names_never_anything_else_reach_the_report():
    """A key that isn't a plain lower-case name (a hand-edited state file) is left out, not read back."""
    text = _counted(**_BASE, **{"agent_end_Priya's mail": 4, "tool_fail_<script>": 2, "agent_end_done": 1})
    assert "Priya" not in text and "<script>" not in text
    assert "Agent runs: 1 (1 finished)." in text


def test_the_lines_come_in_a_fixed_order_before_the_closing_caveat():
    text = _counted(**_BASE, agent_end_done=1, tool_fail=1, mic_stall=1, wake_voice=1)
    order = [
        text.index(s)
        for s in (
            "Wake word:",
            "Microphone:",
            "Computer control:",
            "Agent runs:",
            "These are interaction signals",
        )
    ]
    assert order == sorted(order)
