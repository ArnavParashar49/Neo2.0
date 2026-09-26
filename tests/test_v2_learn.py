"""Laya learns from use: labels from what actually happened, idle retraining, safe swaps."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from neo.reflex import learn


def _rows():
    return learn._read(learn._path(learn.EXTRA))  # as training sees them: latest verdict per sentence


def test_record_keeps_real_requests_and_skips_noise():
    assert learn.record("open notes and type hello", "quick_action", "open_app", "live")
    assert not learn.record("neo", "chat")  # the wake word alone
    assert not learn.record("okay", "chat")
    assert not learn.record("stop", "chat")
    assert not learn.record("what", "chat")  # one word
    assert not learn.record("do the thing", "stop")  # not a trainable intent
    assert [r["text"] for r in _rows()] == ["open notes and type hello"]


def test_latest_verdict_wins_for_the_same_sentence():
    learn.record("Check my email", "agent_task", "", "live")
    learn.record("check my  email", "quick_action", "mail_unread", "live")
    rows = _rows()
    assert len(rows) == 1 and rows[0]["intent"] == "quick_action" and rows[0]["tool"] == "mail_unread"


def test_due_needs_enough_new_examples_and_time(monkeypatch):
    assert not learn.due()
    for i in range(learn._MIN_NEW):
        learn.record(f"example request number {i}", "chat")
    assert learn.pending() == learn._MIN_NEW
    assert learn.due()  # never trained → the time gap is satisfied
    learn._path(learn.STATE).write_text(json.dumps({"trained_at": time.time(), "trained_rows": 0}))
    assert not learn.due()  # trained a moment ago, and not a burst yet
    for i in range(learn._BURST):
        learn.record(f"burst request number {i}", "chat")
    assert learn.due()  # a burst retrains regardless of the gap


def test_retrain_swaps_the_head_only_when_saved(monkeypatch):
    import neo.reflex as reflex_mod
    from neo.reflex import finetune

    swapped = []

    class Laya:
        _agent = object()

        def reload_head(self):
            swapped.append(True)

    monkeypatch.setattr(reflex_mod, "get_laya", lambda: Laya())
    reports = iter([{"saved": True, "intent": 0.91, "previous": 0.90}, {"saved": False, "intent": 0.80, "previous": 0.90}])
    monkeypatch.setattr(finetune, "train", lambda args, agent=None, lock=None: next(reports))
    learn.record("what's on my calendar today", "quick_action", "calendar_today")
    assert learn.retrain()["saved"] and swapped == [True]
    assert json.loads(learn._path(learn.STATE).read_text())["trained_rows"] == 1
    assert not learn.retrain()["saved"] and swapped == [True]  # a worse head is never swapped in


def test_learn_loop_trains_only_after_neo_has_been_idle(monkeypatch):
    runs = []
    monkeypatch.setattr(learn, "due", lambda now=None: True)
    monkeypatch.setattr(learn, "retrain", lambda: runs.append(1))
    monkeypatch.setattr(learn, "_IDLE_S", 0.0)
    ticks = {"n": 0}

    async def fast_sleep(_):
        ticks["n"] += 1
        if ticks["n"] > 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(learn.asyncio, "sleep", fast_sleep)
    idle = iter([False, False, True, False])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(learn.learn_loop(lambda: next(idle)))
    assert runs == [1]  # never while busy


def _live():
    import neo.voice.live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._label, v._call_labels = None, {}
    return v


@pytest.mark.parametrize(
    "tools, spoke, expected",
    [
        ([], True, ("chat", "")),
        (["weather"], True, ("quick_action", "weather")),
        (["hotkey", "open_app"], False, ("quick_action", "")),
        (["agent_task"], True, ("agent_task", "")),
        ([], False, None),  # nothing came of it: room noise, not recorded
    ],
)
def test_live_labels_each_sentence_by_what_the_model_did(tools, spoke, expected):
    v = _live()
    v._label = {"text": "what's it like in Paris tomorrow", "tools": list(tools), "spoke": spoke, "open": False}
    v._file_label()
    rows = _rows()
    if expected is None:
        assert rows == []
    else:
        assert (rows[0]["intent"], rows[0]["tool"]) == expected and rows[0]["source"] == "live"


def test_a_partial_transcript_at_tool_time_and_the_final_one_make_one_label():
    """The tool call arrives while the transcript is still partial; the final wording must
    replace it on the same label — not open a second, tool-less one that files as chat."""
    import asyncio

    import neo.voice.live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._label, v._call_labels, v._tool_tasks, v._utterance = None, {}, {}, ""
    v._dialog, v._lookups, v._seen_calls = [], {}, {}
    v._turn_open = v._speaking = False
    v._last_activity = v._last_user_speech = v._last_turn_end = 0.0

    class Tr:
        def __init__(self, t):
            self.text = t

    class SC:
        def __init__(self, text=None, done=False):
            self.interrupted = False
            self.input_transcription = Tr(text) if text else None
            self.interim_input_transcription = None
            self.output_transcription = None
            self.model_turn = None
            self.turn_complete = done

    class Msg:
        def __init__(self, sc=None, calls=None):
            self.server_content, self.go_away, self.tool_call_cancellation = sc, None, None
            self.tool_call = type("TC", (), {"function_calls": calls})() if calls else None

    class FC:
        id, name, args = "c1", "weather", {"place": "Paris"}

    async def fake_run_tool(live, fc, asked=""):
        return None

    v._run_tool = fake_run_tool
    v._refresh_state = lambda: asyncio.sleep(0)
    v._after_playback = lambda: asyncio.sleep(0)
    user_buf, model_buf = [], []

    async def run():
        await v._on_message(Msg(SC("what's the weather")), user_buf, model_buf, None)
        await v._on_message(Msg(calls=[FC()]), user_buf, model_buf, None)
        await v._on_message(Msg(SC(" in Paris")), user_buf, model_buf, None)
        await v._on_message(Msg(SC(done=True)), user_buf, model_buf, None)

    asyncio.run(run())
    assert v._label["text"] == "what's the weather in Paris" and v._label["tools"] == ["weather"]
    v._file_label()
    assert [(r["text"], r["intent"], r["tool"]) for r in _rows()] == [("what's the weather in Paris", "quick_action", "weather")]


def test_private_content_is_never_stored(monkeypatch):
    assert not learn.record("type my password is hunter2", "quick_action", "type_text")
    assert not learn.record("remember that my locker code is 4471", "quick_action", "memory")
    assert not learn.record("my otp is 123456", "chat")
    assert not learn.record("call me on 050 123 4567", "chat")
    assert learn.record("what's the weather in Dubai", "quick_action", "weather")
    monkeypatch.setenv("NEO_LEARN", "off")
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    assert not learn.record("what time is it in Tokyo", "quick_action", "clock")
    assert [r["text"] for r in _rows()] == ["what's the weather in Dubai"]


def test_retention_and_forget(monkeypatch):
    old = time.time() - (learn._KEEP_DAYS + 1) * 86400
    learn.record("an old request from long ago", "chat")
    p = learn._path(learn.EXTRA)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["ts"] = old
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    learn.record("a recent request", "chat")
    assert [r["text"] for r in _rows()] == ["a recent request"]  # past the retention window
    assert learn.forget_examples() == 1 and _rows() == []


def test_agent_outcome_corrects_the_reflex(monkeypatch):
    from neo.agent.loop import AgentResult, Step
    from neo.agent.session import _learn_from
    from neo.reflex.schema import Decision
    from neo.tools import load_all

    load_all()
    d = Decision("agent_task", 0.7, False, 0, False, 0, "laya+head")
    _learn_from("tell me about black holes", d, AgentResult("Black holes are regions of spacetime where gravity is so strong that nothing escapes.", "done", []))
    _learn_from("do the usual thing", d, AgentResult("Which one do you mean — the report or the email?", "done", []))  # a question back
    _learn_from("how many tabs do i have", d, AgentResult("3", "done", [Step("apps_running", {}, "ok", True)]))
    _learn_from("tidy my desktop", d, AgentResult("Done.", "done", [Step("shell", {}, "", True), Step("shell", {}, "", True)]))
    rows = {r["text"]: (r["intent"], r["tool"]) for r in _rows()}
    assert rows == {"tell me about black holes": ("chat", ""), "how many tabs do i have": ("quick_action", "apps_running")}
