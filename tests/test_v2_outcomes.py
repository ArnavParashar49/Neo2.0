"""Laya learns from what actually happened: the outcome ledger's labels, correction detection,
settle timing, priority retention, the user's own rows winning, the instant correction memory,
and erase — all hermetic (fake clock, fake tools, fake embeddings)."""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time

import numpy as np
import pytest

from neo.reflex import learn, memory, outcomes
from neo.reflex.schema import Decision

_ids = itertools.count(1)


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s
        outcomes.tick()


@pytest.fixture
def clock(monkeypatch):
    from neo.tools import load_all

    load_all()  # which tools are fast decides quick_action vs agent_task
    c = Clock()
    monkeypatch.setattr(outcomes, "_ledger", outcomes.Ledger(clock=c))
    return c


def D(intent="quick_action", conf=0.8, tool="", source="laya+head", tool_p=0.0):
    return Decision(intent, conf, False, 0.0, False, 0.0, source, tool=tool, tool_p=tool_p)


def rows():
    return learn._read(learn._path(learn.EXTRA))


def by_text():
    return {r["text"]: r for r in rows()}


def counts() -> dict:
    """Today's counters, wherever they are (RAM, or already written with a row)."""
    learn.flush_counts()
    out: dict = {}
    for day in (learn._state().get("daily") or {}).values():
        for k, v in day.items():
            out[k] = out.get(k, 0) + v
    return out


def ask(text, intent="quick_action", conf=0.8, tool=""):
    """A typed request, as session.handle reports it."""
    k = f"j{next(_ids)}"
    outcomes.begin(k, text, decision=D(intent, conf, tool))
    return k


def quick(k, *steps):
    outcomes.route(k, "quick")
    for tool, ok in steps:
        outcomes.step(k, tool, ok)
    outcomes.end(k)


# ---- labelling rules ------------------------------------------------------------------------------
def test_a_verified_outcome_settles_20s_after_its_last_event(clock):
    k = ask("open safari please", conf=0.82, tool="open_app")
    quick(k, ("open_app", True))
    clock.advance(19)
    assert rows() == []  # it might still be corrected
    clock.advance(1.5)
    (r,) = rows()
    assert (r["intent"], r["tool"], r["w"], r["verified"]) == ("quick_action", "open_app", 1.0, True)
    assert r["conf"] == 0.82 and r["script"] == "latin" and r["source"] == "neo"
    assert r["ts"] % 86400 == 0  # the day, never the moment


@pytest.mark.parametrize(
    "steps, answer, expected",
    [
        ([], "The capital of France is Paris, on the Seine.", ("chat", "")),  # answered, nothing to run
        ([("weather", True)], "", ("quick_action", "weather")),  # one fast tool
        ([("open_app", True), ("hotkey", True)], "", ("agent_task", "")),  # more than one tool
        ([("safari_read", True)], "", ("agent_task", "")),  # one slow tool
        ([("more_tools", True), ("clock", True)], "", ("quick_action", "clock")),  # asking for tools isn't one
    ],
)
def test_what_ran_is_the_label(clock, steps, answer, expected):
    k = ask("some request here", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, steps, stopped="done", answer=answer or "Done, all set for you.")
    clock.advance(30)
    (r,) = rows()
    assert (r["intent"], r["tool"], r["verified"]) == (*expected, True)


def test_the_chat_route_counts_unless_it_could_not_help(clock):
    k = ask("how does a rainbow form", intent="chat")
    outcomes.route(k, "chat")
    outcomes.reply(k, "Sunlight bends inside raindrops.")
    k2 = ask("open my downloads folder", intent="chat")
    outcomes.route(k2, "chat")
    outcomes.reply(k2, "I can't open folders from here.")  # the reflex sent an action to chat
    clock.advance(30)
    assert {t: (r["intent"], r["verified"]) for t, r in by_text().items()} == {"how does a rainbow form": ("chat", True)}


def test_a_failed_step_then_another_tool_that_worked_is_the_label(clock):
    k = ask("the forecast for sales this quarter", tool="weather")
    outcomes.route(k, "quick")
    outcomes.step(k, "weather", False)  # the fast path failed and handed over to the agent
    outcomes.route(k, "agent")
    outcomes.agent_done(k, [("web_search", True)], stopped="done", answer="Analysts expect sales to rise about 4% this quarter.")
    clock.advance(30)
    (r,) = rows()
    assert (r["intent"], r["tool"], r["verified"], r["w"]) == ("quick_action", "web_search", True, 1.0)


@pytest.mark.parametrize(
    "steps, stopped, answer",
    [
        ([("open_app", False), ("ax_press", False)], "done", "That didn't work."),  # nothing recovered
        ([("open_app", True), ("ax_press", False)], "done", "Done."),  # the last thing failed
        ([("ax_tree", True)], "max_steps", "Still working on it."),  # ran out of steps
        ([("ax_tree", True), ("ax_press", True)], "done", "I couldn't find the button."),  # says it failed
        ([], "done", "Which one do you mean, the report or the email?"),  # asked back
    ],
)
def test_a_failure_teaches_nothing(clock, steps, stopped, answer):
    k = ask("do the thing in the app", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, steps, stopped=stopped, answer=answer)
    clock.advance(30)
    assert rows() == []


def test_a_cancelled_request_teaches_nothing(clock):
    k = ask("organise my desktop files", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.cancelled(k)
    clock.advance(30)
    assert rows() == []


def test_a_stop_within_3s_is_a_defect_not_a_row(clock):
    k = ask("open safari please")
    quick(k, ("open_app", True))
    clock.advance(1.0)
    outcomes.stop()  # "stop!" — whatever NEO just did was wrong
    clock.advance(30)
    assert rows() == [] and counts()["barge_ins"] == 1

    k = ask("open notes please")
    quick(k, ("open_app", True))
    clock.advance(5.0)
    outcomes.stop()  # much later: the user is just done with it
    clock.advance(30)
    assert [r["text"] for r in rows()] == ["open notes please"] and counts()["barge_ins"] == 1


def test_no_at_a_confirmation_teaches_nothing_and_yes_counts_the_action(clock):
    k = ask("delete the old screenshots", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, [("finder_reveal", True)], stopped="needs_confirm", answer="Move 12 files to the Trash?")
    clock.advance(60)
    assert rows() == []  # still waiting for the answer
    outcomes.confirm(False)
    clock.advance(30)
    assert rows() == []

    k = ask("empty the downloads folder", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, [("finder_reveal", True)], stopped="needs_confirm", answer="Move 40 files to the Trash?")
    outcomes.confirm(True, "trash", True)
    clock.advance(30)
    (r,) = rows()
    assert (r["text"], r["intent"], r["verified"]) == ("empty the downloads folder", "agent_task", True)


def test_a_spoken_yes_to_the_question_is_not_a_request(clock):
    k = ask("delete the old screenshots", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, [("finder_reveal", True)], stopped="needs_confirm", answer="Move 12 files to the Trash?")
    lab = {"text": "", "tools": [], "spoke": False, "id": outcomes.begin("", "", source="live", final=False, bind=False)}
    outcomes.confirm(True, "trash", True)  # the user's "yes" ran it
    lab.update(text="yes please go ahead", tools=["agent_task"], spoke=True)
    outcomes.file_live(lab)
    clock.advance(30)
    assert [r["text"] for r in rows()] == ["delete the old screenshots"]


def test_a_request_after_declining_is_its_own(clock):
    k = ask("delete the old screenshots", intent="agent_task")
    outcomes.route(k, "agent")
    outcomes.agent_done(k, [("finder_reveal", True)], stopped="needs_confirm", answer="Move 12 files to the Trash?")
    outcomes.confirm(False)
    b = ask("no, I meant the downloads folder instead", intent="agent_task")  # not a fix of a declined request
    outcomes.route(b, "agent")
    outcomes.agent_done(b, [("finder_reveal", True), ("trash", True)], stopped="done", answer="Moved them to the Trash.")
    clock.advance(30)
    assert [(r["text"], r.get("wrong")) for r in rows()] == [("no, I meant the downloads folder instead", None)]


def test_a_halted_quick_step_waits_for_the_answer(clock):
    k = ask("quit all my apps")
    outcomes.route(k, "quick")
    outcomes.step(k, "quit_app", True, halt=True)  # NEEDS_CONFIRM, not done yet
    outcomes.end(k)
    clock.advance(outcomes.CONFIRM_WAIT_S + 1)
    assert rows() == []  # nobody answered: unknown, not a label


# ---- corrections ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "no, I meant the volume",
        "nope, the other one",
        "not that one",
        "I meant Spotify",
        "i said open notes",
        "wrong app",
        "Neo, no the music app",
        "galat hai, awaaz kam karo",
        "nahi yaar volume",
        "nahin",
        "arre nahi volume",
        "mera matlab tha volume",
        "matlab awaaz",
        "uska nahi",
        "dusra wala",
        "doosra gaana",
    ],
)
def test_correction_words_in_english_and_hinglish(text):
    assert outcomes._CORRECTION.match(text)


@pytest.mark.parametrize("text", ["note that I'm late", "nothing much", "open notes", "know any jokes", "is this wrong?"])
def test_ordinary_requests_are_not_corrections(text):
    assert not outcomes._CORRECTION.match(text)


def test_a_correction_relabels_the_request_it_corrects(clock):
    a = ask("turn it down", conf=0.52, tool="brightness")
    quick(a, ("brightness", True))
    clock.advance(4)
    b = ask("no, I meant the volume")
    quick(b, ("volume", True))
    clock.advance(30)
    got = by_text()
    fixed = got["turn it down"]
    assert (fixed["intent"], fixed["tool"], fixed["verified"], fixed["w"]) == ("quick_action", "volume", True, 1.0)
    assert fixed["wrong"] == ["quick_action:brightness"] and fixed["conf"] == 0.52  # a hard negative
    assert got["no, I meant the volume"]["tool"] == "volume"
    assert counts()["corrections"] == 1


def test_a_hinglish_correction(clock):
    a = ask("thoda kam karo", tool="brightness")
    quick(a, ("brightness", True))
    clock.advance(3)
    b = ask("galat, awaaz kam karo")
    quick(b, ("volume", True))
    clock.advance(30)
    assert by_text()["thoda kam karo"]["wrong"] == ["quick_action:brightness"]


def test_a_rephrase_after_a_failure_gets_what_finally_worked(clock):
    a = ask("what is the weather in pune", tool="weather")
    outcomes.route(a, "quick")
    outcomes.step(a, "weather", False)
    outcomes.agent_done(a, [], stopped="done", answer="Sorry, I couldn't get that.")
    clock.advance(6)
    b = ask("what is the weather like in pune today")  # most of the same words: a rephrase
    quick(b, ("web_search", True))
    clock.advance(30)
    r = by_text()["what is the weather in pune"]
    assert (r["tool"], r["wrong"], r["verified"]) == ("web_search", ["quick_action:weather"], True)


def test_more_words_on_a_request_that_worked_is_a_follow_up(clock):
    a = ask("play despacito now", tool="play_music")
    quick(a, ("play_music", True))
    clock.advance(4)
    b = ask("play despacito now on youtube", intent="agent_task")
    outcomes.route(b, "agent")
    outcomes.agent_done(b, [("open_url", True)], stopped="done", answer="Playing it on YouTube.")
    clock.advance(30)
    got = by_text()
    assert (got["play despacito now"]["tool"], got["play despacito now"].get("wrong")) == ("play_music", None)
    assert got["play despacito now on youtube"]["intent"] == "agent_task"


def test_a_correction_nothing_verified_leaves_no_row_but_counts(clock):
    a = ask("open the report", tool="open_app")
    quick(a, ("open_app", True))
    clock.advance(3)
    b = ask("no not that report", intent="agent_task")
    outcomes.route(b, "agent")
    outcomes.agent_done(b, [], stopped="done", answer="Which report do you mean?")
    clock.advance(30)
    assert rows() == [] and counts()["corrections"] == 1


def test_a_bare_no_then_the_fix(clock):
    a = ask("check the weather", tool="web_search")
    quick(a, ("web_search", True))
    clock.advance(3)
    outcomes.begin("jno", "no", decision=D("chat"))  # not a request of its own
    clock.advance(4)
    c = ask("check the weather app")
    quick(c, ("open_app", True))
    clock.advance(30)
    got = by_text()
    assert set(got) == {"check the weather", "check the weather app"}
    assert (got["check the weather"]["tool"], got["check the weather"]["wrong"]) == ("open_app", ["quick_action:web_search"])


def test_a_bare_no_with_nothing_after_it_drops_the_request(clock):
    a = ask("check the weather", tool="web_search")
    quick(a, ("web_search", True))
    clock.advance(3)
    outcomes.begin("jno", "no", decision=D("chat"))
    clock.advance(60)
    assert rows() == [] and counts()["corrections"] == 1


def test_a_chain_of_corrections_all_get_the_fix(clock):
    a = ask("play the jungle book", tool="play_music")
    quick(a, ("play_music", True))
    clock.advance(3)
    b = ask("no, the movie jungle book")
    quick(b, ("play_music", False))
    clock.advance(3)
    c = ask("i said the movie")
    quick(c, ("open_app", True))
    clock.advance(30)
    got = by_text()
    for t in ("play the jungle book", "no, the movie jungle book"):
        assert (got[t]["tool"], got[t]["wrong"]) == ("open_app", ["quick_action:play_music"])


# ---- settle timing ---------------------------------------------------------------------------------
def test_the_next_unrelated_request_settles_the_last_one_at_once(clock):
    a = ask("open safari please")
    quick(a, ("open_app", True))
    clock.advance(2)
    ask("what time is it in tokyo", tool="clock")
    assert [r["text"] for r in rows()] == ["open safari please"]  # no 20 s wait


def test_a_request_long_after_is_unrelated_even_if_similar(clock):
    a = ask("turn it down", tool="brightness")
    quick(a, ("brightness", True))
    clock.advance(25)  # already settled on its own
    b = ask("no, turn it down")
    quick(b, ("volume", True))
    clock.advance(30)
    assert by_text()["turn it down"]["tool"] == "brightness"


def test_a_live_sentence_holds_the_previous_request_until_its_words_are_final(clock):
    a = ask("turn it down", tool="brightness")
    quick(a, ("brightness", True))
    clock.advance(3)
    lab = {"text": "", "tools": [], "spoke": False, "id": outcomes.begin("", "", source="live", final=False, bind=False)}
    clock.advance(40)
    assert rows() == []  # the words still arriving may be a correction
    outcomes.call_started(lab["id"])  # the model's call for it ran, and worked
    outcomes.step("", "volume", True)
    outcomes.call_done(lab["id"])
    lab.update(text="no I meant volume", tools=["volume"])
    outcomes.file_live(lab)
    clock.advance(30)
    assert by_text()["turn it down"]["wrong"] == ["quick_action:brightness"]


def test_a_request_that_never_finishes_is_dropped(clock):
    k = ask("organise my photos", intent="agent_task")
    outcomes.route(k, "agent")
    clock.advance(outcomes.STALE_S - 1)
    assert len(outcomes.ledger()) == 1
    clock.advance(2)
    assert len(outcomes.ledger()) == 0 and rows() == []


def test_nothing_is_tracked_with_learning_off(clock, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LEARN", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert outcomes.begin("j1", "open safari please", decision=D()) == ""
    assert len(outcomes.ledger()) == 0


# ---- neo/decide's task kinds -----------------------------------------------------------------------
def test_a_task_kind_is_learned_only_when_it_worked(clock):
    k = ask("play some lofi beats", tool="play_music")
    assert outcomes.decided(k, "play_media", "spotify", True)
    quick(k, ("play_music", True))
    k2 = ask("play the new album by queen", tool="play_music")
    outcomes.decided("play the new album by queen", "play_media", "music", False)  # by its words
    quick(k2, ("play_music", True))
    clock.advance(30)
    got = by_text()
    assert got["play some lofi beats"]["kind"] == "play_media" and got["play some lofi beats"]["verified"]
    assert "kind" not in got["play the new album by queen"]
    assert not outcomes.decided("something nobody asked", "", "", True)


def test_a_corrected_request_learns_no_kind(clock):
    a = ask("message mom", intent="agent_task")
    outcomes.decided(a, "send_message", "messages", True)
    quick(a, ("open_app", True))
    clock.advance(2)
    b = ask("no, email mom", intent="agent_task")
    quick(b, ("open_app", True), ("hotkey", True))
    clock.advance(30)
    assert "kind" not in by_text()["message mom"]


# ---- learn.py: rows, retention, counters ------------------------------------------------------------
def test_rows_keep_the_day_and_a_counter_so_pending_and_due_stay_right(monkeypatch):
    for i in range(3):
        learn.record(f"open app number {i}", "quick_action", "open_app", w=1.0, verified=True)
    assert all(r["ts"] % 86400 == 0 for r in rows()) and [r["seq"] for r in rows()] == [1, 2, 3]
    assert learn.pending() == 3
    learn._write_state({"trained_at": time.time(), "trained_rows": 3})  # trained just now
    assert learn.pending() == 0  # same day, yet nothing is new
    learn.record("open app number 3", "quick_action", "open_app")
    assert learn.pending() == 1 and not learn.due()
    for i in range(learn._MIN_FIXES - 1):
        learn.record(f"turn it down {i}", "quick_action", "volume", verified=True, wrong=["quick_action:brightness"])
    assert not learn.due()  # 4 corrections, trained a moment ago
    learn.record("turn that down", "quick_action", "volume", verified=True, wrong=["quick_action:brightness"])
    assert learn.due()  # 5 corrections retrain sooner


def test_rows_from_before_the_counter_still_count_by_time():
    p = learn._path(learn.EXTRA)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"text": "an older request", "intent": "chat", "tool": "", "ts": time.time() - 10}) + "\n")
    learn._write_state({"trained_at": time.time() - 3600, "trained_rows": 0, "trained_seq": 0})
    assert learn.pending() == 1


def test_eviction_keeps_corrections_then_unsure_verified_then_verified_then_guesses(monkeypatch):
    monkeypatch.setattr(learn, "_KEEP_ROWS", 4)
    learn.record("teacher guess one", "chat", w=0.5, verified=False)
    learn.record("verified sure one", "chat", w=1.0, verified=True, conf=0.95)
    learn.record("verified unsure one", "chat", w=1.0, verified=True, conf=0.4)
    learn.record("corrected one here", "quick_action", "volume", verified=True, wrong=["chat"], conf=0.97)
    learn.record("teacher guess two", "chat", w=0.5, verified=False)
    learn.record("verified sure two", "chat", w=1.0, verified=True, conf=0.9)
    assert [r["text"] for r in rows()] == ["verified sure one", "verified unsure one", "corrected one here", "verified sure two"]
    learn.record("verified sure three", "chat", w=1.0, verified=True, conf=0.9)
    assert [r["text"] for r in rows()] == ["verified unsure one", "corrected one here", "verified sure two", "verified sure three"]
    learn._compact(learn._path(learn.EXTRA))  # the file on disk holds to the same rule
    assert len(learn._read_raw(learn._path(learn.EXTRA))) == 4


def test_guesses_expire_sooner_and_never_override_a_verified_outcome():
    learn.record("old verified request", "chat", verified=True)
    learn.record("old guessed request", "chat", verified=False)
    p = learn._path(learn.EXTRA)
    old = [json.loads(line) for line in p.read_text().splitlines()]
    for r in old:
        r["ts"] = learn._day(time.time() - 31 * 86400)
    p.write_text("".join(json.dumps(r) + "\n" for r in old))
    assert [r["text"] for r in rows()] == ["old verified request"]  # 30 days for a guess, 180 verified
    learn.record("Old verified  request", "agent_task", verified=False)
    assert [(r["intent"], r["verified"]) for r in rows()] == [("chat", True)]
    learn.record("old verified request", "quick_action", "clock", verified=True)
    assert [(r["intent"], r["tool"]) for r in rows()] == [("quick_action", "clock")]  # a newer outcome does


def test_devanagari_is_tagged_and_the_counters_carry_no_text():
    assert learn.record("आवाज़ कम करो", "quick_action", "volume", verified=True)
    assert rows()[0]["script"] == "devanagari"
    learn.count("corrections")
    learn.count("requests", 3)
    learn.flush_counts()
    st = json.loads(learn._path(learn.STATE).read_text())
    (today,) = st["daily"].values()
    assert today == {"corrections": 1, "requests": 3}
    assert "3 requests" in learn.summary() and "1 corrections" in learn.summary()
    assert "आवाज़" not in learn._path(learn.STATE).read_text()


def test_a_correction_to_a_content_tool_keeps_nothing():
    assert not learn.record("tell priya hi", "quick_action", "volume", wrong=["quick_action:type_text"])


# ---- dataset: the user's verified rows win --------------------------------------------------------
def test_own_verified_rows_override_every_other_source():
    from neo.reflex import dataset

    d = learn._path("x").parent
    d.mkdir(parents=True, exist_ok=True)
    (d / "reflex_generated.jsonl").write_text(
        json.dumps({"text": "Turn it  DOWN", "intent": "chat"}) + "\n" + json.dumps({"text": "play some jazz", "intent": "quick_action"}) + "\n"
    )
    learn.record("turn it down", "quick_action", "volume", w=1.0, verified=True, wrong=["quick_action:brightness"], conf=0.5)
    learn.record("play some jazz", "agent_task", w=0.5, verified=False)  # a guess overrides nothing
    built = dataset.build(5)
    down = [r for r in built if learn.norm(r["text"]) == "turn it down"]
    assert len(down) == 1 and down[0]["source"] == "reflex_extra.jsonl"
    assert (down[0]["w"], down[0]["verified"], down[0]["wrong"], down[0]["conf"]) == (1.0, True, ["quick_action:brightness"], 0.5)
    jazz = sorted(r["source"] for r in built if r["text"] == "play some jazz")
    assert jazz == ["reflex_extra.jsonl", "reflex_generated.jsonl"]


# ---- the instant correction memory -----------------------------------------------------------------
DIM = 16
C = np.zeros(DIM, np.float32)
C[0] = 1.0  # every embedding leans this way (Laya's anisotropy)


def emb(angle_deg: float, along: int = 1) -> np.ndarray:
    """An embedding whose centred direction sits `angle_deg` from axis 1 (towards `along`+1)."""
    a = np.radians(angle_deg)
    v = C.copy()
    v[along] += 0.2 * np.cos(a)
    v[along + 1] += 0.2 * np.sin(a)
    return v / np.linalg.norm(v)


def remember(text="thoda kam karo", intent="quick_action", tool="volume"):
    memory.memory().seen(text, emb(0))
    assert memory.memory().learned(text, intent, tool)


@pytest.mark.parametrize(
    "angle, head, override",
    [
        (15, D("quick_action", 0.6, "brightness", tool_p=0.6), True),  # centred cosine 0.97
        (30, D("quick_action", 0.6, "brightness", tool_p=0.6), False),  # 0.87: not the same request
        (15, D("chat", 0.93), False),  # the head is sure of something else
        (15, D("chat", 0.7), True),
        (15, D("quick_action", 0.95, "brightness", tool_p=0.95), False),  # sure of its tool
        (15, D("quick_action", 0.95, "volume", tool_p=0.95), False),  # already agrees
    ],
)
def test_memory_overrides_only_a_real_repeat(angle, head, override):
    remember()
    out = memory.recall(head, "thoda kam karo na", emb(angle), C)
    if override:
        assert (out.intent, out.tool, out.source) == ("quick_action", "volume", "laya+memory")
        assert out.intent_confidence >= memory.TAU and out.tool_p >= memory.TAU
        assert counts()["memory_hits"] == 1
    else:
        assert out is head


def test_memory_needs_the_same_script_and_centring():
    remember()
    head = D("chat", 0.7)
    assert memory.recall(head, "थोड़ा कम करो", emb(15), C) is head  # Devanagari vs Latin
    unrelated = emb(0, along=5)  # raw cosine with the stored one ≈ 0.96, centred ≈ 0
    assert float(unrelated @ emb(0)) > 0.9
    assert memory.recall(head, "something else entirely", unrelated, C) is head


def test_memory_without_an_embedding_from_the_reflex_does_nothing():
    remember()
    head = D("chat", 0.7)
    assert memory.recall(head, "thoda kam karo", None, C) is head
    assert memory.recall(head, "thoda kam karo", emb(0), None) is head


def test_the_reflex_consults_the_memory(monkeypatch):
    import neo.reflex as reflex_mod

    class Laya:
        def __init__(self, attrs=True):
            self.attrs = attrs

        def decide(self, text):
            if self.attrs:
                self.last_embedding, self.center = emb(5), C
            return D("chat", 0.7)

    remember()
    monkeypatch.setattr(reflex_mod, "get_laya", lambda: Laya())
    assert reflex_mod.Reflex()._laya_decide("thoda kam karo").source == "laya+memory"
    monkeypatch.setattr(reflex_mod, "get_laya", lambda: Laya(attrs=False))  # an older reflex
    assert reflex_mod.Reflex()._laya_decide("thoda kam karo").source == "laya+head"


def test_a_correction_the_reflex_just_decided_is_remembered_at_once(clock, monkeypatch):
    memory.memory().seen("turn it down", emb(0))  # what _laya_decide saw a moment ago
    a = ask("turn it down", tool="brightness")
    quick(a, ("brightness", True))
    clock.advance(3)
    b = ask("no, I meant the volume")
    quick(b, ("volume", True))
    clock.advance(30)
    assert len(memory.memory()) == 1
    out = memory.recall(D("quick_action", 0.6, "brightness", tool_p=0.6), "turn it down", emb(0), C)
    assert (out.tool, out.source) == ("volume", "laya+memory")


def test_memory_is_rebuilt_from_stored_corrections_and_stops_when_busy():
    learn.record("thoda kam karo", "quick_action", "volume", verified=True, wrong=["quick_action:brightness"])
    learn.record("open the music app", "quick_action", "open_app", verified=True)  # not a correction
    m = memory.memory()
    assert m.stale and len(m) == 0
    assert m.rebuild(lambda t: emb(0), keep_going=lambda: False) == 0 and m.stale  # busy: later
    assert m.rebuild(lambda t: emb(0)) == 1 and not m.stale and len(m) == 1


def test_memory_rebuild_only_uses_an_already_loaded_laya(monkeypatch):
    import neo.reflex as reflex_mod

    monkeypatch.setattr(reflex_mod, "_laya", None)
    assert not memory.rebuild_soon(lambda: True)

    class Laya:
        def decide(self, text):
            self.last_embedding = emb(0)

    learn.record("thoda kam karo", "quick_action", "volume", verified=True, wrong=["quick_action:brightness"])
    monkeypatch.setattr(reflex_mod, "_laya", Laya())
    assert memory.rebuild_soon(lambda: True)
    for _ in range(100):
        if not memory.memory()._rebuilding:
            break
        time.sleep(0.01)
    assert len(memory.memory()) == 1


# ---- erase ------------------------------------------------------------------------------------------
def test_erase_forgets_requests_in_flight_and_the_memory(clock):
    from neo import privacy

    remember()
    k = ask("open safari please")
    outcomes.route(k, "quick")
    assert len(outcomes.ledger()) == 1 and len(memory.memory()) == 1
    asyncio.run(privacy.forget_everything())
    assert len(outcomes.ledger()) == 0 and len(memory.memory()) == 0
    outcomes.step(k, "open_app", True)  # the run finishing afterwards
    outcomes.end(k)
    clock.advance(60)
    assert rows() == []


# ---- the hooks: Gemini Live ------------------------------------------------------------------------
def _settle():
    outcomes.tick(time.time() + 3600)


@pytest.fixture
def voice(monkeypatch):
    from test_v2_speech_gate import FakeLive, FakeSpk

    from neo import config, events
    from neo.agent.registry import Registry, ToolOutput
    from neo.events import EventBus
    from neo.tools import load_all
    from neo.voice import live as live_mod

    load_all()
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    gate = threading.Event()

    async def fake_invoke(self, name, args, ctx):
        if name == "web_fetch":  # slow: still running when the user stops it
            while not gate.is_set():
                await asyncio.sleep(0.01)
        return ToolOutput(f"did {name}", ok=name != "open_url")

    monkeypatch.setattr(Registry, "invoke", fake_invoke)

    class Session:
        memory_context = ""

        async def handle(self, text, **kw):  # what the real session reports for a relayed run
            from neo.agent.session import Reply

            outcomes.route("j-relay", "agent")
            outcomes.agent_done("j-relay", [("web_search", True), ("web_fetch", True)], stopped="done", answer="Here's the summary.")
            return Reply("Here's the summary.", "agent", silent=False)

    v = live_mod.LiveVoice(lambda t: None, Session())
    v.spk, v._live = FakeSpk(), FakeLive()
    return v


def _run(v, *msgs, settle=0.1):
    from test_v2_speech_gate import feed

    return asyncio.run(feed(v, *msgs, settle=settle))


def test_live_a_tool_that_worked_is_a_verified_row(voice):
    from test_v2_speech_gate import DONE, call, heard

    _run(voice, heard("what's the weather in paris"), call("weather", {"place": "Paris"}), DONE)
    voice._file_label()
    _settle()
    (r,) = rows()
    assert (r["text"], r["intent"], r["tool"], r["verified"], r["source"]) == ("what's the weather in paris", "quick_action", "weather", True, "live")


def test_live_a_failed_call_is_no_row(voice):
    from test_v2_speech_gate import DONE, call, heard

    _run(voice, heard("open example dot com"), call("open_url", {"url": "https://example.com"}), DONE)
    voice._file_label()
    _settle()
    assert rows() == []


def test_live_a_relayed_run_labels_the_users_own_words(voice):
    from test_v2_speech_gate import DONE, call, heard

    _run(voice, heard("summarise the top story on hacker news"), call("agent_task", {"goal": "Summarise the top HN story"}), DONE)
    voice._file_label()
    _settle()
    (r,) = rows()
    assert (r["text"], r["intent"], r["verified"]) == ("summarise the top story on hacker news", "agent_task", True)


def test_live_a_call_the_user_stopped_is_no_row(voice):
    from test_v2_speech_gate import call, heard

    async def run():
        from test_v2_speech_gate import feed

        await feed(voice, heard("read that article for me"), call("web_fetch", {"url": "https://example.com"}), settle=0.05)
        await voice.interrupt()
        await asyncio.sleep(0.05)

    asyncio.run(run())
    voice._file_label()
    _settle()
    assert rows() == []


def test_live_talking_over_the_answer_at_once_is_a_defect(voice):
    from test_v2_speech_gate import audio, heard, said, sc

    _run(voice, heard("tell me a joke please"), said("Why did the"), audio(), sc(interrupted=True))
    voice._file_label()
    _settle()
    assert rows() == [] and counts()["barge_ins"] == 1


# ---- the hooks: NEO's own session ------------------------------------------------------------------
@pytest.fixture
def session(monkeypatch):
    from test_v2_session import FakeChain

    from neo import events
    from neo.agent import session as sess_mod
    from neo.agent.registry import Registry, ToolOutput
    from neo.events import EventBus
    from neo.tools import load_all

    load_all()
    monkeypatch.setattr(events, "_bus", EventBus())
    ran = []

    async def fake_invoke(self, name, args, ctx):  # never launches anything
        ran.append(name)
        return ToolOutput(f"did {name}", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": FakeChain())

    class Reflex:
        intent = "quick_action"

        async def decide(self, text):
            from neo.reflex import is_bare_stop

            if is_bare_stop(text):
                return Decision("stop", 1.0, False, 0.0, False, 0.0, "rules")
            return D(self.intent, 0.66)

    s = sess_mod.Session(reflex=Reflex())
    s.ran = ran
    return s


def test_session_a_quick_command_that_worked_is_a_verified_row(session):
    asyncio.run(session.handle("open safari", from_user=True))
    _settle()
    (r,) = rows()
    assert (r["intent"], r["tool"], r["verified"], r["conf"]) == ("quick_action", "open_app", True, 0.66)
    assert session.ran == ["open_app"]


def test_session_a_chat_answer_is_a_verified_row(session):
    session.reflex.intent = "chat"
    asyncio.run(session.handle("how are you doing today", from_user=True))
    _settle()
    assert [(r["text"], r["intent"], r["verified"]) for r in rows()] == [("how are you doing today", "chat", True)]


def test_session_stop_right_after_a_command_is_a_defect(session):
    async def run():
        await session.handle("open safari", from_user=True)
        await session.handle("stop", from_user=True)

    asyncio.run(run())
    _settle()
    assert rows() == [] and counts()["barge_ins"] == 1


def test_session_a_relayed_request_is_not_an_episode_of_its_own(session):
    asyncio.run(session.handle("open safari", conversation="user: open safari"))
    _settle()
    assert rows() == [] and len(outcomes.ledger()) == 0
