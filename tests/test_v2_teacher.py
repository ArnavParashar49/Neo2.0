"""The idle-time teacher: off by default, only models that already saw a request, two-model agreement."""

from __future__ import annotations

import json
import time

import pytest

from neo import config
from neo.providers.base import Turn
from neo.reflex import learn, teacher


@pytest.fixture
def teach(monkeypatch):
    monkeypatch.delenv("NEO_LEARN_TEACHER", raising=False)  # whatever the developer's shell says

    def set_mode(value: str, *, learn_on: bool = True):
        monkeypatch.setenv("NEO_LEARN_TEACHER", value)
        monkeypatch.setenv("NEO_LEARN", "on" if learn_on else "off")
        monkeypatch.setattr(config, "_settings", None)

    return set_mode


def _row(text, intent="chat", margin=0.1, **kw):
    return {"text": text, "intent": intent, "tool": "", "source": "agent-outcome", "ts": time.time() - 3600, "margin": margin, **kw}


def _write(rows):
    learn._path(learn.EXTRA).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _stored():
    return {r["text"]: r for r in learn._read(learn._path(learn.EXTRA))}


class Labeller:
    """Answers each utterance from `answers` (default: agree with nothing) and remembers what it saw."""

    def __init__(self, answers):
        self.answers, self.seen = answers, []

    async def complete(self, messages, *, system="", tools=None, effort="low", max_tokens=4000):
        texts = [line.split(". ", 1)[1] for line in messages[0].text.splitlines()[1:]]
        self.seen += texts
        return Turn(text=json.dumps([self.answers.get(t, "stop") for t in texts]))


def _providers(**by_name):
    async def provider(name):
        if name not in by_name:
            raise RuntimeError(f"{name} has no key")
        return by_name[name]

    return provider


def test_it_is_off_unless_asked_for_exactly(teach):
    assert teacher.mode() == "off"
    for value, want in (("local", "local"), (" Cloud ", "cloud"), ("on", "off"), ("clod", "off"), ("", "off")):
        teach(value)
        assert teacher.mode() == want
    teach("cloud", learn_on=False)
    assert teacher.mode() == "off"  # NEO_LEARN=off turns every kind of learning off


def test_cloud_only_uses_the_provider_that_already_heard_the_request():
    assert teacher.provider_of({"source": "live"}) == "gemini"
    assert teacher.provider_of({"source": "agent-outcome", "brain": "groq"}) == "groq"
    assert teacher.provider_of({"source": "agent-outcome", "brain": "gemini_lite"}) == "gemini"
    assert teacher.provider_of({"source": "agent-outcome"}) is None  # which brain ran it wasn't kept
    assert teacher.provider_of({"source": "agent-outcome", "brain": "someone-else"}) is None
    assert teacher.labeller_for({"source": "agent-outcome"}, "cloud") is None
    assert teacher.labeller_for({"source": "agent-outcome"}, "local") == "local"
    assert teacher.labeller_for({"source": "live"}, "off") is None


def test_picks_unsettled_rows_least_sure_first():
    rows = [
        _row("fairly sure", margin=0.5),
        _row("torn", margin=0.01),
        _row("settled", margin=0.0, verified=True),
        _row("corrected", margin=0.0, wrong=["quick_action"]),
        _row("seen by a teacher", margin=0.0, teacher="local"),
        _row("no margin", margin=None),
        _row("stopped", intent="nonsense", margin=0.0),
        _row("heard by live", margin=0.2, source="live"),
    ]
    assert [r["text"] for r in teacher.pick(rows, 10, "local")] == ["torn", "heard by live", "fairly sure"]
    assert [r["text"] for r in teacher.pick(rows, 2, "local")] == ["torn", "heard by live"]
    assert [r["text"] for r in teacher.pick(rows, 10, "cloud")] == ["heard by live"]  # the rest: provider unknown
    assert teacher.pick(rows, 10, "off") == []


def test_fifty_a_day(monkeypatch):
    assert teacher.budget_left() == teacher.DAILY_MAX == 50
    teacher._spend(30)
    teacher._spend(15)
    assert teacher.budget_left() == 5
    monkeypatch.setattr(teacher, "_today", lambda: "2999-01-01")
    assert teacher.budget_left() == 50  # a new day
    state = json.loads((config.settings().data_dir / teacher.STATE).read_text())
    assert set(state) == {"day", "sent"}  # counts only, never text


async def test_agreement_keeps_a_row_and_disagreement_drops_it():
    rows = [_row("tell me a joke", "chat"), _row("open the docs", "quick_action"), _row("hmm", "agent_task")]
    local = Labeller({"tell me a joke": "chat", "open the docs": "agent_task", "hmm": "not-a-class"})
    got = await teacher.verdicts(rows, "local", provider=_providers(local=local))
    by = {r["text"]: v for r, v in got}
    assert set(by) == {"tell me a joke", "open the docs"}  # no usable answer: no verdict
    assert by["tell me a joke"]["w"] == 1.0 and by["tell me a joke"]["agreed"] is True
    assert by["open the docs"]["w"] == 0.0 and by["open the docs"]["agreed"] is False
    assert by["open the docs"]["intent"] == "quick_action"  # a label is never changed
    assert all(v["teacher"] == "local" for v in by.values())


async def test_cloud_sends_each_request_only_to_its_own_provider():
    rows = [_row("live one", source="live"), _row("groq one", brain="groq"), _row("unknown one")]
    gem, grq = Labeller({"live one": "chat"}), Labeller({"groq one": "chat"})
    got = await teacher.verdicts(rows, "cloud", provider=_providers(gemini=gem, groq=grq))
    assert gem.seen == ["live one"] and grq.seen == ["groq one"]
    assert {r["text"] for r, _ in got} == {"live one", "groq one"}


async def test_a_missing_provider_or_a_misaligned_answer_labels_nothing():
    rows = [_row("a request", brain="claude"), _row("b request", brain="claude")]
    assert await teacher.verdicts(rows, "cloud", provider=_providers()) == []

    class Short(Labeller):
        async def complete(self, messages, **kw):
            await super().complete(messages, **kw)
            return Turn(text='["chat"]')  # one label for two utterances: whose?

    assert await teacher.verdicts(rows, "local", provider=_providers(local=Short({}))) == []


def test_a_verdict_never_outlives_an_erase_or_overrides_a_newer_label():
    first, second = _row("first request"), _row("second request")
    _write([first, second])
    gen = learn._erase_gen
    verdict = [(r, {**r, "w": 1.0, "teacher": "local", "agreed": True}) for r in (first, second)]
    learn.record("second request", "agent_task", source="live")  # a newer word on that sentence
    assert teacher.store(verdict, gen) == 1
    rows = _stored()
    assert rows["first request"]["teacher"] == "local" and rows["first request"]["ts"] == first["ts"]
    assert rows["second request"]["intent"] == "agent_task" and "teacher" not in rows["second request"]
    _write([first])
    learn.forget_examples()
    assert teacher.store(verdict, gen) == 0
    assert not learn._path(learn.EXTRA).exists()  # nothing written back after an erase


async def test_a_round_labels_within_the_budget_and_stores_the_verdicts(teach, monkeypatch):
    teach("local")
    _write([_row(f"request number {i}", "chat", margin=i / 100) for i in range(60)])
    learn._write_state({"trained_at": time.time() - 60, "trained_rows": 60})
    local = Labeller({f"request number {i}": "chat" if i % 2 else "agent_task" for i in range(60)})
    monkeypatch.setattr(teacher, "_provider", _providers(local=local))
    got = await teacher.run_once()
    assert got == {"labelled": 50, "agreed": 25, "stored": 50}
    assert local.seen == [f"request number {i}" for i in range(50)]  # the 50 least sure
    rows = _stored()
    assert sum(r.get("w") == 1.0 for r in rows.values()) == 25 and sum(r.get("w") == 0.0 for r in rows.values()) == 25
    assert learn.pending() == 0  # nothing counts as new: verdicts keep the rows' own times
    assert await teacher.run_once() == {"labelled": 0, "agreed": 0, "stored": 0}  # today's budget is spent
    assert len(local.seen) == 50


async def test_nothing_runs_when_it_is_off(teach, monkeypatch):
    _write([_row("a request")])

    async def boom(name):
        raise AssertionError("no model may be asked")

    monkeypatch.setattr(teacher, "_provider", boom)
    assert await teacher.run_once() == {"labelled": 0, "agreed": 0, "stored": 0}
    await teacher.teacher_loop(lambda: True)  # returns at once instead of looping
