"""The zero-shot Laya model: loaded on first use off the caller's thread, one forward pass at a time
under the shared Laya lock, dropped when idle. A fake loader stands in for the checkpoint; nothing
here downloads or loads a model."""

from __future__ import annotations

import threading
import time
import types

import pytest

import neo.reflex
from neo.reflex import zeroshot

Q = {"app": {"type": "choice", "instructions": "Which?", "criteria": {"A": "a", "B": "b"}}}


class FakeAgent:
    def __init__(self):
        self.calls: list[object] = []
        self.locked: list[bool] = []
        self.closed = False

    def system_one(self, state, questions):
        self.locked.append(neo.reflex.laya_lock().locked())
        self.calls.append(state)
        answer = {"type": "choice", "choice": "A", "probabilities": {"A": 0.9, "B": 0.1}}
        return {"answers": {q: answer for q in questions}}

    def __exit__(self, *exc):
        self.closed = True


@pytest.fixture
def loader(monkeypatch):
    """Laya on, the checkpoint faked: ``loader.gate`` holds the load until set."""
    from neo import config

    monkeypatch.setenv("NEO_REFLEX", "laya")
    monkeypatch.setattr(config, "_settings", None)
    state = types.SimpleNamespace(gate=threading.Event(), opened=[], locked_while_opening=[], fail=False)
    state.gate.set()

    def fetch():
        return "/nowhere/typed-decisions"

    def open_(path):
        state.locked_while_opening.append(neo.reflex.laya_lock().locked())
        state.gate.wait(5)
        if state.fail:
            raise RuntimeError("no torch here")
        agent = FakeAgent()
        state.opened.append(agent)
        return agent

    monkeypatch.setattr(zeroshot, "_fetch", fetch)
    monkeypatch.setattr(zeroshot, "_open", open_)
    return state


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(zeroshot, "_agent", None)
    monkeypatch.setattr(zeroshot, "_loading", None)
    monkeypatch.setattr(zeroshot, "_failed_at", float("-inf"))
    monkeypatch.setattr(zeroshot, "_used", 0.0)
    yield
    t = zeroshot._timer
    if t is not None:
        t.cancel()


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_nothing_loads_until_asked_then_once_and_under_the_lock(loader):
    assert zeroshot._agent is None and not loader.opened  # importing it costs nothing
    res = zeroshot.ask({"request": "x"}, Q, wait_s=2.0)
    assert res["answers"]["app"]["choice"] == "A"
    agent = loader.opened[0]
    assert loader.locked_while_opening == [True]  # the load onto the GPU holds the shared lock
    assert agent.calls[0] == {"text": "warm up"}  # the slow first pass is taken while loading
    assert agent.locked == [True, True]  # every forward pass holds it
    zeroshot.ask({"request": "y"}, Q, wait_s=2.0)
    assert len(loader.opened) == 1 and agent.calls[-1] == {"request": "y"}


def test_concurrent_first_questions_share_one_load(loader):
    loader.gate.clear()
    out = []
    threads = [threading.Thread(target=lambda: out.append(zeroshot.ask("s", Q, wait_s=2.0))) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    loader.gate.set()
    for t in threads:
        t.join()
    assert len(loader.opened) == 1 and all(r is not None for r in out)


def test_a_slow_load_answers_none_in_time_and_never_computes_the_stale_question(loader):
    loader.gate.clear()
    stale = threading.Event()
    t = time.perf_counter()
    assert zeroshot.ask("s", Q, wait_s=0.1, stale=stale) is None  # the caller's prior decides
    assert time.perf_counter() - t < 1.0
    stale.set()  # the caller gave up
    loader.gate.set()
    assert wait_until(lambda: zeroshot._agent is not None)
    assert loader.opened[0].calls == [{"text": "warm up"}]
    assert zeroshot.ask("s2", Q, wait_s=1.0) is not None  # loaded now: the next question is answered


def test_a_question_that_went_stale_waiting_for_the_lock_is_skipped(loader):
    zeroshot.ask("s", Q, wait_s=2.0)
    stale = threading.Event()
    with neo.reflex.laya_lock():
        out = []
        t = threading.Thread(target=lambda: out.append(zeroshot.ask("late", Q, wait_s=2.0, stale=stale)))
        t.start()
        time.sleep(0.1)
        stale.set()
    t.join()
    assert out == [None] and loader.opened[0].calls[-1] == "s"


def test_a_busy_laya_costs_the_wait_not_more(loader):
    zeroshot.ask("s", Q, wait_s=2.0)
    held, done = threading.Event(), threading.Event()

    def reflex_busy():
        with neo.reflex.laya_lock():
            held.set()
            done.wait(3)

    holder = threading.Thread(target=reflex_busy)
    holder.start()
    held.wait(1)
    try:
        t = time.perf_counter()
        assert zeroshot.ask("s", Q, wait_s=0.05) is None
        assert time.perf_counter() - t < 0.5
    finally:
        done.set()
        holder.join()


def test_idle_model_is_unloaded_and_comes_back_on_the_next_question(loader, monkeypatch):
    monkeypatch.setattr(zeroshot, "IDLE_S", 0.5)
    zeroshot.ask("s", Q, wait_s=2.0)
    first = loader.opened[0]
    time.sleep(0.25)
    zeroshot.ask("s", Q, wait_s=2.0)  # use postpones the unload
    time.sleep(0.3)  # past IDLE_S since the load, not since the last question
    assert zeroshot._agent is first and not first.closed
    assert wait_until(lambda: zeroshot._agent is None)
    assert first.closed  # its memory handed back
    assert zeroshot.ask("s", Q, wait_s=2.0) is not None
    assert len(loader.opened) == 2


def test_unload_waits_for_a_busy_laya(loader, monkeypatch):
    monkeypatch.setattr(zeroshot, "IDLE_S", 0.05)
    monkeypatch.setattr(zeroshot, "BUSY_RETRY_S", 0.05)
    zeroshot.ask("s", Q, wait_s=2.0)
    agent = loader.opened[0]
    with neo.reflex.laya_lock():
        time.sleep(0.3)  # idle, but Laya is busy: it must not free the model mid-pass
        assert not agent.closed and zeroshot._agent is agent
    assert wait_until(lambda: agent.closed)


def test_laya_off_or_a_failed_load_answers_none_without_retrying_at_once(loader, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_REFLEX", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert zeroshot.ask("s", Q, wait_s=0.5) is None and not loader.locked_while_opening
    monkeypatch.setenv("NEO_REFLEX", "laya")
    monkeypatch.setattr(config, "_settings", None)
    loader.fail = True
    assert zeroshot.ask("s", Q, wait_s=1.0) is None
    assert zeroshot.ask("s", Q, wait_s=1.0) is None
    assert len(loader.locked_while_opening) == 1  # tried once; not again within RETRY_S
    assert not neo.reflex.laya_lock().locked()


def test_the_checkpoint_is_fetched_at_the_pinned_revision_with_only_its_own_files(monkeypatch):
    import huggingface_hub

    from neo import models

    asked = {}

    def snapshot_download(repo, revision=None, allow_patterns=None, **kw):
        asked.update(repo=repo, revision=revision, patterns=allow_patterns)
        return "/cache/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert zeroshot.checkpoint_dir() == "/cache/snap/typed-decisions"
    assert asked["repo"] == models.LAYA and asked["revision"] == models.LAYA_REVISION
    assert asked["patterns"] and all(p.startswith("typed-decisions/") for p in asked["patterns"])
    assert "typed-decisions/model.safetensors" in asked["patterns"]
