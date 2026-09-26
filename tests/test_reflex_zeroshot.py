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

    def find():  # on this Mac already, its weights read
        return "/nowhere/typed-decisions"

    def open_(path):
        state.locked_while_opening.append(neo.reflex.laya_lock().locked())
        state.gate.wait(5)
        if state.fail:
            raise RuntimeError("no torch here")
        agent = FakeAgent()
        state.opened.append(agent)
        return agent

    monkeypatch.setattr(zeroshot, "_find", find)
    monkeypatch.setattr(zeroshot, "_open", open_)
    return state


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(zeroshot, "_agent", None)
    monkeypatch.setattr(zeroshot, "_loading", None)
    monkeypatch.setattr(zeroshot, "_fetching", None)
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


def test_a_slow_load_answers_none_within_the_wait_and_drops_the_question(loader):
    loader.gate.clear()
    t = time.perf_counter()
    assert zeroshot.ask("s", Q, wait_s=0.1) is None  # the caller's prior decides
    assert time.perf_counter() - t < 1.0
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


def test_a_checkpoint_not_on_this_mac_downloads_in_the_background_and_nobody_waits_for_it(loader, monkeypatch):
    here, gate, downloads = threading.Event(), threading.Event(), []
    monkeypatch.setattr(zeroshot, "_find", lambda: "/nowhere/typed-decisions" if here.is_set() else None)

    def hub(download=True):  # ~0.8 GB to download on a fresh Mac
        if not download:
            raise FileNotFoundError("not in the cache")
        downloads.append(download)
        gate.wait(5)
        here.set()
        return "/nowhere/typed-decisions"

    monkeypatch.setattr(zeroshot, "checkpoint_dir", hub)
    for _ in range(3):  # the first question starts the download; none waits its wait_s for it
        t = time.perf_counter()
        assert zeroshot.ask("s", Q, wait_s=1.0) is None
        assert time.perf_counter() - t < 0.5
    assert downloads == [True] and not loader.opened
    assert not zeroshot.available()  # neo/decide doesn't even shortlist for it meanwhile
    gate.set()
    assert wait_until(lambda: zeroshot._fetching is None)
    assert zeroshot.available()
    assert zeroshot.ask("s", Q, wait_s=2.0) is not None  # here now: loaded and answered
    assert downloads == [True]


def test_only_a_complete_checkpoint_on_this_mac_is_found_and_looking_never_downloads(tmp_path, monkeypatch):
    folder = tmp_path / "typed-decisions"

    def cached(download=True):
        assert download is False
        if not tmp_path.joinpath("cached").exists():
            raise FileNotFoundError("nothing of this revision in the cache")
        return str(folder)

    monkeypatch.setattr(zeroshot, "checkpoint_dir", cached)
    assert zeroshot._find() is None
    tmp_path.joinpath("cached").touch()  # the revision's folder: the reflex's checkpoint shares it
    assert zeroshot._find() is None
    folder.mkdir()
    (folder / "rl_agent_config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"\0" * 1024)
    assert zeroshot._find() == str(folder)


def test_prefetch_downloads_without_loading_and_a_failed_download_waits_before_retrying(loader, monkeypatch):
    from neo import config

    got, offline = [], []

    def hub(download=True):
        got.append(download)
        if not download:
            raise FileNotFoundError("not in the cache")
        if offline:
            raise OSError("offline")
        return "/x"

    monkeypatch.setattr(zeroshot, "checkpoint_dir", hub)
    zeroshot.prefetch()
    assert wait_until(lambda: zeroshot._fetching is None) and got == [False, True]
    assert not loader.opened and zeroshot._agent is None
    got.clear()
    offline.append(True)
    zeroshot.prefetch()
    assert wait_until(lambda: zeroshot._fetching is None) and got == [False, True]
    zeroshot.prefetch()
    assert zeroshot.ask("s", Q, wait_s=0.5) is None
    time.sleep(0.05)
    assert got == [False, True]  # not again within RETRY_S (each try may pull ~0.8 GB)
    monkeypatch.setattr(zeroshot, "_failed_at", float("-inf"))
    monkeypatch.setenv("NEO_REFLEX", "off")
    monkeypatch.setattr(config, "_settings", None)
    zeroshot.prefetch()
    assert zeroshot._fetching is None and got == [False, True]  # Laya off: nothing to fetch


def test_prefetch_of_a_checkpoint_already_here_never_waits_for_the_reflex(loader, monkeypatch):
    got = []
    monkeypatch.setattr(zeroshot, "_local", lambda: "/nowhere/typed-decisions")
    monkeypatch.setattr(zeroshot, "checkpoint_dir", lambda download=True: got.append(download))
    with neo.reflex.laya_lock():  # the reflex is still loading at NEO's start
        zeroshot.prefetch()
        assert wait_until(lambda: zeroshot._fetching is None) and not got
        assert zeroshot.available()


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
