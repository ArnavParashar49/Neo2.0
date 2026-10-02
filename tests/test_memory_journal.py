"""The journal: every line is on disk moments after it is said, survives a kill, is redacted,
never comes back after an erase — and conversations get their summaries from it."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from neo2.core import config, effects, privacy, runtime
from neo2.core.memory import extract, journal, summarize
from neo2.core.memory import store as store_mod
from neo2.core.memory.embed import embed_texts as _REAL_EMBED_TEXTS  # before conftest swaps it out
from neo2.core.memory.redact import MASK
from neo2.core.memory.store import Store, store
from neo2.core.providers.base import Turn
from tests.neo_lib import make_runtime, scripted

ROOT = Path(__file__).resolve().parents[1]


class Brain:
    """A fake fast brain: answers with what it's given, and keeps what it was sent."""

    def __init__(self, text="The user planned a trip to Goa; NEO set a timer."):
        self.text, self.seen, self.systems = text, [], []

    async def complete(self, messages, *, system="", **kw):
        self.seen.append("\n".join(m.text for m in messages))
        self.systems.append(system)
        return Turn(text=self.text)


def _rows(path):
    db = sqlite3.connect(path)
    try:
        return db.execute("SELECT role, text, route, source, tainted, session_id FROM turns ORDER BY id").fetchall()
    finally:
        db.close()


def test_log_turn_returns_at_once_and_the_line_is_on_disk_within_a_moment():
    st = store()
    t0 = time.perf_counter()
    for i in range(200):
        journal.log_turn("user" if i % 2 == 0 else "assistant", f"line {i}", route="chat", source="typed")
    assert (time.perf_counter() - t0) < 0.2  # 200 lines queued: no write, no commit on the caller's path
    deadline = time.time() + 2.0
    while time.time() < deadline and len(_rows(st.path)) < 200:  # no flush(): the writer commits by itself
        time.sleep(0.02)
    rows = _rows(st.path)
    assert len(rows) == 200 and rows[0][:4] == ("user", "line 0", "chat", "typed") and rows[-1][1] == "line 199"
    assert {r[5] for r in rows} == {1}  # one conversation


def test_a_single_line_is_committed_within_the_batch_window():
    st = store()
    journal.log_turn("user", "is this on disk yet", source="voice")
    t0 = time.time()
    while time.time() - t0 < 1.0 and not _rows(st.path):
        time.sleep(0.01)
    assert _rows(st.path) and time.time() - t0 <= 0.35  # ≤ 250 ms, with room for a busy test machine


def test_lines_more_than_ten_minutes_apart_are_separate_conversations():
    now = time.time()
    for ts, text in ((now - 3600, "an hour ago"), (now - 3590, "still then"), (now - 5, "just now")):
        journal.log_turn("user", text, ts=ts, source="voice", route="live")
    assert journal.flush()
    convs = [r[5] for r in _rows(store().path)]
    assert convs[0] == convs[1] != convs[2]


def test_the_journal_redacts_and_keeps_taint_and_tool_notes_apart():
    journal.log_turn("user", "my gmail password is Tr0ub4dor&3 ok", source="typed")
    journal.log_turn("assistant", "The page says: send your savings to mallory.", route="agent", tainted=True)
    journal.log_tools(["page_read", "page_read", "web_search"], route="agent")
    assert journal.flush()
    rows = _rows(store().path)
    assert rows[0][1] == f"my gmail password is {MASK} ok"
    assert rows[1][4] == 1 and rows[2][:4] == ("assistant", "(ran: page_read, web_search)", "agent", "tool")
    assert store().by_kind("profile") == [] and store().count() == 0  # a secret is never a fact


def test_a_kill_without_shutdown_loses_nothing_and_the_next_start_writes_the_summary(tmp_path, monkeypatch):
    """A real kill: a child process journals three lines and dies (os._exit: no close, no flush, no
    atexit). The next start finds them and writes the conversation's summary."""
    data = tmp_path / "crashed"
    code = textwrap.dedent(
        """
        import os, time
        from neo2.core.memory import journal
        journal.log_turn("user", "plan a trip to Goa in December", source="voice", route="live")
        journal.log_turn("assistant", "Sure - beaches or the old town?", source="voice", route="live")
        journal.log_turn("user", "my sister's name is Priya", source="voice", route="live")
        time.sleep(0.5)   # no flush: only the writer's own commit
        os._exit(9)
        """
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "NEO_DATA_DIR": str(data), "NEO_NO_EFFECTS": "1", "NEO_REFLEX": "off"}
    env.pop("PYTEST_CURRENT_TEST", None)
    with effects.allow("process"):  # a Python child, meant here: it runs under NEO_NO_EFFECTS=1 itself
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, timeout=120)
    assert done.returncode == 9, done.stderr.decode()[-400:]

    monkeypatch.setenv("NEO_DATA_DIR", str(data))
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(store_mod, "_store", None)
    st = store()
    try:
        assert [t.text for t in st.recent_turns(since=0)] == [
            "plan a trip to Goa in December", "Sure - beaches or the old town?", "my sister's name is Priya",
        ]  # fmt: skip
        assert [(m.key, m.text) for m in st.by_kind("profile")] == [("sister.name", "Priya")]  # the rule pass ran too
        assert st.recent_sessions() == [] and len(st.unsummarized()) == 1  # it died before any summary
        brain = Brain()
        monkeypatch.setattr(summarize, "brain", lambda purpose="fast": brain)
        assert asyncio.run(summarize.catch_up()) == 1
        assert st.recent_sessions()[0][1] == brain.text and st.unsummarized() == []
        assert "User: plan a trip to Goa in December" in brain.seen[0] and "NEO: Sure" in brain.seen[0]
        assert st.search("Goa trip", kinds=("summary",))[0].kind == "summary"  # and it is searchable
        assert asyncio.run(summarize.catch_up()) == 0  # once
    finally:
        st.close()
        monkeypatch.setattr(store_mod, "_store", None)


def test_an_erase_drops_what_is_still_queued(monkeypatch):
    st = store()
    journal.log_turn("user", "said before the erase, already written")
    assert journal.flush() and st.turn_count() == 1
    with st._lock:  # the writer can't write: these lines are still on their way
        journal.log_turn("user", "said before the erase, still queued")
        journal.log_turn("user", "my sister's name is Priya")
        time.sleep(0.3)  # the writer has taken the batch and waits for the database
        monkeypatch.setattr(privacy, "_generation", privacy._generation + 1)  # "Erase NEO's Memory"
        st.forget_all()
    assert journal.flush()
    assert st.turn_count() == 0 and st.count() == 0  # neither the lines nor the fact in them came back
    journal.log_turn("user", "said after the erase")
    assert journal.flush() and [t.text for t in st.recent_turns(since=0)] == ["said after the erase"]


def test_a_request_older_than_the_erase_writes_nothing(monkeypatch):
    gen = privacy.generation()
    monkeypatch.setattr(privacy, "_generation", gen + 1)
    journal.log_turn("assistant", "an answer that finished after the erase", generation=gen)
    assert journal.flush() and store().turn_count() == 0


def test_with_the_journal_off_nothing_is_written(monkeypatch):
    monkeypatch.setenv("NEO_MEMORY_JOURNAL", "false")
    monkeypatch.setattr(config, "_settings", None)
    assert not journal.enabled() and journal.journal() is None
    journal.log_turn("user", "this is not kept")
    assert journal.flush() and store().turn_count() == 0 and store()._journal is None  # not even a writer thread


def test_the_writer_stops_with_its_store(tmp_path):
    st = Store(tmp_path / "m.sqlite3")
    j = journal.journal_of(st)
    j.log("user", "hello there")
    assert j.flush() and j._thread is not None and j._thread.is_alive()
    t = j._thread
    st.close()
    t.join(2.0)
    assert not t.is_alive()
    j.log("user", "after close")  # dropped, never an error


# ---- library mode -----------------------------------------------------------------------------------


async def test_an_embedded_agent_stores_nothing_unless_its_host_turns_it_on(tmp_path, monkeypatch):
    from neo2.core.agent.session import Session

    started: list[str] = []
    real_start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", lambda self: (started.append(self.name), real_start(self))[1])

    async def ask(rt, text):
        sc = rt.scope()
        with runtime.bind(sc):
            sc.session = sc.session or Session()
            reply = await sc.session.handle(text, origin="user")
            journal.flush()
            return reply

    # memory on, the journal not asked for: the host's default
    quiet = make_runtime(tmp_path / "quiet", "quiet", ["clock"], memory_enabled=True)
    scripted(quiet, ["Hello!"])
    assert (await ask(quiet, "my sister's name is Priya, say hello")).text == "Hello!"
    with runtime.bind(quiet.scope()):
        assert not journal.enabled() and store().turn_count() == 0 and store().count() == 0
    assert "neo-journal" not in started
    # no memory at all: no database file either
    bare = make_runtime(tmp_path / "bare", "bare", ["clock"])
    scripted(bare, ["Hi."])
    await ask(bare, "my sister's name is Priya")
    assert not (tmp_path / "bare" / "memory.sqlite3").exists()
    # the host turned the journal on (and the rules): now it is kept — in the agent's own folder
    keen = make_runtime(tmp_path / "keen", "keen", ["clock"], memory_enabled=True, memory_journal=True, memory_extract="rules")
    scripted(keen, ["Nice to meet Priya."])
    await ask(keen, "my sister's name is Priya")
    with runtime.bind(keen.scope()):
        assert [(t.role, t.text) for t in store().recent_turns(since=0)] == [
            ("user", "my sister's name is Priya"), ("assistant", "Nice to meet Priya."),
        ]  # fmt: skip
        assert [(m.key, m.text) for m in store().by_kind("profile")] == [("sister.name", "Priya")]
        assert Path(store().path).parent == tmp_path / "keen"
        store().close()
    for rt in (quiet, bare, keen):
        rt.close()


def test_library_settings_default_to_storing_nothing(tmp_path):
    from neo2.core.config import LibrarySettings, Settings

    lib = LibrarySettings(data_dir=tmp_path / "a")
    assert (lib.memory_enabled, lib.memory_journal, lib.memory_extract, lib.memory_live_inject) == (False, False, "off", False)
    app = Settings()
    assert (app.memory_journal, app.memory_retention_days, app.memory_extract, app.memory_recall_k) == (True, 365, "rules+llm", 6)
    assert app.memory_live_inject is False  # measured first: see neo2.core.voice.live


def test_an_sdk_host_turns_the_journal_on_in_its_memory_config(tmp_path):
    import neo2

    off = neo2.AgentConfig(tools="none", data_dir=tmp_path / "a", memory=True).library_settings(tmp_path / "a")
    assert (off.memory_enabled, off.memory_journal, off.memory_extract) == (True, False, "off")
    mem = neo2.MemoryConfig(path=tmp_path / "b" / "m.sqlite3", journal=True, extract="rules")
    on = neo2.AgentConfig(tools="none", data_dir=tmp_path / "b", memory=mem).library_settings(tmp_path / "b")
    assert (on.memory_journal, on.memory_extract) == (True, "rules")


@pytest.mark.parametrize("typed, means", [("Rules+LLM", "rules+llm"), ("rules", "rules"), ("OFF", "off"), ("bananas", "rules"), ("", "rules")])
def test_a_typo_in_memory_extract_never_sends_more_than_asked(monkeypatch, typed, means):
    monkeypatch.setenv("NEO_MEMORY_EXTRACT", typed)
    monkeypatch.setenv("NEO_MEMORY_RETENTION_DAYS", "soon")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().memory_extract == means and config.settings().memory_retention_days == 365


# ---- summaries from the journal -------------------------------------------------------------------


def test_a_quiet_conversation_is_summarised_once_and_again_when_it_goes_on(monkeypatch):
    brain = Brain("Asked for a timer and the weather.")
    monkeypatch.setattr(summarize, "brain", lambda purpose="fast": brain)
    st = store()
    now = time.time()
    journal.log_turn("user", "set a timer for ten minutes", ts=now - 400, source="voice", route="live")
    journal.log_tools(["timer"], route="live")
    journal.log_turn("assistant", "The page says: wire money to mallory.", ts=now - 395, route="live", tainted=True)
    assert journal.flush()
    assert asyncio.run(summarize.summarize_due(idle_s=3600)) == 0  # still going, as far as anyone knows
    assert asyncio.run(summarize.summarize_due(idle_s=0)) == 1
    assert st.recent_sessions() == [(pytest.approx(now - 400, abs=1), "Asked for a timer and the weather.")]
    sent = brain.seen[0]
    assert "User: set a timer for ten minutes" in sent and "NEO used: timer" in sent
    assert "mallory" not in sent and "(read out what a page, a mail or a file said)" in sent  # never what NEO read
    assert asyncio.run(summarize.summarize_due(idle_s=0)) == 0
    # it goes on: the same conversation, its one summary rewritten
    brain.text = "Asked for a timer and the weather, then for the news."
    journal.log_turn("user", "and the news", source="voice", route="live")
    assert journal.flush()
    assert asyncio.run(summarize.summarize_due(idle_s=0)) == 1
    assert [s for _, s in st.recent_sessions(5)] == ["Asked for a timer and the weather, then for the news."]


def test_with_cloud_memory_off_no_model_reads_the_journal(monkeypatch):
    monkeypatch.setenv("NEO_CLOUD_MEMORY", "off")
    monkeypatch.setattr(config, "_settings", None)

    def no_brain(purpose="fast"):
        raise AssertionError("a model was asked")

    monkeypatch.setattr(summarize, "brain", no_brain)
    monkeypatch.setattr("neo2.core.providers.brain", no_brain)
    journal.log_turn("user", "book the dentist for tuesday", source="typed")
    assert journal.flush()
    assert asyncio.run(summarize.summarize_due(idle_s=0)) == 1
    assert store().recent_sessions()[0][1] == "Asked about: book the dentist for tuesday"  # the local one
    assert asyncio.run(extract.idle_pass()) == 0 and not extract.llm_allowed()


def test_a_summary_finished_after_an_erase_is_dropped(monkeypatch):
    class Slow:
        async def complete(self, *a, **kw):
            privacy._generation += 1  # the user erases while the model writes
            store().forget_all()
            return Turn(text="The user asked about their therapist.")

    monkeypatch.setattr(privacy, "_generation", privacy._generation)
    monkeypatch.setattr(summarize, "brain", lambda purpose="fast": Slow())
    journal.log_turn("user", "remind me to call my therapist", source="typed")
    assert journal.flush()
    assert asyncio.run(summarize.summarize_due(idle_s=0)) == 0
    assert store().recent_sessions(5) == [] and store().turn_count() == 0


async def test_the_sessions_checkpoint_summarises_from_the_journal(monkeypatch):
    from neo2.core.agent.session import Session

    brain = Brain("Chatted about tea.")
    monkeypatch.setattr(summarize, "brain", lambda purpose="fast": brain)
    journal.log_turn("user", "I like oolong tea", source="typed")
    s = Session()
    assert await s.checkpoint() == ""  # the conversation isn't over (nothing quiet for three minutes)
    assert await s.checkpoint(final=True) == "1 conversation"  # shutdown: every open one
    assert store().recent_sessions()[0][1] == "Chatted about tea."


# ---- NEO.app's background loop, the erase, the embedder ---------------------------------------------


async def test_the_memory_loop_catches_up_summarises_extracts_and_prunes_only_while_idle(monkeypatch):
    from neo2.core import app as app_mod

    st = store()
    now = time.time()
    # what an earlier run left behind: a conversation with no summary, and a line past its keep-by date
    st.add_turns([(now - 86400 * 400, "user", "a line from long ago about tulips", "live", "voice", False)])
    st.add_turns([(now - 7200, "user", "we moved to Pune last year", "live", "voice", False)])
    brain = Brain("Said they moved to Pune.")
    facts = Brain('[{"key": "city", "value": "Pune", "kind": "profile", "importance": 0.8}]')
    monkeypatch.setattr(summarize, "brain", lambda purpose="fast": brain)
    monkeypatch.setattr("neo2.core.providers.brain", lambda purpose="fast": facts)
    monkeypatch.setattr(extract, "_retry_at", 0.0)
    idle = {"now": False}
    task = asyncio.create_task(app_mod._memory_loop(NS(session=None), lambda: idle["now"], tick_s=0.05))
    await asyncio.sleep(2.3)
    assert st.recent_sessions() == [] and st.count() == 0 and st.turn_count() == 2  # busy: nothing runs
    idle["now"] = True
    for _ in range(100):
        await asyncio.sleep(0.05)
        if st.count() and st.recent_sessions():
            break
    task.cancel()
    assert st.turn_count() == 1  # the old line is gone…
    assert [s for _, s in st.recent_sessions()] == ["Said they moved to Pune."]  # …the crashed conversation has its summary
    assert [(m.key, m.text) for m in st.by_kind("profile")] == [("city", "Pune")]  # …and the idle pass read it for facts


async def test_erase_everything_clears_the_journal_and_what_is_on_its_way():
    st = store()
    st.remember("my therapist is Dr. Mehta", "note")
    journal.log_turn("user", "call my therapist tomorrow", source="voice", route="live")
    assert journal.flush()
    st.set_summary(st.recent_turns()[0].conv, "Asked to call the therapist.", 1)
    journal.log_turn("user", "my sister's name is Priya", source="voice", route="live")  # still queued when the erase comes
    counts = await privacy.forget_everything()
    assert counts["memory"] >= 1 and counts["turns"] >= 1 and counts["sessions"] == 1
    assert journal.flush()
    assert st.turn_count() == 0 and st.count() == 0 and st.recent_sessions() == [] and st.context("therapist") == ""
    assert "therapist" not in st.path.read_bytes().decode("latin-1")


def test_the_reflex_line_no_longer_calls_its_correction_cache_memory():
    from neo2.core.reflex import learn

    learn.count("requests")
    learn.count("memory_hits")
    learn.flush_counts()
    line = learn.summary()
    assert "correction hits 1" in line and "memory hits" not in line


class _FakeST:
    made: list[dict] = []

    def __init__(self, name, **kw):
        type(self).made.append({"name": name, **kw})
        if kw.get("local_files_only") and getattr(type(self), "broken_cache", False):
            raise OSError("half a cache")

    def encode(self, texts, **kw):
        import numpy as np

        return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def embedder(monkeypatch, tmp_path):
    """The real embed module (not conftest's stand-in), with a fake SentenceTransformer."""
    from neo2.core import models
    from neo2.core.memory import embed

    monkeypatch.setattr(embed, "embed_texts", _REAL_EMBED_TEXTS)
    for name, val in (("_model", None), ("_failed", False), ("_failed_at", 0.0), ("_loading", False)):
        monkeypatch.setattr(embed, name, val)
    monkeypatch.setattr(embed, "ready_event", threading.Event())
    _FakeST.made, _FakeST.broken_cache = [], False
    monkeypatch.setitem(sys.modules, "sentence_transformers", NS(SentenceTransformer=_FakeST))
    monkeypatch.setenv("NEO_MODELS_PATH", str(tmp_path / "models"))
    monkeypatch.setattr(config, "_settings", None)
    snap = tmp_path / "models" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / models.EMBEDDER_REVISION
    return embed, snap


def test_embed_texts_never_loads_the_model_on_the_callers_path(embedder, monkeypatch):
    embed, _snap = embedder
    release = threading.Event()
    loads = []

    def slow_load(cache):
        loads.append(threading.current_thread().name)
        release.wait(5)
        embed._model, embed._loading = _FakeST("x"), False

    monkeypatch.setattr(embed, "_load", slow_load)
    t0 = time.perf_counter()
    assert embed.embed_texts(["hello"]) is None and embed.embed_texts(["again"]) is None  # no model yet: keywords
    assert time.perf_counter() - t0 < 0.2 and not embed.ready()
    time.sleep(0.05)
    assert loads == ["neo-embedder"]  # one load, started in the background, not in this thread
    release.set()
    for _ in range(100):
        if embed.ready():
            break
        time.sleep(0.01)
    assert embed.embed_texts(["hello"]).shape == (1, 384)


def test_the_cached_model_is_loaded_without_asking_the_hub(embedder):
    embed, snap = embedder
    assert embed._get() is not None and "local_files_only" not in _FakeST.made[0]  # first run: it may download
    embed._model = None
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    _FakeST.made.clear()
    assert embed._get() is not None and _FakeST.made[0]["local_files_only"] is True and embed.ready_event.is_set()
    # a half-written cache: fetch what's missing rather than run without the model
    embed._model, _FakeST.broken_cache = None, True
    _FakeST.made.clear()
    assert embed._get() is not None and [m.get("local_files_only") for m in _FakeST.made] == [True, None]


def test_a_failed_load_is_tried_again_later_and_a_held_load_is_nobody_elses(embedder, monkeypatch):
    embed, _snap = embedder

    class Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("no torch today")

    monkeypatch.setitem(sys.modules, "sentence_transformers", NS(SentenceTransformer=Boom))
    assert embed._get() is None and embed._failed
    monkeypatch.setitem(sys.modules, "sentence_transformers", NS(SentenceTransformer=_FakeST))
    assert embed._get() is None  # not hammered: nothing for a while after a failure
    monkeypatch.setattr(embed, "_failed_at", time.time() - embed._RETRY_S - 1)
    cache = embed.hold()  # the reflex's warm-up takes the load (it runs after Laya's)
    assert cache is not None
    embed.warmup()
    assert embed.embed_texts(["x"]) is None and _FakeST.made == []  # a lazy warm-up doesn't start a second one
    embed.load_held(cache)
    assert embed.ready() and embed.embed_texts(["x"]) is not None
