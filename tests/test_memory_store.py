"""The memory store: the upgrade of an older database, facts that supersede, redaction, the hybrid
search (keywords, meaning, Devanagari, time phrases, recency, a minimum score), its speed on a
20,000-line journal, and the erase."""

from __future__ import annotations

import hashlib
import random
import sqlite3
import statistics
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from neo2.core.memory import embed as emb
from neo2.core.memory import redact, timephrase
from neo2.core.memory.store import SCHEMA_VERSION, Store, snippet, tokens

# The schema NEO shipped before the journal (user_version 2): what a user's database looks like today.
_V2_SCHEMA = """
CREATE TABLE memory (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, key TEXT, text TEXT NOT NULL,
  created REAL NOT NULL, updated REAL NOT NULL, uses INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE memory_fts USING fts5(text, key, content='memory', content_rowid='id');
CREATE TRIGGER mem_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TRIGGER mem_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
END;
CREATE TRIGGER mem_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TABLE memory_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE TABLE playbook (
  id INTEGER PRIMARY KEY, goal TEXT NOT NULL, steps TEXT NOT NULL, answer TEXT, created REAL NOT NULL, uses INTEGER DEFAULT 0
);
CREATE TABLE playbook_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE TABLE sessions (id INTEGER PRIMARY KEY, started REAL, ended REAL, summary TEXT);
"""


def fake_embed(texts):
    """Deterministic bag-of-words vectors: shared words → a high cosine."""
    out = np.zeros((len(texts), emb.DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokens(t) or t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % emb.DIM] += 1.0
        out[i] /= np.linalg.norm(out[i]) or 1.0
    return out


@pytest.fixture
def st(tmp_path):
    s = Store(tmp_path / "m.sqlite3")
    yield s
    s.close()


@pytest.fixture
def vst(tmp_path, monkeypatch):
    """A store with (fake) embeddings."""
    monkeypatch.setattr(emb, "embed_texts", fake_embed)
    s = Store(tmp_path / "m.sqlite3")
    yield s
    s.close()


def _v2_database(path, now):
    db = sqlite3.connect(path)
    db.executescript(_V2_SCHEMA)
    rows = [
        ("profile", "city", "Dubai", now - 86400 * 40),
        ("profile", "name", "Arnav", now - 86400 * 40),
        ("lesson", "", "Reply in Hinglish when I speak Hinglish", now - 86400 * 9),
        ("note", "", "the gym locker is number 214", now - 86400 * 3),
    ]
    for kind, key, text, t in rows:
        cur = db.execute("INSERT INTO memory(kind,key,text,created,updated,uses) VALUES (?,?,?,?,?,1)", (kind, key, text, t, t))
        db.execute("INSERT INTO memory_vec(id, vec) VALUES (?,?)", (cur.lastrowid, fake_embed([text])[0].tobytes()))
    db.execute("INSERT INTO playbook(goal, steps, answer, created) VALUES ('email my boss', '[{\"tool\": \"mail_send\", \"args\": {}}]', '', ?)", (now,))
    for i in range(30):
        db.execute(
            "INSERT INTO sessions(started, ended, summary) VALUES (?,?,?)",
            (now - 86400 * (i + 1), now - 86400 * (i + 1) + 600, f"Day {i}: asked about the Lisbon trip and set a timer."),
        )
    db.execute("PRAGMA user_version=2")
    db.commit()
    db.close()


# ---- migration ------------------------------------------------------------------------------------


def test_an_older_database_is_copied_then_upgraded_and_keeps_every_row(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "embed_texts", fake_embed)
    path, now = tmp_path / "memory.sqlite3", time.time()
    _v2_database(path, now)
    st = Store(path)
    try:
        bak = tmp_path / "memory.sqlite3.bak-v2"
        assert bak.exists() and (bak.stat().st_mode & 0o777) == 0o600
        old = sqlite3.connect(bak)  # the copy is the database as it was: old version, old columns, every row
        assert old.execute("PRAGMA user_version").fetchone()[0] == 2
        assert old.execute("SELECT count(*) FROM memory").fetchone()[0] == 4
        assert "source" not in {r[1] for r in old.execute("PRAGMA table_info(memory)")}
        old.close()
        db = sqlite3.connect(path)
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
        cols = {r[1] for r in db.execute("PRAGMA table_info(memory)")}
        assert {"source", "importance", "confidence", "superseded_by", "last_seen"} <= cols
        assert db.execute("SELECT count(*), sum(uses) FROM memory").fetchone() == (4, 4)
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 30  # all thirty: no 90-day cut
        assert db.execute("SELECT count(*) FROM playbook").fetchone()[0] == 1
        assert {r[0] for r in db.execute("SELECT DISTINCT model FROM memory_vec")} == {emb.MODEL_ID}
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"turns", "conversations", "meta", "sessions_vec", "turns_fts", "sessions_fts"} <= tables
        db.close()
        # the old rows are found by the new search — facts, and the summaries that weren't searchable before
        assert [m.text for m in st.by_kind("profile")] and {m.key for m in st.by_kind("profile")} == {"city", "name"}
        assert st.recall("gym locker")[0].text == "the gym locker is number 214"
        assert st.search("Lisbon trip", kinds=("summary",))[0].kind == "summary"
        assert len(st._vec["memory_vec"]) == 4  # the old vectors are in RAM, not made again
    finally:
        st.close()
    # opened again: nothing to upgrade, and the copy is left alone
    before = bak.stat().st_mtime
    st = Store(path)
    st.close()
    assert bak.stat().st_mtime == before and not (tmp_path / "memory.sqlite3.bak-v3").exists()


def test_a_new_database_has_no_backup_and_the_current_schema(tmp_path):
    st = Store(tmp_path / "m.sqlite3")
    st.close()
    assert list(tmp_path.glob("*.bak-*")) == []
    assert sqlite3.connect(tmp_path / "m.sqlite3").execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_an_agents_database_with_the_new_tables_is_still_its_own(tmp_path):
    from neo2.core.config import agent_memory_db

    st = Store(tmp_path / "agent.sqlite3", agent=True)
    st.remember("likes tea", "profile", "pref.tea")
    st.add_turns([(time.time(), "user", "hello there", "chat", "typed", False)])
    st.close()
    assert agent_memory_db(tmp_path / "agent.sqlite3")


# ---- facts ----------------------------------------------------------------------------------------


def test_a_newer_value_supersedes_the_old_one_and_only_one_is_live(st):
    a = st.remember("Dubai", "profile", "city", source="t1")
    assert st.remember("dubai", "profile", "city", source="t7") == a  # told again: the same fact, seen again
    b = st.remember("Abu Dhabi", "profile", "city", source="t9")
    assert b != a
    live = st.by_kind("profile")
    assert [(m.id, m.text) for m in live] == [(b, "Abu Dhabi")]
    hist = st.history_of("profile", "city")
    assert [m.text for m in hist] == ["Abu Dhabi", "Dubai"] and hist[1].source == "t1,t7"
    assert [h.text for h in st.search("city")] == ["Abu Dhabi"]  # the old value is never recalled
    assert "Dubai" not in st.context("where do I live").replace("Abu Dhabi", "")
    assert st.forget(b) and st.by_kind("profile") == [] and st.history_of("profile", "city") == []  # never back to the old one


def test_the_same_fact_under_another_spelling_of_its_key_is_one_fact(st):
    a = st.remember("Priya", "profile", "sister.name")
    assert st.remember("Priya", "profile", "sister_name") == a
    assert len(st.by_kind("profile")) == 1


def test_a_near_duplicate_note_updates_instead_of_adding(vst):
    a = vst.remember("I park the grey car on level three of the blue garage", "note")
    assert vst.remember("i park the grey car on level three of the blue garage", "note") == a  # the same words
    # nearly the same: one fact, and the newer wording stays
    assert vst.remember("I park the grey car on level three of the blue garage downtown", "note") == a
    assert vst.count() == 1 and vst.recall("park car garage")[0].text.endswith("downtown")
    assert vst.remember("my dentist is Dr. Rao", "note") != a


def test_a_secret_is_never_stored_in_a_fact_or_a_journal_line(st):
    mid = st.remember("my bank password is Tr0ub4dor&3", "note")
    st.add_turns([(time.time(), "user", "the wifi pin is 48291 and my card is 4111 1111 1111 1111", "chat", "typed", False)])
    raw = (st.path.read_bytes() + st.path.with_name(st.path.name + "-wal").read_bytes()).decode("latin-1")
    assert "Tr0ub4dor" not in raw and "48291" not in raw and "4111 1111" not in raw
    assert st.facts([mid])[0].text == f"my bank password is {redact.MASK}"
    assert redact.MASK in st.recent_turns()[0].text


@pytest.mark.parametrize(
    "text",
    [
        "my password is hunter2",
        "mera password hunter2 hai",
        "the OTP is 482913",
        "pin 4321",
        "api key: sk-proj-abcdefghijklmnop1234567890",
        "use AIzaSyA1234567890abcdefghijklmnopqrstuv for maps",
        "card 4111 1111 1111 1111, cvv 123",
        "token eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4",
        "my locker code is 4471",
        "Authorization: Bearer abcdef1234567890abcdef",
    ],
)
def test_realistic_secrets_are_masked(text):
    clean, n = redact.redact(text)
    assert n >= 1 and redact.MASK in clean and redact.has_secret(text) and not redact.has_secret(clean)


@pytest.mark.parametrize(
    "text",
    [
        "the wifi password is on the fridge",
        "what's my wifi password?",
        "my phone number is 9876543210",
        "my flight is on 2026-10-12",
        "my pin code is 560001",
        "pin the tab",
        "I forgot my password",
        "my sister's name is Priya",
        "flat 1204, tower B",
    ],
)
def test_what_is_not_a_secret_is_kept_as_said(text):
    assert redact.redact(text) == (text, 0)


# ---- search ---------------------------------------------------------------------------------------


def _day(n: float, hour: int = 15) -> float:
    """`n` days ago, at `hour` o'clock."""
    return (datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0) - timedelta(days=n)).timestamp()


def test_keyword_search_finds_facts_summaries_and_lines_and_ranks_facts_first(st):
    st.remember("Lisbon", "profile", "honeymoon.city")
    st.add_turns([(_day(3), "user", "book the hotel in lisbon for march", "live", "voice", False)])
    conv = st.recent_turns(limit=5)[0].conv
    st.set_summary(conv, "Planned the Lisbon trip and compared two hotels.", 1)
    hits = st.search("lisbon")
    assert [h.kind for h in hits] == ["profile", "summary", "turn"]  # a fact over a summary over a line
    assert [h.ref[0] for h in hits] == ["f", "s", "t"] and all(h.score >= 0.36 for h in hits)
    assert snippet(hits[0]).startswith("[fact, ") and "honeymoon city: Lisbon" in snippet(hits[0])
    assert [h.kind for h in st.search("lisbon", kinds=("turn",))] == ["turn"]
    assert [h.kind for h in st.search("lisbon", kinds=("fact",))] == ["profile"]


def test_semantic_search_finds_what_shares_no_exact_keyword_form(vst):
    vst.remember("the dentist appointment is on Tuesday", "note")
    vst.remember("favourite editor is Zed", "profile", "editor")
    # keyword: nothing ("appointments" ≠ "appointment" for an exact token) — the vector side finds it
    hits = vst.search("dentist appointments timing")
    assert hits and "dentist" in hits[0].text
    assert vst.search("quantum chromodynamics lecture") == []  # nothing related: nothing returned


def test_stop_words_and_junk_find_nothing(st):
    st.remember("the gym locker is number 214", "note")
    st.add_turns([(time.time(), "user", "what is the weather", "chat", "typed", False)])
    assert st.search("???") == [] and st.search("") == [] and st.search("what is the") == []


def test_devanagari_and_hinglish_are_searchable(st):
    st.remember("मुझे अदरक वाली चाय पसंद है", "note")
    st.add_turns(
        [
            (_day(2), "user", "कल ऑफिस में मीटिंग है", "live", "voice", False),
            (_day(2), "user", "mujhe adrakwali chai pasand hai", "live", "voice", False),
        ]
    )
    assert st.search("चाय")[0].text == "मुझे अदरक वाली चाय पसंद है"
    assert st.search("मीटिंग")[0].text == "कल ऑफिस में मीटिंग है"
    assert st.search("ऑफिस की मीटिंग", kinds=("turn",))[0].text == "कल ऑफिस में मीटिंग है"
    assert st.search("adrak")[0].text.startswith("mujhe adrakwali")  # a substring inside a longer word
    assert tokens("मुझे चाय पसंद है") == ["मुझे", "चाय", "पसंद", "है"]  # vowel signs stay in their word


def test_substring_search_works_without_the_trigram_tokenizer(st):
    st.remember("मुझे अदरक वाली चाय पसंद है", "note")
    st.add_turns([(_day(1), "user", "mujhe adrakwali chai pasand hai", "live", "voice", False)])
    st.trigram = False  # an SQLite built without it: LIKE does the same job
    assert st.search("चाय") and st.search("adrak")[0].kind == "turn"


def test_time_phrases_limit_lines_and_summaries_to_that_window(st):
    st.add_turns(
        [
            (_day(8), "user", "remind me about the passport renewal", "live", "voice", False),
            (_day(1), "user", "the passport photos are ready", "live", "voice", False),
            (_day(1, 16), "assistant", "Good, I'll note the photos are ready.", "live", "voice", False),
            (_day(0, 9), "user", "what time is the passport office open", "live", "voice", False),
        ]
    )
    assert [h.text for h in st.search("passport yesterday", kinds=("turn",))] == ["the passport photos are ready"]
    assert len(st.search("passport", kinds=("turn",))) == 3
    week = st.search("what did I say about the passport last week", kinds=("turn",))
    lw = timephrase.find("last week", datetime.now())
    assert all(lw.since <= h.ts < lw.until for h in week)
    # only a time: that window's lines, in the order they were said
    listing = st.search("what did we talk about yesterday")
    assert [h.text for h in listing] == ["the passport photos are ready", "Good, I'll note the photos are ready."]
    assert st.search("kal humne kya baat ki thi") == listing  # Hinglish: "kal" is yesterday here
    assert st.search("passport", since=_day(0, 0), until=time.time() + 1, kinds=("turn",))[0].text.startswith("what time")


def test_a_windows_summaries_come_before_its_lines(st):
    st.add_turns([(_day(1), "user", "plan the goa trip", "live", "voice", False), (_day(1), "assistant", "Sure.", "live", "voice", False)])
    st.set_summary(st.recent_turns(since=0, limit=1)[0].conv, "Planned a trip to Goa in December.", 2)
    hits = st.search("", since=_day(1, 0), until=_day(0, 0))
    assert hits[0].kind == "summary" and "Goa" in hits[0].text


@pytest.mark.parametrize(
    "phrase, days_ago",
    [("yesterday", 1), ("kal", 1), ("कल", 1), ("parso", 2), ("the day before yesterday", 2), ("3 days ago", 3), ("do din pehle", 2)],
)
def test_day_phrases(phrase, days_ago):
    now = datetime(2026, 10, 2, 15, 30)
    w = timephrase.find(f"what did I say {phrase}", now)
    day = datetime(2026, 10, 2) - timedelta(days=days_ago)
    assert w is not None and datetime.fromtimestamp(w.since) == day and datetime.fromtimestamp(w.until) == day + timedelta(days=1)
    assert w.rest == "what did I say"


def test_week_weekday_and_month_phrases():
    now = datetime(2026, 10, 2, 15, 30)  # a Friday
    for phrase in ("last week", "pichle hafte", "पिछले हफ्ते"):
        w = timephrase.find(phrase, now)
        assert (datetime.fromtimestamp(w.since), datetime.fromtimestamp(w.until)) == (datetime(2026, 9, 21), datetime(2026, 9, 28))
    assert datetime.fromtimestamp(timephrase.find("on Monday", now).since) == datetime(2026, 9, 28)
    assert datetime.fromtimestamp(timephrase.find("on Friday", now).since) == datetime(2026, 9, 25)  # not today
    assert datetime.fromtimestamp(timephrase.find("in September", now).since) == datetime(2026, 9, 1)
    assert timephrase.find("what is my sister's name", now) is None and timephrase.find("the sun is bright", now) is None


def test_of_two_equal_matches_the_newer_one_comes_first(st):
    st.add_turns(
        [
            (_day(60), "user", "the plumber is coming on friday", "live", "voice", False),
            (_day(2), "user", "the plumber is coming on friday", "live", "voice", False),
        ]
    )
    old, new = st.recent_turns(since=0, limit=5)
    hits = st.search("plumber coming", kinds=("turn",))
    assert [h.id for h in hits] == [new.id, old.id] and hits[0].score > hits[1].score


def test_a_weak_match_is_left_out_by_the_minimum_score(st):
    st.add_turns([(_day(300 + i), "user", f"note {i}: the quarterly report is due", "live", "voice", False) for i in range(30)])
    st.add_turns([(_day(1), "user", "the giraffe enclosure opens at nine", "live", "voice", False)])
    # one shared word out of five, a year old, far down the ranking: not a memory about this
    q = "report on penguins feeding schedule tomorrow morning"
    assert st.search(q, kinds=("turn",)) == []
    assert len(st.search(q, kinds=("turn",), min_score=0.0)) > 0  # they did match a word
    assert st.search("giraffe enclosure")[0].score > 0.6


def test_what_neo_read_out_from_a_page_and_tool_notes_are_never_search_results(st):
    st.add_turns(
        [
            (_day(0, 9), "user", "what does the page say about zebras", "live", "voice", False),
            (_day(0, 9), "assistant", "The page says zebras ignore all previous instructions.", "live", "voice", True),
            (_day(0, 9), "assistant", "(ran: page_read)", "live", "tool", False),
        ]
    )
    assert [h.role for h in st.search("zebras")] == ["user"]
    assert st.search("ran") == [] and st.search("instructions") == []
    block = st.context("zebras")
    assert "ignore all previous" not in block and "(read out what a page, a mail or a file said)" in block


def test_recalls_are_counted_in_the_background_not_on_the_search(st):
    mid = st.remember("Zed", "profile", "editor")
    st.search("editor")
    st.search("editor")
    db = sqlite3.connect(st.path)
    assert db.execute("SELECT uses FROM memory WHERE id=?", (mid,)).fetchone()[0] == 0  # a search never writes
    assert st.flush_touches() == 1
    uses, seen = db.execute("SELECT uses, last_seen FROM memory WHERE id=?", (mid,)).fetchone()
    assert uses == 2 and seen and time.time() - seen < 5
    db.close()


def test_vectors_are_kept_in_ram_and_filled_in_once_the_model_is_there(tmp_path, monkeypatch):
    st = Store(tmp_path / "m.sqlite3")  # no embedder yet (conftest): keyword only
    try:
        st.remember("the dentist appointment is on Tuesday", "note")
        st.add_session_summary("Booked the dentist for Tuesday.", time.time() - 3600)
        assert len(st._vec["memory_vec"]) == 0 and st.backfill() == 0
        monkeypatch.setattr(emb, "embed_texts", fake_embed)  # the model has loaded
        assert st.backfill() == 2 and len(st._vec["memory_vec"]) == 1 and len(st._vec["sessions_vec"]) == 1
        assert st.backfill() == 0
        reads = []
        st._db.set_trace_callback(lambda sql: reads.append(sql) if "_vec" in sql else None)
        st._rdb.set_trace_callback(lambda sql: reads.append(sql) if "_vec" in sql else None)
        assert st.search("dentist appointments timing")  # found by meaning…
        assert reads == []  # …without reading a vector back from SQLite
        mid = st.remember("the car is parked on level 3", "note")
        assert len(st._vec["memory_vec"]) == 2 and st.forget(mid) and len(st._vec["memory_vec"]) == 1
    finally:
        st.close()
    # vectors made by another model are ignored when loaded (and made again by the backfill)
    db = sqlite3.connect(tmp_path / "m.sqlite3")
    db.execute("UPDATE memory_vec SET model='some/other-model@1'")
    db.commit()
    db.close()
    st = Store(tmp_path / "m.sqlite3")
    try:
        assert len(st._vec["memory_vec"]) == 0 and st.backfill() == 1 and len(st._vec["memory_vec"]) == 1
    finally:
        st.close()


def test_search_is_fast_on_a_20k_line_journal(tmp_path, monkeypatch):
    """Warm, a search stays under 50 ms (typically under 20) on 20,000 journal lines, 500 facts
    and 300 summaries — without the embedding model's own time (here a hash), and never loading it."""
    monkeypatch.setattr(emb, "embed_texts", fake_embed)
    rng = random.Random(7)
    topics = ["trip", "hotel", "flight", "passport", "dentist", "plumber", "invoice", "recipe", "guitar", "budget"]
    vocab = [f"w{i:04d}" for i in range(3000)] + topics
    st = Store(tmp_path / "big.sqlite3")
    try:
        now = time.time()
        rows = []
        for i in range(20_000):
            words = " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 14)))
            rows.append((now - (20_000 - i) * 900, "user" if i % 2 == 0 else "assistant", words, "live", "voice", False))
        for i in range(0, len(rows), 2000):
            st.add_turns(rows[i : i + 2000])
        assert st.turn_count() == 20_000
        with st._lock:  # (straight in: 800 single-row commits would only slow the test down)
            for _ in range(500):
                text = " ".join(rng.choice(vocab) for _ in range(6))
                cur = st._db.execute("INSERT INTO memory(kind,key,text,created,updated) VALUES ('note','',?,?,?)", (text, now, now))
                st._put_vec("memory_vec", cur.lastrowid, fake_embed([text])[0])
            for i in range(300):
                text = " ".join(rng.choice(vocab) for _ in range(30))
                cur = st._db.execute("INSERT INTO sessions(started, ended, summary) VALUES (?,?,?)", (now - i * 86400, now, text))
                st._put_vec("sessions_vec", cur.lastrowid, fake_embed([text])[0])
            st._db.commit()
        queries = [
            "hotel trip", "what did I say about the passport last week", "dentist", "plumber invoice budget",
            "w0042 w1999 guitar", "recipe yesterday", "flight to lisbon in march", "w0007", "kal budget", "trip",
        ]  # fmt: skip
        for q in queries:  # warm: the page cache, the statements
            st.search(q)
        times = []
        for _ in range(6):
            for q in queries:
                t0 = time.perf_counter()
                hits = st.search(q)
                times.append((time.perf_counter() - t0) * 1000)
                assert len(hits) <= 8
        assert st.search("hotel trip") and st.search("w0007")
        typical, worst = statistics.median(times), max(times)
        print(f"\n[memory] search on 20k lines: median {typical:.2f} ms, p95 {sorted(times)[int(len(times) * 0.95)]:.2f} ms, max {worst:.2f} ms")
        assert typical < 20, f"typical search took {typical:.1f} ms"
        assert worst < 50, f"slowest search took {worst:.1f} ms"
        t0 = time.perf_counter()
        block = st.context("what did I say about the hotel")
        assert (time.perf_counter() - t0) * 1000 < 50 and len(block) <= 1800
    finally:
        st.close()


# ---- the prompt block -----------------------------------------------------------------------------


def test_the_memory_block_is_labelled_budgeted_and_has_todays_lines(st, monkeypatch):
    st.remember("Arnav", "profile", "name")
    st.remember("Reply in Hinglish when I speak Hinglish", "lesson")
    st.add_session_summary("Set up the Node project and emailed Priya the notes.", _day(2))
    now = time.time()
    st.add_turns([(now - 3000 + i, "user" if i % 2 == 0 else "assistant", f"line {i} about the garden " + "x" * 60, "live", "voice", False) for i in range(40)])
    block = st.context("garden")
    assert block.startswith("About the user: name: Arnav") and "Lessons: Reply in Hinglish" in block
    assert "Recent sessions: " in block and "Priya" in block
    assert "Earlier today" in block and "line 39 about the garden" in block and "line 5 about" not in block  # the latest
    assert block.rstrip().endswith("search your memory before saying you don't remember something.)")
    assert len(block) <= 1800
    assert len(st.context("garden", turns=20, budget=2600)) <= 2600
    # NEO_CLOUD_MEMORY limits what a model is sent
    from neo2.core import config

    monkeypatch.setenv("NEO_CLOUD_MEMORY", "profile")
    monkeypatch.setattr(config, "_settings", None)
    assert st.context("garden") == "About the user: name: Arnav"
    monkeypatch.setenv("NEO_CLOUD_MEMORY", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert st.context("garden") == ""


def test_a_sessions_own_lines_are_not_repeated_in_its_memory_block(st):
    now = time.time()
    st.add_turns(
        [
            (now - 900, "user", "said to the voice a while ago", "live", "voice", False),
            (now - 600, "user", "typed before this session began", "chat", "typed", False),
            (now - 60, "user", "typed in this very session", "chat", "typed", False),
            (now - 30, "user", "said to the voice just now", "live", "voice", False),
        ]
    )
    block = st.context("", not_after=now - 300)  # the session began five minutes ago
    assert "typed in this very session" not in block  # it is in the session's own history
    assert "said to the voice just now" in block and "typed before this session began" in block


# ---- erase ----------------------------------------------------------------------------------------


def test_erase_clears_every_table_the_ram_vectors_and_the_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "embed_texts", fake_embed)
    path = tmp_path / "memory.sqlite3"
    _v2_database(path, time.time())
    st = Store(path)
    try:
        st.remember("my therapist is Dr. Mehta", "note")
        st.add_turns([(time.time(), "user", "call my therapist tomorrow", "live", "voice", False)])
        st.set_summary(st.recent_turns()[0].conv, "Asked to call the therapist.", 1)
        st.set_meta("extract_upto", "1")
        st.search("therapist")
        assert (tmp_path / "memory.sqlite3.bak-v2").exists()
        counts = st.forget_all()
        assert counts["memory"] == 5 and counts["turns"] == 1 and counts["sessions"] == 31 and counts["playbook"] == 1
        assert not list(tmp_path.glob("*.bak-*"))
        assert all(len(v) == 0 for v in st._vec.values()) and st._touched == {}
        assert st.search("therapist") == [] and st.context("therapist") == "" and st.turn_count() == 0
        assert st.unsummarized() == [] and st.get_meta("extract_upto") == ""
    finally:
        st.close()
    raw = path.read_bytes().decode("latin-1")
    assert "therapist" not in raw and "Lisbon" not in raw  # gone from the file, not just unlinked


def test_old_journal_lines_are_pruned_and_facts_and_summaries_stay(st):
    st.remember("Arnav", "profile", "name")
    st.add_turns([(_day(400), "user", "an old line about tulips", "live", "voice", False)])
    st.set_summary(st.recent_turns(since=0, limit=1)[0].conv, "Talked about tulips.", 1)
    st.add_turns([(_day(3), "user", "a recent line about roses", "live", "voice", False)])
    assert st.prune_turns(365) == 1
    assert st.turn_count() == 1 and st.search("tulips", kinds=("turn",)) == []
    assert st.search("tulips")[0].kind == "summary" and st.by_kind("profile")[0].text == "Arnav"
    assert st.prune_turns(0) == 0  # 0 = keep everything


def test_english_endings_are_stemmed_and_devanagari_words_are_whole(st):
    st.remember("I park on level 3", "note")
    st.add_turns([(_day(2), "user", "the meetings ran late again", "live", "voice", False)])
    st.add_turns([(_day(2), "user", "मुझे अदरक वाली चाय पसंद है", "live", "voice", False)])
    assert st.search("parking")[0].text == "I park on level 3" and st.search("parked")[0].score > 0.6
    assert st.search("meeting")[0].text == "the meetings ran late again"
    assert st.find_about("parking")["memory"] and st.find_about("late meeting")["turns"]
    db = sqlite3.connect(st.path)
    terms = {r[0] for r in db.execute("SELECT term FROM turns_vocab")}
    db.close()
    assert {"चाय", "पसंद", "meet"} <= terms  # whole words (not split at vowel signs), English stemmed


def test_the_question_being_asked_is_not_its_own_answer(st):
    st.add_turns(
        [
            (_day(1), "user", "the plumber is coming on friday", "live", "voice", False),
            (time.time() - 1, "user", "when is the plumber coming", "live", "voice", False),
        ]
    )
    assert [h.text for h in st.search("when is the plumber coming")] == ["the plumber is coming on friday"]
    assert [h.text for h in st.search("plumber", asking="When is the plumber coming")] == ["the plumber is coming on friday"]
    assert "when is the plumber coming" not in st.context("when is the plumber coming", not_after=0.0).split("Earlier today")[0]
