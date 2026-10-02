"""Local long-term memory: one SQLite file — facts, the journal of every turn, conversation
summaries and playbooks — with a fast hybrid search over all of it.

What is kept (schema version 3, ``PRAGMA user_version``):

- ``memory``        facts. kind ``profile`` (stable facts about the user: always in the prompt),
                    ``lesson`` (how to behave: always in the prompt), ``note`` (everything else:
                    recalled by search). Each has where it came from (``source``: journal turn
                    ids, or who wrote it), an ``importance`` and a ``confidence`` (0-1), when it
                    was last recalled (``last_seen``) and ``superseded_by``: a newer value for the
                    same key replaces the old one, which is kept but no longer live — never two
                    contradictory live facts.
- ``turns``         the journal: every line the user said or typed and every line NEO actually
                    said, as it happened (neo2.core.memory.journal writes them). ``tainted`` marks
                    a line of NEO's that quotes a page, a mail or a file: it is never put back in
                    a system prompt and never read as the user's words.
- ``conversations`` runs of turns less than ten minutes apart (a few "Neo…" in a row, a typed
                    exchange): what a summary is written for.
- ``sessions``      one summary per conversation (and the summaries older versions wrote).
- ``playbook``      tool sequences of successful multi-step tasks, recalled by goal similarity.
- ``meta``          small bookkeeping (how far the idle fact extraction has read).

Search (:meth:`Store.search`) is keyword and meaning together: FTS5 (Unicode words, diacritics
folded, English endings stemmed) over facts, summaries and turns, a trigram index for substrings
(Hinglish spelled as one word, part of a Devanagari word), and — once the embedding model is loaded, never waiting
for it — cosine similarity against vectors held in RAM (loaded once, updated on insert). Ranks
are fused (reciprocal rank) and weighed by how much of the query a row really covers — its rare
words count for more than its common ones — then recency, importance and the kind (a fact over a
summary over a turn) decide the order; what scores too low is left out. "yesterday", "last week", "kal" in the
question become a time window. A warm search is a few milliseconds on tens of thousands of turns.

An older database is migrated in place when first opened: it is copied to ``<file>.bak-v<old>``
first (once), and every row it held survives.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neo2.core import logger, runtime
from neo2.core.config import settings
from neo2.core.memory import embed as _emb
from neo2.core.memory import timephrase
from neo2.core.memory.redact import clean as _clean

log = logger(__name__)

Kind = Literal["profile", "lesson", "note"]
SCHEMA_VERSION = 3
# How text is split into search words: Unicode words with their combining marks kept (so a
# Devanagari word is one token, not split at every vowel sign), diacritics folded (café = cafe),
# and English endings stemmed (parking = park) — the same for what is stored and what is asked.
_TOKENIZER = "porter unicode61 remove_diacritics 2 categories 'L* N* Co Mc Mn'"
_FTS = f'tokenize="{_TOKENIZER}"'

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  key TEXT,
  text TEXT NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  uses INTEGER DEFAULT 0,
  source TEXT DEFAULT '',
  importance REAL DEFAULT 0.5,
  confidence REAL DEFAULT 1.0,
  superseded_by INTEGER,
  last_seen REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, key, content='memory', content_rowid='id', {_FTS});
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE OF text, key ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TABLE IF NOT EXISTS memory_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL, model TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS playbook (
  id INTEGER PRIMARY KEY,
  goal TEXT NOT NULL,
  steps TEXT NOT NULL,
  answer TEXT,
  created REAL NOT NULL,
  uses INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS playbook_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL, model TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY, started REAL, ended REAL, summary TEXT, conv_id INTEGER, turns INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(summary, content='sessions', content_rowid='id', {_FTS});
CREATE TRIGGER IF NOT EXISTS ses_ai AFTER INSERT ON sessions BEGIN
  INSERT INTO sessions_fts(rowid, summary) VALUES (new.id, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS ses_ad AFTER DELETE ON sessions BEGIN
  INSERT INTO sessions_fts(sessions_fts, rowid, summary) VALUES('delete', old.id, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS ses_au AFTER UPDATE OF summary ON sessions BEGIN
  INSERT INTO sessions_fts(sessions_fts, rowid, summary) VALUES('delete', old.id, old.summary);
  INSERT INTO sessions_fts(rowid, summary) VALUES (new.id, new.summary);
END;
CREATE TABLE IF NOT EXISTS sessions_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL, model TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  started REAL NOT NULL,
  last_ts REAL NOT NULL,
  source TEXT DEFAULT '',
  turns INTEGER DEFAULT 0,
  last_turn INTEGER DEFAULT 0,
  summary_id INTEGER,
  summarized_upto INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  route TEXT DEFAULT '',
  source TEXT DEFAULT '',
  tainted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(text, content='turns', content_rowid='id', {_FTS});
CREATE TRIGGER IF NOT EXISTS turn_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS turn_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vocab USING fts5vocab('memory_fts', 'row');
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_vocab USING fts5vocab('sessions_fts', 'row');
CREATE VIRTUAL TABLE IF NOT EXISTS turns_vocab USING fts5vocab('turns_fts', 'row');
"""

# Substring search (and scripts unicode61 can't split into words): a trigram index beside each
# FTS table. An SQLite built without the trigram tokenizer does without (LIKE on the few rows).
_TRIGRAM = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_tri USING fts5(text, key, content='memory', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS memtri_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_tri(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TRIGGER IF NOT EXISTS memtri_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_tri(memory_tri, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
END;
CREATE TRIGGER IF NOT EXISTS memtri_au AFTER UPDATE OF text, key ON memory BEGIN
  INSERT INTO memory_tri(memory_tri, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
  INSERT INTO memory_tri(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_tri USING fts5(summary, content='sessions', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS sestri_ai AFTER INSERT ON sessions BEGIN
  INSERT INTO sessions_tri(rowid, summary) VALUES (new.id, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS sestri_ad AFTER DELETE ON sessions BEGIN
  INSERT INTO sessions_tri(sessions_tri, rowid, summary) VALUES('delete', old.id, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS sestri_au AFTER UPDATE OF summary ON sessions BEGIN
  INSERT INTO sessions_tri(sessions_tri, rowid, summary) VALUES('delete', old.id, old.summary);
  INSERT INTO sessions_tri(rowid, summary) VALUES (new.id, new.summary);
END;
CREATE VIRTUAL TABLE IF NOT EXISTS turns_tri USING fts5(text, content='turns', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS turntri_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_tri(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS turntri_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_tri(turns_tri, rowid, text) VALUES('delete', old.id, old.text);
END;
"""
_FTS_TABLES = ("memory_fts", "sessions_fts", "turns_fts")
_TRI_TABLES = ("memory_tri", "sessions_tri", "turns_tri")


@dataclass
class Memory:
    id: int
    kind: str
    key: str
    text: str
    created: float
    updated: float = 0.0
    importance: float = 0.5
    source: str = ""

    @property
    def label(self) -> str:
        """The fact as one phrase: "sister name: Priya", "city: Dubai", "likes cold coffee"."""
        return fact_label(self.key, self.text)


@dataclass
class Playbook:
    id: int
    goal: str
    steps: list[dict[str, Any]]
    answer: str
    created: float
    similarity: float = 0.0


@dataclass
class Turn:
    id: int
    conv: int
    ts: float
    role: str  # "user" | "assistant"
    text: str
    route: str = ""
    source: str = ""  # "voice" | "typed" | "tool" (a note of what NEO ran: names only)
    tainted: bool = False


@dataclass
class Hit:
    """One search result, from any of the three tables."""

    table: str  # "memory" | "sessions" | "turns"
    id: int
    kind: str  # "profile" | "lesson" | "note" | "summary" | "turn"
    text: str
    ts: float
    score: float = 0.0
    key: str = ""
    role: str = ""

    @property
    def ref(self) -> str:
        """A short tag a model can cite or forget by: f12 (fact), s5 (summary), t881 (turn)."""
        return {"memory": "f", "sessions": "s", "turns": "t"}[self.table] + str(self.id)


_KEEP_PLAYBOOKS = 300
_CONV_GAP_S = 600.0  # turns further apart than this start a new conversation
_TURN_MAX_CHARS = 2000  # one journal line (a long answer is cut: the gist is in the summary)
_SEM_MIN = 0.4  # a semantic hit below this isn't about the query
_NEAR_DUP = 0.9  # two facts of one kind this close are the same fact, said again
MIN_SCORE = 0.36  # a fused score below this is left out of search results
HIGH_SCORE = 0.62  # clearly about the query (what a voice turn is told unasked)
_KIND_BOOST = {"profile": 0.12, "lesson": 0.10, "note": 0.08, "summary": 0.05, "turn": 0.0}
_FACT_KINDS = ("profile", "lesson", "note")
_HALF_LIFE_DAYS = 30.0
# What a playbook may remember of a step: the tool and its settings — an allowlist, so a new
# tool's free-text argument (a title, a label, a place, a path) never slips through.
SETTING_ARGS = {
    "app", "name", "keys", "direction", "level", "minutes", "action", "kind", "button", "count",
    "new_tab", "send", "submit", "site", "calendar", "role", "news", "limit", "max_chars", "start", "lines",
}
# Words that carry no meaning for a search (English and Hinglish): they'd match everything.
_STOP = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with", "about", "and", "or", "but",
    "if", "so", "as", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "she", "it", "its", "they", "them",
    "their", "this", "that", "these", "those", "what", "which", "who", "whom", "when", "where", "why", "how",
    "can", "could", "would", "should", "will", "shall", "may", "might", "not", "no", "yes", "any", "some",
    "all", "just", "also", "there", "here", "then", "than", "too", "very", "please", "tell", "say", "said",
    "know", "remember", "recall", "anything", "something", "neo", "hey", "ok", "okay", "talk", "talked",
    "discuss", "discussed", "mera", "meri", "mere", "mujhe", "mujhko", "main", "mai", "hum", "humne", "hamara",
    "hamari", "aap", "tum", "tumhe", "tumhara", "kya", "kab", "kahan", "kaun", "kaise", "kyun", "kyon", "hai",
    "hain", "tha", "thi", "ho", "hoon", "hu", "ka", "ki", "ke", "ko", "se", "mein", "par", "aur", "ya", "bhi",
    "nahi", "nahin", "toh", "na", "ye", "yeh", "wo", "woh", "vo", "kuch", "koi", "sab", "baare", "bare",
    "batao", "bata", "yaad", "baat",
})  # fmt: skip
_TOKEN = re.compile(r"(?:[^\W_]|[\u0300-\u036F\u0900-\u0DFF\u200C\u200D])+")


def _host_only(v: Any) -> Any:
    from urllib.parse import urlsplit

    s = str(v)
    if "://" in s:
        u = urlsplit(s)
        return f"{u.scheme}://{u.hostname or ''}"
    return v


def path_only(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in steps:
        args = s.get("args") or {}
        keep = {k: v for k, v in args.items() if k in SETTING_ARGS}
        for k in ("url", "target"):  # a link keeps its site, never its path or query
            if k in args and "://" in str(args[k]):
                keep[k] = _host_only(args[k])
        out.append({"tool": s.get("tool", ""), "args": keep})
    return out


def fact_label(key: str, text: str) -> str:
    """A fact as one phrase. The key names what the value is ("city: Dubai", "sister name:
    Priya") unless the text already says it ("likes cold coffee", "allergic to peanuts")."""
    key = (key or "").strip()
    if not key or key.startswith(("pref.", "allergy.")):
        return text
    words = [w for w in re.split(r"[._\s]+", key.lower()) if w]
    low = text.lower()
    if words and all(w in low for w in words):
        return text
    return f"{' '.join(words)}: {text}"


def tokens(s: str) -> list[str]:
    """The words of a query worth searching for: Unicode words (Devanagari with its vowel
    signs), lower-cased, without the stop words — unless nothing else is left."""
    all_ = [t.lower() for t in _TOKEN.findall(s or "")]
    keep = [t for t in all_ if t not in _STOP and (len(t) > 1 or not t.isascii())]
    return list(dict.fromkeys(keep))


class MemoryDisabled(RuntimeError):
    """This agent runs without long-term memory."""


class _Stemmer:
    """The form the search index keeps a word in ("parking" → "park"), asked of SQLite's own
    tokenizer (a one-row FTS table in RAM) and remembered: what a query word weighs is looked up
    in the index's vocabulary by that form."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._known: dict[str, str] = {}

    def __call__(self, word: str) -> str:
        hit = self._known.get(word)
        if hit is not None:
            return hit
        with self._lock:
            try:
                if self._db is None:
                    self._db = sqlite3.connect(":memory:", check_same_thread=False)
                    self._db.execute(f"CREATE VIRTUAL TABLE w USING fts5(x, {_FTS})")
                    self._db.execute("CREATE VIRTUAL TABLE v USING fts5vocab('w', 'row')")
                self._db.execute("DELETE FROM w")
                self._db.execute("INSERT INTO w(x) VALUES (?)", (word,))
                rows = self._db.execute("SELECT term FROM v").fetchall()
                stem = rows[0][0] if len(rows) == 1 else word
            except sqlite3.Error:
                stem = word
            if len(self._known) > 4096:
                self._known.clear()
            self._known[word] = stem
        return stem


_stem = _Stemmer()


class _Vectors:
    """One table's vectors, in RAM: loaded once, kept in step on every insert and delete, so a
    search never reads them back from SQLite. Rows made by another embedding model are left out
    (and made again by Store.backfill)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._at: dict[int, int] = {}
        self._mat = np.zeros((0, _emb.DIM), dtype=np.float32)
        self._n = 0

    def load(self, rows: Iterable[tuple[int, bytes, str]]) -> None:
        ids, vecs = [], []
        for rid, blob, model in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape[0] == _emb.DIM and (model or "") == _emb.MODEL_ID:
                ids.append(int(rid))
                vecs.append(v)
        with self._lock:
            self._ids = ids
            self._at = {i: n for n, i in enumerate(ids)}
            self._mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, _emb.DIM), dtype=np.float32)
            self._n = len(ids)

    def put(self, rid: int, vec: np.ndarray) -> None:
        with self._lock:
            at = self._at.get(rid)
            if at is not None:
                self._mat[at] = vec
                return
            if self._n == self._mat.shape[0]:  # grow by doubling: an insert is O(1) amortised
                grown = np.zeros((max(64, self._n * 2), _emb.DIM), dtype=np.float32)
                grown[: self._n] = self._mat[: self._n]
                self._mat = grown
            self._mat[self._n] = vec
            self._ids.append(rid)
            self._at[rid] = self._n
            self._n += 1

    def remove(self, rid: int) -> None:
        with self._lock:
            at = self._at.pop(rid, None)
            if at is None:
                return
            last = self._n - 1
            if at != last:  # the last row takes its place
                self._mat[at] = self._mat[last]
                moved = self._ids[last]
                self._ids[at] = moved
                self._at[moved] = at
            self._ids.pop()
            self._n = last

    def clear(self) -> None:
        self.load([])

    def has(self, rid: int) -> bool:
        return rid in self._at

    def __len__(self) -> int:
        return self._n

    def top(self, q: np.ndarray | None, k: int) -> list[tuple[int, float]]:
        if q is None:
            return []
        with self._lock:
            if not self._n:
                return []
            return [(self._ids[i], s) for i, s in _emb.cosine_top(q, self._mat[: self._n], k)]


class Store:
    """Thread-safe: requests run concurrently (worker threads and the event loop), so every
    public method holds a lock for its SQL — one for the connection that writes, one for the
    connection searches read from (WAL: a search never waits for the journal's commit).
    Embeddings are computed before taking either — a lock is never held while a model runs."""

    def __init__(self, path: Path | None = None, *, agent: bool = False) -> None:
        """`agent`: an embedded agent's memory — a database it creates is marked as NEO's own
        (neo2.core.config.AGENT_DB_ID), so it's never mistaken for a host's, nor a host's for it."""
        self.path = Path(path or settings().memory_db)
        fresh = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if not os.path.exists(self.path):  # private from the start, whatever the process umask
            os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        self._lock = threading.RLock()
        self._rlock = threading.RLock()
        self._closed = False
        self._erased_at = 0.0  # time.monotonic() of the last erase: older queued writes are dropped
        self._journal: Any = None  # neo2.core.memory.journal.Journal, on first use
        self._backfilled = False
        self._touched: dict[int, int] = {}  # fact id -> recalls not yet written (see touch_later)
        self._touch_lock = threading.Lock()
        self.trigram = True
        self._vec = {"memory_vec": _Vectors(), "sessions_vec": _Vectors(), "playbook_vec": _Vectors()}
        if not fresh:
            self._backup_before_migration()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        with self._lock:
            if agent and fresh:  # before WAL: the mark goes into the file's own header at once
                from neo2.core.config import AGENT_DB_ID

                self._db.execute(f"PRAGMA application_id = {int(AGENT_DB_ID)}")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")  # WAL: safe against a crash, no fsync per turn
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("PRAGMA secure_delete=ON")  # what's deleted is overwritten, not left in free pages
            self._migrate(fresh)
            self._db.commit()
            for table, vecs in self._vec.items():
                vecs.load(self._db.execute(f"SELECT id, vec, model FROM {table}").fetchall())
        self._rdb = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        self._rdb.execute("PRAGMA busy_timeout=5000")
        self._rdb.execute("PRAGMA query_only=ON")

    # ---- schema ------------------------------------------------------------------------
    def _backup_before_migration(self) -> None:
        """Before an older database is changed, a copy of it as it was: ``<file>.bak-v<old>``,
        made once (SQLite's own backup: consistent, with whatever is still in the WAL)."""
        try:
            src = sqlite3.connect(self.path, timeout=5)
        except sqlite3.Error:
            return
        try:
            old = int(src.execute("PRAGMA user_version").fetchone()[0])
            has_rows = src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory'").fetchone()
            if old >= SCHEMA_VERSION or not has_rows:
                return
            bak = self.path.with_name(f"{self.path.name}.bak-v{old}")
            if bak.exists():
                return
            os.close(os.open(bak, os.O_CREAT | os.O_WRONLY, 0o600))
            dst = sqlite3.connect(bak)
            try:
                src.backup(dst)
            finally:
                dst.close()
            log.info(f"[memory] kept a copy of the memory database before upgrading it (v{old} → v{SCHEMA_VERSION})")
        except (sqlite3.Error, OSError) as e:
            log.warning(f"[memory] couldn't copy the database before upgrading it: {str(e)[:80]}")
            with contextlib.suppress(OSError, UnboundLocalError):
                bak.unlink()
        finally:
            src.close()

    def _columns(self, table: str) -> set[str]:
        return {r[1] for r in self._db.execute(f"PRAGMA table_info({table})").fetchall()}

    def _migrate(self, fresh: bool) -> None:
        db = self._db
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        legacy = "memory" in tables and version < SCHEMA_VERSION
        if legacy:
            # An index made with the old tokenizer (and its triggers) is dropped and built again
            # below; the rows themselves are only ever added to.
            for trig in ("mem_ai", "mem_ad", "mem_au"):
                db.execute(f"DROP TRIGGER IF EXISTS {trig}")
            db.execute("DROP TABLE IF EXISTS memory_vocab")
            db.execute("DROP TABLE IF EXISTS memory_fts")
            have = self._columns("memory")
            for col, decl in (
                ("source", "TEXT DEFAULT ''"),
                ("importance", "REAL DEFAULT 0.5"),
                ("confidence", "REAL DEFAULT 1.0"),
                ("superseded_by", "INTEGER"),
                ("last_seen", "REAL"),
            ):
                if col not in have:
                    db.execute(f"ALTER TABLE memory ADD COLUMN {col} {decl}")
            for t in ("memory_vec", "playbook_vec"):
                if t in tables and "model" not in self._columns(t):
                    db.execute(f"ALTER TABLE {t} ADD COLUMN model TEXT DEFAULT ''")
                    db.execute(f"UPDATE {t} SET model=?", (_emb.MODEL_ID,))  # the only model there was
            if "sessions" in tables:
                have = self._columns("sessions")
                if "conv_id" not in have:
                    db.execute("ALTER TABLE sessions ADD COLUMN conv_id INTEGER")
                if "turns" not in have:
                    db.execute("ALTER TABLE sessions ADD COLUMN turns INTEGER DEFAULT 0")
        db.executescript(_SCHEMA)
        try:
            db.executescript(_TRIGRAM)
        except sqlite3.OperationalError:  # this SQLite has no trigram tokenizer: LIKE does it
            self.trigram = False
        if legacy:
            # A profile fact is worth more than a note; what the user told NEO is certain.
            db.execute("UPDATE memory SET importance=0.8 WHERE kind='profile' AND importance=0.5")
            db.execute("UPDATE memory SET importance=0.7 WHERE kind='lesson' AND importance=0.5")
            for t in _FTS_TABLES + (_TRI_TABLES if self.trigram else ()):
                db.execute(f"INSERT INTO {t}({t}) VALUES('rebuild')")
        if version < 2:
            # Playbooks saved by older versions kept message bodies, typed text, commands,
            # titles and full links. Rewrite them, then VACUUM so the old bytes are gone from
            # the file too (not just unreferenced).
            for pid, steps in db.execute("SELECT id, steps FROM playbook").fetchall():
                try:
                    steps_clean = path_only(json.loads(steps))
                except (ValueError, TypeError, AttributeError):
                    steps_clean = []
                db.execute("UPDATE playbook SET steps=?, answer='' WHERE id=?", (json.dumps(steps_clean), pid))
        if version < SCHEMA_VERSION:
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.commit()
            if version < 2 and not fresh:
                db.execute("VACUUM")
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ---- vectors ---------------------------------------------------------------------
    @staticmethod
    def _embed(text: str) -> np.ndarray | None:
        v = _emb.embed_texts([text])
        return None if v is None else v[0]

    def _put_vec(self, table: str, row_id: int, vec: np.ndarray | None) -> None:
        """Under the write lock. No vector (the model isn't loaded): the old one, made for other
        words, goes — Store.backfill makes the new one later."""
        if vec is None:
            self._db.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
            self._vec[table].remove(row_id)
            self._backfilled = False
            return
        self._db.execute(
            f"INSERT OR REPLACE INTO {table}(id, vec, model) VALUES (?, ?, ?)", (row_id, vec.tobytes(), _emb.MODEL_ID)
        )
        self._vec[table].put(row_id, vec)

    def _drop_vec(self, table: str, row_id: int) -> None:
        self._db.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        self._vec[table].remove(row_id)

    def backfill(self, limit: int = 256) -> int:
        """Make the vectors that are missing — rows written while the embedding model wasn't
        loaded, or by another model. Does nothing until it is loaded (never waits for it).
        Returns how many it made."""
        if self._closed:
            return 0
        made = 0
        plan = (
            ("memory", "memory_vec", "SELECT id, key, text FROM memory WHERE superseded_by IS NULL"),
            ("sessions", "sessions_vec", "SELECT id, '', summary FROM sessions WHERE summary IS NOT NULL AND summary != ''"),
            ("playbook", "playbook_vec", "SELECT id, '', goal FROM playbook"),
        )
        for _table, vtable, sql in plan:
            with self._lock:
                rows = [r for r in self._db.execute(sql).fetchall() if not self._vec[vtable].has(int(r[0]))]
            rows = rows[: max(0, limit - made)]
            if not rows:
                continue
            vecs = _emb.embed_texts([fact_label(r[1], r[2]) for r in rows])
            if vecs is None:  # no model (yet): try again later
                return made
            with self._lock:
                if self._closed:
                    return made
                for r, v in zip(rows, vecs, strict=True):
                    self._put_vec(vtable, int(r[0]), v)
                self._db.commit()
            made += len(rows)
        if made < limit:
            self._backfilled = True  # nothing is missing any more (until a row is written without one)
        return made

    # ---- facts: write ------------------------------------------------------------------
    def remember(
        self,
        text: str,
        kind: Kind = "note",
        key: str = "",
        *,
        source: str = "",
        importance: float | None = None,
        confidence: float = 1.0,
        since: float | None = None,
    ) -> int:
        """Keep a fact; returns its id. A secret's value in `text` is never stored
        (neo2.core.memory.redact). The same fact again is not added twice: the same text, or — with
        the embedding model loaded — one of the same kind that means the same, is refreshed. With
        a `key`, a different value supersedes the old one: the old row stays (marked
        ``superseded_by``), only the new one is live.

        `since`: when (time.monotonic()) the work that found this fact began — a fact found in
        what was said before an erase is not written after it (returns 0)."""
        now = time.time()
        text = _clean(" ".join(text.split()))
        key = (key or "").strip()
        if importance is None:
            importance = {"profile": 0.8, "lesson": 0.7}.get(kind, 0.5)
        importance = min(1.0, max(0.0, float(importance)))
        vec = self._embed(fact_label(key, text))
        with self._lock:
            db = self._db
            if since is not None and self._erased_at >= since:
                return 0  # the user erased NEO's memory meanwhile: it stays erased
            old = None
            if key:  # keyed facts: one live value per (kind, key)
                old = db.execute(
                    "SELECT id, text, source FROM memory WHERE kind=? AND key=? AND superseded_by IS NULL "
                    "ORDER BY updated DESC LIMIT 1",
                    (kind, key),
                ).fetchone()
                if old is None:
                    # The same value under a key that says the same thing in other words
                    # ("sister.name" / "sister_name", from the rules and from a model): one fact.
                    words = {w for w in re.split(r"[._\s]+", key.lower()) if w}
                    for row in db.execute(
                        "SELECT id, text, source, key FROM memory WHERE kind=? AND lower(text)=lower(?) AND superseded_by IS NULL",
                        (kind, text),
                    ).fetchall():
                        if words & {w for w in re.split(r"[._\s]+", (row[3] or "").lower()) if w}:
                            self._refresh(int(row[0]), now, source, row[2], importance)
                            db.commit()
                            return int(row[0])
                if old is not None and old[1].strip().casefold() == text.casefold():
                    self._refresh(int(old[0]), now, source, old[2], importance)
                    db.commit()
                    return int(old[0])
            else:
                dup = db.execute(
                    "SELECT id, source FROM memory WHERE kind=? AND lower(text)=lower(?) AND superseded_by IS NULL",
                    (kind, text),
                ).fetchone()
                if dup is None and vec is not None:
                    for mid, sim in self._vec["memory_vec"].top(vec, 3):
                        if sim < _NEAR_DUP:
                            break
                        row = db.execute(
                            "SELECT id, source FROM memory WHERE id=? AND kind=? AND superseded_by IS NULL AND "
                            "(key IS NULL OR key='')",
                            (mid, kind),
                        ).fetchone()
                        if row is not None:  # said again in other words: the newer wording stays
                            db.execute("UPDATE memory SET text=? WHERE id=?", (text, mid))
                            self._put_vec("memory_vec", mid, vec)
                            dup = row
                            break
                if dup is not None:
                    self._refresh(int(dup[0]), now, source, dup[1], importance)
                    db.commit()
                    return int(dup[0])
            cur = db.execute(
                "INSERT INTO memory(kind,key,text,created,updated,source,importance,confidence,last_seen) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (kind, key, text, now, now, source, importance, float(confidence), now),
            )
            mid = int(cur.lastrowid or 0)
            if old is not None:
                db.execute("UPDATE memory SET superseded_by=? WHERE kind=? AND key=? AND superseded_by IS NULL AND id!=?", (mid, kind, key, mid))
                self._drop_vec("memory_vec", int(old[0]))  # only live facts are searched by meaning
            self._put_vec("memory_vec", mid, vec)
            db.commit()
            return mid

    def _refresh(self, mid: int, now: float, source: str, had: str | None, importance: float) -> None:
        """The same fact, told again: newer, seen now, from one more place, never less important."""
        srcs = [s for s in (had or "").split(",") if s]
        for s in source.split(","):
            if s and s not in srcs:
                srcs.append(s)
        self._db.execute(
            "UPDATE memory SET updated=?, last_seen=?, source=?, importance=max(importance, ?) WHERE id=?",
            (now, now, ",".join(srcs[-12:]), importance, mid),
        )

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            n = self._db.execute("DELETE FROM memory WHERE id=?", (memory_id,)).rowcount
            self._drop_vec("memory_vec", memory_id)
            # The values it had superseded go with it (and theirs): forgetting a fact never
            # brings an older one back, and never leaves one behind.
            gone = [memory_id]
            while gone:
                marks = ",".join("?" * len(gone))
                older = [int(r[0]) for r in self._db.execute(f"SELECT id FROM memory WHERE superseded_by IN ({marks})", gone)]
                for old in older:
                    self._db.execute("DELETE FROM memory WHERE id=?", (old,))
                    self._drop_vec("memory_vec", old)
                gone = older
            self._db.commit()
            return n > 0

    def close(self) -> None:
        """Close the database (an embedded agent's, when its host closes or erases it). What the
        journal still held is written first."""
        j = self._journal
        if j is not None:
            with contextlib.suppress(Exception):
                j.close()
        with contextlib.suppress(Exception):
            self.flush_touches()
        with self._lock, self._rlock:
            self._closed = True
            self._db.close()
            self._rdb.close()

    def forget_all(self) -> dict[str, int]:
        """Delete every memory, playbook, session summary and journal line (and the copy kept
        from before an upgrade), and reclaim the space on disk. Lines still waiting to be
        written are dropped, not written afterwards."""
        self._erased_at = time.monotonic()
        with self._lock, self._rlock:
            counts = {
                t: int(self._db.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
                for t in ("memory", "playbook", "sessions", "turns")
            }
            for t in ("memory", "memory_vec", "playbook", "playbook_vec", "sessions", "sessions_vec", "turns", "conversations", "meta"):
                self._db.execute(f"DELETE FROM {t}")
            for t in _FTS_TABLES + (_TRI_TABLES if self.trigram else ()):
                self._db.execute(f"INSERT INTO {t}({t}) VALUES('rebuild')")
            self._db.commit()
            for v in self._vec.values():
                v.clear()
            with self._touch_lock:
                self._touched.clear()
            self._erased_at = time.monotonic()
            self._db.execute("VACUUM")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        for bak in self.path.parent.glob(self.path.name + ".bak-v*"):
            with contextlib.suppress(OSError):
                bak.unlink()
        return counts

    # ---- facts: read -------------------------------------------------------------------
    def recall(self, query: str, limit: int = 8) -> list[Memory]:
        """The facts that match `query`, best first (see :meth:`search`)."""
        return self.facts(h.id for h in self.search(query, kinds=_FACT_KINDS, k=limit))

    def by_kind(self, kind: Kind, limit: int = 40) -> list[Memory]:
        """The live facts of one kind, the most important and the newest first."""
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT id, kind, key, text, created, updated, importance, source FROM memory "
                "WHERE kind=? AND superseded_by IS NULL ORDER BY importance DESC, updated DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def facts(self, ids: Iterable[int]) -> list[Memory]:
        """These facts, in the order asked for (ids that are gone are left out)."""
        ids = [int(i) for i in ids]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._rlock:
            rows = self._rdb.execute(
                f"SELECT id, kind, key, text, created, updated, importance, source FROM memory WHERE id IN ({marks})", ids
            ).fetchall()
        by_id = {int(r[0]): Memory(*r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def history_of(self, kind: Kind, key: str) -> list[Memory]:
        """Every value a keyed fact has had, the live one first."""
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT id, kind, key, text, created, updated, importance, source FROM memory "
                "WHERE kind=? AND key=? ORDER BY superseded_by IS NOT NULL, updated DESC",
                (kind, key),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def touch_later(self, ids: Iterable[int]) -> None:
        """These facts were recalled. Only noted here: a search never writes. The journal's
        writer (or the next write, or close) puts the counts in the database: flush_touches."""
        with self._touch_lock:
            for i in ids:
                self._touched[i] = self._touched.get(i, 0) + 1

    def flush_touches(self) -> int:
        """Write the recalls noted since the last time (``uses``, ``last_seen``)."""
        with self._touch_lock:
            got, self._touched = self._touched, {}
        if not got or self._closed:
            return 0
        now = time.time()
        with self._lock:
            self._db.executemany(
                "UPDATE memory SET uses=uses+?, last_seen=? WHERE id=?", [(n, now, i) for i, n in got.items()]
            )
            self._db.commit()
        return len(got)

    # ---- playbooks ---------------------------------------------------------------------
    def save_playbook(self, goal: str, steps: list[dict[str, Any]], answer: str) -> int:
        """Store the tool sequence of a successful task. Near-duplicate goals replace the old entry."""
        goal = goal.strip()
        vec = self._embed(goal)
        with self._lock:
            for pid, sim in self._vec["playbook_vec"].top(vec, 1):
                if sim >= 0.92:
                    self._db.execute(
                        "UPDATE playbook SET steps=?, answer=?, created=? WHERE id=?",
                        (json.dumps(steps), answer[:500], time.time(), pid),
                    )
                    self._db.commit()
                    return pid
            cur = self._db.execute(
                "INSERT INTO playbook(goal, steps, answer, created) VALUES (?,?,?,?)",
                (goal, json.dumps(steps), answer[:500], time.time()),
            )
            pid = int(cur.lastrowid or 0)
            if vec is not None:
                self._put_vec("playbook_vec", pid, vec)
            # Keep the newest few hundred: old paths go stale anyway.
            stale = [
                int(r[0])
                for r in self._db.execute(
                    "SELECT id FROM playbook ORDER BY created DESC LIMIT -1 OFFSET ?", (_KEEP_PLAYBOOKS,)
                ).fetchall()
            ]
            for sid in stale:
                self._db.execute("DELETE FROM playbook WHERE id=?", (sid,))
                self._drop_vec("playbook_vec", sid)
            self._db.commit()
            return pid

    def similar_playbooks(self, goal: str, k: int = 2, min_sim: float = 0.6) -> list[Playbook]:
        q = self._embed(goal)
        hits = [(pid, s) for pid, s in self._vec["playbook_vec"].top(q, k) if s >= min_sim]
        if not hits:
            return []
        with self._lock:
            out = []
            for pid, s in hits:
                r = self._db.execute(
                    "SELECT id, goal, steps, answer, created FROM playbook WHERE id=?", (pid,)
                ).fetchone()
                if r:
                    out.append(Playbook(int(r[0]), r[1], json.loads(r[2]), r[3] or "", r[4], s))
                    self._db.execute("UPDATE playbook SET uses=uses+1 WHERE id=?", (pid,))
            self._db.commit()
            return out

    def playbook_context(self, goal: str) -> str:
        # Past requests are memory too: they go to the cloud model only with NEO_CLOUD_MEMORY=full.
        if settings().cloud_memory != "full":
            return ""
        pbs = self.similar_playbooks(goal)
        if not pbs:
            return ""
        lines = [
            "[PLAYBOOKS] Similar requests you completed before — follow the same path unless something differs:"
        ]
        for pb in pbs:
            steps = " → ".join(f"{s['tool']}({_short_args(s.get('args', {}))})" for s in pb.steps[:10])
            lines.append(f"- “{pb.goal}”: {steps}")
        return "\n".join(lines)

    # ---- the journal (written by neo2.core.memory.journal's one writer thread) -----------
    def add_turns(
        self, rows: list[tuple[float, str, str, str, str, bool]], *, keep: Callable[[int], bool] | None = None
    ) -> list[Turn]:
        """Write journal lines — (ts, role, text, route, source, tainted) — in one transaction and
        return them with their ids. Each joins the open conversation, or starts one when the
        last line is more than ten minutes old. `keep(i)`: asked for row i with the database
        locked, so "was this queued before an erase?" can't change between the answer and the
        write."""
        if not rows or self._closed:
            return []
        out: list[Turn] = []
        with self._lock:
            if keep is not None:
                rows = [r for i, r in enumerate(rows) if keep(i)]
                if not rows:
                    return []
            db = self._db
            last = db.execute("SELECT id, last_ts FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
            conv, last_ts = (int(last[0]), float(last[1])) if last else (0, 0.0)
            for ts, role, text, route, source, tainted in rows:
                text = _clean(text.strip())[:_TURN_MAX_CHARS]
                if not text:
                    continue
                if not conv or ts - last_ts > _CONV_GAP_S:
                    cur = db.execute("INSERT INTO conversations(started, last_ts, source) VALUES (?,?,?)", (ts, ts, source))
                    conv = int(cur.lastrowid or 0)
                cur = db.execute(
                    "INSERT INTO turns(session_id, ts, role, text, route, source, tainted) VALUES (?,?,?,?,?,?,?)",
                    (conv, ts, role, text, route, source, int(bool(tainted))),
                )
                tid = int(cur.lastrowid or 0)
                last_ts = max(last_ts, ts)
                db.execute(
                    "UPDATE conversations SET last_ts=?, turns=turns+1, last_turn=? WHERE id=?", (last_ts, tid, conv)
                )
                out.append(Turn(tid, conv, ts, role, text, route, source, bool(tainted)))
            db.commit()
        return out

    def recent_turns(self, *, since: float = 0.0, limit: int = 20, spoken_only: bool = True) -> list[Turn]:
        """The latest journal lines since `since`, oldest first. `spoken_only`: without the notes
        of what NEO ran."""
        where = "ts >= ?" + (" AND source != 'tool'" if spoken_only else "")
        with self._rlock:
            rows = self._rdb.execute(
                f"SELECT id, session_id, ts, role, text, route, source, tainted FROM turns WHERE {where} "
                "ORDER BY ts DESC, id DESC LIMIT ?",  # by when it was said: a spoken line is written once it is final
                (since, limit),
            ).fetchall()
        return [Turn(int(r[0]), int(r[1]), float(r[2]), r[3], r[4], r[5] or "", r[6] or "", bool(r[7])) for r in reversed(rows)]

    def turns_after(self, turn_id: int, limit: int = 80) -> list[Turn]:
        """Journal lines with an id above `turn_id`, oldest first (the idle fact extraction's)."""
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT id, session_id, ts, role, text, route, source, tainted FROM turns WHERE id > ? ORDER BY id LIMIT ?",
                (turn_id, limit),
            ).fetchall()
        return [Turn(int(r[0]), int(r[1]), float(r[2]), r[3], r[4], r[5] or "", r[6] or "", bool(r[7])) for r in rows]

    def turns_of(self, conv: int) -> list[Turn]:
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT id, session_id, ts, role, text, route, source, tainted FROM turns WHERE session_id=? ORDER BY ts, id",
                (conv,),
            ).fetchall()
        return [Turn(int(r[0]), int(r[1]), float(r[2]), r[3], r[4], r[5] or "", r[6] or "", bool(r[7])) for r in rows]

    def turn_count(self) -> int:
        with self._rlock:
            return int(self._rdb.execute("SELECT count(*) FROM turns").fetchone()[0])

    def prune_turns(self, keep_days: int) -> int:
        """Journal lines older than `keep_days` go (facts and summaries stay). 0 or less: keep all."""
        if keep_days <= 0 or self._closed:
            return 0
        cutoff = time.time() - keep_days * 86400
        with self._lock:
            n = self._db.execute("DELETE FROM turns WHERE ts < ?", (cutoff,)).rowcount
            if n:
                self._db.execute("DELETE FROM conversations WHERE id NOT IN (SELECT DISTINCT session_id FROM turns)")
            self._db.commit()
        return int(n or 0)

    def get_meta(self, key: str, default: str = "") -> str:
        with self._rlock:
            r = self._rdb.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
            self._db.commit()

    # ---- conversations and their summaries -----------------------------------------------
    def unsummarized(self, *, idle_s: float = 0.0, now: float | None = None) -> list[tuple[int, float, float]]:
        """(conversation id, started, last line's time) of each conversation with lines its
        summary doesn't cover yet and nothing said for `idle_s` — oldest first."""
        now = time.time() if now is None else now
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT id, started, last_ts FROM conversations WHERE last_turn > summarized_upto AND last_ts <= ? ORDER BY id",
                (now - idle_s,),
            ).fetchall()
        return [(int(r[0]), float(r[1]), float(r[2])) for r in rows]

    def set_summary(self, conv: int, summary: str, upto: int, *, since: float | None = None) -> int:
        """The summary of a conversation, covering its lines up to turn `upto`: written once per
        conversation and rewritten when it went on. `since`: as for :meth:`remember` — a summary
        of what was said before an erase is not written after it."""
        summary = _clean(summary.strip())
        vec = self._embed(summary) if summary else None
        with self._lock:
            if since is not None and self._erased_at >= since:
                return 0
            db = self._db
            c = db.execute("SELECT started, last_ts, turns, summary_id FROM conversations WHERE id=?", (conv,)).fetchone()
            if c is None:
                return 0
            sid = int(c[3] or 0)
            if not summary:  # nothing worth saying: the lines count as covered all the same
                db.execute("UPDATE conversations SET summarized_upto=? WHERE id=?", (upto, conv))
                db.commit()
                return sid
            if sid and db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
                db.execute("UPDATE sessions SET summary=?, ended=?, turns=? WHERE id=?", (summary, c[1], c[2], sid))
            else:
                cur = db.execute(
                    "INSERT INTO sessions(started, ended, summary, conv_id, turns) VALUES (?,?,?,?,?)",
                    (c[0], c[1], summary, conv, c[2]),
                )
                sid = int(cur.lastrowid or 0)
            db.execute("UPDATE conversations SET summary_id=?, summarized_upto=? WHERE id=?", (sid, upto, conv))
            self._put_vec("sessions_vec", sid, vec)
            db.commit()
            return sid

    def add_session_summary(
        self, summary: str, started: float, ended: float | None = None, *, since: float | None = None
    ) -> int:
        """A summary that isn't of a journal conversation (an agent that keeps no journal).
        `since`: as for :meth:`remember` — nothing is written after an erase that came later."""
        summary = _clean(summary.strip())
        vec = self._embed(summary)
        with self._lock:
            if since is not None and self._erased_at >= since:
                return 0
            cur = self._db.execute(
                "INSERT INTO sessions(started, ended, summary) VALUES (?,?,?)", (started, ended or time.time(), summary)
            )
            sid = int(cur.lastrowid or 0)
            if vec is not None:
                self._put_vec("sessions_vec", sid, vec)
            self._db.commit()
            return sid

    def recent_sessions(self, n: int = 3) -> list[tuple[float, str]]:
        with self._rlock:
            rows = self._rdb.execute(
                "SELECT started, summary FROM sessions WHERE summary IS NOT NULL AND summary != '' "
                "ORDER BY started DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [(float(r[0]), r[1]) for r in rows]

    # ---- forgetting by what it's about ---------------------------------------------------
    def find_about(self, query: str) -> dict[str, list[int]]:
        """What a "forget X" would remove: the facts about X, the journal lines that mention it
        (NEO's own too) and the summaries that do. Every word of X must be there: a line that
        shares one word with it is not about it, and nothing is matched by meaning — what goes
        is exactly what the user is shown."""
        toks = tokens(query)
        if not toks:
            return {"memory": [], "turns": [], "sessions": []}
        out: dict[str, list[int]] = {}
        with self._rlock:
            for table, col in (("memory", "text"), ("turns", "text"), ("sessions", "summary")):
                ids = self._keyword(table, toks, 400, every=True, spoken_only=False)
                if not ids:
                    ids = self._substring(table, col, toks, 400, every=True, spoken_only=False)
                out[table] = ids
            # What NEO answered to a line of the user's about it goes too (its reply, the note of
            # what it ran), whether or not the answer repeats the words: up to the user's next line.
            replies: list[int] = []
            for tid in out["turns"]:
                row = self._rdb.execute("SELECT session_id, role FROM turns WHERE id=?", (tid,)).fetchone()
                if row is None or row[1] != "user":
                    continue
                nxt = self._rdb.execute(
                    "SELECT min(id) FROM turns WHERE session_id=? AND id>? AND role='user'", (row[0], tid)
                ).fetchone()[0]
                replies += [
                    int(r[0])
                    for r in self._rdb.execute(
                        "SELECT id FROM turns WHERE session_id=? AND id>? AND id<? AND role='assistant'",
                        (row[0], tid, nxt if nxt is not None else 1 << 62),
                    )
                ]
            out["turns"] = sorted(set(out["turns"]) | set(replies))
        return out

    def forget_items(self, *, memory: Iterable[int] = (), turns: Iterable[int] = (), sessions: Iterable[int] = ()) -> dict[str, int]:
        """Delete these facts, journal lines and summaries. A conversation that loses a line
        loses its summary too (it may say what was forgotten) and is summarised again from what
        is left; one with nothing left goes."""
        memory, turns, sessions = [int(i) for i in memory], [int(i) for i in turns], {int(i) for i in sessions}
        done = {"memory": 0, "turns": 0, "sessions": 0}
        for mid in memory:
            done["memory"] += int(self.forget(mid))
        with self._lock:
            db = self._db
            convs: set[int] = set()
            for i in range(0, len(turns), 400):
                chunk = turns[i : i + 400]
                marks = ",".join("?" * len(chunk))
                convs |= {int(r[0]) for r in db.execute(f"SELECT DISTINCT session_id FROM turns WHERE id IN ({marks})", chunk)}
                done["turns"] += db.execute(f"DELETE FROM turns WHERE id IN ({marks})", chunk).rowcount
            for conv in convs:
                row = db.execute("SELECT summary_id FROM conversations WHERE id=?", (conv,)).fetchone()
                if row and row[0]:
                    sessions.add(int(row[0]))
                left = db.execute("SELECT count(*), max(id) FROM turns WHERE session_id=?", (conv,)).fetchone()
                if not left[0]:
                    db.execute("DELETE FROM conversations WHERE id=?", (conv,))
                else:
                    db.execute(
                        "UPDATE conversations SET summary_id=NULL, summarized_upto=0, turns=?, last_turn=? WHERE id=?",
                        (left[0], left[1], conv),
                    )
            for sid in sessions:
                done["sessions"] += db.execute("DELETE FROM sessions WHERE id=?", (sid,)).rowcount
                self._drop_vec("sessions_vec", sid)
                db.execute("UPDATE conversations SET summary_id=NULL, summarized_upto=0 WHERE summary_id=?", (sid,))
            db.commit()
        return done

    # ---- search ------------------------------------------------------------------------
    @staticmethod
    def _match(toks: list[str], *, every: bool = False) -> str:
        """An FTS5 query for these words: any of them (or `every` one), longer ones by prefix
        too ("allerg" finds "allergic"). Quoted, so no word is read as an FTS operator."""
        parts = ['"' + t.replace('"', "") + '"' + ("*" if len(t) >= 4 else "") for t in toks]
        return (" AND " if every else " OR ").join(parts)

    @staticmethod
    def _needs_substring(toks: list[str]) -> bool:
        """Words in a script whose words unicode61 may not split as people write them."""
        return any(not t.isascii() for t in toks)

    def _keyword(
        self, table: str, toks: list[str], limit: int, *, every: bool = False, since: float | None = None,
        until: float | None = None, fts: str = "", spoken_only: bool = True,
    ) -> list[int]:  # fmt: skip
        """Row ids of `table` matching the words, best (BM25) first. Under the read lock."""
        fts = fts or f"{table}_fts"
        if fts.endswith("_tri"):
            toks = [t for t in toks if len(t) >= 3]  # a trigram index can't see shorter ones
            if not toks:
                return []
            q = (" AND " if every else " OR ").join('"' + t.replace('"', "") + '"' for t in toks)
        else:
            q = self._match(toks, every=every)
        ts_col = {"turns": "t.ts", "sessions": "t.started"}.get(table)
        where, args = "", [q]
        if ts_col and since is not None:
            where += f" AND {ts_col} >= ?"
            args.append(since)
        if ts_col and until is not None:
            where += f" AND {ts_col} < ?"
            args.append(until)
        if table == "memory" and spoken_only:
            where += " AND t.superseded_by IS NULL"
        elif table == "turns" and spoken_only:
            # never a note of what NEO ran, nor a line of NEO's that quoted a page, a mail or a file
            where += " AND t.source != 'tool' AND NOT (t.role = 'assistant' AND t.tainted = 1)"
        try:
            rows = self._rdb.execute(
                f"SELECT t.id FROM {fts} f JOIN {table} t ON t.id = f.rowid WHERE {fts} MATCH ?{where} "
                f"ORDER BY f.rank LIMIT ?",
                (*args, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(r[0]) for r in rows]

    def _substring(
        self, table: str, col: str, toks: list[str], limit: int, *, every: bool = False, since: float | None = None,
        until: float | None = None, spoken_only: bool = True,
    ) -> list[int]:  # fmt: skip
        """Rows whose text contains the words anywhere inside a word: the trigram index, or LIKE
        for words too short for it (and where SQLite has no trigram tokenizer). Newest first."""
        if self.trigram and all(len(t) >= 3 for t in toks):
            return self._keyword(
                table, toks, limit, every=every, since=since, until=until, fts=f"{table}_tri", spoken_only=spoken_only
            )
        ts_col = {"turns": "ts", "sessions": "started"}.get(table)
        likes = [f"{col} LIKE ? ESCAPE '\\'" for _ in toks]
        args: list[Any] = ["%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in toks]
        where = "(" + (" AND " if every else " OR ").join(likes) + ")"
        if ts_col and since is not None:
            where += f" AND {ts_col} >= ?"
            args.append(since)
        if ts_col and until is not None:
            where += f" AND {ts_col} < ?"
            args.append(until)
        if table == "memory" and spoken_only:
            where += " AND superseded_by IS NULL"
        elif table == "turns" and spoken_only:
            where += " AND source != 'tool' AND NOT (role = 'assistant' AND tainted = 1)"
        rows = self._rdb.execute(f"SELECT id FROM {table} WHERE {where} ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [int(r[0]) for r in rows]

    def _weights(self, table: str, toks: list[str]) -> dict[str, float]:
        """How much each query word says in this table: a word few rows have (or none: the
        query asks for something that isn't here) weighs more than one most rows have. From the
        FTS index's own vocabulary. Under the read lock."""
        if not toks:
            return {}
        try:
            n = int(self._rdb.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0)
            out = {}
            for t in toks:
                r = self._rdb.execute(f"SELECT doc FROM {table}_vocab WHERE term = ?", (_stem(t),)).fetchone()
                out[t] = math.log(1.0 + n / max(float(r[0]) if r else 0.0, 0.5))
            return out
        except sqlite3.OperationalError:  # no vocabulary table here: every word weighs the same
            return dict.fromkeys(toks, 1.0)

    def _matching(self, table: str, toks: list[str], ids: list[int], marks: str) -> dict[int, set[str]]:
        """Which of the query words each candidate row has, as the index sees it (stemmed, by
        prefix): row id → words. Under the read lock."""
        out: dict[int, set[str]] = {}
        fts = f"{table}_fts"
        for t in toks:
            try:
                rows = self._rdb.execute(
                    f"SELECT rowid FROM {fts} WHERE {fts} MATCH ? AND rowid IN ({marks})", (self._match([t]), *ids)
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for (rid,) in rows:
                out.setdefault(int(rid), set()).add(t)
        return out

    def _query_vec(self, query: str, toks: list[str]) -> np.ndarray | None:
        """The query's vector — only once the model is loaded (never waiting for it) and only for
        a real query: one word is a keyword, not a meaning."""
        if len(toks) < 2:
            return None
        return self._embed(query)  # None until the model is loaded: embed_texts never loads it

    def search(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        k: int = 8,
        min_score: float = MIN_SCORE,
        now: float | None = None,
        touch: bool = True,
        asking: str = "",
    ) -> list[Hit]:
        """What NEO remembers about `query`, best first: facts, conversation summaries and
        journal lines (`kinds`: any of profile, lesson, note, summary, turn — or "fact" for the
        first three; default all). Keyword and meaning ranks are fused, then recency, importance
        and the kind decide; hits below `min_score` are left out.

        A time phrase in the query ("yesterday", "last week", "on Monday", "kal", "pichle
        hafte") — or `since` / `until` — limits summaries and journal lines to that window;
        facts have no date and are matched by the other words alone. A query that is only a
        time ("what did we talk about yesterday") lists that window's conversations.

        `asking`: the request being answered, when it isn't the query itself — its own line in
        the journal (written a moment ago) is the question, not a memory, and is left out."""
        now = time.time() if now is None else now
        own = {" ".join(x.split()).casefold() for x in (query, asking) if x and x.strip()}
        want = set(kinds) if kinds is not None else {*_FACT_KINDS, "summary", "turn"}
        if "fact" in want:
            want = (want - {"fact"}) | set(_FACT_KINDS)
        if since is None and until is None and (win := timephrase.find(query, datetime.fromtimestamp(now))) is not None:
            since, until, query = win.since, win.until, win.rest  # the time words are not what is searched for
        words = tokens(query)
        windowed = since is not None or until is not None
        qvec = self._query_vec(query, words)
        fact_kinds = [x for x in _FACT_KINDS if x in want]
        ranked: dict[tuple[str, int], list[float]] = {}  # (table, id) -> [rrf, similarity]

        def fuse(table: str, ids: list[int], sims: dict[int, float] | None = None) -> None:
            for rank, rid in enumerate(ids):
                slot = ranked.setdefault((table, rid), [0.0, 0.0])
                slot[0] += 1.0 / (10 + rank)
                if sims:
                    slot[1] = max(slot[1], sims.get(rid, 0.0))

        n = max(24, k * 4)
        listing = windowed and not words  # only a time was asked about: that window's conversations
        with self._rlock:
            if self._closed:
                return []
            if fact_kinds and words:
                ids = self._keyword("memory", words, n)
                if self._needs_substring(words) or not ids:
                    ids = list(dict.fromkeys(ids + self._substring("memory", "text", words, n)))
                fuse("memory", ids)
            for kind, table, col in (("summary", "sessions", "summary"), ("turn", "turns", "text")):
                if kind not in want:
                    continue
                if listing:
                    order = "started" if table == "sessions" else "id"
                    ts_col = "started" if table == "sessions" else "ts"
                    extra = "" if table == "sessions" else " AND source != 'tool' AND NOT (role='assistant' AND tainted=1)"
                    rows = self._rdb.execute(
                        f"SELECT id FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ?{extra} ORDER BY {order} DESC LIMIT ?",
                        (since if since is not None else 0.0, until if until is not None else now + 1, n),
                    ).fetchall()
                    fuse(table, [int(r[0]) for r in rows])
                elif words:
                    ids = self._keyword(table, words, n, since=since, until=until)
                    if self._needs_substring(words) or not ids:
                        ids = list(dict.fromkeys(ids + self._substring(table, col, words, n, since=since, until=until)))
                    fuse(table, ids)
        if qvec is not None and not listing:
            if fact_kinds:
                sem = [(i, s) for i, s in self._vec["memory_vec"].top(qvec, n) if s >= _SEM_MIN]
                fuse("memory", [i for i, _ in sem], dict(sem))
            if "summary" in want:
                sem = [(i, s) for i, s in self._vec["sessions_vec"].top(qvec, n) if s >= _SEM_MIN]
                fuse("sessions", [i for i, _ in sem], dict(sem))
        if not ranked:
            return []
        lists = 2 if qvec is not None else 1
        hits: list[Hit] = []
        with self._rlock:
            if self._closed:
                return []
            for table in ("memory", "sessions", "turns"):
                ids = [rid for (t, rid) in ranked if t == table]
                if not ids:
                    continue
                weight = self._weights(table, words)
                total = sum(weight.values()) or 1.0
                marks = ",".join("?" * len(ids))
                has = self._matching(table, words, ids, marks)
                if table == "memory":
                    sql = f"SELECT id, kind, text, updated, importance, key, '' FROM memory WHERE id IN ({marks}) AND superseded_by IS NULL"
                elif table == "sessions":
                    sql = f"SELECT id, 'summary', summary, started, 0.5, '', '' FROM sessions WHERE id IN ({marks})"
                else:
                    sql = f"SELECT id, 'turn', text, ts, 0.4, '', role FROM turns WHERE id IN ({marks})"
                for rid, kind, text, ts, imp, key, role in self._rdb.execute(sql, ids).fetchall():
                    if kind not in want or not text:
                        continue
                    if table == "turns" and text.casefold() in own:
                        continue  # the question itself
                    ts = float(ts or 0.0)
                    if table != "memory" and ((since is not None and ts < since) or (until is not None and ts >= until)):
                        continue
                    rrf, sim = ranked[(table, int(rid))]
                    low = f"{key or ''} {text}".casefold()
                    # How much of the query this row covers, each word weighed by how rare it is
                    # here: sharing "the report" with a question about penguins is no match.
                    mine = has.get(int(rid), ())
                    cover = (
                        sum(weight[w] for w in words if w in mine or w in low) / total
                        if words
                        else (1.0 if listing else 0.0)
                    )
                    about = max(cover, sim)
                    base = min(1.0, rrf / (lists * 0.1))
                    age_days = max(0.0, now - ts) / 86400.0
                    score = (
                        0.45 * base * (0.35 + 0.65 * about)
                        + 0.20 * about
                        + 0.12 * 0.5 ** (age_days / _HALF_LIFE_DAYS)
                        + 0.10 * float(imp or 0.5)
                        + _KIND_BOOST.get(kind, 0.0)
                    )
                    if score >= min_score:
                        hits.append(Hit(table, int(rid), kind, text, ts, round(score, 4), key or "", role or ""))
        if listing:
            # the window's conversations first (newest first), then its latest lines in the order said
            hits.sort(key=lambda h: (h.kind != "summary", -h.ts))
            hits = hits[:k]
            hits.sort(key=lambda h: (h.kind != "summary", -h.ts if h.kind == "summary" else h.ts))
        else:
            hits.sort(key=lambda h: (-h.score, -h.ts))
            hits = hits[:k]
        if touch and (ids := [h.id for h in hits if h.table == "memory"]):
            self.touch_later(ids)
        return hits

    # ---- prompt --------------------------------------------------------------------------
    def context(
        self,
        query: str = "",
        *,
        turns: int = 12,
        budget: int = 1800,
        not_after: float | None = None,
        now: float | None = None,
    ) -> str:
        """What a model is told from memory for one request, within `budget` characters:

        * the live profile facts and the lessons — always;
        * the last three conversation summaries;
        * the last `turns` lines said today (so a new "Neo" a moment later continues the same
          conversation). `not_after`: leave out lines NEO's own session already holds — those
          it journaled itself since that moment (the caller sends them as the conversation);
        * the best search hits for `query` that aren't in the lines above.

        A line of NEO's that quoted a page, a mail or a file is never repeated here (this goes
        into a system prompt). It goes to the cloud models with each request, so NEO_CLOUD_MEMORY
        limits it: "profile" sends only the profile facts, "off" nothing."""
        s = settings()
        level = s.cloud_memory
        if level == "off" or self._closed:
            return ""
        now = time.time() if now is None else now
        prof = self.by_kind("profile")
        head = ("About the user: " + "; ".join(m.label for m in prof))[:600] if prof else ""
        if level == "profile":
            return head
        parts: list[str] = [head] if head else []
        shown_facts = {m.id for m in prof}
        les = self.by_kind("lesson", 15)
        if les:
            parts.append(("Lessons: " + " | ".join(m.text for m in les))[:300])
            shown_facts |= {m.id for m in les}
        sess = self.recent_sessions(3)
        if sess:
            parts.append(
                "Recent sessions: "
                + " | ".join(f"{time.strftime('%a %d %b', time.localtime(t))}: {x[:170]}" for t, x in sess)
            )
        day = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        today = self.recent_turns(since=min(day, now - 3 * 3600), limit=turns * 2 if not_after is not None else turns)
        if not_after is not None:
            today = [t for t in today if t.ts < not_after or t.route == "live"][-turns:]
        if query:
            shown_turns = {t.id for t in today}
            texts = {x for _, x in sess}
            hits = [
                h
                for h in self.search(query, k=max(1, int(getattr(s, "memory_recall_k", 6))), now=now, touch=False)
                if not (h.table == "memory" and h.id in shown_facts)
                and not (h.table == "turns" and h.id in shown_turns)
                and not (h.table == "sessions" and h.text in texts)
            ]
            if hits:
                related = "Related, from a search of your memory for this request: " + " | ".join(
                    snippet(h, 160) for h in hits
                )
                parts.append(related[:520])
        used = sum(len(p) + 1 for p in parts) + len(_MORE) + 1
        lines = [_turn_line(t) for t in today]
        head_line = "Earlier today (oldest first — what was said, for context):"
        while lines and used + len(head_line) + sum(len(x) + 1 for x in lines) > budget:
            lines.pop(0)  # the oldest lines go first
        if lines:
            parts.append(head_line + "\n" + "\n".join(lines))
        if parts:
            parts.append(_MORE)
        return "\n".join(parts)[:budget]

    def prompt_context(self, query: str = "") -> str:
        """:meth:`context` under its older name."""
        return self.context(query)

    def count(self) -> int:
        with self._rlock:
            return int(self._rdb.execute("SELECT count(*) FROM memory").fetchone()[0])


# The last line of every memory block: what is shown is not all there is.
_MORE = "(More is kept than is shown here: search your memory before saying you don't remember something.)"


def _turn_line(t: Turn, width: int = 150) -> str:
    who = "user" if t.role == "user" else "NEO"
    if t.role != "user" and t.tainted:
        text = "(read out what a page, a mail or a file said)"
    else:
        text = t.text if len(t.text) <= width else t.text[: width - 1].rstrip() + "…"
    return f"{time.strftime('%H:%M', time.localtime(t.ts))} {who}: {text}"


def snippet(h: Hit, width: int = 200) -> str:
    """A hit as one short, dated line: "[fact, 30 Sep] sister name: Priya"."""
    day = time.strftime("%a %d %b", time.localtime(h.ts)) if h.ts else ""
    if h.table == "memory":
        what, text = "fact", fact_label(h.key, h.text)
    elif h.table == "sessions":
        what, text = "conversation", h.text
    else:
        what, text = ("you said" if h.role == "user" else "NEO said"), h.text
    if len(text) > width:
        text = text[: width - 1].rstrip() + "…"
    return f"[{what}, {day}] {text}" if day else f"[{what}] {text}"


def _short_args(args: dict[str, Any]) -> str:
    bits = []
    for k, v in list(args.items())[:3]:
        if k in ("confirm", "cancel"):
            continue
        s = str(v)
        bits.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={s!r}")
    return ", ".join(bits)


_store: Store | None = None


def store() -> Store:
    """The bound agent's memory (neo2.core.runtime), else NEO.app's. An agent without memory has none:
    MemoryDisabled."""
    rt = runtime.current_runtime()
    if rt is not None:
        if not rt.settings.memory_enabled:
            raise MemoryDisabled(f"memory is off for the agent {rt.name!r}")
        return rt.get_or_create("store", lambda: Store(rt.settings.memory_db, agent=rt.library))
    runtime.check_unbound("store()")
    global _store
    if _store is None:
        _store = Store()
    return _store
