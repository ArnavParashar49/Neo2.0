"""Local long-term memory: SQLite + FTS5 keyword search + local embeddings (hybrid recall).

Kinds of memory rows:
- profile  — stable facts about the user (name, city, preferences)  → always in the prompt
- lesson   — corrections / how-to-behave feedback                     → always in the prompt
- note     — everything else (facts, events, task outcomes)           → recalled by search

Also here:
- playbooks — tool sequences from successful multi-step tasks, recalled by goal similarity so a
  repeat request follows a known path instead of exploring again
- sessions  — short summaries written at shutdown so "what did we do yesterday" works
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neo.config import settings
from neo.memory import embed as _emb

Kind = Literal["profile", "lesson", "note"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  key TEXT,
  text TEXT NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  uses INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, key, content='memory', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text, key) VALUES('delete', old.id, old.text, old.key);
  INSERT INTO memory_fts(rowid, text, key) VALUES (new.id, new.text, new.key);
END;
CREATE TABLE IF NOT EXISTS memory_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS playbook (
  id INTEGER PRIMARY KEY,
  goal TEXT NOT NULL,
  steps TEXT NOT NULL,
  answer TEXT,
  created REAL NOT NULL,
  uses INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS playbook_vec (id INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY, started REAL, ended REAL, summary TEXT
);
"""


@dataclass
class Memory:
    id: int
    kind: str
    key: str
    text: str
    created: float


@dataclass
class Playbook:
    id: int
    goal: str
    steps: list[dict[str, Any]]
    answer: str
    created: float
    similarity: float = 0.0


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings().memory_db
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript(_SCHEMA)

    # ---- vectors ---------------------------------------------------------------------
    def _put_vec(self, table: str, row_id: int, text: str) -> None:
        v = _emb.embed_texts([text])
        if v is None:
            return
        self._db.execute(f"INSERT OR REPLACE INTO {table}(id, vec) VALUES (?, ?)", (row_id, v[0].tobytes()))

    def _load_vecs(self, table: str) -> tuple[list[int], np.ndarray]:
        rows = self._db.execute(f"SELECT id, vec FROM {table}").fetchall()
        if not rows:
            return [], np.zeros((0, _emb.DIM), dtype=np.float32)
        ids, vecs = [], []
        for r in rows:
            v = np.frombuffer(r[1], dtype=np.float32)
            if v.shape[0] == _emb.DIM:  # rows written by a different embedding model are ignored
                ids.append(int(r[0]))
                vecs.append(v)
        if not vecs:
            return [], np.zeros((0, _emb.DIM), dtype=np.float32)
        return ids, np.vstack(vecs)

    def _semantic(self, table: str, query: str, k: int) -> list[tuple[int, float]]:
        q = _emb.embed_texts([query])
        if q is None:
            return []
        ids, mat = self._load_vecs(table)
        return [(ids[i], s) for i, s in _emb.cosine_top(q[0], mat, k)]

    # ---- write -----------------------------------------------------------------------
    def remember(self, text: str, kind: Kind = "note", key: str = "") -> int:
        now = time.time()
        text = text.strip()
        if key:  # keyed facts upsert (e.g. profile.city)
            row = self._db.execute("SELECT id FROM memory WHERE kind=? AND key=?", (kind, key)).fetchone()
            if row:
                self._db.execute("UPDATE memory SET text=?, updated=? WHERE id=?", (text, now, row[0]))
                self._put_vec("memory_vec", int(row[0]), text)
                self._db.commit()
                return int(row[0])
        dup = self._db.execute(
            "SELECT id FROM memory WHERE kind=? AND lower(text)=lower(?)", (kind, text)
        ).fetchone()
        if dup:
            return int(dup[0])
        cur = self._db.execute(
            "INSERT INTO memory(kind,key,text,created,updated) VALUES (?,?,?,?,?)",
            (kind, key, text, now, now),
        )
        mid = int(cur.lastrowid or 0)
        self._put_vec("memory_vec", mid, text)
        self._db.commit()
        return mid

    def forget(self, memory_id: int) -> bool:
        n = self._db.execute("DELETE FROM memory WHERE id=?", (memory_id,)).rowcount
        self._db.execute("DELETE FROM memory_vec WHERE id=?", (memory_id,))
        self._db.commit()
        return n > 0

    # ---- read ------------------------------------------------------------------------
    def _keyword(self, query: str, limit: int) -> list[int]:
        toks = [t for t in _tokens(query) if len(t) > 2]
        if not toks:
            return []
        q = " OR ".join('"' + t + '"' for t in toks)  # quoted → no FTS5 operators / reserved words
        try:
            rows = self._db.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts) LIMIT ?",
                (q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(r[0]) for r in rows]

    def recall(self, query: str, limit: int = 8) -> list[Memory]:
        """Hybrid recall: keyword (BM25) and semantic ranks fused (reciprocal-rank fusion)."""
        kw = self._keyword(query, limit * 2)
        # The semantic path only runs for real queries and only keeps clearly related rows.
        sem = []
        if len([t for t in _tokens(query) if len(t) > 2]) >= 2:
            sem = [i for i, s in self._semantic("memory_vec", query, limit * 2) if s >= 0.4]
        score: dict[int, float] = {}
        for rank, mid in enumerate(kw):
            score[mid] = score.get(mid, 0.0) + 1.0 / (10 + rank)
        for rank, mid in enumerate(sem):
            score[mid] = score.get(mid, 0.0) + 1.0 / (10 + rank)
        ids = [mid for mid, _ in sorted(score.items(), key=lambda kv: -kv[1])[:limit]]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT id, kind, key, text, created FROM memory WHERE id IN ({marks})", ids
        ).fetchall()
        by_id = {int(r[0]): Memory(*r) for r in rows}
        self._db.execute(f"UPDATE memory SET uses=uses+1 WHERE id IN ({marks})", ids)
        self._db.commit()
        return [by_id[i] for i in ids if i in by_id]

    def by_kind(self, kind: Kind, limit: int = 40) -> list[Memory]:
        rows = self._db.execute(
            "SELECT id, kind, key, text, created FROM memory WHERE kind=? ORDER BY updated DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [Memory(*r) for r in rows]

    # ---- playbooks ---------------------------------------------------------------------
    def save_playbook(self, goal: str, steps: list[dict[str, Any]], answer: str) -> int:
        """Store the tool sequence of a successful task. Near-duplicate goals replace the old entry."""
        goal = goal.strip()
        for pid, sim in self._semantic("playbook_vec", goal, 1):
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
        self._put_vec("playbook_vec", pid, goal)
        self._db.commit()
        return pid

    def similar_playbooks(self, goal: str, k: int = 2, min_sim: float = 0.6) -> list[Playbook]:
        hits = [(pid, s) for pid, s in self._semantic("playbook_vec", goal, k) if s >= min_sim]
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

    # ---- sessions ----------------------------------------------------------------------
    def add_session_summary(self, summary: str, started: float, ended: float | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO sessions(started, ended, summary) VALUES (?,?,?)",
            (started, ended or time.time(), summary.strip()),
        )
        self._db.commit()
        return int(cur.lastrowid or 0)

    def recent_sessions(self, n: int = 3) -> list[tuple[float, str]]:
        rows = self._db.execute(
            "SELECT started, summary FROM sessions ORDER BY started DESC LIMIT ?", (n,)
        ).fetchall()
        return [(float(r[0]), r[1]) for r in rows if r[1]]

    # ---- prompt --------------------------------------------------------------------------
    def prompt_context(self, query: str = "") -> str:
        """Profile + lessons always; recent sessions; relevant notes when a query is given."""
        parts: list[str] = []
        prof = self.by_kind("profile")
        if prof:
            parts.append(
                "About the user: " + "; ".join((f"{m.key}: " if m.key else "") + m.text for m in prof)
            )
        les = self.by_kind("lesson", 15)
        if les:
            parts.append("Lessons: " + " | ".join(m.text for m in les))
        sess = self.recent_sessions(2)
        if sess:
            parts.append(
                "Recent sessions: "
                + " | ".join(f"{time.strftime('%a %d %b', time.localtime(t))}: {s}" for t, s in sess)
            )
        if query:
            notes = [m for m in self.recall(query, 6) if m.kind == "note"]
            if notes:
                parts.append("Relevant notes: " + " | ".join(m.text for m in notes))
        return "\n".join(parts)

    def count(self) -> int:
        return int(self._db.execute("SELECT count(*) FROM memory").fetchone()[0])


def _short_args(args: dict[str, Any]) -> str:
    bits = []
    for k, v in list(args.items())[:3]:
        if k in ("confirm", "cancel"):
            continue
        s = str(v)
        bits.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={s!r}")
    return ", ".join(bits)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())  # no apostrophes: they're FTS5 string delimiters


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
