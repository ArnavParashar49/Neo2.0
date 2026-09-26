"""Instant correction memory: a fix takes effect on the very next request.

A retrain waits for dozens of new rows and a few idle minutes. A verified correction ("no, I
meant…" followed by something that worked) is also kept here, as the reflex's own embedding of
the corrected words, and when the same request comes back its label beats the head's pick.

Laya's mean-pooled embeddings all point roughly one way (random pairs sit at cosine ~0.83), so a
raw-cosine match fires on everything. Centred on the head's mean vector, random pairs drop to
~0.02 and τ=0.9 only fires on real repeats (measured precision 1.00 on the user's own rows, 0.4%
of unrelated ones). It fixes repeats; generalising is the next retrain's job. It never overrides
a head that is ≥0.9 sure of something else, nor across scripts (Devanagari vs Latin).

RAM only: embeddings never touch disk — they give the words back (Vec2Text recovers most short
inputs). After a restart it is rebuilt from the stored corrections, re-embedded in the
background while NEO is idle with the Laya model already loaded.

Interface with the reflex: after ``LayaReflex.decide(text)`` the instance exposes
``last_embedding`` (float32, L2-normalised embedding of that text) and ``center`` (float32, the
head's mean embedding). Without them this module does nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from neo.reflex import learn
from neo.reflex.schema import Decision

TAU = 0.9  # centred cosine for "the same request again"
HEAD_SURE = 0.9  # a head this sure of something else keeps its pick
CAP = 2000
_RECENT = 64  # embeddings of the latest decisions, so a correction can be remembered at once


@dataclass
class _Item:
    key: str
    intent: str
    tool: str
    script: str


def _key(text: str) -> str:
    """A sentence's identity without keeping its words: a hash of what learn stores for it."""
    return hashlib.sha256(learn.norm(learn.skeleton(text) or text).encode()).hexdigest()[:20]


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


class Memory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[_Item] = []
        self._vecs: list[np.ndarray] = []  # the reflex's embeddings, as given
        self._mat: np.ndarray | None = None  # centred, unit rows — rebuilt when the centre moves
        self._center: np.ndarray | None = None
        self._recent: OrderedDict[str, np.ndarray] = OrderedDict()
        self._stale = True  # the stored corrections aren't all in RAM yet (e.g. after a restart)
        self._gen = 0  # bumped by forget(): a rebuild that started before it throws its work away
        self._rebuilding = False

    # ---- what the reflex sees -----------------------------------------------------------------
    def seen(self, text: str, emb: np.ndarray) -> None:
        with self._lock:
            k = _key(text)
            self._recent.pop(k, None)
            self._recent[k] = np.array(emb, dtype=np.float32).reshape(-1)
            while len(self._recent) > _RECENT:
                self._recent.popitem(last=False)

    def match(self, text: str, emb: np.ndarray, center: np.ndarray) -> tuple[_Item, float] | None:
        """The remembered correction this request repeats, if any (same script, cosine ≥ TAU)."""
        with self._lock:
            if not self._items:
                return None
            mat = self._centred(center)
            if mat is None:
                return None
            q = _unit(np.asarray(emb, dtype=np.float32).reshape(-1) - self._center)
            if q is None or q.shape[0] != mat.shape[1]:
                return None
            sims = mat @ q
            j = int(sims.argmax())
            item, sim = self._items[j], float(sims[j])
        if sim < TAU or item.script != learn.script(text):
            return None
        return item, sim

    def _centred(self, center: np.ndarray) -> np.ndarray | None:
        c = np.asarray(center, dtype=np.float32).reshape(-1)
        if self._mat is None or self._center is None or (c is not self._center and not np.array_equal(c, self._center)):
            if any(v.shape != c.shape for v in self._vecs):
                return None  # a head for another encoder: these vectors don't compare
            rows = np.stack(self._vecs) - c
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            self._mat, self._center = (rows / np.maximum(norms, 1e-8)).astype(np.float32), c
        return self._mat

    # ---- what it remembers --------------------------------------------------------------------
    def learned(self, text: str, intent: str, tool: str) -> bool:
        """A verified correction was just stored: remember it now if its embedding is at hand
        (the reflex decided these words a moment ago), else at the next idle rebuild."""
        with self._lock:
            emb = self._recent.get(_key(text))
            if emb is None:
                self._stale = True
                return False
            self._put(_key(text), emb, intent, tool, learn.script(text))
            return True

    def _put(self, key: str, emb: np.ndarray, intent: str, tool: str, script: str) -> None:
        item = _Item(key, intent, tool if intent == "quick_action" else "", script)
        for i, it in enumerate(self._items):
            if it.key == key:
                del self._items[i], self._vecs[i]
                break
        self._items.append(item)
        self._vecs.append(np.array(emb, dtype=np.float32).reshape(-1))
        if len(self._items) > CAP:
            del self._items[0], self._vecs[0]
        self._mat = None

    @property
    def stale(self) -> bool:
        return self._stale

    def __len__(self) -> int:
        return len(self._items)

    def forget(self) -> None:
        with self._lock:
            self._gen += 1
            self._items, self._vecs, self._mat, self._center = [], [], None, None
            self._recent.clear()
            self._stale = True

    def rebuild(self, embed: Callable[[str], np.ndarray | None], keep_going: Callable[[], bool] = lambda: True) -> int:
        """Re-embed stored corrections not in RAM yet (the newest CAP) and drop ones no longer
        stored. Stops early — staying stale — when keep_going() turns False. Returns how many
        were added."""
        rows = learn.corrections()[-CAP:]
        with self._lock:
            gen, have = self._gen, {it.key for it in self._items}
        wanted = {_key(r["text"]): r for r in rows}
        new: list[tuple[str, np.ndarray, dict]] = []
        for k, r in wanted.items():
            if k in have:
                continue
            if not keep_going():
                return 0
            e = embed(r["text"])
            if e is None:
                break  # this reflex exposes no embedding: nothing to remember with
            new.append((k, e, r))
        with self._lock:
            if gen != self._gen:
                return 0  # erased while embedding
            keep = [i for i, it in enumerate(self._items) if it.key in wanted]
            self._items = [self._items[i] for i in keep]
            self._vecs = [self._vecs[i] for i in keep]
            self._mat = None
            for k, e, r in new:
                self._put(k, e, r["intent"], r.get("tool", ""), r.get("script") or learn.script(r["text"]))
            self._stale = False
        return len(new)


_memory = Memory()


def memory() -> Memory:
    return _memory


def forget() -> None:
    _memory.forget()


def recall(d: Decision, text: str, emb, center) -> Decision:
    """The head's decision, or the remembered correction when this request repeats one."""
    if emb is None or center is None or not learn.enabled() or d.intent == "stop":
        return d
    try:
        _memory.seen(text, emb)
        hit = _memory.match(text, emb, center)
        if hit is None:
            return d
        item, sim = hit
        if item.intent == d.intent and (not item.tool or item.tool == d.tool):
            return d  # the head already agrees
        if item.intent != d.intent and d.intent_confidence >= HEAD_SURE:
            return d  # sure of something else: a near-repeat isn't enough to overrule that
        if item.intent == d.intent and d.tool_p >= HEAD_SURE:
            return d
        learn.count("memory_hits")
        return dataclasses.replace(
            d,
            intent=item.intent,  # type: ignore[arg-type]
            intent_confidence=max(sim, d.intent_confidence if item.intent == d.intent else 0.0),
            tool=item.tool,
            tool_p=sim if item.tool else 0.0,
            source="laya+memory",
        )
    except Exception as e:  # noqa: BLE001 — the memory must never break a decision
        print(f"[reflex] correction memory skipped: {str(e)[:80]}")
        return d


def rebuild_soon(is_idle: Callable[[], bool]) -> bool:
    """Start a background rebuild when the memory is behind the stored corrections, Laya is
    already loaded (never load it just for this) and NEO is idle."""
    from neo import reflex

    laya = reflex._laya
    if laya is None or not _memory.stale or _memory._rebuilding or not learn.enabled():
        return False

    def embed(text: str) -> np.ndarray | None:
        with reflex.laya_lock():
            laya.decide(text)
            e = getattr(laya, "last_embedding", None)
            return None if e is None else np.array(e, dtype=np.float32).reshape(-1)

    def run() -> None:
        try:
            n = _memory.rebuild(embed, is_idle)
            if n:
                print(f"[reflex] correction memory: {len(_memory)} remembered")
        except Exception as e:  # noqa: BLE001
            print(f"[reflex] correction memory rebuild failed: {str(e)[:80]}")
        finally:
            _memory._rebuilding = False

    _memory._rebuilding = True
    threading.Thread(target=run, daemon=True, name="laya-memory").start()
    return True
