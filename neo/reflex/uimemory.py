"""Per-app memory of UI flows that worked, so a request that worked once replays without a model.

A *recipe* is the steps one request took in one app — press "New Game" in the menu bar, then
"Start" in the sheet it opened — with what each step was seen to do (a sheet with these buttons
opened, the sheet closed, the title changed, the focus moved). It is kept only when the outcome
ledger (neo.reflex.outcomes) settles that request as verified: every step was seen to change
something (an unknown effect is no recipe), the user didn't stop it, correct it or decline a
question.

**Key.** The owning app's bundle id (evidence from the element's process, never a name), the
script (Devanagari or Latin) and the request's reflex embedding, centred on the head's mean as
neo.reflex.memory does (τ = 0.9: the user's own repeats, not look-alikes). Without an embedding at
hand only the same words (skeleton, normalised) match.

**Replay** (:func:`replay`) runs the steps through callables the caller gives it — the tool
handlers, so every press still passes the gate and the protected-app rules — and checks each
step's effect. At the first mismatch it stops and hands over "did X, expected Y, saw Z"; two
failed replays in a row retire the recipe until it works again from the model's run. A recipe is
recalled only for the user's own words (``origin == "user"``): a model's or Live's never replay one.

**Aliases.** When the model chose a control from the resolver's shortlist and that press was
verified, the wording (stripped of its click frame) is remembered for that app: next time the
resolver's alias stage finds it without the model.

**Privacy.** Only skeleton text (learn.skeleton: never a message's or a note's body), bundle ids and
UI chrome labels (buttons, menu items, tabs…, never rows, cells, static text or field values) are
written, to ``ui_memory.json`` (0600). What a step saw is kept as facts: that the title or the
focus changed (never the title or the focused text: a document's name, a message), whether a sheet
opened or closed, and a sheet's controls by role with the chrome ones' labels only. Embeddings stay in RAM (rebuilt at idle, as memory.py
does). NEO_LEARN=off turns it off; the privacy erase deletes it; recipes unused for 180 days go;
at most 500. Not in library mode.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neo import runtime
from neo.reflex import learn

log = logging.getLogger(__name__)

__all__ = [
    "CAP",
    "KEEP_S",
    "TAU",
    "Recipe",
    "Replayed",
    "Step",
    "UIMemory",
    "alias",
    "forget",
    "learn_alias",
    "memory",
    "recall",
    "remember",
    "rebuild_soon",
    "replay",
    "settled",
]

TAU = 0.9  # centred cosine for "the same request again" (memory.py's measured bar)
CAP = 500  # recipes kept
ALIAS_CAP = 1000
KEEP_S = 180 * 86400.0
DEMOTE_AFTER = 2  # failed replays in a row
FILE = "ui_memory.json"
_RECENT = 64
_KINDS = ("menu", "press", "key")
# What a step's label may be: an app's own chrome (the resolver's CHROME_ROLES).
_CHROME = frozenset(
    {"AXButton", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
     "AXMenuButton", "AXDisclosureTriangle", "AXSegment", "AXToolbarButton"}
)  # fmt: skip
_EXPECT = ("sheet", "sheet_closed", "title", "focus", "changed")
_FLAGS = ("sheet_closed", "changed")  # facts: kept as booleans
_CHANGED = ("title", "focus")  # kept only as "it changed" (their text can be the user's content)
_SHEET_MAX = 12


@dataclass
class Step:
    """One step of a flow: a menu item (``path``: menu titles, ending with the item), a press of a
    control (``role``, ``label``), or a key chord (``keys``); and what it did (``expect``)."""

    kind: str
    role: str = ""
    label: str = ""
    path: tuple[str, ...] = ()
    keys: str = ""
    expect: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "menu":
            return " › ".join(self.path) or self.label
        if self.kind == "key":
            return self.keys
        return f"“{self.label}”"

    @classmethod
    def of(cls, d: dict[str, Any]) -> Step | None:
        """A step from a signature dict; None when it isn't one this memory may keep (a label that
        isn't UI chrome, a text step, a coordinate)."""
        kind = str(d.get("kind", ""))
        if kind not in _KINDS:
            return None
        role = str(d.get("role", ""))
        label = " ".join(str(d.get("label", "")).split())[:80]
        path = tuple(" ".join(str(p).split())[:80] for p in (d.get("path") or ()))
        keys = str(d.get("keys", ""))[:40]
        if kind == "press" and (role not in _CHROME or not label):
            return None
        if kind == "menu" and not (path or label):
            return None
        if kind == "key" and not keys:
            return None
        exp = {k: _fact(k, v) for k, v in (d.get("expect") or {}).items() if k in _EXPECT}
        return cls(kind, role, label, path, keys, {k: v for k, v in exp.items() if v is not None})


def _fact(key: str, v: Any) -> Any:
    """What a step saw, as it may be kept: booleans for the flags; "it changed" for a title or the
    focus (their text is never kept); a sheet's controls by role, with a label only on UI chrome
    (a plain string can't say what it is: only that something was there)."""
    if key in _FLAGS:
        return bool(v) if isinstance(v, (bool, int)) else None
    if key in _CHANGED:
        if isinstance(v, bool):
            return v
        return True if isinstance(v, str) and v.strip() else None
    if key == "sheet":
        if isinstance(v, bool):
            return v
        if not isinstance(v, (list, tuple)):
            return None
        out: list[dict[str, str]] = []
        for x in list(v)[:_SHEET_MAX]:
            role = str(x.get("role", "") or "")[:40] if isinstance(x, dict) else ""
            label = " ".join(str(x.get("label", "") or "").split())[:40] if isinstance(x, dict) else ""
            out.append({"role": role, "label": label} if role in _CHROME and label else {"role": role})
        return out
    return None


@dataclass
class Recipe:
    key: str
    bundle: str
    script: str
    skeleton: str
    steps: list[Step]
    ok: int = 1
    bad: int = 0
    streak: int = 0  # failed replays in a row
    made: float = 0.0
    used: float = 0.0

    @property
    def retired(self) -> bool:
        return self.streak >= DEMOTE_AFTER

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [{k: v for k, v in asdict(s).items() if v not in ("", (), {}, [])} for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Recipe | None:
        try:
            steps = [s for s in (Step.of(x) for x in d.get("steps", [])) if s is not None]
            if not steps:
                return None
            return cls(
                str(d["key"]), str(d["bundle"]), str(d.get("script", "latin")), str(d["skeleton"]), steps,
                int(d.get("ok", 1)), int(d.get("bad", 0)), int(d.get("streak", 0)), float(d.get("made", 0)),
                float(d.get("used", 0)),
            )  # fmt: skip
        except (KeyError, TypeError, ValueError):
            return None


def _norm_words(text: str) -> str:
    return learn.norm(text)


def _key(bundle: str, skeleton: str) -> str:
    return hashlib.sha256(f"{bundle}\n{_norm_words(skeleton)}".encode()).hexdigest()[:20]


def _text_key(text: str) -> str:
    return hashlib.sha256(_norm_words(learn.skeleton(text) or text).encode()).hexdigest()[:20]


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def _active() -> bool:
    return learn.enabled() and not runtime.library_bound()


class UIMemory:
    def __init__(self, path: Callable[[], Path] | None = None, clock: Callable[[], float] = time.time) -> None:
        self._path = path or (lambda: learn._path(FILE))
        self._clock = clock
        self._lock = threading.RLock()
        self._recipes: dict[str, Recipe] | None = None  # loaded on first use
        self._from: Path | None = None  # the file they were loaded from (a new data dir reloads)
        self._aliases: dict[str, dict[str, Any]] = {}
        self._vecs: dict[str, np.ndarray] = {}  # recipe key -> the reflex embedding (RAM only)
        self._recent: OrderedDict[str, np.ndarray] = OrderedDict()
        self._gen = 0

    # ---- disk -----------------------------------------------------------------------------------
    def _load(self) -> dict[str, Recipe]:
        where = self._path()
        if self._recipes is not None and where == self._from:
            return self._recipes
        if where != self._from:
            self._vecs.clear()
        self._from = where
        recipes: dict[str, Recipe] = {}
        aliases: dict[str, dict[str, Any]] = {}
        try:
            data = json.loads(where.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        now = self._clock()
        for d in data.get("recipes", []) if isinstance(data, dict) else []:
            r = Recipe.from_json(d) if isinstance(d, dict) else None
            if r is not None and now - max(r.used, r.made) < KEEP_S:
                recipes[r.key] = r
        for k, a in (data.get("aliases", {}) if isinstance(data, dict) else {}).items():
            if isinstance(a, dict) and now - float(a.get("used", 0)) < KEEP_S and a.get("label"):
                aliases[str(k)] = a
        self._recipes, self._aliases = recipes, aliases
        return recipes

    def _save(self) -> None:
        recipes = self._load()
        if len(recipes) > CAP:
            for k, _ in sorted(recipes.items(), key=lambda kv: max(kv[1].used, kv[1].made))[: len(recipes) - CAP]:
                recipes.pop(k, None)
                self._vecs.pop(k, None)
        if len(self._aliases) > ALIAS_CAP:
            for k, _ in sorted(self._aliases.items(), key=lambda kv: kv[1].get("used", 0))[: len(self._aliases) - ALIAS_CAP]:
                self._aliases.pop(k, None)
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        body = json.dumps({"recipes": [r.to_json() for r in recipes.values()], "aliases": self._aliases}, ensure_ascii=False)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, p)

    # ---- what the reflex sees ---------------------------------------------------------------------
    def seen(self, text: str, emb: Any) -> None:
        """The reflex's embedding of a request it just decided (kept for a moment, in RAM, so the
        flow can be keyed by it when the request settles as verified)."""
        if emb is None:
            return
        with self._lock:
            k = _text_key(text)
            self._recent.pop(k, None)
            self._recent[k] = np.array(emb, dtype=np.float32).reshape(-1)
            while len(self._recent) > _RECENT:
                self._recent.popitem(last=False)

    # ---- learning -------------------------------------------------------------------------------------
    def remember(self, text: str, bundle: str, steps: list[dict[str, Any]] | list[Step]) -> Recipe | None:
        """A verified flow for this request in this app (bundle id). Refused: learning off, library
        mode, no bundle id, words that aren't a keepable skeleton, or any step it may not keep."""
        if not _active() or not bundle:
            return None
        skel = learn.skeleton(text)
        if not skel:
            return None
        got = [s if isinstance(s, Step) else Step.of(s) for s in steps]
        if not got or any(s is None for s in got):
            return None  # one step it can't keep (a text, a coordinate): the flow isn't replayable
        with self._lock:
            recipes = self._load()
            k = _key(bundle, skel)
            now = self._clock()
            old = recipes.get(k)
            if old is not None and [asdict(s) for s in old.steps] == [asdict(s) for s in got]:
                old.ok, old.streak, old.used = old.ok + 1, 0, now
                r = old
            else:
                r = Recipe(k, bundle, learn.script(text), skel, got, made=now, used=now)  # type: ignore[arg-type]
                recipes[k] = r
            emb = self._recent.get(_text_key(text))
            if emb is not None:
                self._vecs[k] = emb
            self._save()
            return r

    def learn_alias(self, text: str, bundle: str, label: str) -> bool:
        """The model chose ``label`` from the resolver's shortlist for this wording, in this app, and
        the press was verified: next time the resolver's alias stage finds it."""
        if not _active() or not bundle or not label:
            return False
        phrase = _alias_phrase(text)
        if not phrase:
            return False
        with self._lock:
            self._load()
            self._aliases[_key(bundle, phrase)] = {"label": " ".join(label.split())[:80], "used": self._clock()}
            self._save()
        return True

    def alias(self, text: str, bundle: str) -> str | None:
        if not _active() or not bundle:
            return None
        phrase = _alias_phrase(text)
        if not phrase:
            return None
        with self._lock:
            self._load()
            a = self._aliases.get(_key(bundle, phrase))
            return str(a["label"]) if a else None

    # ---- recall ---------------------------------------------------------------------------------------
    def recall(self, text: str, bundle: str, emb: Any = None, center: Any = None, *, origin: str = "") -> Recipe | None:
        """This app's recipe for this request: same bundle and script, and the centred reflex
        embedding at τ (or, without one, the same words). Never a retired recipe, and only for the
        user's own words (``origin == "user"``): a replay presses without a model, so a model's,
        Live's or an approval's words never get one."""
        if origin != "user" or not _active() or not bundle:
            return None
        with self._lock:
            recipes = [r for r in self._load().values() if r.bundle == bundle and not r.retired and r.script == learn.script(text)]
            if not recipes:
                return None
            skel = learn.skeleton(text) or text
            same = next((r for r in recipes if _norm_words(r.skeleton) == _norm_words(skel)), None)
            if emb is None or center is None:
                return same
            self.seen(text, emb)
            c = np.asarray(center, dtype=np.float32).reshape(-1)
            q = _unit(np.asarray(emb, dtype=np.float32).reshape(-1) - c)
            if q is None:
                return same
            best, best_s = None, -1.0
            for r in recipes:
                v = self._vecs.get(r.key)
                if v is None or v.shape != c.shape:
                    continue
                u = _unit(v - c)
                if u is None:
                    continue
                s = float(u @ q)
                if s > best_s:
                    best, best_s = r, s
            if best is not None and best_s >= TAU:
                return best
            return same

    def result(self, key: str, ok: bool) -> None:
        """How a replay of this recipe went."""
        with self._lock:
            r = self._load().get(key)
            if r is None:
                return
            r.used = self._clock()
            if ok:
                r.ok, r.streak = r.ok + 1, 0
            else:
                r.bad, r.streak = r.bad + 1, r.streak + 1
            with contextlib.suppress(OSError):
                self._save()

    # ---- upkeep ------------------------------------------------------------------------------------------
    def stale(self) -> list[Recipe]:
        """Recipes with no embedding in RAM yet (after a restart): what :meth:`rebuild` re-embeds."""
        with self._lock:
            return [r for r in self._load().values() if r.key not in self._vecs]

    def rebuild(self, embed: Callable[[str], Any], keep_going: Callable[[], bool] = lambda: True) -> int:
        """Re-embed the stored skeletons (at idle, with Laya already loaded). Returns how many."""
        with self._lock:
            gen = self._gen
            todo = [(r.key, r.skeleton) for r in self.stale()]
        n = 0
        for k, skel in todo:
            if not keep_going():
                break
            e = embed(skel)
            if e is None:
                break
            with self._lock:
                if gen != self._gen:
                    return 0
                self._vecs[k] = np.array(e, dtype=np.float32).reshape(-1)
                n += 1
        return n

    def forget(self) -> int:
        """The privacy erase: every recipe and alias, on disk and in RAM. Returns how many."""
        with self._lock:
            self._gen += 1
            n = len(self._load()) + len(self._aliases)
            self._recipes, self._aliases = {}, {}
            self._vecs.clear()
            self._recent.clear()
            with contextlib.suppress(OSError):
                self._path().unlink(missing_ok=True)
            self._recipes = None
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._load())


def _alias_phrase(text: str) -> str:
    """The wording an alias is kept under: the request's skeleton without its click frame."""
    from neo.tools.computer import phrases

    skel = learn.skeleton(text) or (text if len(text.split()) == 1 else "")
    if not skel:
        return ""
    return phrases.norm(phrases.strip_frame(skel))[:120]


# ---- replay -----------------------------------------------------------------------------------------------
@dataclass
class Replayed:
    ok: bool
    done: int  # steps that ran and matched
    handoff: str = ""  # on a mismatch: "did X, expected Y, saw Z", for the agent


async def _call(fn: Callable, *a: Any) -> Any:
    out = fn(*a)
    return await out if inspect.isawaitable(out) else out


async def replay(
    recipe: Recipe,
    run_step: Callable[[Step], Awaitable[Any] | Any],
    check: Callable[[Step, Any], Awaitable[tuple[bool, str]] | tuple[bool, str]],
    *,
    mem: UIMemory | None = None,
) -> Replayed:
    """Run a recipe's steps through ``run_step`` (the tool handlers: the gate and the protected-app
    rules apply to each press) and ``check`` each one's effect against what it did before. Stops at
    the first failure or mismatch and says what happened; the outcome is counted on the recipe."""
    m = mem or _memory
    for i, step in enumerate(recipe.steps):
        try:
            got = await _call(run_step, step)
        except Exception as e:  # noqa: BLE001 - a step that broke hands over like one that missed
            got, err = None, f"{type(e).__name__}"
        else:
            err = ""
        matched, saw = (False, err or "it failed") if err else await _call(check, step, got)
        if not matched:
            m.result(recipe.key, False)
            want = ", ".join(f"{k}={v}" for k, v in step.expect.items()) or "a change"
            done = "; ".join(s.describe() for s in recipe.steps[:i]) or "nothing yet"
            return Replayed(False, i, f"replayed {done}; then {step.describe()} was expected to give {want}, but saw {saw or 'no change'}")
    m.result(recipe.key, True)
    return Replayed(True, len(recipe.steps))


# ---- the outcome ledger's hook ----------------------------------------------------------------------------
def settled(text: str, sigs: list[dict[str, Any]]) -> Recipe | None:
    """neo.reflex.outcomes: a request settled as verified with these step signatures. Kept as a
    recipe when every step is keepable, changed something and belongs to one app; a shortlist pick
    in it becomes an alias."""
    if not sigs or not _active():
        return None
    bundles = {str(s.get("bundle", "")) for s in sigs}
    if len(bundles) != 1 or not next(iter(bundles)):
        return None  # several apps, or one whose bundle id is unknown
    bundle = next(iter(bundles))
    if any(s.get("changed") is not True for s in sigs):
        return None  # a step that changed nothing, or whose effect nobody saw, replays nothing
    for s in sigs:
        if s.get("alias") and s.get("label") and s.get("role") in _CHROME:
            _memory.learn_alias(text, bundle, str(s["label"]))
    return _memory.remember(text, bundle, sigs)


_memory = UIMemory()
_rebuilding = False


def rebuild_soon(is_idle: Callable[[], bool]) -> bool:
    """Start a background re-embedding of the stored recipes' skeletons (after a restart) when NEO is
    idle and Laya is already loaded (never loaded for this), as neo.reflex.memory.rebuild_soon does."""
    global _rebuilding
    from neo import reflex

    laya = reflex._laya
    if laya is None or _rebuilding or not _active() or getattr(laya, "center", None) is None or not _memory.stale():
        return False

    def embed(text: str) -> np.ndarray | None:
        with reflex.laya_lock():
            laya.decide(text)
            e = getattr(laya, "last_embedding", None)
            return None if e is None else np.array(e, dtype=np.float32).reshape(-1)

    def run() -> None:
        global _rebuilding
        try:
            _memory.rebuild(embed, is_idle)
        except Exception as e:  # noqa: BLE001
            log.info(f"[uimemory] rebuild failed: {str(e)[:80]}")
        finally:
            _rebuilding = False

    _rebuilding = True
    threading.Thread(target=run, daemon=True, name="laya-uimemory").start()
    return True


def memory() -> UIMemory:
    return _memory


def remember(text: str, bundle: str, steps) -> Recipe | None:
    return _memory.remember(text, bundle, steps)


def recall(text: str, bundle: str, emb: Any = None, center: Any = None, *, origin: str = "") -> Recipe | None:
    return _memory.recall(text, bundle, emb, center, origin=origin)


def alias(text: str, bundle: str) -> str | None:
    return _memory.alias(text, bundle)


def learn_alias(text: str, bundle: str, label: str) -> bool:
    return _memory.learn_alias(text, bundle, label)


def forget() -> int:
    return _memory.forget()
