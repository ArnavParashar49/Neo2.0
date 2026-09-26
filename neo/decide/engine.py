"""Which app should do this? Laya decides, the user's own habits and NEO's track record weigh in.

For a request NEO works out, from evidence only (no app names anywhere in this file):

1. what kind of task it is (:func:`decide_kind`: the reflex's trained kind head; without one a
   Laya zero-shot choice over :data:`TASKS`, trusted only when it is very sure);
2. which installed apps can do that kind (their scripting dictionary's capabilities, the URL
   schemes and document types they declare, their category, media evidence), and in what order
   to try them (:func:`choose`):

   * an app the user named ("on <app>", "in <app>") that fits, or the website they named
     ("on <site>", :func:`site_hint`);
   * the apps this user actually uses, ordered by Laya ("which app should do this?", each option
     described by its evidence, on the zero-shot checkpoint of :mod:`neo.reflex.zeroshot`)
     combined with the bandit prior (``score = laya_p**LAYA_EXP * prior**PRIOR_EXP``); when Laya
     is unsure or unavailable, by the prior alone;
   * the task's fallback (the web, through the default browser);
   * used apps that are less likely to work than the fallback (see WEB_AFF), installed-but-unused
     apps, then apps that keep failing (still tried, so they can recover).

3. what worked (:func:`record`): a discounted Beta(a, b) per (kind, app, method) in
   ``<data_dir>/decide.json``. Three blamed failures in a row demote an arm; failures that are not
   the arm's fault (permission, login, offline, not_found) never count against it. Each settled
   attempt of a chosen app also goes to the reflex's outcome ledger (the request's text, kind,
   app key and whether it worked), which teaches its kind head.

The prior of a candidate = best method's posterior mean x how much this user uses the app
(:func:`neo.decide.inventory.affinity`). The fall-through to the next candidate on failure *is*
the exploration, so the voice path needs no randomness.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib
import json
import os
import re
import tempfile
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neo.config import settings

if TYPE_CHECKING:
    from neo.decide.inventory import App


# ---- task kinds -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One kind of request.

    ``description`` is what Laya reads (it shares a ~250-token budget with the other kinds, so
    keep it short). ``wants`` are capability tags (:func:`neo.decide.sdef.caps`) that make an app a
    candidate; ``signals`` are other generic evidence that does too:

    * ``media`` - the inventory found media evidence for the app;
    * ``category:<x>`` - its App Store category's last segment is exactly ``x``;
    * ``handles:<scheme>`` - it is the default handler for that standard URL scheme;
    * ``scheme:<scheme>`` - it declares that URL scheme;
    * ``type:<uti or extension>`` - it opens or edits that content type.

    A kind with neither (open an app, search the web, anything else) only takes the app the user
    named. ``fallback`` is tried after the fitting apps (``'web'`` = the default browser); it is
    also the key the fallback's outcomes are recorded under. ``methods`` are the generic ways to
    drive an app when no driver module is registered for the kind (see :data:`DRIVERS`).
    """

    description: str
    wants: tuple[str, ...] = ()
    fallback: str | None = None
    signals: tuple[str, ...] = ()
    methods: tuple[str, ...] = ("script", "url", "ax")


TASKS: dict[str, Task] = {
    "play_media": Task(
        "play a song, album, artist, playlist, podcast, video or show",
        wants=("media.search_play", "media.play_uri", "media.resume", "media.state", "media.now_playing"),
        fallback="web",
        signals=("media",),  # the inventory weighs category, frameworks, types (editors aren't players)
    ),
    "send_message": Task(
        "send a text or chat message to someone",
        wants=("message.send",),
        signals=("handles:sms", "category:social-networking"),
    ),
    "send_email": Task("write or send an email", wants=("mail.compose",), signals=("handles:mailto",)),
    "write_note": Task("write down or save a note", wants=("note.make",)),
    "add_reminder": Task("remind me later or add a to-do", wants=("reminder.make",)),
    "add_event": Task(
        "put a meeting or event on the calendar",
        wants=("event.make",),
        signals=("type:ics", "type:public.calendar-event"),
    ),
    "open_app": Task("open, launch or switch to an app", fallback="web", methods=("open",)),
    "web_search": Task("search the web or look something up online", fallback="web", methods=("url", "ax")),
    "other": Task("anything else", methods=("ax",)),
}

# kind -> module exposing ``methods_for(app) -> tuple[str, ...]`` (the kind's driver). A kind
# without one gets the generic Task.methods, filtered by what the app exposes.
DRIVERS: dict[str, str] = {"play_media": "neo.decide.media"}

WEB = "web"  # app key (and method) of the web fallback candidate


# ---- tuning -----------------------------------------------------------------------------------

MAX_CANDIDATES = 6  # attempts handed to a driver (it also caps total time)
LAYA_POOL = 6  # at most this many apps in one Laya choice (options share one ~250-token budget)
SHORTLIST_ABOVE = 20  # more used apps than this: an embedding shortlist picks the pool
LAYA_SURE_P = 0.5  # Laya counts as sure when its top pick has at least this probability...
LAYA_MARGIN = 0.15  # ...and leads the runner-up by at least this much
# A zero-shot task kind needs far more: on 390 mapped real requests it was right 0.57 of the time.
# At 0.95 (so a margin of at least 0.9) 5% of them count, 1 in 390 wrongly (a reminder that was
# labelled an event); at the app-choice bar 23% were sure and wrong.
ZS_KIND_P = 0.95
ZS_KIND_MARGIN = 0.9
LAYA_EXP = 1.0
PRIOR_EXP = 1.0
LAYA_LOCK_S = 1.5  # wait this long for the shared Laya lock (the reflex may be using it)
LAYA_TIMEOUT_S = 3.0  # whole Laya call incl. a model still loading: past this, use the prior
GAMMA = 0.97  # discount per update: old evidence fades, a switch of apps shows in ~3-10 uses
DEMOTE_AFTER = 3  # consecutive blamed failures that demote an arm
DEMOTE_FOR_S = 12 * 3600  # a demotion lapses after this, so a quiet arm gets re-probed
WEAK_MEAN = 0.25  # an app whose best arm is this bad is tried after the fallback too
USED_WITHIN_S = 90 * 86400  # "the user uses it": opened within this window
AFF_FLOOR = 0.3  # prior = arm mean x (AFF_FLOOR + (1 - AFF_FLOOR) x affinity)
WEB_AFF = 0.5  # a used app goes before the fallback when its prior beats the fallback's mean
# weighted like an app of this affinity (so an app opened once, weeks ago, goes after it)
# not the arm's fault: nothing is learned from these ('unconfirmed' = it may well have worked, but
# NEO couldn't hear it start)
NO_BLAME = frozenset({"permission", "login", "offline", "not_found", "unconfirmed"})
KEEP_ARMS_S = 365 * 86400  # arms untouched this long are dropped from decide.json

# (successes, failures) seeds before any evidence on this Mac, per method; unknown methods: even.
METHOD_PRIOR: dict[str, tuple[float, float]] = {
    "learned_uri": (4.0, 1.0),  # replay an item id the app itself reported for this query
    "script_search_play": (3.0, 1.0),  # scripting dictionary: search + play
    "script": (3.0, 1.0),
    "open": (4.0, 1.0),
    "url": (3.0, 2.0),  # the app's own deep-link scheme
    "local_file": (2.0, 1.0),
    "ax_search_play": (2.0, 2.0),  # search field + play button through the AX tree
    "ax": (2.0, 2.0),
    WEB: (2.0, 2.0),
}
DEFAULT_PRIOR = (2.0, 2.0)


@dataclass
class Candidate:
    app: App | None  # None = the task's fallback (the web)
    score: float
    laya_p: float
    evidence: str
    methods: tuple[str, ...]
    site: str = ""  # the fallback only: the website the user named ("... on <site>")

    @property
    def key(self) -> str:
        """The app key its outcomes are recorded under (``record(kind, c.key, method, ok)``)."""
        return self.app.bundle_id if self.app is not None else (self.methods[0] if self.methods else WEB)


def app_key(app: App | None) -> str:
    return app.bundle_id if app is not None else WEB


def _now() -> float:
    return time.time()


# ---- Laya -------------------------------------------------------------------------------------


def _laya_call(state: Any, questions: dict, stale: threading.Event) -> dict | None:
    """One zero-shot pass on :mod:`neo.reflex.zeroshot` (in a worker thread); None if it can't answer."""
    try:
        from neo.reflex import zeroshot

        return zeroshot.ask(state, questions, LAYA_LOCK_S, stale)
    except Exception as e:  # noqa: BLE001 - Laya must never break a decision
        print(f"[decide] Laya unavailable for this decision ({str(e)[:80]})")
        return None


def _reflex_call(text: str, stale: threading.Event) -> Any:
    """The reflex's Decision for ``text`` when it has trained heads (in a worker thread), else None."""
    try:
        from neo.reflex import get_laya, laya_lock

        lock = laya_lock()
        # get_laya() takes this same (non-reentrant) lock with no timeout, so only call it once the
        # lock is free: a busy Laya then costs LAYA_LOCK_S, not the whole LAYA_TIMEOUT_S.
        if not lock.acquire(timeout=LAYA_LOCK_S):
            return None  # busy (retraining, loading, another decision)
        lock.release()
        laya = get_laya()
        if laya is None or not getattr(laya, "has_head", False):
            return None
        if not lock.acquire(timeout=LAYA_LOCK_S):
            return None
        try:
            return None if stale.is_set() else laya.decide(text)
        finally:
            lock.release()
    except Exception as e:  # noqa: BLE001
        print(f"[decide] the reflex couldn't tell the task kind ({str(e)[:80]})")
        return None


async def _off_loop(fn: Any, *args: Any) -> Any:
    """``fn(*args, stale)`` in a worker thread, None past LAYA_TIMEOUT_S (a model still loading, a
    busy GPU); a worker still waiting then skips its forward pass."""
    stale = threading.Event()
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, stale), LAYA_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - timeout or anything else
        return None
    finally:
        stale.set()


async def _ask_laya(state: Any, questions: dict) -> dict | None:
    return await _off_loop(_laya_call, state, questions)


def _choice(res: dict | None, qid: str) -> dict[str, float]:
    """Label -> probability from a Laya result (empty if it gave none)."""
    if not isinstance(res, dict):
        return {}
    ans = (res.get("answers") or {}).get(qid) or {}
    probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
    if not any(probs.values()) and ans.get("choice") is not None:
        probs = {str(ans["choice"]): 1.0}
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()} if z > 0 else {}


def _sure(probs: dict[str, float], top_p: float = LAYA_SURE_P, margin: float = LAYA_MARGIN) -> bool:
    if not probs:
        return False
    top = sorted(probs.values(), reverse=True) + [0.0]
    return top[0] >= top_p and top[0] - top[1] >= margin


async def decide_kind(text: str, decision: Any = None) -> tuple[str, float]:
    """The task kind of ``text`` and its probability.

    The reflex's trained kind head decides when it has one: ``decision`` is the reflex's Decision
    for this text if the caller has it, else the reflex is asked (one embedding pass). Without a
    kind head, Laya's zero-shot choice over :data:`TASKS` counts only when it is very sure
    (:data:`ZS_KIND_P`); otherwise ('other', 0.0) - only an app the user named is tried, so a
    confidently wrong kind can't send the request to the wrong kind of app.
    """
    if not text or not text.strip():
        return "other", 0.0
    text = text.strip()
    if decision is None:
        decision = await _off_loop(_reflex_call, text)
    kind = getattr(decision, "kind", None)
    if kind in TASKS:
        return kind, float(getattr(decision, "kind_p", 0.0) or 0.0)
    q = {
        "kind": {
            "type": "choice",
            "instructions": "What does `text` ask the computer to do?",
            "criteria": {k: t.description for k, t in TASKS.items()},
        }
    }
    probs = {k: p for k, p in _choice(await _ask_laya({"text": text}, q), "kind").items() if k in TASKS}
    if not _sure(probs, ZS_KIND_P, ZS_KIND_MARGIN):
        return "other", 0.0
    kind = max(probs, key=lambda k: probs[k])
    return kind, float(probs[kind])


# ---- the learning store (decide.json) ---------------------------------------------------------

_lock = threading.RLock()
_cache: dict[str, tuple[int, dict]] = {}  # path -> (mtime_ns, data)


def _path() -> Path:
    return settings().data_dir / "decide.json"


def _fresh() -> dict:
    return {"version": 1, "arms": {}}


def forget() -> int:
    """Erase what NEO learned about this user's apps (the privacy erase). Returns arms dropped."""
    with _lock:
        n = sum(len(m) for per_app in _load()["arms"].values() for m in per_app.values())
        _cache.clear()
        for p in _path().parent.glob(f"{_path().name}*"):  # the store and any set-aside copy
            p.unlink(missing_ok=True)
        return n


def _clean(raw: Any) -> dict:
    """Keep only well-formed arms: a hand-edited or half-migrated file must not break decisions."""
    out = _fresh()
    arms = raw.get("arms") if isinstance(raw, dict) else None
    if not isinstance(arms, dict):
        raise ValueError("no arms")
    for kind, per_app in arms.items():
        if not isinstance(per_app, dict):
            continue
        for key, per_method in per_app.items():
            if not isinstance(per_method, dict):
                continue
            for method, arm in per_method.items():
                try:
                    a, b = float(arm["a"]), float(arm["b"])
                    if not (0 < a < 1e6 and 0 < b < 1e6):  # also rejects NaN / Infinity
                        continue
                    out["arms"].setdefault(str(kind), {}).setdefault(str(key), {})[str(method)] = {
                        "a": a,
                        "b": b,
                        "n": int(arm.get("n", 0)),
                        "ok": int(arm.get("ok", 0)),
                        "fails": int(arm.get("fails", 0)),
                        "last_ok": float(arm.get("last_ok", 0.0)),
                        "last_fail": float(arm.get("last_fail", 0.0)),
                        "t": float(arm.get("t", 0.0)),
                        "reason": str(arm.get("reason", ""))[:40],
                    }
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
    return out


def _load() -> dict:
    """The store (cached until the file changes). A corrupt file is set aside, never fatal."""
    p = _path()
    with _lock:
        hit = _cache.get(str(p))
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:  # no file yet (or it was set aside): what this session learned, if anything
            if hit is None or hit[0] != -1:
                hit = _cache[str(p)] = (-1, _fresh())
            return hit[1]
        if hit is not None and hit[0] == mtime:
            return hit[1]
        try:
            data = _clean(json.loads(p.read_text()))
        except (OSError, ValueError, TypeError) as e:
            print(f"[decide] {p.name} was unreadable ({str(e)[:60]}); starting fresh")
            with contextlib.suppress(OSError):
                p.replace(p.with_name(p.name + ".corrupt"))
            data = _fresh()
            try:  # couldn't set it aside: don't re-read (and re-report) it until it changes
                mtime = p.stat().st_mtime_ns
            except OSError:
                mtime = -1
        _cache[str(p)] = (mtime, data)
        return data


def _save(data: dict) -> None:
    """Atomic: write a temp file in the same folder, fsync, rename over the old one."""
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".decide.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    with contextlib.suppress(OSError):
        _cache[str(p)] = (p.stat().st_mtime_ns, data)


def _seed(method: str) -> dict:
    a, b = METHOD_PRIOR.get(method, DEFAULT_PRIOR)
    return {"a": a, "b": b, "n": 0, "ok": 0, "fails": 0, "last_ok": 0.0, "last_fail": 0.0, "t": 0.0, "reason": ""}


def _arm(data: dict, kind: str, key: str, method: str) -> dict:
    return data["arms"].get(kind, {}).get(key, {}).get(method) or _seed(method)


def _mean(arm: dict) -> float:
    return arm["a"] / (arm["a"] + arm["b"])


def _demoted(arm: dict, now: float) -> bool:
    return arm["fails"] >= DEMOTE_AFTER and now - arm["last_fail"] < DEMOTE_FOR_S


def record(kind: str, app_key: str, method: str, ok: bool, reason: str = "") -> None:
    """Learn from one attempt. ``reason`` in NO_BLAME: not the arm's fault, nothing changes."""
    reason = (reason or "").strip().lower()
    now = _now()
    with _lock:
        data = copy.deepcopy(_load())
        per_method = data["arms"].setdefault(kind, {}).setdefault(app_key, {})
        arm = per_method.get(method) or _seed(method)
        if not ok and reason in NO_BLAME:
            arm["reason"], arm["t"] = reason, now  # remembered for the log, not held against it
        else:
            arm["a"], arm["b"] = arm["a"] * GAMMA, arm["b"] * GAMMA
            arm["n"] += 1
            arm["t"], arm["reason"] = now, ("" if ok else reason)
            if ok:
                arm["a"] += 1.0
                arm["ok"] += 1
                arm["fails"] = 0
                arm["last_ok"] = now
            else:
                arm["b"] += 1.0
                arm["fails"] += 1
                arm["last_fail"] = now
        per_method[method] = arm
        _prune(data, now)
        try:
            _save(data)
        except OSError as e:
            print(f"[decide] couldn't save what was learned ({str(e)[:80]})")
            try:  # still learn for this session: keep it until the file itself changes
                mtime = _path().stat().st_mtime_ns
            except OSError:
                mtime = -1
            _cache[str(_path())] = (mtime, data)
    if ok or reason not in NO_BLAME:
        _tell_ledger(kind, app_key, ok)


@dataclass
class _Asked:
    """The request :func:`choose` last ordered in this task: what its outcomes are reported as."""

    kind: str
    text: str
    keys: frozenset[str]
    told: set[tuple[str, bool]] = field(default_factory=set)


_asked: ContextVar[_Asked | None] = ContextVar("neo_decide_asked", default=None)


def _tell_ledger(kind: str, key: str, ok: bool) -> None:
    """One settled attempt of a chosen app, to the reflex's outcome ledger when it has one: the
    request text choose() was given, the kind, the app key and ok - never a query, tool arguments
    or results. Each (app, outcome) once per request; an attempt outside a decision (a per-site
    sub-arm, another task) is not reported."""
    a = _asked.get()
    if a is None or a.kind != kind or key not in a.keys or (key, ok) in a.told:
        return
    a.told.add((key, ok))
    try:
        from neo.reflex import outcomes

        decided = getattr(outcomes, "decided", None)
        if callable(decided):
            decided(a.text, kind, key, ok)
    except Exception as e:  # noqa: BLE001 - no ledger (yet), or a broken one: learning is optional
        if not isinstance(e, ImportError):
            print(f"[decide] couldn't report the outcome ({str(e)[:60]})")


def _prune(data: dict, now: float) -> None:
    for kind in list(data["arms"]):
        per_app = data["arms"][kind]
        for key in list(per_app):
            for method in list(per_app[key]):
                if now - per_app[key][method]["t"] > KEEP_ARMS_S:
                    del per_app[key][method]
            if not per_app[key]:
                del per_app[key]
        if not per_app:
            del data["arms"][kind]


def prior(kind: str, app_key: str, method: str) -> float:
    """Posterior mean of the (kind, app, method) arm: how often this way of doing it works here."""
    with _lock:
        return _mean(_arm(_load(), kind, app_key, method))


# ---- evidence ---------------------------------------------------------------------------------


def _inventory() -> list[App]:
    from neo.decide.inventory import inventory

    return inventory()


def warm() -> None:
    """Scan the apps in the background now, so the first request doesn't wait for it."""
    threading.Thread(target=lambda: _inventory() and None, name="neo-inventory-warm", daemon=True).start()


def _self_ids() -> frozenset[str]:
    """NEO's own app (the one that launched this core), so it never picks itself."""
    if os.environ.get("NEO_LAUNCHED_BY_APP") != "1":
        return frozenset()
    try:
        from neo.decide.media import _bundle_id_of_pid, responsible_pid

        ppid = os.getppid()
        return frozenset(b.lower() for b in {_bundle_id_of_pid(ppid), _bundle_id_of_pid(responsible_pid(ppid))} if b)
    except Exception:  # noqa: BLE001
        return frozenset()


def _affinity(app: App, now: float) -> float:
    try:
        from neo.decide.inventory import affinity

        return max(0.0, min(1.0, float(affinity(app, now))))
    except Exception:  # noqa: BLE001
        return 0.0


def _named(text: str, apps: list[App]) -> App | None:
    try:
        from neo.decide.inventory import named_in

        return named_in(text, apps)
    except Exception:  # noqa: BLE001
        return None


def _app_caps(app: App) -> frozenset[str]:
    if getattr(app, "script", None) is None:
        return frozenset()
    try:
        from neo.decide.sdef import caps

        return frozenset(caps(app.script))
    except Exception:  # noqa: BLE001
        return frozenset()


def _last_segment(category: str) -> str:
    return (category or "").rsplit(".", 1)[-1].lower()


def _has_type(app: App, t: str) -> bool:
    t = t.lower()
    for x in (*app.viewer_types, *app.editor_types):
        x = str(x).lower()
        if x == t or x.endswith("." + t):
            return True
    return False


_NOT_PLAYERS = frozenset({"audio/video editor"})  # media evidence, but they record/edit, not play


def _signal(app: App, sig: str) -> bool:
    name, _, arg = sig.partition(":")
    if name == "media":
        return bool(app.media_evidence) and app.media_evidence[0] not in _NOT_PLAYERS
    if name == "category":
        return bool(arg) and _last_segment(app.category) == arg
    if name == "handles":
        return arg in app.handles
    if name == "scheme":
        return arg in app.schemes
    if name == "type":
        return _has_type(app, arg)
    return False


def _is_browser(app: App) -> bool:
    """A web browser: the default http(s) handler, or an app that claims http(s) AND opens HTML
    (owning an http scheme alone doesn't make one). A web-app wrapper is its own app, not a browser."""
    if "https" in app.handles or "http" in app.handles:
        return True
    return (
        not app.web_start_url
        and bool({"http", "https"}.intersection(app.schemes))
        and _has_type(app, "html")
    )


def _fits(task: Task, app: App, caps: frozenset[str]) -> bool:
    return bool(caps.intersection(task.wants)) or any(_signal(app, s) for s in task.signals)


_SITE_RX = re.compile(
    r"\s(?:on|from|via|using|at)\s+(?:the\s+|my\s+)?(?P<site>[a-z0-9][a-z0-9.\-]*(?:\s+[a-z0-9][a-z0-9.\-]*)?)\s*[.!?]*\s*$",
    re.I,
)


def site_hint(text: str, apps: list[App] | None = None) -> str:
    """'... on <site>' closing a request, where <site> is no app on this Mac and not an ordinary
    word ('on repeat'): the website the user asked for. A hint only - the web driver keeps it when
    a result really is on that site."""
    m = _SITE_RX.search(" " + " ".join(str(text or "").split()))
    if not m:
        return ""
    site = m.group("site").lower().strip(".-")
    if apps is not None and _named(site, apps) is not None:
        return ""
    if "." not in site:
        words = set(site.split())
        try:
            from neo.decide.inventory import _common_words

            common = _common_words(words)
        except Exception:  # noqa: BLE001
            return ""
        if all(common.get(w, True) for w in words):
            return ""
    return site


def _open_ended(task: Task) -> bool:
    """A kind any app could be meant for: only the app the user named is a candidate."""
    return not task.wants and not task.signals


_PREP_RX = re.compile(r"\b(?:on|in|with|using|via|through)\s+(?:the\s+|my\s+)?")


def _explicit(text: str, app: App) -> bool:
    """'... on <app>' / 'in <app>' / 'with <app>': the user clearly meant that app, by any name the
    inventory knows it by (its display name, a short form of it, its own URL scheme)."""
    low = " ".join(text.lower().split())
    for m in _PREP_RX.finditer(low):
        words = low[m.end() :].split()[:6]
        for k in range(1, len(words) + 1):
            hit = _named(" ".join(words[:k]), [app])
            if hit is None or hit.bundle_id != app.bundle_id:
                continue
            # the name has to start right after the preposition, not somewhere further on
            if k == 1 or _named(" ".join(words[1:k]), [app]) is None:
                return True
            break
    return False


def _methods(kind: str, task: Task, app: App, caps: frozenset[str]) -> tuple[str, ...]:
    mod = DRIVERS.get(kind)
    if mod:
        try:
            fn = importlib.import_module(mod).methods_for
        except (ImportError, AttributeError):
            fn = None
        if fn is not None:
            try:
                return tuple(fn(app))
            except Exception:  # noqa: BLE001 - a driver bug must not sink the whole decision
                return ()
    out = []
    for m in task.methods:
        if m == "script" and not caps.intersection(task.wants):
            continue
        if m == "url" and not app.own_schemes:
            continue
        out.append(m)
    return tuple(out)


def _usage_words(app: App, now: float) -> str:
    if app.days_used_30:
        return f"used {app.days_used_30} of last 30 days"
    if app.last_used:
        days = int(max(0.0, now - app.last_used) // 86400)
        return "used today" if days == 0 else f"last used {days} days ago"
    return "never used"


def _record_words(data: dict, kind: str, key: str) -> str:
    arms = list(data["arms"].get(kind, {}).get(key, {}).values())
    n = sum(a["n"] for a in arms)
    if not n:
        return "not tried yet"
    fails = max(a["fails"] for a in arms)
    if fails >= DEMOTE_AFTER:
        return f"failed last {fails} tries"
    return f"worked {sum(a['ok'] for a in arms)} of {n} tries"


def _describe(task: Task, app: App, caps: frozenset[str], rec: str, now: float, short: bool) -> str:
    """What Laya reads about one candidate: all of it evidence, none of it a name list."""
    parts = []
    if cat := _last_segment(app.category).replace("-", " "):
        parts.append(cat)
    can = [t.split(".", 1)[-1].replace("_", " ") for t in task.wants if t in caps]
    if can and not short:
        parts.append("can " + ", ".join(can))
    parts.append(_usage_words(app, now))
    if app.running and not short:
        parts.append("open now")
    if rec != "not tried yet" or not short:
        parts.append(rec)
    return "; ".join(parts)


@dataclass
class _Info:
    app: App
    caps: frozenset[str]
    methods: tuple[str, ...]
    aff: float
    prior: float
    used: bool
    weak: bool
    rec: str
    mean: float = 0.0  # its best method's posterior mean
    proven: bool = False  # it has done this kind of task for NEO before
    label: str = ""


def _info(kind: str, task: Task, app: App, caps: frozenset[str], data: dict, now: float) -> _Info | None:
    methods = _methods(kind, task, app, caps)
    if not methods:
        return None
    key = app.bundle_id
    arms = {m: _arm(data, kind, key, m) for m in methods}
    # demoted methods last; otherwise best-first (stable: the driver's order breaks ties)
    methods = tuple(sorted(methods, key=lambda m: (_demoted(arms[m], now), -_mean(arms[m]))))
    live = [m for m in methods if not _demoted(arms[m], now)]
    # Once NEO has tried this app, judge it by the methods it actually tried: a never-attempted
    # method's optimistic seed (e.g. one the driver only uses when it has something to replay)
    # must not keep an app whose real attempts keep failing ahead of the fallback.
    tried = [m for m in methods if arms[m]["n"] > 0]
    basis = [m for m in live if arms[m]["n"] > 0] if tried else live
    best = max((_mean(arms[m]) for m in basis), default=0.0)
    aff = _affinity(app, now)
    succeeded = any(a["ok"] for a in data["arms"].get(kind, {}).get(key, {}).values())
    used = bool(
        app.running
        or app.days_used_30
        or (app.last_used and now - app.last_used < USED_WITHIN_S)
        or succeeded
    )
    return _Info(
        app=app,
        caps=caps,
        methods=methods,
        aff=aff,
        prior=(best or min(_mean(a) for a in arms.values())) * (AFF_FLOOR + (1 - AFF_FLOOR) * aff),
        used=used,
        weak=not basis or best < WEAK_MEAN,
        rec=_record_words(data, kind, key),
        mean=best,
        proven=succeeded,
    )


def _evidence(task: Task, i: _Info, now: float, named: bool = False) -> str:
    parts = ["named by the user"] if named else []
    parts.append(_describe(task, i.app, i.caps, i.rec, now, short=False))
    parts.extend(str(e) for e in i.app.media_evidence[:3])
    return "; ".join(p for p in parts if p)


def _candidate(task: Task, i: _Info, now: float, score: float, laya_p: float, named: bool = False) -> Candidate:
    return Candidate(i.app, score, laya_p, _evidence(task, i, now, named), i.methods)


def _dedupe(apps: list[App]) -> list[App]:
    seen: set[str] = set()
    out = []
    for a in apps:
        if a.bundle_id and a.bundle_id not in seen:
            seen.add(a.bundle_id)
            out.append(a)
    return out


def _labels(infos: list[_Info]) -> None:
    taken: dict[str, int] = {}
    for i in infos:
        base = (i.app.name or i.app.bundle_id).strip() or i.app.bundle_id
        n = taken.get(base.lower(), 0) + 1
        taken[base.lower()] = n
        i.label = base if n == 1 else f"{base} ({n})"


def _shortlist(task: Task, text: str, infos: list[_Info], now: float) -> list[_Info]:
    """The LAYA_POOL apps whose evidence reads most like the request, by NEO's pinned MiniLM (the
    memory embedder: already loaded, CPU, ~13 ms for 25 apps); [] when it isn't available.

    Over 24 used apps (32 requests) the right one was among the 6 kept for 0.84 of requests;
    Laya's own mean-pooled encoder (laya.shortlist.embed_fn_from_agent) managed 0.31, about
    chance, in 150 ms of GPU time."""
    from neo.memory import embed

    opts = [f"{i.label}: {_describe(task, i.app, i.caps, i.rec, now, short=True)}" for i in infos]
    try:
        vecs = embed.embed_texts([text, *opts])
    except Exception as e:  # noqa: BLE001 - a shortlist is an optimisation, never a failure
        print(f"[decide] no shortlist ({str(e)[:60]})")
        return []
    if vecs is None or len(vecs) != len(opts) + 1:
        return []
    sims = vecs[1:] @ vecs[0]  # unit vectors: cosine
    keep = set(sorted(range(len(infos)), key=lambda j: -float(sims[j]))[:LAYA_POOL])
    return [i for j, i in enumerate(infos) if j in keep]


async def _laya_order(task: Task, text: str, used: list[_Info], now: float) -> list[tuple[_Info, float, float]]:
    """(info, score, laya_p), best first. Laya's pick counts only when it is sure."""
    if len(used) < 2:
        return [(i, i.prior, 0.0) for i in used]
    by_prior = sorted(used, key=lambda i: -i.prior)
    _labels(by_prior)
    pool = by_prior[:LAYA_POOL]
    if len(by_prior) > SHORTLIST_ABOVE:
        pool = await asyncio.to_thread(_shortlist, task, text, by_prior, now) or pool
    # The request is the whole state; each option is its app's short evidence. Restating every
    # candidate's evidence in the state (as JSON) was worse and costlier on decide's own questions:
    # Laya's pick right 0.59 vs 0.85, 277 vs 82 tokens, 82 vs 35 ms a question.
    q = {
        "app": {
            "type": "choice",
            "instructions": "Which app on this Mac should do the request?",
            "criteria": {i.label: _describe(task, i.app, i.caps, i.rec, now, short=True) for i in pool},
        }
    }
    probs = _choice(await _ask_laya({"request": text}, q), "app")
    if not _sure(probs):  # unsure or unavailable: the prior orders them
        return [(i, i.prior, probs.get(i.label, 0.0)) for i in by_prior]
    scored = []
    for i in pool:
        p = probs.get(i.label, 0.0)
        scored.append((i, (max(p, 1e-3) ** LAYA_EXP) * (max(i.prior, 1e-6) ** PRIOR_EXP), p))
    scored.sort(key=lambda t: -t[1])
    in_pool = {id(i) for i in pool}
    return scored + [(i, i.prior, 0.0) for i in by_prior if id(i) not in in_pool]


async def choose(kind: str, text: str, apps: list[App] | None = None) -> list[Candidate]:
    """Ordered attempts for this request on this Mac (see the module docstring)."""
    task = TASKS.get(kind) or TASKS["other"]
    now = _now()
    if apps is None:
        try:
            apps = await asyncio.to_thread(_inventory)  # the first scan takes a while: off the loop
        except Exception as e:  # noqa: BLE001
            print(f"[decide] app inventory failed ({str(e)[:80]})")
            apps = []
    me = _self_ids()
    apps = _dedupe([a for a in apps if a.bundle_id.lower() not in me])
    with _lock:
        data = copy.deepcopy(_load())
    named = _named(text, apps)
    site = site_hint(text, apps) if task.fallback == WEB and named is None else ""

    first: list[Candidate] = []
    used: list[_Info] = []
    unused: list[_Info] = []
    weak: list[_Info] = []
    for app in apps:
        caps = _app_caps(app)
        is_named = named is not None and app.bundle_id == named.bundle_id
        fits = _fits(task, app, caps)
        if is_named:
            if (fits or _open_ended(task) or _explicit(text, app)) and (
                i := _info(kind, task, app, caps, data, now)
            ):
                first.append(_candidate(task, i, now, i.prior, 0.0, named=True))
            continue
        if not fits:
            continue
        if task.fallback == WEB and _is_browser(app):
            continue  # browsers are the web fallback's (it opens the default one), unless named
        i = _info(kind, task, app, caps, data, now)
        if i is None:
            continue
        (weak if i.weak else used if i.used else unused).append(i)

    fallback: list[Candidate] = []
    fb_mean = 0.0
    if task.fallback:
        fb = task.fallback
        fb_mean = _mean(_arm(data, kind, fb, fb))
        on = f" ({site})" if site else ""
        fallback.append(
            Candidate(None, fb_mean, 0.0, f"fallback: {fb}{on}; {_record_words(data, kind, fb)}", (fb,), site=site)
        )
    # A used app goes before the fallback when its prior beats the fallback's (see WEB_AFF), when
    # it has done this before at least as reliably, or when Laya is sure it's the one; the others
    # (e.g. a player opened once, weeks ago, with nothing in its library) go after it.
    bar = fb_mean * (AFF_FLOOR + (1 - AFF_FLOOR) * WEB_AFF)
    before: list[Candidate] = []
    after: list[Candidate] = []
    for n, (i, s, p) in enumerate(await _laya_order(task, text, used, now)):
        keep = (
            not fallback
            or i.prior >= bar
            or (i.proven and i.mean >= fb_mean)
            or (n == 0 and p >= LAYA_SURE_P)
        )
        (before if keep else after).append(_candidate(task, i, now, s, p))
    tail = [_candidate(task, i, now, i.prior, 0.0) for i in sorted(unused, key=lambda i: -i.prior)]
    tail += [_candidate(task, i, now, i.prior, 0.0) for i in sorted(weak, key=lambda i: -i.prior)]

    # the site the user named goes first; otherwise the apps they use lead
    seq = first + (fallback + before if site else before + fallback) + after + tail
    out = seq[:MAX_CANDIDATES]
    if fallback and not any(c is fallback[0] for c in out):  # the fallback always keeps a slot
        out = out[: MAX_CANDIDATES - 1] + fallback
    _asked.set(_Asked(kind, text, frozenset(c.key for c in out)))
    return out


__all__ = [
    "TASKS",
    "DRIVERS",
    "WEB",
    "Task",
    "Candidate",
    "app_key",
    "decide_kind",
    "choose",
    "site_hint",
    "warm",
    "record",
    "prior",
    "forget",
]
