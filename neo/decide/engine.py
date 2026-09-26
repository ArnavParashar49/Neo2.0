"""Which app should do this? Laya decides, the user's own habits and NEO's track record weigh in.

For a request NEO works out, from evidence only (no app names anywhere in this file):

1. what kind of task it is (:func:`decide_kind`: the reflex's trained kind head; without one,
   Laya's zero-shot choice over :data:`TASKS` when it is very sure, marked as advice only);
2. which installed apps can do that kind (their scripting dictionary's capabilities, the URL
   schemes and document types they declare, their category, media evidence), and in what order
   to try them (:func:`choose`):

   * an app the user named ("on <app>", "in <app>") that fits, or the website they named
     ("on <site>", :func:`site_hint`);
   * the apps this user actually uses, ordered by Laya ("which app should do this?", each option
     described by its evidence, on the zero-shot checkpoint of :mod:`neo.reflex.zeroshot`)
     combined with the bandit prior (``score = laya_p**LAYA_EXP * prior**PRIOR_EXP``); when Laya
     is unsure or unavailable, by the prior alone. A destination that is no app (the Mac's shared
     calendar or reminders store, :mod:`neo.decide.routes`) is weighed like one, and for kinds
     with preferences the user's corrections and per-person choices scale the prior;
   * the task's fallback (the web, through the default browser; a local last resort such as a
     file for some kinds);
   * used apps that are less likely to work than the web fallback (see WEB_AFF), installed-but-
     unused apps, then apps that keep failing (still tried, so they can recover).

3. what worked (:func:`record`): a discounted Beta(a, b) per (kind, app, method) in
   ``<data_dir>/decide.json``. Three blamed failures in a row demote an arm; failures that are not
   the arm's fault (permission, login, offline, not_found) never count against it. Each settled
   attempt of a chosen app also goes to the reflex's outcome ledger (the user's own words for the
   request, when the caller has them; the kind, app key and whether it worked), which teaches
   its kind head.

The prior of a candidate = best method's posterior mean x how much this user uses the app
(:func:`neo.decide.inventory.affinity`). The fall-through to the next candidate on failure *is*
the exploration, so the voice path needs no randomness.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib
import inspect
import json
import os
import re
import tempfile
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from neo.config import settings

if TYPE_CHECKING:
    from neo.decide.inventory import App


# ---- task kinds -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One kind of request.

    ``description`` is what Laya reads (it shares a ~190-token budget with the other kinds, so
    keep it short and contrastive). ``wants`` are capability tags (:func:`neo.decide.sdef.caps`,
    :func:`neo.decide.sdef.read_caps`) that make an app a candidate; ``signals`` are other generic
    evidence that does too:

    * ``media`` - the inventory found media evidence for the app;
    * ``category:<x>`` - its App Store category's last segment is exactly ``x``;
    * ``handles:<scheme>`` - it is the default handler for that standard URL scheme;
    * ``scheme:<scheme>`` - it declares that URL scheme;
    * ``type:<uti or extension>`` - it opens or edits that content type;
    * ``intent:<INIntent>`` - one of its extensions handles that SiriKit intent;
    * ``protocol:<P>`` / ``schema:<domain.Name>`` - it ships that App Intents protocol / schema;
    * ``share:text`` - a share extension of it accepts text;
    * ``library_text_editor`` - it edits plain text or Markdown, keeps no documents of its own
      and is filed under productivity (or nothing): a notes library, not a document editor.

    A trailing ``~`` makes a signal WEAK: it counts only for an app this user uses (or that has
    done the task before) that is not the default handler of a call scheme (see :func:`fits_kind`).

    A kind with neither (open an app, search the web, anything else) only takes the app the user
    named. ``fallback`` is tried after the fitting apps (``'web'`` = the default browser); it is
    also the key the fallback's outcomes are recorded under. ``methods`` are the generic ways to
    drive an app when no driver module is registered for the kind (see :data:`DRIVERS`).
    ``routes`` are destinations that are no app (the OS's shared calendar and reminder stores:
    'eventkit.events', 'eventkit.reminders'). Laya orders the candidates once ``laya_min_used``
    used ones fit (at 1, capable apps not used yet join the question too). ``proven_fit``: an app
    that has done this kind for NEO before is a candidate without declared evidence. ``prefs``:
    the user's corrections and per-person choices weigh in (:func:`record_pref`).
    """

    description: str
    wants: tuple[str, ...] = ()
    fallback: str | None = None
    signals: tuple[str, ...] = ()
    methods: tuple[str, ...] = ("script", "url", "ax")
    routes: tuple[str, ...] = ()
    laya_min_used: int = 2
    proven_fit: bool = False
    prefs: bool = False


# The kinds and their evidence. Sends (send_message, send_email) have no driver yet (R4/R5); every
# other kind is driven by its module in DRIVERS once the user switched it on (neo.config
# decide_modes).
TASKS: dict[str, Task] = {
    "play_media": Task(
        "play a song, album, artist, playlist, podcast, video or show",
        wants=("media.search_play", "media.play_uri", "media.resume", "media.state", "media.now_playing"),
        fallback="web",
        signals=("media",),  # the inventory weighs category, frameworks, types (editors aren't players)
    ),
    "send_message": Task(
        "text or chat message to a person, not email",
        wants=("message.send",),
        # a social app counts only when used and not a call handler (calls aren't messages)
        signals=("handles:sms", "intent:INSendMessageIntent", "category:social-networking~"),
        laya_min_used=1,
        proven_fit=True,
        prefs=True,
    ),
    "send_email": Task(
        "write or send an email",
        wants=("mail.compose",),
        signals=("handles:mailto", "protocol:SendMail", "schema:mail.CreateDraftIntent"),
        laya_min_used=1,
        proven_fit=True,
        prefs=True,
    ),
    "read_mail": Task(
        "check or read my email inbox",
        wants=("mail.read", "mail.unread_count"),
        signals=("handles:mailto",),
        laya_min_used=1,
        proven_fit=True,
    ),
    "write_note": Task(
        "write down, jot or save a note",
        wants=("note.make",),
        signals=("intent:INCreateNoteIntent", "intent:INAppendToNoteIntent", "library_text_editor~"),
        fallback="file",
        laya_min_used=1,
        proven_fit=True,
        prefs=True,
    ),
    "add_reminder": Task(
        "remind me later, or add a to-do or list item",
        wants=("reminder.make",),
        signals=("intent:INAddTasksIntent",),
        routes=("eventkit.reminders",),
        laya_min_used=1,
        proven_fit=True,
        prefs=True,
    ),
    # an app that only opens calendar files is no candidate: the .ics hand-off is the fallback
    "add_event": Task(
        "put a meeting or appointment on my calendar",
        wants=("event.make",),
        routes=("eventkit.events",),
        fallback="ics",
        laya_min_used=1,
        proven_fit=True,
        prefs=True,
    ),
    "read_calendar": Task(
        "what's on my calendar or schedule",
        wants=("event.read",),
        routes=("eventkit.events",),
        laya_min_used=1,
    ),
    "open_app": Task("open, launch or switch to an app", fallback="web", methods=("open",)),
    "web_search": Task("search the web or a website", fallback="web", methods=("url", "ax")),
    "other": Task("anything else", methods=("ax",)),
}

# kind -> module exposing ``methods_for(app[, kind]) -> tuple[str, ...]`` (the kind's driver). A
# kind without one gets the generic Task.methods, filtered by what the app exposes.
DRIVERS: dict[str, str] = {
    "play_media": "neo.decide.media",
    "write_note": "neo.decide.kinds.note",
    "add_event": "neo.decide.kinds.schedule",
    "add_reminder": "neo.decide.kinds.schedule",
    "read_calendar": "neo.decide.kinds.schedule",
    "read_mail": "neo.decide.kinds.mail",
    "open_app": "neo.decide.kinds.launch",
    "web_search": "neo.decide.kinds.site",
}

WEB = "web"  # app key (and method) of the web fallback candidate


# ---- tuning -----------------------------------------------------------------------------------

MAX_CANDIDATES = 6  # attempts handed to a driver (it also caps total time)
LAYA_POOL = 6  # at most this many apps in one Laya choice (options share one ~250-token budget)
SHORTLIST_ABOVE = 20  # more used apps than this: an embedding shortlist picks the pool
LAYA_SURE_P = 0.5  # Laya counts as sure when its top pick has at least this probability...
LAYA_MARGIN = 0.15  # ...and leads the runner-up by at least this much
# A zero-shot task kind needs far more, and is still only advice (Kind.advisory): on 390 mapped
# real requests it was right 0.57 of the time. At 0.95 (so a margin of at least 0.9) 21 answers
# count and 1 is wrong (95% upper bound 24%); 19 of them are events or emails, 1 is play_media.
# A bar picked on half the requests and tested on the other half (200 random splits) was 0.95 or
# 0.97 in 374 of 400 picks, and wrong for 9% of the answers it let through.
ZS_KIND_P = 0.95
ZS_KIND_MARGIN = 0.9
LAYA_EXP = 1.0
PRIOR_EXP = 1.0
LAYA_LOCK_S = 1.5  # wait this long for the shared Laya lock (the reflex may be using it)
LAYA_TIMEOUT_S = 3.0  # a whole Laya step (a model still loading, the shortlist): past it, the prior
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
# ('ambiguous' = NEO asked who, 'declined' = the user said no to a card, 'needs_user' = NEO asked)
NO_BLAME = frozenset(
    {"permission", "login", "offline", "not_found", "unconfirmed", "ambiguous", "declined", "needs_user"}
)
ROUTE_BAD = "route_bad"  # the app's dictionary route itself is wrong: demoted at once, not after 3
KEEP_ARMS_S = 365 * 86400  # arms (and preferences) untouched this long are dropped from decide.json
WEAK_AFF = 0.3  # a weak signal counts for an app used at least this much (or one that did the task)
CALL_SCHEMES = ("tel", "facetime", "facetime-audio")  # their default handler places calls

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
    "eventkit": (5.0, 1.0),  # the OS store itself: saved and fetched back
    "script_make": (3.0, 1.0),
    "script_read": (3.0, 1.0),
    "script_send": (3.0, 1.0),
    "script_draft": (3.0, 1.0),
    "mailto": (3.0, 1.5),
    "mailto_ax": (2.0, 2.0),
    "handler_url": (2.5, 2.0),
    "ax_chat": (2.0, 2.0),
    "ax_new": (2.0, 2.0),
    "badge": (2.0, 2.0),
    "file": (4.0, 1.0),
    "ics": (1.5, 2.0),
    "learned_template": (4.0, 1.0),
    "opensearch": (3.0, 1.0),
    "jsonld": (3.0, 1.0),
    "site_form": (2.5, 1.5),
    "site_box": (2.0, 2.0),
    "address_bar": (3.0, 1.5),
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


class Kind(NamedTuple):
    """:func:`decide_kind`'s answer. ``advisory``: Laya's zero-shot guess while the reflex has no
    kind head - a hint (to ask the user, or to order what they named), never a reason to hand the
    request to that kind of app."""

    kind: str
    p: float
    advisory: bool = False


_OTHER = Kind("other", 0.0)
_NO_KIND_HEAD = object()  # _reflex_call: the reflex has no kind head (zero-shot may guess)


def _laya_call(state: Any, questions: dict, stale: threading.Event) -> dict | None:
    """One zero-shot pass on :mod:`neo.reflex.zeroshot` (in a worker thread); None if it can't answer."""
    try:
        from neo.reflex import zeroshot

        return zeroshot.ask(state, questions, LAYA_LOCK_S, stale)
    except Exception as e:  # noqa: BLE001 - Laya must never break a decision
        print(f"[decide] Laya unavailable for this decision ({str(e)[:80]})")
        return None


def _has_kind_head(laya: Any) -> bool:
    """Whether the loaded reflex has a trained kind head: its own flag, else its head's kind entry."""
    flag = getattr(laya, "has_kind_head", None)
    if flag is not None:
        return bool(flag)
    head = getattr(laya, "_head", None)
    return isinstance(head, dict) and head.get("kind") is not None


def _reflex_call(text: str, known: Any, stale: threading.Event) -> Any:
    """In a worker thread: the reflex's Decision for ``text`` (``known`` when the caller has it)
    if the reflex has a kind head; _NO_KIND_HEAD if it has none; None if it is busy or broken."""
    try:
        from neo.reflex import get_laya, laya_lock

        lock = laya_lock()
        # get_laya() takes this same (non-reentrant) lock with no timeout, so only call it once the
        # lock is free: a busy Laya then costs LAYA_LOCK_S, not the whole LAYA_TIMEOUT_S.
        if not lock.acquire(timeout=LAYA_LOCK_S):
            return None  # busy (retraining, loading, another decision)
        lock.release()
        laya = get_laya()
        if laya is None or not _has_kind_head(laya):
            return _NO_KIND_HEAD
        if known is not None:
            return known
        if not lock.acquire(timeout=LAYA_LOCK_S):
            return None
        try:
            return None if stale.is_set() else laya.decide(text)
        finally:
            lock.release()
    except Exception as e:  # noqa: BLE001
        print(f"[decide] the reflex couldn't tell the task kind ({str(e)[:80]})")
        return None


async def _off_loop(fn: Any, *args: Any, timeout: float | None = None) -> Any:
    """``fn(*args, stale)`` in a worker thread, None past ``timeout`` (default LAYA_TIMEOUT_S: a
    model still loading, a busy GPU); a worker still waiting then skips its work."""
    wait = LAYA_TIMEOUT_S if timeout is None else timeout
    if wait <= 0:
        return None  # the budget is spent: start nothing nobody will wait for
    stale = threading.Event()
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args, stale), wait)
    except Exception:  # noqa: BLE001 - timeout or anything else
        return None
    finally:
        stale.set()


async def _ask_laya(state: Any, questions: dict, timeout: float | None = None) -> dict | None:
    return await _off_loop(_laya_call, state, questions, timeout=timeout)


def _zero_shot_can_answer() -> bool:
    try:
        from neo.reflex import zeroshot

        return zeroshot.available()
    except Exception:  # noqa: BLE001
        return False


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


# "open <app>" once the app's name is masked ("an app"): which kind needs no model. The verbs are a
# language table (English, Hinglish, Hindi), never app names.
_OPEN_AN_APP = re.compile(
    r"^(?:please\s+)?(?:open|launch|start|switch\s+to|bring\s+up|show)\s+an\s+app[.!?]*$"
    r"|^an\s+app\s+(?:kholo|khol\s+do|chalu\s+karo|खोलो|खोल\s+दो)[.!?]*$",
    re.I,
)


def mask_apps(text: str, apps: list[App]) -> str:
    """``text`` with every installed app's name the user said replaced by "an app", so the kind
    is read from what they asked, not from which app they named ("message john on an app")."""
    try:
        from neo.decide.inventory import named_span
    except ImportError:
        return text
    for _ in range(4):  # a request names an app or two, not more
        hit = named_span(text, apps)
        if hit is None:
            break
        _app, a, b = hit
        text = f"{text[:a]}an app{text[b:]}"
    return text


async def decide_kind(text: str, decision: Any = None, *, apps: list[App] | None = None) -> Kind:
    """The task kind of ``text`` and its probability, all within LAYA_TIMEOUT_S.

    The reflex's trained kind head decides when it has one: ``decision`` is the reflex's Decision
    for this text if the caller has it, else the reflex is asked (one embedding pass). Its
    abstaining (not an action), or being too busy to answer, gives ('other', 0.0): a head may well
    exist then, so nothing is guessed. Only without a kind head does Laya answer zero-shot, and
    that counts only when it is very sure (:data:`ZS_KIND_P`) and only as advice
    (``advisory=True``): even then it was wrong too often to route a request by.

    With ``apps``, the names of those apps in ``text`` are masked first (:func:`mask_apps`); what
    is then just "open an app" is ``open_app`` without asking anyone.
    """
    if not text or not text.strip():
        return _OTHER
    text = text.strip()
    if apps is not None:
        text = mask_apps(text, apps)
        if _OPEN_AN_APP.match(text):
            return Kind("open_app", 1.0)
    kind = getattr(decision, "kind", None)
    if kind in TASKS:
        return Kind(kind, float(getattr(decision, "kind_p", 0.0) or 0.0))
    end = time.monotonic() + LAYA_TIMEOUT_S
    decision = await _off_loop(_reflex_call, text, decision)
    if decision is None:
        return _OTHER
    if decision is not _NO_KIND_HEAD:
        kind = getattr(decision, "kind", None)
        return Kind(kind, float(getattr(decision, "kind_p", 0.0) or 0.0)) if kind in TASKS else _OTHER
    q = {
        "kind": {
            "type": "choice",
            "instructions": "What does `text` ask the computer to do?",
            "criteria": {k: t.description for k, t in TASKS.items()},
        }
    }
    res = await _ask_laya({"text": text}, q, end - time.monotonic())
    probs = {k: p for k, p in _choice(res, "kind").items() if k in TASKS}
    if not _sure(probs, ZS_KIND_P, ZS_KIND_MARGIN):
        return _OTHER
    kind = max(probs, key=lambda k: probs[k])
    return Kind(kind, float(probs[kind]), advisory=True)


# ---- the learning store (decide.json) ---------------------------------------------------------

_lock = threading.RLock()
_cache: dict[str, tuple[int, dict]] = {}  # path -> (mtime_ns, data)


def _path() -> Path:
    return settings().data_dir / "decide.json"


def _fresh() -> dict:
    return {"version": 2, "arms": {}, "prefs": {}}


def forget() -> int:
    """Erase what NEO learned about this user's apps and preferences (the privacy erase).
    Returns how many arms and preferences were dropped."""
    with _lock:
        data = _load()
        n = sum(len(m) for per_app in data["arms"].values() for m in per_app.values())
        n += sum(len(per_key) for per_key in data["prefs"].values())
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
    prefs = raw.get("prefs")  # version 1 files have none
    for scope, per_key in (prefs.items() if isinstance(prefs, dict) else ()):
        if not isinstance(per_key, dict):
            continue
        for key, pref in per_key.items():
            try:
                a, b = float(pref["a"]), float(pref["b"])
                if not (0 < a < 1e6 and 0 < b < 1e6):
                    continue
                out["prefs"].setdefault(str(scope), {})[str(key)] = {
                    "a": a,
                    "b": b,
                    "t": float(pref.get("t", 0.0)),
                    "up": int(pref.get("up", 0)),
                    "down": int(pref.get("down", 0)),
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
                if reason == ROUTE_BAD:  # the route itself is wrong for this app: stop trying it now
                    arm["fails"] = max(arm["fails"], DEMOTE_AFTER)
        per_method[method] = arm
        _prune(data, now)
        _persist(data)
    if ok or reason not in NO_BLAME:
        _tell_ledger(kind, app_key, ok)


@dataclass
class _Asked:
    """The request :func:`choose` last ordered in this task: what its outcomes are reported as."""

    kind: str
    said: str
    keys: frozenset[str]
    told: set[tuple[str, bool]] = field(default_factory=set)


_asked: ContextVar[_Asked | None] = ContextVar("neo_decide_asked", default=None)


def _tell_ledger(kind: str, key: str, ok: bool) -> None:
    """One settled attempt of a chosen app, to the reflex's outcome ledger when it has one: the
    user's own words choose() was given (``said``), the kind, the app key and ok - never a query,
    tool arguments or results. Each (app, outcome) once per request; an attempt outside a decision
    (a per-site sub-arm, another task, a request without the user's words) is not reported."""
    a = _asked.get()
    if a is None or a.kind != kind or key not in a.keys or (key, ok) in a.told:
        return
    a.told.add((key, ok))
    try:
        from neo.reflex import outcomes

        decided = getattr(outcomes, "decided", None)
        if callable(decided):
            decided(a.said, kind, key, ok)
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
    for scope in list(data["prefs"]):
        per_key = data["prefs"][scope]
        for key in [k for k, v in per_key.items() if now - v["t"] > KEEP_ARMS_S]:
            del per_key[key]
        if not per_key:
            del data["prefs"][scope]


def _persist(data: dict) -> None:
    try:
        _save(data)
    except OSError as e:
        print(f"[decide] couldn't save what was learned ({str(e)[:80]})")
        try:  # still learn for this session: keep it until the file itself changes
            mtime = _path().stat().st_mtime_ns
        except OSError:
            mtime = -1
        _cache[str(_path())] = (mtime, data)


def prior(kind: str, app_key: str, method: str) -> float:
    """Posterior mean of the (kind, app, method) arm: how often this way of doing it works here."""
    with _lock:
        return _mean(_arm(_load(), kind, app_key, method))


def proven(kind: str, key: str) -> bool:
    """Whether this app (or route) has done this kind of task for NEO before."""
    with _lock:
        return _succeeded(_load(), kind, key)


def habitual(kind: str, key: str) -> bool:
    """Whether the last success of this kind was with this app (or route): no need to say where."""
    with _lock:
        per_app = _load()["arms"].get(kind, {})
    last = {k: max((a["last_ok"] for a in arms.values()), default=0.0) for k, arms in per_app.items()}
    best = max(last, key=lambda k: last[k], default=None)
    return best == key and last[key] > 0


# ---- preferences (kinds with Task.prefs) ---------------------------------------------------------
# Kept apart from reliability: an app that worked can still be the wrong choice for this user (or
# for this person). Only a correction and an approved send move them, never a plain success.


def _scope(kind: str, about: str) -> str:
    return f"{kind}@{about}" if about else kind


def _pref_mean(data: dict, scope: str, key: str) -> float:
    p = data["prefs"].get(scope, {}).get(key)
    return p["a"] / (p["a"] + p["b"]) if p else 0.5  # Beta(1, 1): no opinion


def pref_factor(kind: str, key: str, about: str = "", data: dict | None = None) -> float:
    """(0.5 + kind preference) x (0.5 + this person's): exactly 1.0 with no evidence either way."""
    if data is None:
        with _lock:
            data = _load()
    person = _pref_mean(data, _scope(kind, about), key) if about else 0.5
    return (0.5 + _pref_mean(data, kind, key)) * (0.5 + person)


def record_pref(kind: str, key: str, liked: bool, *, about: str = "") -> None:
    """The user chose (``liked``) or turned down this app for this kind, for one person when
    ``about`` (a hashed contact id) is given. Nothing is kept once the user erased NEO's memory
    since the request began."""
    from neo.decide import run

    if run.erased():
        return
    now = _now()
    with _lock:
        data = copy.deepcopy(_load())
        per_key = data["prefs"].setdefault(_scope(kind, about), {})
        p = per_key.get(key) or {"a": 1.0, "b": 1.0, "t": 0.0, "up": 0, "down": 0}
        p["a"], p["b"] = p["a"] * GAMMA, p["b"] * GAMMA
        p["a" if liked else "b"] += 1.0
        p["up" if liked else "down"] += 1
        p["t"] = now
        per_key[key] = p
        _prune(data, now)
        _persist(data)


def correct(kind: str, wrong_key: str, *, about: str = "") -> None:
    """The user redirected a finished task elsewhere ("no, in <app>"): the choice was wrong, the
    app didn't fail, so this moves preference only."""
    record_pref(kind, wrong_key, False, about=about)


def _pref_words(data: dict, kind: str, key: str, about: str) -> str:
    parts = []
    if about and (p := data["prefs"].get(_scope(kind, about), {}).get(key)) and p["up"] + p["down"]:
        parts.append(f"used for this person {p['up']} of {p['up'] + p['down']} times")
    if (p := data["prefs"].get(kind, {}).get(key)) and p["down"]:
        parts.append("you corrected this once" if p["down"] == 1 else f"you corrected this {p['down']} times")
    return "; ".join(parts)


# ---- evidence ---------------------------------------------------------------------------------


def _inventory() -> list[App]:
    from neo.decide.inventory import inventory

    return inventory()


def warm() -> None:
    """Scan the apps in the background now, then make sure the zero-shot checkpoint is on this Mac
    (downloaded, not loaded), so the first request waits for neither."""

    def run() -> None:
        try:
            _inventory()
        finally:
            from neo.reflex import zeroshot

            zeroshot.prefetch()

    threading.Thread(target=run, name="neo-inventory-warm", daemon=True).start()


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
        from neo.decide.sdef import caps, read_caps

        out = frozenset(caps(app.script) | read_caps(app.script))
    except Exception:  # noqa: BLE001
        return frozenset()
    # a text search in a player or a photo library is no web search: only a browser's counts
    return out if "web.search" not in out or _is_browser(app) else out - {"web.search"}


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
# Content types (type vocabulary, not app names) a notes library edits.
_TEXT_TYPES = ("public.plain-text", "net.daringfireball.markdown")


def _library_text_editor(app: App) -> bool:
    edits = {str(t).lower() for t in app.editor_types}
    return (
        any(t in edits for t in _TEXT_TYPES)
        and not app.doc_based
        and _last_segment(app.category) in ("", "productivity")
    )


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
    if name == "intent":
        return arg in app.intents
    if name in ("protocol", "schema"):
        return sig in app.ai_schemas
    if name == "share":
        return arg == "text" and app.share_text
    if name == "library_text_editor":
        return _library_text_editor(app)
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


def _succeeded(data: dict, kind: str, key: str) -> bool:
    return any(a["ok"] for a in data["arms"].get(kind, {}).get(key, {}).values())


def _fits(kind: str, task: Task, app: App, caps: frozenset[str], data: dict, now: float) -> bool:
    """Declared evidence, a weak signal this user backs by use, or (``proven_fit``) a track record."""
    if caps.intersection(task.wants):
        return True
    done_before = _succeeded(data, kind, app.bundle_id)
    for sig in task.signals:
        weak = sig.endswith("~")
        if not _signal(app, sig.rstrip("~")):
            continue
        # e.g. a social app that is really the default call handler, or one this user never opens
        if not weak or (
            not set(CALL_SCHEMES).intersection(app.handles) and (done_before or _affinity(app, now) >= WEAK_AFF)
        ):
            return True
    return task.proven_fit and done_before


def fits_kind(kind: str, app: App) -> bool:
    """Whether the app can do this kind of task, from its evidence, this user's use of it (weak
    signals) and NEO's record with it: no routes, no Laya, no disk scan (fast enough for a key
    press gate)."""
    task = TASKS.get(kind)
    if task is None or app.kind != "app":
        return False
    with _lock:
        data = _load()  # replaced, never changed in place, by a later write: safe to read unlocked
    return _fits(kind, task, app, _app_caps(app), data, _now())


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
                # a driver of several kinds gives this kind's methods only
                return tuple(fn(app, kind) if len(inspect.signature(fn).parameters) > 1 else fn(app))
            except Exception:  # noqa: BLE001 - a driver bug must not sink the whole decision
                return ()
    if app.kind == "route":  # 'route:<framework>.<store>' is driven through that framework
        return (app.bundle_id.removeprefix("route:").split(".", 1)[0],)
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
    pref: str = ""  # the user's own choices, in words ("you corrected this once")


def _describe(task: Task, i: _Info, now: float, short: bool) -> str:
    """What Laya reads about one candidate: all of it evidence, none of it a name list."""
    app = i.app
    if app.kind == "route":  # a store, told by counts only (never its calendars' or items' titles)
        parts = [app.note]
    else:
        parts = [cat] if (cat := _last_segment(app.category).replace("-", " ")) else []
        can = [t.split(".", 1)[-1].replace("_", " ") for t in task.wants if t in i.caps]
        if can and not short:
            parts.append("can " + ", ".join(can))
        parts.append(_usage_words(app, now))
        if app.running and not short:
            parts.append("open now")
    if i.rec != "not tried yet" or not short:
        parts.append(i.rec)
    parts.append(i.pref)
    return "; ".join(p for p in parts if p)


def _info(
    kind: str,
    task: Task,
    app: App,
    caps: frozenset[str],
    data: dict,
    now: float,
    *,
    aff: float | None = None,
    used: bool = False,
) -> _Info | None:
    """``aff`` stands in for the app's own affinity and ``used`` counts it as used (a route whose
    store NEO hasn't been allowed to read yet)."""
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
    aff = _affinity(app, now) if aff is None else aff
    succeeded = _succeeded(data, kind, key)
    used = bool(
        used
        or app.running
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
    parts.append(_describe(task, i, now, short=False))
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


def _embedder_ready() -> bool:
    """NEO's memory embedder is loaded: a decision never starts (or waits for) its load."""
    try:
        from neo.memory import embed

        return embed.ready()
    except Exception:  # noqa: BLE001
        return False


def _shortlist(task: Task, text: str, infos: list[_Info], now: float, stale: threading.Event) -> list[_Info]:
    """The LAYA_POOL apps whose evidence reads most like the request, by NEO's pinned MiniLM (the
    memory embedder, loaded at NEO's start: CPU, ~13 ms for 25 apps); [] when it isn't available.

    Over 24 used apps (32 requests) the right one was among the 6 kept for 0.84 of requests;
    Laya's own mean-pooled encoder (laya.shortlist.embed_fn_from_agent) managed 0.31, about
    chance, in 150 ms of GPU time."""
    from neo.memory import embed

    if stale.is_set():
        return []
    opts = [f"{i.label}: {_describe(task, i, now, short=True)}" for i in infos]
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


async def _laya_order(
    task: Task, text: str, used: list[_Info], unused: list[_Info], now: float
) -> tuple[list[tuple[_Info, float, float]], list[_Info]]:
    """((info, score, laya_p) best first, the unused candidates left for the tail). Laya's pick
    counts only when it is sure; at ``laya_min_used`` 1 capable unused apps join its question, and
    one it is sure about leaves the tail."""
    by_prior = sorted(used, key=lambda i: -i.prior)
    extra = sorted(unused, key=lambda i: -i.prior) if task.laya_min_used < 2 else []
    cands = by_prior + extra
    if len(used) < max(1, task.laya_min_used) or len(cands) < 2 or not _zero_shot_can_answer():
        return [(i, i.prior, 0.0) for i in by_prior], unused  # nothing to choose between, or nobody to ask
    _labels(cands)
    pool = cands[:LAYA_POOL]
    end = time.monotonic() + LAYA_TIMEOUT_S  # the shortlist and the question share one budget
    if len(cands) > SHORTLIST_ABOVE and _embedder_ready():
        pool = await _off_loop(_shortlist, task, text, cands, now, timeout=end - time.monotonic()) or pool
    # The request is the whole state; each option is its app's short evidence. Restating every
    # candidate's evidence in the state (as JSON) was worse and costlier on decide's own questions:
    # Laya's pick right 0.59 vs 0.85, 277 vs 82 tokens, 82 vs 35 ms a question.
    q = {
        "app": {
            "type": "choice",
            "instructions": "Which app on this Mac should do the request?",
            "criteria": {i.label: _describe(task, i, now, short=True) for i in pool},
        }
    }
    probs = _choice(await _ask_laya({"request": text}, q, end - time.monotonic()), "app")
    if not _sure(probs):  # unsure or unavailable: the prior orders them
        return [(i, i.prior, probs.get(i.label, 0.0)) for i in by_prior], unused
    scored = []
    for i in pool:
        p = probs.get(i.label, 0.0)
        scored.append((i, (max(p, 1e-3) ** LAYA_EXP) * (max(i.prior, 1e-6) ** PRIOR_EXP), p))
    scored.sort(key=lambda t: -t[1])
    in_pool = {id(i) for i in pool}
    rest = [(i, i.prior, 0.0) for i in by_prior if id(i) not in in_pool]
    return scored + rest, [i for i in unused if id(i) not in in_pool]


def _routes(kind: str, task: Task, exclude: frozenset[str]) -> list[tuple[App, bool]]:
    """(route candidate, whether its use is still unknown) for each usable route of the kind
    (see neo.decide.routes; counting a store's items can take a moment: called off the loop)."""
    from neo.decide import routes

    out = []
    for route in task.routes:
        try:
            app = routes.route_app(kind, route)
            if app is not None and app.bundle_id not in exclude:
                out.append((app, routes.pending(app)))
        except Exception as e:  # noqa: BLE001 - an unusable store is simply no candidate
            print(f"[decide] route {route} unavailable ({str(e)[:60]})")
    return out


def _store_view(kind: str, key: str) -> bool:
    """The app is known to write into a route's store (its items showed up there)."""
    try:
        from neo.decide import learned

        return learned.get("store_views", f"{kind}:{key}") is not None
    except Exception:  # noqa: BLE001
        return False


async def choose(
    kind: str,
    text: str,
    apps: list[App] | None = None,
    *,
    said: str = "",
    about: str = "",
    exclude: frozenset[str] = frozenset(),
) -> list[Candidate]:
    """Ordered attempts for this request on this Mac (see the module docstring). ``text`` is what
    the decision reads (the user's words, or the query when they aren't known); ``said`` is the
    user's own words when the caller has them - the only text the outcome ledger is ever told.
    ``about`` (a hashed contact id) brings in this person's preferences; ``exclude`` drops the
    apps (keys) already tried for this request."""
    _asked.set(None)  # a choose() that fails leaves nothing for a later outcome to be filed under
    task = TASKS.get(kind) or TASKS["other"]
    now = _now()
    if apps is None:
        try:
            apps = await asyncio.to_thread(_inventory)  # the first scan takes a while: off the loop
        except Exception as e:  # noqa: BLE001
            print(f"[decide] app inventory failed ({str(e)[:80]})")
            apps = []
    me = _self_ids()
    apps = _dedupe([a for a in apps if a.bundle_id.lower() not in me and a.bundle_id not in exclude])
    with _lock:
        data = copy.deepcopy(_load())
    named = _named(text, apps)
    site = site_hint(text, apps) if task.fallback == WEB and named is None else ""
    routes = await asyncio.to_thread(_routes, kind, task, exclude) if task.routes else []

    first: list[Candidate] = []
    used: list[_Info] = []
    unused: list[_Info] = []
    weak: list[_Info] = []
    fitting: list[_Info] = []
    for app in apps:
        caps = _app_caps(app)
        is_named = named is not None and app.bundle_id == named.bundle_id
        fits = _fits(kind, task, app, caps, data, now)
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
        fitting.append(i)
        if routes and _store_view(kind, app.bundle_id):
            continue  # what it writes lands in the route's store: the route is the way there
        (weak if i.weak else used if i.used else unused).append(i)
    for r, pending in routes:
        # Before NEO may read the store its use is unknown: it counts as much as the most-used app
        # that fits (every app of this kind here shows that same store).
        aff = max((i.aff for i in fitting), default=0.0) if pending else None
        i = _info(kind, task, r, frozenset(), data, now, aff=aff, used=pending and any(f.used for f in fitting))
        if i is not None:
            (weak if i.weak else used if i.used else unused).append(i)
    if task.prefs:
        for i in (*used, *unused, *weak):
            i.prior *= pref_factor(kind, i.app.bundle_id, about, data)
            i.pref = _pref_words(data, kind, i.app.bundle_id, about)

    fallback: list[Candidate] = []
    fb_mean = 0.0
    if task.fallback and task.fallback not in exclude:
        fb = task.fallback
        fb_mean = _mean(_arm(data, kind, fb, fb))
        on = f" ({site})" if site else ""
        fallback.append(
            Candidate(None, fb_mean, 0.0, f"fallback: {fb}{on}; {_record_words(data, kind, fb)}", (fb,), site=site)
        )
    # A used app goes before the web fallback when its prior beats the fallback's (see WEB_AFF),
    # when it has done this before at least as reliably, or when Laya is sure it's the one; the
    # others (e.g. a player opened once, weeks ago, with nothing in its library) go after it. A
    # local fallback (a file) is only what's left when the apps fail: every used app goes first.
    bar = fb_mean * (AFF_FLOOR + (1 - AFF_FLOOR) * WEB_AFF)
    before: list[Candidate] = []
    after: list[Candidate] = []
    ordered, unused = await _laya_order(task, text, used, unused, now)
    for n, (i, s, p) in enumerate(ordered):
        keep = (
            not fallback
            or task.fallback != WEB
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
    if said.strip():
        _asked.set(_Asked(kind, " ".join(said.split()), frozenset(c.key for c in out)))
    return out


__all__ = [
    "TASKS",
    "DRIVERS",
    "WEB",
    "Task",
    "Candidate",
    "Kind",
    "app_key",
    "decide_kind",
    "mask_apps",
    "choose",
    "site_hint",
    "warm",
    "record",
    "prior",
    "proven",
    "habitual",
    "fits_kind",
    "pref_factor",
    "record_pref",
    "correct",
    "forget",
]
