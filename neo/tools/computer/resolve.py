"""Which on-screen control does the user mean? A cascade that resolves UI commands by meaning, in
English, Hinglish and Hindi, without a model call where it is sure enough.

    resolve(utterance, candidates, want="press") -> Resolved | Ambiguous

Stages, in order (the first that answers wins):

1. **lexical** — the control's name inside the request (phrases.strip_frame: "new game pe click
   karo" → "new game") is exactly one label, or a whole word of exactly one distinct label.
   Today's click_text rule plus the multilingual frame stripper; offline precision 0.99-1.00.
2. **alias** — a wording that, once handed to the model, ended on this label and was verified
   (neo.reflex.uimemory aliases, per app).
3. **os_strings** — macOS's own English↔Hindi UI strings (neo.decide.uistrings): a Devanagari
   request whose content words hold all of a label's OS Hindi form's, or a Latin request that
   names a Hindi label's English source.
4. **meaning** (Latin script) — MiniLM's top 6 by cosine, and Laya's typed-decisions checkpoint
   choosing among them (neo.reflex.zeroshot): acted on only when Laya picks MiniLM's top-1 with
   p ≥ ``resolver_laya_p`` (0.9; 0.95 above 60 candidates). Laya only when it is loaded already:
   a cold or busy Laya is never waited for (MiniLM alone then needs a far bigger margin).
5. **devanagari** — a trained Devanagari grounding head (not built yet: P2).
6. Otherwise :class:`Ambiguous`: at most 6 candidates for the model as a multiple choice.

**Live or shadow.** Only the lexical stage acts by default (``settings.resolver = "shadow"``). The
later stages run and are logged (neo.decide.shadow ``ui_resolve``) but hand off; each is promoted
on its own, once it passes ≥97% precision overall and ≥95% per script on
tests/data/ground_eval.jsonl and on the user's real requests (``resolver = "live"`` plus the
stage in ``resolver_promoted``). ``resolver = "off"``: the lexical stage alone, nothing logged.

**Safety** (on top of the frozen confirmation gate, which still judges every label pressed):

* Only the user's own words get the by-meaning stages (``origin == "user"``); a model's or Gemini
  Live's words get the lexical stage only.
* A request that sounds destructive with a label that doesn't commit → abstain (hand off).
* A label the gate can't read (not plain Latin) without an OS back-translation → ``ask``: the
  caller asks the user before pressing, whatever the gate says (fail closed while the OS string
  index isn't built).
* Candidates come from the caller, already without NEO's own and protected apps' elements (the
  ``skip`` predicate of :func:`candidates` is where the caller's gate helpers go); disabled,
  empty-label, off-screen and secure elements never become candidates.
* Screen labels are untrusted data: they are only ever compared with the user's words, and a
  handoff lists them as quoted data with element ids, never coordinates.

Nothing here presses anything, loads a model, starts a scan or touches the network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from neo.tools.computer import phrases

log = logging.getLogger(__name__)

__all__ = [
    "Ambiguous",
    "Candidate",
    "Resolved",
    "STAGES",
    "after_press",
    "aresolve",
    "candidates",
    "choose",
    "forget",
    "ground",
    "handoff_choices",
    "handoff_text",
    "label_commits",
    "live_stages",
    "matches",
    "mode",
    "os_index",
    "outcome",
    "resolve",
    "signature",
    "utterance_of",
]

Want = Literal["press", "field", "menu"]
STAGES = ("lexical", "alias", "os_strings", "meaning", "devanagari")
SHORTLIST = 6  # at most this many choices in a handoff (the MindAct multiple choice)
# Laya's probability for its pick, when it equals MiniLM's top-1. On the benchmark's 914 Latin rows
# the lexical and OS stages leave (real MiniLM + typed-decisions, offline): p ≥ 0.7 acted on 47% at
# 93.5% precision, 0.9 on 28% at 98.4% (100% / 97.5% / 96.5% at 12 / 50 / 200 elements); above
# CROWDED candidates 0.95 gave 29/29. Chosen on the benchmark itself, so optimistic: the stage
# stays in shadow until the user's own requests say the same (neo.reflex.evaluate.shadow_report).
LAYA_P = 0.9
LAYA_P_CROWDED = 0.95
CROWDED = 60
# MiniLM without Laya (Laya cold, busy or off). Held-out margin bars of 0.08-0.175 gave only
# 0.95-0.97 precision offline, so alone it needs far more than that; it is logged in shadow.
MINILM_ONLY_MARGIN = 0.25
MINILM_ONLY_SIM = 0.55
LAYA_WAIT_LIVE_S = 1.5  # engine.LAYA_LOCK_S: the shared Laya lock, when the stage acts
LAYA_WAIT_SHADOW_S = 0.2  # only logging: never hold up the handoff for long
BUDGET_S = 0.6  # the by-meaning stages together; past it they are skipped (logged as such)
_VEC_CAP = 20000  # label vectors kept in RAM (never on disk: labels can be the user's content)

# Roles a press / a field / a menu command may land on, best first (click_text's order: a control
# before a row, a text or a picture of the same name).
FIELD_ROLES = ("AXTextField", "AXSearchField", "AXComboBox", "AXTextArea")
_WEAK_ROLES = ("AXCell", "AXRow", "AXStaticText", "AXImage", "AXHeading")
PRESS_ROLES = (
    "AXButton", "AXLink", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXCheckBox", "AXRadioButton",
    "AXPopUpButton", "AXMenuButton", "AXDisclosureTriangle", "AXSegment", "AXToolbarButton", "AXSlider",
    "AXIncrementor", *FIELD_ROLES, *_WEAK_ROLES,
)  # fmt: skip
MENU_ROLES = ("AXMenuItem", "AXMenuBarItem")
# A control proper (a text, a row or a picture of the same name elsewhere is not a second one).
_CONTROL_ROLES = frozenset(PRESS_ROLES) - set(_WEAK_ROLES)
_TEXT_ROLES = frozenset({"AXStaticText", "AXImage", "AXHeading"})
# UI chrome: an app's own words, which a log line may carry. Never rows, cells, static text, links
# or field values — those can be the user's content.
CHROME_ROLES = frozenset(
    {"AXButton", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
     "AXMenuButton", "AXDisclosureTriangle", "AXSegment", "AXToolbarButton"}
)  # fmt: skip
_ROLE_WORD = {
    "AXButton": "button", "AXLink": "link", "AXMenuItem": "menu item", "AXMenuBarItem": "menu",
    "AXTab": "tab", "AXCheckBox": "checkbox", "AXRadioButton": "option", "AXPopUpButton": "pop-up menu",
    "AXMenuButton": "menu button", "AXCell": "cell", "AXRow": "row", "AXStaticText": "text", "AXImage": "image",
    "AXTextField": "text field", "AXTextArea": "text area", "AXSearchField": "search field",
    "AXComboBox": "combo box",
}  # fmt: skip


# ---- what the resolver sees ----------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One element the request could mean: its label (the app's words, untrusted), role, whether it
    sits in an open sheet or dialog, a menu path for menu items, and the element itself (``ref``,
    handed back untouched)."""

    label: str
    role: str = "AXButton"
    ref: Any = field(default=None, compare=False, repr=False)
    modal: bool = False
    path: tuple[str, ...] = ()
    box: tuple[float, float, float, float] | None = field(default=None, compare=False, repr=False)
    id: str = ""

    @property
    def ui_chrome(self) -> bool:
        return self.role in CHROME_ROLES

    def describe(self) -> str:
        where = " in the open dialog" if self.modal else ""
        path = f" ({' › '.join(self.path)})" if self.path else ""
        return f"{_ROLE_WORD.get(self.role, self.role.removeprefix('AX').lower())}{path}{where}"


@dataclass
class Resolved:
    """The control the request means. ``stage``: which stage found it. ``p``: its confidence (1.0
    for a lexical match). ``commits``: the label sends, buys, deletes… (the frozen gate asks
    anyway). ``ask``: why the caller must ask the user before pressing even when the gate
    wouldn't ("" = no reason): "unreadable_label" (not plain Latin and no OS back-translation) or
    "commits" (it commits by meaning — a back-translation, a mail "Forward" — though the gate's
    English rule doesn't see it)."""

    candidate: Candidate
    stage: str
    p: float = 1.0
    commits: bool = False
    ask: str = ""
    rid: str = ""
    shadow: dict[str, Any] = field(default_factory=dict)

    @property
    def element(self) -> Any:
        return self.candidate.ref

    @property
    def label(self) -> str:
        return self.candidate.label


@dataclass
class Ambiguous:
    """Not sure enough to act: at most 6 candidates for the model to choose from (possibly none),
    and why. ``reason``: "nothing" (no candidates), "no_match", "same_label" (the same label on two
    controls outside a dialog), "several" (a phrase in several labels), "destructive_mismatch",
    "unsure" (the by-meaning stages didn't agree or are in shadow), "off"."""

    shortlist: list[Candidate]
    reason: str
    rid: str = ""
    shadow: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # `if (r := resolve(...)):` reads as "resolved"
        return False


Result = Resolved | Ambiguous


class StringIndex(Protocol):
    """The OS's own UI strings (neo.decide.uistrings): what the resolver asks of it."""

    def ready(self) -> bool: ...

    def forms(self, label: str) -> list[str]:  # an English label's OS translations (Hindi)
        ...

    def sources(self, label: str) -> list[str]:  # a localized label's English source strings
        ...


# ---- settings (Step 0 adds the keys; until then the defaults) --------------------------------------
def _setting(name: str, default: Any) -> Any:
    try:
        from neo.config import settings

        return getattr(settings(), name, default)
    except Exception:  # noqa: BLE001 - a broken config must not break a click
        return default


def mode() -> str:
    """``settings.resolver``: live | shadow (default) | off."""
    m = str(_setting("resolver", "shadow") or "shadow").strip().lower()
    return m if m in ("live", "shadow", "off") else "shadow"


def live_stages() -> frozenset[str]:
    """The stages that act. Lexical always (it is today's click_text rule); a later stage only in
    "live" mode and only when listed in ``settings.resolver_promoted`` (a comma list or a list)."""
    if mode() != "live":
        return frozenset({"lexical"})
    raw = _setting("resolver_promoted", ())
    names = raw.split(",") if isinstance(raw, str) else list(raw or ())
    return frozenset({"lexical"} | {str(n).strip() for n in names if str(n).strip() in STAGES})


def _laya_p() -> float:
    try:
        return min(0.99, max(0.5, float(_setting("resolver_laya_p", LAYA_P))))
    except (TypeError, ValueError):
        return LAYA_P


# ---- candidates --------------------------------------------------------------------------------
def _roles(want: Want) -> Collection[str]:
    return FIELD_ROLES if want == "field" else MENU_ROLES if want == "menu" else PRESS_ROLES


def _box(e: Any) -> tuple[float, float, float, float] | None:
    vis = getattr(e, "visible", None)
    if isinstance(vis, tuple) and len(vis) == 4:
        return tuple(float(v) for v in vis)  # type: ignore[return-value]
    try:
        return (float(e.x), float(e.y), float(e.w), float(e.h))
    except (AttributeError, TypeError, ValueError):
        return None


def _meets(box, frame) -> bool:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    if frame is None:
        return True
    fx, fy, fw, fh = frame
    return x + w > fx and y + h > fy and x < fx + fw and y < fy + fh


def _secure(e: Any) -> bool:
    return "secure" in f"{getattr(e, 'role', '')} {getattr(e, 'subrole', '')}".lower()


def candidates(
    els: Iterable[Any],
    frame: tuple[float, float, float, float] | None = None,
    *,
    want: Want = "press",
    modal: Callable[[Any], bool] | Collection[str] | None = None,
    skip: Callable[[Any], bool] | None = None,
) -> list[Candidate]:
    """The elements a request may land on (ax.Element-like objects): enabled, on screen (inside
    ``frame`` when given), a role that fits ``want``, a label, and not ``skip(e)`` — where the
    caller's gate helpers go (NEO's own and protected apps' elements). A field's label is its
    title or description, never its value (the user's text). ``modal``: which elements sit in an
    open sheet or dialog (a predicate or a set of element ids; else the element's own ``sheet`` or
    ``modal`` flag)."""
    roles = _roles(want)
    out: list[Candidate] = []
    for e in els:
        role = str(getattr(e, "role", "") or "")
        if role not in roles or not getattr(e, "enabled", True) or _secure(e):
            continue
        box = _box(e)
        if box is None or not _meets(box, frame):
            continue
        if skip is not None:
            try:
                if skip(e):
                    continue
            except Exception:  # noqa: BLE001 - a check that can't say keeps the element out
                continue
        title = str(getattr(e, "title", "") or "")
        desc = str(getattr(e, "desc", "") or getattr(e, "description", "") or "")
        value = "" if want == "field" else str(getattr(e, "value", "") or "")
        label = " ".join((title or desc or value).split())[:120]
        if not label:
            continue
        eid = str(getattr(e, "id", "") or "")
        if callable(modal):
            try:
                m = bool(modal(e))
            except Exception:  # noqa: BLE001
                m = False
        elif modal is not None:
            m = eid in modal
        else:
            m = bool(getattr(e, "sheet", False) or getattr(e, "modal", False))
        path = tuple(str(p) for p in (getattr(e, "path", ()) or ()))
        out.append(Candidate(label, role, e, m, path, box, eid))
    return out


def _as_candidates(items: Sequence[Any], want: Want) -> list[Candidate]:
    out: list[Candidate] = []
    for c in items:
        if isinstance(c, Candidate):
            out.append(c)
        elif c is not None:
            out += candidates([c], want=want)
    return out


# ---- the OS string index ----------------------------------------------------------------------------
class _UIStrings:
    """neo.decide.uistrings behind :class:`StringIndex`, whatever its lookups are called. Asks only
    whether it is ready and looks words up: it never starts a build (library mode runs no scans)."""

    def __init__(self, mod: Any) -> None:
        self._m = mod
        self._forms = next((getattr(mod, n) for n in ("forms", "hindi_forms", "translations", "localized") if callable(getattr(mod, n, None))), None)
        self._sources = next((getattr(mod, n) for n in ("english_sources", "sources") if callable(getattr(mod, n, None))), None)

    def ready(self) -> bool:
        fn = getattr(self._m, "ready", None)
        try:
            return bool(fn()) if callable(fn) else False
        except Exception:  # noqa: BLE001
            return False

    def forms(self, label: str) -> list[str]:
        return _lookup(self._forms, label)

    def sources(self, label: str) -> list[str]:
        return _lookup(self._sources, label)


def _lookup(fn: Callable | None, label: str) -> list[str]:
    if fn is None:
        return []
    try:
        got = fn(label)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(got, str):
        return [got]
    return [str(x) for x in (got or ()) if x]


def os_index() -> StringIndex | None:
    """The OS string index when it is built (neo.decide.uistrings), else None."""
    try:
        from neo.decide import uistrings
    except Exception:  # noqa: BLE001 - not built on this branch yet, or no macOS strings
        return None
    idx = _UIStrings(uistrings)
    return idx if idx.ready() else None


# Labels whose press has consequences the gate's English regex doesn't name (yet): a mail forward
# sends, a restart or quit loses work, "Don't Save" discards it. Only ever adds asks.
_MORE_COMMITS = re.compile(
    r"\b(?:forward|restart|shut\s*down|force\s+quit|quit|log\s*out|revert|don'?t\s+save|discard|overwrite|"
    r"replace\s+all|erase)\b",
    re.I,
)


def label_commits(label: str, index: StringIndex | None = None) -> bool:
    """Does pressing this label send, buy, delete, grant, quit…? The gate's English rules on the
    label, on its OS English sources (a Hindi "भेजें" is "Send"), the destructive words in all
    three languages, and :data:`_MORE_COMMITS`. Only ever stricter than policy.commits."""
    from neo.reflex.schema import sounds_destructive
    from neo.security import policy

    def one(s: str) -> bool:
        return policy.commits(s) or sounds_destructive(s) or bool(_MORE_COMMITS.search(s))

    if one(label):
        return True
    return index is not None and any(one(s) for s in index.sources(label)[:8])


def _gate_reads(label: str) -> bool:
    """Would the frozen gate ask about this label by itself (its English rule)?"""
    from neo.security import policy

    return policy.commits(label)


def _readable(label: str, index: StringIndex | None) -> bool:
    """Can the gate judge this label? Plain Latin it can; anything else only through an OS
    back-translation to English."""
    if phrases.latin_only(label):
        return True
    return index is not None and bool(index.sources(label))


# ---- the stages --------------------------------------------------------------------------------------
_B = r"(?<![\wऀ-ॿ])"
_E = r"(?![\wऀ-ॿ])"


def _whole(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"{_B}{re.escape(phrase)}{_E}")


def _phrase_ok(phrase: str) -> bool:
    words = phrase.split()
    return bool(words) and len(phrase) >= 2 and not all(w in phrases._VAGUE for w in words)


def _same_control(a: Candidate, b: Candidate) -> bool:
    """Two elements of one control (a button and the text inside it): one's box holds the other's
    centre."""
    if a.box is None or b.box is None:
        return False
    for p, q in ((a.box, b.box), (b.box, a.box)):
        cx, cy = q[0] + q[2] / 2, q[1] + q[3] / 2
        if p[0] <= cx <= p[0] + p[2] and p[1] <= cy <= p[1] + p[3]:
            return True
    return False


def _rank(c: Candidate) -> tuple[int, float]:
    order = PRESS_ROLES
    area = c.box[2] * c.box[3] if c.box else 0.0
    return (order.index(c.role) if c.role in order else len(order), area)


def _pick(hits: list[Candidate]) -> Candidate | str:
    """One control out of lexical hits, or why not ("same_label" / "several")."""
    modal = [h for h in hits if h.modal]
    hits = modal or hits
    groups: list[list[Candidate]] = []
    for h in sorted(hits, key=_rank):
        for g in groups:
            if _same_control(g[0], h):
                g.append(h)
                break
        else:
            groups.append([h])
    best = [g[0] for g in groups]
    if len(best) == 1:
        return best[0]
    controls = [b for b in best if b.role in _CONTROL_ROLES]
    if len(controls) == 1:
        return controls[0]  # a button, and a heading or a picture of the same name elsewhere
    return "same_label" if len({phrases.norm(b.label) for b in best}) == 1 else "several"


def _lexical(phrase: str, cands: list[Candidate]) -> Candidate | str | None:
    """Today's click_text rule on the stripped phrase: an exact label, else the phrase as a whole
    word of exactly one distinct label. A Candidate, a reason to ask, or None (nothing)."""
    if not _phrase_ok(phrase):
        return None
    exact = [c for c in cands if phrases.norm(c.label) == phrase]
    if exact:
        return _pick(exact)
    w = _whole(phrase)
    # A few words inside a longer text or heading ("back" in "Welcome back") are no control's name:
    # a partial match counts on a control, a row or a cell only (0.9965 → 0.999 precision offline).
    hits = [c for c in cands if c.role not in _TEXT_ROLES and w.search(phrases.norm(c.label))]
    if not hits:
        return None
    if len({phrases.norm(c.label) for c in hits}) > 1:
        modal = [h for h in hits if h.modal]
        if modal and len({phrases.norm(c.label) for c in modal}) == 1:
            return _pick(modal)
        return "several"
    return _pick(hits)


def _os_strings(utterance: str, phrase: str, cands: list[Candidate], index: StringIndex) -> Candidate | str | None:
    """A Devanagari request against each label's OS Hindi forms (every content word of a form in
    the request), or a Latin request against a localized label's English sources."""
    hits: list[Candidate] = []
    if phrases.script(utterance) == "devanagari":
        want = phrases.content_words(phrase)
        if not want:
            return None
        for c in cands:
            for v in [c.label, *index.forms(c.label)[:6]]:
                if phrases.script(v) != "devanagari":
                    continue
                have = phrases.content_words(v)
                if have and have <= want:
                    hits.append(c)
                    break
    else:
        if not _phrase_ok(phrase):
            return None
        w = _whole(phrase)
        for c in cands:
            if phrases.latin_only(c.label):
                continue
            srcs = [phrases.norm(s) for s in index.sources(c.label)[:8]]
            if phrase in srcs or any(w.search(s) for s in srcs):
                hits.append(c)
    if not hits:
        return None
    if len({phrases.norm(c.label) for c in hits}) > 1:
        return "several"
    return _pick(hits)


class _Vectors:
    """Label vectors by label string, in RAM only (labels can be the user's content)."""

    def __init__(self, cap: int = _VEC_CAP) -> None:
        self._cap = cap
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mu = threading.Lock()
        self._owner: Any = None  # the embed function the vectors came from

    def get(self, texts: list[str], embed: Callable[[list[str]], Any]) -> np.ndarray | None:
        with self._mu:
            if self._owner is not embed:
                self._d.clear()
                self._owner = embed
            todo = [t for t in dict.fromkeys(texts) if t not in self._d]
        if todo:
            got = embed(todo)
            if got is None:
                return None
            got = np.asarray(got, dtype=np.float32)
            if got.ndim != 2 or got.shape[0] != len(todo):
                return None
            with self._mu:
                for t, v in zip(todo, got, strict=True):
                    self._d[t] = v
                    self._d.move_to_end(t)
                while len(self._d) > self._cap:
                    self._d.popitem(last=False)
        with self._mu:
            try:
                return np.stack([self._d[t] for t in texts])
            except KeyError:  # evicted meanwhile by a huge window
                return None

    def clear(self) -> None:
        with self._mu:
            self._d.clear()


_vectors = _Vectors()


def _default_embed() -> Callable[[list[str]], Any] | None:
    """NEO's MiniLM (neo.memory.embed) when it is loaded already: never starts (or waits for) a load."""
    try:
        from neo.memory import embed
    except Exception:  # noqa: BLE001
        return None
    if not embed.ready():
        return None
    return embed.embed_texts


def _probs(res: Any, qid: str) -> dict[str, float]:
    if not isinstance(res, dict):
        return {}
    ans = (res.get("answers") or {}).get(qid) or {}
    probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
    if not any(probs.values()) and ans.get("choice") is not None:
        probs = {str(ans["choice"]): 1.0}
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()} if z > 0 else {}


def _laya_choice(request: str, options: list[str], instructions: str, wait_s: float) -> tuple[int | None, float]:
    """Laya's typed-decisions choice among ``options`` for ``request``: (index, p), or (None, 0)
    when it isn't loaded (never waited for), busy past ``wait_s``, or off."""
    try:
        from neo.reflex import zeroshot
    except Exception:  # noqa: BLE001
        return None, 0.0
    if not options or not zeroshot.loaded():
        return None, 0.0
    keys: list[str] = []
    crit: dict[str, str] = {}
    for i, lab in enumerate(options):
        k = lab if lab not in crit else f"{lab} ({i})"
        keys.append(k)
        crit[k] = f"the on-screen control labelled “{lab}”"
    q = {"el": {"type": "choice", "instructions": instructions, "criteria": crit}}
    try:
        res = zeroshot.ask({"request": request}, q, wait_s)
    except Exception as e:  # noqa: BLE001 - Laya must never break a click
        log.info(f"[resolve] Laya unavailable ({str(e)[:60]})")
        return None, 0.0
    probs = _probs(res, "el")
    if not probs:
        return None, 0.0
    pick = max(probs, key=probs.__getitem__)
    return (keys.index(pick) if pick in keys else None), float(probs[pick])


Chooser = Callable[[str, list[str], float], tuple[int | None, float]]


def _default_choose(request: str, options: list[str], wait_s: float) -> tuple[int | None, float]:
    return _laya_choice(request, options, "Which on-screen control does the request mean?", wait_s)


@dataclass
class _Meaning:
    top: list[int]  # candidate indices, MiniLM's order
    sims: list[float]
    laya: int | None = None  # the candidate index Laya picked
    laya_p: float = 0.0
    pick: int | None = None  # what the stage would press
    p: float = 0.0
    how: str = ""  # "agree" | "minilm_only" | ""


def _meaning(
    utterance: str,
    phrase: str,
    cands: list[Candidate],
    embed: Callable[[list[str]], Any],
    choose: Chooser,
    wait_s: float,
) -> _Meaning | None:
    labels = [c.label for c in cands]
    lv = _vectors.get(labels, embed)
    if lv is None:
        return None
    qv = embed([phrase or utterance])
    if qv is None:
        return None
    q = np.asarray(qv, dtype=np.float32).reshape(-1)
    if q.shape[0] != lv.shape[1]:
        return None
    sims = lv @ q
    order = [int(i) for i in np.argsort(-sims)]
    top: list[int] = []
    seen: set[str] = set()
    for i in order:  # one option per distinct label: a choice between twins says nothing
        k = phrases.norm(labels[i])
        if k in seen:
            continue
        seen.add(k)
        top.append(i)
        if len(top) == SHORTLIST:
            break
    m = _Meaning(top, [float(sims[i]) for i in top])
    li, lp = choose(utterance, [labels[i] for i in top], wait_s)
    if li is not None and 0 <= li < len(top):
        m.laya, m.laya_p = top[li], lp
        bar = LAYA_P_CROWDED if len(cands) > CROWDED else _laya_p()
        if m.laya == top[0] and lp >= bar:
            m.pick, m.p, m.how = top[0], lp, "agree"
    elif len(m.sims) >= 1:
        margin = m.sims[0] - (m.sims[1] if len(m.sims) > 1 else 0.0)
        if margin >= MINILM_ONLY_MARGIN and m.sims[0] >= MINILM_ONLY_SIM:
            m.pick, m.p, m.how = top[0], float(m.sims[0]), "minilm_only"
    return m


# ---- the cascade ---------------------------------------------------------------------------------------
@dataclass
class _Run:
    rid: str
    utterance: str
    phrase: str
    want: str
    origin: str
    cands: list[Candidate]
    t0: float
    picks: dict[str, int | None] = field(default_factory=dict)  # stage -> candidate index it chose
    notes: dict[str, Any] = field(default_factory=dict)


def resolve(
    utterance: str,
    items: Sequence[Any],
    *,
    want: Want = "press",
    origin: str = "user",
    destructive: bool | None = None,
    app: str = "",
    index: StringIndex | None | Literal[False] = None,
    embed: Callable[[list[str]], Any] | None | Literal[False] = None,
    choose: Chooser | None | Literal[False] = None,
    alias: Callable[[str, str], str | None] | None | Literal[False] = None,
    stages: Collection[str] | None = None,
    log_it: bool = True,
) -> Result:
    """The control ``utterance`` (the user's whole request, or a label) means among ``items``
    (Candidates, or ax.Element-like objects run through :func:`candidates`).

    ``origin``: who chose the words ("user" gets every stage; a model's or Live's words the lexical
    one). ``destructive``: the reflex's destructive head, OR'd with the words' own. ``app``: the
    working app's bundle id (per-app aliases). ``index`` / ``embed`` / ``choose`` / ``alias``:
    the OS string index, MiniLM, Laya's choice and the alias lookup — None for NEO's own (only when
    already loaded or built), False for none. ``stages``: the stages that act (default
    :func:`live_stages`); the others run in shadow when the mode isn't "off"."""
    m = mode()
    cands = _as_candidates(items, want)
    phrase = phrases.norm(phrases.strip_frame(utterance))
    run = _Run(secrets.token_hex(4), utterance, phrase, want, origin, cands, time.perf_counter())
    acting = frozenset(stages) if stages is not None else live_stages()
    user = origin == "user"
    shadow_ok = user and m != "off"
    # The index is evidence for the safety rules whoever chose the words; its stage is the user's.
    if index is False:
        idx: StringIndex | None = None
    elif index is None:
        idx = os_index()
    else:
        idx = index if index.ready() else None
    danger = bool(destructive)
    if not danger:
        from neo.reflex.schema import sounds_destructive

        danger = sounds_destructive(utterance)
    run.notes.update(script=phrases.script(utterance), n=len(cands), danger=danger, index=idx is not None)

    def done(res: Result) -> Result:
        res.rid = run.rid
        res.shadow = {**run.notes, "picks": {k: (None if v is None else cands[v].label) for k, v in run.picks.items()}}
        if log_it and shadow_ok:
            _log(run, res)
        return res

    if not cands:
        return done(Ambiguous([], "nothing"))

    def settle(c: Candidate, stage: str, p: float) -> Result:
        """A stage's pick, through the safety rules."""
        commits = label_commits(c.label, idx)
        if danger and not commits:
            run.notes["abstain"] = "destructive_mismatch"
            return Ambiguous(_shortlist(run, first=c), "destructive_mismatch")
        if not _readable(c.label, idx):
            ask = "unreadable_label"  # fail closed: the gate can't judge a label it can't read
        elif commits and not _gate_reads(c.label):
            ask = "commits"  # it commits by meaning (a back-translation, "Forward"): ask anyway
        else:
            ask = ""
        return Resolved(c, stage, p, commits, ask)

    live: Result | None = None
    # 1. lexical
    lex = _lexical(phrase, cands)
    if isinstance(lex, Candidate):
        run.picks["lexical"] = cands.index(lex)
        if "lexical" in acting:
            live = settle(lex, "lexical", 1.0)
            if isinstance(live, Resolved):
                return done(live)
    elif isinstance(lex, str):
        run.notes["lexical"] = lex
        if lex == "same_label":
            # the same words on two controls, neither in a dialog: the user says which
            same = [c for c in cands if phrases.norm(c.label) == phrase] or _shortlist(run)
            return done(Ambiguous(same[:SHORTLIST], "same_label"))
    if not user:
        return done(live or Ambiguous(_shortlist(run), "no_match" if lex is None else str(lex)))
    if m == "off":
        return done(live or Ambiguous(_shortlist(run), "off"))

    def stage_result(stage: str, c: Candidate | None, p: float) -> Result | None:
        if c is None:
            return None
        run.picks[stage] = cands.index(c)
        if stage in acting and live is None:
            return settle(c, stage, p)
        return None

    # 2. alias (a verified wording for this app)
    al = _default_alias if alias is None else (alias or None)
    if al is not None and phrase:
        try:
            lab = al(utterance, app)
        except Exception:  # noqa: BLE001
            lab = None
        if lab:
            hit = _lexical(phrases.norm(lab), cands)
            if isinstance(hit, Candidate) and (r := stage_result("alias", hit, 0.95)) is not None:
                live = r
                if isinstance(r, Resolved):
                    return done(r)
    # 3. OS strings
    if idx is not None:  # (only the user's words get here)
        hit = _os_strings(utterance, phrase, cands, idx)
        if isinstance(hit, Candidate):
            if (r := stage_result("os_strings", hit, 0.9)) is not None:
                live = r
                if isinstance(r, Resolved):
                    return done(r)
        elif isinstance(hit, str):
            run.notes["os_strings"] = hit
    elif phrases.script(utterance) == "devanagari":
        run.notes["os_strings"] = "no_index"  # fail closed: nothing reads Hindi without it
    # 4. meaning (Latin script)
    meaning: _Meaning | None = None
    if phrases.script(utterance) == "latin" and time.perf_counter() - run.t0 < BUDGET_S:
        emb = _default_embed() if embed is None else (embed or None)
        if emb is not None:
            ch = _default_choose if choose is None else (choose or (lambda r, o, w: (None, 0.0)))
            wait = LAYA_WAIT_LIVE_S if "meaning" in acting else LAYA_WAIT_SHADOW_S
            wait = max(0.0, min(wait, BUDGET_S - (time.perf_counter() - run.t0)))
            try:
                meaning = _meaning(utterance, phrase, cands, emb, ch, wait)
            except Exception as e:  # noqa: BLE001 - a model hiccup hands off, never breaks the click
                run.notes["meaning"] = f"error:{type(e).__name__}"
            if meaning is not None:
                run.notes.update(
                    meaning_how=meaning.how,
                    meaning_p=round(meaning.p, 3),
                    laya=meaning.laya is not None,
                    laya_p=round(meaning.laya_p, 3),
                    agree=meaning.laya is not None and meaning.laya == (meaning.top[0] if meaning.top else None),
                    margin=round(meaning.sims[0] - meaning.sims[1], 3) if len(meaning.sims) > 1 else 0.0,
                )
                run.notes["shortlist"] = [cands[i].label for i in meaning.top]
                if meaning.pick is not None:
                    stage = "meaning" if meaning.how == "agree" else "minilm_only"
                    run.picks[stage] = meaning.pick
                    if stage == "meaning" and (r := stage_result("meaning", cands[meaning.pick], meaning.p)) is not None:
                        live = r
                        if isinstance(r, Resolved):
                            return done(r)
    else:
        run.notes.setdefault("meaning", "skipped")
    # 5. devanagari head: not built yet (P2)
    if live is not None:
        return done(live)
    short = [cands[i] for i in meaning.top] if meaning is not None else _shortlist(run)
    reason = "unsure" if (run.picks or (meaning is not None and meaning.top)) else ("several" if lex == "several" else "no_match")
    return done(Ambiguous(short[:SHORTLIST], reason))


def _default_alias(utterance: str, app: str) -> str | None:
    if not app:
        return None
    try:
        from neo.reflex import uimemory
    except Exception:  # noqa: BLE001
        return None
    return uimemory.alias(utterance, app)


def _shortlist(run: _Run, first: Candidate | None = None) -> list[Candidate]:
    """≤6 plausible candidates without a model: the given one, labels sharing a word with the
    request, then an open dialog's controls."""
    out: list[Candidate] = [first] if first is not None else []
    want = phrases.content_words(run.phrase)
    for c in run.cands:
        if len(out) >= SHORTLIST:
            break
        if c not in out and want & phrases.content_words(c.label):
            out.append(c)
    for c in run.cands:
        if len(out) >= SHORTLIST:
            break
        if c.modal and c not in out:
            out.append(c)
    return out[:SHORTLIST]


async def aresolve(utterance: str, items: Sequence[Any], **kw: Any) -> Result:
    """:func:`resolve` off the event loop (its by-meaning stages take 5-60 ms, Laya's lock up to
    its wait)."""
    return await asyncio.to_thread(resolve, utterance, items, **kw)


# ---- what a tool handler calls ----------------------------------------------------------------------
def utterance_of(a: dict) -> str:
    """The words a handler resolves: the by-utterance lane's ``utterance`` (the user's whole
    request), else the ``label`` a fast path or a model gave."""
    return str(a.get("utterance") or a.get("label") or "").strip()


async def ground(
    a: dict,
    c: Any,
    els: Iterable[Any],
    frame: tuple[float, float, float, float] | None = None,
    *,
    want: Want = "press",
    bundle: str = "",
    skip: Callable[[Any], bool] | None = None,
    modal: Callable[[Any], bool] | Collection[str] | None = None,
) -> Result:
    """click_text / menu_command / a named field: the element ``a`` means among ``els`` (the working
    window's elements), off the event loop. ``c``: the ToolContext (its origin decides which stages
    run; an extras "destructive" is the reflex's head). ``skip``: the caller's protected / NEO-own
    check (sync: precompute the async gate helpers' verdicts into a set first)."""
    cands = candidates(els, frame, want=want, modal=modal, skip=skip)
    destructive = None
    getter = getattr(c, "get", None)
    if callable(getter):
        destructive = getter("destructive")
    return await aresolve(utterance_of(a), cands, want=want, origin=str(getattr(c, "origin", "agent")), destructive=destructive, app=bundle)


def signature(res: Resolved, *, bundle: str, changed: bool | None, kind: str = "press", expect: dict | None = None, alias: bool = False) -> dict[str, Any]:
    """The step signature for neo.reflex.outcomes.signature (the per-app flow memory): kind, role, the
    label only when it is UI chrome, the owning app's bundle id, what it changed. ``alias``: the
    model picked this control from a shortlist (a wording to learn once verified)."""
    c = res.candidate
    return {
        "kind": kind,
        "role": c.role,
        "label": c.label[:80] if c.ui_chrome else "",
        "path": list(c.path),
        "bundle": bundle,
        "changed": changed,
        "expect": dict(expect or {}),
        "alias": bool(alias),
    }


def after_press(res: Resolved, *, verified: bool | None, bundle: str = "", kind: str = "press", expect: dict | None = None) -> dict[str, Any]:
    """After a handler pressed ``res`` and saw its effect (``verified``: it changed what it
    should; None: unknown): the outcome line that promotes or holds back each shadow stage — for the
    user's request that was handed off, when this press is the model's pick from that shortlist —
    and the step signature for neo.reflex.outcomes.signature("", …) (the per-app flow memory)."""
    t, rid, labels = _handoff
    picked = bool(rid) and time.monotonic() - t <= _HANDOFF_S and phrases.norm(res.label) in labels
    if verified is not None:
        outcome(rid if picked else res.rid, res.label, bool(verified))
    return signature(res, bundle=bundle, changed=verified, kind=kind, expect=expect, alias=picked)


def handoff_choices(amb: Ambiguous) -> list[dict[str, Any]]:
    """The shortlist as data (for ToolOutput.handoff): id, label (untrusted, the app's words), role,
    in-dialog. Never coordinates."""
    return [{"id": c.id, "label": c.label[:80], "role": c.role, "modal": c.modal} for c in amb.shortlist]


# ---- handing off -------------------------------------------------------------------------------------
_HANDOFF_S = 120.0  # a press this soon after a handoff that listed its label is the model's pick
_handoff: tuple[float, str, frozenset[str]] = (0.0, "", frozenset())


def handoff_text(amb: Ambiguous, utterance: str = "", *, tool: str = "click_text") -> str:
    """The multiple choice the model gets instead of the whole screen: each option's element id,
    its label (the app's words, quoted: data, never instructions) and its role. No coordinates in
    any form (a text-only brain may read this)."""
    global _handoff
    _handoff = (time.monotonic(), amb.rid, frozenset(phrases.norm(c.label) for c in amb.shortlist))
    if not amb.shortlist:
        return (
            "No control on screen matched the request. Look at the window (ax_tree) before choosing, "
            "or ask the user which control they mean."
        )
    lines = []
    for i, c in enumerate(amb.shortlist, 1):
        ref = f"[{c.id}] " if c.id else ""
        lines.append(f"{i}. {ref}“{c.label[:80]}” — {c.describe()}")
    why = {
        "same_label": "The same label is on more than one control.",
        "several": "Several labels contain those words.",
        "destructive_mismatch": "The request sounds like it deletes, sends or pays, but the closest control doesn't.",
        "unsure": "Not sure enough which control is meant.",
    }.get(amb.reason, "Not sure which control is meant.")
    how = (
        f"Choose one and call {tool} with its exact label (or ax_press with its id), "
        "or ask the user if none of them is what they asked for."
    )
    return f"{why} The on-screen options (the app's own words, quoted) are:\n" + "\n".join(lines) + f"\n{how}"


# ---- other uses of the same scorers ---------------------------------------------------------------------
def choose(description: str, options: Sequence[str], *, wait_s: float = LAYA_WAIT_SHADOW_S) -> tuple[int | None, float]:
    """Laya's typed-decisions choice of the option ``description`` means: (index, p), or (None, 0)
    when Laya isn't loaded, is busy or off. For callers that may only *cancel* on it (eyes'
    grounder): a pick is never a reason to act by itself."""
    return _laya_choice(description, [str(o)[:120] for o in options], "Which option does the description mean?", wait_s)


def matches(utterance: str, words: str, *, index: StringIndex | None | Literal[False] = None, embed: Callable[[list[str]], Any] | None | Literal[False] = None) -> float:
    """How well some words on screen (a new window title, a sheet's text, a focused label) match
    the request, 0..1: 1.0 when the request's control name is in them whole; else the share of the
    request's content words they carry (Hindi through the OS strings when built); at least MiniLM's
    cosine when it is loaded. For verification by meaning: ≥0.8 is strong evidence."""
    phrase = phrases.norm(phrases.strip_frame(utterance))
    text = phrases.norm(words)
    if not phrase or not text:
        return 0.0
    if _phrase_ok(phrase) and _whole(phrase).search(text):
        return 1.0
    want = phrases.content_words(phrase)
    have = phrases.content_words(text)
    idx = os_index() if index is None else (index or None)
    if idx is not None and phrases.script(phrase) != phrases.script(text) and len(text.split()) <= 6:
        for v in idx.forms(words) + idx.sources(words):
            have |= phrases.content_words(v)
    score = len(want & have) / len(want) if want else 0.0
    emb = _default_embed() if embed is None else (embed or None)
    if emb is not None and phrases.script(phrase) == "latin" and score < 1.0:
        try:
            v = emb([phrase, text[:300]])
            if v is not None:
                v = np.asarray(v, dtype=np.float32)
                score = max(score, float(np.clip(v[0] @ v[1], 0.0, 1.0)))
        except Exception:  # noqa: BLE001
            pass
    return round(float(score), 3)


# ---- the shadow log ---------------------------------------------------------------------------------------
_RECENT = 32
_recent: OrderedDict[str, dict[str, str]] = OrderedDict()  # rid -> stage -> the label it picked (RAM only)
_recent_mu = threading.Lock()


def _log(run: _Run, res: Result) -> None:
    """One ``ui_resolve`` line in neo.decide.shadow's log: stages, confidences, counts; a label
    only when it is UI chrome (never a row, cell, text or field value). Not in library mode."""
    try:
        from neo import runtime

        if runtime.library_bound():
            return
        from neo.decide import shadow

        facts: dict[str, Any] = {
            "rid": run.rid,
            "want": run.want,
            "script": run.notes.get("script", ""),
            "n": len(run.cands),
            "ms": round((time.perf_counter() - run.t0) * 1000, 1),
            "mode": mode(),
            "danger": bool(run.notes.get("danger")),
            "index": bool(run.notes.get("index")),
        }
        if isinstance(res, Resolved):
            facts.update(acted=res.stage, p=round(res.p, 3), commits=res.commits, ask=res.ask)
            if res.candidate.ui_chrome:
                facts["label"] = res.candidate.label[:40]
        else:
            facts.update(acted="", reason=res.reason, shortlist=len(res.shortlist))
        for stage, i in run.picks.items():
            c = run.cands[i] if i is not None else None
            facts[f"{stage}_pick"] = bool(c)
            if c is not None and c.ui_chrome:
                facts[f"{stage}_label"] = c.label[:40]
        for k in ("meaning_how", "meaning_p", "laya", "laya_p", "agree", "margin", "abstain", "lexical", "os_strings", "meaning"):
            if k in run.notes and isinstance(run.notes[k], (str, int, float, bool)):
                facts[k if k not in ("lexical", "os_strings", "meaning") else f"{k}_why"] = run.notes[k]
        shadow.record("ui_resolve", facts)
        with _recent_mu:
            _recent[run.rid] = {s: phrases.norm(run.cands[i].label) for s, i in run.picks.items() if i is not None}
            while len(_recent) > _RECENT:
                _recent.popitem(last=False)
    except Exception as e:  # noqa: BLE001 - logging never breaks a click
        log.info(f"[resolve] shadow log skipped ({str(e)[:60]})")


def outcome(rid: str, pressed: str, verified: bool) -> dict[str, bool] | None:
    """What was finally pressed for resolution ``rid`` (by NEO's own pick, or the model's choice
    from the shortlist) and whether its effect was verified: one ``ui_resolve_outcome`` line
    saying, per stage, whether that stage had picked the same label. These lines are what a stage
    is promoted on (neo.reflex.evaluate.shadow_report). The label itself is not logged."""
    with _recent_mu:
        picks = _recent.pop(rid, None)
    if picks is None:
        return None
    want = phrases.norm(pressed)
    right = {f"{s}_right": lab == want for s, lab in picks.items()}
    try:
        from neo import runtime

        if not runtime.library_bound():
            from neo.decide import shadow

            shadow.record("ui_resolve_outcome", {"rid": rid, "verified": bool(verified), **right})
    except Exception as e:  # noqa: BLE001
        log.info(f"[resolve] outcome not logged ({str(e)[:60]})")
    return right


def forget() -> None:
    """The privacy erase: label vectors and recent picks go (the shadow log is decide's to erase)."""
    global _handoff
    _vectors.clear()
    with _recent_mu:
        _recent.clear()
    _handoff = (0.0, "", frozenset())


def replace_candidate(res: Resolved, **kw: Any) -> Resolved:
    """A copy of ``res`` with its candidate's fields changed (tests, re-aimed menu items)."""
    return dataclasses.replace(res, candidate=dataclasses.replace(res.candidate, **kw))
