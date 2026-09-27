"""The replay lane: a flow that worked, done again without a model.

When the user asks for the same thing in the same app, and a model has twice done it the same way —
every action one NEO saw work — NEO does it the third time itself: the same actions, in order,
each through the registry exactly as a model's call would go (the confirmation gate and the
protected-app checks inside every tool, origin "agent"), each checked for its effect. The first
action that doesn't work as before stops the replay and hands the request to the agent, told what
already ran and what didn't work. A flow that misses twice in a row is dropped.

What can be replayed is narrow on purpose:

* only the user's own words start one (never a model's words: Gemini Live, an SDK host), and only
  as a command: words Laya reads as wanting an answer get a model;
* only actions whose arguments are the user's own words, as they said them (an app, a menu item, a
  button they named — punctuation and all, no control characters) or a key chord, and numbers they
  said — never an element id, a position or "at the pointer" (all go stale), never text read off the
  screen or written by the model. A chord counts only from a run that read nothing untrusted;
* only a run that ended done, without a question, with every action NEO saw change something, and
  that was a command with nothing to say (neo2.core.agent.session._quietly) — so a replay is silent too.
  Never a run that took over from a replay or a fast path: it did only the tail of the request;
* only in the app the flow was worked out in (the app in front when it started, by its name).

Nothing is written to disk: flows live in memory, per NEO.app process. With learning off
(NEO_LEARN=off) nothing is recorded or replayed; an erase of NEO's memory (neo2.core.privacy) drops every
flow; a stop during a replay, or right after a flow was learned or replayed, drops that flow. An
embedded agent never records or replays.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from neo2.core import logger, runtime
from neo2.core.agent import batch

log = logger(__name__)

MIN_SEEN = 2  # model runs that did it exactly this way before NEO replays it
MAX_MISSES = 2  # replays in a row that didn't work: the flow is dropped
MAX_STEPS = 6
MAX_FLOWS = 200
_ACT_CATEGORIES = frozenset({"computer", "apps"})  # actions on the screen and on apps
_PIXEL_OR_ID = batch._PIXEL_ARGS | {"id", "element", "ref"}
_ID_VALUE = re.compile(r"^[A-Za-z]{1,3}\d+$")  # an element id (e12): stale by the next look
_KEY = r"(?:cmd|command|ctrl|control|alt|opt|option|shift|fn|return|enter|escape|esc|tab|space|delete|backspace|up|down|left|right|home|end|pageup|pagedown|f\d{1,2}|[a-z0-9]|[\[\]\\;',./`=-])"
_CHORD = re.compile(rf"^{_KEY}(?:\s*\+\s*{_KEY})*$", re.I)
_WORDS = re.compile(r"[^\w\s]", re.UNICODE)
# Positions by another name: where the pointer happens to be when the replay runs.
_POSITION_FLAGS = frozenset({"at_pointer"})
# Characters no one says: control characters, line breaks, text-direction overrides.
_BIDI = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f")
STOP_S = 3.0  # a stop this soon after a flow was learned or replayed says it was wrong (outcomes.EARLY_S)
_clock = time.monotonic


def _norm(text: str) -> str:
    """The request as it's matched: lower-case, punctuation gone, one space between words."""
    return " ".join(_WORDS.sub(" ", (text or "").casefold()).split())


@dataclass
class Flow:
    request: str  # the request, normalized
    app: str  # the app in front when the run began (lower-case name)
    steps: tuple[tuple[str, str], ...]  # (tool, its arguments as sorted JSON)
    seen: int = 1  # model runs that did it exactly this way
    wins: int = 0  # replays that worked
    misses: int = 0  # replays in a row that didn't
    generation: int = 0  # neo2.core.privacy's erase generation it was learned in
    at: float = field(default_factory=time.time)

    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [(t, json.loads(a)) for t, a in self.steps]


_flows: dict[tuple[str, str], Flow] = {}
_last: tuple[str, str] | None = None  # the flow the latest request recorded or replayed
_last_at = float("-inf")  # when that flow was learned, or its replay ended (_clock)
_running: set[tuple[str, str]] = set()  # flows being replayed right now


def enabled() -> bool:
    """The replay lane is on (``replay_flows``; NEO_REPLAY_FLOWS=0 turns it off). On in NEO.app;
    off wherever real effects are off (a test, a scratch run) unless asked for — it reads which app
    is in front. Never for an embedded agent, and never with learning off (NEO_LEARN=off: a flow is
    something learned from the user's requests)."""
    if runtime.library_bound():
        return False
    from neo2.core import effects
    from neo2.core.config import settings

    if not settings().learn:
        return False
    return bool(batch._setting("replay_flows", effects.on(), False))


def _generation() -> int:
    from neo2.core import privacy

    return privacy.generation()


def forget() -> None:
    """Drop every flow (NEO's memory was erased — neo2.core.privacy.forget_everything; a test's clean
    slate)."""
    global _last, _last_at
    _flows.clear()
    _running.clear()
    _last, _last_at = None, float("-inf")


def forget_last() -> None:
    """The user said stop. During a replay, or right after a flow was learned or replayed (STOP_S),
    that says the flow was wrong: it is dropped. A stop later on ("stop" to the music it started,
    minutes after) says nothing about it."""
    global _last
    if _last is not None and (_last in _running or _clock() - _last_at <= STOP_S):
        _flows.pop(_last, None)
    _last = None


def working_app() -> str:
    """The app the user is working in (NEO's own window aside), lower-case; "" when it can't tell.
    Only reads which app is in front."""
    try:
        from neo2.core.tools.computer import ax

        name, _pid = ax.working_app()
        return (name or "").strip().casefold()
    except Exception:  # noqa: BLE001 - no Accessibility, no macOS: no app, no replay
        return ""


# ---- learning a flow ---------------------------------------------------------------------------------


_CHORD_ARGS = frozenset({"keys", "key", "shortcut", "chord"})  # where a key chord may stand


def _said(text: str) -> str:
    """The user's words as an argument is matched against them: case aside, one space between
    words — punctuation kept."""
    return " ".join((text or "").casefold().split())


def _unsaid(s: str) -> bool:
    """Characters no one says (a line break, a control character, a text-direction override)."""
    return any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") or ch in _BIDI for ch in s)


def _in_words(s: str, said: str) -> bool:
    """`s` stands in the user's words as they said them, whole words only (case aside)."""
    return bool(s) and re.search(rf"(?<!\w){re.escape(s.casefold())}(?!\w)", said) is not None


def _number_said(v: float, said: str) -> bool:
    """A number the user said (its size: "scroll down 5" is dy=-5)."""
    n = abs(v)
    form = str(int(n)) if float(n).is_integer() else repr(float(n))
    return re.search(rf"(?<![\w.]){re.escape(form)}(?!\w)(?!\.\d)", said) is not None


def _replayable_args(args: dict[str, Any], text: str, *, chords: bool) -> bool:
    """Is every argument the user's own words (`text`: their request, as they said it), a key chord
    (`chords`: from a run that read nothing untrusted) or a number they said? Never an element id,
    a position or "at the pointer"."""
    said = _said(text)
    for k, v in (args or {}).items():
        if k in _PIXEL_OR_ID or k.startswith("_"):
            return False
        if v is None:
            continue
        if isinstance(v, bool):
            if v and k in _POSITION_FLAGS:
                return False  # wherever the pointer happens to be: a position
            continue
        if isinstance(v, int | float):
            if _number_said(v, said):
                continue
            return False
        if not isinstance(v, str):
            return False
        if _unsaid(v) or v != " ".join(v.split()) or _ID_VALUE.match(v):
            return False
        if _in_words(v, said):
            continue  # the user's own words
        if chords and k in _CHORD_ARGS and _CHORD.match(v):
            continue  # a key chord (from a run that read nothing untrusted)
        return False
    return True


def steps_of(res: Any, text: str, *, reg: Any, tainted: bool) -> tuple[tuple[str, str], ...] | None:
    """The run's actions as a replayable flow, or None when any of it can't be replayed. `text`:
    the request, as the user said it."""
    out: list[tuple[str, str]] = []
    for st in getattr(res, "steps", []):
        t = reg.get(st.tool)
        if t is None or t.hidden:
            return None
        if st.result.lstrip().startswith(("NEEDS_", "skipped: ")) or not st.ok:
            return None  # a question, a skip or a failure: not a flow that simply worked
        if t.parallel_safe or t.read_only or st.tool in ("screenshot", "look", "more_tools"):
            continue  # a look: the replay checks each action's effect instead
        if t.category not in _ACT_CATEGORIES or t.risk == "destructive" or t.untrusted:
            return None
        eff = getattr(st, "effect", None)
        worked = (eff is not None and getattr(eff, "changed", None) is True) or batch.verified(
            st.tool, _AsResult(st)
        )
        if not worked:
            return None  # NEO didn't see it change anything: nothing to check a replay against
        if not _replayable_args(st.args, text, chords=not tainted):
            return None
        out.append((st.tool, json.dumps(st.args, sort_keys=True, ensure_ascii=False, default=str)))
    if not out or len(out) > MAX_STEPS:
        return None
    return tuple(out)


@dataclass
class _AsResult:
    """A loop Step seen as a ToolResult (batch.verified reads ok, content and effect)."""

    st: Any

    @property
    def ok(self) -> bool:
        return bool(self.st.ok)

    @property
    def content(self) -> str:
        return str(self.st.result)

    @property
    def effect(self) -> Any:
        return getattr(self.st, "effect", None)


def learn(text: str, app: str, res: Any, *, reg: Any, tainted: bool) -> Flow | None:
    """Remember how a run that simply worked did it (see the module doc for what qualifies). The
    caller has checked the run was a quiet command from the user's own words."""
    global _last, _last_at
    if not enabled() or not app:
        return None
    words = _norm(text)
    if not words or getattr(res, "stopped", "") != "done" or getattr(res, "question", ""):
        return None
    steps = steps_of(res, text, reg=reg, tainted=tainted)
    if steps is None:
        return None
    key = (words, app)
    gen = _generation()
    f = _flows.get(key)
    if f is not None and f.steps == steps and f.generation == gen:
        f.seen += 1
        f.at = time.time()
    else:
        f = Flow(words, app, steps, generation=gen)
        _flows[key] = f
        if len(_flows) > MAX_FLOWS:
            for k, _v in sorted(_flows.items(), key=lambda kv: kv[1].at)[: len(_flows) - MAX_FLOWS]:
                _flows.pop(k, None)
    _last, _last_at = key, _clock()
    log.info(f"[replay] learned a way to “{words[:40]}” in {app} ({len(steps)} steps, seen {f.seen}×)")
    return f


# ---- replaying one -----------------------------------------------------------------------------------


def candidate(text: str) -> bool:
    """Is there a flow for these words at all (no app read yet: a dict lookup)?"""
    words = _norm(text)
    return bool(words) and any(k[0] == words for k in _flows)


def recall(text: str, app: str) -> Flow | None:
    """The flow to replay for this request in this app, if one has earned it."""
    if not enabled() or not app:
        return None
    f = _flows.get((_norm(text), app))
    if f is None or f.generation != _generation():
        return None
    if f.seen < MIN_SEEN or f.misses >= MAX_MISSES:
        return None
    return f


@dataclass
class Replayed:
    """How a replay went. `ran`: the (tool, args, output) of each action that ran, in order.
    `how`: "done" (every action worked as before), "halt" (the last one stopped to ask the user),
    "miss" (the last one didn't work as before: the agent takes over)."""

    how: str
    ran: list[tuple[str, dict[str, Any], Any]]


Invoke = Callable[[str, dict[str, Any]], Awaitable[Any]]


async def replay(f: Flow, invoke: Invoke) -> Replayed:
    """Run the flow's actions in order through `invoke` (the registry, with the request's context),
    each checked for its effect, stopping at the first that asks the user or doesn't work."""
    global _last, _last_at
    key = _last = (f.request, f.app)
    _running.add(key)
    try:
        return await _replay(f, invoke)
    finally:
        _running.discard(key)
        _last_at = _clock()


async def _replay(f: Flow, invoke: Invoke) -> Replayed:
    ran: list[tuple[str, dict[str, Any], Any]] = []
    for tool, args in f.calls():
        out = await invoke(tool, args)
        ran.append((tool, args, out))
        if getattr(out, "halt", False) or str(getattr(out, "text", "")).lstrip().startswith("NEEDS_"):
            return Replayed("halt", ran)
        eff = batch.effect(out)
        verified = batch.verified(tool, _Out(tool, out))
        if not getattr(out, "ok", False) or not (verified or (eff is not None and getattr(eff, "changed", None) is True)):
            f.misses += 1
            if f.misses >= MAX_MISSES:
                _flows.pop((f.request, f.app), None)
            log.info(f"[replay] “{f.request[:40]}”: {tool} didn't work as before — the agent takes over")
            return Replayed("miss", ran)
    f.misses, f.wins, f.at = 0, f.wins + 1, time.time()
    return Replayed("done", ran)


@dataclass
class _Out:
    """A ToolOutput seen as a ToolResult (batch.verified reads ok, content and effect)."""

    name: str
    out: Any

    @property
    def ok(self) -> bool:
        return bool(getattr(self.out, "ok", False))

    @property
    def content(self) -> str:
        return str(getattr(self.out, "text", ""))

    @property
    def effect(self) -> Any:
        return batch.effect(self.out)


def describe(ran: Sequence[tuple[str, dict[str, Any], Any]]) -> str:
    """What ran, for the agent that takes over: tool(args) of each action that worked."""
    return "; ".join(f"{t}({', '.join(f'{k}={v!r}' for k, v in a.items())})" for t, a, _o in ran)
