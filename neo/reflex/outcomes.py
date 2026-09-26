"""What actually happened to each request — the labels Laya learns from.

An *episode* is one request: its words, the reflex's decision, the route NEO took, what the
teacher chose (Gemini Live's tool calls, or NEO's brain for a request the reflex was unsure of),
each step with whether it worked, and what the user did next: stop it, talk over it, answer a
confirmation, or say "no, I meant…". Episodes live in RAM only. One settles ~20 s after its last
event, or as soon as the next, unrelated request starts; only then does it become a row for
``learn.record`` — or none:

* it ran, every step worked (or a failure was recovered by a later step), nothing contradicted
  it → what ran: no tool = chat; one call of a tool that isn't slow = quick_action + that tool;
  several tools = quick_action with no single tool when NEO's fast paths cover the whole
  sentence (and the agent wasn't needed), else agent_task; a slow tool = agent_task. Verified,
  w=1.0.
* only the teacher's choice is known (Live's calls whose results never came back, Live
  answering in words) → that choice, w=0.5.
* the next request within 20 s corrects it ("no, I meant…", "galat…") and *that* one is
  verified → this request's words get its label, with the original in ``wrong`` (a hard
  negative); verified, w=1.0. Mostly the same words again only counts as a fix of a request that
  didn't visibly work; after one that did, it is a request of its own ("turn the volume up" →
  "turn the brightness up too"), and so is a whole new command after a "no" ("no wait, open
  Safari"). A yes or no to a question NEO itself asked is an answer, not a correction.
* a failure nothing recovered, a cancel, "no" at a confirmation → nothing.
* stopped or talked over within 3 s of NEO answering (and no rephrase) → nothing; a defect.

Nothing here touches disk: only settled rows (via learn.record, skeleton text only) and daily
counters without any text. NEO_LEARN=off turns it off; neo.privacy's erase forgets it.
"""

from __future__ import annotations

import contextvars
import functools
import itertools
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from neo import runtime
from neo.reflex import learn

SETTLE_S = 20.0  # an episode nothing followed up on for this long is final
WINDOW_S = 20.0  # a correction has to come this soon after the request it corrects
EARLY_S = 3.0  # stopped or talked over this soon after NEO responded = it was wrong
CONFIRM_WAIT_S = 125.0  # a pending confirmation (confirm._TTL_S) holds its episode open
STALE_S = 600.0  # an episode that never finishes is dropped after this long
JACCARD = 0.5  # word overlap that makes the next request a rephrase of this one
_IGNORE = {"more_tools"}
_INTENTS = ("chat", "quick_action", "agent_task")

# The next sentence says the last one was wrong (English and Hinglish).
_CORRECTION = re.compile(
    r"^\s*(?:(?:hey\s+)?neo[\s,]+)?(?:no|nope|not\s+that|i\s+meant|i\s+said|wrong|galat|nahi|nahin|"
    r"arre\s+nahi|mera\s+matlab|matlab|uska\s+nahi|dusra|doosra)\b",
    re.I,
)
_UNABLE = re.compile(
    r"\b(?:i\s+can'?t|i\s+cannot|i'?m\s+(?:not\s+able|unable)|unable\s+to|i\s+don'?t\s+have\s+access|"
    r"couldn'?t|could\s+not|failed|didn'?t\s+work)\b",
    re.I,
)
_ASKED_BACK = re.compile(r"\?\s*$|\b(?:could\s+you|which\s+one|do\s+you\s+want)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# The user answering a question NEO asked ("Want me to do Downloads too?" → "no, that's all").
_REPLY = re.compile(
    r"^\s*(?:(?:hey\s+)?neo[\s,]+)?(?:yes|yeah|yep|yup|sure|ok(?:ay)?|no|nope|nah|nahi|nahin|haan|han|"
    r"thanks?|thank\s+you|not\s+now)\b",
    re.I,
)
# A bare "no" (and "wait", "actually"…) can be a change of mind rather than "you got it wrong"…
_SOFT_NO = re.compile(
    r"^\s*(?:(?:hey\s+)?neo[\s,]+)?(?:arre\s+nahi|no|nope|nahi|nahin)\b"
    r"(?:[\s,.!:;-]*(?:wait|actually|sorry|hold\s+on|hang\s+on|yaar|arre)\b)*[\s,.!:;-]*",
    re.I,
)
# …these words never are.
_EXPLICIT = re.compile(
    r"(?:i\s+meant|i\s+said|not\s+that|wrong|galat|mera\s+matlab|matlab|uska\s+nahi|dusra|doosra)\b", re.I
)
_FUNCTION_WORDS = {
    "the", "a", "an", "it", "to", "my", "me", "please", "and", "of", "in", "on", "for", "that", "this",
    "is", "i", "you", "your", "now", "just", "can", "could", "would", "will",
}  # fmt: skip

# The episode a relayed request belongs to: Gemini Live's tool call (or its command lane) binds
# its sentence here, so a session run it hands over reports its steps to that sentence.
_bound: contextvars.ContextVar[str] = contextvars.ContextVar("neo_episode", default="")


@dataclass(eq=False)
class Episode:
    key: str
    text: str
    source: str  # "neo" (typed / local voice) | "live"
    started: float
    last: float
    filed: bool  # its words are final
    running: bool  # NEO's route for it is still going
    decision: dict | None = None  # the reflex's pick: intent, tool, conf, margin, source
    route: str = ""
    responded: float = 0.0  # when NEO first answered or acted
    teacher: list[str] = field(default_factory=list)  # Live's tool calls
    spoke: bool = False  # Live answered in words
    steps: list[tuple[str, bool]] = field(default_factory=list)  # (tool, worked)
    calls: int = 0  # Live tool calls still running
    stopped: str = ""  # how an agent run ended (AgentResult.stopped)
    answer: str = ""  # NEO's own reply, judged: "real" | "unable" | "asked_back" (never kept)
    asked: bool = False  # NEO's last reply asked the user something ("…Want me to do Downloads too?")
    unable: bool = False  # an agent run with steps that says it didn't manage
    cancelled: bool = False
    early_stop: bool = False  # stopped / talked over within EARLY_S
    corrected: bool = False  # the user said it was wrong
    awaiting: bool = False  # a confirmation is pending
    confirm: bool | None = None
    kind: str = ""  # neo/decide's task kind…
    kind_ok: bool = False  # …and whether carrying it out worked
    brain: str = ""  # the provider that read it (chat / agent routes): the teacher's only allowed one
    fix: Episode | None = None  # the next request, which corrected or rephrased this one
    resolved: tuple[str, str] | None = None  # the verified label this episode ended up with
    resolved_tools: tuple[str, ...] = ()  # every tool behind that label, along the chain of fixes
    settled: bool = False


def _safe(fn: Callable) -> Callable:
    """Learning is best effort: a bug here must never break a request. It is NEO.app's alone: the
    ledger, its counters and what it trains are process-wide, so an embedded agent's requests
    (neo.runtime) never reach it."""

    @functools.wraps(fn)
    def wrap(*a, **kw):
        if runtime.library_bound():
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[reflex] outcome ledger: {type(e).__name__}: {str(e)[:100]}")
            return None

    return wrap


def _tokens(text: str) -> set[str]:
    return {w for w in (t.strip(".,!?;:\"'()") for t in text.lower().split()) if w}


def _answers(text: str) -> bool:
    """Just a yes or a no to a question NEO asked ("no, I meant the other file" is more)."""
    from neo.agent.session import _NO, _YES

    return bool(_YES.match(text)) or (bool(_NO.search(text)) and len(text.split()) <= 3)


def _fast(tool: str) -> bool:
    """A registered tool that answers or acts in one quick call: not slow, not the agent itself."""
    if tool == "agent_task":
        return False
    try:
        from neo.agent.registry import registry

        t = registry().get(tool)
    except Exception:  # noqa: BLE001
        return False
    return t is not None and not t.slow


def _covered(text: str) -> set[str]:
    """The tools NEO's fast paths run for this whole sentence without any model (empty: none)."""
    try:
        from neo.agent import fastpath

        return {t for t, _ in fastpath.plan(text) or []}
    except Exception:  # noqa: BLE001
        return set()


def _label_for(tools: list[str], text: str = "", *, agent: bool = False) -> tuple[str, str]:
    """No tool → chat; one tool that isn't slow → quick_action + that tool; several that the
    fast paths cover for these words (without the agent's help) → quick_action with no single
    tool; anything else → agent_task."""
    distinct = list(dict.fromkeys(t for t in tools if t not in _IGNORE))
    if not distinct:
        return "chat", ""
    if len(distinct) == 1:
        return ("quick_action", distinct[0]) if _fast(distinct[0]) else ("agent_task", "")
    if not agent and "agent_task" not in distinct and set(distinct) <= _covered(text):
        return "quick_action", ""
    return "agent_task", ""


def _judge(text: str) -> str:
    """A chat reply: "" (nothing said), "unable", "asked_back" (nothing but a question back —
    "Which file do you mean?") or "real" (an answer, even one ending in a follow-up question)."""
    if not text:
        return ""
    if _UNABLE.search(text):
        return "unable"
    if not any(s.strip() and not _ASKED_BACK.search(s) for s in _SENTENCE.split(text)):
        return "asked_back"
    return "real"


def _new_command(before: str, text: str) -> bool:
    """ "no wait, open Safari" after something unrelated: a whole command of its own after a bare
    no, sharing nothing with the request before it — a change of mind, not a correction."""
    m = _SOFT_NO.match(text)
    if m is None:
        return False  # "I meant…", "galat…": that says the last one was wrong
    rest = text[m.end() :].strip()
    if not rest or _EXPLICIT.match(rest) or len(rest.split()) < 2:
        return False  # "no, I meant the volume", "no, weather": the fix itself
    if (_tokens(rest) & _tokens(before)) - _FUNCTION_WORDS:
        return False  # "no, turn the VOLUME down" after "turn it down": about the same thing
    return bool(_covered(rest))


def _fmt(label: tuple[str, str]) -> str:
    return f"{label[0]}:{label[1]}" if label[1] else label[0]


def _outcome(ep: Episode) -> tuple[tuple[str, str], bool] | None:
    """(label, verified) from what the episode itself did, or None when it teaches nothing."""
    if ep.cancelled or ep.confirm is False or ep.awaiting:
        return None  # stopped, declined, or never answered: not what they wanted, or unknown
    if ep.stopped and ep.stopped != "done":
        return None  # ran out of steps, went round in circles, errored, or had to ask
    steps = [(t, ok) for t, ok in ep.steps if t not in _IGNORE]
    if steps:
        fails = [i for i, (_, ok) in enumerate(steps) if not ok]
        if fails and not any(ok for _, ok in steps[fails[-1] + 1 :]):
            return None  # it failed and nothing after that worked
        if ep.unable:
            return None
        return _label_for([t for t, ok in steps if ok], ep.text, agent=ep.route == "agent"), True
    if ep.answer == "real":
        return ("chat", ""), True  # NEO answered, with nothing to run
    if ep.answer:
        return None  # it couldn't, or asked back: the request was unclear
    if ep.teacher:
        return _label_for(ep.teacher, ep.text), False  # Live chose these; no result says how it went
    if ep.spoke:
        return ("chat", ""), False  # Live answered in words we never see
    return None


def _attempted(ep: Episode) -> str:
    """What this episode was taken to be — the label a correction says it was *not*."""
    if tools := [t for t, _ in ep.steps if t not in _IGNORE]:
        return _fmt(_label_for(tools, ep.text, agent=ep.route == "agent"))
    if ep.teacher:
        return _fmt(_label_for(ep.teacher, ep.text))
    d = ep.decision or {}
    if d.get("intent") in _INTENTS:
        return _fmt((d["intent"], d.get("tool", "") if d["intent"] == "quick_action" else ""))
    return "chat" if (ep.spoke or ep.answer or ep.route == "chat") else ""


class Ledger:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._eps: dict[str, Episode] = {}  # unsettled, by key
        self._recent: list[Episode] = []  # in the order they began (settled ones until trimmed)
        self._ids = itertools.count(1)

    # ---- lookup ----------------------------------------------------------------------------------
    def _get(self, key: str = "") -> Episode | None:
        """This key's episode, else the one bound to the current task (a relayed run)."""
        return self._eps.get(key) if key and key in self._eps else self._eps.get(_bound.get())

    def _latest(self, pred: Callable[[Episode], bool] = lambda e: True) -> Episode | None:
        return next((e for e in reversed(self._recent) if not e.settled and pred(e)), None)

    def _neighbour(self, ep: Episode, step: int) -> Episode | None:
        try:
            i = self._recent.index(ep) + step
        except ValueError:
            return None
        return self._recent[i] if 0 <= i < len(self._recent) else None

    def _touch(self, ep: Episode, now: float, *, responded: bool = True) -> None:
        ep.last = max(ep.last, now)
        if responded and not ep.responded:
            ep.responded = now

    # ---- events -----------------------------------------------------------------------------------
    def begin(self, key: str, text: str, *, source: str = "neo", decision=None, final: bool = True, bind: bool = True) -> str:
        """A new request. `final`: its words are complete and NEO's route for it is starting (a
        typed or locally heard request); Live's words keep arriving until file_live()."""
        if not learn.enabled():
            return ""
        with self._lock:
            now = self._clock()
            key = key or f"u{next(self._ids)}"
            if key in self._eps:
                return key
            self._tick(now)  # whatever was due settles first: a late sentence can't reopen it
            ep = Episode(key, " ".join(str(text).split()), source, now, now, filed=final, running=final)
            if decision is not None:
                laya = str(getattr(decision, "source", "")).startswith("laya")
                ep.decision = {
                    "intent": decision.intent,
                    "tool": decision.tool,
                    "conf": decision.intent_confidence if laya else None,
                    "margin": getattr(decision, "margin", None) if laya else None,
                    "source": decision.source,
                }
                if laya:
                    learn.count("decisions")
                    learn.count("unsure", int(decision.source == "laya+unsure"))
            self._eps[key] = ep
            self._recent.append(ep)
            if bind:
                _bound.set(key)
            if final:
                self._finalize(ep)
            self._tick(now)
            return key

    def route(self, key: str, route: str) -> None:
        """NEO's route for it starts (NEO hasn't answered yet: an early stop counts from later)."""
        with self._lock:
            if ep := self._get(key):
                ep.route = route
                self._touch(ep, self._clock(), responded=False)

    def brain(self, key: str, name: str) -> None:
        """Which provider read this request (the chain's last_used, after it answered)."""
        with self._lock:
            if name and (ep := self._get(key)):
                ep.brain = name

    def step(self, key: str, tool: str, ok: bool, *, halt: bool = False) -> None:
        """One tool ran. `halt`: it stopped to ask the user (NEEDS_CONFIRM) — not done yet."""
        with self._lock:
            if ep := self._get(key):
                if halt:
                    ep.awaiting = True
                else:
                    ep.steps.append((tool, bool(ok)))
                self._touch(ep, self._clock())

    def reply(self, key: str, text: str) -> None:
        """NEO's chat route answered (the words are judged, never kept)."""
        with self._lock:
            if ep := self._get(key):
                text = (text or "").strip()
                ep.answer = _judge(text)
                ep.asked = bool(text and _ASKED_BACK.search(text))
                self._end(ep, key)

    def agent_done(self, key: str, steps: list[tuple[str, bool]], *, stopped: str, answer: str, question: bool = False) -> None:
        """An agent run for this request ended — its first, or the one that finished the job
        after the user said yes to a question (then it may stop at another question)."""
        with self._lock:
            if ep := self._get(key):
                ep.steps += [(t, bool(ok)) for t, ok in steps]
                ep.stopped = stopped
                ep.awaiting = ep.awaiting or stopped == "needs_confirm"
                text = (answer or "").strip()
                ep.asked = question or bool(text and _ASKED_BACK.search(text))
                if not [t for t, _ in ep.steps if t not in _IGNORE]:
                    real = len(text) > 40 and not _ASKED_BACK.search(text) and not _UNABLE.search(text)
                    ep.answer = "real" if real and not question else "asked_back"
                else:
                    ep.unable = ep.unable or bool(_UNABLE.search(text))
                self._end(ep, key)

    def end(self, key: str) -> None:
        with self._lock:
            if ep := self._get(key):
                self._end(ep, key)

    def cancelled(self, key: str = "") -> None:
        with self._lock:
            if ep := self._get(key):
                ep.cancelled = True
                self._end(ep, key, responded=False)

    def _end(self, ep: Episode, key: str, *, responded: bool = True) -> None:
        """NEO's route for its own request is over (a relayed run's end isn't its sentence's)."""
        now = self._clock()
        if key == ep.key:
            ep.running = False
        self._touch(ep, now, responded=responded)
        self._tick(now)

    def call_started(self, key: str) -> str:
        """A Live tool call (or its command lane) for this sentence starts, in the current task."""
        _bound.set(key or "")
        with self._lock:
            if ep := self._eps.get(key or ""):
                ep.calls += 1
                self._touch(ep, self._clock())
        return key or ""

    def call_done(self, key: str) -> None:
        with self._lock:
            if ep := self._eps.get(key or ""):
                ep.calls = max(0, ep.calls - 1)
                now = self._clock()
                self._touch(ep, now)
                self._tick(now)

    def touch(self, key: str) -> None:
        """Something happened for this request (Live finished speaking its answer)."""
        with self._lock:
            if ep := self._eps.get(key or ""):
                self._touch(ep, self._clock())

    def stop(self, key: str = "", *, early: bool | None = None) -> None:
        """The user said stop / pressed stop. `early` is known when the caller can tell how long
        NEO had been speaking; otherwise it's measured from when NEO responded."""
        with self._lock:
            now = self._clock()
            self._tick(now)  # never a verdict on a request that should have settled already
            if ep := (self._eps.get(key) if key else None) or self._latest():
                self._interrupt(ep, now, early, stop=True)

    def barge_in(self, key: str = "", *, early: bool) -> None:
        """The user talked over NEO. Only right at the start is that a verdict on the answer."""
        with self._lock:
            now = self._clock()
            self._tick(now)
            if early and (ep := (self._eps.get(key) if key else None) or self._latest()):
                self._interrupt(ep, now, True, stop=False)

    def _interrupt(self, ep: Episode, now: float, early: bool | None, *, stop: bool) -> None:
        if early is None:
            early = now - (ep.responded or ep.started) <= EARLY_S
        if stop and (ep.running or ep.calls):
            ep.cancelled = True
        if early and not ep.early_stop:
            ep.early_stop = True
            learn.count("barge_ins")
        if early or ep.running or ep.calls:
            self._touch(ep, now, responded=False)  # a stop long after says nothing: no new window

    def confirm(self, yes: bool, tool: str = "", ok: bool = True, *, more: bool = False) -> str:
        """The user answered the pending confirmation; a yes ran `tool`. `more`: an agent run now
        finishes the job — the request stays open until it reports (agent_done with the key
        returned here)."""
        with self._lock:
            now = self._clock()
            self._tick(now)  # a question that lapsed has settled: a late answer isn't its outcome
            if ep := self._latest(lambda e: e.awaiting):
                ep.awaiting, ep.confirm = False, bool(yes)
                if yes and tool:
                    ep.steps.append((tool, bool(ok)))
                    ep.stopped = "done" if ep.stopped == "needs_confirm" else ep.stopped
                ep.running = ep.running or (bool(yes) and more)
                self._touch(ep, now)
                self._tick(now)
                return ep.key
            return ""

    def decided(self, utt_or_text: str, kind: str, app_key: str = "", ok: bool = True) -> bool:
        """neo/decide carried out a task kind for this request (by episode key, the bound
        episode, or its words). A kind is only learned when it worked and nothing corrected it."""
        with self._lock:
            ep = self._eps.get(utt_or_text) or self._eps.get(_bound.get())
            if ep is None:
                want = learn.norm(utt_or_text)
                ep = self._latest(lambda e: learn.norm(e.text) == want)
            if ep is None or not kind:
                return False
            ep.kind, ep.kind_ok = str(kind), bool(ok)
            return True

    def file_live(self, lab: dict) -> None:
        """Gemini Live is done with a sentence: its final words, the tools the model called,
        whether it spoke. Room noise that nothing came of is dropped."""
        with self._lock:
            key = str(lab.get("id") or "")
            if not key:
                key = self.begin("", str(lab.get("text", "")), source="live", final=False, bind=False)
            if not (ep := self._eps.get(key)):
                return  # settled already, or erased
            ep.text = " ".join(str(lab.get("text", "")).split())
            ep.teacher = [t for t in lab.get("tools", []) if t not in _IGNORE]
            ep.spoke = ep.spoke or bool(lab.get("spoke"))
            ep.filed = True
            if not ep.text or not (ep.teacher or ep.spoke or ep.steps or ep.answer):
                self._drop(ep)
            else:
                self._finalize(ep)
            self._tick(self._clock())

    # ---- settling ---------------------------------------------------------------------------------
    def _finalize(self, ep: Episode) -> None:
        """The words are final: a bare stop or a yes/no/thanks isn't a request of its own."""
        prev = self._neighbour(ep, -1)
        prev = prev if prev is not None and not prev.settled else None
        from neo.reflex import is_bare_stop

        if is_bare_stop(ep.text):
            if prev is not None:
                self._interrupt(prev, ep.started, None, stop=True)
            self._drop(ep)
        elif prev is not None and (prev.awaiting or prev.confirm is not None or prev.asked) and _answers(ep.text):
            self._drop(ep)  # "yes, go ahead" / "no, that's all" answered NEO's question: not a request
        elif learn._SKIP.match(ep.text):
            # A bare "no" right after a request (not at a question): that one was wrong.
            if (
                prev is not None
                and _CORRECTION.match(ep.text)
                and prev.confirm is None
                and not prev.awaiting
                and not prev.asked
                and ep.started - prev.last <= WINDOW_S
            ):
                prev.corrected = True
                self._touch(prev, ep.started, responded=False)  # the fix may follow
            self._drop(ep)

    def _drop(self, ep: Episode) -> None:
        ep.settled = True
        self._eps.pop(ep.key, None)
        if ep in self._recent:
            self._recent.remove(ep)

    def tick(self, now: float | None = None) -> None:
        with self._lock:
            self._tick(self._clock() if now is None else now)

    def _tick(self, now: float) -> None:
        progress = True
        while progress:
            progress = any([self._try_settle(ep, now) for ep in list(self._eps.values())])
        while self._recent and (self._recent[0].settled or len(self._recent) > 64):
            if not self._recent[0].settled:  # a runaway backlog: nothing that old is still news
                self._recent[0].cancelled = True
                self._settle(self._recent[0])
            self._recent.pop(0)

    def _try_settle(self, ep: Episode, now: float) -> bool:
        if ep.settled:
            return False
        if not ep.filed or ep.running or ep.calls:
            if now - ep.last >= STALE_S:
                ep.cancelled = True  # never finished: nothing to learn from it
                self._settle(ep)
                return True
            return False
        if ep.awaiting and now - ep.last < CONFIRM_WAIT_S:
            return False
        if ep.fix is not None:
            if ep.fix.settled:
                self._settle(ep, ep.fix)
                return True
            return False
        nxt = self._neighbour(ep, 1)
        if nxt is None:
            if now - ep.last >= SETTLE_S:
                self._settle(ep)
                return True
            return False
        if nxt.started - ep.last > WINDOW_S:
            self._settle(ep)  # the next request came much later: unrelated
            return True
        if not nxt.filed:
            if now - ep.last >= STALE_S:
                self._settle(ep)
                return True
            return False  # its words aren't final yet: it may still be a correction
        if self._corrects(ep, nxt):
            ep.fix = nxt
            return False  # settles with it
        self._settle(ep)
        return True

    def _corrects(self, a: Episode, b: Episode) -> bool:
        """Is `b` the user correcting or rephrasing `a`?"""
        if a.confirm is not None or a.awaiting or b.started - a.last > WINDOW_S:
            return False  # a "no" that answered a question isn't about the request
        if a.asked and _REPLY.match(b.text):
            return False  # "no thanks, that's all" answered what NEO asked at the end
        if _CORRECTION.match(b.text):
            if _new_command(a.text, b.text):
                return False
            a.corrected = True
            return True
        out = _outcome(a)
        if out is not None and out[1] and not (a.corrected or a.early_stop):
            # It worked: only saying so makes it wrong. Similar words are a request of their own
            # ("turn the volume up" → "turn the brightness up too", "…on YouTube").
            return False
        ta, tb = _tokens(a.text), _tokens(b.text)
        return bool(ta and tb and len(ta & tb) / len(ta | tb) >= JACCARD)

    def _settle(self, ep: Episode, fix: Episode | None = None) -> None:
        ep.settled = True
        self._eps.pop(ep.key, None)
        learn.count("requests")
        # `resolved` carries a verified label back along a chain of fixes ("open spotfy" → "no,
        # spotify" → "I said Spotify"), even when a row itself isn't kept (too short, content…),
        # and `resolved_tools` every tool behind it: a content tool two fixes on still means the
        # first request's words were content.
        ep.resolved_tools = tuple(ep.teacher) + tuple(t for t, _ in ep.steps)
        if fix is not None and fix.resolved is not None:
            before = _attempted(ep)
            wrong = [before] if before and before != _fmt(fix.resolved) else []
            ep.corrected = ep.corrected or bool(wrong)
            ep.resolved = fix.resolved
            ep.resolved_tools += fix.resolved_tools
            self._emit(ep, fix.resolved, True, wrong=wrong, also=fix.resolved_tools)
        elif fix is None and not (ep.early_stop or ep.corrected) and (out := _outcome(ep)) is not None:
            label, verified = out
            ep.resolved = label if verified else None
            self._emit(ep, label, verified)
        if ep.corrected:
            learn.count("corrections")
        ep.fix = None  # nothing further needs it; don't keep a chain of old requests alive

    def _emit(self, ep: Episode, label: tuple[str, str], verified: bool, *, wrong=(), also=()) -> bool:
        d = ep.decision or {}
        ok = learn.record(
            ep.text,
            label[0],
            label[1],
            source=ep.source,
            tools=tuple(ep.teacher) + tuple(t for t, _ in ep.steps) + tuple(also),
            w=1.0 if verified else 0.5,
            verified=verified,
            wrong=tuple(wrong),
            conf=d.get("conf"),
            margin=d.get("margin"),
            kind=ep.kind if (ep.kind_ok and not ep.corrected and not wrong) else "",
            brain=ep.brain,
        )
        if ok and wrong:
            from neo.reflex import memory

            memory.memory().learned(ep.text, *label)
        return bool(ok)

    def forget(self) -> None:
        with self._lock:
            for ep in self._eps.values():
                ep.settled = True
            self._eps.clear()
            self._recent.clear()

    def __len__(self) -> int:
        return len(self._eps)


_ledger = Ledger()


def ledger() -> Ledger:
    return _ledger


# Module-level hooks: what session.py and live.py call. All best effort.
begin = _safe(lambda *a, **kw: _ledger.begin(*a, **kw))
route = _safe(lambda *a, **kw: _ledger.route(*a, **kw))
brain = _safe(lambda *a, **kw: _ledger.brain(*a, **kw))
step = _safe(lambda *a, **kw: _ledger.step(*a, **kw))
reply = _safe(lambda *a, **kw: _ledger.reply(*a, **kw))
agent_done = _safe(lambda *a, **kw: _ledger.agent_done(*a, **kw))
end = _safe(lambda *a, **kw: _ledger.end(*a, **kw))
cancelled = _safe(lambda *a, **kw: _ledger.cancelled(*a, **kw))
call_started = _safe(lambda *a, **kw: _ledger.call_started(*a, **kw))
call_done = _safe(lambda *a, **kw: _ledger.call_done(*a, **kw))
touch = _safe(lambda *a, **kw: _ledger.touch(*a, **kw))
stop = _safe(lambda *a, **kw: _ledger.stop(*a, **kw))
barge_in = _safe(lambda *a, **kw: _ledger.barge_in(*a, **kw))
confirm = _safe(lambda *a, **kw: _ledger.confirm(*a, **kw))
file_live = _safe(lambda *a, **kw: _ledger.file_live(*a, **kw))
tick = _safe(lambda *a, **kw: _ledger.tick(*a, **kw))
count = _safe(learn.count)


@_safe
def decided(utt_or_text: str, kind: str, app_key: str = "", ok: bool = True) -> bool:
    """For neo/decide: the task kind it chose for this request (an episode key or the request's
    words) and whether carrying it out worked. A (text, kind) row is learned only when it worked
    and nothing corrected the request."""
    return bool(_ledger.decided(utt_or_text, kind, app_key, ok))


def forget() -> None:
    """Erase: every request in flight is forgotten; nothing more will be learned from them."""
    _ledger.forget()
