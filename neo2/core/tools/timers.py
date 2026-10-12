"""NEO's running timers: list them, cancel one, say how long is left.

The `timer` tool (neo2/core/tools/system.py) sets them and keeps them in timers.json; this tool only
reads that file and takes entries out of it (system.cancel_timers), so setting a timer is exactly
what it was. Which timer the user means is read from their words alone: its name ("the pasta
timer"), or its place when several share a name ("the second one"). A duration is no name ("the 10
minute timer" says nothing timers.json keeps), so with several running NEO asks which — a plain
question, not a halt: the answer is just the next request, and for a minute a bare answer ("the
pasta one", "the second one") is read as one. Nothing here is written anywhere but timers.json.

The words reach it only while one of NEO's own timers runs, and only when a name they carry is one of
those timers' ("the pasta timer"), a place ("the second timer") or none at all ("the timer"): "turn
off the sleep timer" (the music app's) and "stop the focus timer" (another app's) go on to the other
routes, as they did before this tool.
"""

from __future__ import annotations

import math
import re
import time

from neo2.core.agent.registry import ToolContext, tool
from neo2.core.tools import system
from neo2.core.tools.system import _MAC, _P

# ---- the words ------------------------------------------------------------------------------
# Whole-utterance and anchored, like every fast path: a sentence that merely mentions a timer ("the
# timer app is broken") never matches. The name, when there is one, is the words before "timer".
_END = r"(?:\s+(?:please|now|na|yaar))*[\s?.!।]*$"
_THE = r"(?:the\s+|my\s+|that\s+|this\s+)?"
_NAME = r"(?:(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,4}?)\s+)?"
_STOP = r"(?:cancel|stop|delete|remove|clear|kill|end|turn\s+off|switch\s+off)"
_LEFT = r"(?:left|remaining|to\s+go)"
_HOW_LONG = r"how\s+(?:much\s+(?:more\s+)?time|long|many\s+(?:more\s+)?minutes)\s+(?:is\s+|are\s+|do\s+i\s+have\s+|have\s+i\s+got\s+)?"
_STILL = r"(?:\s+(?:hai|he|h|hain))?"
_WALA = r"\s+(?:wala|wale|waala|waale|vala|vale)\s+"
_HI_STOP = r"(?:(?:band|bandh|cancel|stop|khatam)(?:\s+(?:karo|kar\s*do|kardo|kar\s+dijiye|kijiye|kar))?|hatao|hata\s+do|roko|rok\s+do)"
_HI_LEFT = r"(?:kitna|kitni|kitne)\s+(?:time|der|samay|minute|minutes|mins?)\s+(?:bacha|bachi|bache|baaki|baki)" + _STILL


def _a_timer_runs(_args: dict) -> bool:
    """A bare "how much time is left?" is about a timer only while one runs (else it's left to
    the conversation: the exam, the match)."""
    return bool(system.running_timers())


def _neos_timer(args: dict) -> bool:
    """Do these words mean one of NEO's own running timers? Only while one runs, and only when the
    name they carry is one of theirs ("the pasta timer"), a place ("the second timer") or no name at all
    ("the timer", "the 10 minute timer"). "turn off the sleep timer" is the music app's, "stop the focus
    timer" another app's: not NEO's to answer, so they go on to the other routes as before."""
    ts = system.running_timers()
    if not ts:
        return False
    label = str(args.get("label") or "")
    if parts := _parts(label):  # "the pasta timer and the eggs timer"
        return any(_pick(ts, p)[0] for p in parts)
    return bool(_pick(ts, label)[0])


# NEO's "Which one should I cancel?": when it was asked and which timers it named. For a minute after,
# a bare answer ("the pasta one", "the second one") is read as the answer — a chat model can't see
# the timers, so without this it would only say it cancelled one.
_WHICH_S = 60.0
_asked: tuple[float, tuple[str, ...]] = (0.0, ())


def _answers_which(args: dict) -> bool:
    """Is this the answer to NEO's "Which one should I cancel?" — asked within the minute, naming
    exactly one of the timers it listed, by its own name (every word of the answer is in it) or by
    its place? ("open notes" is no answer, even with a timer called Notes running.)"""
    at, ids = _asked
    if not ids or not 0 <= time.time() - at <= _WHICH_S:
        return False
    key = _norm(str(args.get("label") or ""))
    if not key:
        return False
    ts = system.running_timers()
    if key in _ORDINALS:
        hits, _ = _pick(ts, key)
    else:
        hits = [t for t in ts if set(key.split()) <= set(_norm(t["label"]).split())]
    return len(hits) == 1 and hits[0]["id"] in ids


_FAST = [
    # cancel every one: "cancel all my timers", "stop the timers", "sab timers band karo"
    (_P + _STOP + r"\s+(?:all\s+(?:of\s+)?|every\s+|each\s+)?(?:the\s+|my\s+)?timers" + _END, {"action": "cancel", "all": True}, _a_timer_runs),
    (_P + _STOP + r"\s+(?:all|every)\s+(?:the\s+|my\s+)?timer" + _END, {"action": "cancel", "all": True}, _a_timer_runs),
    (r"^\s*(?:sab(?:hi)?|saare|sare|saray)\s+timers?\s+" + _HI_STOP + _END, {"action": "cancel", "all": True}, _a_timer_runs),
    # cancel one: "cancel the timer", "stop the pasta timer", "pasta wala timer band karo"
    (_P + _STOP + r"\s+" + _THE + _NAME + r"timer" + _END, {"action": "cancel", "label": "<label>"}, _neos_timer),
    (_P + _STOP + r"\s+(?:the\s+|my\s+)?timer\s+(?:for|called|named)\s+(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,3}?)" + _END,
     {"action": "cancel", "label": "<label>"}, _neos_timer),
    (r"^\s*(?:(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,3}?)" + _WALA + r")?timer\s+" + _HI_STOP + _END, {"action": "cancel", "label": "<label>"}, _neos_timer),
    (r"^\s*(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,2}?)\s+timer\s+" + _HI_STOP + _END, {"action": "cancel", "label": "<label>"}, _neos_timer),
    (r"^\s*(?:सारे\s+|सब\s+)?टाइमर\s+(?:बंद|रद्द|कैंसल)\s+(?:करो|कर\s+दो|कीजिए|करें)" + _END, {"action": "cancel"}, _neos_timer),
    # the answer to NEO's own "Which one should I cancel?": "the pasta one", "cancel the second one"
    (r"^\s*(?:(?:no\s+|nahi\s+)?(?:cancel|stop|delete|remove|kill|end)\s+)?(?:the\s+|that\s+)?(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,2}?)"
     r"(?:\s+(?:one|timer|wala|wali|wale|waala|vala))?" + _END, {"action": "cancel", "label": "<label>"}, _answers_which),
    # how long is left: "how long is left on the pasta timer", "when will my timer go off"
    (_P + _HOW_LONG + _LEFT + r"\s+(?:on|in|for)\s+" + _THE + _NAME + r"timer" + _END, {"action": "left", "label": "<label>"}, _neos_timer),
    (_P + r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?time\s+(?:left|remaining)\s+(?:on|in|for)\s+" + _THE + _NAME + r"timer" + _END,
     {"action": "left", "label": "<label>"}, _neos_timer),
    (_P + r"how(?:'s|\s+is)\s+" + _THE + _NAME + r"timer(?:\s+(?:doing|going))?" + _END, {"action": "left", "label": "<label>"}, _neos_timer),
    (_P + r"when\s+(?:does|will|is)\s+" + _THE + _NAME + r"timer\s+(?:go(?:ing)?\s+off|going\s+to\s+go\s+off|end|finish|ring|be\s+done|done)" + _END,
     {"action": "left", "label": "<label>"}, _neos_timer),
    (_P + r"how\s+long\s+(?:until|till|til|before)\s+" + _THE + _NAME + r"timer\s+(?:goes\s+off|is\s+(?:done|over|up)|rings|ends|finishes)" + _END,
     {"action": "left", "label": "<label>"}, _neos_timer),
    (_P + r"how\s+much\s+longer(?:\s+(?:on|for)\s+" + _THE + _NAME + r"timer)?" + _END, {"action": "left", "label": "<label>"}, _neos_timer),
    (r"^\s*(?:(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,3}?)" + _WALA + r")?timer\s+(?:mein|me|mai|pe|par)\s+" + _HI_LEFT + _END,
     {"action": "left", "label": "<label>"}, _neos_timer),
    (r"^\s*(?:(?P<label>[\w'’-]+(?:\s+[\w'’-]+){0,2}?)\s+)?timer\s+(?:kitna|kitni|kitne)(?:\s+(?:time|der|samay|minute|minutes|mins?))?\s+"
     r"(?:bacha|bachi|bache|baaki|baki)" + _STILL + _END, {"action": "left", "label": "<label>"}, _neos_timer),
    # …said bare ("how much time is left?", "kitna time bacha hai"): only while a timer runs
    (_P + _HOW_LONG + _LEFT + _END, {"action": "left"}, _a_timer_runs),
    (r"^\s*" + _HI_LEFT + _END, {"action": "left"}, _a_timer_runs),
    (r"^\s*(?:टाइमर\s+में\s+)?कितना\s+(?:समय|टाइम|वक़्त|वक्त)\s+(?:बचा|बाकी|बाक़ी)(?:\s+है)?" + _END, {"action": "left"}, _a_timer_runs),
    # which ones run: "what timers do I have", "list my timers", "mere timers dikhao"
    (_P + r"(?:what|which)\s+timers\s+(?:do\s+i\s+have|have\s+i\s+(?:got|set)|are\s+(?:there|running|on|set|going))(?:\s+(?:right\s+)?now)?" + _END,
     {"action": "list"}),
    (_P + r"(?:list|show)(?:\s+me)?\s+(?:all\s+)?(?:of\s+)?(?:the\s+|my\s+)?timers" + _END, {"action": "list"}),
    (_P + r"(?:do\s+i\s+have|are\s+there|is\s+there|have\s+i\s+(?:got|set))\s+(?:any\s+|a\s+)?timers?(?:\s+(?:running|on|set|going))?(?:\s+(?:right\s+)?now)?" + _END,
     {"action": "list"}),
    (r"^\s*(?:mere\s+|sab(?:hi)?\s+|saare\s+)?timers?\s+(?:dikhao|batao|list\s+karo)" + _END, {"action": "list"}),
    (r"^\s*(?:kaun\s*(?:kaun\s*)?se|kitne)\s+timers?\s+(?:chal\s+rahe|lage|set)" + _STILL + _END, {"action": "list"}),
]  # fmt: skip

# ---- which timer the user means -----------------------------------------------------------------
_FILLER = frozenset((
    "the", "my", "a", "an", "that", "this", "these", "those", "one", "ones", "timer", "timers", "for", "of", "wala",
    "wale", "waala", "waale", "vala", "vale", "ka", "ki", "ke", "mera", "meri", "mere",
))  # fmt: skip
_NUMBERS = frozenset((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "ninety", "hundred", "half", "quarter", "and", "a", "an",
))  # fmt: skip
_UNITS = frozenset((
    "sec", "secs", "seconds", "min", "mins", "minute", "minutes", "hour", "hours", "hr", "hrs", "ghanta", "ghante", "minat",
))  # fmt: skip
_ORDINALS = {
    **dict.fromkeys(("first", "1st", "pehla", "pehle", "pahla", "pehli"), 0),
    **dict.fromkeys(("second", "2nd", "doosra", "dusra", "doosre", "dusre", "doosri", "dusri"), 1),
    **dict.fromkeys(("third", "3rd", "teesra", "tisra", "teesri"), 2),
    **dict.fromkeys(("fourth", "4th"), 3),
    **dict.fromkeys(("fifth", "5th"), 4),
    **dict.fromkeys(("last", "aakhri", "akhri", "aakhiri"), -1),
}
_PLACE = ("first", "second", "third", "fourth", "fifth")


def _norm(s: str) -> str:
    """Lower case, letters and digits only, without the words that name no timer ("the", "timer")."""
    words = re.sub(r"[^\w\s]", " ", s.casefold()).split()
    return " ".join(w for w in words if w not in _FILLER)


def _generic(key: str) -> bool:
    """No name at all: nothing, or only a duration ("10 minute", "five min"). "second" alone is a
    place ("the second timer"), not a unit."""
    words = key.split()
    if words == ["second"]:
        return False
    return all(w.isdigit() or w in _NUMBERS or w in _UNITS for w in words)


def _called(key: str, label: str) -> bool:
    """Does the timer's name answer to these words? ("pasta" names "Pasta al dente"; "the pasta
    water timer" names "Pasta".) A timer with no name of its own ("Timer") answers to none."""
    name = _norm(label)
    return bool(name) and (key in name or set(name.split()) <= set(key.split()))


def _parts(label: str) -> list[str]:
    """"pasta timer and the eggs" → ["pasta timer", "the eggs"]: two timers named at once ([] when the
    words name one)."""
    parts = [p for p in re.split(r"\s+(?:and|aur|or|&)\s+|\s*,\s*", label or "") if _norm(p)]
    return parts if len(parts) > 1 else []


def _pick(ts: list[dict], label: str) -> tuple[list[dict], bool]:
    """The timers these words mean (soonest first), and whether the words named any."""
    key = _norm(label)
    if _generic(key):
        return ts, False
    named = [t for t in ts if _called(key, t["label"])]
    if named:
        return named, True
    if key in _ORDINALS:  # "the second one": its place among them, soonest first (as NEO lists them)
        i = _ORDINALS[key]
        return ([ts[i]] if -len(ts) <= i < len(ts) else []), True
    return [], True


# ---- what NEO says ----------------------------------------------------------------------------


def _n(k: int, unit: str) -> str:
    return f"{k} {unit}{'' if k == 1 else 's'}"


def _left_words(seconds: float) -> str:
    """How long is left, the way a person says it: to the second under ten minutes, then to the
    minute. Rounded up, as a countdown shows it."""
    s = math.ceil(seconds)
    if s < 60:
        return _n(max(s, 0), "second")
    if s < 600:
        m, sec = divmod(s, 60)
        return _n(m, "minute") + (f" {_n(sec, 'second')}" if sec else "")
    m = round(s / 60)
    if m < 60:
        return _n(m, "minute")
    h, m = divmod(m, 60)
    return _n(h, "hour") + (f" {_n(m, 'minute')}" if m else "")


def _left(t: dict, now: float) -> str:
    s = t["due"] - now
    return "going off now" if s <= 0 else f"{_left_words(s)} left"


def _items(ts: list[dict], everyone: list[dict], now: float) -> str:
    """"Tea, 2 minutes left; Pasta, 5 minutes left" — by place ("the first, …") when names don't
    tell them apart."""
    titles = [system.timer_title(t) for t in ts]
    if len(set(titles)) == len(titles):
        return "; ".join(f"{name}, {_left(t, now)}" for name, t in zip(titles, ts, strict=True))
    # (its place among all of them, soonest first: what "the second one" picks)
    place = {t["id"]: i for i, t in enumerate(everyone)}

    def at(i: int) -> str:
        return f"the {_PLACE[i]}" if i < len(_PLACE) else f"number {i + 1}"

    return "; ".join(f"{at(place[t['id']])}, {_left(t, now)}" for t in ts)


def _listing(ts: list[dict], now: float) -> str:
    if len(ts) == 1:
        return f"You have one timer: {_items(ts, ts, now)}."
    return f"You have {len(ts)} timers: {_items(ts, ts, now)}."


def _name(t: dict) -> str:
    """"the Pasta timer", or "your timer" for one with no name of its own."""
    title = system.timer_title(t)
    if not _norm(title):
        return "your timer"
    return f"the {title}" if title.casefold().endswith("timer") else f"the {title} timer"


def _together(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _unknown(label: str, ts: list[dict], now: float) -> str:
    said = " ".join(label.split())[:40]
    return f"You don't have a timer called {said}. {_listing(ts, now)}"


@tool(
    "timers",
    "NEO's running timers: list them, cancel one, or say how long is left. action: list|cancel|left. "
    "label: which timer — its name (\"pasta\") or its place (\"second\"); all=true cancels every one.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "cancel", "left"]},
            "label": {"type": "string"},
            "all": {"type": "boolean"},
        },
        "required": ["action"],
    },
    category="system",
    fast_path=_FAST,
    requires=_MAC,
)
async def timers(a: dict, c: ToolContext) -> str:
    global _asked
    action = str(a.get("action") or "list").strip().lower()
    label = str(a.get("label") or "")
    ts = system.running_timers()
    if not ts:
        return "You have no timers running."
    now = time.time()
    if action == "cancel":
        _asked = (0.0, ())  # any cancel answers (or replaces) the question NEO asked
    if action == "cancel" and a.get("all") in (True, "true", "True", 1):
        gone = await system.cancel_timers([t["id"] for t in ts])
        return f"Cancelled {_name(gone[0])}." if len(gone) == 1 else f"Cancelled all {len(gone)} timers."
    if action == "cancel" and (parts := _parts(label)):
        # "the pasta timer and the eggs timer": each name, when each names exactly one of them
        picked = [_pick(ts, p) for p in parts]
        if all(named and len(hits) == 1 for hits, named in picked):
            ids = list(dict.fromkeys(hits[0]["id"] for hits, _ in picked))
            gone = {t["id"]: t for t in await system.cancel_timers(ids)}
            if gone:  # (said in the order the user named them)
                return f"Cancelled {_together([_name(gone[i]) for i in ids if i in gone])}."
    hits, named = _pick(ts, label)
    if not hits:
        return _unknown(label, ts, now)
    if action == "cancel":
        if len(hits) > 1:
            _asked = (now, tuple(t["id"] for t in hits))
            return f"You have {len(hits)} timers: {_items(hits, ts, now)}. Which one should I cancel?"
        gone = await system.cancel_timers([hits[0]["id"]])
        return f"Cancelled {_name(gone[0])}." if gone else "That timer has just gone off."
    if action == "left":
        if len(hits) == 1:
            return f"{system.timer_title(hits[0])}: {_left(hits[0], now)}."
        return "; ".join(f"{system.timer_title(t)}: {_left(t, now)}" for t in hits) + "."
    return _listing(hits if named else ts, now)
