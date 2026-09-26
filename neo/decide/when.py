"""Time phrases in English, Hinglish and Hindi -> datetimes, for NEO's own fast paths.

"at 5", "tomorrow 5:30 pm", "kal subah 9 baje", "saadhe 5", "agle somvar", "20 minute mein",
"परसों शाम 7 बजे", "3 oct at 10". A model passes ISO times; this only reads the user's own words,
and only a small grammar of them: a phrase with any word outside it is not understood (None),
so the request goes to the model instead of being guessed.

When NEO had to decide something the words didn't say (AM or PM, the hour of "tonight", that 5
o'clock has passed so it means tomorrow), :attr:`When.inferred` says the result in words, to be
said back. Pure: no clock is read (``now`` is passed in) and nothing is stored.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

__all__ = ["When", "parse"]


@dataclass
class When:
    start: datetime | None
    end: datetime | None
    date_only: bool  # a day with no time (an all-day event, a reminder without an alert)
    inferred: str  # '' or what NEO decided, in words: "5 PM tomorrow"


# Hindi words (Devanagari) -> the Hinglish spelling the grammar below reads.
_HINDI = {
    "आज": "aaj", "कल": "kal", "परसों": "parso", "परसो": "parso", "सुबह": "subah", "दोपहर": "dopahar",
    "शाम": "shaam", "रात": "raat", "बजे": "baje", "बजकर": "bajkar", "साढ़े": "saadhe", "साढे": "saadhe",
    "सवा": "sawa", "पौने": "paune", "डेढ़": "dedh", "डेढ": "dedh", "ढाई": "dhai", "मिनट": "minute",
    "घंटे": "ghante", "घंटा": "ghanta", "घंटों": "ghante", "में": "mein", "बाद": "baad", "अगले": "agle",
    "अगला": "agle", "हफ्ते": "hafte", "हफ़्ते": "hafte", "सोमवार": "somvar", "मंगलवार": "mangalvar",
    "बुधवार": "budhvar", "गुरुवार": "guruvar", "बृहस्पतिवार": "guruvar", "शुक्रवार": "shukravar",
    "शनिवार": "shanivar", "रविवार": "ravivar", "इतवार": "ravivar", "था": "tha", "थी": "thi", "थे": "the",
    "को": "ko", "पर": "par", "के": "ke", "आधे": "aadhe", "आधा": "aadhe", "एक": "ek",
}  # fmt: skip
_WEEKDAYS = {
    **dict.fromkeys(("monday", "mon", "somvar", "somvaar"), 0),
    **dict.fromkeys(("tuesday", "tue", "tues", "mangalvar", "mangalvaar", "mangal"), 1),
    **dict.fromkeys(("wednesday", "wed", "budhvar", "budhvaar", "budh"), 2),
    **dict.fromkeys(("thursday", "thu", "thur", "thurs", "guruvar", "guruvaar", "brihaspativar", "veervar"), 3),
    **dict.fromkeys(("friday", "fri", "shukravar", "shukravaar", "shukra"), 4),
    **dict.fromkeys(("saturday", "sat", "shanivar", "shanivaar", "shani"), 5),
    **dict.fromkeys(("sunday", "sun", "ravivar", "ravivaar", "itwaar", "itvaar", "itwar"), 6),
}
_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        (("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
         ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
         ("november", "nov"), ("december", "dec"))
    )
    for m in names
}  # fmt: skip
# Parts of the day and the hour each means when no clock time is given.
_PARTS = {
    **dict.fromkeys(("morning", "subah"), ("morning", 9)),
    **dict.fromkeys(("noon", "midday"), ("noon", 12)),
    **dict.fromkeys(("afternoon", "dopahar"), ("afternoon", 14)),
    **dict.fromkeys(("evening", "shaam", "sham"), ("evening", 18)),
    **dict.fromkeys(("night", "raat", "tonight"), ("night", 21)),
    "midnight": ("midnight", 0),
}
_TODAY = ("today", "aaj")
_TOMORROW = ("tomorrow", "tmrw", "tmr")
_AFTER = ("parso", "parson", "parsoon")
_PAST = ("tha", "thi", "the", "was", "were")  # a past-tense request: "kal" is yesterday (reads only)
_FILLER = frozenset({"at", "on", "by", "around", "about", "the", "of", "this", "from", "ko", "pe", "par", "ke", "for", "o'clock",
                     "oclock", "baje", "bajkar", "mein", "me", "coming", "is"})  # fmt: skip
_MINUTES = ("minute", "minutes", "min", "mins", "minat")
_HOURS = ("hour", "hours", "hr", "hrs", "ghante", "ghanta", "ghanton")
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "ek": 1, "two": 2, "do": 2, "three": 3, "teen": 3}
_ORDINAL = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$")


def _normalise(phrase: str) -> list[str]:
    s = unicodedata.normalize("NFKC", phrase or "").casefold()
    s = s.translate({0x0966 + i: str(i) for i in range(10)})  # Devanagari digits
    s = " ".join(_HINDI.get(w, w) for w in s.split())
    s = re.sub(r"(\d)\s*([ap])\.?\s?m\b\.?", r"\1 \2m", s)  # 5pm, 5 p.m. -> 5 pm
    s = re.sub(r"(\d)\s*(?:-|–|—)\s*(\d)", r"\1 to \2", s)  # 3-4 pm
    s = re.sub(r"[,;!?]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return s.split()


@dataclass
class _Clock:
    hour: int
    minute: int
    ampm: str = ""  # 'am' | 'pm' | '' (unknown)
    h24: bool = False  # written as a 24-hour time (17:30, 09:00)


def _clock(tok: list[str], i: int) -> tuple[_Clock, int] | None:
    """A clock time at tok[i]: (the time, the index after it)."""
    t = tok[i]
    j = i + 1
    frac = {"saadhe": 30, "saade": 30, "sadhe": 30, "sawa": 15, "sava": 15, "paune": -15}.get(t)
    if frac is not None:
        if j < len(tok) and tok[j].isdigit() and 1 <= int(tok[j]) <= 12:
            h = int(tok[j])
            c = _Clock(h - 1, 45) if frac < 0 else _Clock(h, frac)
            if c.hour == 0:
                c.hour = 12
            return c, j + 1
        return None
    if t in ("dedh", "dhai", "dhaai"):
        return _Clock(1 if t == "dedh" else 2, 30), j
    m = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?", t)
    if not m:
        return None
    h, mins = int(m.group(1)), int(m.group(2) or 0)
    if h > 23 or mins > 59:
        return None
    c = _Clock(h, mins, h24=h > 12 or h == 0 or (m.group(2) is not None and t.startswith("0")))
    if j < len(tok) and tok[j] in ("am", "pm"):
        if h > 12 or h == 0:
            return None
        c.ampm, j = tok[j], j + 1
    return c, j


@dataclass
class _Parsed:
    days: list = field(default_factory=list)  # what day words said: a date, "kal" or (weekday, next)
    part: tuple[str, int] | None = None
    clock: _Clock | None = None
    end: _Clock | None = None
    delta: timedelta | None = None
    past: bool = False


def _scan(tok: list[str], now: datetime) -> _Parsed | None:
    """What the phrase's words say, or None at the first word outside the grammar (or a second
    time where one fits: "5 at 6")."""
    p = _Parsed()
    today = now.date()
    i = 0
    while i < len(tok):
        t = tok[i]
        nxt = tok[i + 1] if i + 1 < len(tok) else ""
        if t in _FILLER:
            i += 1
        elif t in _PAST:
            p.past, i = True, i + 1
        elif (rel := (_relative(tok, i + 1) if t == "in" else None) or _relative(tok, i, postfix=True)) is not None:
            if p.delta is not None:
                return None
            p.delta, i = rel
        elif t in _TODAY:
            p.days.append(today)
            i += 1
        elif t in _TOMORROW:
            p.days.append(today + timedelta(days=1))
            i += 1
        elif t == "kal":
            p.days.append("kal")
            i += 1
        elif t in _AFTER or (t == "day" and tok[i + 1 : i + 3] == ["after", "tomorrow"]):
            p.days.append(today + timedelta(days=2))
            i += 1 if t in _AFTER else 3
        elif t == "yesterday":
            p.days.append(today - timedelta(days=1))
            i += 1
        elif t in ("next", "agle", "agla") and nxt in _WEEKDAYS:
            p.days.append((_WEEKDAYS[nxt], True))
            i += 2
        elif t in ("next", "agle", "agla") and nxt in ("week", "hafte"):
            p.days.append(today + timedelta(days=7))
            i += 2
        elif t in _WEEKDAYS:
            p.days.append((_WEEKDAYS[t], False))
            i += 1
        elif t in _PARTS:
            if p.part is not None:
                return None
            if t == "tonight":
                p.days.append(today)
            p.part, i = _PARTS[t], i + 1
        elif (d := _date(tok, i, now)) is not None:
            p.days.append(d[0])
            i = d[1]
        elif (c := _clock(tok, i)) is not None:
            if p.clock is None:
                p.clock, i = c
            elif p.end is None and tok[i - 1] in ("to", "till", "until"):
                p.end, i = c
            else:
                return None
        elif t in ("to", "till", "until") and p.clock is not None and p.end is None:
            i += 1
        else:
            return None  # a word outside the grammar: not sure what was meant
    return p


def _relative(tok: list[str], i: int, postfix: bool = False) -> tuple[timedelta, int] | None:
    """'20 minutes' / 'an hour' / 'half an hour' at tok[i] ('in' already read); with ``postfix``
    the Hinglish order '20 minute mein', 'aadhe ghante mein'."""
    n: float | None = None
    j = i
    if j < len(tok) and tok[j] in ("half", "aadhe", "adha", "aadha"):
        n, j = 0.5, j + 1
        if j < len(tok) and tok[j] in ("an", "a"):
            j += 1
    elif j < len(tok) and (tok[j].isdigit() or tok[j] in _NUMBER_WORDS):
        n, j = float(tok[j]) if tok[j].isdigit() else float(_NUMBER_WORDS[tok[j]]), j + 1
    if n is None or j >= len(tok):
        return None
    unit = tok[j]
    if unit not in _MINUTES and unit not in _HOURS:
        return None
    try:
        d = timedelta(minutes=n) if unit in _MINUTES else timedelta(hours=n)
    except OverflowError:  # past any calendar: not a time NEO can use
        return None
    j += 1
    if postfix:
        if j >= len(tok) or tok[j] not in ("mein", "me", "baad"):
            return None
        j += 1
    return (d, j) if d > timedelta(0) else None


def _date(tok: list[str], i: int, now: datetime) -> tuple[date, int] | None:
    """'3 oct', '3rd october', 'oct 3', 'the 3rd' (+ an optional year): the next such date."""
    def num(t: str) -> int | None:
        m = _ORDINAL.fullmatch(t) or re.fullmatch(r"(\d{1,2})", t)
        return int(m.group(1)) if m else None

    t = tok[i]
    j = i + 1
    nxt = tok[j] if j < len(tok) else ""
    day, month = None, None
    if t in _MONTHS and num(nxt) is not None:
        month, day, j = _MONTHS[t], num(nxt), j + 1
    elif (n := num(t)) is not None and nxt in _MONTHS:
        day, month, j = n, _MONTHS[nxt], j + 1
    elif _ORDINAL.fullmatch(t):
        day = num(t)
    if day is None:
        return None
    year = now.year
    if j < len(tok) and re.fullmatch(r"20\d\d", tok[j]):
        year, j = int(tok[j]), j + 1
        explicit_year = True
    else:
        explicit_year = False
    try:
        if month is None:  # "the 3rd": this month, or next once it has passed
            d = date(year, now.month, day)
            if d < now.date():
                d = date(year + (now.month == 12), now.month % 12 + 1, day)
        else:
            d = date(year, month, day)
            if d < now.date() and not explicit_year:
                d = date(year + 1, month, day)
    except ValueError:
        return None
    return d, j


def _hour24(c: _Clock, part: str) -> int | None:
    """The hour on the 24-hour clock, when the words say (AM/PM, a 24-hour time, a part of the
    day); None when only a guess could tell."""
    h = c.hour
    if c.ampm:
        return (0 if h == 12 else h) if c.ampm == "am" else (12 if h == 12 else h + 12)
    if c.h24:
        return h
    if part == "morning":
        return 0 if h == 12 else h
    if part in ("afternoon", "evening"):
        return 12 if h == 12 else h + 12
    if part == "night":
        return 0 if h == 12 else h if h <= 4 else h + 12
    return None


def _say(dt: datetime, now: datetime) -> str:
    """'5 PM tomorrow', '9:30 AM Monday', '10 AM Saturday 3 October'."""
    h12 = dt.hour % 12 or 12
    clock = f"{h12}{f':{dt.minute:02d}' if dt.minute else ''} {'AM' if dt.hour < 12 else 'PM'}"
    days = (dt.date() - now.date()).days
    if days == 0:
        when = "today"
    elif days == 1:
        when = "tomorrow"
    elif 1 < days < 7:
        when = dt.strftime("%A")
    else:
        when = f"{dt.strftime('%A')} {dt.day} {dt.strftime('%B')}"
    return f"{clock} {when}"


def parse(phrase: str, now: datetime, *, future: bool = True) -> When | None:
    """The time a phrase names, relative to ``now``; None when it isn't sure. ``future``: the
    phrase is about something to do (a reminder, an event), so "kal" is tomorrow and a time that
    has passed means its next occurrence; a read about the past says so ("kal kya tha")."""
    tok = _normalise(phrase)
    if not tok:
        return None
    p = _scan(tok, now)
    if p is None or len(p.days) > 1 or not (p.days or p.part or p.clock or p.delta):
        return None
    if p.delta is not None:  # "in 20 minutes" stands alone
        if p.days or p.part or p.clock:
            return None
        try:
            return When(now + p.delta, None, False, "")
        except OverflowError:  # after the year 9999
            return None
    spec = p.days[0] if p.days else None
    day_explicit = spec is not None
    if spec == "kal":
        day = now.date() + timedelta(days=-1 if p.past and not future else 1)
    elif isinstance(spec, tuple):
        wd, is_next = spec
        if p.past and not future:
            day = now.date() - timedelta(days=(now.weekday() - wd) % 7 or 7)
        else:
            ahead = (wd - now.weekday()) % 7
            day = now.date() + timedelta(days=ahead or (7 if is_next or p.clock is None else 0))
    else:
        day = spec or now.date()
    guessed = False
    part = p.part[0] if p.part else ""
    if p.clock is None:
        if p.part is None:
            return When(datetime.combine(day, time()), None, True, "")  # a day only
        start = datetime.combine(day, time(p.part[1]))
        if part == "midnight" and not day_explicit:
            start += timedelta(days=1)
        if future and start <= now and not day_explicit:
            return None  # "this morning" at noon: already over
        return When(start, None, False, _say(start, now))
    if part in ("noon", "midnight"):
        return None  # "noon at 5": the words disagree
    if p.end is not None and p.end.ampm and not (p.clock.ampm or p.clock.h24):
        # "3 to 4 pm": the start is in the end's half of the day unless that would put it after
        p.clock.ampm = p.end.ampm if p.clock.hour % 12 <= p.end.hour % 12 else "am"
    hour = _hour24(p.clock, part)
    if hour is not None:
        start = datetime.combine(day, time(hour, p.clock.minute))
        if part == "night" and hour <= 4:
            start += timedelta(days=1)  # "raat 2 baje": the small hours after that night
        if future and start <= now and not day_explicit:
            start += timedelta(days=1)
            guessed = True
    elif day_explicit and day != now.date():
        # a bare hour (1 to 12) on another day: a daytime hour (8 AM to 7 PM; 12 is noon)
        h = p.clock.hour
        start = datetime.combine(day, time(h + 12 if h <= 7 else h, p.clock.minute))
        guessed = True
    else:
        # a bare hour today: the next time the clock shows it (the last, for the past), within 12 hours
        h = p.clock.hour % 12
        days = (day,) if day_explicit else (day, day + timedelta(days=1))  # "today at 5" stays today
        opts = [datetime.combine(d, time(hh, p.clock.minute)) for d in days for hh in (h, h + 12)]
        if future:
            start = min((o for o in opts if o > now), default=None)
        else:
            back = [o - timedelta(days=1) for o in opts] if not day_explicit else opts
            start = max((o for o in {*opts, *back} if o <= now), default=None)
        if start is None:
            return None
        guessed = True
    if future and isinstance(spec, tuple) and start <= now:
        start += timedelta(days=7)  # "monday at 9" said on a Monday at 10: next Monday
    end = None
    if p.end is not None:
        eh = _hour24(p.end, part)
        if eh is None:  # "3 pm to 4": the first such hour after the start
            h = p.end.hour % 12
            eh = next((x for x in (h, h + 12) if (x, p.end.minute) > (start.hour, start.minute)), h)
        end = datetime.combine(start.date(), time(eh, p.end.minute))
        if end <= start:
            end += timedelta(days=1)
    return When(start, end, False, _say(start, now) if guessed else "")
