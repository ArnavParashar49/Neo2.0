"""When a memory question is about: "yesterday", "last week", "on Monday", "this morning",
"3 days ago", "in September", "kal", "parso", "pichle hafte", "कल" → a window of time.

Memory looks back, so every phrase is read as the past: "kal" is yesterday (never tomorrow),
"on Monday" the latest Monday that has been, "in March" the latest March. Only a small grammar,
found anywhere in the question; the words it used are taken out, so "what did we say about the
trip last week" searches for "the trip" inside last week. Pure: no clock is read (``now`` is
passed in) and nothing is stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = ["Window", "find"]


@dataclass
class Window:
    since: float  # epoch seconds, inclusive
    until: float  # epoch seconds, exclusive
    label: str  # the window in words: "yesterday", "last week", "on Monday"
    rest: str  # the question without the time words


_DAYS = {
    "monday": 0, "mon": 0, "somvar": 0, "somwar": 0, "सोमवार": 0,
    "tuesday": 1, "tue": 1, "tues": 1, "mangalvar": 1, "mangalwar": 1, "मंगलवार": 1,
    "wednesday": 2, "wed": 2, "budhvar": 2, "budhwar": 2, "बुधवार": 2,
    "thursday": 3, "thu": 3, "thurs": 3, "guruvar": 3, "guruwar": 3, "गुरुवार": 3,
    "friday": 4, "fri": 4, "shukravar": 4, "shukrawar": 4, "शुक्रवार": 4,
    "saturday": 5, "sat": 5, "shanivar": 5, "shaniwar": 5, "शनिवार": 5,
    "sunday": 6, "sun": 6, "ravivar": 6, "raviwar": 6, "itwar": 6, "रविवार": 6, "इतवार": 6,
}  # fmt: skip
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "couple": 2,
    "few": 3, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
}  # fmt: skip
_B = r"(?<![\w\u0900-\u097F])"  # a word starts here (Devanagari vowel signs aren't \w)
_E = r"(?![\w\u0900-\u097F])"
_DAY_WORDS = "|".join(sorted(map(re.escape, _DAYS), key=len, reverse=True))
_MONTH_WORDS = "|".join(sorted(_MONTHS, key=len, reverse=True))
_N = r"(?P<n>\d{1,3}|" + "|".join(_NUM) + r")"

# (pattern, kind). The first that matches anywhere wins; longer phrases come first.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(_B + r"(?:(?:the\s+)?day\s+before\s+yesterday|parso[n]?|परसों)" + _E, re.I), "day-2"),
    (re.compile(_B + r"(?:yesterday|kal|कल)\s+(?P<part>morning|afternoon|evening|night|subah|dopahar|shaam|sham|raat)" + _E, re.I), "day-1"),
    (re.compile(_B + r"(?:last\s+night|kal\s+raat|कल\s+रात)" + _E, re.I), "lastnight"),
    (re.compile(_B + r"(?:yesterday|kal|कल)" + _E, re.I), "day-1"),
    (re.compile(_B + r"(?:this|aaj|आज)\s+(?P<part>morning|afternoon|evening|subah|dopahar|shaam|sham)" + _E, re.I), "day0"),
    (re.compile(_B + r"(?:tonight|aaj\s+raat)" + _E, re.I), "tonight"),
    (re.compile(_B + r"(?:earlier\s+)?(?:today|aaj|आज)" + _E, re.I), "day0"),
    (re.compile(_B + _N + r"\s+(?:of\s+)?(?P<unit>day|din|week|hafte|hafta|month|mahine|mahina)s?\s+(?:ago|back|pehle|pahle)" + _E, re.I), "ago"),
    (re.compile(_B + r"(?:(?:in\s+the\s+|over\s+the\s+)?(?:last|past)|pichhle|pichle|पिछले)\s+" + _N + r"\s+(?P<unit>day|din|week|hafte|hafta|month|mahine|mahina)s?" + _E, re.I), "span"),
    (re.compile(_B + r"(?:last\s+week|pichh?le\s+haft[ea]|पिछले\s+हफ्ते|पिछले\s+हफ़्ते)" + _E, re.I), "lastweek"),
    (re.compile(_B + r"(?:this\s+week|is\s+hafte|इस\s+हफ्ते)" + _E, re.I), "thisweek"),
    (re.compile(_B + r"(?:last\s+month|pichh?le\s+mahine|पिछले\s+महीने)" + _E, re.I), "lastmonth"),
    (re.compile(_B + r"(?:this\s+month|is\s+mahine|इस\s+महीने)" + _E, re.I), "thismonth"),
    (re.compile(_B + r"(?:the\s+other\s+day|recently|lately|a\s+while\s+ago|kuch\s+din\s+pehle)" + _E, re.I), "recent"),
    (re.compile(_B + rf"(?:on\s+|last\s+|this\s+past\s+|pichh?le\s+)?(?P<day>{_DAY_WORDS})(?:\s+ko)?" + _E, re.I), "weekday"),
    (re.compile(_B + rf"(?:on\s+)?(?:the\s+)?(?P<dom>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>{_MONTH_WORDS})" + _E, re.I), "date"),
    (re.compile(_B + rf"(?:on\s+)?(?P<month>{_MONTH_WORDS})\s+(?P<dom>\d{{1,2}})(?:st|nd|rd|th)?" + _E, re.I), "date"),
    (re.compile(_B + rf"(?:in|during|last)\s+(?P<month>{_MONTH_WORDS})" + _E, re.I), "month"),
]
_PARTS = {
    "morning": (5, 12), "subah": (5, 12), "afternoon": (12, 17), "dopahar": (12, 17),
    "evening": (17, 22), "shaam": (17, 22), "sham": (17, 22), "night": (20, 28), "raat": (20, 28),
}  # fmt: skip
# Words that only joined the time phrase to the question ("about the trip *from* last week").
_DANGLING = re.compile(r"\b(?:on|in|from|during|since|of|ko|se|mein|me)\s*$", re.I)


def _midnight(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _number(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM.get(tok.lower(), 1)


def _unit_days(unit: str) -> int:
    u = unit.lower()
    if u.startswith(("week", "haft")):
        return 7
    if u.startswith(("month", "mahin")):
        return 30
    return 1


def _window(kind: str, m: re.Match[str], now: datetime) -> tuple[datetime, datetime, str] | None:
    today = _midnight(now)
    g = m.groupdict()
    if kind in ("day0", "day-1", "day-2"):
        day = today - timedelta(days={"day0": 0, "day-1": 1, "day-2": 2}[kind])
        label = {"day0": "today", "day-1": "yesterday", "day-2": "the day before yesterday"}[kind]
        if part := (g.get("part") or "").lower():
            a, b = _PARTS[part]
            return day + timedelta(hours=a), day + timedelta(hours=b), f"{label} {part}"
        return day, day + timedelta(days=1), label
    if kind == "lastnight":
        y = today - timedelta(days=1)
        return y + timedelta(hours=20), today + timedelta(hours=4), "last night"
    if kind == "tonight":
        return today + timedelta(hours=18), today + timedelta(hours=28), "tonight"
    if kind == "ago":
        n, per = _number(g["n"]), _unit_days(g["unit"])
        if per == 1:
            day = today - timedelta(days=n)
            return day, day + timedelta(days=1), f"{n} day{'s' if n != 1 else ''} ago"
        mid = today - timedelta(days=n * per)
        half = timedelta(days=per / 2 + 1)
        return mid - half, mid + half, m.group(0).strip()
    if kind == "span":
        n, per = _number(g["n"]), _unit_days(g["unit"])
        return today - timedelta(days=n * per), now + timedelta(seconds=1), m.group(0).strip()
    if kind == "lastweek":
        monday = today - timedelta(days=today.weekday())
        return monday - timedelta(days=7), monday, "last week"
    if kind == "thisweek":
        return today - timedelta(days=today.weekday()), now + timedelta(seconds=1), "this week"
    if kind == "lastmonth":
        first = today.replace(day=1)
        prev = (first - timedelta(days=1)).replace(day=1)
        return prev, first, "last month"
    if kind == "thismonth":
        return today.replace(day=1), now + timedelta(seconds=1), "this month"
    if kind == "recent":
        return today - timedelta(days=7), now + timedelta(seconds=1), "the last few days"
    if kind == "weekday":
        wd = _DAYS[g["day"].lower()]
        back = (today.weekday() - wd) % 7 or 7  # "on Monday", said on a Monday, is last week's
        day = today - timedelta(days=back)
        return day, day + timedelta(days=1), f"on {day.strftime('%A')}"
    if kind in ("date", "month"):
        month = _MONTHS[g["month"].lower()]
        year = now.year if month <= now.month else now.year - 1
        if kind == "month":
            first = datetime(year, month, 1)
            nxt = datetime(year + (month == 12), month % 12 + 1, 1)
            return first, nxt, f"in {first.strftime('%B')}"
        try:
            day = datetime(year, month, int(g["dom"]))
        except ValueError:
            return None
        if day > now:
            day = day.replace(year=year - 1)
        return day, day + timedelta(days=1), f"on {day.strftime('%-d %B')}"
    return None


def find(text: str, now: datetime) -> Window | None:
    """The time window `text` names (the first phrase of the grammar found in it), or None."""
    if not text:
        return None
    for rx, kind in _RULES:
        m = rx.search(text)
        if not m:
            continue
        if kind == "weekday" and len(m.group("day")) <= 3 and not re.match(r"(?:on|last)\s", m.group(0), re.I):
            continue  # "sun", "sat", "mon" alone are words too: only as "on Sun" / "last Sat"
        if kind in ("date", "month") and m.group("month").lower() == "may" and not re.match(r"(?:on|in|during|last|the|\d)", m.group(0), re.I):
            continue  # "may 5 people come": a verb, not a month
        w = _window(kind, m, now)
        if w is None:
            continue
        rest = _DANGLING.sub("", text[: m.start()]).rstrip() + " " + text[m.end() :].lstrip()
        return Window(w[0].timestamp(), w[1].timestamp(), w[2], " ".join(rest.split()))
    return None
