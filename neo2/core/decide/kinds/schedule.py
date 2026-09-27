"""Calendar and reminders: add an event, add a reminder, say what's on the calendar.

Where an item goes is the engine's choice among the Mac's shared stores (:mod:`neo2.core.decide.routes`,
written and read through EventKit) and the apps whose own dictionary can make or list events and
reminders (:mod:`neo2.core.decide.aescript`, the way in when the store is closed to NEO); a new event
can also be handed to the app that opens calendar files (``ics``). Every add is read back before
NEO calls it done, and one that may have happened but couldn't be checked is never made twice.

Times come as ISO from a model, or in the user's own words from a fast path (English, Hinglish,
Hindi: :mod:`neo2.core.decide.when`); a time NEO had to infer is said back.

* :func:`carry_out` - do the request (a kind the user switched on);
* :func:`would_choose` - only say what would be done (a kind in shadow): nothing is asked of
  macOS, opened, scripted or learned, and no title leaves this module;
* :func:`check` - the fast paths' test, on cached data only;
* :func:`parse_args` - the tool's arguments as times and names, for any mode;
* :func:`prune` / :func:`forget` - the calendar files NEO handed over go (at start / the erase).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neo2.core.decide import aescript, eventkit, routes, when
from neo2.core.decide.run import Effect, Outcome, Req, Result

if TYPE_CHECKING:
    from neo2.core.decide.engine import Candidate
    from neo2.core.decide.inventory import App

__all__ = [
    "KINDS",
    "OVERALL_S",
    "BUDGET_S",
    "Spec",
    "parse_args",
    "methods_for",
    "accepts",
    "describe",
    "attempt",
    "fallback",
    "carry_out",
    "would_choose",
    "check",
    "ics_text",
    "prune",
    "forget",
]

KINDS = ("add_event", "add_reminder", "read_calendar")
_ENTITY: dict[str, eventkit.Entity] = {"add_event": "event", "add_reminder": "reminder", "read_calendar": "event"}
_ITEM: dict[str, aescript.Item] = {"add_event": "event", "add_reminder": "reminder"}
_WANT = {
    "event": ("title", "start", "end", "all_day", "location", "body"),
    "reminder": ("title", "body", "due", "due_day", "alert"),
}
_NOUN = {"event": ("calendar", "calendars", "events"), "reminder": ("list", "lists", "reminders")}
_PANE = {"event": ("Calendar", "Calendars"), "reminder": ("Reminders", "Reminders")}
_STORE = {"event": "calendar", "reminder": "reminders"}  # "your calendar", "your reminders"

OVERALL_S = {"add_event": 20.0, "add_reminder": 20.0, "read_calendar": 45.0}
BUDGET_S = {"eventkit": 10.0, "script_make": 18.0, "script_read": 40.0, "ics": 20.0}
MAKE_S = 10.0  # the make script's own limit, inside script_make's budget: its timeout is seen here
CHECK_S = 3.0  # a read-back, a search-back or a container listing
READ_S = 20.0  # one calendar's listing by script
READ_AT_ONCE = 3  # calendars listed by script at the same time
ASK_NOTE_S = 5.0  # macOS's question still open after this: tell the user where to click
ICS_KEEP_S = 600.0  # calendar files NEO wrote are deleted after this
MAX_OCCURRENCES = 500  # of one repeating event in one window
EVENT_LEN = timedelta(hours=1)

# ---- what the user asked for ------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """A request's parsed arguments. For a read, ``start``/``end`` are the window."""

    kind: str
    title: str = ""
    start: datetime | None = None  # the event's start, the reminder's due time, the window's start
    end: datetime | None = None
    date_only: bool = False  # an all-day event, a reminder due on a day (no alert)
    inferred: str = ""  # a time NEO decided the words didn't say, in words ("5 PM tomorrow")
    location: str = ""
    notes: str = ""
    container: str = ""  # the calendar or list the user named
    app: str = ""  # the app the user named (the tool's `app`: the words alone never name one)
    phrase: str = ""  # the time in the user's own words, normalised ('' for an ISO one)

    @property
    def sig(self) -> tuple:
        """What makes two adds the same one (a retried request adds nothing new): the title, the
        calendar or list named, and when it is, to the minute (the user's words and a model's ISO
        time for them agree on that)."""
        if self.kind == "read_calendar":
            return ()
        return (_norm(self.title), _norm(self.container), aescript.iso(self.start) if self.start else "")

    @property
    def said_sig(self) -> tuple:
        """The same with the time as said: "in 20 minutes" asked again moves with the clock."""
        if self.kind == "read_calendar" or not self.phrase:
            return ()
        return (_norm(self.title), _norm(self.container), self.phrase)

    @property
    def said_back(self) -> bool:
        """Something about when it is that the user should hear: a time NEO inferred, or an
        all-day event over several days (a range read one day off is heard)."""
        days = self.date_only and self.start and self.end and self.end - self.start > timedelta(days=1)
        return bool(self.inferred or (self.kind == "add_event" and days))


_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _minute(dt: datetime) -> datetime:
    return (dt + timedelta(seconds=30)).replace(second=0, microsecond=0)


def _time(value: Any, now: datetime, *, future: bool) -> when.When | None:
    """An ISO date or time (a model's), else the user's own words (to the nearest minute: "in 20
    minutes" is counted from now); None when unsure."""
    v = " ".join(str(value or "").split())
    if not v:
        return None
    if _ISO_DAY.fullmatch(v):
        try:
            return when.When(datetime.fromisoformat(v), None, True, "")
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        got = when.parse(v, now, future=future)
        if got is None or got.start is None:
            return got
        return replace(got, start=_minute(got.start), end=_minute(got.end) if got.end else None)
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return when.When(dt.replace(second=0, microsecond=0), None, False, "")


def _phrase(value: Any) -> str:
    """A time in the user's own words, normalised; '' for an ISO date or time."""
    v = " ".join(str(value or "").split())
    try:
        datetime.fromisoformat(v)
    except ValueError:
        return _norm(v)
    return ""


def _text(args: dict, key: str) -> str:
    return " ".join(str(args.get(key) or "").split())


def _flag(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "yes", "1")


def parse_args(kind: str, args: dict, now: datetime | None = None) -> Spec | str:
    """The tool's arguments as a :class:`Spec`, or what to say when they can't be used."""
    now = now or datetime.now()
    base = Spec(kind, app=_text(args, "app"))
    if kind == "read_calendar":
        return _window(base, args, now)
    title = _text(args, "title")
    if not title:
        return "What should it be called?"
    if kind == "add_reminder":
        due = None
        if _text(args, "due") and (due := _time(args["due"], now, future=True)) is None:
            return f"I couldn't tell when “{_text(args, 'due')}” is."
        return replace(
            base,
            title=title,
            start=due.start if due else None,
            date_only=bool(due and due.date_only),
            inferred=due.inferred if due else "",
            notes=str(args.get("notes") or "").strip(),
            container=_text(args, "list"),
            phrase=_phrase(args.get("due")) if due else "",
        )
    start = _time(args.get("start"), now, future=True)
    if start is None or start.start is None:
        return f"I couldn't tell when “{_text(args, 'start')}” is." if _text(args, "start") else "When is it?"
    all_day = start.date_only or _flag(args.get("all_day"))
    first = datetime.combine(start.start.date(), datetime.min.time()) if all_day else start.start
    end, end_day = start.end, False
    if _text(args, "end"):
        got = _time(args["end"], start.start, future=True)
        if got is None or got.start is None:
            return f"I couldn't tell when “{_text(args, 'end')}” is."
        end, end_day = got.start, got.date_only
    if all_day:
        # a date alone is the last day (as a read's end is); a midnight is the moment the last
        # day ends (iCalendar's exclusive end), so it is the day before
        last = (end or first).date()
        if end is not None and not end_day and end.time() == datetime.min.time() and end > first:
            last -= timedelta(days=1)
        end = datetime.combine(max(last, first.date()) + timedelta(days=1), datetime.min.time())
    elif end is None:
        end = first + EVENT_LEN
    if end <= first:
        return "That would end before it starts."
    return replace(
        base,
        title=title,
        start=first,
        end=end,
        date_only=all_day,
        inferred=start.inferred,
        location=_text(args, "location"),
        notes=str(args.get("notes") or "").strip(),
        container=_text(args, "calendar"),
        phrase=_phrase(args.get("start")),
    )


def _window(base: Spec, args: dict, now: datetime) -> Spec | str:
    """The day (or ISO range) to read: today unless the words or the range say otherwise."""
    if _text(args, "start"):
        got = _time(args["start"], now, future=False)
        if got is None or got.start is None:
            return f"I couldn't tell when “{_text(args, 'start')}” is."
        t0 = got.start
        t1 = None
        if _text(args, "end"):
            end = _time(args["end"], t0, future=True)
            if end is None or end.start is None:
                return f"I couldn't tell when “{_text(args, 'end')}” is."
            t1 = end.start + (timedelta(days=1) if end.date_only else timedelta(0))
        t1 = t1 or datetime.combine(t0.date() + timedelta(days=1), datetime.min.time())
    elif span := _week(_text(args, "day"), now):
        t0, t1 = span
    else:
        day = now.date()
        if _text(args, "day"):
            got = _time(args["day"], now, future=False)
            if got is None or got.start is None:
                return f"I couldn't tell which day “{_text(args, 'day')}” is."
            day = got.start.date()
        t0 = datetime.combine(day, datetime.min.time())
        t1 = t0 + timedelta(days=1)
    if t1 <= t0:
        return "That range ends before it starts."
    t1 = min(t1, t0 + timedelta(days=eventkit.MAX_RANGE_DAYS))
    return replace(base, start=t0, end=t1, date_only=True)


# "next week", "agle hafte", "this week": a week, not the one day a week from now
_WEEK = frozenset(
    unicodedata.normalize("NFKC", w) for w in ("week", "hafte", "hafta", "हफ्ते", "हफ़्ते", "हफ्ता", "सप्ताह")
)
_NEXT = frozenset({"next", "agle", "agla", "अगले", "अगला"})
_THIS = frozenset({"this", "is", "iss", "इस"})
_AROUND = frozenset({"for", "the", "in", "mein", "me", "में"})  # "for next week", "agle hafte mein"


def _first_weekday() -> int:
    """The first day of the user's week (0 = Monday), from their own calendar settings."""
    try:
        from Foundation import NSCalendar

        return (int(NSCalendar.currentCalendar().firstWeekday()) + 5) % 7  # 1 = Sunday ... 7 = Saturday
    except Exception:  # noqa: BLE001 - no Foundation: the ISO week
        return 0


def _week(phrase: str, now: datetime) -> tuple[datetime, datetime] | None:
    """Next week, or what is left of this one, for a phrase that says so; else None."""
    words = [w for w in re.findall(r"[\wऀ-ॿ]+", _norm(phrase)) if w not in _AROUND]
    if len(words) != 2 or words[1] not in _WEEK or words[0] not in _NEXT | _THIS:
        return None
    today = datetime.combine(now.date(), datetime.min.time())
    begun = today - timedelta(days=(today.weekday() - _first_weekday()) % 7)
    if words[0] in _NEXT:
        return begun + timedelta(days=7), begun + timedelta(days=14)
    return today, begun + timedelta(days=7)


def _spec(req: Req) -> Spec:
    return Spec(**req.fields)


async def _request(spec: Spec, text: str) -> Req:
    """What the engine reads: the user's words, where only the tool's `app` names an app. A
    calendar or to-do app is often called by the kind's own noun ("show my calendar", "put it in
    calendar"), so any app the words seem to name is masked unless the model passed it."""
    said = " ".join((text or "").split()) or spec.title or spec.kind.replace("_", " ")
    if spec.app:
        if spec.app.casefold() not in said.casefold():
            said = f"{said} in {spec.app}"
    else:
        from neo2.core.decide import engine, inventory

        try:
            said = engine.mask_apps(said, await asyncio.to_thread(inventory.inventory))
        except Exception:  # noqa: BLE001 - no inventory: the engine finds no app named either
            pass
    return Req(spec.kind, asdict(spec), said, named=spec.app, sig=spec.sig, also=(spec.said_sig,))


# ---- words: names, times ----------------------------------------------------------------------

# words around a calendar's or list's name that aren't the name ("my work calendar")
_CONTAINER_WORDS = frozenset(
    {"my", "the", "a", "calendar", "calendars", "cal", "list", "lists", "mera", "meri", "mere", "wala", "wali",
     "वाला", "वाली", "मेरा", "मेरी", "कैलेंडर", "लिस्ट"}
)  # fmt: skip
_PRONOUNS = frozenset(
    {"it", "that", "this", "them", "those", "these", "one", "ye", "yeh", "woh", "wo", "use", "usko", "isko",
     "यह", "वह", "इसे", "उसे"}
)  # fmt: skip


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s or "").casefold().split())


def _words(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"[\wऀ-ॿ]+", _norm(s))) - _CONTAINER_WORDS


def _match(named: str, options: list[tuple[str, str]]) -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
    """(the one (id, title) ``named`` means, else None; every option that fits it): the same
    name, else one with every word of the name."""
    want = _norm(named)
    exact = [o for o in options if _norm(o[1]) == want]
    words = _words(named)
    pool = exact or [o for o in options if words and words <= _words(o[1])]
    return (pool[0] if len(pool) == 1 else None), pool


async def _pick(named: str, options: list[tuple[str, str]], text: str) -> tuple[tuple[str, str] | None, bool]:
    """(the (id, title) the user meant by ``named``, else None; whether it fit several). A tie,
    or no fit at all, is Laya's to settle among the user's own titles (local only), and only
    when it is sure."""
    hit, pool = _match(named, options)
    if hit is not None:
        return hit, False
    ask = pool if pool else (options if _words(named) else [])
    return (await _laya_pick(named, ask, text) if len(ask) > 1 else None), len(pool) > 1


def _missed(one: str, named: str, tied: bool, where: str) -> str:
    """What to say when the named calendar or list wasn't used."""
    if tied:
        return f" I couldn't tell which {one} “{named}” is, so I put it in {where}."
    return f" There's no {one} called {named}, so I put it in {where}."


async def _laya_pick(named: str, pool: list[tuple[str, str]], text: str) -> tuple[str, str] | None:
    from neo2.core.decide import engine

    if not engine._zero_shot_can_answer():
        return None
    labels: dict[str, tuple[str, str]] = {}
    for o in pool:
        label, n = o[1] or o[0], 2
        while label in labels:
            label, n = f"{o[1]} ({n})", n + 1
        labels[label] = o
    q = {
        "which": {
            "type": "choice",
            "instructions": f"Which of these did the user mean by “{named}”?",
            "criteria": {label: label for label in labels},
        }
    }
    probs = engine._choice(await engine._ask_laya({"request": text or named}, q), "which")
    if not engine._sure(probs):
        return None
    return labels.get(max(probs, key=lambda k: probs[k]))


def _clock(dt: datetime) -> str:
    """'5 PM', '9:30 AM'."""
    h = dt.hour % 12 or 12
    return f"{h}{f':{dt.minute:02d}' if dt.minute else ''} {'AM' if dt.hour < 12 else 'PM'}"


def _day(d: date, now: datetime) -> str:
    """'today', 'tomorrow', 'Friday', 'Friday 9 October'."""
    days = (d - now.date()).days
    if days in (-1, 0, 1):
        return ("yesterday", "today", "tomorrow")[days + 1]
    name = d.strftime("%A")
    return name if 1 < days < 7 else f"{name} {d.day} {d.strftime('%B')}"


def _when_words(spec: Spec, now: datetime) -> str:
    """When an added item is, in words: 'all day Friday', '1 PM to 2 PM tomorrow', '5 PM today'."""
    if spec.start is None:
        return ""
    day = _day(spec.start.date(), now)
    if spec.date_only:
        if spec.kind != "add_event":
            return day
        last = (spec.end - timedelta(days=1)).date() if spec.end else spec.start.date()
        return f"all day {day}" if last <= spec.start.date() else f"all day {day} to {_day(last, now)}"
    if spec.end is not None and spec.end.date() == spec.start.date():
        return f"{_clock(spec.start)} to {_clock(spec.end)} {day}"
    return f"{_clock(spec.start)} {day}"


def _added(spec: Spec, where: str, now: datetime) -> str:
    words = _when_words(spec, now)
    return f"Added “{spec.title}” to {where}{f', {words}' if words else ''}."


def _one_of(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


# ---- the driver: candidates' methods ----------------------------------------------------------


def _route_entity(app: App) -> eventkit.Entity | None:
    return routes.ROUTES.get(app.bundle_id.removeprefix("route:")) if app.kind == "route" else None


def methods_for(app: App, kind: str | None = None) -> tuple[str, ...]:
    """How this app (or store) can do ``kind`` (every schedule kind when None): the store through
    EventKit; an app through its dictionary's make or list route."""
    if app.kind == "route":
        entity = _route_entity(app)
        return ("eventkit",) if entity is not None and (kind is None or _ENTITY.get(kind) == entity) else ()
    d = getattr(app, "script", None)
    if d is None:
        return ()
    from neo2.core.decide import sdef

    tags = sdef.caps(d) | sdef.read_caps(d)
    out = []
    for k, tag, item in (("add_event", "event.make", "event"), ("add_reminder", "reminder.make", "reminder")):
        if kind in (None, k) and tag in tags and aescript.create_route(d, item, _WANT[item]) is not None:
            out.append("script_make")
    if kind in (None, "read_calendar") and "event.read" in tags and aescript.list_route(d) is not None:
        out.append("script_read")
    return tuple(dict.fromkeys(out))


def accepts(app: App, method: str, req: Req) -> bool:
    return method in methods_for(app, req.kind)


def describe(app: App) -> str:
    """What the app can do here, in words for Laya."""
    if app.kind == "route":
        return app.note
    words = {"script_make": "can add by script", "script_read": "can list events by script"}
    return ", ".join(words[m] for m in methods_for(app))


class _Driver:
    """One schedule kind as :func:`neo2.core.decide.run.run_best` drives it."""

    def __init__(self, kind: str) -> None:
        self.KIND, self.OVERALL_S, self.BUDGET_S = kind, OVERALL_S[kind], BUDGET_S

    def methods_for(self, app: App, kind: str | None = None) -> tuple[str, ...]:
        return methods_for(app, kind or self.KIND)

    def accepts(self, app: App, method: str, req: Req) -> bool:
        return accepts(app, method, req)

    def describe(self, app: App) -> str:
        return describe(app)

    async def attempt(self, req: Req, cand: Candidate, method: str) -> Outcome:
        return await attempt(req, cand, method)

    async def fallback(self, req: Req, key: str) -> Outcome | None:
        return await fallback(req, key)


_DRIVERS = {k: _Driver(k) for k in KINDS}


async def attempt(req: Req, cand: Candidate, method: str) -> Outcome:
    spec = _spec(req)
    app = cand.app
    assert app is not None  # the fallback candidate goes to fallback()
    if method == "eventkit":
        return await (_store_read(spec) if spec.kind == "read_calendar" else _store_add(spec, req))
    if method == "script_make":
        return await _script_make(spec, req, app)
    if method == "script_read":
        return await _script_read(spec, app)
    return Outcome(False, f"{app.name} can't do that", reason="route_bad")


# ---- the store (EventKit) -------------------------------------------------------------------------


def _settings_line(entity: eventkit.Entity) -> str:
    noun, pane = _PANE[entity]
    return f"{noun} access is off for NEO: System Settings › Privacy & Security › {pane}"


def _no_access(entity: eventkit.Entity, st: str, *, reading: bool = False) -> Outcome:
    noun = _STORE[entity]
    if st == "write_only" and reading:
        text = f"NEO may only add to your {noun}, not read it: System Settings › Privacy & Security › {_PANE[entity][1]}"
    elif st == "not_determined":
        text = f"NEO hasn't been allowed to use your {noun} yet"
    else:
        text = _settings_line(entity)
    return Outcome(False, text, reason="permission")


async def _store_add(spec: Spec, req: Req) -> Outcome:
    entity = _ENTITY[spec.kind]
    one = _NOUN[entity][0]
    st = eventkit.status(entity)
    if st not in ("full", "write_only"):
        return _no_access(entity, st)
    now = datetime.now()
    cal_id, where, note = "", f"your default {one}", ""
    try:
        if st == "full":
            cals = await asyncio.to_thread(eventkit.calendars, entity)
            default = await asyncio.to_thread(eventkit.default_calendar, entity)
            where = default.title if default is not None and default.title else where
            if spec.container:
                pick, tied = await _pick(spec.container, [(c.id, c.title) for c in cals], req.text)
                if pick is not None:
                    cal_id, where = pick
                else:
                    note = _missed(one, spec.container, tied, where)
        elif spec.container:
            note = f" NEO can't see your {_NOUN[entity][1]}, so it went to your default {one}."
        if entity == "event":
            assert spec.start is not None and spec.end is not None
            # an all-day event ends on its last day in the store (the day after would add a day)
            end = spec.end - timedelta(days=1) if spec.date_only else spec.end
            ev = eventkit.EventSpec(spec.title, spec.start, max(end, spec.start), spec.date_only, spec.location, spec.notes, cal_id)
            got = await _commit_thread(eventkit.save_event, ev)
        else:
            rem = eventkit.ReminderSpec(spec.title, spec.start, spec.date_only, spec.notes, cal_id)
            got = await _commit_thread(eventkit.save_reminder, rem)
    except eventkit.NoAccess:  # taken back meanwhile
        return _no_access(entity, eventkit.status(entity))
    except RuntimeError as e:  # the store refused it: nothing was saved
        return Outcome(False, str(e)[:120])
    receipt = {"id": got.id, "where": f"route:{_route(entity)}", "container": got.calendar_id, "verified": got.verified}
    if st == "full" and not got.verified:
        return Outcome(
            False,
            "I may have added it but couldn't check. Have a look in your calendar.",
            reason="unconfirmed",
            effect=Effect.UNKNOWN,
            receipt=receipt,
            say=True,
        )
    clash = await _overlap(spec, got.id) if st == "full" and entity == "event" and not spec.date_only else ""
    text = _added(spec, where, now) + note + clash
    if st == "write_only":
        text += " (I can't double-check with the access I have.)"
    say = True if (note or clash or spec.said_back or not got.verified) else None
    return Outcome(True, text, effect=Effect.DONE, receipt=receipt, say=say)


def _route(entity: eventkit.Entity) -> str:
    return next(r for r, e in routes.ROUTES.items() if e == entity)


async def _commit_thread(fn: Any, arg: Any) -> Any:
    """The one call that saves: past it the item may exist (see run.commit)."""
    from neo2.core.decide import run

    return await run.commit(asyncio.to_thread(fn, arg))


async def _overlap(spec: Spec, new_id: str) -> str:
    """' It overlaps with another event at 10 AM.' (never another event's title: an invitation's
    title is someone else's text)."""
    assert spec.start is not None and spec.end is not None
    try:
        rows = await asyncio.to_thread(eventkit.events_between, spec.start, spec.end)
    except Exception:  # noqa: BLE001 - a conflict note is a nicety
        return ""
    others = [r for r in rows if r.id != new_id and not r.all_day and r.start < spec.end and r.end > spec.start]
    if not others:
        return ""
    first = min(others, key=lambda r: r.start)
    more = "another event" if len(others) == 1 else f"{len(others)} other events"
    return f" It overlaps with {more} at {_clock(first.start)}."


@dataclass(frozen=True)
class _Occ:
    """One occurrence of an event in the window being read."""

    title: str
    start: datetime
    end: datetime
    all_day: bool
    calendar: str


async def _store_read(spec: Spec) -> Outcome:
    st = eventkit.status("event")
    if st != "full":
        return _no_access("event", st, reading=True)
    assert spec.start is not None and spec.end is not None
    try:
        rows = await asyncio.to_thread(eventkit.events_between, spec.start, spec.end)
    except eventkit.NoAccess:
        return _no_access("event", eventkit.status("event"), reading=True)
    occ = [_Occ(r.title, r.start, r.end, r.all_day, r.calendar) for r in rows if r.start < spec.end and r.end > spec.start]
    return _read_outcome(spec, occ, "route:eventkit.events")


def _read_outcome(spec: Spec, occ: list[_Occ], where: str, extra: str = "") -> Outcome:
    """What a successful read says: the events, all-day ones first, one line each; or that the
    calendar is empty for that time (only ever after a read that worked)."""
    assert spec.start is not None and spec.end is not None
    now = datetime.now()
    one_day = spec.end - spec.start <= timedelta(days=1)
    first = _day(spec.start.date(), now)
    if one_day:
        span = f"on {first}" if first[0].isupper() else first
    else:
        span = f"from {first} to {_day((spec.end - timedelta(seconds=1)).date(), now)}"
    receipt = {"where": where, "items": len(occ)}
    if not occ:
        text = f"Nothing on your calendar {span}.{extra}"
        return Outcome(True, text, effect=Effect.DONE, receipt=receipt, data=text)
    occ = sorted(set(occ), key=lambda o: (o.start.date(), not o.all_day, o.start, o.title))
    lines = [f"{len(occ)} event{'s' if len(occ) != 1 else ''} {span}:"]
    day = None
    for o in occ:
        if not one_day and o.start.date() != day:
            day = o.start.date()
            lines.append(f"{_day(day, now).capitalize()}:")
        cal = f" ({o.calendar})" if o.calendar else ""
        lines.append(f"all day: {o.title}{cal}" if o.all_day else f"{o.start:%H:%M}–{o.end:%H:%M} {o.title}{cal}")
    text = "\n".join(lines) + extra
    return Outcome(True, lines[0], effect=Effect.DONE, receipt=receipt, data=text)


# ---- an app's dictionary (AppleScript) ------------------------------------------------------------


async def _listing(r: aescript.CreateRoute, app: App) -> list[tuple[str, str, bool]] | Outcome:
    """The app's calendars or lists: (id or name, name, whether NEO may add to it)."""
    got = aescript.containers(r, app.bundle_id)
    if got is None:
        return []
    code, out, err = await aescript.run(*got, CHECK_S)
    if code != 0:
        return aescript.fail(app.name, code, err, committed=False)
    _status, rows = aescript.rows(out)
    return [(row[0], row[1], row[2] != "false") for row in rows if len(row) >= 3 and row[0]]


def _values(r: aescript.CreateRoute, spec: Spec) -> dict[str, Any]:
    """Slot -> value for the make: a timed reminder's due time is also its alert."""
    have = {s for s, _c, _k in r.sets}
    v: dict[str, Any] = {"title": spec.title, "body": spec.notes or None}
    if r.item == "event":
        v.update(start=spec.start, end=spec.end, location=spec.location or None, all_day=spec.date_only or None)
    elif spec.start is not None:
        if spec.date_only:
            v["due_day" if "due_day" in have else "due"] = spec.start
        else:
            v.update(due=spec.start, alert=spec.start)
    return v


def _same_time(got: str, want: datetime | None) -> bool:
    try:
        return want is not None and datetime.fromisoformat(got).replace(second=0) == want.replace(second=0, microsecond=0)
    except ValueError:
        return False


async def _check(r: aescript.CreateRoute, app: App, spec: Spec, oid: str, cid: str) -> bool | None:
    """Read the new item back by its id: True when its title (and time) are what was asked,
    False when it isn't there or differs, None when the app couldn't say."""
    got = aescript.readback(r, app.bundle_id, oid, cid)
    if got is None:
        return None
    code, out, _err = await aescript.run(*got, CHECK_S)
    status, rows = aescript.rows(out) if code == 0 else ("", [])
    if status == "EMPTY":
        return False
    if status != "OK" or not rows:
        return None
    vals = dict(zip((s for s, _c, _k in r.reads), rows[0], strict=False))
    if _norm(vals.get("title", "")) != _norm(spec.title):
        return False
    timed = "start" if r.item == "event" else "due"
    if spec.start is not None and not spec.date_only and timed in vals:
        return _same_time(vals[timed], spec.start)
    return True


async def _search_back(r: aescript.CreateRoute, app: App, spec: Spec, since: datetime, cid: str) -> str:
    """After a make that timed out: the id of the item it made after all ('' if none is found)."""
    got = aescript.search_back(r, app.bundle_id, spec.title, since, spec.start, cid)
    if got is None:
        return ""
    code, out, _err = await aescript.run(*got, CHECK_S)
    status, rows = aescript.rows(out) if code == 0 else ("", [])
    return rows[0][0] if status == "OK" and rows and rows[0] else ""


# "<kind>:<bundle id>" -> when NEO asked which calendar or list to use for new items there: only
# the answer to that question is remembered for later adds that name none
_ASKED: dict[str, float] = {}


def _answering(key: str) -> bool:
    """This request names a calendar or list soon after NEO asked which one to use there."""
    from neo2.core.decide import run

    return time.monotonic() - _ASKED.get(key, float("-inf")) <= run.RECENT_S


async def _script_make(spec: Spec, req: Req, app: App) -> Outcome:
    from neo2.core.decide import learned, run

    item = _ITEM[spec.kind]
    one = _NOUN[item][0]
    r = aescript.create_route(app.script, item, _WANT[item])
    if r is None:
        return Outcome(False, f"{app.name} can't add that", reason="route_bad")
    key = f"{spec.kind}:{app.bundle_id}"
    now = datetime.now()
    cid, where, remember, tied = "", app.name, False, False
    listed: list[tuple[str, str, bool]] | None = None
    if spec.container and r.listable:
        got = await _listing(r, app)
        if isinstance(got, Outcome):
            return got
        listed = got
        pick, tied = await _pick(spec.container, [(i, n) for i, n, w in listed if w], req.text)
        if pick is not None:
            cid, where = pick[0], f"{pick[1]} in {app.name}"
            remember = not r.default and _answering(key)
    missing = bool(spec.container) and not cid
    if not cid and not r.default and r.container:
        # no default of its own: the one NEO was told before, the only one, or ask once
        hit = learned.get("containers", key)
        if hit and isinstance(hit.get("id"), str):
            cid = hit["id"]
        else:
            if listed is None:
                got = await _listing(r, app)
                if isinstance(got, Outcome):
                    return got
                listed = got
            open_ = [(i, n) for i, n, w in listed if w]
            if not open_:
                return Outcome(False, f"{app.name} has no {one} NEO could add to", reason="not_found")
            if len(open_) > 1:
                from neo2.core.agent import confirm

                names = [n for _i, n in open_][:6]
                ask = f"Which {one} should I use for new {_NOUN[item][2]} in {app.name}: {_one_of(names)}?"
                _ASKED[key] = time.monotonic()
                return Outcome(False, confirm.ask_user(ask), reason="needs_user")
            cid, where = open_[0][0], f"{open_[0][1]} in {app.name}"
    script, argv = aescript.render_create(r, app.bundle_id, _values(r, spec), cid)
    since = datetime.now() - timedelta(seconds=2)
    code, out, err = await run.commit(aescript.run(script, argv, MAKE_S))
    status, rows = aescript.rows(out) if code == 0 else ("", [])
    oid, name = "", ""
    if code != 0:
        failed = aescript.fail(app.name, code, err, committed=True)
        # a make cut short may have made it after all, and an object the app couldn't resolve may
        # have been one it made (never a second copy): look before trying anything else
        if failed.effect is Effect.UNKNOWN or failed.reason == "not_found":
            oid = await _search_back(r, app, spec, since, cid)
        if not oid and failed.effect is not Effect.UNKNOWN:
            return failed
        if not oid:
            return Outcome(
                False,
                f"I may have added it in {app.name} but couldn't check. Have a look there.",
                reason="unconfirmed",
                effect=Effect.UNKNOWN,
                say=True,
            )
    elif status == "NO_CONTAINER":
        if cid and not spec.container:
            learned.drop("containers", key)  # the remembered one is gone
        return Outcome(False, f"{app.name} couldn't find that {one}", reason="not_found")
    elif status != "OK":
        return Outcome(False, f"{app.name} gave an answer NEO didn't understand", reason="route_bad")
    else:
        oid, name = (rows[0] + ["", ""])[:2] if rows else ("", "")
        if not oid and r.id_code:  # made, but its id didn't come back: find it rather than guess
            oid = await _search_back(r, app, spec, since, cid)
    if name:  # the calendar or list the app put it in, by its own name
        where = f"{name} in {app.name}"
    note = _missed(one, spec.container, tied, name or where) if missing else ""
    ok = await _check(r, app, spec, oid, cid) if oid else None
    if ok is not True:
        return Outcome(
            False,
            f"I added it in {app.name} but couldn't check it. Have a look there.",
            reason="unconfirmed",
            effect=Effect.UNKNOWN,
            say=True,
        )
    if remember:
        learned.put("containers", key, {"id": cid})
        _ASKED.pop(key, None)
    await _note_store_view(spec, app, since)
    receipt = {"id": oid, "where": app.bundle_id, "container": cid, "verified": True}
    return Outcome(True, _added(spec, where, now) + note, effect=Effect.DONE, receipt=receipt,
                   say=True if (note or spec.said_back) else None)


async def _note_store_view(spec: Spec, app: App, since: datetime) -> None:
    """An app whose item showed up in the store writes into it: from now on the store is the way
    there (the engine drops the app while the store can be used)."""
    from neo2.core.decide import learned

    entity = _ENTITY[spec.kind]
    if eventkit.status(entity) != "full" or spec.start is None and entity == "event":
        return
    try:
        if entity == "event":
            assert spec.start is not None
            found = await asyncio.to_thread(eventkit.find_event, spec.title, spec.start)
        else:
            found = await asyncio.to_thread(eventkit.find_reminder, spec.title, since)
    except Exception:  # noqa: BLE001 - learning where things land is a nicety
        return
    if found:
        learned.put("store_views", f"{spec.kind}:{app.bundle_id}", {"view": True})


# ---- reading a calendar by script -------------------------------------------------------------------

_UNTIL_UTC = re.compile(r"UNTIL=(\d{8}T\d{6})Z", re.I)


def _rule(recur: str, start: datetime) -> Any:
    """The repeat rule of an event starting at ``start`` (naive local time, so a weekly 9 AM stays
    9 AM across a daylight-saving change); None when it can't be read."""
    from dateutil.rrule import rrulestr

    text = recur.strip()
    text = text[6:] if text.upper().startswith("RRULE:") else text

    def local(m: re.Match) -> str:
        utc = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        return f"UNTIL={utc.astimezone().replace(tzinfo=None):%Y%m%dT%H%M%S}"

    try:
        return rrulestr(_UNTIL_UTC.sub(local, text), dtstart=start)
    except (ValueError, TypeError, OverflowError):
        return None


def _dates(text: str) -> set[datetime]:
    out = set()
    for part in (text or "").split(","):
        try:
            out.add(datetime.fromisoformat(part.strip()).replace(tzinfo=None))
        except ValueError:
            continue
    return out


def _occurrences(cols: tuple[str, ...], rows: list[list[str]], t0: datetime, t1: datetime, cal: str) -> tuple[list[_Occ], bool]:
    """The window's occurrences of one calendar's events (repeating ones expanded, excluded dates
    left out) and whether any repeat was expanded."""
    out: list[_Occ] = []
    seen: set[tuple[str, datetime]] = set()
    expanded = False
    for row in rows:
        f = dict(zip(cols, row, strict=False))
        try:
            start, end = datetime.fromisoformat(f.get("start", "")), datetime.fromisoformat(f.get("end", ""))
        except ValueError:
            continue
        dur, all_day, uid = max(end - start, timedelta(0)), f.get("all_day") == "true", f.get("id", "")
        starts = [start]
        rule = _rule(f["recur"], start) if f.get("recur", "").strip() else None
        if rule is not None:
            expanded = True
            skip = _dates(f.get("exdates", ""))
            starts = []
            for s in rule.xafter(t0 - dur, count=MAX_OCCURRENCES, inc=True):
                if s >= t1:
                    break
                if s not in skip:
                    starts.append(s)
        for s in starts:
            if s < t1 and s + dur > t0 and (uid, s) not in seen:
                seen.add((uid, s))
                out.append(_Occ(f.get("title", ""), s, s + dur, all_day, cal))
    return out, expanded


async def _script_read(spec: Spec, app: App) -> Outcome:
    from neo2.core.decide import engine

    lr = aescript.list_route(app.script)
    r = aescript.create_route(app.script, "event", ())
    if lr is None or r is None:
        return Outcome(False, f"{app.name} can't list events", reason="route_bad")
    assert spec.start is not None and spec.end is not None
    listed = await _listing(r, app)
    if isinstance(listed, Outcome):
        return listed
    cols = tuple(s for s, _c, _k in lr.cols)
    deadline = time.monotonic() + BUDGET_S["script_read"] - 3.0
    gate = asyncio.Semaphore(READ_AT_ONCE)

    async def one(cid: str, name: str) -> tuple[list[_Occ], bool] | Outcome:
        """This calendar's occurrences and whether a repeat was expanded, else why not."""
        async with gate:
            left = min(READ_S, deadline - time.monotonic())
            if left < 1.0:
                return Outcome(False, f"{app.name} took too long", reason="timeout")
            code, out, err = await aescript.run(*aescript.render_list(lr, app.bundle_id, cid, spec.start, spec.end), left)
        if code != 0:
            return aescript.fail(app.name, code, err, committed=False)
        status, rows = aescript.rows(out)
        return _occurrences(cols, rows, spec.start, spec.end, name) if status == "OK" else ([], False)

    results = await asyncio.gather(*(one(i, n) for i, n, _w in listed))
    fails = {n: x for (_i, n, _w), x in zip(listed, results, strict=True) if isinstance(x, Outcome)}
    if perm := next((f for f in fails.values() if f.reason == "permission"), None):
        return perm
    done = [x for x in results if not isinstance(x, Outcome)]
    if not done and fails:
        return next(iter(fails.values()))
    occ = [o for got, _e in done for o in got]
    extra = ""
    if fails:
        names = list(fails)
        extra += f"\n(I couldn't read {', '.join(names[:-1]) + ' and ' if len(names) > 1 else ''}{names[-1]} in time.)"
    if any(e for _o, e in done) and not engine.proven(spec.kind, app.bundle_id):
        extra += "\n(Moved occurrences of repeating events may be missing.)"
    return _read_outcome(spec, occ, app.bundle_id, extra)


# ---- the calendar-file fallback -----------------------------------------------------------------------


def _ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\r\n", "\n").replace("\n", "\\n")


def _fold(line: str) -> str:
    """RFC 5545 lines are at most 75 octets: longer ones continue on the next, after a space."""
    out, cur = [], b""
    for ch in line:
        b = ch.encode()
        if len(cur) + len(b) > (75 if not out else 74):
            out.append(cur.decode())
            cur = b""
        cur += b
    out.append(cur.decode())
    return "\r\n ".join(out)


def ics_text(spec: Spec, uid: str, stamp: datetime, tzid: str) -> str:
    """The event as an iCalendar file (a VEVENT at local time in ``tzid``; floating without one)."""
    assert spec.start is not None and spec.end is not None
    if spec.date_only:
        when_ = [f"DTSTART;VALUE=DATE:{spec.start:%Y%m%d}", f"DTEND;VALUE=DATE:{spec.end:%Y%m%d}"]
    else:
        tz = f";TZID={tzid}" if tzid else ""
        when_ = [f"DTSTART{tz}:{spec.start:%Y%m%dT%H%M%S}", f"DTEND{tz}:{spec.end:%Y%m%dT%H%M%S}"]
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//NEO//Calendar//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp.astimezone(UTC):%Y%m%dT%H%M%SZ}",
        *when_,
        f"SUMMARY:{_ics_escape(spec.title)}",
    ]
    if spec.location:
        lines.append(f"LOCATION:{_ics_escape(spec.location)}")
    if spec.notes:
        lines.append(f"DESCRIPTION:{_ics_escape(spec.notes)}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(_fold(x) for x in lines) + "\r\n"


def _tzid() -> str:
    try:
        from Foundation import NSTimeZone

        return str(NSTimeZone.localTimeZone().name() or "")
    except Exception:  # noqa: BLE001 - a floating local time still imports at the right hour
        return ""


def _tmp() -> Path:
    from neo2.core.config import settings

    return settings().data_dir / "tmp"


def _drop_files(older_than_s: float | None) -> None:
    """Delete the calendar files NEO wrote (only those older than ``older_than_s``, if given)."""
    for f in _tmp().glob("neo-*.ics"):
        try:
            if older_than_s is None or time.time() - f.stat().st_mtime > older_than_s:
                f.unlink()
        except OSError:
            continue


def _drop_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def prune() -> None:
    """Delete the calendar files handed over more than ICS_KEEP_S ago. For NEO's start: the
    timers of the run that wrote them are gone."""
    _drop_files(ICS_KEEP_S)


def forget() -> None:
    """The privacy erase: every calendar file NEO wrote (an event's title, place and notes), and
    any "which calendar?" still waiting for its answer."""
    _drop_files(None)
    _ASKED.clear()


def _write_ics(spec: Spec) -> str:
    """The event's file in NEO's own tmp folder (0600); files older than ICS_KEEP_S go first."""
    d = _tmp()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    _drop_files(ICS_KEEP_S)
    uid = f"{uuid.uuid4()}@neo.local"
    path = d / f"neo-{uuid.uuid4().hex}.ics"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", newline="") as f:
        f.write(ics_text(spec, uid, datetime.now(UTC), _tzid()))
    return str(path)


async def _open_with(bid: str, path: str) -> bool:
    """Open a file with that app (its import window shows; nothing is added until the user says)."""
    p = await asyncio.create_subprocess_exec(
        "open", "-b", bid, path, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await p.wait() == 0


async def fallback(req: Req, key: str) -> Outcome | None:
    """A new event as a calendar file, opened in the app that imports those: the user presses Add
    there. Only while the store is closed to NEO (it would take the event from NEO itself)."""
    if key != "ics" or req.kind != "add_event" or eventkit.status("event") in ("full", "write_only"):
        return None
    from neo2.core.decide import inventory, run, workspace

    spec = _spec(req)
    bid = await asyncio.to_thread(workspace.default_for_extension, "ics")
    if not bid:
        return Outcome(False, "Nothing on this Mac opens calendar files", reason="not_found")
    path = await asyncio.to_thread(_write_ics, spec)
    if not await run.commit(_open_with(bid, path)):
        _drop_file(path)
        return Outcome(False, "The calendar file didn't open")
    # the app has it open for the user to add; the file itself holds the event's words
    asyncio.get_running_loop().call_later(ICS_KEEP_S, _drop_file, path)
    app = inventory.app_for_bundle(bid)
    name = app.name if app is not None else "your calendar app"
    receipt = {"where": "ics", "handoff": True, "verified": False}
    return Outcome(True, f"It's open in {name}. Press Add there.", effect=Effect.DONE, receipt=receipt, say=True)


# ---- entry points ---------------------------------------------------------------------------------


async def _ask_store(entity: eventkit.Entity) -> None:
    """macOS's one-time question for the store (full access), with a pointer if it lingers."""
    task = asyncio.create_task(eventkit.request(entity))
    done, _ = await asyncio.wait({task}, timeout=ASK_NOTE_S)
    if not done:
        try:
            from neo2.core.events import bus

            await bus().note(f"macOS is asking whether NEO can use your {_STORE[entity]}. Click Allow.")
        except Exception:  # noqa: BLE001 - a note is a nicety
            pass
    await task


async def carry_out(kind: str, args: dict, text: str = "", *, ask_access: bool = False) -> Result:
    """Do one request of a schedule kind (the kind is switched on). ``ask_access``: macOS may be
    asked, once, for the store - only when no app was named for it (the tool's `app`); never in
    shadow or tests. When nothing worked, a store NEO couldn't use says why (the setting the user
    turned off, or the app NEO runs in not declaring it)."""
    from neo2.core.agent.confirm import Halt
    from neo2.core.decide import run

    spec = parse_args(kind, args)
    if isinstance(spec, str):
        return Result(spec, False, say=True)
    entity = _ENTITY[kind]
    req = await _request(spec, text)
    unasked = not eventkit.available(entity) and eventkit.status(entity) == "not_determined"
    if ask_access and unasked and not spec.app:
        await _ask_store(entity)
    res = await run.run_best(_DRIVERS[kind], req)
    if res.ok or res.unsure or isinstance(res.text, Halt):
        return res
    if why := eventkit.available(entity):
        line = f"(The Mac's {_STORE[entity]} store can't be used: {why}.)"
    elif eventkit.status(entity) in ("denied", "restricted"):
        line = _settings_line(entity)
    else:
        return res
    if line not in res.text:
        res.text = f"{res.text.rstrip('.')}. {line}"
    return res


async def would_choose(kind: str, args: dict, text: str = "") -> dict[str, str | int | float | bool]:
    """What :func:`carry_out` would do, as facts for the shadow log: the candidates' keys in
    order, the first one's methods, the store's access, where an add would go (a calendar or list
    id, never a title) and, for a read NEO may already do, how many events it would list. Nothing
    is asked of macOS, opened, scripted or learned."""
    facts: dict[str, str | int | float | bool] = {"kind": kind}
    try:
        spec = parse_args(kind, args)
        facts["parsed"] = not isinstance(spec, str)
        if isinstance(spec, str):
            return facts
        entity = _ENTITY[kind]
        st = "unavailable" if eventkit.available(entity) else eventkit.status(entity)
        facts.update(store=st, inferred=bool(spec.inferred), named_app=bool(spec.app))
        if kind == "add_event":
            facts["all_day"] = spec.date_only
        elif kind == "add_reminder":
            facts["alarm"] = spec.start is not None and not spec.date_only
        from neo2.core.decide import engine

        # its own task: whatever the engine notes for this request stays out of the caller's
        cands = await asyncio.create_task(engine.choose(kind, (await _request(spec, text)).text))
        facts["candidates"] = ",".join(c.key for c in cands)
        first = cands[0] if cands else None
        facts["first"] = first.key if first else ""
        if first is not None:  # the ways this kind would be tried (an app's candidate lists every kind's)
            facts["methods"] = "+".join(m for m in first.methods if first.app is None or m in methods_for(first.app, kind))
        if first is None or first.app is None or first.app.kind != "route" or st != "full":
            return facts
        if kind == "read_calendar":
            assert spec.start is not None and spec.end is not None
            facts["items"] = len(await asyncio.to_thread(eventkit.events_between, spec.start, spec.end))
            return facts
        cals = await asyncio.to_thread(eventkit.calendars, entity)
        default = await asyncio.to_thread(eventkit.default_calendar, entity)
        hit = _match(spec.container, [(c.id, c.title) for c in cals])[0] if spec.container else None
        facts["container"] = hit[0] if hit else (default.id if default else "")
        if spec.container:
            facts["named_found"] = hit is not None
    except Exception as e:  # noqa: BLE001 - a shadow run must never touch the real one
        facts["error"] = type(e).__name__
    return facts


# A time a fast path captured after a lazy title can start with the title's own last word ("put the
# kettle | on at 5", "turn the oven | on tomorrow at 7", "finish | this at 5"): a time opens with
# one of these words at most, bar a few pairs, and never "on/in/at" a day that takes none.
_OPENERS = frozenset({"at", "on", "in", "by", "around", "about", "this", "from", "for", "is"})
_OPEN_PAIRS = frozenset(
    {("at", "around"), ("at", "about"), ("in", "about"), ("in", "around"), ("on", "this"), ("by", "this")}
)
_BARE_DAYS = frozenset({"today", "tonight", "tomorrow", "tmrw", "tmr", "aaj", "kal", "parso", "parson"})


def _cut_title(phrase: str) -> bool:
    w = _norm(phrase).split()
    if len(w) < 2 or w[0] not in _OPENERS:
        return False
    if w[1] in _OPENERS:
        return (w[0], w[1]) not in _OPEN_PAIRS
    return w[0] in ("on", "in", "at", "this") and w[1] in _BARE_DAYS


def check(kind: str, filled: dict) -> bool:
    """A fast path's test, on cached data only: a title that isn't a pronoun, a time
    :mod:`neo2.core.decide.when` reads (an event needs a clock time) that doesn't begin with the title's
    own last word, and a named list NEO has already seen in the store. Captured words count only
    for a kind the user switched on: legacy and shadow run today's scripts, which take ISO times.
    False hands the words to the model."""
    from neo2.core.config import settings

    if any(_text(filled, k) for k in filled) and settings().decide_modes.get(kind) != "on":
        return False
    now = datetime.now()
    title = _norm(_text(filled, "title"))
    if kind in _ITEM and (not title or title in _PRONOUNS):
        return False
    if kind == "add_event":
        got = _time(filled.get("start"), now, future=True)
        timed = got is not None and got.start is not None and not got.date_only
        return timed and not _cut_title(_text(filled, "start"))
    if kind == "add_reminder":
        due = _text(filled, "due")
        if due and (_time(due, now, future=True) is None or _cut_title(due)):
            return False
        if _text(filled, "list"):
            known = eventkit.titles("reminder")
            return bool(known) and _match(_text(filled, "list"), [(t, t) for t in known])[0] is not None
        return True
    if kind == "read_calendar":
        return not _text(filled, "day") or not isinstance(parse_args(kind, {"day": filled["day"]}, now), str)
    return False
