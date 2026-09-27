"""The Mac's shared calendar and reminder stores (EventKit): every calendar app here shows them.

NEO saves and reads through the store itself, so an event lands in the user's default calendar
whichever app they look at it in, and recurring events come back expanded. :func:`status` only
reads what the user already decided, and every read or write below refuses to run unless access
was granted (so none of them can put up macOS's question). Only :func:`request` asks, once, and
only when NEO's app declares why it wants the store. Calendar and list titles are never logged.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from neo import effects
from neo.decide._objc import pooled

__all__ = [
    "Entity",
    "Status",
    "Cal",
    "EventSpec",
    "ReminderSpec",
    "Receipt",
    "EventRow",
    "Counts",
    "available",
    "usage_declared",
    "status",
    "request",
    "calendars",
    "titles",
    "default_calendar",
    "save_event",
    "save_reminder",
    "fetch",
    "events_between",
    "find_event",
    "find_reminder",
    "counts",
    "forget",
]

Entity = Literal["event", "reminder"]
Status = Literal["not_determined", "restricted", "denied", "full", "write_only"]

# EKAuthorizationStatus, EKEntityType, EKParticipantStatus (stable framework values)
_STATUS: dict[int, Status] = {0: "not_determined", 1: "restricted", 2: "denied", 3: "full", 4: "write_only"}
_ENTITY = {"event": 0, "reminder": 1}
_DECLINED = 3
# Usage keys the responsible app must declare before access could ever be requested.
_USAGE = {
    "event": ("NSCalendarsFullAccessUsageDescription", "NSCalendarsUsageDescription"),
    "reminder": ("NSRemindersFullAccessUsageDescription", "NSRemindersUsageDescription"),
}
_NOUN = {"event": "calendar", "reminder": "reminders"}
COUNTS_TTL_S = 600.0
MAX_RANGE_DAYS = 31
ASK_S = 60.0  # how long NEO waits for the user to answer macOS's question


class NoAccess(PermissionError):
    """NEO hasn't been allowed to use this store (it never asks from here)."""


@dataclass(frozen=True)
class Cal:
    id: str
    title: str
    source_type: int = 0
    writable: bool = True


@dataclass(frozen=True)
class EventSpec:
    title: str
    start: datetime
    end: datetime | None = None  # start + 1 h
    all_day: bool = False
    location: str = ""
    notes: str = ""
    calendar_id: str = ""  # '' = the user's default calendar for new events


@dataclass(frozen=True)
class ReminderSpec:
    title: str
    due: datetime | None = None
    date_only: bool = False  # a day with no time: no alert
    notes: str = ""
    list_id: str = ""  # '' = the default list
    priority: int = 0  # 0 = none (set only when the user asked)


@dataclass(frozen=True)
class Receipt:
    id: str
    calendar_id: str
    verified: bool  # read back from the store (impossible with write-only access)


@dataclass(frozen=True)
class EventRow:
    id: str
    title: str
    start: datetime
    end: datetime
    all_day: bool
    calendar: str


@dataclass(frozen=True)
class Counts:
    sources: int  # accounts holding calendars / lists
    writable: int  # calendars / lists NEO could add to
    days_30: int  # days in the last 30 with an event (or a reminder created)
    last: float | None  # epoch of the latest such item


def _ek() -> Any:
    import EventKit

    return EventKit


def available(entity: Entity = "event") -> str:
    """'' when this store could be used at all, else why not (checked before any request)."""
    try:
        _ek()
    except ImportError:
        return "EventKit isn't installed"
    if not usage_declared(entity):
        return f"NEO's app doesn't declare {_NOUN[entity]} access"
    return ""


def usage_declared(entity: Entity) -> bool:
    from neo.decide import evidence

    return bool(evidence.responsible_usage().intersection(_USAGE[entity]))


@pooled
def status(entity: Entity) -> Status:
    """What the user decided about NEO and this store (never asks)."""
    try:
        ek = _ek()
        return _STATUS.get(int(ek.EKEventStore.authorizationStatusForEntityType_(_ENTITY[entity])), "denied")
    except Exception:  # noqa: BLE001 - no EventKit: as good as denied
        return "denied"


_store_obj: Any = None
_store_lock = threading.Lock()


def _the_store() -> Any:
    global _store_obj
    with _store_lock:
        if _store_obj is None:
            cls = _ek().EKEventStore
            effects.guard("calendar", "open the user's Calendar and Reminders store", via=cls)
            _store_obj = cls.alloc().init()
        return _store_obj


def _store(entity: Entity, *, write: bool = False) -> Any:
    """The process's one store, only once access to ``entity`` was granted (full, or write-only
    for a write)."""
    st = status(entity)
    if st != "full" and not (write and st == "write_only"):
        raise NoAccess(f"no {_NOUN[entity]} access ({st})")
    return _the_store()


async def request(entity: Entity) -> Status:
    """Ask the user (macOS's own question) for full access to this store, and wait up to ASK_S
    for the answer; the status afterwards. Nothing is asked when NEO's app doesn't declare the
    usage (asking without it may end the process) or when the user already decided: then this is
    just :func:`status`. Only ever called for a kind the user switched on."""
    if available(entity) or status(entity) != "not_determined":
        return status(entity)
    return await asyncio.to_thread(_ask, entity)


@pooled
def _ask(entity: Entity) -> Status:
    s = _the_store()
    done = threading.Event()

    def answered(_granted: Any, _error: Any = None) -> None:
        done.set()

    full = "requestFullAccessToEventsWithCompletion:" if entity == "event" else "requestFullAccessToRemindersWithCompletion:"
    if s.respondsToSelector_(full):  # macOS 14+: full access only (a write-only grant is the user's choice)
        getattr(s, full.replace(":", "_"))(answered)
    else:
        s.requestAccessToEntityType_completion_(_ENTITY[entity], answered)
    done.wait(ASK_S)
    _counts.pop(entity, None)
    return status(entity)


def _nsdate(dt: datetime) -> Any:
    from Foundation import NSDate

    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def _py(d: Any) -> datetime | None:
    return datetime.fromtimestamp(float(d.timeIntervalSince1970())) if d is not None else None


def _cal(c: Any) -> Cal:
    src = c.source()
    return Cal(
        id=str(c.calendarIdentifier()),
        title=str(c.title() or ""),
        source_type=int(src.sourceType()) if src is not None else 0,
        writable=bool(c.allowsContentModifications()),
    )


_titles: dict[str, tuple[str, ...]] = {}  # entity -> writable calendar / list titles (memory only)


@pooled
def calendars(entity: Entity, *, writable: bool = True) -> list[Cal]:
    """The user's calendars (or reminder lists), only those NEO could add to by default."""
    out = [_cal(c) for c in _store(entity).calendarsForEntityType_(_ENTITY[entity]) or []]
    _titles[entity] = tuple(c.title for c in out if c.writable)
    return [c for c in out if c.writable or not writable]


def titles(entity: Entity) -> tuple[str, ...] | None:
    """The writable calendars' (or lists') titles as last read from the store, never reading it:
    for checks that must answer at once. None until NEO has read them."""
    return _titles.get(entity)


@pooled
def default_calendar(entity: Entity) -> Cal | None:
    s = _store(entity)
    c = s.defaultCalendarForNewEvents() if entity == "event" else s.defaultCalendarForNewReminders()
    return _cal(c) if c is not None else None


def _target(s: Any, entity: Entity, cal_id: str) -> Any:
    if cal_id:
        c = s.calendarWithIdentifier_(cal_id)
        if c is not None and c.allowsContentModifications():
            return c
    return s.defaultCalendarForNewEvents() if entity == "event" else s.defaultCalendarForNewReminders()


@pooled
def save_event(spec: EventSpec) -> Receipt:
    """Add one event (this occurrence only; no invitees) and read it back when access allows."""
    ek = _ek()
    s = _store("event", write=True)
    ev = ek.EKEvent.eventWithEventStore_(s)
    ev.setTitle_(spec.title)
    ev.setStartDate_(_nsdate(spec.start))
    ev.setEndDate_(_nsdate(spec.end or spec.start + timedelta(hours=1)))
    ev.setAllDay_(spec.all_day)
    if spec.location:
        ev.setLocation_(spec.location)
    if spec.notes:
        ev.setNotes_(spec.notes)
    cal = _target(s, "event", spec.calendar_id)
    ev.setCalendar_(cal)
    ok, err = s.saveEvent_span_commit_error_(ev, ek.EKSpanThisEvent, True, None)
    if not ok:
        raise RuntimeError(f"the calendar store didn't save it ({err})")
    eid = str(ev.eventIdentifier() or "")
    verified = False
    if status("event") == "full" and eid:
        try:
            got = s.eventWithIdentifier_(eid)
            verified = got is not None and str(got.title()) == spec.title and _py(got.startDate()) == _py(ev.startDate())
        except Exception as e:  # noqa: BLE001 - it is saved; only the check failed: unverified
            print(f"[eventkit] reading the new event back failed ({type(e).__name__})")
    return Receipt(eid, str(cal.calendarIdentifier()) if cal is not None else "", verified)


def _components(d: datetime, timed: bool) -> Any:
    from Foundation import NSDateComponents, NSTimeZone

    c = NSDateComponents.alloc().init()
    c.setYear_(d.year)
    c.setMonth_(d.month)
    c.setDay_(d.day)
    if timed:
        c.setHour_(d.hour)
        c.setMinute_(d.minute)
        c.setTimeZone_(NSTimeZone.localTimeZone())
    return c


@pooled
def save_reminder(spec: ReminderSpec) -> Receipt:
    """Add one reminder; one with a time gets an alert at that time (a day alone gets none)."""
    ek = _ek()
    s = _store("reminder", write=True)
    r = ek.EKReminder.reminderWithEventStore_(s)
    r.setTitle_(spec.title)
    if spec.notes:
        r.setNotes_(spec.notes)
    if spec.priority:
        r.setPriority_(spec.priority)
    lst = _target(s, "reminder", spec.list_id)
    r.setCalendar_(lst)
    if spec.due is not None:
        r.setDueDateComponents_(_components(spec.due, not spec.date_only))
        if not spec.date_only:
            r.addAlarm_(ek.EKAlarm.alarmWithAbsoluteDate_(_nsdate(spec.due)))
    ok, err = s.saveReminder_commit_error_(r, True, None)
    if not ok:
        raise RuntimeError(f"the reminders store didn't save it ({err})")
    rid = str(r.calendarItemIdentifier() or "")
    verified = False
    if status("reminder") == "full" and rid:
        try:
            got = s.calendarItemWithIdentifier_(rid)
            verified = got is not None and str(got.title()) == spec.title
        except Exception as e:  # noqa: BLE001 - it is saved; only the check failed: unverified
            print(f"[eventkit] reading the new reminder back failed ({type(e).__name__})")
    return Receipt(rid, str(lst.calendarIdentifier()) if lst is not None else "", verified)


@pooled
def fetch(identifier: str) -> dict | None:
    """An event or reminder NEO saved, read back from the store (None if it isn't there)."""
    for entity in ("event", "reminder"):
        try:
            s = _store(entity)
        except NoAccess:
            continue
        item = s.eventWithIdentifier_(identifier) if entity == "event" else s.calendarItemWithIdentifier_(identifier)
        if item is None:
            continue
        out = {"id": identifier, "entity": entity, "title": str(item.title() or "")}
        cal = item.calendar()
        out["calendar_id"] = str(cal.calendarIdentifier()) if cal is not None else ""
        if entity == "event":
            out["start"], out["end"] = _py(item.startDate()), _py(item.endDate())
        else:
            out["alarms"] = len(item.alarms() or [])
        return out
    return None


def _declined(ev: Any) -> bool:
    return any(a.isCurrentUser() and int(a.participantStatus()) == _DECLINED for a in ev.attendees() or [])


@pooled
def events_between(start: datetime, end: datetime) -> list[EventRow]:
    """Every occurrence overlapping [start, end) on every calendar (repeats expanded), minus the
    ones the user declined, by start time. The range is capped at MAX_RANGE_DAYS."""
    s = _store("event")
    end = min(end, start + timedelta(days=MAX_RANGE_DAYS))
    pred = s.predicateForEventsWithStartDate_endDate_calendars_(_nsdate(start), _nsdate(end), None)
    rows = []
    for ev in s.eventsMatchingPredicate_(pred) or []:
        if _declined(ev):
            continue
        cal = ev.calendar()
        rows.append(
            EventRow(
                id=str(ev.eventIdentifier() or ""),
                title=str(ev.title() or ""),
                start=_py(ev.startDate()),
                end=_py(ev.endDate()),
                all_day=bool(ev.isAllDay()),
                calendar=str(cal.title() or "") if cal is not None else "",
            )
        )
    return sorted(rows, key=lambda r: (not r.all_day, r.start))


def find_event(title: str, start: datetime, tol_s: float = 60) -> str | None:
    """The id of an event with this title starting within ``tol_s`` of ``start`` (after an
    import NEO couldn't watch)."""
    want = " ".join(title.split()).casefold()
    tol = timedelta(seconds=tol_s)
    for r in events_between(start - tol, start + tol + timedelta(minutes=1)):
        if " ".join(r.title.split()).casefold() == want and abs(r.start - start) <= tol:
            return r.id
    return None


@pooled
def find_reminder(title: str, since: datetime) -> str | None:
    """The id of a reminder with this title created since ``since`` (one an app just added)."""
    want = " ".join(title.split()).casefold()
    hit = next((i for c, t, i in _reminders_created(_store("reminder"), since) if t == want), None)
    return hit or None


_counts: dict[str, tuple[float, Counts]] = {}


def _reminders_created(s: Any, since: datetime) -> list[tuple[datetime, str, str]]:
    """(created, folded title, id) of reminders made since ``since`` (EventKit answers on its
    own queue)."""
    done, got = threading.Event(), []

    def handler(items: Any) -> None:
        for r in items or []:
            if (c := _py(r.creationDate())) is not None and c >= since:
                got.append((c, " ".join(str(r.title() or "").split()).casefold(), str(r.calendarItemIdentifier() or "")))
        done.set()

    s.fetchRemindersMatchingPredicate_completion_(s.predicateForRemindersInCalendars_(None), handler)
    done.wait(5.0)
    return list(got)


@pooled
def counts(entity: Entity) -> Counts:
    """How much the user keeps in the store (cached for COUNTS_TTL_S): numbers only."""
    hit = _counts.get(entity)
    if hit and time.monotonic() - hit[0] < COUNTS_TTL_S:
        return hit[1]
    s = _store(entity)
    cals = list(s.calendarsForEntityType_(_ENTITY[entity]) or [])
    now = datetime.now()
    since = now - timedelta(days=30)
    if entity == "event":
        stamps = [r.start for r in events_between(since, now)]
    else:
        stamps = [c for c, _t, _i in _reminders_created(s, since)]
    days = {d.date() for d in stamps}
    _titles[entity] = tuple(str(c.title() or "") for c in cals if c.allowsContentModifications())
    c = Counts(
        sources=len({str(c.source().sourceIdentifier()) for c in cals if c.source() is not None}),
        writable=sum(1 for c in cals if c.allowsContentModifications()),
        days_30=len(days),
        last=max((d.timestamp() for d in stamps), default=None),
    )
    _counts[entity] = (time.monotonic(), c)
    return c


def forget() -> None:
    """The privacy erase: the cached counts and titles go (nothing else is kept)."""
    _counts.clear()
    _titles.clear()
