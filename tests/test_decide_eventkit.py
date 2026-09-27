"""neo2.core.decide.eventkit + routes: the Mac's calendar and reminder stores, on a fake EventKit
installed as sys.modules["EventKit"]. No store is opened for real, and macOS is never asked for
access: every request method of the fake fails the test unless a test lets the fake answer it."""

from __future__ import annotations

import itertools
import sys
import threading
import types
from datetime import datetime, timedelta

import pytest
from Foundation import NSDate

from neo2.core.decide import eventkit, evidence, routes

FULL, NOT_ASKED, RESTRICTED, DENIED, WRITE_ONLY = 3, 0, 1, 2, 4
_ids = itertools.count(1)


def ns(dt: datetime):
    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def py(d) -> datetime:
    return datetime.fromtimestamp(d.timeIntervalSince1970())


class Source:
    def __init__(self, sid, kind=0):
        self.sid, self.kind = sid, kind

    def sourceIdentifier(self):
        return self.sid

    def sourceType(self):
        return self.kind


class Cal:
    def __init__(self, cid, title, writable=True, source="local"):
        self.cid, self.t, self.w, self.src = cid, title, writable, Source(source)

    def calendarIdentifier(self):
        return self.cid

    def title(self):
        return self.t

    def allowsContentModifications(self):
        return self.w

    def source(self):
        return self.src


class Attendee:
    def __init__(self, me, status):
        self.me, self.st = me, status

    def isCurrentUser(self):
        return self.me

    def participantStatus(self):
        return self.st


class Item:
    """An event or a reminder: setters record, getters answer."""

    def __init__(self, store, **kw):
        self.store, self.v, self.alarms_ = store, {"allDay": False, "attendees": [], **kw}, []
        self.ident = f"id{next(_ids)}"

    def __getattr__(self, name):
        if name.startswith("set") and name.endswith("_"):
            key = name[3].lower() + name[4:-1]
            return lambda value: self.v.__setitem__(key, value)
        raise AttributeError(name)

    def title(self):
        return self.v.get("title")

    def startDate(self):
        return self.v.get("startDate")

    def endDate(self):
        return self.v.get("endDate")

    def isAllDay(self):
        return self.v["allDay"]

    def calendar(self):
        return self.v.get("calendar")

    def attendees(self):
        return self.v["attendees"]

    def eventIdentifier(self):
        return self.ident

    def calendarItemIdentifier(self):
        return self.ident

    def creationDate(self):
        return self.v.get("created")

    def addAlarm_(self, alarm):
        self.alarms_.append(alarm)

    def alarms(self):
        return self.alarms_


class FakeEK:
    """The EventKit module: one store class with a status knob per entity."""

    EKSpanThisEvent = 0

    def __init__(self):
        self.status = {0: FULL, 1: FULL}
        self.created = 0
        self.cals = {0: [Cal("home", "Home"), Cal("work", "Work"), Cal("holidays", "Holidays", writable=False, source="sub")],
                     1: [Cal("inbox", "Inbox"), Cal("shopping", "Shopping")]}
        self.default = {0: "work", 1: "inbox"}
        self.events: list[Item] = []
        self.reminders: list[Item] = []
        self.saved: list[Item] = []
        self.queries: list[tuple] = []
        fake = self

        class Store:
            @staticmethod
            def authorizationStatusForEntityType_(entity):
                return fake.status[entity]

            @classmethod
            def alloc(cls):
                fake.created += 1
                return cls()

            def init(self):
                return self

            def __getattr__(self, name):
                if name.startswith("request"):
                    pytest.fail(f"asked for access: {name}")
                raise AttributeError(name)

            def calendarsForEntityType_(self, e):
                return fake.cals[e]

            def defaultCalendarForNewEvents(self):
                return self.calendarWithIdentifier_(fake.default[0])

            def defaultCalendarForNewReminders(self):
                return self.calendarWithIdentifier_(fake.default[1])

            def calendarWithIdentifier_(self, cid):
                return next((c for cs in fake.cals.values() for c in cs if c.cid == cid), None)

            def saveEvent_span_commit_error_(self, ev, span, commit, err):
                assert span == 0 and commit is True
                fake.saved.append(ev)
                fake.events.append(ev)
                return True, None

            def saveReminder_commit_error_(self, r, commit, err):
                fake.saved.append(r)
                fake.reminders.append(r)
                return True, None

            def eventWithIdentifier_(self, i):
                return next((e for e in fake.events if e.ident == i), None)

            def calendarItemWithIdentifier_(self, i):
                return next((e for e in fake.events + fake.reminders if e.ident == i), None)

            def predicateForEventsWithStartDate_endDate_calendars_(self, s, e, cals):
                return ("events", py(s), py(e))

            def eventsMatchingPredicate_(self, pred):
                fake.queries.append(pred)
                _k, s, e = pred
                return [ev for ev in fake.events if py(ev.startDate()) < e and py(ev.endDate()) > s]

            def predicateForRemindersInCalendars_(self, cals):
                return ("reminders",)

            def fetchRemindersMatchingPredicate_completion_(self, pred, handler):
                threading.Thread(target=handler, args=(list(fake.reminders),)).start()

        class Event:
            @staticmethod
            def eventWithEventStore_(store):
                return Item(store)

        class Reminder:
            @staticmethod
            def reminderWithEventStore_(store):
                return Item(store)

        class Alarm:
            @staticmethod
            def alarmWithAbsoluteDate_(d):
                return ("alarm", py(d))

        self.module = types.SimpleNamespace(
            EKEventStore=Store, EKEvent=Event, EKReminder=Reminder, EKAlarm=Alarm, EKSpanThisEvent=0
        )

    def add_event(self, title, start, hours=1.0, cal="home", **kw):
        ev = Item(None, title=title, startDate=ns(start), endDate=ns(start + timedelta(hours=hours)),
                  calendar=next(c for c in self.cals[0] if c.cid == cal), **kw)
        self.events.append(ev)
        return ev


USAGE = frozenset({"NSCalendarsFullAccessUsageDescription", "NSRemindersFullAccessUsageDescription"})


@pytest.fixture
def ek(monkeypatch):
    fake = FakeEK()
    monkeypatch.setitem(sys.modules, "EventKit", fake.module)
    monkeypatch.setattr(evidence, "responsible_usage", lambda: USAGE)
    monkeypatch.setattr(eventkit, "_store_obj", None)
    eventkit.forget()
    yield fake
    eventkit.forget()


def test_status_is_read_never_asked(ek):
    assert eventkit.status("event") == "full"
    for code, name in ((NOT_ASKED, "not_determined"), (RESTRICTED, "restricted"), (DENIED, "denied"), (WRITE_ONLY, "write_only")):
        ek.status[1] = code
        assert eventkit.status("reminder") == name
    assert ek.created == 0  # reading the status opens no store


def test_available_needs_the_framework_and_the_usage_key(ek, monkeypatch):
    assert eventkit.available("event") == "" and eventkit.usage_declared("reminder")
    monkeypatch.setattr(evidence, "responsible_usage", lambda: frozenset({"NSCalendarsUsageDescription"}))
    assert eventkit.available("event") == ""  # the older key counts too
    assert eventkit.available("reminder") == "NEO's app doesn't declare reminders access"
    monkeypatch.setitem(sys.modules, "EventKit", None)
    assert eventkit.available("event") == "EventKit isn't installed" and eventkit.status("event") == "denied"


def test_nothing_is_read_or_written_without_access(ek):
    ek.status = {0: NOT_ASKED, 1: DENIED}
    with pytest.raises(eventkit.NoAccess):
        eventkit.save_event(eventkit.EventSpec("Lunch", datetime(2026, 10, 1, 13)))
    with pytest.raises(eventkit.NoAccess):
        eventkit.calendars("reminder")
    assert eventkit.fetch("id1") is None and ek.created == 0
    ek.status[0] = WRITE_ONLY  # add-only access: may write, may not read
    with pytest.raises(eventkit.NoAccess):
        eventkit.events_between(datetime(2026, 10, 1), datetime(2026, 10, 2))


def test_an_event_goes_to_the_default_calendar_and_is_read_back(ek):
    r = eventkit.save_event(eventkit.EventSpec("Lunch with Sam", datetime(2026, 10, 1, 13), location="Cafe", notes="bring book"))
    ev = ek.saved[-1]
    assert r.calendar_id == "work" and r.verified and r.id == ev.ident
    assert (py(ev.v["startDate"]), py(ev.v["endDate"])) == (datetime(2026, 10, 1, 13), datetime(2026, 10, 1, 14))
    assert ev.v["location"] == "Cafe" and ev.v["notes"] == "bring book" and ev.v["allDay"] is False
    named = eventkit.save_event(eventkit.EventSpec("Standup", datetime(2026, 10, 2, 9), calendar_id="home"))
    assert named.calendar_id == "home"
    readonly = eventkit.save_event(eventkit.EventSpec("x", datetime(2026, 10, 2, 9), calendar_id="holidays"))
    assert readonly.calendar_id == "work"  # a calendar NEO can't add to: the default instead
    day = eventkit.save_event(eventkit.EventSpec("Trip", datetime(2026, 10, 3), all_day=True))
    assert ek.saved[-1].v["allDay"] is True and day.verified


def test_write_only_access_adds_but_cannot_check(ek):
    ek.status[0] = WRITE_ONLY
    r = eventkit.save_event(eventkit.EventSpec("Lunch", datetime(2026, 10, 1, 13)))
    assert r.id and not r.verified


def test_a_timed_reminder_gets_an_alert_and_a_day_does_not(ek):
    due = datetime(2026, 10, 1, 17, 30)
    r = eventkit.save_reminder(eventkit.ReminderSpec("call mom", due=due))
    item = ek.saved[-1]
    comps = item.v["dueDateComponents"]
    assert (comps.year(), comps.month(), comps.day(), comps.hour(), comps.minute()) == (2026, 10, 1, 17, 30)
    assert item.alarms() == [("alarm", due)] and r.calendar_id == "inbox" and r.verified
    assert "priority" not in item.v  # only when the user asked
    eventkit.save_reminder(eventkit.ReminderSpec("pay rent", due=datetime(2026, 10, 3), date_only=True, list_id="shopping", priority=1))
    item = ek.saved[-1]
    assert item.alarms() == [] and item.v["calendar"].cid == "shopping" and item.v["priority"] == 1
    from Foundation import NSDateComponentUndefined

    assert item.v["dueDateComponents"].hour() == NSDateComponentUndefined
    eventkit.save_reminder(eventkit.ReminderSpec("someday"))
    assert "dueDateComponents" not in ek.saved[-1].v


def test_fetch_reads_back_what_was_saved(ek):
    r = eventkit.save_event(eventkit.EventSpec("Lunch", datetime(2026, 10, 1, 13)))
    got = eventkit.fetch(r.id)
    assert got["entity"] == "event" and got["title"] == "Lunch" and got["start"] == datetime(2026, 10, 1, 13)
    rr = eventkit.save_reminder(eventkit.ReminderSpec("call mom", due=datetime(2026, 10, 1, 17)))
    assert eventkit.fetch(rr.id)["alarms"] == 1 and eventkit.fetch("nope") is None


def test_the_days_events_repeats_expanded_declined_skipped(ek):
    day = datetime(2026, 10, 1)
    ek.add_event("Standup", day.replace(hour=9), 0.25)  # an occurrence of a repeating event
    ek.add_event("Offsite", day, 24, allDay=True)
    ek.add_event("Declined sync", day.replace(hour=11), attendees=[Attendee(True, 3), Attendee(False, 2)])
    ek.add_event("Accepted sync", day.replace(hour=10), attendees=[Attendee(True, 2)])
    ek.add_event("Tomorrow", day + timedelta(days=1, hours=9))
    rows = eventkit.events_between(day, day + timedelta(days=1))
    assert [r.title for r in rows] == ["Offsite", "Standup", "Accepted sync"]
    assert rows[1].calendar == "Home" and rows[0].all_day
    eventkit.events_between(day, day + timedelta(days=400))
    assert ek.queries[-1][2] == day + timedelta(days=eventkit.MAX_RANGE_DAYS)  # capped
    assert eventkit.find_event("  standup ", day.replace(hour=9, second=30)) == rows[1].id
    assert eventkit.find_event("Standup", day.replace(hour=12)) is None


def test_counts_are_numbers_only_and_cached(ek):
    now = datetime.now()
    # noon two days ago, and two hours before that: the same calendar day at any time of night
    day = now.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=2)
    ek.add_event("a", day)
    ek.add_event("b", day - timedelta(hours=2))
    ek.add_event("c", now - timedelta(days=9))
    ek.add_event("old", now - timedelta(days=45))
    c = eventkit.counts("event")
    assert (c.writable, c.sources, c.days_30) == (2, 2, 2)
    assert c.last == pytest.approx(day.timestamp(), abs=2)
    ek.add_event("d", now - timedelta(days=1))
    assert eventkit.counts("event").days_30 == 2  # cached
    eventkit.forget()
    assert eventkit.counts("event").days_30 == 3
    r = Item(None, title="x", created=ns(now - timedelta(days=3)))
    ek.reminders += [r, Item(None, title="y", created=ns(now - timedelta(days=60)))]
    rc = eventkit.counts("reminder")
    assert rc.days_30 == 1 and rc.writable == 2


# ---- routes ----------------------------------------------------------------------------------------


def test_a_store_route_is_a_candidate_only_when_usable(ek, monkeypatch):
    assert routes.route_app("add_event", "eventkit.nothing") is None
    ek.status[0] = DENIED
    assert routes.route_app("add_event", "eventkit.events") is None
    ek.status[0] = RESTRICTED
    assert routes.route_app("add_event", "eventkit.events") is None
    ek.status[0] = FULL
    monkeypatch.setattr(evidence, "responsible_usage", lambda: frozenset())
    assert routes.route_app("add_event", "eventkit.events") is None  # it could never be asked for


def test_before_access_the_route_says_so_and_its_use_is_unknown(ek):
    ek.status[1] = NOT_ASKED
    app = routes.route_app("add_reminder", "eventkit.reminders")
    assert app.kind == "route" and app.bundle_id == "route:eventkit.reminders" and app.path == ""
    assert app.note.endswith("access not asked yet") and app.days_used_30 == 0 and app.last_used is None
    assert routes.pending(app) and ek.created == 0
    ek.status[1] = WRITE_ONLY
    assert routes.pending(routes.route_app("add_reminder", "eventkit.reminders"))


def test_with_access_the_route_counts_the_users_own_use(ek):
    now = datetime.now()
    for d in (1, 1, 4, 20):
        ek.add_event("x", now - timedelta(days=d))
    app = routes.route_app("add_event", "eventkit.events")
    assert app.days_used_30 == 3 and app.last_used == pytest.approx((now - timedelta(days=1)).timestamp(), abs=2)
    assert "2 writable calendars" in app.note and "events on 3 of the last 30 days" in app.note
    assert "Home" not in app.note and "Work" not in app.note  # counts only, never titles
    assert not routes.pending(app)


# ---- asking (a kind switched on only), cached titles, finding a reminder ------------------------------------


@pytest.fixture
def answers(ek, monkeypatch):
    """Let the fake store answer access requests (the real one is never there): full access on
    macOS 14+, the older one-for-all request before it."""
    asked: list[tuple[str, int]] = []
    store = ek.module.EKEventStore
    modern = {"on": True}

    def grant(how, entity, handler):
        asked.append((how, entity))
        ek.status[entity] = FULL
        threading.Thread(target=handler, args=(True, None)).start()

    monkeypatch.setattr(store, "respondsToSelector_", lambda self, sel: modern["on"], raising=False)
    monkeypatch.setattr(store, "requestFullAccessToEventsWithCompletion_", lambda self, h: grant("full", 0, h), raising=False)
    monkeypatch.setattr(store, "requestFullAccessToRemindersWithCompletion_", lambda self, h: grant("full", 1, h), raising=False)
    monkeypatch.setattr(store, "requestAccessToEntityType_completion_", lambda self, e, h: grant("old", e, h), raising=False)
    return asked, modern


def test_request_asks_once_for_full_access_and_only_with_the_usage_key(ek, answers, monkeypatch):
    import asyncio

    asked, modern = answers
    ek.status = {0: NOT_ASKED, 1: NOT_ASKED}
    assert asyncio.run(eventkit.request("event")) == "full" and asked == [("full", 0)]
    assert asyncio.run(eventkit.request("event")) == "full" and asked == [("full", 0)]  # decided: not again
    modern["on"] = False
    assert asyncio.run(eventkit.request("reminder")) == "full" and asked[-1] == ("old", 1)
    ek.status[0] = DENIED
    assert asyncio.run(eventkit.request("event")) == "denied" and len(asked) == 2
    ek.status[0] = NOT_ASKED
    monkeypatch.setattr(evidence, "responsible_usage", lambda: frozenset())
    assert asyncio.run(eventkit.request("event")) == "not_determined" and len(asked) == 2


def test_titles_are_cached_from_reads_only_and_erased(ek):
    assert eventkit.titles("reminder") is None
    eventkit.calendars("reminder")
    assert eventkit.titles("reminder") == ("Inbox", "Shopping")
    eventkit.counts("event")
    assert eventkit.titles("event") == ("Home", "Work")  # writable ones only
    eventkit.forget()
    assert eventkit.titles("event") is None and eventkit.titles("reminder") is None


def test_a_reminder_an_app_just_added_is_found_by_title(ek):
    now = datetime.now()
    old = Item(None, title="call mom", created=ns(now - timedelta(days=3)))
    new = Item(None, title="Call  Mom", created=ns(now))
    ek.reminders += [old, new]
    assert eventkit.find_reminder("call mom", now - timedelta(seconds=5)) == new.ident
    assert eventkit.find_reminder("call dad", now - timedelta(seconds=5)) is None
