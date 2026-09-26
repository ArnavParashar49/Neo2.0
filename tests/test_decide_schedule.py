"""neo.decide.kinds.schedule: events and reminders added, the calendar read - through the Mac's
store (a fake EventKit installed as sys.modules["EventKit"]), through an app's dictionary
(osascript answered by a fake app that reads only the generated scripts' property codes), or as
a calendar file (never opened). macOS is never asked anything: the fake store's access requests
fail the test unless a test lets the fake answer them."""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import sys
import threading
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from Foundation import NSDate

from neo import privacy
from neo.agent.confirm import Halt
from neo.config import settings
from neo.decide import aescript, engine, eventkit, evidence, inventory, learned, media, run, sdef, workspace
from neo.decide.inventory import App
from neo.decide.kinds import schedule

FULL, NOT_ASKED, RESTRICTED, DENIED, WRITE_ONLY = 3, 0, 1, 2, 4
FIX = Path(__file__).parent / "fixtures" / "decide" / "sdef"
DAY = 86400.0
US, RS = aescript.US, aescript.RS
USAGE = frozenset({"NSCalendarsFullAccessUsageDescription", "NSRemindersFullAccessUsageDescription"})
_ids = itertools.count(1)


def ns(dt: datetime):
    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def py(d) -> datetime:
    return datetime.fromtimestamp(d.timeIntervalSince1970())


def at(days: int, hour: int, minute: int = 0) -> datetime:
    return (datetime.now() + timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)


# ---- a fake EventKit ------------------------------------------------------------------------------


class Source:
    def __init__(self, sid):
        self.sid = sid

    def sourceIdentifier(self):
        return self.sid

    def sourceType(self):
        return 0


class Cal:
    def __init__(self, cid, title, writable=True):
        self.cid, self.t, self.w, self.src = cid, title, writable, Source("local")

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

    def __init__(self, **kw):
        self.v, self.alarms_ = {"allDay": False, "attendees": [], "created": ns(datetime.now()), **kw}, []
        self.ident = f"ek{next(_ids)}"

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
    """EventKit: a status knob per entity, calendars and lists with defaults, and access requests
    that fail the test unless ``answer`` says how the user answers."""

    def __init__(self):
        self.status = {0: FULL, 1: FULL}
        self.answer: int | None = None
        self.asks: list[int] = []
        self.cals = {0: [Cal("home", "Home"), Cal("work", "Work"), Cal("holidays", "Holidays", writable=False)],
                     1: [Cal("inbox", "Inbox"), Cal("errands", "Errands")]}  # fmt: skip
        self.default = {0: "work", 1: "inbox"}
        self.events: list[Item] = []
        self.reminders: list[Item] = []
        self.saved: list[Item] = []
        self.on_save = None
        fake = self

        def ask(entity, handler):
            fake.asks.append(entity)
            if fake.answer is None:
                pytest.fail("macOS would have been asked for access")
            fake.status[entity] = fake.answer
            threading.Thread(target=handler, args=(fake.answer == FULL, None)).start()

        class Store:
            @staticmethod
            def authorizationStatusForEntityType_(entity):
                return fake.status[entity]

            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def respondsToSelector_(self, sel):
                return True

            def requestFullAccessToEventsWithCompletion_(self, handler):
                ask(0, handler)

            def requestFullAccessToRemindersWithCompletion_(self, handler):
                ask(1, handler)

            def calendarsForEntityType_(self, e):
                return fake.cals[e]

            def defaultCalendarForNewEvents(self):
                return self.calendarWithIdentifier_(fake.default[0])

            def defaultCalendarForNewReminders(self):
                return self.calendarWithIdentifier_(fake.default[1])

            def calendarWithIdentifier_(self, cid):
                return next((c for cs in fake.cals.values() for c in cs if c.cid == cid), None)

            def saveEvent_span_commit_error_(self, ev, span, commit, err):
                if fake.on_save:
                    fake.on_save()
                fake.saved.append(ev)
                fake.events.append(ev)
                return True, None

            def saveReminder_commit_error_(self, r, commit, err):
                if fake.on_save:
                    fake.on_save()
                fake.saved.append(r)
                fake.reminders.append(r)
                return True, None

            def eventWithIdentifier_(self, i):
                return next((e for e in fake.events if e.ident == i), None)

            def calendarItemWithIdentifier_(self, i):
                return next((e for e in fake.events + fake.reminders if e.ident == i), None)

            def predicateForEventsWithStartDate_endDate_calendars_(self, s, e, cals):
                return (py(s), py(e))

            def eventsMatchingPredicate_(self, pred):
                s, e = pred
                return [ev for ev in fake.events if py(ev.startDate()) < e and py(ev.endDate()) > s]

            def predicateForRemindersInCalendars_(self, cals):
                return ("reminders",)

            def fetchRemindersMatchingPredicate_completion_(self, pred, handler):
                threading.Thread(target=handler, args=(list(fake.reminders),)).start()

        class Alarm:
            @staticmethod
            def alarmWithAbsoluteDate_(d):
                return ("alarm", py(d))

        self.module = types.SimpleNamespace(
            EKEventStore=Store,
            EKEvent=types.SimpleNamespace(eventWithEventStore_=lambda s: Item()),
            EKReminder=types.SimpleNamespace(reminderWithEventStore_=lambda s: Item()),
            EKAlarm=Alarm,
            EKSpanThisEvent=0,
        )

    def add_event(self, title, start, hours=1.0, cal="home", **kw):
        c = next(c for c in self.cals[0] if c.cid == cal)
        ev = Item(title=title, startDate=ns(start), endDate=ns(start + timedelta(hours=hours)), calendar=c, **kw)
        self.events.append(ev)
        return ev

    def busy_month(self):
        """The user keeps their calendar and reminders in the store: items on six of the last 30
        days (all in the past, so no read of today or later sees them)."""
        for d in (1, 3, 6, 9, 14, 20):
            self.add_event("x", at(-d, 10))
            self.reminders.append(Item(title="y", created=ns(at(-d, 10))))


# ---- a fake scriptable app, answering the generated scripts by their codes -----------------------------


class FakeApp:
    """What a calendar or to-do app would answer to NEO's scripts: it keeps what was made (by
    property code) and gives it back. Tests override single answers."""

    def __init__(self, containers=(("home", "Home", True), ("work", "Work", True), ("subs", "Holidays", False))):
        self.calls: list[tuple[str, list[str]]] = []
        self.containers = list(containers)
        self.items: dict[str, dict[str, str]] = {}
        self.lists: dict[str, tuple[int, str, str]] = {}  # calendar id -> the listing's answer
        self.make: tuple[int, str, str] | None = None
        self.listing: tuple[int, str, str] | None = None
        self.on_make = None
        self.default = "home"  # the container an item goes to when none is given
        self.no_id = False  # the new item's id can't be read after the make
        self.made_then: tuple[int, str, str] | None = None  # made it, then answered this

    def kinds(self, script: str) -> list[str]:
        return [k for k, mark in (("make", "make new"), ("readback", "set oid to item 3 of argv"),
                                  ("search", "set t to my neoDate(item 4 of argv)"), ("containers", "set ks to"),
                                  ("list", "set t0 to my neoDate(item 3 of argv)")) if mark in script]  # fmt: skip

    async def __call__(self, script, argv, timeout):
        self.calls.append((script, list(argv)))
        kind = self.kinds(script)[0]
        if kind == "containers":
            if self.listing is not None:
                return self.listing
            return 0, "OK" + "".join(f"{RS}{i}{US}{n}{US}{'true' if w else 'false'}" for i, n, w in self.containers), ""
        if kind == "make":
            cid = argv[1]
            if self.make is not None:
                if self.make[0] == 124:  # timed out after making it
                    self.items["late"] = {"_cid": cid, **dict(zip(re.findall(r"«property (.{4})»:\(", script), argv[2:], strict=False))}
                return self.make
            if cid and cid not in {i for i, _n, _w in self.containers}:
                return 0, "NO_CONTAINER", ""
            oid = f"obj{len(self.items) + 1}"
            self.items[oid] = {"_cid": cid, **dict(zip(re.findall(r"«property (.{4})»:\(", script), argv[2:], strict=False))}
            if self.on_make:
                self.on_make(self.items[oid])
            if self.made_then is not None:
                return self.made_then
            name = next((n for i, n, _w in self.containers if i == (cid or self.default)), "")
            return 0, f"OK{US}{'' if self.no_id else oid}{US}{name}", ""
        if kind == "readback":
            item = self.items.get(argv[2])
            if item is None:
                return 0, "EMPTY", ""
            codes = re.findall(r'neoText\(get «property (.{4})» of o, "\w+"\)', script)
            return 0, "OK" + "".join(US + item.get(c, "") for c in codes), ""
        if kind == "search":
            hit = next((i for i, it in self.items.items() if argv[2] in it.values()), None)
            return (0, f"OK{US}{hit}", "") if hit else (0, "EMPTY", "")
        return self.lists.get(argv[1], (0, "OK", ""))


def listing(*events: dict) -> tuple[int, str, str]:
    """A calendar's answer to the listing script: rows of id, title, start, end, all day, repeat
    rule, excluded dates."""
    cols = ("id", "title", "start", "end", "all_day", "recur", "exdates")
    return 0, "OK" + "".join(RS + US.join(str(e.get(c, "")) for c in cols) for e in events), ""


# ---- the world ------------------------------------------------------------------------------------------

NOW = time.time()
DATEBOOK = App(bundle_id="test.datebook", name="Datebook", path="/Applications/Datebook.app",
               script=sdef.load_file(str(FIX / "datebook.sdef")), last_used=NOW - 5 * DAY, days_used_30=2)  # fmt: skip
TASKLIST = App(bundle_id="test.tasklist", name="Tasklist", path="/Applications/Tasklist.app",
               script=sdef.load_file(str(FIX / "tasklist.sdef")), last_used=NOW - 5 * DAY, days_used_30=2)  # fmt: skip
# the kinds as the plan defines them in engine.TASKS / DRIVERS (installed per test)
TASKS = {
    "add_reminder": engine.Task("remind me later, or add a to-do or list item", wants=("reminder.make",),
                                signals=("intent:INAddTasksIntent",), routes=("eventkit.reminders",),
                                laya_min_used=1, proven_fit=True, prefs=True),
    "add_event": engine.Task("put a meeting or appointment on my calendar", wants=("event.make",),
                             routes=("eventkit.events",), fallback="ics", laya_min_used=1, proven_fit=True, prefs=True),
    "read_calendar": engine.Task("what's on my calendar or schedule", wants=("event.read",),
                                 routes=("eventkit.events",), laya_min_used=1),
}  # fmt: skip


@pytest.fixture
def world(monkeypatch):
    ek = FakeEK()
    monkeypatch.setitem(sys.modules, "EventKit", ek.module)
    monkeypatch.setattr(eventkit, "_ek", lambda: ek.module)  # never the real framework
    monkeypatch.setattr(evidence, "responsible_usage", lambda: USAGE)
    monkeypatch.setattr(eventkit, "_store_obj", None)
    for kind, task in TASKS.items():
        monkeypatch.setitem(engine.TASKS, kind, task)
        monkeypatch.setitem(engine.DRIVERS, kind, "neo.decide.kinds.schedule")
    w = SimpleNamespace(ek=ek, apps=[DATEBOOK, TASKLIST], opened=[], osa=FakeApp())
    monkeypatch.setattr(engine, "_inventory", lambda: list(w.apps))
    monkeypatch.setattr(inventory, "inventory", lambda refresh=False: list(w.apps))
    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: False)
    monkeypatch.setattr(media, "_osascript", w.osa)

    async def allowed(app):
        return None

    async def open_with(bid, path):
        w.opened.append((bid, path))
        return True

    monkeypatch.setattr(media, "_control_gate", allowed)
    monkeypatch.setattr(schedule, "_open_with", open_with)
    monkeypatch.setattr(workspace, "default_for_extension", lambda ext: "test.importer" if ext == "ics" else None)
    ek.busy_month()
    token = run._gen.set(None)
    run.forget()
    eventkit.forget()
    yield w
    run.forget()
    eventkit.forget()
    run._gen.reset(token)


@pytest.fixture
def switched_on(monkeypatch):
    """The schedule kinds switched on (a fast path's captures count only then)."""
    for kind in schedule.KINDS:
        monkeypatch.setitem(settings().decide_modes, kind, "on")


async def add_event(title="Lunch with Sam", start=None, text="", **kw):
    args = {"title": title, "start": (start or at(5, 13)).isoformat(timespec="minutes"), **kw}
    return await schedule.carry_out("add_event", args, text)


# ---- adding an event ------------------------------------------------------------------------------------


async def test_an_event_goes_to_the_default_calendar_through_the_store(world):
    res = await add_event(location="Cafe", notes="bring the book", text="put lunch with sam on my calendar")
    assert res.ok and "Added “Lunch with Sam” to Work, 1 PM to 2 PM" in res.text
    ev = world.ek.saved[-1]
    assert ev.v["calendar"].cid == "work" and (py(ev.v["startDate"]), py(ev.v["endDate"])) == (at(5, 13), at(5, 14))
    assert ev.v["location"] == "Cafe" and ev.v["notes"] == "bring the book" and ev.v["allDay"] is False
    assert world.osa.calls == [] and world.opened == [] and world.ek.asks == []
    assert engine.proven("add_event", "route:eventkit.events") and res.say is True  # the first time: where it went
    assert run.last("add_event") == ("route:eventkit.events", "")
    again = await add_event("Coffee", at(6, 10))
    assert again.ok and again.say is False  # where this kind always goes: nothing to say


async def test_a_named_calendar_by_its_words_a_tie_by_a_sure_laya_else_the_default_said(world, monkeypatch):
    res = await add_event(calendar="my home calendar")
    assert res.ok and world.ek.saved[-1].v["calendar"].cid == "home" and "to Home" in res.text
    world.ek.cals[0] += [Cal("fam1", "Family"), Cal("fam2", "Family")]
    answers = {"Family": 0.5, "Family (2)": 0.5}
    asked: list[set] = []

    async def laya(state, q, timeout=None):
        if "which" not in q:
            return None  # the engine's own question (which app): no opinion
        asked.append(set(q["which"]["criteria"]))  # only ever the user's own titles, locally
        return {"answers": {"which": {"probabilities": dict(answers)}}}

    monkeypatch.setattr(engine, "_zero_shot_can_answer", lambda: True)
    monkeypatch.setattr(engine, "_ask_laya", laya)
    res = await add_event("Picnic", calendar="family")
    assert world.ek.saved[-1].v["calendar"].cid == "work"  # Laya unsure: the default, and said
    assert "couldn't tell which calendar “family” is, so I put it in Work" in res.text and res.say is True
    answers.update({"Family": 0.05, "Family (2)": 0.95})
    res = await add_event("Picnic 2", calendar="family")
    assert world.ek.saved[-1].v["calendar"].cid == "fam2" and "to Family" in res.text
    assert asked == [{"Family", "Family (2)"}] * 2  # a tie: only the tied ones
    answers.clear()
    res = await add_event("Picnic 3", calendar="Gardening")  # no fit at all: every title, and Laya has no idea
    assert asked[-1] == {"Home", "Work", "Family", "Family (2)"}
    assert world.ek.saved[-1].v["calendar"].cid == "work" and "There's no calendar called Gardening" in res.text


async def test_a_day_alone_is_an_all_day_event(world):
    day = at(9, 0).date()
    res = await schedule.carry_out("add_event", {"title": "Trip", "start": day.isoformat()})
    ev = world.ek.saved[-1]
    assert res.ok and ev.v["allDay"] is True and "all day" in res.text
    assert py(ev.v["startDate"]) == py(ev.v["endDate"]) == datetime.combine(day, datetime.min.time())
    await schedule.carry_out("add_event", {"title": "Fair", "start": at(10, 15).isoformat(), "all_day": True})
    assert world.ek.saved[-1].v["allDay"] is True and py(world.ek.saved[-1].v["startDate"]).hour == 0


async def test_write_only_access_adds_but_says_it_cant_check(world):
    world.ek.status[0] = WRITE_ONLY
    res = await add_event(calendar="home")
    assert res.ok and world.ek.saved and res.say is True
    assert "your default calendar" in res.text and "can't double-check" in res.text and "can't see your calendars" in res.text


async def test_an_overlap_is_said_without_the_other_events_title(world):
    world.ek.add_event("Secret board meeting", at(5, 13, 30), 0.5, cal="work")
    world.ek.add_event("Holiday", at(5, 0), 24, allDay=True)
    res = await add_event()
    assert "It overlaps with another event at 1:30 PM." in res.text and "Secret" not in res.text and res.say is True


async def test_the_same_add_twice_is_added_once(world):
    first = await add_event()
    again = await add_event(title="lunch  with sam")
    assert first.ok and again.ok and again.text == "Already added a moment ago." and len(world.ek.saved) == 1


async def test_a_repeat_is_told_by_its_words_and_its_list(world):
    early, late = datetime(2026, 9, 26, 10, 0, 55), datetime(2026, 9, 26, 10, 1, 2)  # 7 s apart, across a minute
    a = schedule.parse_args("add_reminder", {"title": "check the oven", "due": "in 20 minutes"}, early)
    b = schedule.parse_args("add_reminder", {"title": "check the oven", "due": "in 20 minutes"}, late)
    assert a.sig == b.sig and a.start == b.start == datetime(2026, 9, 26, 10, 21)  # to the nearest minute
    await schedule.carry_out("add_reminder", {"title": "tea", "due": "in 20 minutes"})
    again = await schedule.carry_out("add_reminder", {"title": "tea", "due": "in 20 minutes"})
    assert again.text == "Already added a moment ago." and len(world.ek.saved) == 1
    await schedule.carry_out("add_reminder", {"title": "milk", "list": "Inbox"})
    other = await schedule.carry_out("add_reminder", {"title": "milk", "list": "Errands"})
    assert other.ok and "to Errands" in other.text and len(world.ek.saved) == 3  # another list: another reminder


def test_the_users_words_and_a_models_iso_time_for_them_are_the_same_add():
    now = datetime(2026, 9, 26, 10, 0)
    said = schedule.parse_args("add_event", {"title": "lunch with sam", "start": "tomorrow at 1"}, now)
    iso = schedule.parse_args("add_event", {"title": "Lunch with Sam", "start": "2026-09-27T13:00"}, now)
    assert said.start == iso.start and said.sig == iso.sig and iso.said_sig == ()
    assert said.said_sig == ("lunch with sam", "", "tomorrow at 1")  # "in 20 minutes" moves with the clock
    other = schedule.parse_args("add_event", {"title": "Lunch with Sam", "start": "2026-09-27T14:00"}, now)
    assert other.sig != said.sig


async def test_a_read_back_that_breaks_after_the_save_is_unconfirmed_and_never_saved_twice(world, monkeypatch):
    def broken(self, i):
        raise ValueError("objc hiccup")

    monkeypatch.setattr(world.ek.module.EKEventStore, "eventWithIdentifier_", broken)
    first = await add_event()
    assert not first.ok and first.unsure and "may have added it" in first.text and first.say is True
    again = await add_event(title="lunch with sam")  # the model retrying it
    assert again.unsure and again.text == first.text and len(world.ek.saved) == 1


# ---- through the session: the user's words, a fast path, the driver ------------------------------------------


def _heard(text: str, monkeypatch) -> tuple:
    """(the session's reply to the user's ``text``, the session): with the kind on, a fast path
    runs the tool; no model may be asked anything."""
    import neo.agent.session as sess_mod
    from neo import events
    from neo.agent import early
    from neo.agent.session import Session
    from neo.events import EventBus
    from neo.reflex.schema import Decision

    class Reflex:
        async def decide(self, t):
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head")

    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(early, "_actor", None)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": pytest.fail("handed to a model"))
    session = Session(reflex=Reflex())
    return asyncio.run(session.handle(text, from_user=True)), session


def test_an_add_that_may_have_happened_is_said_and_never_handed_to_a_model(world, decide_modes, monkeypatch):
    decide_modes(add_event="on")
    monkeypatch.setitem(schedule.BUDGET_S, "eventkit", 0.2)
    world.ek.on_save = lambda: time.sleep(0.4)  # saved, but past the attempt's budget
    reply, _s = _heard("put lunch with sam on my calendar tomorrow at 1", monkeypatch)
    assert reply.route == "quick" and not reply.silent and "may have" in reply.text
    assert len(world.ek.saved) == 1
    # a model's own retry of it (its casing, an ISO time) adds nothing either
    from neo.agent.registry import ToolContext, registry

    start = at(1, 13).isoformat(timespec="minutes")
    out = asyncio.run(registry().invoke("calendar_add", {"title": "Lunch with Sam", "start": start}, ToolContext()))
    assert out.ok and "may have" in out.text and len(world.ek.saved) == 1


def test_the_ask_once_calendar_question_is_asked_not_handed_to_a_model(world, decide_modes, monkeypatch):
    decide_modes(add_event="on")
    world.ek.status[0] = DENIED  # the calendar app's own dictionary, with two calendars to add to
    reply, session = _heard("put gym on my calendar tomorrow at 7", monkeypatch)
    assert reply.route == "quick" and not reply.silent
    assert "Which calendar should I use for new events in Datebook" in reply.text
    assert "Which calendar" in session.history[-1].text  # the conversation holds the question
    assert not any("make new" in s for s, _a in world.osa.calls) and world.ek.saved == []


async def test_a_bad_time_or_title_asks_and_touches_nothing(world):
    for args, want in (({"title": "x", "start": "whenever you like"}, "couldn't tell when"),
                       ({"title": "", "start": "2026-10-01T10:00"}, "What should it be called?"),
                       ({"title": "x", "start": "2026-10-01T10:00", "end": "2026-10-01T09:00"}, "end before it starts")):  # fmt: skip
        res = await schedule.carry_out("add_event", args)
        assert not res.ok and want in res.text and res.say is True
    assert world.ek.saved == [] and world.osa.calls == []


# ---- access: asked only when switched on, never without the usage key -----------------------------------


async def test_the_store_is_asked_for_only_by_a_kind_switched_on(world):
    world.ek.status = {0: NOT_ASKED, 1: NOT_ASKED}
    world.ek.answer = FULL
    res = await add_event(calendar="Work")  # not switched on: the store stays unasked, the app is next
    assert world.ek.asks == [] and world.ek.saved == [] and res.ok and "Work in Datebook" in res.text
    facts = await schedule.would_choose("add_event", {"title": "x", "start": "2026-10-01T10:00"})
    assert world.ek.asks == [] and facts["store"] == "not_determined"
    res = await schedule.carry_out("add_event", {"title": "Dentist", "start": at(6, 9).isoformat()}, ask_access=True)
    assert world.ek.asks == [0] and res.ok and world.ek.saved[-1].v["title"] == "Dentist"
    await schedule.carry_out("add_event", {"title": "Dentist 2", "start": at(7, 9).isoformat()}, ask_access=True)
    assert world.ek.asks == [0]  # asked once


async def test_naming_an_app_or_a_missing_usage_key_asks_nothing(world, monkeypatch):
    world.ek.status = {0: NOT_ASKED, 1: NOT_ASKED}
    res = await schedule.carry_out("add_event", {"title": "Gym", "start": at(4, 7).isoformat(), "calendar": "home",
                                                 "app": "Datebook"}, "add gym in datebook", ask_access=True)  # fmt: skip
    assert world.ek.asks == [] and res.ok and "Home in Datebook" in res.text
    monkeypatch.setattr(evidence, "responsible_usage", lambda: frozenset())
    res = await schedule.carry_out("add_reminder", {"title": "call mom"}, ask_access=True)
    assert world.ek.asks == [] and res.ok and "Tasklist" in res.text  # the store could never be asked: the app
    assert await eventkit.request("reminder") == "not_determined" and world.ek.asks == []
    world.apps = []
    res = await schedule.carry_out("add_reminder", {"title": "call dad"}, ask_access=True)
    assert not res.ok and "store can't be used: NEO's app doesn't declare reminders access" in res.text


async def test_only_the_tools_app_names_an_app_never_the_kinds_own_noun(world):
    """'show my calendar', 'put it in calendar': a calendar app is often called by the kind's own
    noun, so the words alone never pick an app over the store (nor keep the store from being
    asked); the tool's `app` does (the model heard the user name one)."""
    world.ek.add_event("Standup", at(1, 9), cal="work")
    res = await schedule.carry_out("read_calendar", {"day": "tomorrow"}, "show my datebook for tomorrow")
    assert res.ok and "Standup" in res.text and world.osa.calls == []
    res = await add_event("Gym", at(4, 7), text="put it in datebook")
    assert res.ok and world.ek.saved[-1].v["title"] == "Gym" and world.osa.calls == []
    res = await add_event("Swim", at(4, 8), text="put swim in datebook", calendar="home", app="Datebook")
    assert res.ok and "Home in Datebook" in res.text and world.ek.saved[-1].v["title"] == "Gym"
    world.ek.status = {0: NOT_ASKED, 1: NOT_ASKED}
    world.ek.answer = FULL
    res = await schedule.carry_out("read_calendar", {"day": "tomorrow"}, "show my datebook for tomorrow", ask_access=True)
    assert world.ek.asks == [0] and res.ok and "Standup" in res.text


async def test_a_store_the_user_turned_down_is_named_when_nothing_else_works(world):
    world.ek.status = {0: DENIED, 1: DENIED}
    world.apps = []
    res = await schedule.carry_out("add_reminder", {"title": "call mom", "due": at(1, 17).isoformat()})
    assert not res.ok and "Reminders access is off for NEO: System Settings › Privacy & Security › Reminders" in res.text


# ---- when the store is closed: the app's dictionary ------------------------------------------------------


async def test_denied_store_goes_to_the_app_which_asks_once_which_calendar(world):
    world.ek.status[0] = DENIED
    res = await add_event()
    assert isinstance(res.text, Halt) and not res.ok
    assert "Which calendar should I use for new events in Datebook: Home or Work?" in res.text  # not the read-only one
    assert [world.osa.kinds(s)[0] for s, _a in world.osa.calls] == ["containers"]  # nothing made
    res = await add_event(calendar="work")  # the user's answer
    assert res.ok and "Added “Lunch with Sam” to Work in Datebook" in res.text
    make = next((s, a) for s, a in world.osa.calls if "make new" in s)
    assert make[1][:3] == ["test.datebook", "work", "Lunch with Sam"] and "Lunch" not in make[0]
    assert world.osa.items["obj1"]["wr1s"] == at(5, 13).isoformat() and "wrad" not in world.osa.items["obj1"]
    assert learned.get("containers", "add_event:test.datebook") == {"id": "work"}
    world.osa.calls.clear()
    res = await add_event("Standup", at(6, 9))  # remembered: no question, no listing
    assert res.ok and [world.osa.kinds(s)[0] for s, _a in world.osa.calls] == ["make", "readback"]
    assert "“Standup” to Work in Datebook" in res.text  # where it went, by the calendar's own name
    assert world.osa.items["obj2"]["_cid"] == "work" and world.ek.saved == []


async def test_only_the_answer_to_which_calendar_is_remembered(world):
    world.ek.status[0] = DENIED
    res = await add_event("Mum's birthday dinner", at(5, 19), calendar="home")
    assert res.ok and "to Home in Datebook" in res.text
    assert learned.get("containers", "add_event:test.datebook") is None  # named for this one event only
    res = await add_event("Sprint review", at(6, 10))
    assert isinstance(res.text, Halt) and "Which calendar should I use" in res.text  # so still asked
    res = await add_event("Sprint review", at(6, 10), calendar="work")  # the answer
    assert res.ok and learned.get("containers", "add_event:test.datebook") == {"id": "work"}
    res = await add_event("Retro", at(7, 10))
    assert res.ok and "“Retro” to Work in Datebook" in res.text and world.osa.items["obj3"]["_cid"] == "work"


async def test_an_id_the_app_couldnt_give_back_is_searched_for_never_made_twice(world):
    world.ek.status[0] = DENIED
    world.osa.no_id = True  # made, but reading its id failed
    res = await add_event(calendar="work")
    assert res.ok and "to Work in Datebook" in res.text
    assert [world.osa.kinds(s)[0] for s, _a in world.osa.calls] == ["containers", "make", "search", "readback"]
    assert len(world.osa.items) == 1 and world.opened == []
    world.osa.no_id = False
    world.osa.made_then = (1, "", "execution error: Can't get object. (-1728)")  # made, then an object went missing
    res = await add_event("Dentist", at(6, 9), calendar="work")
    assert res.ok and len(world.osa.items) == 2 and world.opened == []
    world.osa.made_then = None
    world.osa.make = (1, "", "execution error: Can't get calendar. (-1728)")  # nothing made
    res = await add_event("Physio", at(7, 9), calendar="work")
    assert res.ok and len(world.osa.items) == 2 and len(world.opened) == 1  # so the next way: the file


async def test_a_missing_list_by_script_names_the_list_it_went_to(world):
    world.ek.status[1] = DENIED
    world.osa.containers = [("inbox", "Inbox", True), ("errands", "Errands", True)]
    world.osa.default = "inbox"
    res = await schedule.carry_out("add_reminder", {"title": "bread", "list": "shopping"})
    assert res.ok and res.say is True and world.osa.items["obj1"]["_cid"] == ""
    assert res.text.startswith("Added “bread” to Inbox in Tasklist.")
    assert "There's no list called shopping, so I put it in Inbox." in res.text


async def test_a_make_that_timed_out_is_found_again_or_never_made_twice(world):
    world.ek.status[0] = DENIED
    world.osa.make = (124, "", "timed out after 10s")
    res = await add_event(calendar="work")
    assert res.ok  # the search-back found it: done, once
    assert [world.osa.kinds(s)[0] for s, _a in world.osa.calls] == ["containers", "make", "search", "readback"]
    world.osa.items.clear()
    world.osa.make = (1, "", "AppleEvent timed out. (-1712)")
    res = await add_event("Other", calendar="work")
    assert not res.ok and "may have added it in Datebook" in res.text and res.say is True
    assert sum("make new" in s for s, _a in world.osa.calls) == 2 and world.opened == []  # no second copy, no file
    arms = engine._load()["arms"]["add_event"]["test.datebook"]["script_make"]
    assert arms["fails"] == 0  # unconfirmed: not the app's fault


async def test_an_app_that_writes_into_the_store_steps_aside_afterwards(world):
    world.osa.on_make = lambda item: world.ek.add_event(item["wr11"], datetime.fromisoformat(item["wr1s"]))
    res = await add_event(calendar="home", text="lunch in datebook", app="Datebook")
    assert res.ok and "Datebook" in res.text and world.ek.saved == []
    assert learned.get("store_views", "add_event:test.datebook") == {"view": True}
    cands = await engine.choose("add_event", "lunch")
    assert [c.key for c in cands] == ["route:eventkit.events", "ics"]


async def test_a_reminder_by_script_gets_its_due_time_as_its_alert(world):
    world.ek.status[1] = DENIED
    due = at(1, 17)
    res = await schedule.carry_out("add_reminder", {"title": "call mom", "due": due.isoformat(), "notes": "about Sunday"})
    assert res.ok and "“call mom” to Home in Tasklist" in res.text  # the list the app put it in
    item = world.osa.items["obj1"]
    assert item["dued"] == item["rmdt"] == due.isoformat() and item["body"] == "about Sunday" and item["_cid"] == ""
    await schedule.carry_out("add_reminder", {"title": "pay rent", "due": at(3, 0).date().isoformat()})
    item = world.osa.items["obj2"]
    assert "adue" in item and "dued" not in item and "rmdt" not in item  # a day: no alert
    res = await schedule.carry_out("add_reminder", {"title": "milk", "list": "errands"})
    assert "There's no list called errands, so I put it in Home." in res.text  # this app's lists: none named so


# ---- reminders through the store --------------------------------------------------------------------------


async def test_a_timed_reminder_gets_an_alarm_and_a_day_does_not(world):
    res = await schedule.carry_out("add_reminder", {"title": "call mom", "due": at(1, 17).isoformat()})
    r = world.ek.saved[-1]
    assert res.ok and r.alarms() == [("alarm", at(1, 17))] and r.v["calendar"].cid == "inbox" and "Inbox" in res.text
    await schedule.carry_out("add_reminder", {"title": "pay rent", "due": "tomorrow"})
    r = world.ek.saved[-1]
    from Foundation import NSDateComponentUndefined

    assert r.alarms() == [] and r.v["dueDateComponents"].hour() == NSDateComponentUndefined
    await schedule.carry_out("add_reminder", {"title": "someday"})
    assert "dueDateComponents" not in world.ek.saved[-1].v and world.ek.saved[-1].alarms() == []


@pytest.mark.parametrize(
    "phrase, days, hour, minute",
    [("kal subah 9 baje", 1, 9, 0), ("परसों शाम 7 बजे", 2, 19, 0), ("tomorrow at 5:30 pm", 1, 17, 30),
     ("saadhe 8 baje kal raat", 1, 20, 30)],
)
async def test_the_users_own_words_for_a_time(world, phrase, days, hour, minute):
    res = await schedule.carry_out("add_reminder", {"title": "x", "due": phrase})
    assert res.ok and world.ek.saved[-1].alarms() == [("alarm", at(days, hour, minute))]


async def test_an_inferred_time_is_said_back(world):
    await schedule.carry_out("add_reminder", {"title": "tea", "due": "in 20 minutes"})
    res = await schedule.carry_out("add_reminder", {"title": "tea 2", "due": "in 30 minutes"})
    assert res.ok and res.say is False  # the words said exactly when, and it went where it always goes
    guess = schedule.parse_args("add_reminder", {"title": "tea 3", "due": "5 baje"})
    res = await schedule.carry_out("add_reminder", {"title": "tea 3", "due": "5 baje"})
    assert res.ok and res.say is True and guess.inferred and _clock_words(guess.start) in res.text


def _clock_words(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12} {'AM' if dt.hour < 12 else 'PM'}"


async def test_a_missing_list_goes_to_the_default_and_is_said(world):
    res = await schedule.carry_out("add_reminder", {"title": "milk", "list": "shopping"})
    assert res.ok and world.ek.saved[-1].v["calendar"].cid == "inbox" and res.say is True
    assert "There's no list called shopping, so I put it in Inbox." in res.text
    res = await schedule.carry_out("add_reminder", {"title": "stamps", "list": "my errands list"})
    assert world.ek.saved[-1].v["calendar"].cid == "errands" and "to Errands" in res.text


# ---- reading the calendar ---------------------------------------------------------------------------------


async def test_the_days_events_repeats_included_declined_skipped(world):
    today = at(0, 0)
    world.ek.add_event("Standup", today.replace(hour=9), 0.25)  # an occurrence of a repeating event
    world.ek.add_event("Offsite", today, 24, allDay=True, cal="work")
    world.ek.add_event("Declined sync", today.replace(hour=11), attendees=[Attendee(True, 3)])
    world.ek.add_event("Tomorrow's", at(1, 9))
    res = await schedule.carry_out("read_calendar", {})
    assert res.ok and res.text.splitlines() == ["2 events today:", "all day: Offsite (Work)", "09:00–09:15 Standup (Home)"]
    res = await schedule.carry_out("read_calendar", {"day": "kal"})
    assert res.text.splitlines()[1:] == ["09:00–10:00 Tomorrow's (Home)"] and world.osa.calls == []


async def test_empty_is_said_only_after_a_read_that_worked(world):
    res = await schedule.carry_out("read_calendar", {"day": "tomorrow"})
    assert res.ok and res.text == "Nothing on your calendar tomorrow."
    world.ek.status[0] = DENIED
    world.apps = []
    res = await schedule.carry_out("read_calendar", {"day": "tomorrow"})
    assert not res.ok and "on your calendar" not in res.text and "System Settings › Privacy & Security › Calendars" in res.text
    world.ek.status[0] = WRITE_ONLY
    res = await schedule.carry_out("read_calendar", {})
    assert not res.ok and "may only add to your calendar" in res.text


async def test_a_calendar_read_by_script_names_the_one_that_was_too_slow(world):
    world.ek.status[0] = DENIED
    day = at(0, 0)
    world.osa.lists["home"] = listing({"id": "a", "title": "Dentist", "start": day.replace(hour=15).isoformat(),
                                       "end": day.replace(hour=16).isoformat(), "all_day": "false"})  # fmt: skip
    world.osa.lists["work"] = (124, "", "timed out after 20s")
    res = await schedule.carry_out("read_calendar", {})
    assert res.ok and res.text.splitlines() == ["1 event today:", "15:00–16:00 Dentist (Home)", "(I couldn't read Work in time.)"]
    world.osa.listing = (1, "", "Not authorized to send Apple events to Datebook. (-1743)")
    res = await schedule.carry_out("read_calendar", {})
    assert not res.ok and "needs your OK" in res.text


@pytest.fixture
def new_york():
    """A time zone with daylight saving (this Mac's may have none)."""
    was = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if was is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = was
    time.tzset()


def test_script_rows_overlap_repeats_and_excluded_dates_across_dst(new_york):
    t0, t1 = datetime(2026, 11, 1), datetime(2026, 11, 15)  # daylight saving ends on 1 November
    cols = ("id", "title", "start", "end", "all_day", "recur", "exdates")
    rows = [
        ["w", "Weekly", "2026-10-26T09:00:00", "2026-10-26T09:30:00", "false", "FREQ=WEEKLY;UNTIL=20261110T140000Z",
         "2026-11-02T09:00:00"],
        ["e", "Ends at the start", "2026-10-31T23:00:00", "2026-11-01T00:00:00", "false", "", ""],
        ["o", "Overnight", "2026-10-31T23:00:00", "2026-11-01T01:00:00", "false", "", ""],
        ["d", "Daily", "2026-11-13T08:00:00", "2026-11-13T08:15:00", "false", "RRULE:FREQ=DAILY;COUNT=5", ""],
        ["x", "Broken rule", "2026-11-03T10:00:00", "2026-11-03T11:00:00", "false", "FREQ=SOMETIMES", ""],
        ["b", "Bad", "not a date", "", "", "", ""],
    ]  # fmt: skip
    occ, expanded = schedule._occurrences(cols, rows, t0, t1, "Home")
    got = sorted((o.title, o.start) for o in occ)
    assert expanded and got == [
        ("Broken rule", datetime(2026, 11, 3, 10)),  # an unreadable rule: the event itself
        ("Daily", datetime(2026, 11, 13, 8)),
        ("Daily", datetime(2026, 11, 14, 8)),  # clipped at the window's end
        ("Overnight", datetime(2026, 10, 31, 23)),
        ("Weekly", datetime(2026, 11, 9, 9)),  # 9 AM on both sides of the change; 2 Nov excluded; until 10 Nov
    ]


# ---- the calendar-file fallback ------------------------------------------------------------------------------


def test_the_calendar_file_escapes_folds_and_carries_the_time_zone():
    spec = schedule.parse_args("add_event", {"title": "Plan; a,b \\ c", "start": "2026-10-01T13:00", "notes": "one\ntwo",
                                             "location": "Room " + "प" * 40}, datetime(2026, 9, 26))  # fmt: skip
    text = schedule.ics_text(spec, "u-1@neo.local", datetime(2026, 9, 26, 10, 0, tzinfo=schedule.UTC), "Asia/Kolkata")
    lines = text.split("\r\n")
    assert text.endswith("\r\n") and "UID:u-1@neo.local" in lines and "DTSTAMP:20260926T100000Z" in lines
    assert "DTSTART;TZID=Asia/Kolkata:20261001T130000" in lines and "DTEND;TZID=Asia/Kolkata:20261001T140000" in lines
    assert "SUMMARY:Plan\\; a\\,b \\\\ c" in lines and "DESCRIPTION:one\\ntwo" in lines
    assert all(len(line.encode()) <= 75 for line in lines) and any(line.startswith(" ") for line in lines)
    unfolded = text.replace("\r\n ", "")
    assert "LOCATION:Room " + "प" * 40 in unfolded
    day = schedule.parse_args("add_event", {"title": "Trip", "start": "2026-10-03"}, datetime(2026, 9, 26))
    day_text = schedule.ics_text(day, "u", datetime(2026, 9, 26, tzinfo=schedule.UTC), "")
    assert "DTSTART;VALUE=DATE:20261003\r\nDTEND;VALUE=DATE:20261004" in day_text


async def test_the_file_is_handed_over_only_when_nothing_else_can(world):
    world.ek.status[0] = NOT_ASKED
    world.apps = []
    tmp = settings().data_dir / "tmp"
    tmp.mkdir(parents=True)
    stale = tmp / "neo-old.ics"
    stale.write_text("x")
    os.utime(stale, (time.time() - 1200, time.time() - 1200))
    res = await add_event()
    assert res.ok and res.text == "It's open in your calendar app. Press Add there." and res.say is True
    (bid, path), = world.opened
    assert bid == "test.importer" and "SUMMARY:Lunch with Sam" in Path(path).read_text()
    assert oct(os.stat(path).st_mode & 0o777) == "0o600" and not stale.exists()
    world.ek.status[0] = FULL  # the store itself: no file
    res = await add_event("Tea")
    assert res.ok and len(world.opened) == 1 and world.ek.saved[-1].v["title"] == "Tea"
    assert await schedule.fallback(run.Req("add_event", {}, ""), "ics") is None


async def test_a_handed_over_file_goes_after_a_while_and_with_the_erase(world, monkeypatch):
    world.ek.status[0] = DENIED
    world.apps = []
    monkeypatch.setattr(schedule, "ICS_KEEP_S", 0.05)
    res = await add_event("Therapy session", notes="private notes")
    path = Path(world.opened[-1][1])
    assert res.ok and path.exists()
    await asyncio.sleep(0.3)
    assert not path.exists()
    monkeypatch.setattr(schedule, "ICS_KEEP_S", 600.0)
    await add_event("Therapy session 2", at(6, 9))
    path = Path(world.opened[-1][1])
    assert path.exists()
    await privacy.forget_everything()
    assert not path.exists() and not list((settings().data_dir / "tmp").glob("neo-*.ics"))


async def test_stale_files_go_at_start_and_a_file_that_didnt_open_goes_at_once(world, monkeypatch):
    tmp = settings().data_dir / "tmp"
    tmp.mkdir(parents=True)
    stale, fresh = tmp / "neo-old.ics", tmp / "neo-new.ics"
    for p in (stale, fresh):
        p.write_text("x")
    os.utime(stale, (time.time() - 1200, time.time() - 1200))
    schedule.prune()
    assert not stale.exists() and fresh.exists()
    world.ek.status[0] = DENIED
    world.apps = []

    async def fails(bid, path):
        world.opened.append((bid, path))
        return False

    monkeypatch.setattr(schedule, "_open_with", fails)
    res = await add_event()
    assert not res.ok and not Path(world.opened[-1][1]).exists()


# ---- nothing learned after an erase ----------------------------------------------------------------------------


async def test_an_erase_during_the_request_leaves_nothing_learned(world, monkeypatch):
    world.ek.on_save = lambda: monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)
    res = await add_event()
    assert res.ok and not engine.proven("add_event", "route:eventkit.events") and run.last("add_event") is None
    world.ek.on_save = None
    world.ek.status[0] = DENIED
    world.osa.on_make = lambda item: monkeypatch.setattr(privacy, "_generation", privacy.generation() + 1)
    res = await add_event("Other", calendar="work")
    assert res.ok and learned.get("containers", "add_event:test.datebook") is None
    assert not engine.proven("add_event", "test.datebook")


# ---- shadow: what would be chosen, and nothing else ------------------------------------------------------------


async def test_shadow_says_what_it_would_do_and_does_nothing(world):
    world.ek.add_event("Private title", at(0, 12), cal="work")
    world.ek.status[1] = NOT_ASKED  # would need asking: never asked here
    facts = {
        "event": await schedule.would_choose("add_event", {"title": "Secret plan", "start": "2026-10-01T10:00",
                                                           "calendar": "home"}),  # fmt: skip
        "reminder": await schedule.would_choose("add_reminder", {"title": "Secret errand", "due": "kal 5 baje"}),
        "read": await schedule.would_choose("read_calendar", {"day": "today"}),
        "bad": await schedule.would_choose("add_event", {"title": "x", "start": "whenever"}),
    }
    assert world.ek.saved == [] and world.ek.asks == [] and world.osa.calls == [] and world.opened == []
    assert not (settings().data_dir / "decide.json").exists() and not (settings().data_dir / "decide_learned").exists()
    ev, rem, read = facts["event"], facts["reminder"], facts["read"]
    assert ev["first"] == "route:eventkit.events" and ev["methods"] == "eventkit" and ev["container"] == "home"
    assert ev["named_found"] is True and ev["candidates"].split(",")[-1] == "ics" and ev["all_day"] is False
    assert rem["store"] == "not_determined" and rem["alarm"] is True and rem["inferred"] is True
    assert read["items"] == 1 and read["first"] == "route:eventkit.events"
    assert facts["bad"] == {"kind": "add_event", "parsed": False}
    flat = " ".join(str(v) for f in facts.values() for v in f.values())
    for private in ("Secret", "Private", "Home", "Work", "Inbox"):
        assert private not in flat
    assert all(isinstance(v, str | int | float | bool) for f in facts.values() for v in f.values())


async def test_shadow_never_breaks_the_real_request(world, monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("engine fell over")

    monkeypatch.setattr(engine, "choose", broken)
    facts = await schedule.would_choose("add_event", {"title": "x", "start": "2026-10-01T10:00"})
    assert facts["error"] == "RuntimeError"


# ---- the fast paths' test ---------------------------------------------------------------------------------------


def test_fast_path_check_uses_cached_data_only(world, switched_on):
    assert schedule.check("add_reminder", {"title": "call mom", "due": "kal 5 baje"})
    assert schedule.check("add_reminder", {"title": "call mom"})
    assert not schedule.check("add_reminder", {"title": "it", "due": "at 5"})
    assert not schedule.check("add_reminder", {"title": "call mom", "due": "when the cows come home"})
    assert not schedule.check("add_reminder", {"title": "milk", "list": "errands"})  # never read: yield to the model
    eventkit.calendars("reminder")
    assert schedule.check("add_reminder", {"title": "milk", "list": "errands"})
    assert not schedule.check("add_reminder", {"title": "milk", "list": "shopping"})
    assert schedule.check("add_event", {"title": "dentist", "start": "friday at 3 pm"})
    assert not schedule.check("add_event", {"title": "dentist", "start": "friday"})  # no clock time: the model asks
    assert not schedule.check("add_event", {"title": "that", "start": "friday at 3 pm"})
    assert schedule.check("read_calendar", {}) and schedule.check("read_calendar", {"day": "kal tha"})
    assert not schedule.check("read_calendar", {"day": "someday"}) and not schedule.check("open_app", {})
    eventkit.forget()
    assert eventkit.titles("reminder") is None


def test_a_fast_path_time_that_starts_with_the_titles_own_word_goes_to_the_model(world, switched_on):
    """'remind me to put the kettle on at 5': a lazy title stops at 'on', and 'on at 5' still
    reads as a time - the model splits it instead."""
    for due in ("on at 5", "on tomorrow at 7", "in at 5", "this at 5", "in on friday", "on in 10 minutes"):
        assert not schedule.check("add_reminder", {"title": "put the kettle", "due": due}), due
        assert not schedule.check("add_event", {"title": "put the kettle", "start": due}), due
    for due in ("at 5", "on friday", "in 20 minutes", "by tomorrow", "on this friday", "at around 5", "tomorrow at 7"):
        assert schedule.check("add_reminder", {"title": "call mom", "due": due}), due


def test_fast_path_captures_count_only_for_a_kind_switched_on(world, monkeypatch):
    """Legacy and shadow run today's scripts, which take ISO times only: the words a fast path
    captured go to the model instead."""
    monkeypatch.setitem(settings().decide_modes, "add_reminder", "shadow")
    assert not schedule.check("add_reminder", {"title": "call mom", "due": "kal 5 baje"})
    assert not schedule.check("read_calendar", {"day": "kal"}) and schedule.check("read_calendar", {})
    monkeypatch.setitem(settings().decide_modes, "add_reminder", "on")
    assert schedule.check("add_reminder", {"title": "call mom", "due": "kal 5 baje"})


# ---- parts -----------------------------------------------------------------------------------------------------------


def test_methods_per_kind():
    store = App(bundle_id="route:eventkit.events", name="store", path="", kind="route")
    lists = App(bundle_id="route:eventkit.reminders", name="store", path="", kind="route")
    assert schedule.methods_for(store, "add_event") == schedule.methods_for(store, "read_calendar") == ("eventkit",)
    assert schedule.methods_for(store, "add_reminder") == () and schedule.methods_for(lists, "add_reminder") == ("eventkit",)
    assert schedule.methods_for(DATEBOOK) == ("script_make", "script_read")
    assert schedule.methods_for(DATEBOOK, "add_event") == ("script_make",)
    assert schedule.methods_for(DATEBOOK, "add_reminder") == ()
    assert schedule.methods_for(TASKLIST, "add_reminder") == ("script_make",)
    assert schedule.methods_for(TASKLIST, "read_calendar") == ()
    plain = App(bundle_id="test.plain", name="Plain", path="/Applications/Plain.app")
    assert schedule.methods_for(plain) == () and schedule.describe(DATEBOOK) == "can add by script, can list events by script"
    req = run.Req("read_calendar", {}, "")
    assert schedule.accepts(DATEBOOK, "script_read", req) and not schedule.accepts(DATEBOOK, "script_make", req)


def test_parse_args():
    now = datetime(2026, 9, 26, 10, 0)  # a Saturday
    ev = schedule.parse_args("add_event", {"title": " Lunch ", "start": "2026-10-01T13:00:00+05:30"}, now)
    local = datetime.fromisoformat("2026-10-01T13:00:00+05:30").astimezone().replace(tzinfo=None)
    assert ev.title == "Lunch" and ev.start == local and ev.end == local + timedelta(hours=1) and not ev.date_only
    ev = schedule.parse_args("add_event", {"title": "x", "start": "monday 3 to 4 pm"}, now)
    assert (ev.start, ev.end) == (datetime(2026, 9, 28, 15), datetime(2026, 9, 28, 16))
    ev = schedule.parse_args("add_event", {"title": "x", "start": "2026-10-03", "end": "2026-10-04"}, now)
    assert ev.date_only and (ev.start, ev.end) == (datetime(2026, 10, 3), datetime(2026, 10, 5))  # through the 4th
    rem = schedule.parse_args("add_reminder", {"title": "x", "due": "5 baje", "list": " Errands "}, now)
    assert rem.start == datetime(2026, 9, 26, 17) and rem.inferred == "5 PM today" and rem.container == "Errands"
    read = schedule.parse_args("read_calendar", {"day": "kal tha"}, now)
    assert (read.start, read.end) == (datetime(2026, 9, 25), datetime(2026, 9, 26))
    read = schedule.parse_args("read_calendar", {"start": "2026-10-01", "end": "2027-03-01"}, now)
    assert read.end - read.start == timedelta(days=eventkit.MAX_RANGE_DAYS)  # capped
    assert schedule.parse_args("read_calendar", {"day": "someday"}, now) == "I couldn't tell which day “someday” is."
    assert schedule.parse_args("add_event", {"title": "x"}, now) == "When is it?"
    assert schedule.parse_args("add_event", {"title": "x", "start": "2026-10-01T10:00", "app": "Datebook"}, now).app == "Datebook"
    assert schedule.parse_args("add_event", {"title": "x", "start": "2026-10-01T10:00"}, now).sig == ("x", "", "2026-10-01T10:00:00")


def test_an_all_day_end_at_midnight_ends_the_day_before():
    """A date alone is the last day (as for a read's end); a midnight is the moment the last day
    ends (iCalendar's exclusive end)."""
    now = datetime(2026, 9, 26, 10, 0)
    for args in ({"start": "2026-10-03T00:00", "end": "2026-10-04T00:00", "all_day": True},
                 {"start": "2026-10-03", "end": "2026-10-04T00:00"}, {"start": "2026-10-03", "end": "2026-10-03"},
                 {"start": "2026-10-03T00:00", "end": "2026-10-03T00:00", "all_day": True}):  # fmt: skip
        ev = schedule.parse_args("add_event", {"title": "Offsite", **args}, now)
        assert (ev.start, ev.end) == (datetime(2026, 10, 3), datetime(2026, 10, 4)), args
        assert schedule._when_words(ev, now) == "all day Saturday 3 October"
    two = schedule.parse_args("add_event", {"title": "Offsite", "start": "2026-10-03", "end": "2026-10-04"}, now)
    assert two.end == datetime(2026, 10, 5) and schedule._when_words(two, now) == "all day Saturday 3 October to Sunday 4 October"


async def test_an_all_day_event_ending_at_midnight_is_one_day_in_the_store(world):
    d = at(9, 0)
    res = await schedule.carry_out("add_event", {"title": "Offsite", "start": d.isoformat(timespec="minutes"),
                                                 "end": (d + timedelta(days=1)).isoformat(timespec="minutes"), "all_day": True})  # fmt: skip
    ev = world.ek.saved[-1]
    assert res.ok and py(ev.v["startDate"]) == py(ev.v["endDate"]) == d
    assert res.text.endswith(f", all day {schedule._day(d.date(), datetime.now())}.")  # one day, no range
    fair = await schedule.carry_out("add_event", {"title": "Fair", "start": (d + timedelta(days=3)).date().isoformat()})
    assert fair.ok and fair.say is False  # one day, where events always go: nothing to say
    last = d + timedelta(days=5)
    trip = await schedule.carry_out("add_event", {"title": "Trip", "start": (d + timedelta(days=4)).date().isoformat(),
                                                  "end": last.date().isoformat()})  # fmt: skip
    assert trip.ok and trip.say is True and trip.text.endswith(f" to {schedule._day(last.date(), datetime.now())}.")


def test_a_week_is_read_as_that_week(monkeypatch):
    now = datetime(2026, 9, 26, 10, 0)  # a Saturday
    monkeypatch.setattr(schedule, "_first_weekday", lambda: 0)  # weeks start on Monday
    for day in ("next week", "agle hafte", "अगले हफ्ते में", "for next week"):
        read = schedule.parse_args("read_calendar", {"day": day}, now)
        assert (read.start, read.end) == (datetime(2026, 9, 28), datetime(2026, 10, 5)), day
    read = schedule.parse_args("read_calendar", {"day": "this week"}, now)
    assert (read.start, read.end) == (datetime(2026, 9, 26), datetime(2026, 9, 28))  # what's left of it
    monkeypatch.setattr(schedule, "_first_weekday", lambda: 6)  # on Sunday
    read = schedule.parse_args("read_calendar", {"day": "is hafte"}, now)
    assert (read.start, read.end) == (datetime(2026, 9, 26), datetime(2026, 9, 27))
    read = schedule.parse_args("read_calendar", {"day": "next week"}, now)
    assert (read.start, read.end) == (datetime(2026, 9, 27), datetime(2026, 10, 4))
    assert isinstance(schedule.parse_args("read_calendar", {"day": "some week"}, now), str)


async def test_an_app_named_through_the_tool_reaches_the_engine(world):
    spec = schedule.parse_args("add_event", {"title": "x", "start": "2026-10-01T10:00", "app": "Datebook"})
    assert (await schedule._request(spec, "lunch friday")).text == "lunch friday in Datebook"
    assert (await schedule._request(spec, "lunch in datebook")).text == "lunch in datebook"
    assert (await schedule._request(spec, "")).text == "x in Datebook"
    unnamed = schedule.parse_args("add_event", {"title": "x", "start": "2026-10-01T10:00"})
    assert (await schedule._request(unnamed, "put lunch in datebook")).text == "put lunch in an app"


async def test_an_app_with_no_calendars_reads_as_empty(world):
    world.ek.status[0] = DENIED
    world.osa.containers = []
    res = await schedule.carry_out("read_calendar", {})
    assert res.ok and res.text == "Nothing on your calendar today."
