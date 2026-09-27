"""The dictionary-built scripts do at least what today's app-specific scripts (legacy.py) do:
every property a legacy create sets is set by the generated one too (compared by the property's
code in a dictionary shaped like the one macOS ships). New ones only where meant: the alert of a
timed reminder, and an event's own calendar instead of "the first writable one"."""

from __future__ import annotations

import inspect
import re
from datetime import datetime
from pathlib import Path

import pytest

from neo2.core.decide import aescript, sdef
from neo2.core.decide.kinds import schedule
from neo2.core.tools.computer import legacy

FIX = Path(__file__).parent / "fixtures" / "decide" / "sdef"
WHEN = datetime(2026, 10, 1, 13, 0)


def legacy_props(fn, klass: str) -> set[str]:
    """The property names a legacy script sets on a new ``klass``."""
    src = inspect.getsource(fn)
    m = re.search(rf"make new {klass} with properties \{{\{{(.*?)\}}\}}", src)
    assert m, f"{fn.__name__} makes no {klass}"
    names = {p.split(":", 1)[0].strip() for p in m.group(1).split(",")}
    names |= set(re.findall(r'", (\w[\w ]*):my parseISO', src))  # a property added only when given
    return names


def codes(d: sdef.Dictionary, klass: str, names: set[str]) -> set[str]:
    props = d.class_props[klass]
    return {props[n][0] for n in names}


def made(script: str) -> set[str]:
    return set(re.findall(r"«property (.{4})»:\(", script))


@pytest.mark.parametrize(
    "item, fixture, fn, klass, args, extra",
    [
        ("event", "datebook", legacy.calendar_add, "event",
         {"title": "Lunch", "start": "2026-10-01T13:00", "end": "2026-10-01T14:00"}, set()),
        ("reminder", "tasklist", legacy.reminder_add, "reminder",
         {"title": "call mom", "due": "2026-10-01T17:00"}, {"rmdt"}),
    ],
)  # fmt: skip
def test_generated_creates_set_every_property_the_legacy_ones_set(item, fixture, fn, klass, args, extra):
    d = sdef.load_file(str(FIX / f"{fixture}.sdef"))
    kind = "add_event" if item == "event" else "add_reminder"
    spec = schedule.parse_args(kind, args, datetime(2026, 9, 26))
    r = aescript.create_route(d, item, schedule._WANT[item])
    script, _argv = aescript.render_create(r, "b", schedule._values(r, spec), "cal" if item == "event" else "")
    want = codes(d, klass, legacy_props(fn, klass))
    assert want and want <= made(script)
    assert made(script) - want == extra  # nothing else new unless asked for (a place, notes)
    assert "wr05" not in script  # the chosen calendar, not "the first writable one"


def test_a_day_long_reminder_sets_its_day_and_no_alert():
    d = sdef.load_file(str(FIX / "tasklist.sdef"))
    spec = schedule.parse_args("add_reminder", {"title": "rent", "due": "2026-10-03"}, datetime(2026, 9, 26))
    r = aescript.create_route(d, "reminder", schedule._WANT["reminder"])
    script, _argv = aescript.render_create(r, "b", schedule._values(r, spec))
    assert made(script) == {"pnam", "adue"}
