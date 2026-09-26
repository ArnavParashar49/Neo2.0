"""neo.decide.aescript: the shared pieces of dictionary routes - slots found by code or name,
read-only properties never written, HTML bodies detected from the dictionary's own words, the
answer protocol, the error map with its commit semantics, and the facade over media's runner and
Automation gate (so media's own tests keep patching the same functions). osascript is faked."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from neo.decide import aescript, media, sdef
from neo.decide.inventory import App
from neo.decide.run import Effect


def parse(body: str) -> sdef.Dictionary:
    return sdef.parse(ET.fromstring(f'<dictionary><suite name="S" code="SSSS">{body}</suite></dictionary>'))


NOTE = """<class name="note" code="note">
  <property name="name" code="pnam" type="text"/>
  <property name="body" code="body" type="text" description="the HTML content of the note"/>
  <property name="plaintext" code="text" type="text" access="r"/>
  <property name="id" code="ID  " type="text" access="r"/>
</class>"""
RENAMED = """<class name="task" code="TkTk">
  <property name="title" code="ttl1" type="text"/>
  <property name="due date" code="dDue" type="date"/>
  <property name="remind me date" code="rmd2" type="date"/>
  <property name="notes" code="nOts" type="text" description="plain notes"/>
  <property name="bad code" code='a"b' type="text"/>
</class>"""


def test_slots_by_de_facto_code_then_by_name():
    d = parse(NOTE)
    assert aescript.slot_prop(d, "note", "title") == ("name", "pnam")
    assert aescript.slot_prop(d, "note", "body") == ("body", "body")
    assert aescript.slot_prop(d, "note", "plain") == ("plaintext", "text")
    r = parse(RENAMED)
    assert aescript.slot_prop(r, "task", "title") == ("title", "ttl1")
    assert aescript.slot_prop(r, "task", "due") == ("due date", "dDue")
    assert aescript.slot_prop(r, "task", "alert") == ("remind me date", "rmd2")
    assert aescript.slot_prop(r, "task", "body") == ("notes", "nOts")
    assert aescript.slot_prop(r, "task", "start") is None and aescript.slot_prop(d, "missing", "title") is None


def test_read_only_properties_are_never_written():
    d = parse(NOTE)
    assert aescript.slot_prop(d, "note", "plain", write=True) is None  # a read-back slot
    assert aescript.slot_prop(d, "note", "title", write=True) == ("name", "pnam")
    ro = parse(NOTE.replace('code="body" type="text"', 'code="body" type="text" access="r"'))
    assert aescript.slot_prop(ro, "note", "body", write=True) is None
    assert aescript.slot_prop(ro, "note", "body") == ("body", "body")  # still readable


def test_a_code_that_could_break_out_of_the_source_is_refused():
    r = parse(RENAMED.replace('code="ttl1"', "code='t\"l1'"))
    assert aescript.slot_prop(r, "task", "title") is None


def test_html_bodies_are_detected_from_the_dictionarys_own_words():
    assert aescript.is_html(parse(NOTE), "note", "body")
    assert not aescript.is_html(parse(RENAMED), "task", "notes")
    assert not aescript.is_html(parse(NOTE.replace("HTML", "HTMLish")), "note", "body")


def test_answers_are_status_token_then_rows():
    us, rs = aescript.US, aescript.RS
    assert aescript.rows(f"OK{us}id1{us}Title") == ("OK", [["id1", "Title"]])
    assert aescript.rows(f"OK{rs}a{us}b{rs}c{us}d{rs}") == ("OK", [["a", "b"], ["c", "d"]])
    assert aescript.rows("EMPTY") == ("EMPTY", [])
    assert aescript.rows("NO_CONTAINER") == ("NO_CONTAINER", [])
    assert aescript.rows("something else") == ("", [])


@pytest.mark.parametrize(
    "code, err, committed, reason, effect",
    [
        (1, "execution error: Not authorized to send Apple events to X. (-1743)", False, "permission", Effect.NONE),
        (1, "... (-1744)", True, "permission", Effect.NONE),
        (1, "application isn't running. (-600)", True, "", Effect.NONE),
        (1, "Can't get note id \"x\". (-1728)", True, "not_found", Effect.NONE),
        (1, "Can't get note id \"x\". (-1728)", False, "not_found", Effect.NONE),
        (124, "timed out after 15s", False, "timeout", Effect.NONE),
        (124, "timed out after 15s", True, "unconfirmed", Effect.UNKNOWN),
        (-9, "", False, "timeout", Effect.NONE),  # killed
        (1, "AppleEvent timed out. (-1712)", True, "unconfirmed", Effect.UNKNOWN),
        (1, "Connection is invalid. (-609)", False, "timeout", Effect.NONE),
        (1, "Connection is invalid. (-609)", True, "unconfirmed", Effect.UNKNOWN),
        (1, "handler failed (-10000)", False, "route_bad", Effect.NONE),
        (1, "doesn't understand the message (-1708)", False, "route_bad", Effect.NONE),
        (1, "doesn't understand the message (-1708)", True, "unconfirmed", Effect.UNKNOWN),
        (1, "some other failure (-2700)", False, "", Effect.NONE),
    ],
)
def test_the_error_map_before_and_after_the_commit_point(code, err, committed, reason, effect):
    out = aescript.fail("Notebook", code, err, committed=committed)
    assert not out.ok and out.reason == reason and out.effect is effect
    assert "Notebook" in out.text


def test_running_and_the_gate_go_through_media_at_call_time(monkeypatch):
    calls = []

    async def fake_osascript(script, args, timeout):
        calls.append((script, list(args), timeout))
        return 0, "OK", ""

    monkeypatch.setattr(media, "_osascript", fake_osascript)
    assert asyncio.run(aescript.run("return 1", ["a b", "«x»"], 5.0)) == (0, "OK", "")
    assert calls == [("return 1", ["a b", "«x»"], 5.0)]

    app = App(bundle_id="test.notebook", name="Notebook", path="/Applications/Notebook.app")

    async def denied(a):
        return SimpleNamespace(ok=False, text=f"{a.name} needs your OK before NEO can control it", reason="permission")

    async def allowed(a):
        return None

    monkeypatch.setattr(media, "_control_gate", denied)
    got = asyncio.run(aescript.control_gate(app))
    assert got is not None and got.reason == "permission" and got.effect is Effect.NONE and "Notebook" in got.text
    monkeypatch.setattr(media, "_control_gate", allowed)
    assert asyncio.run(aescript.control_gate(app)) is None
    assert aescript.ok4("pnam") and not aescript.ok4('pn"m') and aescript.ok8("corecrel") and not aescript.ok8("short")
