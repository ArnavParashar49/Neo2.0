"""neo.decide.aescript: the shared pieces of dictionary routes - slots found by code or name,
read-only properties never written, HTML bodies detected from the dictionary's own words, the
answer protocol, the error map with its commit semantics, and the facade over media's runner and
Automation gate (so media's own tests keep patching the same functions). osascript is faked."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
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


# ---- making, reading back, finding again, listing ---------------------------------------------------

FIX = Path(__file__).parent / "fixtures" / "decide" / "sdef"
DATEBOOK = sdef.load_file(str(FIX / "datebook.sdef"))
TASKLIST = sdef.load_file(str(FIX / "tasklist.sdef"))
NOTEBOOK = parse(
    """<command name="make" code="corecrel"/>
<class-extension extends="application">
  <property name="default account" code="dfac" type="account"/>
  <element type="account"/><element type="folder"/><element type="note"/>
</class-extension>
<class name="account" code="acct">
  <property name="default folder" code="dfol" type="folder"/>
  <property name="id" code="ID  " type="text" access="r"/>
  <element type="folder"/><element type="note"/>
</class>
<class name="folder" code="cfol">
  <property name="name" code="pnam" type="text"/><property name="id" code="ID  " type="text" access="r"/>
  <element type="folder"/><element type="note"/>
</class>"""
    + NOTE.replace('<class name="note" code="note">', '<class name="note" code="note">\n  <property name="creation date" code="ascd" type="date" access="r"/>')
)
EVENT = ("title", "start", "end", "all_day", "location", "body")
WHEN = datetime(2026, 10, 1, 13, 0)


def test_create_routes_from_the_dictionarys_own_defaults():
    note = aescript.create_route(NOTEBOOK, "note", ("title", "body"))
    assert note.default == ("dfac", "dfol") and note.container == "cfol"  # the default folder of the default account
    assert note.html and note.created_code == "ascd" and note.id_code == "ID  "
    rem = aescript.create_route(TASKLIST, "reminder", ("title", "body", "due", "due_day", "alert"))
    assert rem.default == ("dlis",) and rem.container == "list"  # the default list, not the default account
    assert [s for s, _c, _k in rem.sets] == ["title", "body", "due", "due_day", "alert"]
    ev = aescript.create_route(DATEBOOK, "event", EVENT)
    assert ev.default == () and ev.container == "wres" and ev.listable and ev.writable_key == "wr05"
    assert [c for _s, c, _k in ev.sets] == ["wr11", "wr1s", "wr5s", "wrad", "wr14", "wr12"]
    assert aescript.create_route(DATEBOOK, "reminder", ()) is None and aescript.create_route(None, "event", ()) is None
    no_make = parse(NOTE)
    assert aescript.create_route(no_make, "note", ("title",)) is None


def test_values_travel_only_in_argv():
    nasty = 'He said "hi" «x» & do shell script "rm -rf ~"\nline two प्रिया'
    rem = aescript.create_route(TASKLIST, "reminder", ("title", "body", "due", "alert"))
    script, argv = aescript.render_create(rem, "test.tasklist", {"title": nasty, "body": nasty, "due": WHEN, "alert": WHEN})
    assert argv == ["test.tasklist", "", nasty, nasty, "2026-10-01T13:00:00", "2026-10-01T13:00:00"]
    for bad in ('"hi"', "«x»", "do shell script", "line two", "प्रिया", "test.tasklist", "2026"):
        assert bad not in script
    assert "«property dlis»" in script and "make new «class remi»" in script
    ev = aescript.create_route(DATEBOOK, "event", EVENT)
    script, argv = aescript.render_create(ev, "test.datebook", {"title": "x", "start": WHEN, "end": WHEN, "all_day": True, "location": None}, "cal-1")
    assert argv == ["test.datebook", "cal-1", "x", "2026-10-01T13:00:00", "2026-10-01T13:00:00", "true"]
    assert "wr14" not in script  # a value not given is not set
    assert '(every «class wres» whose «property ID  » is cid)' in script and "NO_CONTAINER" in script


def test_after_the_make_nothing_can_fail_the_script():
    """The id and the container's name are read back in `try`: an error the make script gives
    is the make's own (nothing made), never a made item whose id couldn't be read."""
    ev = aescript.create_route(DATEBOOK, "event", EVENT)
    script, _argv = aescript.render_create(ev, "b", {"title": "x"}, "c")
    after = script[script.index("make new") : script.index("end run")].split("\n")[1:]
    reads = [n for n, line in enumerate(after) if "(get «property" in line]
    assert len(reads) == 2 and all(after[n - 1].strip() == "try" and after[n + 1].strip() == "end try" for n in reads)
    assert 'return "OK" & US & newId & US & cName' in script and "of c," in after[reads[1]]
    flat = parse('<command name="make" code="corecrel"/><class-extension extends="application"><element type="note"/></class-extension>' + NOTE)
    no_holder = aescript.create_route(flat, "note", ("title",))
    assert "cName to my" not in aescript.render_create(no_holder, "b", {"title": "t"})[0]  # the app itself holds it


def test_an_html_body_is_escaped_line_by_line():
    note = aescript.create_route(NOTEBOOK, "note", ("title", "body"))
    _s, argv = aescript.render_create(note, "b", {"title": "t", "body": "a < b\n\nfish & chips"})
    assert argv[-1] == "<div>a &lt; b</div><div><br></div><div>fish &amp; chips</div>"
    rem = aescript.create_route(TASKLIST, "reminder", ("title", "body"))
    assert aescript.render_create(rem, "b", {"title": "t", "body": "a < b"})[1][-1] == "a < b"


def test_read_back_search_back_and_listing_scripts():
    ev = aescript.create_route(DATEBOOK, "event", EVENT)
    script, argv = aescript.readback(ev, "b", "EV-1", "cal-1")
    assert argv == ["b", "cal-1", "EV-1"] and "EMPTY" in script
    assert script.index("wr11") < script.index("wr1s") < script.index("wr5s")  # the order of ev.reads
    rem = aescript.create_route(TASKLIST, "reminder", ("title", "due"))
    script, argv = aescript.search_back(rem, "b", "call mom", WHEN)
    assert argv == ["b", "", "call mom", "2026-10-01T13:00:00"] and "«property ascd» of o) ≥ t" in script
    script, argv = aescript.search_back(ev, "b", "Lunch", WHEN, start=datetime(2026, 10, 2, 9), container_id="c")
    assert argv[-1] == "2026-10-02T09:00:00" and "«property wr1s» of o) = t" in script  # no creation date: the start
    assert aescript.search_back(ev, "b", "Lunch", WHEN) is None  # nothing tells a new one from an old one
    script, argv = aescript.containers(ev, "b")
    assert argv == ["b", ""] and "«property wr05»" in script and "«property pnam»" in script
    lr = aescript.list_route(DATEBOOK)
    assert [s for s, _c, _k in lr.cols] == ["id", "title", "start", "end", "all_day", "recur", "exdates"]
    script, argv = aescript.render_list(lr, "b", "cal-1", WHEN, datetime(2026, 10, 2))
    assert argv == ["b", "cal-1", "2026-10-01T13:00:00", "2026-10-02T00:00:00"]
    assert "«property wr1s» < t1 and «property wr5s» > t0" in script  # overlap, not only the start
    assert "«property wr1s» < t0 and «property wr15» is not missing value" in script  # repeats begun before
    assert aescript.list_route(TASKLIST) is None and aescript.list_route(None) is None


@pytest.mark.skipif(not Path("/usr/bin/osacompile").exists(), reason="macOS only")
def test_every_generated_script_compiles(tmp_path):
    """Compiled only: the target is `application id (item 1 of argv)`, so compiling loads no app's
    terminology and sends nothing."""
    import subprocess

    ev = aescript.create_route(DATEBOOK, "event", EVENT)
    rem = aescript.create_route(TASKLIST, "reminder", ("title", "body", "due", "due_day", "alert"))
    note = aescript.create_route(NOTEBOOK, "note", ("title", "body"))
    scripts = [
        aescript.render_create(ev, "b", {"title": "t", "start": WHEN, "end": WHEN, "all_day": False, "location": "l", "body": "n"}, "c"),
        aescript.render_create(rem, "b", {"title": "t", "due_day": WHEN, "body": "n"}),
        aescript.render_create(note, "b", {"title": "t", "body": "n"}),
        aescript.readback(ev, "b", "i", "c"),
        aescript.readback(rem, "b", "i"),
        aescript.search_back(rem, "b", "t", WHEN),
        aescript.search_back(ev, "b", "t", WHEN, WHEN, "c"),
        aescript.containers(ev, "b"),
        aescript.containers(rem, "b"),
        aescript.render_list(aescript.list_route(DATEBOOK), "b", "c", WHEN, WHEN),
    ]
    for n, (script, _argv) in enumerate(scripts):
        src = tmp_path / f"s{n}.applescript"
        src.write_text(script)
        done = subprocess.run(["osacompile", "-o", str(tmp_path / f"s{n}.scpt"), str(src)], capture_output=True, text=True)
        assert done.returncode == 0, (n, done.stderr)
