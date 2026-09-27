"""neo2.core.decide.kinds: the tables every kind driver shares (tools per kind, nouns, correction cues)
and the driver lookup, which finds nothing until a kind's driver exists."""

from __future__ import annotations

import sys
import types

import pytest

from neo2.core.decide import engine, kinds


def test_every_kind_with_tools_is_an_engine_kind():
    assert set(kinds.KIND_TOOLS) <= set(engine.TASKS)
    assert kinds.KIND_TOOLS["web_search"] == ("web_search", "search_site")
    assert set(kinds.NOUNS.values()) <= set(engine.TASKS)


@pytest.mark.parametrize(
    "word, kind",
    [("email", "read_mail"), ("my inbox", "read_mail"), ("the calendar", "read_calendar"), ("notes", "write_note"),
     ("to-dos", "add_reminder"), ("messages", "send_message"), ("browser", None), ("banana", None), ("", None)],
)
def test_nouns(word, kind):
    assert kinds.noun_kind(word) == kind


@pytest.mark.parametrize(
    "text, app, yes",
    [("no, in notebook", "notebook", True), ("not there, on jotter", "jotter", True),
     ("actually use jotter", "jotter", True), ("nahi, jotter pe", "jotter", True), ("नहीं, जोटर में", "जोटर", True),
     ("jotter pe nahin", "jotter", True), ("put it in jotter instead", "jotter", True), ("No. Jotter!", None, True),
     ("note that the knot is tight", None, False), ("nothing to add", None, False), ("another note", None, False),
     ("", None, False),
     # a cue inside what is being written or sent is content, not a redirect
     ("add a note in jotter do not forget the milk", "jotter", False),
     ("add a note in jotter: actually the meeting moved", "jotter", False),
     ("text priya on chatwave that there is no class today", "chatwave", False),
     ("jotter mein likho ki kal nahi aaunga", "jotter", False),
     ("note in jotter buy oat milk instead", "jotter", False),
     ("put it in jotter instead", None, False)],  # after the app's name only when it is known
)
def test_correction_cues_open_the_request_or_follow_the_app(text, app, yes):
    span = (text.index(app), text.index(app) + len(app)) if app else None
    assert kinds.is_correction(text, span) is yes


def test_driver_lookup(monkeypatch):
    assert kinds.driver("send_message") is None  # no driver yet
    mod = types.ModuleType("fake_note_driver")
    monkeypatch.setitem(sys.modules, "fake_note_driver", mod)
    monkeypatch.setitem(engine.DRIVERS, "write_note", "fake_note_driver")
    assert kinds.driver("write_note") is mod
    monkeypatch.setitem(engine.DRIVERS, "write_note", "neo2.core.decide.kinds.not_written_yet")
    assert kinds.driver("write_note") is None
    assert kinds.driver("play_media").__name__ == "neo2.core.decide.media"
