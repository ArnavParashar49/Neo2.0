"""neo.decide.kinds: the tables every kind driver shares (tools per kind, nouns, correction cues)
and the driver lookup, which finds nothing until a kind's driver exists."""

from __future__ import annotations

import sys
import types

import pytest

from neo.decide import engine, kinds


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
    "text, yes",
    [("no, in notebook", True), ("not there, on jotter", True), ("actually use jotter", True),
     ("nahi, jotter pe", True), ("jotter pe nahin", True), ("नहीं, जोटर में", True), ("put it in jotter instead", True),
     ("note that the knot is tight", False), ("nothing to add", False), ("another note", False), ("", False)],
)
def test_correction_cues_are_whole_words(text, yes):
    assert kinds.is_correction(text) is yes


def test_driver_lookup(monkeypatch):
    assert kinds.driver("send_message") is None  # no driver yet
    mod = types.ModuleType("fake_note_driver")
    monkeypatch.setitem(sys.modules, "fake_note_driver", mod)
    monkeypatch.setitem(engine.DRIVERS, "write_note", "fake_note_driver")
    assert kinds.driver("write_note") is mod
    monkeypatch.setitem(engine.DRIVERS, "write_note", "neo.decide.kinds.not_written_yet")
    assert kinds.driver("write_note") is None
    assert kinds.driver("play_media").__name__ == "neo.decide.media"
