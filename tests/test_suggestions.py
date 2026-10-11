"""Shortcut suggestions (neo2.core.reflex.suggestions): after the same app-and-site launch works three
times NEO offers a spoken shortcut, saying the exact sentence that saves one; on disk it keeps only a
hash and a count. Every test has its own data dir (conftest)."""

from __future__ import annotations

import json

import pytest

from neo2.core.reflex import suggestions
from neo2.core.tools import routines

STEPS = [
    {"tool": "open_app", "args": {"target": "Notion"}},
    {"tool": "open_app", "args": {"target": "Safari"}},
    {"tool": "open_url", "args": {"url": "https://www.youtube.com/feed/subscriptions?x=1"}},
]


@pytest.fixture(autouse=True)
def _suggestions_on(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_LEARN", "on")
    monkeypatch.setenv("NEO_WORKFLOW_SUGGESTIONS", "true")
    monkeypatch.setattr(config, "_settings", None)


def test_the_third_success_offers_the_exact_sentence():
    assert suggestions.record_success(STEPS) == ""
    assert suggestions.record_success(STEPS) == ""
    offer = suggestions.record_success(STEPS)
    assert "“When I say [your phrase], open Notion, Safari, and youtube.com”" in offer
    assert suggestions.record_success(STEPS) == ""  # once, not every time after


def test_the_offered_sentence_saves_the_same_launches():
    for _ in range(3):
        offer = suggestions.record_success(STEPS)
    said = offer.split("“", 1)[1].split("”", 1)[0].replace("[your phrase]", "I want to study")
    parsed = routines.parse_local_setup(said)
    assert parsed is not None and parsed["trigger"] == "I want to study"
    assert parsed["actions"] == [
        {"kind": "open_app", "target": "Notion"},
        {"kind": "open_app", "target": "Safari"},
        {"kind": "open_url", "target": "https://youtube.com"},
    ]


def test_only_a_hash_and_a_count_are_kept():
    for _ in range(3):
        suggestions.record_success(STEPS)
    text = suggestions._path().read_text(encoding="utf-8")
    for word in ("Notion", "notion", "Safari", "youtube", "subscriptions"):
        assert word not in text
    (row,) = json.loads(text)["patterns"].values()
    assert row["count"] == 3 and row["actions"] == 3


def test_a_saved_routine_is_never_offered_again():
    routines._write([{"id": "r1", "name": "study", "trigger": "I want to study", "match": "i want to study",
                      "actions": [{"kind": "open_app", "target": "Notion"}, {"kind": "open_app", "target": "Safari"}]}])
    two = STEPS[:2]
    assert [suggestions.record_success(two) for _ in range(3)] == ["", "", ""]


def test_the_cap_keeps_the_most_recent_successes(monkeypatch):
    monkeypatch.setattr(suggestions, "_MAX", 3)

    def steps(app: str) -> list[dict]:
        return [{"tool": "open_app", "args": {"target": app}}, {"tool": "open_app", "args": {"target": "Safari"}}]

    for app in ("A", "B", "C"):
        suggestions.record_success(steps(app))
    suggestions.record_success(steps("A"))  # A again: now the most recent
    suggestions.record_success(steps("D"))  # one over the cap: the oldest (B) goes, not A
    keys = list(json.loads(suggestions._path().read_text())["patterns"])
    assert keys == [suggestions._signature(steps(x))[0] for x in ("C", "A", "D")]


def test_off_means_nothing_is_counted(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_WORKFLOW_SUGGESTIONS", "false")
    monkeypatch.setattr(config, "_settings", None)
    assert [suggestions.record_success(STEPS) for _ in range(3)] == ["", "", ""]
    assert not suggestions._path().exists()
