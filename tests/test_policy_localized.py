"""The click gate on localized labels (neo.security.policy.commits with back-translations) and on
the words that end or lose something (Quit, Restart…, Don't Save, a forward). The gate only ever gets
stricter: every English check of today still holds; back-translations are OR'd in; the policy module
stays pure stdlib (the core wheel runs it without macOS frameworks)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from neo.security import policy

DATA = Path(__file__).parent / "data" / "uistrings_gate.json"


@pytest.fixture(autouse=True)
def _no_translations():
    """Each test starts without the Mac side's back-translations (and puts back what was there)."""
    before = policy._back_translate
    policy.use_back_translations(None)
    yield
    policy.use_back_translations(before)


def test_todays_english_gate_is_unchanged():
    for label in ("Send", "Place your order", "Delete account", "Move to Trash", "Buy Now", "Allow", "Sign Out"):
        assert policy.commits(label), label
    for label in ("Open", "New Game", "Start", "Cancel", "Next", "Bold", "Show Sidebar", "Alpha", "Go Forward"):
        assert not policy.commits(label), label
    assert policy.commits("OK", "This will permanently delete 3 items. This can't be undone.")
    assert policy.commits("Continue", "Continue to send this message to 40 people?")
    assert not policy.commits("OK", "Your changes were saved.")
    assert not policy.commits("Continue", "")


@pytest.mark.parametrize(
    "label",
    ["Forward", "Quit", "Quit Game", "Restart…", "Shut Down…", "Force Quit…", "Log Off", "Revert to Saved", "Don’t Save",
     "Don't Save", "Do not save", "Resign", "End Game", "Clear History…", "Update Now"],
)  # fmt: skip
def test_ending_or_losing_something_is_gated(label):
    assert policy.commits(label)


@pytest.mark.parametrize("label", ["Go Forward", "Skip Forward", "Fast-Forward", "Move forward", "Start", "New Window"])
def test_moving_forward_is_navigation_not_a_forward(label):
    assert not policy.commits(label)


def test_stricter_labels_leave_scripts_and_drops_alone():
    """The new words are for labels only: an AppleScript that quits an app is judged as before."""
    assert not policy._COMMIT.search("quit")
    assert policy.drop_commits("Trash") and not policy.drop_commits("Quit")


def test_back_translations_gate_a_hindi_label():
    assert not policy.commits("भेजें")  # today: no English word in it
    policy.use_back_translations(lambda t: {"भेजें": ["Send"], "नया गेम": ["New Game"]}.get(t, []))
    assert policy.commits("भेजें")
    assert not policy.commits("नया गेम")


def test_sources_can_be_handed_in_directly():
    assert policy.commits("भेजें", sources=["Send"])
    assert not policy.commits("भेजें", sources=[])  # an explicit empty list: nothing looked up


def test_a_localized_ok_reads_the_dialogs_words_too():
    table = {"ठीक": ["OK"], "इससे 3 आइटम स्थायी रूप से डिलीट हो जाएँगे।": ["This will permanently delete 3 items."]}
    policy.use_back_translations(lambda t: table.get(t, []))
    assert policy.generic_ok("ठीक") and not policy.generic_ok("नया गेम")
    assert policy.commits("ठीक", "इससे 3 आइटम स्थायी रूप से डिलीट हो जाएँगे।")
    assert not policy.commits("ठीक", "")


def test_ascii_is_never_looked_up_and_a_broken_lookup_adds_nothing():
    calls = []

    def lookup(t):
        calls.append(t)
        raise RuntimeError("broken")

    policy.use_back_translations(lookup)
    assert not policy.commits("Open")
    assert calls == []  # ASCII is English already
    assert not policy.commits("नया गेम")  # the failure adds nothing; today's checks still ran
    assert policy.commits("Delete")


@pytest.mark.parametrize(
    "text,loses",
    [
        ("Start a New Game?", False),
        ("Do you want to start a new game? The current game will be lost.", True),
        ("Your changes will be lost if you don’t save them.", True),
        ("This action can’t be undone.", True),
        ("Are you sure you want to quit?", True),
        ("Choose the players and the variant.", False),
        ("Send this message to 40 people?", True),
    ],
)
def test_sheet_consequence(text, loses):
    assert policy.sheet_consequence(text) is loses


def test_sheet_consequence_reads_hindi_through_its_english():
    hi = "इस क्रिया को पहले जैसा नहीं किया जा सकता है।"
    assert not policy.sheet_consequence(hi)
    policy.use_back_translations(lambda t: {hi: ["This action cannot be undone."]}.get(t, []))
    assert policy.sheet_consequence(hi)


def test_policy_stays_pure_stdlib():
    """The core wheel imports it without macOS frameworks or NEO's decision layer: back-translations
    are handed in, never imported."""
    tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))

    def names(nodes) -> set[str]:
        out = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                out |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                out.add(node.module)
        return out

    top = names(tree.body)  # what importing the module imports
    assert {m.split(".")[0] for m in top} <= set(sys.stdlib_module_names) | {"__future__"}, top
    anywhere = names(ast.walk(tree))  # (a few functions look up NEO's own settings lazily)
    banned = ("neo.decide", "neo.tools", "objc", "AppKit", "Foundation", "Quartz", "ApplicationServices")
    assert not [m for m in anywhere if m.startswith(banned)], anywhere


def test_the_os_hindi_forms_of_commit_labels_are_gated():
    """Measured on the OS's own strings (tests/data/uistrings_gate.json, from this Mac's .loctable
    files): every Hindi form of 25 commit labels is gated through its English; of 140 Hindi forms of
    harmless labels only two shared forms are ("बंद करें" is Close and Quit, "आगे" Next and Forward) -
    stricter, never looser - plus Quit itself, which now asks by design."""
    from neo.decide import uistrings

    data = json.loads(DATA.read_text(encoding="utf-8"))
    before = uistrings._system
    uistrings.install(uistrings.Index.of(data["pairs"], "fixture"))
    policy.use_back_translations(uistrings.english_sources)
    try:
        commit = [h for forms in data["labels"]["commit"].values() for h in forms]
        benign = [(en, h) for en, forms in data["labels"]["benign"].items() for h in forms]
        assert len(commit) >= 27 and not any(policy._COMMIT.search(h) for h in commit)  # today: none
        assert all(policy.commits(h) for h in commit)
        flagged = {h for en, h in benign if policy.commits(h)}
        assert flagged <= {"बंद करें", "आगे"}, flagged
        assert len(benign) >= 100
    finally:
        uistrings.install(before)
