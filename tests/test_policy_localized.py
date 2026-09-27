"""The click gate on localized labels (neo2.core.security.policy.commits with back-translations) and on
the words that end or lose something (Quit, Restart…, Don't Save, a forward). The gate only ever gets
stricter: every English check of today still holds; back-translations are OR'd in; the policy module
stays pure stdlib (the core wheel runs it without macOS frameworks)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from neo2.core.security import policy

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
    ok = "खिलाड़ी चुनें।"
    assert policy.sheet_consequence(hi) and policy.sheet_consequence(ok)  # nothing reads them yet: asked (fail closed)
    policy.use_back_translations(lambda t: {hi: ["This action cannot be undone."], ok: ["Choose the players."]}.get(t, []))
    assert policy.sheet_consequence(hi) and not policy.sheet_consequence(ok)


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
    banned = ("neo2.core.decide", "neo2.core.tools", "objc", "AppKit", "Foundation", "Quartz", "ApplicationServices")
    assert not [m for m in anywhere if m.startswith(banned)], anywhere


def test_the_os_hindi_forms_of_commit_labels_are_gated():
    """Measured on the OS's own strings (tests/data/uistrings_gate.json, from this Mac's .loctable
    files): every Hindi form of 25 commit labels is gated through its English; of 140 Hindi forms of
    harmless labels only two shared forms are ("बंद करें" is Close and Quit, "आगे" Next and Forward) -
    stricter, never looser - plus Quit itself, which now asks by design."""
    from neo2.core.decide import uistrings

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


# ---- review fixes: words read piece by piece, and what nothing reads fails closed ----------------------------

_HI = {
    "ठीक है": ["OK"],
    "निर्यात करें": ["Export"],
    "हटाएँ": ["Delete"],
    "सभी आइटम हटाएँ": ["Delete All Items"],
    "इसे पूर्ववत नहीं किया जा सकता।": ["This can't be undone."],
    "आपके बदलाव सहेजे गए।": ["Your changes were saved."],
    "खिलाड़ी चुनें।": ["Choose the players."],
}


def test_a_joined_menu_path_is_read_part_by_part():
    """A submenu command is gated with its path: a joined Hindi path is no one string the OS has,
    so each part is read back on its own ("Export › Delete" asks; so must its Hindi)."""
    policy.use_back_translations(lambda t: _HI.get(t, []))
    assert policy.commits("Export › Delete")
    assert policy.commits("निर्यात करें › हटाएँ")
    assert "Delete" in policy.back_translations("निर्यात करें › हटाएँ")
    assert not policy.commits("निर्यात करें")


def test_a_localized_ok_in_a_dialog_is_read_text_by_text_and_fails_closed():
    policy.use_back_translations(lambda t: _HI.get(t, []))
    joined = "सभी आइटम हटाएँ इसे पूर्ववत नहीं किया जा सकता।"  # a title and a text, joined with a space
    assert policy.commits("ठीक है", joined)  # no one string reads it: asked (fail closed)
    assert policy.commits("ठीक है", ["सभी आइटम हटाएँ", "इसे पूर्ववत नहीं किया जा सकता।"])  # each part read
    assert not policy.commits("ठीक है", ["आपके बदलाव सहेजे गए।"])  # read, and harmless
    assert policy.commits("ठीक है", ["आपके बदलाव सहेजे गए।", "आपकी सारी प्रगति मिट जाएगी।"])  # one part unread
    assert not policy.commits("OK", ["Your changes were saved."]) and policy.commits("OK", ["Saved.", "This can't be undone."])


def test_a_sheet_nothing_can_read_counts_as_saying_something_ends():
    hi = "इस क्रिया को पहले जैसा नहीं किया जा सकता है।"
    assert policy.sheet_consequence(hi)  # no back-translations: can't tell, so yes (fail closed)
    policy.use_back_translations(lambda t: _HI.get(t, []))
    assert not policy.sheet_consequence(["खिलाड़ी चुनें।"])
    assert policy.sheet_consequence(["खिलाड़ी चुनें।", "आपकी सारी प्रगति मिट जाएगी।"])  # the second sentence: unread
    assert policy.sheet_consequence("खिलाड़ी चुनें। आपकी सारी प्रगति मिट जाएगी।")
    assert not policy.sheet_consequence(["Start a New Game?", "खिलाड़ी चुनें।"])
    assert not policy.sheet_consequence("Café au lait?")  # Latin with accents is English-readable


def test_unread_names_foreign_words_nothing_reads():
    assert not policy.unread("New Game") and not policy.unread("Réglages…")
    assert policy.unread("नया गेम")  # nothing registered
    policy.use_back_translations(lambda t: _HI.get(t, []))
    assert not policy.unread("हटाएँ") and not policy.unread("निर्यात करें › हटाएँ")
    assert policy.unread("खिलाड़ी चुनें। आपकी सारी प्रगति मिट जाएगी।") and not policy.unread("खिलाड़ी चुनें।")


def test_the_click_gate_reads_the_dialog_behind_a_localized_ok():
    """The click gate reads a dialog's words only behind a plain OK (policy._GENERIC_OK): a Hindi
    "ठीक है" is that OK, so a Hindi delete dialog behind it asks, as the English one does."""
    from neo2.core.agent import confirm
    from neo2.core.tools import computer as comp
    from neo2.core.tools.computer import ax

    def static(*texts):
        return [ax.Element(f"s{i}", "AXStaticText", "", t, 0, 0, 1, 1) for i, t in enumerate(texts)]

    policy.use_back_translations(lambda t: _HI.get(t, []))
    confirm.cancel("test setup")
    try:
        assert policy._GENERIC_OK.match("ठीक है") and policy._GENERIC_OK.match("OK") and not policy._GENERIC_OK.match("हटाएँ")
        assert policy._GENERIC_OK.pattern.startswith("^")  # still the regex for everything else it's asked
        gate = comp._ask_before_click("click", {"id": "e9"}, "ठीक है", "Some App", static("सभी आइटम हटाएँ", "इसे पूर्ववत नहीं किया जा सकता।"))
        assert gate and gate.startswith("NEEDS_CONFIRM")
        confirm.cancel("next")
        assert comp._ask_before_click("click", {"id": "e9"}, "ठीक है", "Some App", static("आपके बदलाव सहेजे गए।")) is None
        gate = comp._ask_before_click("click", {"id": "e9"}, "OK", "Some App", static("Delete All Items", "This can't be undone."))
        assert gate and gate.startswith("NEEDS_CONFIRM")
    finally:
        confirm.cancel("test over")
